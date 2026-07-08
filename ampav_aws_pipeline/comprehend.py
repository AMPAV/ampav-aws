"""Blocking pipeline adapters for AWS Comprehend tools."""

from typing import Any

from ampav.core.schema import NamedEntities, ToolOutput, Transcript
from ampav.core.schema.transcript import words_to_text

from ampav.aws.comprehend_named_entities import AwsComprehendNamedEntities

from .s3_files import upload_text

_INPUT_PREFIX = "aws_comprehend_named_entities/input"


def extract_named_entities(
    transcript: Transcript,
    *,
    input_bucket: str,
    input_prefix: str = _INPUT_PREFIX,
    output_s3_uri: str | None = None,
    language_code: str = "en",
    job_name_suffix: str | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
    data_access_role_arn: str | None = None,
    entity_recognizer_arn: str | None = None,
    output_kms_key_id: str | None = None,
    volume_kms_key_id: str | None = None,
    delete_user_owned_outputs: bool = False,
    include_tool_private: bool = False,
    polling_interval: float = 30,
    timeout: float | None = 7200,
    comprehend_client: Any | None = None,
    s3_client: Any | None = None,
    keep_uploaded_input: bool = False,
    separator: str = " ",
) -> ToolOutput:
    """Extract named entities from a transcript and align entity timestamps when possible.

    The adapter uploads canonical transcript text to S3, calls the low-level
    Comprehend named-entities tool, then deletes the uploaded text by default.
    """
    if not transcript.words:
        raise ValueError("transcript.words is required")

    client = AwsComprehendNamedEntities(
        region_name=region_name,
        profile_name=profile_name,
        comprehend_client=comprehend_client,
        s3_client=s3_client,
        data_access_role_arn=data_access_role_arn,
        entity_recognizer_arn=entity_recognizer_arn,
        output_kms_key_id=output_kms_key_id,
        volume_kms_key_id=volume_kms_key_id,
        delete_user_owned_outputs=delete_user_owned_outputs,
        include_tool_private=include_tool_private,
        polling_interval=polling_interval,
        timeout=timeout,
    )
    text = words_to_text(transcript.words, separator=separator)
    input_location = upload_text(
        client.s3_client,
        text,
        bucket=input_bucket,
        prefix=input_prefix,
        name="transcript.txt",
        name_prefix="ampav-aws-comprehend-named-entities",
    )
    try:
        result = client.process(
            input_location.uri,
            output_s3_uri=output_s3_uri,
            language_code=language_code,
            job_name_suffix=job_name_suffix,
        )
        if not isinstance(result.output, NamedEntities):
            raise TypeError("aws_comprehend_named_entities returned non-NamedEntities output")
        result.messages.extend(result.output.align_timestamps(transcript.words, separator=separator))
        result.parameters["transcript_text_source"] = "transcript.words"
        result.parameters["transcript_text_separator"] = separator
        return result
    finally:
        if not keep_uploaded_input:
            client.s3_client.delete_object(Bucket=input_location.bucket, Key=input_location.key)
