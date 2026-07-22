"""Blocking pipeline adapters for AWS Comprehend tools."""

from typing import Any

from ampav.core.schema import NamedEntities, ToolOutput, Transcript
from ampav.core.schema.transcript import words_to_text
from ampav.core.text_chunking import words_to_text_units

from ampav.aws.comprehend_named_entities import AwsComprehendNamedEntities
from ampav.aws.comprehend_named_entities_realtime import (
    AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
    DEFAULT_CHUNK_OVERLAP_BYTES,
    AwsComprehendNamedEntitiesRealtime,
)

from .s3_files import upload_text

_INPUT_PREFIX = "aws_comprehend_named_entities/input"


def extract_named_entities_realtime_from_transcript(
    transcript: Transcript,
    *,
    language_code: str = "en",
    separator: str = " ",
    region_name: str | None = None,
    profile_name: str | None = None,
    max_chunk_bytes: int = AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
    chunk_overlap_bytes: int = DEFAULT_CHUNK_OVERLAP_BYTES,
    include_tool_private: bool = False,
    comprehend_client: Any | None = None,
) -> ToolOutput:
    """Extract named entities from canonical transcript words in real time.

    The adapter builds text and byte-weighted chunk units together, calls AWS
    Comprehend without S3, reassembles full-text entity offsets, and then aligns
    timestamps once against the original transcript words.

    Args:
        transcript: Transcript whose non-empty ``words`` define source text.
        language_code: AWS language code for built-in entity recognition.
        separator: Exact text inserted between rendered transcript words.
        region_name: Optional AWS region used when creating a boto3 session.
        profile_name: Optional boto3 profile used for authentication.
        max_chunk_bytes: Maximum UTF-8 bytes sent in one AWS request.
        chunk_overlap_bytes: Best-effort byte context around owned chunk ranges.
        include_tool_private: Include native per-chunk responses for debugging.
        comprehend_client: Optional injected AWS Comprehend client.
    """
    _validate_realtime_transcript(transcript)
    tool = AwsComprehendNamedEntitiesRealtime(
        region_name=region_name,
        profile_name=profile_name,
        comprehend_client=comprehend_client,
        max_chunk_bytes=max_chunk_bytes,
        chunk_overlap_bytes=chunk_overlap_bytes,
        include_tool_private=include_tool_private,
    )
    text, units = words_to_text_units(
        transcript.words,
        separator=separator,
        weight_fn=lambda value: len(value.encode("utf-8")),
    )
    result = tool._process_with_units(
        text,
        units,
        language_code=language_code,
        media_duration=transcript.media_duration,
        extra_parameters={
            "transcript_text_source": "transcript.words",
            "transcript_text_separator": separator,
        },
    )
    if not isinstance(result.output, NamedEntities):
        raise RuntimeError(
            "aws_comprehend_named_entities_realtime returned non-NamedEntities output"
        )
    result.messages.extend(
        result.output.align_timestamps(transcript.words, separator=separator)
    )
    return result


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


def _validate_realtime_transcript(transcript: Transcript) -> None:
    """Reject transcripts that cannot produce meaningful canonical text."""
    if not isinstance(transcript, Transcript):
        raise TypeError("transcript must be a Transcript")
    if not transcript.words:
        raise ValueError("transcript.words is required")
    for index, word in enumerate(transcript.words):
        if not word.word.strip():
            raise ValueError(f"transcript.words[{index}].word must not be empty")
