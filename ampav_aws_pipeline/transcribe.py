"""Blocking pipeline adapters for AWS Transcribe tools."""

from os import PathLike
from typing import Any

from ampav.core.schema import ToolOutput

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings

from .s3_files import upload_file

_INPUT_PREFIX = "aws_transcribe/input"


def transcribe_file(
    source: str | PathLike[str],
    *,
    input_bucket: str,
    input_prefix: str = _INPUT_PREFIX,
    output_s3_uri: str | None = None,
    job_name_suffix: str | None = None,
    transcription_settings: TranscriptionSettings | None = None,
    region_name: str | None = None,
    profile_name: str | None = None,
    delete_user_owned_outputs: bool = False,
    include_tool_private: bool = False,
    polling_interval: float = 30,
    timeout: float | None = 7200,
    transcribe_client: Any | None = None,
    s3_client: Any | None = None,
    keep_uploaded_input: bool = False,
) -> ToolOutput:
    """Upload a local media file, transcribe it, and delete the uploaded media by default."""
    client = AwsTranscribe(
        region_name=region_name,
        profile_name=profile_name,
        transcribe_client=transcribe_client,
        s3_client=s3_client,
        delete_user_owned_outputs=delete_user_owned_outputs,
        include_tool_private=include_tool_private,
        polling_interval=polling_interval,
        timeout=timeout,
    )
    input_location = upload_file(
        client.s3_client,
        source,
        bucket=input_bucket,
        prefix=input_prefix,
        name_prefix="ampav-aws-transcribe",
    )
    try:
        return client.process(
            input_location.uri,
            output_s3_uri=output_s3_uri,
            job_name_suffix=job_name_suffix,
            transcription_settings=transcription_settings,
        )
    finally:
        if not keep_uploaded_input:
            client.s3_client.delete_object(Bucket=input_location.bucket, Key=input_location.key)
