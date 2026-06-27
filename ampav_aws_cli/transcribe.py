"""Command-line entry point for AWS Transcribe."""

import argparse
import logging
from os import PathLike
from pathlib import Path
from typing import Sequence

from botocore.exceptions import BotoCoreError, ClientError

from ampav.core.async_tool import ToolError
from ampav.core.logging import LOG_FORMAT
from ampav.aws.s3 import S3Location
from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings
from ampav_aws_utils.s3_files import upload_file


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the AWS Transcribe CLI parser."""
    parser = argparse.ArgumentParser(description="Transcribe media with AWS Transcribe and print AMPAV ToolOutput YAML.")
    parser.add_argument("media", help="Local media path or s3://bucket/key media URI")
    parser.add_argument("--output-s3-uri", help="Optional exact s3://bucket/key transcript output location")
    parser.add_argument("--delete-output", action="store_true", help="Delete caller-supplied S3 output after reading it")
    parser.add_argument("--input-bucket", help="S3 bucket for uploading a local media file")
    parser.add_argument("--input-key", help="S3 key for uploading a local media file")
    parser.add_argument("--input-prefix", default="aws_transcribe/input", help="S3 prefix for generated input keys")
    parser.add_argument("--keep-input", action="store_true", help="Keep CLI-uploaded input S3 object after the run")
    parser.add_argument("--job-name-suffix", help="Human-readable suffix for the generated AWS job name")
    parser.add_argument("--include-tool-private", action="store_true", help="Include raw AWS payloads in ToolOutput.tool_private")
    parser.add_argument("--media-format", help="AWS media format; inferred from extension when omitted")
    parser.add_argument("--language-code", default="en-US", help="AWS language code when language identification is off")
    parser.add_argument("--identify-language", action="store_true", help="Enable AWS language identification")
    parser.add_argument("--language-option", action="append", default=[], help="Allowed language when identifying language")
    parser.add_argument("--no-speaker-labels", action="store_true", help="Disable AWS speaker diarization")
    parser.add_argument("--max-speaker-labels", type=int, default=10, help="Maximum AWS speaker labels")
    parser.add_argument("--profile", help="AWS profile name for boto3 session")
    parser.add_argument("--region", help="AWS region for boto3 session")
    parser.add_argument("--poll-interval", type=float, default=30, help="Seconds between job status checks")
    parser.add_argument("--timeout", type=float, default=7200, help="Maximum seconds to wait for completion")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AWS Transcribe CLI."""
    args = build_cli_parser().parse_args(argv)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    transcription = TranscriptionSettings(
        media_format=args.media_format,
        language_code=args.language_code,
        identify_language=args.identify_language,
        language_options=args.language_option,
        show_speaker_labels=not args.no_speaker_labels,
        max_speaker_labels=args.max_speaker_labels,
    )
    aws = AwsTranscribe(
        region_name=args.region,
        profile_name=args.profile,
        polling_interval=args.poll_interval,
        timeout=args.timeout,
    )

    uploaded_input = None
    try:
        input_s3_uri, uploaded_input = _prepare_media_uri(
            aws,
            args.media,
            input_bucket=args.input_bucket,
            input_key=args.input_key,
            input_prefix=args.input_prefix,
        )

        result = aws.process(
            input_s3_uri,
            output_s3_uri=args.output_s3_uri,
            delete_output=args.delete_output,
            job_name_suffix=args.job_name_suffix,
            include_tool_private=args.include_tool_private,
            transcription=transcription,
        )
    except Exception as exc:
        cli_errors = (ToolError, BotoCoreError, ClientError, OSError, ValueError)
        if not isinstance(exc, cli_errors):
            raise
        logging.error("%s", exc)
        return 1
    finally:
        if uploaded_input is not None and not args.keep_input:
            aws.s3_client.delete_object(Bucket=uploaded_input.bucket, Key=uploaded_input.key)

    print(result.model_dump_yaml(sort_keys=False))
    return 0


def _prepare_media_uri(
    aws: AwsTranscribe,
    media: str | PathLike[str],
    *,
    input_bucket: str | None,
    input_key: str | None,
    input_prefix: str,
) -> tuple[str, S3Location | None]:
    media_str = str(media)
    if media_str.startswith("s3://"):
        return media_str, None
    if not input_bucket:
        raise ValueError("--input-bucket is required when media is a local file")
    location = upload_file(
        aws.s3_client,
        Path(media),
        bucket=input_bucket,
        key=input_key,
        prefix=input_prefix,
        name_prefix="ampav-aws-transcribe",
    )
    logging.info("Uploaded local media to %s", location.uri)
    return location.uri, location


if __name__ == "__main__":
    raise SystemExit(main())
