"""Command-line entry point for AWS Comprehend named entities."""

import argparse
import logging
from pathlib import Path
from typing import Sequence

from botocore.exceptions import BotoCoreError, ClientError

from ampav.core.async_tool import ToolError
from ampav.core.logging import LOG_FORMAT

from ampav.aws.comprehend_named_entities import AwsComprehendNamedEntities
from ampav_aws_pipeline.s3_files import upload_text


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the AWS Comprehend named-entities CLI parser."""
    parser = argparse.ArgumentParser(
        description="Extract named entities from a UTF-8 text file with AWS Comprehend and print ToolOutput YAML."
    )
    parser.add_argument("text_file", help="Path to a UTF-8 text file")
    parser.add_argument("--input-bucket", required=True, help="S3 bucket for uploading temporary input text")
    parser.add_argument("--input-key", help="Exact S3 key for uploaded input text")
    parser.add_argument(
        "--input-prefix",
        default="aws_comprehend_named_entities/input",
        help="S3 prefix for generated temporary input text keys",
    )
    parser.add_argument("--output-s3-uri", help="Optional s3://bucket/prefix for Comprehend output")
    parser.add_argument("--data-access-role-arn", required=True, help="AWS Comprehend data access role ARN")
    parser.add_argument("--entity-recognizer-arn", help="Optional custom entity recognizer ARN")
    parser.add_argument("--output-kms-key-id", help="Optional KMS key for Comprehend output")
    parser.add_argument("--volume-kms-key-id", help="Optional KMS key for Comprehend volume encryption")
    parser.add_argument(
        "--delete-user-owned-outputs",
        action="store_true",
        help="Delete caller-supplied S3 output after reading it",
    )
    parser.add_argument("--keep-input", action="store_true", help="Keep CLI-uploaded input text after the run")
    parser.add_argument("--job-name-suffix", help="Human-readable suffix for the generated AWS job name")
    parser.add_argument("--include-tool-private", action="store_true", help="Include raw AWS payloads in ToolOutput.tool_private")
    parser.add_argument("--language-code", default="en", help="AWS Comprehend language code")
    parser.add_argument("--profile", help="AWS profile name for boto3 session")
    parser.add_argument("--region", help="AWS region for boto3 session")
    parser.add_argument("--poll-interval", type=float, default=30, help="Seconds between job status checks")
    parser.add_argument("--timeout", type=float, default=7200, help="Maximum seconds to wait for completion")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the AWS Comprehend named-entities CLI."""
    args = build_cli_parser().parse_args(argv)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    client = AwsComprehendNamedEntities(
        region_name=args.region,
        profile_name=args.profile,
        data_access_role_arn=args.data_access_role_arn,
        entity_recognizer_arn=args.entity_recognizer_arn,
        output_kms_key_id=args.output_kms_key_id,
        volume_kms_key_id=args.volume_kms_key_id,
        delete_user_owned_outputs=args.delete_user_owned_outputs,
        include_tool_private=args.include_tool_private,
        polling_interval=args.poll_interval,
        timeout=args.timeout,
    )
    uploaded_input = None
    try:
        text_path = Path(args.text_file)
        text = text_path.read_text(encoding="utf-8")
        uploaded_input = upload_text(
            client.s3_client,
            text,
            bucket=args.input_bucket,
            key=args.input_key,
            prefix=args.input_prefix,
            name=text_path.name,
            name_prefix="ampav-aws-comprehend-named-entities",
        )
        logging.info("Uploaded local text to %s", uploaded_input.uri)
        result = client.process(
            uploaded_input.uri,
            output_s3_uri=args.output_s3_uri,
            language_code=args.language_code,
            job_name_suffix=args.job_name_suffix,
        )
    except Exception as exc:
        cli_errors = (ToolError, BotoCoreError, ClientError, OSError, ValueError, TypeError)
        if not isinstance(exc, cli_errors):
            raise
        logging.error("%s", exc)
        return 1
    finally:
        if uploaded_input is not None and not args.keep_input:
            client.s3_client.delete_object(Bucket=uploaded_input.bucket, Key=uploaded_input.key)

    print(result.model_dump_yaml(sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
