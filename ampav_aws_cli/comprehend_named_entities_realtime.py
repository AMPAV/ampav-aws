"""Command-line entry point for AWS Comprehend real-time named entities."""

import argparse
from collections.abc import Sequence
import logging
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from ampav.core.async_tool import ToolError
from ampav.core.logging import LOG_FORMAT

from ampav.aws.comprehend_named_entities_realtime import (
    AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
    DEFAULT_CHUNK_OVERLAP_BYTES,
    AwsComprehendNamedEntitiesRealtime,
)


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the AWS Comprehend real-time named-entities CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract named entities from a UTF-8 text file with AWS Comprehend "
            "real-time analysis and print AMPAV ToolOutput YAML."
        )
    )
    parser.add_argument("text_file", help="Path to a UTF-8 text file")
    parser.add_argument(
        "--language-code",
        default="en",
        help="AWS Comprehend language code",
    )
    parser.add_argument(
        "--max-chunk-bytes",
        type=int,
        default=AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
        help="Maximum UTF-8 bytes per request",
    )
    parser.add_argument(
        "--chunk-overlap-bytes",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_BYTES,
        help="Byte context added around long-text chunk boundaries",
    )
    parser.add_argument(
        "--include-tool-private",
        action="store_true",
        help="Include raw per-chunk AWS responses in ToolOutput.tool_private",
    )
    parser.add_argument("--profile", help="AWS profile name for boto3 session")
    parser.add_argument("--region", help="AWS region for boto3 session")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one AWS Comprehend real-time entity extraction."""
    args = build_cli_parser().parse_args(argv)
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.DEBUG if args.debug else logging.INFO,
    )
    try:
        text = Path(args.text_file).read_text(encoding="utf-8")
        tool = AwsComprehendNamedEntitiesRealtime(
            region_name=args.region,
            profile_name=args.profile,
            max_chunk_bytes=args.max_chunk_bytes,
            chunk_overlap_bytes=args.chunk_overlap_bytes,
            include_tool_private=args.include_tool_private,
        )
        result = tool.process(text, language_code=args.language_code)
    except (ToolError, BotoCoreError, ClientError, OSError, TypeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1

    print(result.model_dump_yaml(sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
