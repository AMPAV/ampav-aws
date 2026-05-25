"""Example application-owned YAML config for AWS Transcribe.

The core library accepts typed arguments and boto3 session settings. This
example keeps YAML parsing in client code for applications that prefer a
single structured config file.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml

from ampav.aws.transcribe import PollingSettings, TranscriptionSettings, transcribe_file


def main() -> None:
    """Run the YAML-config transcription example from command-line arguments."""
    parser = ArgumentParser(description="Transcribe media using application-owned YAML configuration.")
    parser.add_argument("media")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_yaml(args.config)
    aws_config = config.get("aws", {})
    s3_config = config.get("s3", {})
    transcription_config = config.get("transcription", {})
    polling_config = config.get("polling", {})

    result = transcribe_file(
        args.media,
        output_bucket=s3_config["output_bucket"],
        input_bucket=s3_config.get("input_bucket"),
        input_prefix=s3_config.get("input_prefix", "aws_transcribe/input"),
        output_prefix=s3_config.get("output_prefix", "aws_transcribe/output"),
        job_name_prefix=transcription_config.pop("job_name_prefix", "ampav-aws-transcribe"),
        transcription=TranscriptionSettings(**transcription_config),
        polling=PollingSettings(**polling_config),
        region_name=aws_config.get("region"),
        profile_name=aws_config.get("profile_name"),
    )
    print(result.model_dump_yaml(sort_keys=False))


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping for this example script."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    main()
