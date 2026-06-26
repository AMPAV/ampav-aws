"""Example application-owned YAML config for AWS Transcribe.

The core library accepts typed arguments and boto3 session settings. This
example keeps YAML parsing in client code for applications that prefer a
single structured config file.
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import yaml

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings
from ampav_aws_utils.s3_files import delete_object, upload_file


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

    client = AwsTranscribe(
        region_name=aws_config.get("region"),
        profile_name=aws_config.get("profile_name"),
        polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
        timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
    )
    media_uri = args.media
    input_location = None
    if not media_uri.startswith("s3://"):
        input_bucket = s3_config.get("input_bucket") or s3_config.get("bucket")
        if not input_bucket:
            raise ValueError("s3.input_bucket or s3.bucket is required for local media")
        input_location = upload_file(
            client.s3_client,
            media_uri,
            bucket=input_bucket,
            prefix=s3_config.get("input_prefix", "aws_transcribe/input"),
            name_prefix=transcription_config.get("job_name_prefix", "ampav-aws-transcribe"),
        )
        media_uri = input_location.uri

    try:
        result = client.process(
            media_uri,
            output_s3_uri=s3_config.get("output_s3_uri"),
            delete_output=bool(s3_config.get("delete_output", False)),
            job_name_prefix=transcription_config.pop("job_name_prefix", "ampav-aws-transcribe"),
            transcription=TranscriptionSettings(**transcription_config),
        )
    finally:
        if input_location is not None and not s3_config.get("keep_uploaded_input", False):
            delete_object(client.s3_client, input_location)
    print(result.model_dump_yaml(sort_keys=False))


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping for this example script."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    main()
