"""Transcribe an existing S3 media object with AWS Transcribe.

Copy the examples directory outside the repo, copy
config/aws_config.example.yaml to config/aws_config.yaml, update the copied
config for your AWS account, then run this script with an existing S3 media URI.
"""

from argparse import ArgumentParser
from pathlib import Path

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings, safe_job_part

from example_support import load_config, write_tool_output


def main() -> None:
    """Run the existing-S3 transcription example."""
    parser = ArgumentParser(description="Transcribe an existing s3:// media object.")
    parser.add_argument("input_s3_uri")
    args = parser.parse_args()

    config = load_config()
    aws_config = config.get("aws", {})
    s3_config = config.get("s3", {})
    transcription_config = dict(config.get("transcription", {}))
    polling_config = config.get("polling", {})

    client = AwsTranscribe(
        region_name=aws_config.get("region"),
        profile_name=aws_config.get("profile_name"),
        polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
        timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
    )

    input_stem = safe_job_part(Path(str(args.input_s3_uri).rstrip("/")).stem) or "S3Input"
    result = client.process(
        args.input_s3_uri,
        output_s3_uri=s3_config.get("output_s3_uri"),
        delete_output=bool(s3_config.get("delete_output", False)),
        job_name_suffix=transcription_config.pop("job_name_suffix", input_stem),
        include_tool_private=False,
        transcription=TranscriptionSettings(**transcription_config),
    )

    output_path = write_tool_output(f"{input_stem}-Transcript.yaml", result)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
