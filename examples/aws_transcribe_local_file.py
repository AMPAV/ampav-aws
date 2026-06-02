"""Minimal example for transcribing a local media file.

The local file is uploaded to S3 before AWS Transcribe is started. Credentials
come from boto3's normal provider chain, such as an AWS profile, environment
variables, or instance role credentials.
"""

from argparse import ArgumentParser
from pathlib import Path

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings


def main() -> None:
    """Run the local-file transcription example from command-line arguments."""
    parser = ArgumentParser(description="Transcribe a local media file with AWS Transcribe.")
    parser.add_argument("audiofile", type=Path)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--input-bucket")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    args = parser.parse_args()

    client = AwsTranscribe(profile_name=args.profile, region_name=args.region)
    result = client.process(
        args.audiofile,
        output_bucket=args.output_bucket,
        input_bucket=args.input_bucket,
        transcription=TranscriptionSettings(),
    )
    print(result.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
