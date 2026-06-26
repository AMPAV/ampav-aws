"""Minimal example for transcribing a local media file.

The local file is uploaded to S3 before AWS Transcribe is started. Credentials
come from boto3's normal provider chain, such as an AWS profile, environment
variables, or instance role credentials.
"""

from argparse import ArgumentParser
from pathlib import Path

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings
from ampav_aws_utils.s3_files import delete_object, upload_file


def main() -> None:
    """Run the local-file transcription example from command-line arguments."""
    parser = ArgumentParser(description="Transcribe a local media file with AWS Transcribe.")
    parser.add_argument("audiofile", type=Path)
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--input-prefix", default="aws_transcribe/input")
    parser.add_argument("--output-s3-uri")
    parser.add_argument("--delete-output", action="store_true")
    parser.add_argument("--keep-input", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    args = parser.parse_args()

    client = AwsTranscribe(profile_name=args.profile, region_name=args.region)
    input_location = upload_file(
        client.s3_client,
        args.audiofile,
        bucket=args.input_bucket,
        prefix=args.input_prefix,
        name_prefix="ampav-aws-transcribe",
    )
    try:
        result = client.process(
            input_location.uri,
            output_s3_uri=args.output_s3_uri,
            delete_output=args.delete_output,
            transcription=TranscriptionSettings(),
        )
    finally:
        if not args.keep_input:
            delete_object(client.s3_client, input_location)
    print(result.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
