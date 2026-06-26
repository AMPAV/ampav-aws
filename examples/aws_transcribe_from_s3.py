"""Minimal example for transcribing media that already exists in S3.

Use this when another system owns upload/storage and the AMPAV client only needs
to submit the AWS Transcribe job and read the resulting AMPAV ToolOutput.
"""

from argparse import ArgumentParser

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings


def main() -> None:
    """Run the S3-media transcription example from command-line arguments."""
    parser = ArgumentParser(description="Transcribe an existing s3:// media object with AWS Transcribe.")
    parser.add_argument("media_uri")
    parser.add_argument("--output-s3-uri")
    parser.add_argument("--delete-output", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    args = parser.parse_args()

    client = AwsTranscribe(profile_name=args.profile, region_name=args.region)
    result = client.process(
        args.media_uri,
        output_s3_uri=args.output_s3_uri,
        delete_output=args.delete_output,
        transcription=TranscriptionSettings(),
    )
    print(result.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    main()
