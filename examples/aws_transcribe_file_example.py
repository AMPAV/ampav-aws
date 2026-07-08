"""Transcribe the bundled AMP-Intro media file with AWS Transcribe.

Copy the examples directory outside the repo, copy
config/aws_config.example.yaml to config/aws_config.yaml, update the copied
config for your AWS account, then run this script from the copied directory.
"""

from ampav.aws.transcribe import TranscriptionSettings
from ampav_aws_pipeline.transcribe import transcribe_file

from example_support import DATA_DIR, load_config, write_tool_output


INPUT_FILE = DATA_DIR / "AMP-Intro.m4a"
OUTPUT_FILE = "AMP-Intro-Transcript.yaml"
INPUT_PREFIX = "aws_transcribe/input"


def main() -> None:
    """Run the local-file transcription example."""
    config = load_config()
    aws_config = config.get("aws", {})
    s3_config = config.get("s3", {})
    transcription_config = dict(config.get("transcription", {}))
    polling_config = config.get("polling", {})

    bucket = s3_config.get("bucket")
    if not bucket:
        raise ValueError("s3.bucket is required")

    result = transcribe_file(
        INPUT_FILE,
        input_bucket=bucket,
        input_prefix=INPUT_PREFIX,
        output_s3_uri=s3_config.get("output_s3_uri"),
        job_name_suffix=transcription_config.pop("job_name_suffix", INPUT_FILE.stem),
        transcription_settings=TranscriptionSettings(**transcription_config),
        region_name=aws_config.get("region"),
        profile_name=aws_config.get("profile_name"),
        delete_user_owned_outputs=bool(s3_config.get("delete_user_owned_outputs", False)),
        include_tool_private=bool(s3_config.get("include_tool_private", False)),
        polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
        timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
        keep_uploaded_input=bool(s3_config.get("keep_uploaded_input", False)),
    )

    output_path = write_tool_output(OUTPUT_FILE, result)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
