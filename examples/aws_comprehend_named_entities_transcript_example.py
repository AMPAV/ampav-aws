"""Extract named entities from the Transcribe example output.

Copy the examples directory outside the repo, run
aws_transcribe_file_example.py first, then run this script from the copied
directory using the same config/aws_config.yaml file.
"""

import yaml

from ampav.core.schema import ToolOutput, Transcript
from ampav_aws_pipeline.comprehend import extract_named_entities

from example_support import DATA_DIR, load_config, write_tool_output


INPUT_FILE = DATA_DIR / "AMP-Intro-Transcript.yaml"
OUTPUT_FILE = "AMP-Intro-NamedEntities.yaml"
INPUT_PREFIX = "aws_comprehend_named_entities/input"


def main() -> None:
    """Run the Comprehend named-entities example from a saved transcript output."""
    config = load_config()
    aws_config = config.get("aws", {})
    s3_config = config.get("s3", {})
    comprehend_config = dict(config.get("comprehend_named_entities", {}))
    polling_config = config.get("polling", {})

    bucket = s3_config.get("bucket")
    if not bucket:
        raise ValueError("s3.bucket is required")

    role_arn = comprehend_config.get("data_access_role_arn")
    if not role_arn:
        raise ValueError("comprehend_named_entities.data_access_role_arn is required")

    output_prefix = comprehend_config.pop("output_prefix", None)
    output_s3_uri = f"s3://{bucket}/{output_prefix.strip('/')}" if output_prefix else None
    transcript = load_transcript(INPUT_FILE)
    result = extract_named_entities(
        transcript,
        input_bucket=bucket,
        input_prefix=INPUT_PREFIX,
        output_s3_uri=output_s3_uri,
        language_code=comprehend_config.pop("language_code", "en"),
        job_name_suffix=comprehend_config.pop("job_name_suffix", INPUT_FILE.stem),
        region_name=aws_config.get("region"),
        profile_name=aws_config.get("profile_name"),
        data_access_role_arn=role_arn,
        entity_recognizer_arn=comprehend_config.pop("entity_recognizer_arn", None),
        output_kms_key_id=comprehend_config.pop("output_kms_key_id", None),
        volume_kms_key_id=comprehend_config.pop("volume_kms_key_id", None),
        delete_user_owned_outputs=bool(s3_config.get("delete_user_owned_outputs", False)),
        include_tool_private=bool(s3_config.get("include_tool_private", False)),
        polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
        timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
        keep_uploaded_input=bool(s3_config.get("keep_uploaded_input", False)),
    )

    output_path = write_tool_output(OUTPUT_FILE, result)
    print(f"Wrote {output_path}")


def load_transcript(path) -> Transcript:
    """Load the Transcript payload from a saved ToolOutput YAML file."""
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected ToolOutput YAML mapping in {path}")
    output = ToolOutput.model_validate(data)
    if not isinstance(output.output, Transcript):
        raise ValueError(f"Expected transcript ToolOutput at {path}")
    return output.output


if __name__ == "__main__":
    main()
