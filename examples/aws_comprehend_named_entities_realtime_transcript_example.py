"""Extract named entities from a saved Transcript with Comprehend real time.

Copy the examples directory outside the repo, run
aws_transcribe_file_example.py first, then run this script from the copied
directory using the same config/aws_config.yaml file.
"""

from ampav.aws.comprehend_named_entities_realtime import (
    AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
    DEFAULT_CHUNK_OVERLAP_BYTES,
)
from ampav_aws_pipeline import extract_named_entities_realtime_from_transcript

from example_support import DATA_DIR, load_config, load_transcript, write_tool_output


INPUT_FILE = DATA_DIR / "AMP-Intro-Transcript.yaml"
OUTPUT_FILE = "AMP-Intro-NamedEntities-Realtime.yaml"


def main() -> None:
    """Run real-time entity extraction and align transcript timestamps."""
    config = load_config()
    aws_config = config.get("aws", {})
    comprehend_config = config.get("comprehend_named_entities_realtime", {})
    transcript = load_transcript(INPUT_FILE)

    result = extract_named_entities_realtime_from_transcript(
        transcript,
        language_code=comprehend_config.get("language_code", "en"),
        region_name=aws_config.get("region"),
        profile_name=aws_config.get("profile_name"),
        max_chunk_bytes=comprehend_config.get(
            "max_chunk_bytes",
            AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
        ),
        chunk_overlap_bytes=comprehend_config.get(
            "chunk_overlap_bytes",
            DEFAULT_CHUNK_OVERLAP_BYTES,
        ),
        include_tool_private=bool(
            comprehend_config.get("include_tool_private", False)
        ),
    )

    output_path = write_tool_output(OUTPUT_FILE, result)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
