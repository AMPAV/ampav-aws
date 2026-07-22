"""Extract named entities from a text string with AWS Comprehend real time.

Copy the examples directory outside the repo, copy
config/aws_config.example.yaml to config/aws_config.yaml, update the copied
config for your AWS account, then run this script from the copied directory.
"""

from ampav.aws.comprehend_named_entities_realtime import (
    AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
    DEFAULT_CHUNK_OVERLAP_BYTES,
    AwsComprehendNamedEntitiesRealtime,
)

from example_support import load_config, write_tool_output


TEXT = "Dr. Maya Chen visited Indiana University in Bloomington."
OUTPUT_FILE = "Realtime-Text-NamedEntities.yaml"


def main() -> None:
    """Run real-time entity extraction on a short text string."""
    config = load_config()
    aws_config = config.get("aws", {})
    comprehend_config = config.get("comprehend_named_entities_realtime", {})

    tool = AwsComprehendNamedEntitiesRealtime(
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
    result = tool.process(
        TEXT,
        language_code=comprehend_config.get("language_code", "en"),
    )

    output_path = write_tool_output(OUTPUT_FILE, result)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
