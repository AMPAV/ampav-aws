"""Extract named entities from an existing S3 text object with AWS Comprehend.

This example uses standard boto3 authentication. Pass --profile/--region or
rely on your environment, AWS config files, or IAM role credentials.
"""

from argparse import ArgumentParser
from pathlib import Path

from ampav.aws.comprehend_named_entities import AwsComprehendNamedEntities, safe_job_part

from example_support import write_tool_output


def main() -> None:
    """Run the existing-S3 Comprehend named-entities example."""
    parser = ArgumentParser(description="Extract named entities from an existing s3:// text object.")
    parser.add_argument("input_s3_uri")
    parser.add_argument("--output-s3-uri")
    parser.add_argument("--data-access-role-arn", required=True)
    parser.add_argument("--entity-recognizer-arn")
    parser.add_argument("--output-kms-key-id")
    parser.add_argument("--volume-kms-key-id")
    parser.add_argument("--delete-user-owned-outputs", action="store_true")
    parser.add_argument("--include-tool-private", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--polling-interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--job-name-suffix")
    parser.add_argument("--language-code", default="en")
    args = parser.parse_args()

    client = AwsComprehendNamedEntities(
        region_name=args.region,
        profile_name=args.profile,
        data_access_role_arn=args.data_access_role_arn,
        entity_recognizer_arn=args.entity_recognizer_arn,
        output_kms_key_id=args.output_kms_key_id,
        volume_kms_key_id=args.volume_kms_key_id,
        delete_user_owned_outputs=args.delete_user_owned_outputs,
        include_tool_private=args.include_tool_private,
        polling_interval=args.polling_interval,
        timeout=args.timeout,
    )

    input_stem = safe_job_part(Path(str(args.input_s3_uri).rstrip("/")).stem) or "S3Input"
    result = client.process(
        args.input_s3_uri,
        output_s3_uri=args.output_s3_uri,
        language_code=args.language_code,
        job_name_suffix=args.job_name_suffix or input_stem,
    )

    output_path = write_tool_output(f"{input_stem}-NamedEntities.yaml", result)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
