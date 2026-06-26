"""Example client-side artifact persistence for an AWS Transcribe run.

The library returns a ToolOutput and does not manage run directories. This
example shows how an application can write the public ToolOutput plus private
AWS troubleshooting data to its own artifact folder.
"""

from argparse import ArgumentParser
from datetime import datetime, timezone
import json
from pathlib import Path

from ampav.aws.transcribe import AwsTranscribe
from ampav.aws.transcribe import safe_job_part


def main() -> None:
    """Run the artifact-persistence example from command-line arguments."""
    parser = ArgumentParser(description="Transcribe existing S3 media and save selected run artifacts locally.")
    parser.add_argument("media_uri")
    parser.add_argument("--output-s3-uri")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    media_stem = safe_job_part(Path(str(args.media_uri).rstrip("/")).stem) or "media"
    run_dir = args.run_dir / f"{timestamp}-aws-transcribe-{media_stem}"
    run_dir.mkdir(parents=True, exist_ok=False)

    client = AwsTranscribe(profile_name=args.profile, region_name=args.region)
    result = client.process(
        args.media_uri,
        output_s3_uri=args.output_s3_uri,
    )

    (run_dir / "tool_output.yaml").write_text(result.model_dump_yaml(sort_keys=False))
    if isinstance(result.tool_private, dict):
        (run_dir / "tool_private.json").write_text(json.dumps(result.tool_private, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
