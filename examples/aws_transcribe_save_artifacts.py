"""Example client-side artifact persistence for an AWS Transcribe run.

The library returns a ToolOutput and does not manage run directories. This
example shows how an application can write the public ToolOutput plus private
AWS troubleshooting data to its own artifact folder.
"""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import time

from ampav.aws.transcribe import transcribe_file


def main() -> None:
    """Run the artifact-persistence example from command-line arguments."""
    parser = ArgumentParser(description="Transcribe media and save selected run artifacts locally.")
    parser.add_argument("media")
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    args = parser.parse_args()

    run_dir = args.run_dir / str(int(time()))
    run_dir.mkdir(parents=True, exist_ok=False)

    result = transcribe_file(
        args.media,
        output_bucket=args.output_bucket,
        profile_name=args.profile,
        region_name=args.region,
    )

    (run_dir / "tool_output.yaml").write_text(result.model_dump_yaml(sort_keys=False))
    if isinstance(result.tool_private, dict):
        (run_dir / "tool_private.json").write_text(json.dumps(result.tool_private, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
