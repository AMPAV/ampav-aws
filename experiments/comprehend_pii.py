"""Run and retain a native AWS Comprehend PII detection probe.

The probe deliberately preserves AWS's response shape and does not define an
AMPAV PII schema. Input text, credentials, and generated evidence remain in
the caller's unversioned ``.work`` directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import sys
from time import perf_counter
from typing import Any

import yaml

from ampav.aws.comprehend_pii_realtime import AwsComprehendPiiRealtime


TASK = "AMPAV-187"


def parse_args() -> argparse.Namespace:
    """Parse native PII-probe arguments."""
    parser = argparse.ArgumentParser(
        description="Probe native AWS Comprehend PII detection.",
    )
    parser.add_argument("input", type=Path, help="UTF-8 source text")
    parser.add_argument("output_dir", type=Path, help="new retained-run directory")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--language-code", default="en")
    parser.add_argument("--profile", help="optional boto3 profile name")
    parser.add_argument("--region", help="optional AWS region")
    return parser.parse_args()


def main() -> None:
    """Run the native PII probe and report its retained output location."""
    result = run_probe(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Call ``DetectPiiEntities`` and retain its complete native response."""
    _validate_new_output_dir(args.output_dir)
    text = args.input.read_text(encoding="utf-8")
    text_bytes = text.encode("utf-8")
    if not text.strip():
        raise ValueError("input text must not be empty")

    tool = AwsComprehendPiiRealtime(
        region_name=args.region,
        profile_name=args.profile,
    )
    effective_region = (
        getattr(getattr(tool.comprehend_client, "meta", None), "region_name", None)
        or args.region
    )
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    response = tool.process(text, language_code=args.language_code)
    elapsed_seconds = perf_counter() - started
    completed_at = datetime.now(timezone.utc)

    args.output_dir.mkdir(parents=True)
    _write_json(args.output_dir / "native_pii_entities.json", response)
    entity_types = _entity_types(response)
    manifest = {
        "task": TASK,
        "recorded_at": completed_at.isoformat(),
        "fixture_id": args.fixture_id,
        "input": {
            "utf8_bytes": len(text_bytes),
            "sha256": hashlib.sha256(text_bytes).hexdigest(),
        },
        "native": {
            "service": "Amazon Comprehend",
            "operation": "DetectPiiEntities",
            "boto3_version": _package_version("boto3"),
            "botocore_version": _package_version("botocore"),
        },
        "effective_parameters": {
            "language_code": args.language_code,
            "region": effective_region,
        },
        "timing": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
        },
        "response_summary": {
            "entity_count": len(response.get("Entities", [])),
            "entity_types": entity_types,
            "score_semantics": "AWS provider score; not normalized by AMPAV",
        },
        "cleanup": "No remote artifacts were created by the synchronous API.",
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        "fixture_id=" + args.fixture_id + "\n"
        "input=caller-supplied local UTF-8 text; path intentionally not retained\n"
        "cleanup=No remote artifacts were created by the synchronous API.\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\n"
        "Review the native response for entity types, offsets, and provider score "
        "semantics before deciding whether to adopt a later integration.\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(args.output_dir.resolve()),
        "entity_count": len(response.get("Entities", [])),
        "entity_types": entity_types,
        "manifest": str((args.output_dir / "manifest.yaml").resolve()),
    }


def _entity_types(response: dict[str, Any]) -> list[str]:
    """Return distinct native PII types while retaining the complete response."""
    entities = response.get("Entities", [])
    if not isinstance(entities, list):
        raise ValueError("DetectPiiEntities response must contain an Entities list")
    return sorted(
        {
            entity["Type"]
            for entity in entities
            if isinstance(entity, dict) and isinstance(entity.get("Type"), str)
        }
    )


def _validate_new_output_dir(output_dir: Path) -> None:
    """Require a new directory so selected evidence is never overwritten."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"output directory already exists: {output_dir}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    """Persist a readable native JSON response."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str | None:
    """Return installed package metadata when it is available."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
