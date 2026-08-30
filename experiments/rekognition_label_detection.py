"""Run and retain a native Amazon Rekognition Video label-detection probe."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import sys
from time import perf_counter
from typing import Any

import yaml

from ampav.aws.rekognition_label_detection import (
    AwsRekognitionLabelDetection,
    job_id_from_start_response,
)


TASK = "AMPAV-191"
JOB_TAG = "ampav-aws-rekognition-label"
MAX_RESULTS = 1_000


def parse_args() -> argparse.Namespace:
    """Parse arguments for an opt-in native label-detection probe."""
    parser = argparse.ArgumentParser(description="Probe native Rekognition Video label detection.")
    parser.add_argument("input_s3_uri")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--min-confidence", type=float, default=50.0)
    parser.add_argument("--polling-interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=7200)
    return parser.parse_args()


def main() -> None:
    """Run the retained native label-detection probe."""
    print(json.dumps(run_probe(parse_args()), indent=2, ensure_ascii=False))


def run_probe(args: argparse.Namespace, *, tool: AwsRekognitionLabelDetection | None = None) -> dict[str, Any]:
    """Submit, wait for, and retain one native label-detection job through the wrapper."""
    _validate_args(args)
    _prepare_output_dir(args.output_dir)
    if tool is None:
        tool = AwsRekognitionLabelDetection(profile_name=args.profile, region_name=args.region,
                                             polling_interval=args.polling_interval, timeout=args.timeout)
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    start_response = tool.start(args.input_s3_uri, min_confidence=args.min_confidence, job_tag=JOB_TAG)
    _write_json(args.output_dir / "start_response.json", start_response)
    job_id = job_id_from_start_response(start_response)
    terminal, history = tool.wait_for_terminal_response(job_id)
    completed_at = datetime.now(timezone.utc)
    _write_json(args.output_dir / "status_history.json", history)
    if terminal.get("JobStatus") != "SUCCEEDED":
        _write_json(args.output_dir / "terminal_response.json", terminal)
        raise RuntimeError(f"Rekognition label-detection job {job_id} ended with {terminal.get('JobStatus')!r}: {terminal.get('StatusMessage') or 'no status message'}")
    pages = tool.get_all_pages(job_id, first_response=terminal)
    _write_json(args.output_dir / "native_label_detection_pages.json", pages)
    summary = summarize_pages(pages)
    manifest = {
        "task": TASK,
        "recorded_at": completed_at.isoformat(),
        "fixture_id": args.fixture_id,
        "native": {"service": "Amazon Rekognition Video", "start_operation": "StartLabelDetection", "result_operation": "GetLabelDetection", "boto3_version": _version("boto3"), "botocore_version": _version("botocore")},
        "effective_parameters": {"input_s3_uri": args.input_s3_uri, "min_confidence": args.min_confidence, "job_tag": JOB_TAG, "region": getattr(getattr(tool.rekognition_client, "meta", None), "region_name", None) or args.region, "polling_interval_seconds": args.polling_interval, "timeout_seconds": args.timeout},
        "timing": {"started_at": started_at.isoformat(), "completed_at": completed_at.isoformat(), "elapsed_seconds": perf_counter() - started, "poll_count": len(history)},
        "response_summary": summary,
        "cleanup": "The caller-approved S3 video remains available for the related Rekognition probes. Rekognition does not provide a deletion operation for this job record.",
        "cost_note": "Amazon Rekognition Video analysis is billable; inspect account billing separately.",
    }
    (args.output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (args.output_dir / "input_ref.txt").write_text(f"fixture_id={args.fixture_id}\ninput_s3_uri={args.input_s3_uri}\ncleanup=Input object retained for the approved Rekognition probes.\n", encoding="utf-8")
    (args.output_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    (args.output_dir / "observations.md").write_text("# Observations\n\nReview native labels, timing, confidence, instances, parents, aliases, and categories. Native labels are preserved without an AMPAV aggregation policy in Phase 1.\n", encoding="utf-8")
    return {"output_dir": str(args.output_dir.resolve()), "job_id": job_id, "summary": summary}


def build_start_request(bucket: str, key: str, *, min_confidence: float = 50.0) -> dict[str, Any]:
    """Build the direct native label-detection request for stored video."""
    if not 0 <= min_confidence <= 100:
        raise ValueError("min_confidence must be between 0 and 100")
    return {
        "Video": {"S3Object": {"Bucket": bucket, "Name": key}},
        "MinConfidence": min_confidence,
        "JobTag": JOB_TAG,
    }


def fetch_all_label_pages(client: Any, job_id: str, *, first_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve native pages rather than merging or aggregating labels."""
    pages = [first_response]
    token = first_response.get("NextToken")
    while token:
        if not isinstance(token, str):
            raise TypeError("GetLabelDetection NextToken must be a string")
        response = client.get_label_detection(JobId=job_id, MaxResults=MAX_RESULTS, NextToken=token)
        if not isinstance(response, dict):
            raise TypeError("GetLabelDetection must return a mapping")
        pages.append(response)
        token = response.get("NextToken")
    return pages


def summarize_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Return review diagnostics without defining an AMPAV aggregation policy."""
    records = [record for page in pages for record in page.get("Labels", []) if isinstance(record, dict)]
    labels = [record["Label"] for record in records if isinstance(record.get("Label"), dict)]
    names = [str(label["Name"]) for label in labels if isinstance(label.get("Name"), str)]
    scores = [float(label["Confidence"]) for label in labels if isinstance(label.get("Confidence"), (float, int))]
    timestamps = [int(record["Timestamp"]) for record in records if isinstance(record.get("Timestamp"), int)]
    parent_names = [
        str(parent["Name"]) for label in labels for parent in label.get("Parents", [])
        if isinstance(parent, dict) and isinstance(parent.get("Name"), str)
    ]
    return {
        "page_count": len(pages),
        "label_record_count": len(records),
        "unique_label_name_count": len(set(names)),
        "top_label_names": [{"name": name, "occurrences": count} for name, count in Counter(names).most_common(30)],
        "timestamp_millis": _range(timestamps),
        "confidence": _range(scores),
        "records_with_instances": sum(bool(label.get("Instances")) for label in labels),
        "parent_taxonomy": dict(Counter(parent_names).most_common(30)),
        "native_field_review": ["Timestamp", "Label.Name", "Label.Confidence", "Label.Instances", "Label.Parents", "Label.Aliases", "Label.Categories"],
    }


def _range(values: list[float] | list[int]) -> dict[str, float | int] | None:
    return {"minimum": min(values), "maximum": max(values)} if values else None


def _validate_args(args: argparse.Namespace) -> None:
    if args.polling_interval <= 0 or args.timeout <= 0:
        raise ValueError("polling_interval and timeout must be greater than 0")
    if not 0 <= args.min_confidence <= 100:
        raise ValueError("min_confidence must be between 0 and 100")


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise FileExistsError(f"output directory must not be a symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        return
    output_dir.mkdir(parents=True)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
