"""Run and retain a native Amazon Rekognition Video text-detection probe."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import statistics
import sys
import time
from time import perf_counter
from typing import Any

import boto3
import yaml

from ampav.aws.s3 import parse_s3_uri


TASK = "AMPAV-189"
JOB_TAG = "ampav-aws-rekognition-text"
MAX_RESULTS = 1_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe native Rekognition Video text detection.")
    parser.add_argument("input_s3_uri")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--polling-interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=7200)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run_probe(parse_args()), indent=2, ensure_ascii=False))


def run_probe(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    """Submit, wait for, and retain one native text-detection job."""
    _validate_args(args)
    _prepare_output_dir(args.output_dir)
    location = parse_s3_uri(args.input_s3_uri)
    if client is None:
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        client = session.client("rekognition")
    request = {
        "Video": {"S3Object": {"Bucket": location.bucket, "Name": location.key}},
        "JobTag": JOB_TAG,
    }
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    start_response = client.start_text_detection(**request)
    _write_json(args.output_dir / "start_response.json", start_response)
    job_id = _job_id(start_response)
    terminal, history = wait_for_terminal_response(
        client, job_id, polling_interval=args.polling_interval, timeout=args.timeout
    )
    completed_at = datetime.now(timezone.utc)
    _write_json(args.output_dir / "status_history.json", history)
    if terminal.get("JobStatus") != "SUCCEEDED":
        _write_json(args.output_dir / "terminal_response.json", terminal)
        raise RuntimeError(
            f"Rekognition text-detection job {job_id} ended with {terminal.get('JobStatus')!r}: "
            f"{terminal.get('StatusMessage') or 'no status message'}"
        )
    pages = fetch_all_text_pages(client, job_id, first_response=terminal)
    _write_json(args.output_dir / "native_text_detection_pages.json", pages)
    summary = summarize_pages(pages)
    manifest = {
        "task": TASK,
        "recorded_at": completed_at.isoformat(),
        "fixture_id": args.fixture_id,
        "native": {
            "service": "Amazon Rekognition Video",
            "start_operation": "StartTextDetection",
            "result_operation": "GetTextDetection",
            "boto3_version": _version("boto3"),
            "botocore_version": _version("botocore"),
            "text_model_versions": sorted(
                {str(page["TextModelVersion"]) for page in pages if page.get("TextModelVersion")}
            ),
        },
        "effective_parameters": {
            "input_s3_uri": args.input_s3_uri,
            "job_tag": JOB_TAG,
            "region": getattr(getattr(client, "meta", None), "region_name", None) or args.region,
            "polling_interval_seconds": args.polling_interval,
            "timeout_seconds": args.timeout,
        },
        "timing": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "poll_count": len(history),
        },
        "response_summary": summary,
        "cleanup": "The caller-approved S3 video remains available for the related Rekognition probes. Rekognition does not provide a deletion operation for this job record.",
        "cost_note": "Amazon Rekognition Video analysis is billable; inspect account billing separately.",
    }
    (args.output_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (args.output_dir / "input_ref.txt").write_text(
        f"fixture_id={args.fixture_id}\ninput_s3_uri={args.input_s3_uri}\ncleanup=Input object retained for the approved Rekognition probes.\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8")
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\nReview word and line detections, timestamps, confidence, geometry, and adjacent-frame repetition. Native repeated detections are preserved; do not deduplicate them in Phase 1.\n",
        encoding="utf-8",
    )
    return {"output_dir": str(args.output_dir.resolve()), "job_id": job_id, "summary": summary}


def wait_for_terminal_response(
    client: Any, job_id: str, *, polling_interval: float, timeout: float,
    sleep: Any = time.sleep, monotonic: Any = time.monotonic,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll the direct native job until it reaches a terminal state."""
    started = monotonic()
    history: list[dict[str, Any]] = []
    while True:
        response = client.get_text_detection(JobId=job_id, MaxResults=MAX_RESULTS)
        if not isinstance(response, dict):
            raise TypeError("GetTextDetection must return a mapping")
        history.append(response)
        if response.get("JobStatus") in {"SUCCEEDED", "FAILED"}:
            return response, history
        if monotonic() - started > timeout:
            raise TimeoutError(f"Rekognition text-detection job {job_id} did not finish within {timeout} seconds")
        sleep(polling_interval)


def fetch_all_text_pages(client: Any, job_id: str, *, first_response: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve each native result page rather than merging its detections."""
    pages = [first_response]
    token = first_response.get("NextToken")
    while token:
        if not isinstance(token, str):
            raise TypeError("GetTextDetection NextToken must be a string")
        response = client.get_text_detection(JobId=job_id, MaxResults=MAX_RESULTS, NextToken=token)
        if not isinstance(response, dict):
            raise TypeError("GetTextDetection must return a mapping")
        pages.append(response)
        token = response.get("NextToken")
    return pages


def summarize_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Report direct-review diagnostics while retaining the unmodified native pages."""
    detections = [
        item for page in pages for item in page.get("TextDetections", [])
        if isinstance(item, dict) and isinstance(item.get("TextDetection"), dict)
    ]
    values = [item["TextDetection"] for item in detections]
    by_type = Counter(str(item.get("Type", "<missing>")) for item in values)
    words = [str(item["DetectedText"]) for item in values if isinstance(item.get("DetectedText"), str)]
    confidences = [float(item["Confidence"]) for item in values if isinstance(item.get("Confidence"), (int, float))]
    timestamps = [int(item["Timestamp"]) for item in detections if isinstance(item.get("Timestamp"), int)]
    repeated = Counter(value.casefold().strip() for value in words if value.strip())
    return {
        "page_count": len(pages),
        "detection_count": len(detections),
        "detection_types": dict(sorted(by_type.items())),
        "unique_detected_text_count": len(repeated),
        "top_repeated_detected_text": [
            {"text": text, "occurrences": count}
            for text, count in repeated.most_common(20)
        ],
        "timestamp_millis": _range(timestamps),
        "confidence": _range(confidences, median=True),
        "geometry_presence": sum(isinstance(item.get("Geometry"), dict) for item in values),
        "native_field_review": ["Timestamp", "DetectedText", "Type", "Confidence", "Geometry.BoundingBox", "Geometry.Polygon", "Id", "ParentId"],
    }


def _range(values: list[float] | list[int], *, median: bool = False) -> dict[str, float | int] | None:
    if not values:
        return None
    result: dict[str, float | int] = {"minimum": min(values), "maximum": max(values)}
    if median:
        result["median"] = statistics.median(values)
    return result


def _prepare_output_dir(output_dir: Path) -> None:
    if output_dir.is_symlink():
        raise FileExistsError(f"output directory must not be a symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        return
    output_dir.mkdir(parents=True)


def _validate_args(args: argparse.Namespace) -> None:
    if args.polling_interval <= 0 or args.timeout <= 0:
        raise ValueError("polling_interval and timeout must be greater than 0")


def _job_id(response: dict[str, Any]) -> str:
    value = response.get("JobId")
    if not isinstance(value, str) or not value:
        raise ValueError("StartTextDetection response must contain a non-empty JobId")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
