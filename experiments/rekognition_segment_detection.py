"""Run and retain a native Amazon Rekognition Video segment-detection probe.

The probe preserves paginated ``GetSegmentDetection`` responses without
introducing an AMPAV schema or treating Rekognition shots as semantic scenes.
Inputs, credentials, and retained run data belong outside the repository.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shlex
import sys
import time
from time import perf_counter
from typing import Any

import boto3
import yaml

from ampav.aws.s3 import parse_s3_uri


TASK = "AMPAV-188"
DEFAULT_SEGMENT_TYPES = ("SHOT", "TECHNICAL_CUE")
DEFAULT_POLLING_INTERVAL = 30.0
DEFAULT_TIMEOUT = 7_200.0
MAX_RESULTS = 1_000
JOB_TAG = "ampav-aws-rekognition-segment"


def parse_args() -> argparse.Namespace:
    """Parse arguments for an opt-in native segment-detection probe."""
    parser = argparse.ArgumentParser(
        description="Probe native Amazon Rekognition Video segment detection.",
    )
    parser.add_argument("input_s3_uri", help="existing S3 video URI")
    parser.add_argument("output_dir", type=Path, help="new retained-run directory")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--profile", help="optional boto3 profile name")
    parser.add_argument("--region", help="optional AWS region")
    parser.add_argument(
        "--segment-type",
        dest="segment_types",
        action="append",
        choices=DEFAULT_SEGMENT_TYPES,
        help="native segment type to request; repeat as needed (default: both)",
    )
    parser.add_argument(
        "--polling-interval",
        type=float,
        default=DEFAULT_POLLING_INTERVAL,
        help=f"seconds between job status polls (default: {DEFAULT_POLLING_INTERVAL:g})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"maximum wait in seconds (default: {DEFAULT_TIMEOUT:g})",
    )
    return parser.parse_args()


def main() -> None:
    """Run the probe and print the retained-output location and summary."""
    result = run_probe(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_probe(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    """Submit, wait for, and retain one native segment-detection job."""
    _validate_args(args)
    _prepare_output_dir(args.output_dir)
    location = parse_s3_uri(args.input_s3_uri)
    segment_types = tuple(args.segment_types or DEFAULT_SEGMENT_TYPES)
    if client is None:
        session = boto3.Session(region_name=args.region, profile_name=args.profile)
        client = session.client("rekognition")
    effective_region = getattr(getattr(client, "meta", None), "region_name", None) or args.region

    request = build_start_segment_detection_request(location.bucket, location.key, segment_types)
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    start_response = client.start_segment_detection(**request)
    _write_json(args.output_dir / "start_response.json", start_response)
    job_id = _require_job_id(start_response)

    terminal_response, status_history = wait_for_terminal_response(
        client,
        job_id,
        polling_interval=args.polling_interval,
        timeout=args.timeout,
    )
    elapsed_seconds = perf_counter() - started
    completed_at = datetime.now(timezone.utc)
    _write_json(args.output_dir / "status_history.json", status_history)

    status = terminal_response.get("JobStatus")
    if status != "SUCCEEDED":
        _write_json(args.output_dir / "terminal_response.json", terminal_response)
        _write_failure_record(
            args,
            job_id=job_id,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed_seconds,
            effective_region=effective_region,
            status=status,
            message=terminal_response.get("StatusMessage"),
        )
        raise RuntimeError(
            f"Rekognition segment-detection job {job_id} ended with {status!r}: "
            f"{terminal_response.get('StatusMessage') or 'no status message'}"
        )

    pages = fetch_all_segment_pages(client, job_id, first_response=terminal_response)
    _write_json(args.output_dir / "native_segment_detection_pages.json", pages)
    summary = summarize_pages(pages)
    manifest = {
        "task": TASK,
        "recorded_at": completed_at.isoformat(),
        "fixture_id": args.fixture_id,
        "native": {
            "service": "Amazon Rekognition Video",
            "start_operation": "StartSegmentDetection",
            "result_operation": "GetSegmentDetection",
            "boto3_version": _package_version("boto3"),
            "botocore_version": _package_version("botocore"),
        },
        "effective_parameters": {
            "input_s3_uri": args.input_s3_uri,
            "segment_types": list(segment_types),
            "job_tag": JOB_TAG,
            "region": effective_region,
            "polling_interval_seconds": args.polling_interval,
            "timeout_seconds": args.timeout,
        },
        "timing": {
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "poll_count": len(status_history),
        },
        "response_summary": summary,
        "cleanup": (
            "The caller-owned S3 video is retained for related Rekognition probes. "
            "The native segment-detection API does not provide a job-deletion operation."
        ),
        "cost_note": "Amazon Rekognition Video analysis is billable; inspect account billing separately.",
    }
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (args.output_dir / "input_ref.txt").write_text(
        f"fixture_id={args.fixture_id}\n"
        f"input_s3_uri={args.input_s3_uri}\n"
        "cleanup=Input object retained for the four approved Rekognition probes.\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n", encoding="utf-8"
    )
    (args.output_dir / "observations.md").write_text(
        "# Observations\n\n"
        "Review the native pages for shot versus technical-cue segments, timing, "
        "frame numbers, timecodes, confidence, and technical-cue subtype. "
        "Do not treat Rekognition shots as semantic scenes.\n",
        encoding="utf-8",
    )
    return {
        "output_dir": str(args.output_dir.resolve()),
        "job_id": job_id,
        "summary": summary,
        "manifest": str((args.output_dir / "manifest.yaml").resolve()),
    }


def build_start_segment_detection_request(
    bucket: str,
    key: str,
    segment_types: tuple[str, ...],
) -> dict[str, Any]:
    """Build the direct native ``StartSegmentDetection`` request."""
    return {
        "Video": {"S3Object": {"Bucket": bucket, "Name": key}},
        "SegmentTypes": list(segment_types),
        "JobTag": JOB_TAG,
    }


def wait_for_terminal_response(
    client: Any,
    job_id: str,
    *,
    polling_interval: float,
    timeout: float,
    sleep: Any = time.sleep,
    monotonic: Any = time.monotonic,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll the native job until it reaches a terminal status."""
    started = monotonic()
    history: list[dict[str, Any]] = []
    while True:
        response = client.get_segment_detection(
            JobId=job_id,
            MaxResults=MAX_RESULTS,
        )
        if not isinstance(response, dict):
            raise TypeError("GetSegmentDetection must return a mapping")
        history.append(response)
        status = response.get("JobStatus")
        if status in {"SUCCEEDED", "FAILED"}:
            return response, history
        if monotonic() - started > timeout:
            raise TimeoutError(
                f"Rekognition segment-detection job {job_id} did not finish within {timeout} seconds"
            )
        sleep(polling_interval)


def fetch_all_segment_pages(
    client: Any,
    job_id: str,
    *,
    first_response: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return each native response page without merging or reshaping segments."""
    pages = [first_response]
    token = first_response.get("NextToken")
    while token:
        if not isinstance(token, str):
            raise TypeError("GetSegmentDetection NextToken must be a string")
        response = client.get_segment_detection(
            JobId=job_id,
            MaxResults=MAX_RESULTS,
            NextToken=token,
        )
        if not isinstance(response, dict):
            raise TypeError("GetSegmentDetection must return a mapping")
        pages.append(response)
        token = response.get("NextToken")
    return pages


def summarize_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Report structural native fields for direct qualitative review."""
    segments = [
        segment
        for page in pages
        for segment in page.get("Segments", [])
        if isinstance(segment, dict)
    ]
    segment_types = Counter(
        str(segment.get("Type", "<missing>")) for segment in segments
    )
    technical_cues = Counter(
        str(segment.get("TechnicalCueSegment", {}).get("Type", "<missing>"))
        for segment in segments
        if segment.get("Type") == "TECHNICAL_CUE"
        and isinstance(segment.get("TechnicalCueSegment"), dict)
    )
    return {
        "page_count": len(pages),
        "segment_count": len(segments),
        "segment_types": dict(sorted(segment_types.items())),
        "technical_cue_types": dict(sorted(technical_cues.items())),
        "native_field_review": [
            "Type",
            "StartTimestampMillis",
            "EndTimestampMillis",
            "DurationFrames",
            "StartFrameNumber",
            "EndFrameNumber",
            "StartTimecodeSMPTE",
            "EndTimecodeSMPTE",
            "DurationSMPTE",
            "ShotSegment.Confidence",
            "TechnicalCueSegment.Type",
            "TechnicalCueSegment.Confidence",
        ],
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.polling_interval <= 0:
        raise ValueError("polling_interval must be greater than 0")
    if args.timeout <= 0:
        raise ValueError("timeout must be greater than 0")


def _prepare_output_dir(output_dir: Path) -> None:
    """Create an output directory or accept a pre-created empty log directory."""
    if output_dir.is_symlink():
        raise FileExistsError(f"output directory must not be a symlink: {output_dir}")
    if output_dir.exists():
        if not output_dir.is_dir() or any(output_dir.iterdir()):
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        return
    output_dir.mkdir(parents=True)


def _require_job_id(response: dict[str, Any]) -> str:
    job_id = response.get("JobId")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("StartSegmentDetection response must contain a non-empty JobId")
    return job_id


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_failure_record(
    args: argparse.Namespace,
    *,
    job_id: str,
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
    effective_region: str | None,
    status: Any,
    message: Any,
) -> None:
    (args.output_dir / "failure.yaml").write_text(
        yaml.safe_dump(
            {
                "task": TASK,
                "job_id": job_id,
                "status": status,
                "message": message,
                "recorded_at": completed_at.isoformat(),
                "started_at": started_at.isoformat(),
                "elapsed_seconds": elapsed_seconds,
                "region": effective_region,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
