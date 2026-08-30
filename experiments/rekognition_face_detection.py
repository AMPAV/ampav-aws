"""Run and retain a native Amazon Rekognition Video face-detection probe."""

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
from time import perf_counter
from typing import Any

import yaml

from ampav.aws.rekognition_face_detection import (
    AwsRekognitionFaceDetection,
    job_id_from_start_response,
)


TASK = "AMPAV-190"
JOB_TAG = "ampav-aws-rekognition-face"


def parse_args() -> argparse.Namespace:
    """Parse arguments for an opt-in native face-detection probe."""
    parser = argparse.ArgumentParser(description="Probe native Rekognition Video face detection.")
    parser.add_argument("input_s3_uri")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--region")
    parser.add_argument("--polling-interval", type=float, default=30)
    parser.add_argument("--timeout", type=float, default=7200)
    return parser.parse_args()


def main() -> None:
    """Run the retained native face-detection probe."""
    print(json.dumps(run_probe(parse_args()), indent=2, ensure_ascii=False))


def run_probe(args: argparse.Namespace, *, tool: AwsRekognitionFaceDetection | None = None) -> dict[str, Any]:
    """Submit, wait for, and retain one native face-detection job through the wrapper."""
    _validate_args(args)
    _prepare_output_dir(args.output_dir)
    if tool is None:
        tool = AwsRekognitionFaceDetection(
            profile_name=args.profile,
            region_name=args.region,
            polling_interval=args.polling_interval,
            timeout=args.timeout,
        )
    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    start_response = tool.start(args.input_s3_uri, job_tag=JOB_TAG)
    _write_json(args.output_dir / "start_response.json", start_response)
    job_id = job_id_from_start_response(start_response)
    terminal, history = tool.wait_for_terminal_response(job_id)
    completed_at = datetime.now(timezone.utc)
    _write_json(args.output_dir / "status_history.json", history)
    if terminal.get("JobStatus") != "SUCCEEDED":
        _write_json(args.output_dir / "terminal_response.json", terminal)
        raise RuntimeError(
            f"Rekognition face-detection job {job_id} ended with {terminal.get('JobStatus')!r}: "
            f"{terminal.get('StatusMessage') or 'no status message'}"
        )
    pages = tool.get_all_pages(job_id, first_response=terminal)
    _write_json(args.output_dir / "native_face_detection_pages.json", pages)
    summary = summarize_pages(pages)
    manifest = {
        "task": TASK,
        "recorded_at": completed_at.isoformat(),
        "fixture_id": args.fixture_id,
        "native": {
            "service": "Amazon Rekognition Video",
            "start_operation": "StartFaceDetection",
            "result_operation": "GetFaceDetection",
            "boto3_version": _version("boto3"),
            "botocore_version": _version("botocore"),
            "face_model_versions": sorted({str(page["FaceModelVersion"]) for page in pages if page.get("FaceModelVersion")}),
        },
        "effective_parameters": {
            "input_s3_uri": args.input_s3_uri,
            "job_tag": JOB_TAG,
            "region": getattr(getattr(tool.rekognition_client, "meta", None), "region_name", None) or args.region,
            "polling_interval_seconds": args.polling_interval,
            "timeout_seconds": args.timeout,
            "identity_operations": "not requested",
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
        "# Observations\n\nReview native face records for timestamps, bounding boxes, landmarks, pose, quality, and visual attributes. This Phase 1 probe does not infer identities, track people, or create face collections.\n",
        encoding="utf-8",
    )
    return {"output_dir": str(args.output_dir.resolve()), "job_id": job_id, "summary": summary}


def summarize_pages(pages: list[dict[str, Any]]) -> dict[str, Any]:
    """Report native face fields for review without tracking or identity inference."""
    records = [record for page in pages for record in page.get("Faces", []) if isinstance(record, dict)]
    faces = [record["Face"] for record in records if isinstance(record.get("Face"), dict)]
    timestamps = [int(record["Timestamp"]) for record in records if isinstance(record.get("Timestamp"), int)]
    confidences = [float(face["Confidence"]) for face in faces if isinstance(face.get("Confidence"), (int, float))]
    per_timestamp = Counter(timestamps)
    return {
        "page_count": len(pages),
        "face_record_count": len(records),
        "timestamp_millis": _range(timestamps),
        "confidence": _range(confidences, median=True),
        "timestamps_with_faces": len(per_timestamp),
        "max_faces_at_one_timestamp": max(per_timestamp.values()) if per_timestamp else 0,
        "records_with_bounding_box": sum(isinstance(face.get("BoundingBox"), dict) for face in faces),
        "records_with_landmarks": sum(bool(face.get("Landmarks")) for face in faces),
        "records_with_emotions": sum(bool(face.get("Emotions")) for face in faces),
        "native_field_review": [
            "Timestamp", "Face.BoundingBox", "Face.Confidence", "Face.Landmarks",
            "Face.Pose", "Face.Quality", "Face.Emotions", "Face.Smile",
            "Face.Eyeglasses", "Face.Sunglasses", "Face.Gender", "Face.Beard",
            "Face.Mustache", "Face.EyesOpen", "Face.MouthOpen", "Face.FaceOccluded",
        ],
    }


def _range(values: list[float] | list[int], *, median: bool = False) -> dict[str, float | int] | None:
    if not values:
        return None
    result: dict[str, float | int] = {"minimum": min(values), "maximum": max(values)}
    if median:
        result["median"] = statistics.median(values)
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if args.polling_interval <= 0 or args.timeout <= 0:
        raise ValueError("polling_interval and timeout must be greater than 0")


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
