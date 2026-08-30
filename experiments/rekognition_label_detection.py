"""Run and retain a native Amazon Rekognition Video label-detection probe."""

from __future__ import annotations

from collections import Counter
from typing import Any


TASK = "AMPAV-191"
JOB_TAG = "ampav-aws-rekognition-label"
MAX_RESULTS = 1_000


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
