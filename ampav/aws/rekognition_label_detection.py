"""Native Amazon Rekognition Video label-detection wrapper for Phase 1."""

from __future__ import annotations

import time
from typing import Any

import boto3

from .s3 import parse_s3_uri


MAX_RESULTS = 1_000


class AwsRekognitionLabelDetection:
    """Thin wrapper around Rekognition Video label-detection operations.

    This Phase 1 wrapper deliberately returns each unmodified native result
    page. It does not aggregate labels or convert AWS taxonomy into AMPAV.
    """

    def __init__(self, *, region_name: str | None = None, profile_name: str | None = None,
                 session: Any | None = None, rekognition_client: Any | None = None,
                 polling_interval: float = 30, timeout: float | None = 7200) -> None:
        """Configure native Rekognition clients and polling behavior."""
        if polling_interval <= 0:
            raise ValueError("polling_interval must be greater than 0")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")
        if session is None and rekognition_client is None:
            session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.rekognition_client = rekognition_client or session.client("rekognition")
        self.polling_interval = polling_interval
        self.timeout = timeout

    def start(self, input_s3_uri: str, *, min_confidence: float = 50.0,
              job_tag: str | None = None) -> dict[str, Any]:
        """Start native ``StartLabelDetection`` for an existing S3 video."""
        if not 0 <= min_confidence <= 100:
            raise ValueError("min_confidence must be between 0 and 100")
        location = parse_s3_uri(input_s3_uri)
        request: dict[str, Any] = {
            "Video": {"S3Object": {"Bucket": location.bucket, "Name": location.key}},
            "MinConfidence": min_confidence,
        }
        if job_tag is not None:
            request["JobTag"] = job_tag
        return _response_mapping(self.rekognition_client.start_label_detection(**request), "StartLabelDetection")

    def get_page(self, job_id: str, *, next_token: str | None = None,
                 max_results: int = MAX_RESULTS) -> dict[str, Any]:
        """Return one unmodified native ``GetLabelDetection`` response page."""
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        request: dict[str, Any] = {"JobId": job_id, "MaxResults": max_results}
        if next_token is not None:
            request["NextToken"] = next_token
        return _response_mapping(self.rekognition_client.get_label_detection(**request), "GetLabelDetection")

    def wait_for_terminal_response(self, job_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Poll native result pages until the job reaches a terminal status."""
        started = time.monotonic()
        history: list[dict[str, Any]] = []
        while True:
            response = self.get_page(job_id)
            history.append(response)
            if response.get("JobStatus") in {"SUCCEEDED", "FAILED"}:
                return response, history
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                raise TimeoutError(f"Rekognition label-detection job {job_id} did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)

    def get_all_pages(self, job_id: str, *, first_response: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return all native result pages without merging their label records."""
        first = first_response or self.get_page(job_id)
        if first.get("JobStatus") == "FAILED":
            raise RuntimeError(_failure_message(job_id, first))
        if first.get("JobStatus") != "SUCCEEDED":
            return []
        pages = [first]
        token = first.get("NextToken")
        while token:
            if not isinstance(token, str):
                raise TypeError("GetLabelDetection NextToken must be a string")
            page = self.get_page(job_id, next_token=token)
            pages.append(page)
            token = page.get("NextToken")
        return pages

    def process(self, input_s3_uri: str, *, min_confidence: float = 50.0,
                job_tag: str | None = None) -> list[dict[str, Any]]:
        """Run label detection and return unmodified native result pages."""
        start_response = self.start(input_s3_uri, min_confidence=min_confidence, job_tag=job_tag)
        job_id = job_id_from_start_response(start_response)
        terminal, _ = self.wait_for_terminal_response(job_id)
        return self.get_all_pages(job_id, first_response=terminal)


def job_id_from_start_response(response: dict[str, Any]) -> str:
    """Extract an opaque job identifier from a native start response."""
    value = response.get("JobId")
    if not isinstance(value, str) or not value:
        raise ValueError("StartLabelDetection response must contain a non-empty JobId")
    return value


def _response_mapping(response: Any, operation: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError(f"{operation} must return a mapping")
    return response


def _failure_message(job_id: str, response: dict[str, Any]) -> str:
    return f"Rekognition label-detection job {job_id} failed: {response.get('StatusMessage') or 'no status message'}"
