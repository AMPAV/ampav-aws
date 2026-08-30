"""Native Amazon Rekognition Video text-detection wrapper for Phase 1."""

from __future__ import annotations

import time
from typing import Any

import boto3

from .s3 import parse_s3_uri


MAX_RESULTS = 1_000
"""Largest native result-page size requested by the Phase 1 wrapper."""


class AwsRekognitionVideoTextDetection:
    """Thin wrapper around Rekognition Video text-detection operations.

    This Phase 1 API deliberately returns the unmodified AWS response mappings
    and preserves page boundaries.  An AMPAV OCR conversion is deferred until
    the native timing, geometry, and repetition semantics have been reviewed.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        rekognition_client: Any | None = None,
        polling_interval: float = 30,
        timeout: float | None = 7200,
    ) -> None:
        """Configure Rekognition with boto3 settings or an injected client.

        Args:
            region_name: Optional AWS region used when creating a boto3 session.
            profile_name: Optional boto3 profile used for authentication.
            session: Optional injected boto3-compatible session.
            rekognition_client: Optional injected Rekognition client.
            polling_interval: Seconds between native job-status requests.
            timeout: Maximum wait time for :meth:`process`, or ``None``.
        """
        if polling_interval <= 0:
            raise ValueError("polling_interval must be greater than 0")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")
        if session is None and rekognition_client is None:
            session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.rekognition_client = rekognition_client or session.client("rekognition")
        self.polling_interval = polling_interval
        self.timeout = timeout

    def start(self, input_s3_uri: str, *, job_tag: str | None = None) -> dict[str, Any]:
        """Start native ``StartTextDetection`` for an existing S3 video.

        The AWS start response is returned unchanged.  This wrapper never
        uploads or deletes the caller-owned video object.
        """
        location = parse_s3_uri(input_s3_uri)
        request: dict[str, Any] = {
            "Video": {"S3Object": {"Bucket": location.bucket, "Name": location.key}},
        }
        if job_tag is not None:
            request["JobTag"] = job_tag
        response = self.rekognition_client.start_text_detection(**request)
        return _response_mapping(response, "StartTextDetection")

    def get_page(
        self,
        job_id: str,
        *,
        next_token: str | None = None,
        max_results: int = MAX_RESULTS,
    ) -> dict[str, Any]:
        """Return one unmodified native ``GetTextDetection`` result page."""
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("job_id must be a non-empty string")
        if not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("max_results must be a positive integer")
        request: dict[str, Any] = {"JobId": job_id, "MaxResults": max_results}
        if next_token is not None:
            request["NextToken"] = next_token
        response = self.rekognition_client.get_text_detection(**request)
        return _response_mapping(response, "GetTextDetection")

    def wait_for_terminal_response(self, job_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Poll native result pages until the job completes or fails.

        The terminal page and every polling response are preserved separately;
        callers can retain the history as evidence without recreating requests.
        """
        started = time.monotonic()
        history: list[dict[str, Any]] = []
        while True:
            response = self.get_page(job_id)
            history.append(response)
            if response.get("JobStatus") in {"SUCCEEDED", "FAILED"}:
                return response, history
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                raise TimeoutError(f"Rekognition text-detection job {job_id} did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)

    def get_all_pages(self, job_id: str, *, first_response: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return all unmodified native result pages for a successful job."""
        first = first_response or self.get_page(job_id)
        status = first.get("JobStatus")
        if status == "FAILED":
            raise RuntimeError(_failure_message(job_id, first))
        if status != "SUCCEEDED":
            return []
        pages = [first]
        token = first.get("NextToken")
        while token:
            if not isinstance(token, str):
                raise TypeError("GetTextDetection NextToken must be a string")
            page = self.get_page(job_id, next_token=token)
            pages.append(page)
            token = page.get("NextToken")
        return pages

    def process(self, input_s3_uri: str, *, job_tag: str | None = None) -> list[dict[str, Any]]:
        """Run text detection and return its unmodified native result pages."""
        start_response = self.start(input_s3_uri, job_tag=job_tag)
        job_id = job_id_from_start_response(start_response)
        terminal, _ = self.wait_for_terminal_response(job_id)
        return self.get_all_pages(job_id, first_response=terminal)


def job_id_from_start_response(response: dict[str, Any]) -> str:
    """Extract the required opaque Rekognition job identifier from a start response."""
    value = response.get("JobId")
    if not isinstance(value, str) or not value:
        raise ValueError("StartTextDetection response must contain a non-empty JobId")
    return value


def _response_mapping(response: Any, operation: str) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise TypeError(f"{operation} must return a mapping")
    return response


def _failure_message(job_id: str, response: dict[str, Any]) -> str:
    reason = response.get("StatusMessage") or "no status message"
    return f"Rekognition text-detection job {job_id} failed: {reason}"
