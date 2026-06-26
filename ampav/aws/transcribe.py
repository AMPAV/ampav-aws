"""AWS Transcribe async client for AMPAV."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any
from urllib.request import urlopen

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ampav.core.async_tool import AsyncJobStatus, AsyncStatusCode, AsyncTool, ToolError
from ampav.core.schema import ToolOutput

from .errors import AwsTranscribeError, AwsTranscriptSchemaError
from .s3 import S3Location, join_s3_key, parse_s3_uri
from .transcribe_contract import validate_aws_transcript_contract
from .transcribe_conversion import aws_transcript_to_transcript


_UNSET = object()


class StrictModel(BaseModel):
    """Pydantic base model for user-facing settings objects."""

    model_config = ConfigDict(extra="forbid")


class TranscriptionSettings(StrictModel):
    """AWS Transcribe job settings supported by AMPAV."""

    media_format: str | None = None
    language_code: str | None = "en-US"
    identify_language: bool = False
    language_options: list[str] = Field(default_factory=list)
    show_speaker_labels: bool = True
    max_speaker_labels: int = Field(default=10, ge=2, le=30)

    @model_validator(mode="after")
    def validate_language_settings(self) -> "TranscriptionSettings":
        if not self.identify_language and not self.language_code:
            raise ValueError("language_code is required unless identify_language is true")
        return self


class AwsTranscribeStatus(AsyncJobStatus):
    """AWS Transcribe status mapped onto AMPAV async status."""

    aws_status: str
    failure_reason: str | None = None
    transcript_file_uri: str | None = None


class AwsTranscribe(AsyncTool):
    """Low-level AWS Transcribe client.

    The public lifecycle uses opaque `job_id` strings. The tool API expects
    provider-native S3 media URIs; local file upload belongs in CLI/example code.
    """

    polling_interval: float = 30
    timeout: float | None = 7200

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        transcribe_client: Any | None = None,
        s3_client: Any | None = None,
        polling_interval: float = 30,
        timeout: float | None = 7200,
        job_name_prefix: str = "ampav-aws-transcribe",
    ):
        """Create an AWS Transcribe client from boto3 settings or injected clients."""
        if polling_interval <= 0:
            raise ValueError("polling_interval must be greater than 0")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")

        if session is None and (transcribe_client is None or s3_client is None):
            session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.transcribe_client = transcribe_client or session.client("transcribe")
        self.s3_client = s3_client or session.client("s3")
        self.polling_interval = polling_interval
        self.timeout = timeout
        self.job_name_prefix = job_name_prefix
        self._delete_output_by_job_id: dict[str, bool] = {}
        self._owned_output_by_job_id: dict[str, bool] = {}
        self._media_uri_by_job_id: dict[str, str] = {}

    def submit(
        self,
        media_uri: str,
        *,
        output_s3_uri: str | None = None,
        delete_output: bool = False,
        job_name: str | None = None,
        job_name_prefix: str | None = None,
        transcription: TranscriptionSettings | None = None,
    ) -> str:
        """Submit an AWS Transcribe job for media that already exists in S3."""
        parse_s3_uri(media_uri)
        output_location = parse_s3_uri(output_s3_uri) if output_s3_uri else None
        transcription = transcription or TranscriptionSettings()
        prefix = job_name_prefix or self.job_name_prefix
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = job_name or build_job_name(media_uri, prefix, timestamp)
        request = build_start_job_request(
            job_name=job_id,
            media_uri=media_uri,
            output_location=output_location,
            transcription=transcription,
        )

        logging.info("Starting AWS Transcribe job %s", job_id)
        logging.debug("AWS Transcribe request: %s", json.dumps(request, default=str, sort_keys=True))
        self.transcribe_client.start_transcription_job(**request)
        self._delete_output_by_job_id[job_id] = delete_output
        self._owned_output_by_job_id[job_id] = False
        self._media_uri_by_job_id[job_id] = media_uri
        return job_id

    def get_status(self, job_id: str, details: bool = True) -> AwsTranscribeStatus:
        """Return normalized status information for an AWS Transcribe job."""
        raw_job = self._get_job(job_id)
        job_data = raw_job["TranscriptionJob"]
        return _status_from_job_data(job_data)

    def list_jobs(self) -> list[AsyncJobStatus]:
        """Return AWS Transcribe jobs matching this instance's job-name prefix."""
        response = self.transcribe_client.list_transcription_jobs(JobNameContains=self.job_name_prefix)
        summaries = response.get("TranscriptionJobSummaries", [])
        statuses: list[AsyncJobStatus] = []
        for summary in summaries:
            job_id = summary.get("TranscriptionJobName")
            aws_status = summary.get("TranscriptionJobStatus")
            if not isinstance(job_id, str) or not isinstance(aws_status, str):
                continue
            if not job_id.startswith(self.job_name_prefix):
                continue
            statuses.append(
                AwsTranscribeStatus(
                    job_id=job_id,
                    status=map_aws_transcribe_status(aws_status),
                    message=summary.get("FailureReason"),
                    aws_status=aws_status,
                    failure_reason=summary.get("FailureReason"),
                )
            )
        return statuses

    def get_result(self, job_id: str) -> ToolOutput | None:
        """Return AMPAV output when ready; terminal jobs are cleaned up."""
        raw_job = self._get_job(job_id)
        job_data = raw_job["TranscriptionJob"]
        status = _status_from_job_data(job_data)
        if not status.is_done:
            return None

        if status.status != AsyncStatusCode.SUCCEEDED:
            self.cleanup(job_id)
            reason = status.failure_reason or "no failure reason returned"
            raise AwsTranscribeError(job_id, f"failed: {reason}")

        raw_transcript = self._read_transcript_json(job_data)
        output = self._to_tool_output(job_id, raw_job, raw_transcript)
        self._delete_output_if_needed(job_id, raw_job)
        self._delete_job_record(job_id)
        return output

    def cleanup(self, job_id: str) -> None:
        """Clean up output owned/requested by this tool and delete the job record."""
        raw_job = self._get_job_or_none(job_id)
        if raw_job is None:
            return

        status = _status_from_job_data(raw_job["TranscriptionJob"])
        started = time.monotonic()
        while not status.is_done:
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                raise AwsTranscribeError(job_id, f"cleanup did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)
            raw_job = self._get_job_or_none(job_id)
            if raw_job is None:
                return
            status = _status_from_job_data(raw_job["TranscriptionJob"])

        self._delete_output_if_needed(job_id, raw_job)
        self._delete_job_record(job_id)

    def process(
        self,
        media_uri: str,
        *,
        timeout: float | None | object = _UNSET,
        **kwargs: Any,
    ) -> ToolOutput:
        """Submit a Transcribe job, wait for completion, clean up, and return output."""
        effective_timeout = self.timeout if timeout is _UNSET else timeout
        if effective_timeout is not None and effective_timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")

        job_id = self.submit(media_uri, **kwargs)
        started = time.monotonic()
        while not self.is_done(job_id):
            logging.info("AWS Transcribe job %s is still running", job_id)
            if effective_timeout is not None and time.monotonic() - started > effective_timeout:
                self.cleanup(job_id)
                raise AwsTranscribeError(job_id, f"did not finish within {effective_timeout} seconds")
            time.sleep(self.polling_interval)

        result = self.get_result(job_id)
        if result is None:
            raise AwsTranscribeError(job_id, "finished without an available transcript")
        return result

    @staticmethod
    def native_to_tool_output(native: Any) -> ToolOutput:
        """Convert native AWS transcript JSON into an AMPAV `ToolOutput`."""
        if not isinstance(native, dict):
            raise AwsTranscriptSchemaError("$", "AWS transcript JSON must contain an object")
        aws_model = validate_aws_transcript_contract(native)
        transcript = aws_transcript_to_transcript(aws_model)
        return ToolOutput(
            tool_name="aws_transcribe",
            output=transcript,
            tool_private={"raw_transcript": json_safe(native)},
        )

    def _get_job(self, job_id: str) -> dict[str, Any]:
        try:
            return self.transcribe_client.get_transcription_job(TranscriptionJobName=job_id)
        except ClientError as exc:
            if _is_not_found_error(exc):
                raise KeyError(job_id) from exc
            raise

    def _get_job_or_none(self, job_id: str) -> dict[str, Any] | None:
        try:
            return self._get_job(job_id)
        except KeyError:
            return None

    def _read_transcript_json(self, job_data: dict[str, Any]) -> dict[str, Any]:
        transcript_uri = job_data.get("Transcript", {}).get("TranscriptFileUri")
        if not isinstance(transcript_uri, str):
            raise AwsTranscribeError(job_data.get("TranscriptionJobName"), "completed job did not include transcript URI")

        logging.info("Reading raw AWS transcript from %s", transcript_uri)
        if transcript_uri.startswith("s3://"):
            location = parse_s3_uri(transcript_uri)
            response = self.s3_client.get_object(Bucket=location.bucket, Key=location.key)
            with response["Body"] as body:
                transcript = json.loads(body.read().decode("utf-8"))
        else:
            with urlopen(transcript_uri) as response:
                transcript = json.loads(response.read().decode("utf-8"))

        if not isinstance(transcript, dict):
            raise AwsTranscriptSchemaError("$", "AWS transcript JSON must contain an object")
        return transcript

    def _to_tool_output(
        self,
        job_id: str,
        raw_job: dict[str, Any],
        raw_transcript: dict[str, Any],
    ) -> ToolOutput:
        aws_model = validate_aws_transcript_contract(raw_transcript)
        transcript = aws_transcript_to_transcript(aws_model)
        job_data = raw_job.get("TranscriptionJob", {})
        parameters = tool_parameters(job_id, job_data)

        return ToolOutput(
            tool_name="aws_transcribe",
            parameters=parameters,
            queue_time=timestamp_or_none(job_data.get("CreationTime")),
            start_time=timestamp_or_none(job_data.get("StartTime")) or timestamp_or_none(job_data.get("CreationTime")),
            end_time=timestamp_or_none(job_data.get("CompletionTime")) or time.time(),
            output=transcript,
            tool_private={
                "aws_transcribe_job": {
                    "name": job_id,
                    "media_uri": parameters.get("content_source"),
                    "transcript_file_uri": parameters.get("transcript_file_uri"),
                },
                "raw_transcription_job": json_safe(raw_job),
                "raw_transcript": json_safe(raw_transcript),
            },
        )

    def _delete_output_if_needed(self, job_id: str, raw_job: dict[str, Any]) -> None:
        if not self._should_delete_output(job_id):
            return
        output_uri = _transcript_file_uri(raw_job.get("TranscriptionJob", {}))
        if not output_uri or not output_uri.startswith("s3://"):
            return
        location = parse_s3_uri(output_uri)
        self.s3_client.delete_object(Bucket=location.bucket, Key=location.key)

    def _should_delete_output(self, job_id: str) -> bool:
        return self._delete_output_by_job_id.get(job_id, False) or self._owned_output_by_job_id.get(job_id, False)

    def _delete_job_record(self, job_id: str) -> None:
        try:
            self.transcribe_client.delete_transcription_job(TranscriptionJobName=job_id)
        except ClientError as exc:
            if _is_not_found_error(exc):
                return
            raise


def build_start_job_request(
    *,
    job_name: str,
    media_uri: str,
    output_location: S3Location | None,
    transcription: TranscriptionSettings,
) -> dict[str, Any]:
    """Build a boto3 `start_transcription_job` request body."""
    request: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": media_uri},
        "MediaFormat": transcription.media_format or infer_media_format(media_uri),
    }

    if output_location is not None:
        request["OutputBucketName"] = output_location.bucket
        request["OutputKey"] = output_location.key

    if transcription.identify_language:
        request["IdentifyLanguage"] = True
        if transcription.language_options:
            request["LanguageOptions"] = transcription.language_options
    else:
        request["LanguageCode"] = transcription.language_code

    if transcription.show_speaker_labels:
        request["Settings"] = {
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": transcription.max_speaker_labels,
        }

    return request


def build_job_name(source: str | Path, prefix: str, timestamp: str) -> str:
    safe_prefix = safe_job_part(prefix) or "ampav-aws-transcribe"
    safe_stem = safe_job_part(Path(str(source).rstrip("/")).stem) or "media"
    max_stem_length = max(1, 200 - len(safe_prefix) - len(timestamp) - 2)
    return f"{safe_prefix}-{timestamp}-{safe_stem[:max_stem_length]}".strip("-")


def safe_job_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def infer_media_format(source: str | Path) -> str:
    media_format = Path(str(source)).suffix.lower().lstrip(".")
    if not media_format:
        raise ValueError("media_format is required when the media URI has no extension")
    return media_format


def map_aws_transcribe_status(status: str) -> AsyncStatusCode:
    """Map AWS Transcribe job status onto AMPAV async status."""
    match status:
        case "QUEUED":
            return AsyncStatusCode.QUEUED
        case "IN_PROGRESS":
            return AsyncStatusCode.IN_PROGRESS
        case "COMPLETED":
            return AsyncStatusCode.SUCCEEDED
        case "FAILED":
            return AsyncStatusCode.FAILED
        case _:
            raise AwsTranscribeError(None, f"unknown AWS Transcribe status {status!r}")


def timestamp_or_none(value: object) -> float | None:
    return value.timestamp() if isinstance(value, datetime) else None


def tool_parameters(job_id: str, job_data: dict[str, Any]) -> dict[str, Any]:
    transcript_uri = _transcript_file_uri(job_data)
    media_uri = job_data.get("Media", {}).get("MediaFileUri")
    parameters = {
        "job_id": job_id,
        "content_source": media_uri,
        "transcript_file_uri": transcript_uri,
        "language_code": job_data.get("LanguageCode"),
        "identified_language_score": job_data.get("IdentifiedLanguageScore"),
    }
    return {key: value for key, value in parameters.items() if value is not None}


def json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, default=str))


def _status_from_job_data(job_data: dict[str, Any]) -> AwsTranscribeStatus:
    aws_status = job_data["TranscriptionJobStatus"]
    failure_reason = job_data.get("FailureReason")
    return AwsTranscribeStatus(
        job_id=job_data["TranscriptionJobName"],
        status=map_aws_transcribe_status(aws_status),
        message=failure_reason,
        aws_status=aws_status,
        failure_reason=failure_reason,
        transcript_file_uri=_transcript_file_uri(job_data),
    )


def _transcript_file_uri(job_data: dict[str, Any]) -> str | None:
    uri = job_data.get("Transcript", {}).get("TranscriptFileUri")
    return uri if isinstance(uri, str) else None


def _is_not_found_error(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code")
    return code in {"BadRequestException", "NotFoundException", "ResourceNotFoundException"}
