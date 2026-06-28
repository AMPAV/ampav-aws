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

from ampav.core.async_tool import AsyncJobStatus, AsyncStatusCode, AsyncTool
from ampav.core.schema import ToolOutput

from .errors import AwsTranscribeError, AwsTranscriptSchemaError
from .job import AwsJobStatus
from .s3 import S3Location, parse_s3_uri
from .transcribe_contract import validate_aws_transcript_contract
from .transcribe_conversion import aws_transcript_to_transcript


_JOB_NAME_PREFIX = "ampav-aws-transcribe"


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


class AwsTranscribe(AsyncTool):
    """Low-level AWS Transcribe client.

    The public lifecycle uses opaque `job_id` strings. The tool API expects
    provider-native S3 media URIs; local file upload belongs in CLI/example code.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        transcribe_client: Any | None = None,
        s3_client: Any | None = None,
        delete_user_owned_outputs: bool = False,
        include_tool_private: bool = False,
        polling_interval: float = 30,
        timeout: float | None = 7200,
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
        self.delete_user_owned_outputs = delete_user_owned_outputs
        self.include_tool_private = include_tool_private
        self.polling_interval = polling_interval
        self.timeout = timeout

    def submit(
        self,
        input_s3_uri: str,
        *,
        output_s3_uri: str | None = None,
        job_name_suffix: str | None = None,
        transcription_settings: TranscriptionSettings | None = None,
    ) -> str:
        """Submit an AWS Transcribe job for media that already exists in S3."""
        parse_s3_uri(input_s3_uri)
        output_location = parse_s3_uri(output_s3_uri) if output_s3_uri else None
        transcription_settings = transcription_settings or TranscriptionSettings()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_id = build_job_name(job_name_suffix or input_s3_uri, _JOB_NAME_PREFIX, timestamp)
        request = build_start_job_request(
            job_name=job_id,
            input_s3_uri=input_s3_uri,
            output_location=output_location,
            transcription_settings=transcription_settings,
        )

        logging.info("Starting AWS Transcribe job %s", job_id)
        logging.debug("AWS Transcribe request: %s", json.dumps(request, default=str, sort_keys=True))
        self.transcribe_client.start_transcription_job(**request)
        return job_id

    def get_status(self, job_id: str, details: bool = True) -> AsyncJobStatus:
        """Return normalized status information for an AWS Transcribe job."""
        raw_job = self._get_job(job_id)
        job_data = raw_job["TranscriptionJob"]
        return _status_from_job_data(job_data, details=details)

    def list_jobs(self) -> list[AsyncJobStatus]:
        """Return AWS Transcribe jobs matching this instance's job-name prefix."""
        response = self.transcribe_client.list_transcription_jobs(JobNameContains=_JOB_NAME_PREFIX)
        summaries = response.get("TranscriptionJobSummaries", [])
        statuses: list[AsyncJobStatus] = []
        for summary in summaries:
            job_id = summary.get("TranscriptionJobName")
            aws_status = summary.get("TranscriptionJobStatus")
            if not isinstance(job_id, str) or not isinstance(aws_status, str):
                continue
            if not job_id.startswith(_JOB_NAME_PREFIX):
                continue
            statuses.append(
                AsyncJobStatus(
                    job_id=job_id,
                    status=map_aws_transcribe_status(aws_status),
                    message=summary.get("FailureReason"),
                )
            )
        return statuses

    def get_result(self, job_id: str) -> ToolOutput | None:
        """Return AMPAV output when ready; terminal jobs are cleaned up."""
        raw_job = self._get_job(job_id)
        job_data = raw_job["TranscriptionJob"]
        status = _status_from_job_data(job_data, details=False)
        if not status.is_done:
            return None

        if status.status != AsyncStatusCode.SUCCEEDED:
            self.cleanup(job_id)
            reason = status.message or "no failure reason returned"
            raise AwsTranscribeError(job_id, f"failed: {reason}")

        raw_transcript = self._read_transcript_json(job_data)
        output = self._to_tool_output(job_id, raw_job, raw_transcript)
        self._delete_output_if_needed(raw_job)
        self._delete_job_record(job_id)
        return output

    def cleanup(self, job_id: str) -> None:
        """Clean up output owned/requested by this tool and delete the job record."""
        raw_job = self._get_job_or_none(job_id)
        if raw_job is None:
            return

        status = _status_from_job_data(raw_job["TranscriptionJob"], details=False)
        started = time.monotonic()
        while not status.is_done:
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                raise AwsTranscribeError(job_id, f"cleanup did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)
            raw_job = self._get_job_or_none(job_id)
            if raw_job is None:
                return
            status = _status_from_job_data(raw_job["TranscriptionJob"], details=False)

        self._delete_output_if_needed(raw_job)
        self._delete_job_record(job_id)

    def process(
        self,
        input_s3_uri: str,
        **kwargs: Any,
    ) -> ToolOutput:
        """Submit a Transcribe job, wait for completion, clean up, and return output."""
        job_id = self.submit(input_s3_uri, **kwargs)
        started = time.monotonic()
        while not self.is_done(job_id):
            logging.info("AWS Transcribe job %s is still running", job_id)
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                self.cleanup(job_id)
                raise AwsTranscribeError(job_id, f"did not finish within {self.timeout} seconds")
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
        parameters = tool_parameters(job_data)
        tool_private = None
        if self.include_tool_private:
            tool_private = {
                "aws_transcribe_job": {
                    "name": job_id,
                },
                "raw_transcription_job": json_safe(raw_job),
                "raw_transcript": json_safe(raw_transcript),
            }

        return ToolOutput(
            tool_name="aws_transcribe",
            parameters=parameters,
            queue_time=timestamp_or_none(job_data.get("CreationTime")),
            start_time=timestamp_or_none(job_data.get("StartTime")) or timestamp_or_none(job_data.get("CreationTime")),
            end_time=timestamp_or_none(job_data.get("CompletionTime")) or time.time(),
            output=transcript,
            tool_private=tool_private,
        )

    def _delete_output_if_needed(self, raw_job: dict[str, Any]) -> None:
        if not self.delete_user_owned_outputs:
            return
        output_uri = _transcript_file_uri(raw_job.get("TranscriptionJob", {}))
        if not output_uri or not output_uri.startswith("s3://"):
            return
        location = parse_s3_uri(output_uri)
        self.s3_client.delete_object(Bucket=location.bucket, Key=location.key)

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
    input_s3_uri: str,
    output_location: S3Location | None,
    transcription_settings: TranscriptionSettings,
) -> dict[str, Any]:
    """Build a boto3 `start_transcription_job` request body."""
    request: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": input_s3_uri},
        "MediaFormat": transcription_settings.media_format or infer_media_format(input_s3_uri),
    }

    if output_location is not None:
        request["OutputBucketName"] = output_location.bucket
        request["OutputKey"] = output_location.key

    if transcription_settings.identify_language:
        request["IdentifyLanguage"] = True
        if transcription_settings.language_options:
            request["LanguageOptions"] = transcription_settings.language_options
    else:
        request["LanguageCode"] = transcription_settings.language_code

    if transcription_settings.show_speaker_labels:
        request["Settings"] = {
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": transcription_settings.max_speaker_labels,
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


def tool_parameters(job_data: dict[str, Any]) -> dict[str, Any]:
    settings = job_data.get("Settings") or {}
    parameters = {
        "media_format": job_data.get("MediaFormat"),
        "language_code": job_data.get("LanguageCode"),
        "identified_language_score": job_data.get("IdentifiedLanguageScore"),
        "identify_language": job_data.get("IdentifyLanguage"),
        "show_speaker_labels": settings.get("ShowSpeakerLabels"),
        "max_speaker_labels": settings.get("MaxSpeakerLabels"),
    }
    return {key: value for key, value in parameters.items() if value is not None}


def json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, default=str))


def _status_from_job_data(job_data: dict[str, Any], *, details: bool = True) -> AsyncJobStatus:
    aws_status = job_data["TranscriptionJobStatus"]
    failure_reason = job_data.get("FailureReason")
    status_data = {
        "job_id": job_data["TranscriptionJobName"],
        "status": map_aws_transcribe_status(aws_status),
        "message": failure_reason,
    }
    if not details:
        return AsyncJobStatus(**status_data)
    return AwsJobStatus(
        **status_data,
        job_name=job_data["TranscriptionJobName"],
        input_s3_uri=_media_file_uri(job_data),
        output_s3_uri=_transcript_file_uri(job_data),
    )


def _media_file_uri(job_data: dict[str, Any]) -> str | None:
    uri = job_data.get("Media", {}).get("MediaFileUri")
    return uri if isinstance(uri, str) else None


def _transcript_file_uri(job_data: dict[str, Any]) -> str | None:
    uri = job_data.get("Transcript", {}).get("TranscriptFileUri")
    return uri if isinstance(uri, str) else None


def _is_not_found_error(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code")
    return code in {"BadRequestException", "NotFoundException", "ResourceNotFoundException"}
