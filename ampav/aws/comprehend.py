"""AWS Comprehend async batch entity detection client for AMPAV."""

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import re
import tarfile
import time
from typing import Any, Literal

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict

from ampav.core.async_tool import AsyncJobStatus, AsyncStatusCode, AsyncTool
from ampav.core.schema import ToolOutput

from ._job import _AwsJobMeta
from .errors import AwsComprehendError
from .s3 import join_s3_key, parse_s3_uri

InputFormat = Literal["ONE_DOC_PER_FILE", "ONE_DOC_PER_LINE"]

_JOB_NAME_PREFIX = "ampav-aws-comprehend"


class StrictModel(BaseModel):
    """Pydantic base model for user-facing settings objects."""

    model_config = ConfigDict(extra="forbid")


class AwsComprehendResult(StrictModel):
    """Provider-native result data retrieved from a completed Comprehend job."""

    raw_job: dict[str, Any]
    output_s3_uri: str
    archive_members: list[str]
    records: list[dict[str, Any]]

    @property
    def has_record_errors(self) -> bool:
        """Return true when Comprehend completed but one or more documents failed."""
        return any("ErrorCode" in record or "ErrorMessage" in record for record in self.records)


class AwsComprehend(AsyncTool):
    """Low-level AWS Comprehend client for async batch entity detection."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        comprehend_client: Any | None = None,
        s3_client: Any | None = None,
        data_access_role_arn: str | None = None,
        output_kms_key_id: str | None = None,
        volume_kms_key_id: str | None = None,
        polling_interval: float = 30,
        timeout: float | None = 7200,
    ):
        """Create an AWS Comprehend client from boto3 settings or injected clients."""
        if polling_interval <= 0:
            raise ValueError("polling_interval must be greater than 0")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")

        if session is None and (comprehend_client is None or s3_client is None):
            session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.comprehend_client = comprehend_client or session.client("comprehend")
        self.s3_client = s3_client or session.client("s3")
        self.data_access_role_arn = data_access_role_arn
        self.output_kms_key_id = output_kms_key_id
        self.volume_kms_key_id = volume_kms_key_id
        self.polling_interval = polling_interval
        self.timeout = timeout
        self._job_meta_by_id: dict[str, _AwsJobMeta] = {}

    def submit(
        self,
        input_s3_uri: str,
        *,
        output_s3_uri: str | None = None,
        delete_output: bool = False,
        include_tool_private: bool = False,
        data_access_role_arn: str | None = None,
        language_code: str = "en",
        input_format: InputFormat = "ONE_DOC_PER_FILE",
        job_name_suffix: str | None = None,
        entity_recognizer_arn: str | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> str:
        """Submit an AWS Comprehend entities job for text input in S3."""
        input_location = parse_s3_uri(input_s3_uri)
        role_arn = data_access_role_arn or self.data_access_role_arn
        if not role_arn:
            raise ValueError("data_access_role_arn is required")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = build_job_name(job_name_suffix or input_s3_uri, _JOB_NAME_PREFIX, timestamp)
        owned_output = output_s3_uri is None
        if output_s3_uri is None:
            output_s3_uri = f"s3://{input_location.bucket}/{join_s3_key('aws_comprehend/output', name)}"
        parse_s3_uri(output_s3_uri)

        request = build_start_entities_request(
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
            data_access_role_arn=role_arn,
            language_code=language_code,
            input_format=input_format,
            job_name=name,
            entity_recognizer_arn=entity_recognizer_arn,
            output_kms_key_id=self.output_kms_key_id,
            volume_kms_key_id=self.volume_kms_key_id,
            tags=tags,
        )

        logging.info("Starting AWS Comprehend entities job %s", name)
        logging.debug("AWS Comprehend request: %s", json.dumps(request, default=str, sort_keys=True))
        response = self.comprehend_client.start_entities_detection_job(**request)
        job_id = response["JobId"]
        self._job_meta_by_id[job_id] = _AwsJobMeta(
            delete_output=delete_output,
            owned_output=owned_output,
            include_tool_private=include_tool_private,
        )
        return job_id

    def get_status(self, job_id: str, details: bool = True) -> AsyncJobStatus:
        """Return normalized status information for an AWS Comprehend job."""
        raw_job = self._get_job(job_id)
        return _status_from_job_data(raw_job["EntitiesDetectionJobProperties"])

    def list_jobs(self) -> list[AsyncJobStatus]:
        """Return AWS Comprehend entity jobs matching this instance's job-name prefix."""
        response = self.comprehend_client.list_entities_detection_jobs()
        jobs = response.get("EntitiesDetectionJobPropertiesList", [])
        statuses: list[AsyncJobStatus] = []
        for job_data in jobs:
            job_name = job_data.get("JobName")
            if not isinstance(job_name, str) or not job_name.startswith(_JOB_NAME_PREFIX):
                continue
            try:
                statuses.append(_status_from_job_data(job_data))
            except KeyError:
                continue
        return statuses

    def get_result(self, job_id: str) -> ToolOutput | None:
        """Return a transitional AMPAV `ToolOutput` when the batch job is ready."""
        raw_job = self._get_job(job_id)
        status = _status_from_job_data(raw_job["EntitiesDetectionJobProperties"])
        if not status.is_done:
            return None

        if status.status != AsyncStatusCode.SUCCEEDED:
            self.cleanup(job_id)
            message = status.message or "no provider message"
            raise AwsComprehendError(job_id, f"ended with status {status.status}: {message}")

        native = self._get_native_result(raw_job)
        output = self._to_tool_output(job_id, native)
        self._delete_output_if_needed(job_id, raw_job)
        self._delete_state(job_id)
        return output

    def cleanup(self, job_id: str) -> None:
        """Stop active jobs where possible, delete owned/requested output, and clear state."""
        raw_job = self._get_job_or_none(job_id)
        if raw_job is None:
            self._delete_state(job_id)
            return

        status = _status_from_job_data(raw_job["EntitiesDetectionJobProperties"])
        if not status.is_done:
            self.comprehend_client.stop_entities_detection_job(JobId=job_id)

        started = time.monotonic()
        while not status.is_done:
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                raise AwsComprehendError(job_id, f"cleanup did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)
            raw_job = self._get_job_or_none(job_id)
            if raw_job is None:
                self._delete_state(job_id)
                return
            status = _status_from_job_data(raw_job["EntitiesDetectionJobProperties"])

        self._delete_output_if_needed(job_id, raw_job)
        self._delete_state(job_id)

    def process(
        self,
        input_s3_uri: str,
        **kwargs: Any,
    ) -> ToolOutput:
        """Submit a Comprehend job, wait for completion, clean up, and return output."""
        job_id = self.submit(input_s3_uri, **kwargs)
        started = time.monotonic()
        while not self.is_done(job_id):
            logging.info("AWS Comprehend job %s is still running", job_id)
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                self.cleanup(job_id)
                raise AwsComprehendError(job_id, f"did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)

        result = self.get_result(job_id)
        if result is None:
            raise AwsComprehendError(job_id, "finished without available output")
        return result

    @staticmethod
    def native_to_tool_output(native: Any) -> ToolOutput:
        """Convert native Comprehend result data into transitional AMPAV output."""
        if isinstance(native, AwsComprehendResult):
            result = native
        elif isinstance(native, dict):
            result = AwsComprehendResult.model_validate(native)
        else:
            raise AwsComprehendError(None, "native Comprehend result must be AwsComprehendResult or dict")

        return ToolOutput(
            tool_name="aws_comprehend",
            parameters={
                "archive_members": result.archive_members,
                "record_count": len(result.records),
                "has_record_errors": result.has_record_errors,
            },
            output=None,
        )

    def _to_tool_output(self, job_id: str, result: AwsComprehendResult) -> ToolOutput:
        output = self.native_to_tool_output(result)
        if self._job_meta_by_id.get(job_id, _AwsJobMeta()).include_tool_private:
            output.tool_private = {
                "aws_comprehend_result": result.model_dump(mode="json"),
                "raw_comprehend_job": result.raw_job,
                "raw_records": result.records,
            }
        return output

    def _get_job(self, job_id: str) -> dict[str, Any]:
        try:
            return self.comprehend_client.describe_entities_detection_job(JobId=job_id)
        except ClientError as exc:
            if _is_not_found_error(exc):
                raise KeyError(job_id) from exc
            raise

    def _get_job_or_none(self, job_id: str) -> dict[str, Any] | None:
        try:
            return self._get_job(job_id)
        except KeyError:
            return None

    def _get_native_result(self, raw_job: dict[str, Any]) -> AwsComprehendResult:
        output_s3_uri = get_output_s3_uri(raw_job)
        logging.info("Reading AWS Comprehend output from %s", output_s3_uri)
        output_location = parse_s3_uri(output_s3_uri)
        response = self.s3_client.get_object(Bucket=output_location.bucket, Key=output_location.key)
        with response["Body"] as body:
            archive_bytes = body.read()

        archive_members, records = parse_output_archive(archive_bytes)
        return AwsComprehendResult(
            raw_job=json_safe(raw_job),
            output_s3_uri=output_s3_uri,
            archive_members=archive_members,
            records=records,
        )

    def _delete_output_if_needed(self, job_id: str, raw_job: dict[str, Any]) -> None:
        if not self._should_delete_output(job_id):
            return
        output_s3_uri = get_output_s3_uri_or_none(raw_job)
        if not output_s3_uri:
            return
        output_location = parse_s3_uri(output_s3_uri)
        if output_location.key.endswith(".tar.gz"):
            self.s3_client.delete_object(Bucket=output_location.bucket, Key=output_location.key)

    def _should_delete_output(self, job_id: str) -> bool:
        meta = self._job_meta_by_id.get(job_id)
        return bool(meta and (meta.delete_output or meta.owned_output))

    def _delete_state(self, job_id: str) -> None:
        self._job_meta_by_id.pop(job_id, None)


def build_start_entities_request(
    *,
    input_s3_uri: str,
    output_s3_uri: str,
    data_access_role_arn: str,
    language_code: str,
    input_format: InputFormat,
    job_name: str,
    entity_recognizer_arn: str | None = None,
    output_kms_key_id: str | None = None,
    volume_kms_key_id: str | None = None,
    tags: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a boto3 `start_entities_detection_job` request body."""
    request: dict[str, Any] = {
        "InputDataConfig": {"S3Uri": input_s3_uri, "InputFormat": input_format},
        "OutputDataConfig": {"S3Uri": output_s3_uri},
        "DataAccessRoleArn": data_access_role_arn,
        "JobName": job_name,
        "LanguageCode": language_code,
    }
    if entity_recognizer_arn:
        request["EntityRecognizerArn"] = entity_recognizer_arn
    if output_kms_key_id:
        request["OutputDataConfig"]["KmsKeyId"] = output_kms_key_id
    if volume_kms_key_id:
        request["VolumeKmsKeyId"] = volume_kms_key_id
    if tags:
        request["Tags"] = tags
    return request


def parse_output_archive(data: bytes) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse Comprehend `output.tar.gz` bytes into JSON-line records."""
    archive_members: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.isfile():
                    continue
                archive_members.append(member.name)
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                for line_number, raw_line in enumerate(extracted.read().decode("utf-8").splitlines(), start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise AwsComprehendError(None, f"{member.name}:{line_number}: expected JSON object")
                    records.append(record)
    except (tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AwsComprehendError(None, f"could not parse Comprehend output archive: {exc}") from exc
    return archive_members, records


def get_output_s3_uri(raw_job: dict[str, Any]) -> str:
    output_s3_uri = get_output_s3_uri_or_none(raw_job)
    if not output_s3_uri:
        raise AwsComprehendError(None, "completed job did not include OutputDataConfig.S3Uri")
    return output_s3_uri


def get_output_s3_uri_or_none(raw_job: dict[str, Any]) -> str | None:
    job_data = raw_job.get("EntitiesDetectionJobProperties") or {}
    output_config = job_data.get("OutputDataConfig") or {}
    output_s3_uri = output_config.get("S3Uri")
    return output_s3_uri if isinstance(output_s3_uri, str) else None


def build_job_name(source: str | Path, prefix: str, timestamp: str) -> str:
    safe_prefix = safe_job_part(prefix) or "ampav-aws-comprehend"
    safe_stem = safe_job_part(Path(str(source).rstrip("/")).stem) or "entities"
    max_stem_length = max(1, 200 - len(safe_prefix) - len(timestamp) - 2)
    return f"{safe_prefix}-{timestamp}-{safe_stem[:max_stem_length]}".strip("-")


def safe_job_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def map_aws_comprehend_status(status: str) -> AsyncStatusCode:
    """Map AWS Comprehend job status onto AMPAV async status."""
    match status:
        case "SUBMITTED":
            return AsyncStatusCode.QUEUED
        case "IN_PROGRESS" | "STOP_REQUESTED":
            return AsyncStatusCode.IN_PROGRESS
        case "COMPLETED":
            return AsyncStatusCode.SUCCEEDED
        case "FAILED" | "STOPPED":
            return AsyncStatusCode.FAILED
        case _:
            raise AwsComprehendError(None, f"unknown AWS Comprehend status {status!r}")


def json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, default=str))


def _status_from_job_data(job_data: dict[str, Any]) -> AsyncJobStatus:
    aws_status = job_data["JobStatus"]
    message = job_data.get("Message")
    return AsyncJobStatus(
        job_id=job_data["JobId"],
        status=map_aws_comprehend_status(aws_status),
        message=message,
    )


def _is_not_found_error(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code")
    return code in {"ResourceNotFoundException", "InvalidRequestException", "BadRequestException"}
