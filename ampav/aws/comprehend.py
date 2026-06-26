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
from pydantic import BaseModel, ConfigDict

from ampav.core.async_tool import AsyncJobStatus, AsyncStatusCode, AsyncTool, CleanupPolicy
from ampav.core.schema import ToolOutput

from .errors import AwsComprehendError
from .s3 import S3Location, join_s3_key, parse_s3_uri

InputFormat = Literal["ONE_DOC_PER_FILE", "ONE_DOC_PER_LINE"]


class StrictModel(BaseModel):
    """Pydantic base model for user-facing settings objects."""

    model_config = ConfigDict(extra="forbid")


class AwsComprehendJob(StrictModel):
    """Public handle for a submitted AWS Comprehend entities job."""

    id: str
    name: str | None = None
    arn: str | None = None
    input_s3_uri: str
    output_s3_uri: str
    language_code: str = "en"
    input_format: InputFormat = "ONE_DOC_PER_FILE"

    @property
    def input_location(self) -> S3Location:
        """S3 location submitted as Comprehend input."""
        return parse_s3_uri(self.input_s3_uri)


class AwsComprehendStatus(AsyncJobStatus):
    """AWS Comprehend status mapped onto AMPAV async status."""

    aws_status: str
    output_s3_uri: str | None = None


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


class AwsComprehend(AsyncTool[str, AwsComprehendJob, AwsComprehendResult]):
    """Low-level AWS Comprehend client for async batch entity detection."""

    polling_interval: float = 30
    timeout: float | None = 7200

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        comprehend_client: Any | None = None,
        s3_client: Any | None = None,
        data_access_role_arn: str | None = None,
        polling_interval: float = 30,
        timeout: float | None = 7200,
        cleanup_policy: CleanupPolicy | None = None,
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
        self.polling_interval = polling_interval
        self.timeout = timeout
        if cleanup_policy is not None:
            self.cleanup_policy = cleanup_policy

    def upload_text_input(
        self,
        text: str,
        *,
        bucket: str,
        key: str | None = None,
        prefix: str = "aws_comprehend/input",
        job_name: str | None = None,
        job_name_prefix: str = "ampav-aws-comprehend",
    ) -> S3Location:
        """Upload UTF-8 text input for a Comprehend async job."""
        if not text.strip():
            raise ValueError("text must not be empty")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_name = job_name or build_job_name("entities", job_name_prefix, timestamp)
        key = key or join_s3_key(prefix, f"{job_name}.txt")
        location = S3Location(bucket, key)

        logging.info("Uploading AWS Comprehend input text to %s", location.uri)
        self.s3_client.put_object(
            Bucket=location.bucket,
            Key=location.key,
            Body=text.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return location

    def submit(
        self,
        input_s3_uri: str,
        *,
        output_s3_uri: str,
        data_access_role_arn: str | None = None,
        language_code: str = "en",
        input_format: InputFormat = "ONE_DOC_PER_FILE",
        job_name: str | None = None,
        job_name_prefix: str = "ampav-aws-comprehend",
        entity_recognizer_arn: str | None = None,
        client_request_token: str | None = None,
        output_kms_key_id: str | None = None,
        volume_kms_key_id: str | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> AwsComprehendJob:
        """Submit an AWS Comprehend entities job for text input in S3."""
        parse_s3_uri(input_s3_uri)
        parse_s3_uri(output_s3_uri)
        role_arn = data_access_role_arn or self.data_access_role_arn
        if not role_arn:
            raise ValueError("data_access_role_arn is required")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_name = job_name or build_job_name(input_s3_uri, job_name_prefix, timestamp)
        request = build_start_entities_request(
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
            data_access_role_arn=role_arn,
            language_code=language_code,
            input_format=input_format,
            job_name=job_name,
            entity_recognizer_arn=entity_recognizer_arn,
            client_request_token=client_request_token,
            output_kms_key_id=output_kms_key_id,
            volume_kms_key_id=volume_kms_key_id,
            tags=tags,
        )

        logging.info("Starting AWS Comprehend entities job %s", job_name)
        logging.debug("AWS Comprehend request: %s", json.dumps(request, default=str, sort_keys=True))
        response = self.comprehend_client.start_entities_detection_job(**request)
        return AwsComprehendJob(
            id=response["JobId"],
            name=job_name,
            arn=response.get("JobArn"),
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
            language_code=language_code,
            input_format=input_format,
        )

    def get_job(self, job: AwsComprehendJob) -> dict[str, Any]:
        """Return AWS metadata for a submitted entities detection job."""
        return self.comprehend_client.describe_entities_detection_job(JobId=job.id)

    def get_status(self, job: AwsComprehendJob) -> AwsComprehendStatus:
        """Return normalized status information for an AWS Comprehend job."""
        response = self.get_job(job)
        job_data = response["EntitiesDetectionJobProperties"]
        aws_status = job_data["JobStatus"]
        message = job_data.get("Message")
        output_config = job_data.get("OutputDataConfig") or {}
        return AwsComprehendStatus(
            job_id=job.id,
            status=map_aws_comprehend_status(aws_status),
            message=message,
            aws_status=aws_status,
            output_s3_uri=output_config.get("S3Uri"),
        )

    def list_jobs(self, **kwargs: Any) -> dict[str, Any]:
        """Return AWS Comprehend entities job summaries using boto3 list arguments."""
        return self.comprehend_client.list_entities_detection_jobs(**kwargs)

    def get_external_result(self, job: AwsComprehendJob) -> AwsComprehendResult | None:
        """Return raw AWS Comprehend output records when the job has succeeded."""
        status = self.get_status(job)
        if status.status != AsyncStatusCode.SUCCEEDED:
            return None

        raw_job = self.get_job(job)
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

    def wait(
        self,
        job: AwsComprehendJob,
        *,
        cleanup_policy: CleanupPolicy | None = None,
    ) -> AwsComprehendResult:
        """Wait for a Comprehend job and return provider-native result data."""
        cleanup_policy = cleanup_policy or self.cleanup_policy
        started = time.monotonic()
        status = self.get_status(job)

        while not status.is_done:
            logging.info("AWS Comprehend job %s status: %s", job.id, status.aws_status)
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                self.cleanup(job, cleanup_policy)
                raise AwsComprehendError(job.id, f"did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)
            status = self.get_status(job)

        logging.info("AWS Comprehend job %s status: %s", job.id, status.aws_status)
        if status.status != AsyncStatusCode.SUCCEEDED:
            self.cleanup(job, cleanup_policy)
            message = status.message or "no provider message"
            raise AwsComprehendError(job.id, f"ended with status {status.status}: {message}")

        result = self.get_external_result(job)
        if result is None:
            self.cleanup(job, cleanup_policy)
            raise AwsComprehendError(job.id, "succeeded without an available result")

        self.cleanup(job, cleanup_policy)
        return result

    def to_tool_output(self, job: AwsComprehendJob, result: AwsComprehendResult) -> ToolOutput:
        """Convert raw Comprehend output into AMPAV ToolOutput after schema work lands."""
        raise NotImplementedError("AWS Comprehend ToolOutput conversion is deferred until NamedEntity schema exists")

    def process(self, provider_input: str, **kwargs: Any) -> ToolOutput:
        """Deferred until Comprehend results can be converted to AMPAV schema."""
        raise AwsComprehendError(
            None,
            "process() is deferred until NamedEntity ToolOutput conversion exists; use submit() and wait()",
        )

    def cleanup(
        self,
        job: AwsComprehendJob,
        cleanup_policy: CleanupPolicy | None = None,
    ) -> None:
        """Delete selected S3 objects and stop in-progress jobs when requested.

        S3 cleanup only deletes exact objects. It does not recursively delete
        caller-owned prefixes.
        """
        cleanup_policy = cleanup_policy or self.cleanup_policy
        if cleanup_policy.delete_output:
            output_s3_uri = get_output_s3_uri_or_none(self.get_job(job))
            if output_s3_uri:
                output_location = parse_s3_uri(output_s3_uri)
                if output_location.key.endswith(".tar.gz"):
                    self.s3_client.delete_object(Bucket=output_location.bucket, Key=output_location.key)
        if cleanup_policy.delete_input:
            input_location = job.input_location
            self.s3_client.delete_object(Bucket=input_location.bucket, Key=input_location.key)
        if cleanup_policy.delete_job:
            status = self.get_status(job)
            if not status.is_done:
                self.comprehend_client.stop_entities_detection_job(JobId=job.id)


def build_start_entities_request(
    *,
    input_s3_uri: str,
    output_s3_uri: str,
    data_access_role_arn: str,
    language_code: str,
    input_format: InputFormat,
    job_name: str,
    entity_recognizer_arn: str | None = None,
    client_request_token: str | None = None,
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
    if client_request_token:
        request["ClientRequestToken"] = client_request_token
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
        case "FAILED":
            return AsyncStatusCode.FAILED
        case "STOPPED":
            return AsyncStatusCode.CANCELED
        case _:
            raise AwsComprehendError(None, f"unknown AWS Comprehend status {status!r}")


def json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, default=str))
