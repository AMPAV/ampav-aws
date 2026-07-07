"""AWS Comprehend named-entities async client for AMPAV."""

from datetime import datetime, timezone
import io
import json
import logging
from pathlib import Path
import re
import tarfile
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, ConfigDict, ValidationError

from ampav.core.async_tool import AsyncJobStatus, AsyncStatusCode, AsyncTool
from ampav.core.schema import NamedEntities, NamedEntity, ToolOutput

from ._version import __version__
from .errors import AwsComprehendNamedEntitiesError, AwsComprehendNamedEntitiesSchemaError
from .job import AwsJobStatus
from .s3 import join_s3_key, parse_s3_uri

_JOB_NAME_PREFIX = "ampav-aws-comprehend-named-entities"
_TOOL_MANAGED_OUTPUT_PREFIX = "aws_comprehend_named_entities/_ampav_tmp"
_INPUT_FORMAT = "ONE_DOC_PER_FILE"


class StrictModel(BaseModel):
    """Pydantic base model for user-facing settings objects."""

    model_config = ConfigDict(extra="forbid")


class AwsComprehendNamedEntitiesResult(StrictModel):
    """Raw Comprehend result data plus the source text used for extraction."""

    raw_job: dict[str, Any]
    source_text: str
    output_s3_uri: str
    archive_members: list[str]
    records: list[dict[str, Any]]

    @property
    def has_record_errors(self) -> bool:
        """Return true when Comprehend completed but one or more documents failed."""
        return any("ErrorCode" in record or "ErrorMessage" in record for record in self.records)


class AwsComprehendNamedEntities(AsyncTool):
    """Low-level AWS Comprehend client for async named-entity detection."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        comprehend_client: Any | None = None,
        s3_client: Any | None = None,
        data_access_role_arn: str | None = None,
        entity_recognizer_arn: str | None = None,
        output_kms_key_id: str | None = None,
        volume_kms_key_id: str | None = None,
        delete_user_owned_outputs: bool = False,
        include_tool_private: bool = False,
        polling_interval: float = 30,
        timeout: float | None = 7200,
    ):
        """Create an AWS Comprehend named-entities client from boto3 settings or injected clients."""
        if polling_interval <= 0:
            raise ValueError("polling_interval must be greater than 0")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0 when set")

        if session is None and (comprehend_client is None or s3_client is None):
            session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.comprehend_client = comprehend_client or session.client("comprehend")
        self.s3_client = s3_client or session.client("s3")
        self.data_access_role_arn = data_access_role_arn
        self.entity_recognizer_arn = entity_recognizer_arn
        self.output_kms_key_id = output_kms_key_id
        self.volume_kms_key_id = volume_kms_key_id
        self.delete_user_owned_outputs = delete_user_owned_outputs
        self.include_tool_private = include_tool_private
        self.polling_interval = polling_interval
        self.timeout = timeout

    def submit(
        self,
        input_s3_uri: str,
        *,
        output_s3_uri: str | None = None,
        language_code: str = "en",
        job_name_suffix: str | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> str:
        """Submit a one-document AWS Comprehend named-entities job for a UTF-8 text object in S3."""
        input_location = parse_s3_uri(input_s3_uri)
        role_arn = self.data_access_role_arn
        if not role_arn:
            raise ValueError("data_access_role_arn is required")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        name = build_job_name(job_name_suffix or input_s3_uri, _JOB_NAME_PREFIX, timestamp)
        if output_s3_uri is None:
            output_s3_uri = f"s3://{input_location.bucket}/{join_s3_key(_TOOL_MANAGED_OUTPUT_PREFIX, name)}"
        parse_s3_uri(output_s3_uri)

        request = build_start_entities_request(
            input_s3_uri=input_s3_uri,
            output_s3_uri=output_s3_uri,
            data_access_role_arn=role_arn,
            language_code=language_code,
            job_name=name,
            entity_recognizer_arn=self.entity_recognizer_arn,
            output_kms_key_id=self.output_kms_key_id,
            volume_kms_key_id=self.volume_kms_key_id,
            tags=tags,
        )

        logging.info("Starting AWS Comprehend named-entities job %s", name)
        logging.debug("AWS Comprehend request: %s", json.dumps(request, default=str, sort_keys=True))
        response = self.comprehend_client.start_entities_detection_job(**request)
        return response["JobId"]

    def get_status(self, job_id: str, details: bool = True) -> AsyncJobStatus:
        """Return normalized status information for an AWS Comprehend named-entities job."""
        raw_job = self._get_job(job_id)
        return _status_from_job_data(raw_job["EntitiesDetectionJobProperties"], details=details)

    def list_jobs(self) -> list[AsyncJobStatus]:
        """Return AWS Comprehend named-entities jobs matching this instance's job-name prefix."""
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
        """Return AMPAV named entities when the Comprehend batch job is ready."""
        raw_job = self._get_job(job_id)
        status = _status_from_job_data(raw_job["EntitiesDetectionJobProperties"], details=False)
        if not status.is_done:
            return None

        if status.status != AsyncStatusCode.SUCCEEDED:
            self.cleanup(job_id)
            message = status.message or "no provider message"
            raise AwsComprehendNamedEntitiesError(job_id, f"ended with status {status.status}: {message}")

        native = self._get_native_result(raw_job)
        output = self._to_tool_output(job_id, native)
        self._delete_output_if_needed(raw_job)
        return output

    def cleanup(self, job_id: str) -> None:
        """Stop active jobs where possible, delete owned/requested output, and clear state."""
        raw_job = self._get_job_or_none(job_id)
        if raw_job is None:
            return

        status = _status_from_job_data(raw_job["EntitiesDetectionJobProperties"], details=False)
        if not status.is_done:
            self.comprehend_client.stop_entities_detection_job(JobId=job_id)

        started = time.monotonic()
        while not status.is_done:
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                raise AwsComprehendNamedEntitiesError(job_id, f"cleanup did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)
            raw_job = self._get_job_or_none(job_id)
            if raw_job is None:
                return
            status = _status_from_job_data(raw_job["EntitiesDetectionJobProperties"], details=False)

        self._delete_output_if_needed(raw_job)

    def process(
        self,
        input_s3_uri: str,
        **kwargs: Any,
    ) -> ToolOutput:
        """Submit a Comprehend named-entities job, wait for completion, clean up, and return output."""
        job_id = self.submit(input_s3_uri, **kwargs)
        started = time.monotonic()
        while not self.is_done(job_id):
            logging.info("AWS Comprehend named-entities job %s is still running", job_id)
            if self.timeout is not None and time.monotonic() - started > self.timeout:
                self.cleanup(job_id)
                raise AwsComprehendNamedEntitiesError(job_id, f"did not finish within {self.timeout} seconds")
            time.sleep(self.polling_interval)

        result = self.get_result(job_id)
        if result is None:
            raise AwsComprehendNamedEntitiesError(job_id, "finished without available output")
        return result

    @staticmethod
    def native_to_tool_output(native: Any) -> ToolOutput:
        """Convert native Comprehend result data into AMPAV named entities."""
        if isinstance(native, AwsComprehendNamedEntitiesResult):
            result = native
        elif isinstance(native, dict):
            result = validate_aws_comprehend_named_entities_result(native)
        else:
            raise AwsComprehendNamedEntitiesSchemaError(
                "$",
                "native Comprehend result must be AwsComprehendNamedEntitiesResult or dict",
            )

        return ToolOutput(
            tool_name="aws_comprehend_named_entities",
            tool_version=__version__,
            parameters={
                "archive_members": result.archive_members,
                "record_count": len(result.records),
                "has_record_errors": result.has_record_errors,
            },
            output=aws_comprehend_named_entities_result_to_named_entities(result),
        )

    def _to_tool_output(self, job_id: str, result: AwsComprehendNamedEntitiesResult) -> ToolOutput:
        output = self.native_to_tool_output(result)
        if self.include_tool_private:
            output.tool_private = {
                "aws_comprehend_named_entities_result": result.model_dump(mode="json"),
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

    def _get_native_result(self, raw_job: dict[str, Any]) -> AwsComprehendNamedEntitiesResult:
        output_s3_uri = get_output_s3_uri(raw_job)
        input_s3_uri = get_input_s3_uri(raw_job)
        logging.info("Reading AWS Comprehend input text from %s", input_s3_uri)
        source_text = self._read_text_object(input_s3_uri)
        logging.info("Reading AWS Comprehend output from %s", output_s3_uri)
        output_location = parse_s3_uri(output_s3_uri)
        response = self.s3_client.get_object(Bucket=output_location.bucket, Key=output_location.key)
        with response["Body"] as body:
            archive_bytes = body.read()

        archive_members, records = parse_output_archive(archive_bytes)
        return AwsComprehendNamedEntitiesResult(
            raw_job=json_safe(raw_job),
            source_text=source_text,
            output_s3_uri=output_s3_uri,
            archive_members=archive_members,
            records=records,
        )

    def _read_text_object(self, input_s3_uri: str) -> str:
        input_location = parse_s3_uri(input_s3_uri)
        response = self.s3_client.get_object(Bucket=input_location.bucket, Key=input_location.key)
        with response["Body"] as body:
            return body.read().decode("utf-8")

    def _delete_output_if_needed(self, raw_job: dict[str, Any]) -> None:
        output_s3_uri = get_output_s3_uri_or_none(raw_job)
        if not output_s3_uri:
            return
        if not (self.delete_user_owned_outputs or is_tool_managed_output_uri(output_s3_uri)):
            return
        output_location = parse_s3_uri(output_s3_uri)
        if output_location.key.endswith(".tar.gz"):
            self.s3_client.delete_object(Bucket=output_location.bucket, Key=output_location.key)


def build_start_entities_request(
    *,
    input_s3_uri: str,
    output_s3_uri: str,
    data_access_role_arn: str,
    language_code: str,
    job_name: str,
    entity_recognizer_arn: str | None = None,
    output_kms_key_id: str | None = None,
    volume_kms_key_id: str | None = None,
    tags: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a boto3 `start_entities_detection_job` request body."""
    request: dict[str, Any] = {
        "InputDataConfig": {"S3Uri": input_s3_uri, "InputFormat": _INPUT_FORMAT},
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
                        raise AwsComprehendNamedEntitiesSchemaError(
                            f"$.archive.{member.name}[{line_number}]",
                            "expected JSON object",
                        )
                    records.append(record)
    except (tarfile.TarError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AwsComprehendNamedEntitiesSchemaError(
            "$.archive",
            f"could not parse Comprehend output archive: {exc}",
        ) from exc
    return archive_members, records


def validate_aws_comprehend_named_entities_result(native: object) -> AwsComprehendNamedEntitiesResult:
    """Validate the internal native result bundle consumed by AMPAV conversion."""
    if not isinstance(native, dict):
        raise AwsComprehendNamedEntitiesSchemaError("$", "expected object")
    try:
        return AwsComprehendNamedEntitiesResult.model_validate(native)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise AwsComprehendNamedEntitiesSchemaError(pydantic_loc_to_path(error["loc"]), str(error["msg"])) from exc


def aws_comprehend_named_entities_result_to_named_entities(result: AwsComprehendNamedEntitiesResult) -> NamedEntities:
    """Convert one-document Comprehend entity output to AMPAV named entities."""
    record = single_success_record(result)
    language = get_language_or_none(result.raw_job)
    entities = [
        aws_entity_to_named_entity(entity, language=language)
        for entity in record.get("Entities", [])
    ]
    return NamedEntities(
        text=result.source_text,
        spans=entities,
        languages=[language] if language else None,
    )


def single_success_record(result: AwsComprehendNamedEntitiesResult) -> dict[str, Any]:
    """Return the only successful Comprehend output record."""
    if len(result.records) != 1:
        raise AwsComprehendNamedEntitiesError(None, f"expected one Comprehend output record, got {len(result.records)}")
    record = result.records[0]
    if "ErrorCode" in record or "ErrorMessage" in record:
        code = record.get("ErrorCode", "UNKNOWN")
        message = record.get("ErrorMessage", "no provider message")
        raise AwsComprehendNamedEntitiesError(None, f"record failed with {code}: {message}")
    entities = record.get("Entities")
    if not isinstance(entities, list):
        raise AwsComprehendNamedEntitiesSchemaError("$.records[0].Entities", "expected list")
    return record


def aws_entity_to_named_entity(entity: Any, *, language: str | None = None) -> NamedEntity:
    """Map an AWS Comprehend entity object to the shared AMPAV span schema."""
    if not isinstance(entity, dict):
        raise AwsComprehendNamedEntitiesSchemaError("$.records[0].Entities[]", "expected JSON object")
    try:
        return NamedEntity(
            text=str(entity["Text"]),
            entity_type=str(entity["Type"]),
            confidence=None if entity.get("Score") is None else float(entity["Score"]),
            begin_offset=int(entity["BeginOffset"]),
            end_offset=int(entity["EndOffset"]),
            language=language,
        )
    except KeyError as exc:
        raise AwsComprehendNamedEntitiesSchemaError(
            "$.records[0].Entities[]",
            f"missing required field {exc.args[0]!r}",
        ) from exc
    except (TypeError, ValueError, ValidationError) as exc:
        raise AwsComprehendNamedEntitiesSchemaError("$.records[0].Entities[]", f"invalid entity data: {exc}") from exc


def get_output_s3_uri(raw_job: dict[str, Any]) -> str:
    output_s3_uri = get_output_s3_uri_or_none(raw_job)
    if not output_s3_uri:
        raise AwsComprehendNamedEntitiesSchemaError(
            "$.EntitiesDetectionJobProperties.OutputDataConfig.S3Uri",
            "field required",
        )
    return output_s3_uri


def get_output_s3_uri_or_none(raw_job: dict[str, Any]) -> str | None:
    job_data = raw_job.get("EntitiesDetectionJobProperties") or {}
    return get_output_s3_uri_from_job_data(job_data)


def get_output_s3_uri_from_job_data(job_data: dict[str, Any]) -> str | None:
    output_config = job_data.get("OutputDataConfig") or {}
    output_s3_uri = output_config.get("S3Uri")
    return output_s3_uri if isinstance(output_s3_uri, str) else None


def get_input_s3_uri_or_none(job_data: dict[str, Any]) -> str | None:
    input_config = job_data.get("InputDataConfig") or {}
    input_s3_uri = input_config.get("S3Uri")
    return input_s3_uri if isinstance(input_s3_uri, str) else None


def get_input_s3_uri(raw_job: dict[str, Any]) -> str:
    job_data = raw_job.get("EntitiesDetectionJobProperties") or {}
    input_s3_uri = get_input_s3_uri_or_none(job_data)
    if not input_s3_uri:
        raise AwsComprehendNamedEntitiesSchemaError(
            "$.EntitiesDetectionJobProperties.InputDataConfig.S3Uri",
            "field required",
        )
    return input_s3_uri


def get_language_or_none(raw_job: dict[str, Any]) -> str | None:
    job_data = raw_job.get("EntitiesDetectionJobProperties") or {}
    language = job_data.get("LanguageCode")
    return language if isinstance(language, str) else None


def is_tool_managed_output_uri(output_s3_uri: str) -> bool:
    location = parse_s3_uri(output_s3_uri)
    return location.key.startswith(f"{_TOOL_MANAGED_OUTPUT_PREFIX}/")


def build_job_name(source: str | Path, prefix: str, timestamp: str) -> str:
    safe_prefix = safe_job_part(prefix) or "ampav-aws-comprehend-named-entities"
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
            raise AwsComprehendNamedEntitiesSchemaError(
                "$.EntitiesDetectionJobProperties.JobStatus",
                f"unknown AWS Comprehend status {status!r}",
            )


def json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, default=str))


def pydantic_loc_to_path(loc: tuple[Any, ...]) -> str:
    parts: list[str] = ["$"]
    for item in loc:
        if isinstance(item, int):
            parts[-1] += f"[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts)


def _status_from_job_data(job_data: dict[str, Any], *, details: bool = True) -> AsyncJobStatus:
    aws_status = job_data["JobStatus"]
    message = job_data.get("Message")
    status_data = {
        "job_id": job_data["JobId"],
        "status": map_aws_comprehend_status(aws_status),
        "message": message,
    }
    if not details:
        return AsyncJobStatus(**status_data)
    return AwsJobStatus(
        **status_data,
        job_name=job_data.get("JobName"),
        input_s3_uri=get_input_s3_uri_or_none(job_data),
        output_s3_uri=get_output_s3_uri_from_job_data(job_data),
    )


def _is_not_found_error(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code")
    return code in {"ResourceNotFoundException", "InvalidRequestException", "BadRequestException"}
