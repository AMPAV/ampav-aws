"""AWS Transcribe orchestration and CLI entrypoint for AMPAV."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from ampav.core.logging import LOG_FORMAT
from ampav.core.schema import ToolOutput, Transcript
from ampav.core.utils import pretty_yaml

from .artifacts import create_artifact_writer, read_json as read_file_json, write_json
from .config import (
    AWSSettings,
    PathSettings,
    S3Settings,
    StrictBaseModel,
    load_yaml_mapping,
    redact_aws_credentials,
    resolve_path_from_config,
)
from .errors import (
    AWSConfigError,
    AWSTranscribeJobError,
    AWSTranscriptSchemaError,
    AmpavAWSError,
    is_aws_sdk_error,
)
from .s3 import S3Location, download_file, join_s3_key, read_json as read_s3_json, upload_file
from .session import create_boto3_session
from .transcribe_conversion import aws_transcript_to_transcript


class PollingSettings(StrictBaseModel):
    """Polling behavior for waiting on AWS Transcribe jobs.

    :param interval_seconds: Seconds to wait between AWS status checks.
        Defaults to ``30``.
    :type interval_seconds: int
    :param timeout_seconds: Maximum seconds to wait before failing. Optional;
        when ``None``, polling does not time out.
    :type timeout_seconds: int | None
    """

    interval_seconds: int = Field(default=30, ge=1)
    timeout_seconds: int | None = Field(default=7200, ge=1)


class TranscriptionSettings(StrictBaseModel):
    """AWS Transcribe job settings controlled by AMPAV config.

    :param media_format: AWS media format. Optional; when ``None``, inferred
        from the input file extension.
    :type media_format: str | None
    :param language_code: AWS language code used when language identification
        is disabled. Defaults to ``"en-US"``.
    :type language_code: str | None
    :param identify_language: Enable AWS language identification. Defaults to
        ``False``.
    :type identify_language: bool
    :param language_options: Optional language options passed to AWS when
        language identification is enabled.
    :type language_options: list[str]
    :param show_speaker_labels: Request AWS speaker diarization. Defaults to
        ``True``.
    :type show_speaker_labels: bool
    :param max_speaker_labels: Maximum number of AWS speaker labels. Defaults
        to ``10``.
    :type max_speaker_labels: int
    :param job_name_prefix: Prefix for generated AWS Transcribe job names.
    :type job_name_prefix: str
    """

    media_format: str | None = None
    language_code: str | None = "en-US"
    identify_language: bool = False
    language_options: list[str] = Field(default_factory=list)
    show_speaker_labels: bool = True
    max_speaker_labels: int = Field(default=10, ge=2, le=30)
    job_name_prefix: str = "ampav-aws-transcribe"

    @model_validator(mode="after")
    def validate_language_settings(self) -> TranscriptionSettings:
        """Validate the mutually exclusive AWS language configuration.

        :return: The validated transcription settings instance.
        :rtype: TranscriptionSettings
        :raises ValueError: If no language code is configured while language
            identification is disabled.
        """
        if not self.identify_language and not self.language_code:
            raise ValueError("language_code is required unless identify_language is true")
        return self


class AWSTranscribeConfig(StrictBaseModel):
    """Complete config object for running AWS Transcribe through AMPAV.

    :param aws: AWS credential and region settings.
    :type aws: AWSSettings
    :param s3: S3 bucket and key-prefix settings.
    :type s3: S3Settings
    :param polling: Polling settings for AWS job completion.
    :type polling: PollingSettings
    :param transcription: AWS Transcribe job settings.
    :type transcription: TranscriptionSettings
    :param paths: Optional local artifact path settings.
    :type paths: PathSettings
    """

    aws: AWSSettings = Field(default_factory=AWSSettings)
    s3: S3Settings
    polling: PollingSettings = Field(default_factory=PollingSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


class TranscribeRunResult(BaseModel):
    """Metadata describing one AWS Transcribe execution.

    :param job_name: AWS Transcribe job name.
    :type job_name: str
    :param status: Final AWS Transcribe job status.
    :type status: str
    :param input_uri: S3 URI for the uploaded input media.
    :type input_uri: str
    :param output_bucket: S3 bucket containing the raw AWS transcript JSON.
    :type output_bucket: str
    :param output_key: S3 key for the raw AWS transcript JSON.
    :type output_key: str
    :param run_dir: Optional local run artifact directory.
    :type run_dir: Path | None
    :param transcript_json: Optional local raw AWS transcript JSON path.
    :type transcript_json: Path | None
    :param status_history_json: Optional local polling history JSON path.
    :type status_history_json: Path | None
    :param log_file: Optional local run log file path.
    :type log_file: Path | None
    """

    job_name: str
    status: str
    input_uri: str
    output_bucket: str
    output_key: str
    run_dir: Path | None = None
    transcript_json: Path | None = None
    status_history_json: Path | None = None
    log_file: Path | None = None


class AWSTranscribeService:
    """Thin wrapper around boto3 clients used by the Transcribe workflow.

    :param config: Typed AWS Transcribe configuration.
    :type config: AWSTranscribeConfig
    """

    def __init__(self, config: AWSTranscribeConfig):
        """Create AWS Transcribe and S3 clients from a typed config object.

        :param config: Typed AWS Transcribe configuration.
        :type config: AWSTranscribeConfig
        """
        session = create_boto3_session(config.aws)
        self.config = config
        self.transcribe_client = session.client("transcribe")
        self.s3_client = session.client("s3")

    def upload_input(self, audiofile: Path, destination: S3Location) -> str:
        """Upload an input audio file and return its S3 URI.

        :param audiofile: Local audio file to upload.
        :type audiofile: Path
        :param destination: Destination S3 bucket/key.
        :type destination: S3Location
        :return: Uploaded media URI.
        :rtype: str
        """
        logging.info("Uploading %s to %s", audiofile, destination.uri)
        upload_file(self.s3_client, audiofile, destination)
        return destination.uri

    def start_job(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit an AWS Transcribe start_transcription_job request.

        :param request: boto3 ``start_transcription_job`` request body.
        :type request: dict[str, Any]
        :return: AWS start job response.
        :rtype: dict[str, Any]
        """
        logging.info("Starting AWS Transcribe job %s", request["TranscriptionJobName"])
        return self.transcribe_client.start_transcription_job(**request)

    def get_job(self, job_name: str) -> dict[str, Any]:
        """Fetch one AWS Transcribe job status response.

        :param job_name: AWS Transcribe job name.
        :type job_name: str
        :return: AWS get job response.
        :rtype: dict[str, Any]
        """
        return self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)

    def download_transcript(self, source: S3Location, destination: Path) -> None:
        """Download the raw AWS transcript JSON from S3.

        :param source: S3 location for the raw transcript JSON.
        :type source: S3Location
        :param destination: Local destination path.
        :type destination: Path
        """
        logging.info("Downloading raw AWS transcript from %s to %s", source.uri, destination)
        download_file(self.s3_client, source, destination)

    def read_transcript(self, source: S3Location) -> Any:
        """Read the raw AWS transcript JSON directly from S3.

        :param source: S3 location for the raw transcript JSON.
        :type source: S3Location
        :return: Parsed AWS transcript JSON.
        :rtype: Any
        """
        logging.info("Reading raw AWS transcript from %s", source.uri)
        return read_s3_json(self.s3_client, source)


def load_config(config_path: Path) -> AWSTranscribeConfig:
    """Load and validate an AWS Transcribe YAML config file.

    :param config_path: Path to the local YAML config file.
    :type config_path: Path
    :return: Typed AWS Transcribe config with relative paths resolved.
    :rtype: AWSTranscribeConfig
    :raises AWSConfigError: If the config cannot be read, parsed, or validated.
    """
    try:
        config_path = config_path.expanduser().resolve()
        config = AWSTranscribeConfig.model_validate(load_yaml_mapping(config_path))
        config.paths.runs_dir = resolve_path_from_config(config_path, config.paths.runs_dir)
        return config
    except AWSConfigError:
        raise
    except ValidationError as exc:
        raise AWSConfigError(f"Invalid AWS Transcribe config {config_path}: {exc}") from exc


def transcribe_file(audiofile: Path, config_path: Path, debug: bool = False) -> ToolOutput:
    """Transcribe a local file using config loaded from a YAML file.

    :param audiofile: Local media file to upload and transcribe.
    :type audiofile: Path
    :param config_path: YAML config file containing AWS, S3, polling, and
        transcription settings.
    :type config_path: Path
    :param debug: Enable debug logging. Defaults to ``False``.
    :type debug: bool
    :return: ToolOutput containing the normalized AMPAV Transcript and run metadata.
    :rtype: ToolOutput
    """
    config_path = config_path.expanduser().resolve()
    return transcribe_file_with_config(
        audiofile=audiofile,
        config=load_config(config_path),
        config_path=config_path,
        debug=debug,
    )


def transcribe_file_with_config(
    audiofile: Path,
    config: AWSTranscribeConfig,
    config_path: Path | None = None,
    debug: bool = False,
) -> ToolOutput:
    """Transcribe a local file using an already-loaded config object.

    :param audiofile: Local media file to upload and transcribe.
    :type audiofile: Path
    :param config: Typed AWS Transcribe configuration.
    :type config: AWSTranscribeConfig
    :param config_path: Optional path to the config file that produced
        ``config``. Included in ToolOutput parameters when provided.
    :type config_path: Path | None
    :param debug: Enable debug logging. Defaults to ``False``.
    :type debug: bool
    :return: ToolOutput containing the normalized AMPAV Transcript and run metadata.
    :rtype: ToolOutput
    """
    audiofile = audiofile.expanduser().resolve()
    if not audiofile.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {audiofile}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_name = build_job_name(audiofile, config.transcription.job_name_prefix, timestamp)
    artifacts = create_artifact_writer(config.paths.runs_dir, timestamp, job_name)
    log_file = artifacts.path("aws_transcribe.log")
    configure_logging(log_file, debug=debug)

    output = ToolOutput(
        tool_name="aws_transcribe",
        parameters=initial_tool_parameters(audiofile, config_path, config),
    )
    output.setup_logging()

    logging.info("Starting AWS Transcribe run")
    logging.info("Input audio: %s", audiofile)
    if artifacts.run_dir is not None:
        logging.info("Run directory: %s", artifacts.run_dir)
    else:
        logging.info("Run artifact persistence is disabled")

    service = AWSTranscribeService(config)
    input_location = S3Location(
        bucket=config.s3.bucket,
        key=join_s3_key(config.s3.input_prefix, f"{job_name}{audiofile.suffix}"),
    )
    output_location = S3Location(
        bucket=config.s3.bucket,
        key=join_s3_key(config.s3.output_prefix, f"{job_name}.json"),
    )
    transcript_json_path = artifacts.path("aws_transcript.json")
    status_history_path = artifacts.path("status_history.json")

    input_uri = service.upload_input(audiofile, input_location)
    request = build_start_job_request(config, job_name, input_uri, output_location.key, audiofile)
    artifacts.write_json("request.json", request)
    artifacts.write_json("config.redacted.json", redacted_config(config))
    logging.debug("AWS Transcribe request: %s", json_for_log(request))

    output.queue_time = time.time()
    start_response = service.start_job(request)
    artifacts.write_json("start_response.json", start_response)
    logging.debug("AWS Transcribe start response: %s", json_for_log(start_response))

    final_job, status_history = poll_until_complete(
        service=service,
        job_name=job_name,
        polling=config.polling,
        status_history_path=status_history_path,
    )
    artifacts.write_json("transcription_job.json", final_job)
    logging.debug("AWS Transcribe final job: %s", json_for_log(final_job))
    logging.debug("AWS Transcribe status history: %s", json_for_log(status_history))

    aws_transcript = read_aws_transcript(service, output_location, transcript_json_path)
    logging.info("AWS Transcribe job completed")
    result = TranscribeRunResult(
        job_name=job_name,
        status=final_job["TranscriptionJob"]["TranscriptionJobStatus"],
        input_uri=input_uri,
        output_bucket=output_location.bucket,
        output_key=output_location.key,
        run_dir=artifacts.run_dir,
        transcript_json=transcript_json_path,
        status_history_json=status_history_path,
        log_file=log_file,
    )
    artifacts.write_json("run_result.json", result.model_dump(mode="json"))

    transcript = aws_transcript_to_transcript(aws_transcript)
    logging.info("Saved %d polling status snapshots", len(status_history))
    logging.info(
        "Converted AWS transcript to AMPAV schema with %d paragraphs and %d words",
        len(transcript.paragraphs),
        len(transcript.words),
    )
    return finalize_tool_output(
        output=output,
        transcript=transcript,
        run_result=result,
        final_job=final_job,
    )


def read_aws_transcript(
    service: AWSTranscribeService,
    source: S3Location,
    transcript_json_path: Path | None,
) -> dict[str, Any]:
    """Read or download raw AWS transcript JSON for conversion.

    :param service: AWS Transcribe service wrapper with S3 access.
    :type service: AWSTranscribeService
    :param source: S3 location for the raw AWS transcript JSON.
    :type source: S3Location
    :param transcript_json_path: Optional local path for downloading the raw
        transcript. When ``None``, the JSON is read directly from S3.
    :type transcript_json_path: Path | None
    :return: Raw AWS transcript JSON object.
    :rtype: dict[str, Any]
    :raises AWSTranscriptSchemaError: If the raw JSON root is not an object.
    """
    if transcript_json_path is None:
        aws_transcript = service.read_transcript(source)
    else:
        service.download_transcript(source, transcript_json_path)
        aws_transcript = read_file_json(transcript_json_path)
    if not isinstance(aws_transcript, dict):
        raise AWSTranscriptSchemaError("$", "AWS transcript JSON must contain an object")
    return aws_transcript


def initial_tool_parameters(
    audiofile: Path,
    config_path: Path | None,
    config: AWSTranscribeConfig,
) -> dict[str, Any]:
    """Build initial ToolOutput parameters known before AWS submission.

    :param audiofile: Local media file being transcribed.
    :type audiofile: Path
    :param config_path: Optional config path included for traceability.
    :type config_path: Path | None
    :param config: Typed AWS Transcribe configuration.
    :type config: AWSTranscribeConfig
    :return: ToolOutput parameter dictionary.
    :rtype: dict[str, Any]
    """
    parameters = {
        "content_source": str(audiofile),
        "config": str(config_path) if config_path is not None else None,
        "aws_region": config.aws.region,
        "language_code": config.transcription.language_code,
        "identify_language": config.transcription.identify_language,
        "language_options": config.transcription.language_options,
    }
    return {key: value for key, value in parameters.items() if value is not None}


def finalize_tool_output(
    output: ToolOutput,
    transcript: Transcript,
    run_result: TranscribeRunResult,
    final_job: dict[str, Any],
) -> ToolOutput:
    """Attach final run metadata and normalized transcript to ToolOutput.

    :param output: ToolOutput created before AWS submission.
    :type output: ToolOutput
    :param transcript: Normalized AMPAV transcript.
    :type transcript: Transcript
    :param run_result: Metadata for this AWS Transcribe run.
    :type run_result: TranscribeRunResult
    :param final_job: Final AWS ``get_transcription_job`` response.
    :type final_job: dict[str, Any]
    :return: The updated ToolOutput instance.
    :rtype: ToolOutput
    """
    job_data = final_job.get("TranscriptionJob", {})
    parameters = {
        **output.parameters,
        "status": run_result.status,
        "job_name": run_result.job_name,
        "s3_input_uri": run_result.input_uri,
        "s3_output_bucket": run_result.output_bucket,
        "s3_output_key": run_result.output_key,
        "run_dir": str(run_result.run_dir) if run_result.run_dir is not None else None,
        "raw_transcript_json": str(run_result.transcript_json) if run_result.transcript_json else None,
        "status_history_json": str(run_result.status_history_json) if run_result.status_history_json else None,
        "log_file": str(run_result.log_file) if run_result.log_file else None,
    }
    output.parameters = {key: value for key, value in parameters.items() if value is not None}
    output.start_time = timestamp_or_none(job_data.get("StartTime")) or timestamp_or_none(job_data.get("CreationTime"))
    output.end_time = timestamp_or_none(job_data.get("CompletionTime")) or time.time()
    output.output = transcript
    return output


def timestamp_or_none(value: object) -> float | None:
    """Convert boto3 datetime values to POSIX timestamps.

    :param value: Value returned by boto3, usually ``datetime`` or ``None``.
    :type value: object
    :return: POSIX timestamp or ``None`` when value is not a datetime.
    :rtype: float | None
    """
    if isinstance(value, datetime):
        return value.timestamp()
    return None


def build_start_job_request(
    config: AWSTranscribeConfig,
    job_name: str,
    input_uri: str,
    output_key: str,
    audiofile: Path,
) -> dict[str, Any]:
    """Build the boto3 start_transcription_job request body.

    :param config: Typed AWS Transcribe configuration.
    :type config: AWSTranscribeConfig
    :param job_name: Generated AWS Transcribe job name.
    :type job_name: str
    :param input_uri: S3 URI of the uploaded input media.
    :type input_uri: str
    :param output_key: S3 key where AWS should write transcript JSON.
    :type output_key: str
    :param audiofile: Local media file, used to infer media format when needed.
    :type audiofile: Path
    :return: boto3 ``start_transcription_job`` request body.
    :rtype: dict[str, Any]
    """
    media_format = config.transcription.media_format or infer_media_format(audiofile)
    request: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": input_uri},
        "MediaFormat": media_format,
        "OutputBucketName": config.s3.bucket,
        "OutputKey": output_key,
    }

    if config.transcription.identify_language:
        request["IdentifyLanguage"] = True
        if config.transcription.language_options:
            request["LanguageOptions"] = config.transcription.language_options
    else:
        request["LanguageCode"] = config.transcription.language_code

    settings: dict[str, Any] = {}
    if config.transcription.show_speaker_labels:
        settings["ShowSpeakerLabels"] = True
        settings["MaxSpeakerLabels"] = config.transcription.max_speaker_labels
    if settings:
        request["Settings"] = settings

    return request


def poll_until_complete(
    service: AWSTranscribeService,
    job_name: str,
    polling: PollingSettings,
    status_history_path: Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Poll AWS Transcribe until a job completes, fails, or times out.

    :param service: AWS Transcribe service wrapper.
    :type service: AWSTranscribeService
    :param job_name: AWS Transcribe job name to poll.
    :type job_name: str
    :param polling: Polling interval and timeout settings.
    :type polling: PollingSettings
    :param status_history_path: Optional path for persisting polling snapshots.
    :type status_history_path: Path | None
    :return: Final AWS job response and polling status history.
    :rtype: tuple[dict[str, Any], list[dict[str, Any]]]
    :raises AWSTranscribeJobError: If the job fails or polling times out.
    """
    started = time.monotonic()
    history: list[dict[str, Any]] = []

    while True:
        job = service.get_job(job_name)
        job_data = job["TranscriptionJob"]
        status = job_data["TranscriptionJobStatus"]
        snapshot = {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "failure_reason": job_data.get("FailureReason"),
            "transcript_file_uri": job_data.get("Transcript", {}).get("TranscriptFileUri"),
        }
        history.append(snapshot)
        if status_history_path is not None:
            write_json(status_history_path, history)
        logging.info("AWS Transcribe job %s status: %s", job_name, status)

        if status == "COMPLETED":
            return job, history
        if status == "FAILED":
            reason = job_data.get("FailureReason", "no failure reason returned")
            raise AWSTranscribeJobError(job_name, f"failed: {reason}")
        if polling.timeout_seconds is not None and time.monotonic() - started > polling.timeout_seconds:
            raise AWSTranscribeJobError(job_name, "did not finish within timeout")

        time.sleep(polling.interval_seconds)


def configure_logging(log_file: Path | None, debug: bool = False) -> None:
    """Configure console logging and optional per-run file logging.

    :param log_file: Optional log file path. When ``None``, file logging is
        disabled.
    :type log_file: Path | None
    :param debug: Enable debug logging. Defaults to ``False``.
    :type debug: bool
    """
    level = logging.DEBUG if debug else logging.INFO
    formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    root_logger.addHandler(stream_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


def build_job_name(audiofile: Path, prefix: str, timestamp: str) -> str:
    """Build an AWS-compatible Transcribe job name.

    :param audiofile: Local media file used for the job-name stem.
    :type audiofile: Path
    :param prefix: Configured job-name prefix.
    :type prefix: str
    :param timestamp: Timestamp string included to make names unique.
    :type timestamp: str
    :return: AWS-compatible Transcribe job name.
    :rtype: str
    """
    safe_prefix = safe_job_part(prefix) or "ampav-aws-transcribe"
    safe_stem = safe_job_part(audiofile.stem) or "audio"
    max_stem_length = max(1, 200 - len(safe_prefix) - len(timestamp) - 2)
    return f"{safe_prefix}-{timestamp}-{safe_stem[:max_stem_length]}".strip("-")


def safe_job_part(value: str) -> str:
    """Sanitize a string for use in AWS Transcribe job names.

    :param value: Raw job-name component.
    :type value: str
    :return: Sanitized job-name component.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def infer_media_format(audiofile: Path) -> str:
    """Infer AWS Transcribe MediaFormat from a local file extension.

    :param audiofile: Local media file.
    :type audiofile: Path
    :return: Lowercase extension without the leading dot.
    :rtype: str
    :raises ValueError: If the file has no extension.
    """
    media_format = audiofile.suffix.lower().lstrip(".")
    if not media_format:
        raise ValueError("media_format must be configured when the input file has no extension")
    return media_format


def redacted_config(config: AWSTranscribeConfig) -> dict[str, Any]:
    """Return a serialized config with AWS credentials redacted.

    :param config: Typed AWS Transcribe configuration.
    :type config: AWSTranscribeConfig
    :return: Serialized config dictionary with credential values redacted.
    :rtype: dict[str, Any]
    """
    return redact_aws_credentials(config.model_dump(mode="json"))


def json_for_log(data: Any) -> str:
    """Serialize data for compact debug logging.

    :param data: Value to serialize.
    :type data: Any
    :return: JSON string suitable for logs.
    :rtype: str
    """
    return json.dumps(data, default=str, sort_keys=True)


def cli_aws_transcribe() -> None:
    """CLI entrypoint for AWS Transcribe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Audio file to transcribe using AWS Transcribe")
    parser.add_argument("--config", required=True, type=Path, help="Path to local AWS Transcribe YAML config")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        result = transcribe_file(Path(args.file), args.config, debug=args.debug)
    except Exception as exc:
        if not is_cli_error(exc):
            raise
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
        logging.error("%s", exc)
        raise SystemExit(1) from exc

    print(pretty_yaml(result.model_dump(mode="json", exclude_none=True), sort_keys=False))


def is_cli_error(exc: BaseException) -> bool:
    """Return whether the CLI should report an exception as a user-facing error.

    :param exc: Exception raised while running the CLI.
    :type exc: BaseException
    :return: ``True`` when the CLI should log the error and exit nonzero.
    :rtype: bool
    """
    return isinstance(exc, (AmpavAWSError, ValidationError, OSError, RuntimeError, TimeoutError, ValueError)) or (
        is_aws_sdk_error(exc)
    )


if __name__ == "__main__":
    cli_aws_transcribe()
