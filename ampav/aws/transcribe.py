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
from .s3 import S3Location, download_file, join_s3_key, read_json as read_s3_json, upload_file
from .session import create_boto3_session
from .transcribe_conversion import aws_transcript_to_transcript


class PollingSettings(StrictBaseModel):
    interval_seconds: int = Field(default=30, ge=1)
    timeout_seconds: int | None = Field(default=7200, ge=1)


class TranscriptionSettings(StrictBaseModel):
    media_format: str | None = None
    language_code: str | None = "en-US"
    identify_language: bool = False
    language_options: list[str] = Field(default_factory=list)
    show_speaker_labels: bool = True
    max_speaker_labels: int = Field(default=10, ge=2, le=30)
    job_name_prefix: str = "ampav-aws-transcribe"

    @model_validator(mode="after")
    def validate_language_settings(self) -> TranscriptionSettings:
        if not self.identify_language and not self.language_code:
            raise ValueError("language_code is required unless identify_language is true")
        return self


class AWSTranscribeConfig(StrictBaseModel):
    aws: AWSSettings = Field(default_factory=AWSSettings)
    s3: S3Settings
    polling: PollingSettings = Field(default_factory=PollingSettings)
    transcription: TranscriptionSettings = Field(default_factory=TranscriptionSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


class TranscribeRunResult(BaseModel):
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
    def __init__(self, config: AWSTranscribeConfig):
        session = create_boto3_session(config.aws)
        self.config = config
        self.transcribe_client = session.client("transcribe")
        self.s3_client = session.client("s3")

    def upload_input(self, audiofile: Path, destination: S3Location) -> str:
        logging.info("Uploading %s to %s", audiofile, destination.uri)
        upload_file(self.s3_client, audiofile, destination)
        return destination.uri

    def start_job(self, request: dict[str, Any]) -> dict[str, Any]:
        logging.info("Starting AWS Transcribe job %s", request["TranscriptionJobName"])
        return self.transcribe_client.start_transcription_job(**request)

    def get_job(self, job_name: str) -> dict[str, Any]:
        return self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)

    def download_transcript(self, source: S3Location, destination: Path) -> None:
        logging.info("Downloading raw AWS transcript from %s to %s", source.uri, destination)
        download_file(self.s3_client, source, destination)

    def read_transcript(self, source: S3Location) -> Any:
        logging.info("Reading raw AWS transcript from %s", source.uri)
        return read_s3_json(self.s3_client, source)


def load_config(config_path: Path) -> AWSTranscribeConfig:
    config_path = config_path.expanduser().resolve()
    config = AWSTranscribeConfig.model_validate(load_yaml_mapping(config_path))
    config.paths.runs_dir = resolve_path_from_config(config_path, config.paths.runs_dir)
    return config


def transcribe_file(audiofile: Path, config_path: Path, debug: bool = False) -> ToolOutput:
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
    if transcript_json_path is None:
        aws_transcript = service.read_transcript(source)
    else:
        service.download_transcript(source, transcript_json_path)
        aws_transcript = read_file_json(transcript_json_path)
    if not isinstance(aws_transcript, dict):
        raise ValueError("AWS transcript JSON must contain an object")
    return aws_transcript


def initial_tool_parameters(
    audiofile: Path,
    config_path: Path | None,
    config: AWSTranscribeConfig,
) -> dict[str, Any]:
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
            raise RuntimeError(f"AWS Transcribe job {job_name} failed: {reason}")
        if polling.timeout_seconds is not None and time.monotonic() - started > polling.timeout_seconds:
            raise TimeoutError(f"AWS Transcribe job {job_name} did not finish within timeout")

        time.sleep(polling.interval_seconds)


def configure_logging(log_file: Path | None, debug: bool = False) -> None:
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
    safe_prefix = safe_job_part(prefix) or "ampav-aws-transcribe"
    safe_stem = safe_job_part(audiofile.stem) or "audio"
    max_stem_length = max(1, 200 - len(safe_prefix) - len(timestamp) - 2)
    return f"{safe_prefix}-{timestamp}-{safe_stem[:max_stem_length]}".strip("-")


def safe_job_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def infer_media_format(audiofile: Path) -> str:
    media_format = audiofile.suffix.lower().lstrip(".")
    if not media_format:
        raise ValueError("media_format must be configured when the input file has no extension")
    return media_format


def redacted_config(config: AWSTranscribeConfig) -> dict[str, Any]:
    return redact_aws_credentials(config.model_dump(mode="json"))


def json_for_log(data: Any) -> str:
    return json.dumps(data, default=str, sort_keys=True)


def cli_aws_transcribe() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Audio file to transcribe using AWS Transcribe")
    parser.add_argument("--config", required=True, type=Path, help="Path to local AWS Transcribe YAML config")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    try:
        result = transcribe_file(Path(args.file), args.config, debug=args.debug)
    except (ValidationError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)
        logging.error("%s", exc)
        raise SystemExit(1) from exc

    print(pretty_yaml(result.model_dump(mode="json", exclude_none=True), sort_keys=False))


if __name__ == "__main__":
    cli_aws_transcribe()
