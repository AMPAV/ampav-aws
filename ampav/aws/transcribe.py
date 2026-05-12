from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ampav.core.logging import LOG_FORMAT
from ampav.core.utils import pretty_yaml


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AWSSettings(StrictBaseModel):
    region: str | None = None
    profile_name: str | None = None
    access_key_id: str | None = Field(default=None, alias="aws_access_key_id")
    secret_access_key: str | None = Field(default=None, alias="aws_secret_access_key")
    session_token: str | None = Field(default=None, alias="aws_session_token")

    @model_validator(mode="after")
    def validate_credentials(self) -> AWSSettings:
        explicit_keys = self.access_key_id or self.secret_access_key or self.session_token
        if self.profile_name and explicit_keys:
            raise ValueError("Use either profile_name or explicit AWS credentials, not both")
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise ValueError("access_key_id and secret_access_key must be configured together")
        return self


class S3Settings(StrictBaseModel):
    bucket: str
    input_prefix: str = "aws_transcribe/input"
    output_prefix: str = "aws_transcribe/output"


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


class PathSettings(StrictBaseModel):
    runs_dir: Path = Path("../runs")


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
    run_dir: Path
    transcript_json: Path | None = None
    status_history_json: Path
    log_file: Path


class AWSTranscribeService:
    def __init__(self, config: AWSTranscribeConfig):
        session = create_boto3_session(config.aws)
        self.config = config
        self.transcribe_client = session.client("transcribe")
        self.s3_client = session.client("s3")

    def upload_input(self, audiofile: Path, s3_key: str) -> str:
        logging.info("Uploading %s to s3://%s/%s", audiofile, self.config.s3.bucket, s3_key)
        self.s3_client.upload_file(str(audiofile), self.config.s3.bucket, s3_key)
        return f"s3://{self.config.s3.bucket}/{s3_key}"

    def start_job(self, request: dict[str, Any]) -> dict[str, Any]:
        logging.info("Starting AWS Transcribe job %s", request["TranscriptionJobName"])
        return self.transcribe_client.start_transcription_job(**request)

    def get_job(self, job_name: str) -> dict[str, Any]:
        return self.transcribe_client.get_transcription_job(TranscriptionJobName=job_name)

    def download_transcript(self, output_key: str, destination: Path) -> None:
        logging.info(
            "Downloading raw AWS transcript from s3://%s/%s to %s",
            self.config.s3.bucket,
            output_key,
            destination,
        )
        self.s3_client.download_file(self.config.s3.bucket, output_key, str(destination))


def load_config(config_path: Path) -> AWSTranscribeConfig:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    config = AWSTranscribeConfig.model_validate(raw)
    if not config.paths.runs_dir.is_absolute():
        config.paths.runs_dir = (config_path.parent / config.paths.runs_dir).resolve()
    return config


def create_boto3_session(settings: AWSSettings) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required to run AWS Transcribe jobs") from exc

    kwargs: dict[str, Any] = {}
    if settings.region:
        kwargs["region_name"] = settings.region
    if settings.profile_name:
        kwargs["profile_name"] = settings.profile_name
    elif settings.access_key_id and settings.secret_access_key:
        kwargs["aws_access_key_id"] = settings.access_key_id
        kwargs["aws_secret_access_key"] = settings.secret_access_key
        if settings.session_token:
            kwargs["aws_session_token"] = settings.session_token
    return boto3.Session(**kwargs)


def transcribe_file(audiofile: Path, config_path: Path, debug: bool = False) -> TranscribeRunResult:
    audiofile = audiofile.expanduser().resolve()
    config_path = config_path.expanduser().resolve()
    if not audiofile.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {audiofile}")

    config = load_config(config_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    job_name = build_job_name(audiofile, config.transcription.job_name_prefix, timestamp)
    run_dir = create_run_directory(config.paths.runs_dir, timestamp, job_name)
    log_file = run_dir / "aws_transcribe.log"
    configure_logging(log_file, debug=debug)

    logging.info("Starting AWS Transcribe run")
    logging.info("Input audio: %s", audiofile)
    logging.info("Run directory: %s", run_dir)

    service = AWSTranscribeService(config)
    input_key = join_s3_key(config.s3.input_prefix, f"{job_name}{audiofile.suffix}")
    output_key = join_s3_key(config.s3.output_prefix, f"{job_name}.json")
    transcript_json = run_dir / "aws_transcript.json"
    status_history_json = run_dir / "status_history.json"

    input_uri = service.upload_input(audiofile, input_key)
    request = build_start_job_request(config, job_name, input_uri, output_key, audiofile)
    write_json(run_dir / "request.json", request)
    write_json(run_dir / "config.redacted.json", redacted_config(config))

    start_response = service.start_job(request)
    write_json(run_dir / "start_response.json", start_response)

    final_job, status_history = poll_until_complete(
        service=service,
        job_name=job_name,
        polling=config.polling,
        status_history_path=status_history_json,
    )
    write_json(run_dir / "transcription_job.json", final_job)

    service.download_transcript(output_key, transcript_json)
    logging.info("AWS Transcribe job completed")

    result = TranscribeRunResult(
        job_name=job_name,
        status=final_job["TranscriptionJob"]["TranscriptionJobStatus"],
        input_uri=input_uri,
        output_bucket=config.s3.bucket,
        output_key=output_key,
        run_dir=run_dir,
        transcript_json=transcript_json,
        status_history_json=status_history_json,
        log_file=log_file,
    )
    write_json(run_dir / "run_result.json", result.model_dump(mode="json"))
    logging.info("Saved %d polling status snapshots", len(status_history))
    return result


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
    status_history_path: Path,
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


def configure_logging(log_file: Path, debug: bool = False) -> None:
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

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)


def create_run_directory(runs_dir: Path, timestamp: str, job_name: str) -> Path:
    runs_dir = runs_dir.expanduser()
    run_name = safe_path_part(f"{timestamp}_{job_name}")[:240]
    candidate = runs_dir / run_name
    suffix = 1
    while candidate.exists():
        candidate = runs_dir / f"{run_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def build_job_name(audiofile: Path, prefix: str, timestamp: str) -> str:
    safe_prefix = safe_job_part(prefix) or "ampav-aws-transcribe"
    safe_stem = safe_job_part(audiofile.stem) or "audio"
    max_stem_length = max(1, 200 - len(safe_prefix) - len(timestamp) - 2)
    return f"{safe_prefix}-{timestamp}-{safe_stem[:max_stem_length]}".strip("-")


def safe_job_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def join_s3_key(prefix: str, filename: str) -> str:
    clean_prefix = prefix.strip("/")
    return f"{clean_prefix}/{filename}" if clean_prefix else filename


def infer_media_format(audiofile: Path) -> str:
    media_format = audiofile.suffix.lower().lstrip(".")
    if not media_format:
        raise ValueError("media_format must be configured when the input file has no extension")
    return media_format


def redacted_config(config: AWSTranscribeConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    aws_data = data.get("aws", {})
    for key in ("access_key_id", "secret_access_key", "session_token"):
        if aws_data.get(key):
            aws_data[key] = "***"
    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


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
