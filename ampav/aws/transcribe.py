"""AWS Transcribe client and CLI for AMPAV."""

import argparse
from datetime import datetime, timezone
import json
import logging
from os import PathLike
from pathlib import Path
import re
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ampav.core.logging import LOG_FORMAT
from ampav.core.schema import ToolOutput

from .errors import AwsTranscribeError, AwsTranscriptSchemaError
from .s3 import S3Location, join_s3_key, parse_s3_uri
from .transcribe_contract import validate_aws_transcript_contract
from .transcribe_conversion import aws_transcript_to_transcript


class StrictModel(BaseModel):
    """Pydantic base model for user-facing settings objects."""

    model_config = ConfigDict(extra="forbid")


class PollingSettings(StrictModel):
    """Polling controls for waiting on an AWS Transcribe batch job."""

    interval_seconds: int = Field(default=30, ge=1)
    timeout_seconds: int | None = Field(default=7200, ge=1)


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


class AwsTranscribeJob(StrictModel):
    """Public handle for a submitted AWS Transcribe job."""

    name: str
    media_uri: str
    output_bucket: str
    output_key: str
    media_was_uploaded: bool = False

    @property
    def output_location(self) -> S3Location:
        """S3 location where AWS writes the transcript JSON."""
        return S3Location(self.output_bucket, self.output_key)

    @property
    def media_location(self) -> S3Location:
        """S3 location of the media submitted to AWS Transcribe."""
        return parse_s3_uri(self.media_uri)


class AwsTranscribe:
    """Low-level AWS Transcribe client used by the AMPAV API and CLI."""

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        transcribe_client: Any | None = None,
        s3_client: Any | None = None,
    ):
        """Create an AWS Transcribe client from boto3 session settings or injected clients.

        `profile_name` and `region_name` are passed to boto3's normal session
        constructor. Tests and higher-level callers may pass already-created
        clients to avoid owning credential/session setup here.
        """
        if session is None and (transcribe_client is None or s3_client is None):
            session = boto3.Session(region_name=region_name, profile_name=profile_name)
        self.transcribe_client = transcribe_client or session.client("transcribe")
        self.s3_client = s3_client or session.client("s3")

    def submit(
        self,
        media_uri: str,
        *,
        output_bucket: str,
        output_key: str | None = None,
        output_prefix: str = "aws_transcribe/output",
        job_name: str | None = None,
        job_name_prefix: str = "ampav-aws-transcribe",
        transcription: TranscriptionSettings | None = None,
    ) -> AwsTranscribeJob:
        """Submit an AWS Transcribe job for media that already exists in S3.

        The returned `AwsTranscribeJob` is the public handle callers can store,
        poll, fetch, or clean up later.
        """
        transcription = transcription or TranscriptionSettings()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_name = job_name or build_job_name(media_uri, job_name_prefix, timestamp)
        output_key = output_key or join_s3_key(output_prefix, f"{job_name}.json")
        request = build_start_job_request(
            job_name=job_name,
            media_uri=media_uri,
            output_bucket=output_bucket,
            output_key=output_key,
            transcription=transcription,
        )

        logging.info("Starting AWS Transcribe job %s", job_name)
        logging.debug("AWS Transcribe request: %s", json.dumps(request, default=str, sort_keys=True))
        self.transcribe_client.start_transcription_job(**request)
        return AwsTranscribeJob(
            name=job_name,
            media_uri=media_uri,
            output_bucket=output_bucket,
            output_key=output_key,
        )

    def submit_file(
        self,
        audiofile: str | PathLike[str],
        *,
        output_bucket: str,
        input_bucket: str | None = None,
        input_key: str | None = None,
        input_prefix: str = "aws_transcribe/input",
        output_key: str | None = None,
        output_prefix: str = "aws_transcribe/output",
        job_name: str | None = None,
        job_name_prefix: str = "ampav-aws-transcribe",
        transcription: TranscriptionSettings | None = None,
    ) -> AwsTranscribeJob:
        """Upload a local media file to S3 and submit an AWS Transcribe job."""
        audio_path = Path(audiofile).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio file does not exist: {audio_path}")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        job_name = job_name or build_job_name(str(audio_path), job_name_prefix, timestamp)
        input_bucket = input_bucket or output_bucket
        input_key = input_key or join_s3_key(input_prefix, f"{job_name}{audio_path.suffix}")
        input_location = S3Location(input_bucket, input_key)

        logging.info("Uploading %s to %s", audio_path, input_location.uri)
        self.s3_client.upload_file(str(audio_path), input_location.bucket, input_location.key)

        job = self.submit(
            input_location.uri,
            output_bucket=output_bucket,
            output_key=output_key,
            output_prefix=output_prefix,
            job_name=job_name,
            job_name_prefix=job_name_prefix,
            transcription=transcription,
        )
        return job.model_copy(update={"media_was_uploaded": True})

    def job_info(self, job_name: str | AwsTranscribeJob) -> dict[str, Any]:
        """Return AWS metadata for a submitted transcription job."""
        name = job_name.name if isinstance(job_name, AwsTranscribeJob) else job_name
        return self.transcribe_client.get_transcription_job(TranscriptionJobName=name)

    def list_jobs(self, **kwargs: Any) -> dict[str, Any]:
        """Return AWS Transcribe job summaries using boto3 list arguments."""
        return self.transcribe_client.list_transcription_jobs(**kwargs)

    def wait(
        self,
        job: str | AwsTranscribeJob,
        *,
        polling: PollingSettings | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Poll until an AWS Transcribe job completes, fails, or times out.

        Returns the final `get_transcription_job` response and a lightweight
        status history that can be copied into `ToolOutput.tool_private`.
        """
        polling = polling or PollingSettings()
        job_name = job.name if isinstance(job, AwsTranscribeJob) else job
        started = time.monotonic()
        history: list[dict[str, Any]] = []

        while True:
            response = self.job_info(job_name)
            job_data = response["TranscriptionJob"]
            status = job_data["TranscriptionJobStatus"]
            snapshot = {
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "failure_reason": job_data.get("FailureReason"),
                "transcript_file_uri": job_data.get("Transcript", {}).get("TranscriptFileUri"),
            }
            history.append(snapshot)
            logging.info("AWS Transcribe job %s status: %s", job_name, status)

            if status == "COMPLETED":
                return response, history
            if status == "FAILED":
                reason = job_data.get("FailureReason", "no failure reason returned")
                raise AwsTranscribeError(job_name, f"failed: {reason}")
            if polling.timeout_seconds is not None and time.monotonic() - started > polling.timeout_seconds:
                raise AwsTranscribeError(job_name, "did not finish within timeout")

            time.sleep(polling.interval_seconds)

    def get_transcript_json(self, job: AwsTranscribeJob) -> dict[str, Any]:
        """Read and parse the raw AWS transcript JSON from the job output S3 object."""
        logging.info("Reading raw AWS transcript from %s", job.output_location.uri)
        response = self.s3_client.get_object(Bucket=job.output_bucket, Key=job.output_key)
        with response["Body"] as body:
            transcript = json.loads(body.read().decode("utf-8"))
        if not isinstance(transcript, dict):
            raise AwsTranscriptSchemaError("$", "AWS transcript JSON must contain an object")
        return transcript

    def get_transcription(
        self,
        job: AwsTranscribeJob,
        *,
        final_job: dict[str, Any] | None = None,
        status_history: list[dict[str, Any]] | None = None,
    ) -> ToolOutput:
        """Fetch raw AWS output and convert it to an AMPAV `ToolOutput`.

        Raw AWS job and transcript payloads are stored in `tool_private` for
        troubleshooting. Normal clients should consume `output`.
        """
        final_job = final_job or self.job_info(job)
        aws_transcript = self.get_transcript_json(job)
        aws_model = validate_aws_transcript_contract(aws_transcript)
        transcript = aws_transcript_to_transcript(aws_model)
        job_data = final_job.get("TranscriptionJob", {})

        output = ToolOutput(
            tool_name="aws_transcribe",
            parameters=tool_parameters(job, job_data),
            queue_time=timestamp_or_none(job_data.get("CreationTime")),
            start_time=timestamp_or_none(job_data.get("StartTime")) or timestamp_or_none(job_data.get("CreationTime")),
            end_time=timestamp_or_none(job_data.get("CompletionTime")) or time.time(),
            output=transcript,
            tool_private={
                "aws_transcribe_job": job.model_dump(mode="json"),
                "raw_transcription_job": json_safe(final_job),
                "raw_transcript": json_safe(aws_transcript),
                "status_history": status_history or [],
            },
        )
        return output

    def cleanup(
        self,
        job: AwsTranscribeJob,
        *,
        delete_job: bool = False,
        delete_input: bool = False,
        delete_output: bool = False,
    ) -> None:
        """Delete selected AWS-side job/S3 resources.

        Cleanup is intentionally opt-in. `delete_input` removes the media object
        referenced by the job handle; use it only when this library uploaded or
        otherwise owns that object.
        """
        if delete_output:
            self.s3_client.delete_object(Bucket=job.output_bucket, Key=job.output_key)
        if delete_input:
            media = job.media_location
            self.s3_client.delete_object(Bucket=media.bucket, Key=media.key)
        if delete_job:
            self.transcribe_client.delete_transcription_job(TranscriptionJobName=job.name)


def transcribe_file(
    audiofile: str | PathLike[str],
    *,
    output_bucket: str,
    input_bucket: str | None = None,
    input_key: str | None = None,
    input_prefix: str = "aws_transcribe/input",
    output_key: str | None = None,
    output_prefix: str = "aws_transcribe/output",
    job_name: str | None = None,
    job_name_prefix: str = "ampav-aws-transcribe",
    transcription: TranscriptionSettings | None = None,
    polling: PollingSettings | None = None,
    delete_job: bool = False,
    delete_input: bool = False,
    delete_output: bool = False,
    region_name: str | None = None,
    profile_name: str | None = None,
    session: Any | None = None,
    client: AwsTranscribe | None = None,
) -> ToolOutput:
    """Upload local media, transcribe it, and return AMPAV output.

    Local paths are uploaded before submission. Use `transcribe_uri` for media
    that already exists in S3. Cleanup flags are explicit and disabled by
    default.
    """
    aws = client or AwsTranscribe(region_name=region_name, profile_name=profile_name, session=session)
    job = aws.submit_file(
        audiofile,
        output_bucket=output_bucket,
        input_bucket=input_bucket,
        input_key=input_key,
        input_prefix=input_prefix,
        output_key=output_key,
        output_prefix=output_prefix,
        job_name=job_name,
        job_name_prefix=job_name_prefix,
        transcription=transcription,
    )
    final_job, history = aws.wait(job, polling=polling)
    output = aws.get_transcription(job, final_job=final_job, status_history=history)
    aws.cleanup(job, delete_job=delete_job, delete_input=delete_input, delete_output=delete_output)
    return output


def transcribe_uri(
    media_uri: str,
    *,
    output_bucket: str,
    output_key: str | None = None,
    output_prefix: str = "aws_transcribe/output",
    job_name: str | None = None,
    job_name_prefix: str = "ampav-aws-transcribe",
    transcription: TranscriptionSettings | None = None,
    polling: PollingSettings | None = None,
    delete_job: bool = False,
    delete_output: bool = False,
    region_name: str | None = None,
    profile_name: str | None = None,
    session: Any | None = None,
    client: AwsTranscribe | None = None,
) -> ToolOutput:
    """Submit, wait for, and fetch a transcription for existing S3 media."""
    aws = client or AwsTranscribe(region_name=region_name, profile_name=profile_name, session=session)
    job = aws.submit(
        media_uri,
        output_bucket=output_bucket,
        output_key=output_key,
        output_prefix=output_prefix,
        job_name=job_name,
        job_name_prefix=job_name_prefix,
        transcription=transcription,
    )
    final_job, history = aws.wait(job, polling=polling)
    output = aws.get_transcription(job, final_job=final_job, status_history=history)
    aws.cleanup(job, delete_job=delete_job, delete_output=delete_output)
    return output


def build_start_job_request(
    *,
    job_name: str,
    media_uri: str,
    output_bucket: str,
    output_key: str,
    transcription: TranscriptionSettings,
) -> dict[str, Any]:
    """Build a boto3 `start_transcription_job` request body."""
    request: dict[str, Any] = {
        "TranscriptionJobName": job_name,
        "Media": {"MediaFileUri": media_uri},
        "MediaFormat": transcription.media_format or infer_media_format(media_uri),
        "OutputBucketName": output_bucket,
        "OutputKey": output_key,
    }

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
        raise ValueError("media_format is required when the media URI or file has no extension")
    return media_format


def timestamp_or_none(value: object) -> float | None:
    return value.timestamp() if isinstance(value, datetime) else None


def tool_parameters(job: AwsTranscribeJob, job_data: dict[str, Any]) -> dict[str, Any]:
    parameters = {
        "content_source": job.media_uri,
        "media_was_uploaded": job.media_was_uploaded,
        "output_bucket": job.output_bucket,
        "output_key": job.output_key,
        "language_code": job_data.get("LanguageCode"),
        "identified_language_score": job_data.get("IdentifiedLanguageScore"),
    }
    return {key: value for key, value in parameters.items() if value is not None}


def json_safe(data: Any) -> Any:
    return json.loads(json.dumps(data, default=str))


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the AWS Transcribe CLI parser."""
    parser = argparse.ArgumentParser(description="Transcribe media with AWS Transcribe and print AMPAV ToolOutput YAML.")
    parser.add_argument("media", help="Local media path or s3://bucket/key media URI")
    parser.add_argument("--output-bucket", required=True, help="S3 bucket where AWS writes the transcript JSON")
    parser.add_argument("--output-key", help="S3 key where AWS writes the transcript JSON")
    parser.add_argument("--output-prefix", default="aws_transcribe/output", help="S3 prefix for generated output keys")
    parser.add_argument("--input-bucket", help="S3 bucket for uploading a local media file; defaults to output bucket")
    parser.add_argument("--input-key", help="S3 key for uploading a local media file")
    parser.add_argument("--input-prefix", default="aws_transcribe/input", help="S3 prefix for generated input keys")
    parser.add_argument("--job-name", help="AWS Transcribe job name; generated when omitted")
    parser.add_argument("--job-name-prefix", default="ampav-aws-transcribe", help="Prefix for generated job names")
    parser.add_argument("--media-format", help="AWS media format; inferred from extension when omitted")
    parser.add_argument("--language-code", default="en-US", help="AWS language code when language identification is off")
    parser.add_argument("--identify-language", action="store_true", help="Enable AWS language identification")
    parser.add_argument("--language-option", action="append", default=[], help="Allowed language when identifying language")
    parser.add_argument("--no-speaker-labels", action="store_true", help="Disable AWS speaker diarization")
    parser.add_argument("--max-speaker-labels", type=int, default=10, help="Maximum AWS speaker labels")
    parser.add_argument("--profile", help="AWS profile name for boto3 session")
    parser.add_argument("--region", help="AWS region for boto3 session")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between job status checks")
    parser.add_argument("--timeout", type=int, default=7200, help="Maximum seconds to wait for completion")
    parser.add_argument("--delete-job", action="store_true", help="Delete AWS Transcribe job after fetching output")
    parser.add_argument("--delete-input", action="store_true", help="Delete uploaded input S3 object after fetching output")
    parser.add_argument("--delete-output", action="store_true", help="Delete output S3 object after fetching output")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser


def cli_aws_transcribe() -> None:
    """Console entry point for `ampav_aws_transcribe`."""
    args = build_cli_parser().parse_args()
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG if args.debug else logging.INFO)

    transcription = TranscriptionSettings(
        media_format=args.media_format,
        language_code=args.language_code,
        identify_language=args.identify_language,
        language_options=args.language_option,
        show_speaker_labels=not args.no_speaker_labels,
        max_speaker_labels=args.max_speaker_labels,
    )
    polling = PollingSettings(interval_seconds=args.poll_interval, timeout_seconds=args.timeout)
    try:
        if args.media.startswith("s3://"):
            result = transcribe_uri(
                args.media,
                output_bucket=args.output_bucket,
                output_key=args.output_key,
                output_prefix=args.output_prefix,
                job_name=args.job_name,
                job_name_prefix=args.job_name_prefix,
                transcription=transcription,
                polling=polling,
                delete_job=args.delete_job,
                delete_output=args.delete_output,
                region_name=args.region,
                profile_name=args.profile,
            )
        else:
            result = transcribe_file(
                args.media,
                output_bucket=args.output_bucket,
                input_bucket=args.input_bucket,
                input_key=args.input_key,
                input_prefix=args.input_prefix,
                output_key=args.output_key,
                output_prefix=args.output_prefix,
                job_name=args.job_name,
                job_name_prefix=args.job_name_prefix,
                transcription=transcription,
                polling=polling,
                delete_job=args.delete_job,
                delete_input=args.delete_input,
                delete_output=args.delete_output,
                region_name=args.region,
                profile_name=args.profile,
            )
    except Exception as exc:
        cli_errors = (AwsTranscribeError, BotoCoreError, ClientError, OSError, RuntimeError, TimeoutError, ValueError)
        if not isinstance(exc, cli_errors):
            raise
        logging.error("%s", exc)
        raise SystemExit(1) from exc

    print(result.model_dump_yaml(sort_keys=False))


if __name__ == "__main__":
    cli_aws_transcribe()
