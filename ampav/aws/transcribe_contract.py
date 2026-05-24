"""Pydantic models for the AWS Transcribe fields consumed by AMPAV."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import AwsTranscriptSchemaError


class AwsBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class AwsAlternative(AwsBaseModel):
    content: str
    confidence: float | None = None


class AwsPronunciationItem(AwsBaseModel):
    type: Literal["pronunciation"]
    alternatives: list[AwsAlternative] = Field(min_length=1)
    start_time: float
    end_time: float
    id: int | None = None
    speaker_label: str | None = None
    language_code: str | None = None


class AwsPunctuationItem(AwsBaseModel):
    type: Literal["punctuation"]
    alternatives: list[AwsAlternative] = Field(min_length=1)
    id: int | None = None
    speaker_label: str | None = None
    language_code: str | None = None


AwsTranscriptItem = Annotated[
    AwsPronunciationItem | AwsPunctuationItem,
    Field(discriminator="type"),
]


class AwsTranscriptText(AwsBaseModel):
    transcript: str


class AwsAudioSegment(AwsBaseModel):
    transcript: str
    start_time: float
    end_time: float
    id: int | None = None
    speaker_label: str | None = None
    items: list[int] = Field(default_factory=list)
    language_code: str | None = None


class AwsSpeakerSegmentItem(AwsBaseModel):
    start_time: float
    end_time: float
    speaker_label: str | None = None


class AwsSpeakerSegment(AwsBaseModel):
    start_time: float
    end_time: float
    speaker_label: str
    items: list[int | AwsSpeakerSegmentItem] = Field(default_factory=list)


class AwsSpeakerLabels(AwsBaseModel):
    segments: list[AwsSpeakerSegment] = Field(default_factory=list)
    channel_label: str | None = None
    speakers: int | None = None


class AwsTranscribeResults(AwsBaseModel):
    transcripts: list[AwsTranscriptText] = Field(min_length=1)
    items: list[AwsTranscriptItem] = Field(default_factory=list)
    audio_segments: list[AwsAudioSegment] | None = None
    speaker_labels: AwsSpeakerLabels | None = None


class AwsTranscribeResult(AwsBaseModel):
    results: AwsTranscribeResults
    job_name: str | None = Field(default=None, alias="jobName")
    account_id: str | None = Field(default=None, alias="accountId")
    status: str | None = None


def validate_aws_transcript_contract(aws_transcript: object) -> AwsTranscribeResult:
    """Validate the AWS transcript fields consumed by AMPAV conversion."""
    if not isinstance(aws_transcript, dict):
        raise AwsTranscriptSchemaError("$", "expected object")
    try:
        return AwsTranscribeResult.model_validate(aws_transcript)
    except ValidationError as exc:
        error = exc.errors()[0]
        raise AwsTranscriptSchemaError(pydantic_loc_to_path(error["loc"]), str(error["msg"])) from exc


def pydantic_loc_to_path(loc: tuple[Any, ...]) -> str:
    parts: list[str] = ["$"]
    for item in loc:
        if item in {"pronunciation", "punctuation"}:
            continue
        if isinstance(item, int):
            parts[-1] += f"[{item}]"
        else:
            parts.append(str(item))
    return ".".join(parts)
