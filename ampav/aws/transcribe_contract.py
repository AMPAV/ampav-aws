"""Lightweight contract checks for AWS Transcribe transcript JSON."""

from __future__ import annotations

from typing import Any

from .errors import AWSTranscriptSchemaError

# BDW: This is an interesting case where pydantic may be really helpful -- if you
# model the aws transcript response (or at least the parts you care about) as a 
# pydantic model then it can validate and massage the data.  Alternately, 
# something like the aws_resource_validator library might be helpful.

# BDW: the need for the model becomes especially important later when you're
# doing the conversion.  The classes below normalize and validate everything so
# you don't have to.

from pydantic import BaseModel, Field
class AWSAlternative(BaseModel):
    confidence: float
    content: str

class AWSPunctuation(BaseModel):
    id: int
    alternatives: list[AWSAlternative]
    type: str

class AWSPronunciation(BaseModel):
    id: int
    start_time: float
    end_time: float
    alternatives: list[AWSAlternative]
    type: str

class AWSAudioSegments(BaseModel):
    id: int
    transcript: str
    start_time: float
    end_time: float
    items: list[int]

class AWSTranscribeResults(BaseModel):
    transcripts: list[dict[str, str]]
    items: list[AWSPunctuation | AWSPronunciation] = Field(discriminator="type")
    audio_segments: list[AWSAudioSegments]

class AWSTranscribeResult(BaseModel):
    jobName: str
    accountId: str
    results: AWSTranscribeResults
    status: str

# BDW: then validate becomes something like this, although you'd probably
# just inline it
def bdw_validate_aws_transcript(aws_transcript: dict) -> AWSTranscribeResult:
    # if it fails it throws, otherwise we get our transcript back
    return AWSTranscribeResult(**aws_transcript)


# BDW: require_mapping doesn't really do anything, so it seems like there's
# missing code or soemthign.  Maybe in a different branch?
def validate_aws_transcript_contract(aws_transcript: object) -> None:
    """Validate only the AWS transcript fields consumed by AMPAV conversion.

    This is intentionally not a full AWS schema validator. It checks the
    portions AMPAV relies on so AWS output drift fails with a precise path.

    :param aws_transcript: Raw AWS Transcribe transcript JSON value.
    :type aws_transcript: object
    :raises AWSTranscriptSchemaError: If a consumed field is missing or has an
        incompatible type.
    """
    root = require_mapping(aws_transcript, "$")
    results = require_mapping(root.get("results"), "$.results")
    validate_transcripts(results)
    validate_items(results)
    validate_audio_segments(results)
    validate_speaker_labels(results)


def validate_transcripts(results: dict[str, Any]) -> None:
    """Validate the transcript text section consumed for Transcript.text.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :raises AWSTranscriptSchemaError: If transcript text is missing or invalid.
    """
    transcripts = require_list(results.get("transcripts"), "$.results.transcripts")
    if not transcripts:
        raise AWSTranscriptSchemaError("$.results.transcripts", "expected at least one transcript entry")
    first = require_mapping(transcripts[0], "$.results.transcripts[0]")
    require_string(first.get("transcript"), "$.results.transcripts[0].transcript")


def validate_items(results: dict[str, Any]) -> None:
    """Validate word and punctuation item shapes consumed for WordSegment data.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :raises AWSTranscriptSchemaError: If consumed item fields are missing or invalid.
    """
    items = require_list(results.get("items"), "$.results.items")
    for index, item_value in enumerate(items):
        path = f"$.results.items[{index}]"
        item = require_mapping(item_value, path)
        item_type = require_string(item.get("type"), f"{path}.type")

        if item_type == "pronunciation":
            validate_alternatives(item, path)
            require_numeric(item.get("start_time"), f"{path}.start_time")
            require_numeric(item.get("end_time"), f"{path}.end_time")
            if "speaker_label" in item:
                require_string(item.get("speaker_label"), f"{path}.speaker_label")
        elif item_type == "punctuation":
            validate_alternatives(item, path)
            if "speaker_label" in item:
                require_string(item.get("speaker_label"), f"{path}.speaker_label")


def validate_alternatives(item: dict[str, Any], path: str) -> None:
    """Validate the first AWS alternative consumed for text and confidence.

    :param item: AWS transcript item object.
    :type item: dict[str, Any]
    :param path: JSON-path-like location for error reporting.
    :type path: str
    :raises AWSTranscriptSchemaError: If alternatives are missing or invalid.
    """
    alternatives = require_list(item.get("alternatives"), f"{path}.alternatives")
    if not alternatives:
        raise AWSTranscriptSchemaError(f"{path}.alternatives", "expected at least one alternative")
    first = require_mapping(alternatives[0], f"{path}.alternatives[0]")
    require_string(first.get("content"), f"{path}.alternatives[0].content")
    if "confidence" in first:
        require_numeric(first.get("confidence"), f"{path}.alternatives[0].confidence")


def validate_audio_segments(results: dict[str, Any]) -> None:
    """Validate AWS audio_segments when AWS includes them.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :raises AWSTranscriptSchemaError: If audio segment fields are invalid.
    """
    if "audio_segments" not in results:
        return
    audio_segments = require_list(results.get("audio_segments"), "$.results.audio_segments")
    for index, segment_value in enumerate(audio_segments):
        path = f"$.results.audio_segments[{index}]"
        segment = require_mapping(segment_value, path)
        require_numeric(segment.get("start_time"), f"{path}.start_time")
        require_numeric(segment.get("end_time"), f"{path}.end_time")
        require_string(segment.get("transcript"), f"{path}.transcript")
        if "speaker_label" in segment:
            require_string(segment.get("speaker_label"), f"{path}.speaker_label")
        if "items" in segment:
            require_list(segment.get("items"), f"{path}.items")


def validate_speaker_labels(results: dict[str, Any]) -> None:
    """Validate AWS speaker label segments when AWS includes them.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :raises AWSTranscriptSchemaError: If speaker label fields are invalid.
    """
    if "speaker_labels" not in results:
        return
    speaker_labels = require_mapping(results.get("speaker_labels"), "$.results.speaker_labels")
    if "segments" not in speaker_labels:
        return
    segments = require_list(speaker_labels.get("segments"), "$.results.speaker_labels.segments")
    for index, segment_value in enumerate(segments):
        path = f"$.results.speaker_labels.segments[{index}]"
        segment = require_mapping(segment_value, path)
        require_numeric(segment.get("start_time"), f"{path}.start_time")
        require_numeric(segment.get("end_time"), f"{path}.end_time")
        require_string(segment.get("speaker_label"), f"{path}.speaker_label")
        if "items" in segment:
            validate_speaker_segment_items(segment.get("items"), f"{path}.items")


def validate_speaker_segment_items(items_value: object, path: str) -> None:
    """Validate speaker segment item references used for paragraph fallback.

    :param items_value: AWS speaker segment ``items`` value.
    :type items_value: object
    :param path: JSON-path-like location for error reporting.
    :type path: str
    :raises AWSTranscriptSchemaError: If speaker item references are invalid.
    """
    items = require_list(items_value, path)
    for index, item_value in enumerate(items):
        item_path = f"{path}[{index}]"
        if isinstance(item_value, int):
            continue
        item = require_mapping(item_value, item_path)
        require_numeric(item.get("start_time"), f"{item_path}.start_time")
        require_numeric(item.get("end_time"), f"{item_path}.end_time")
        if "speaker_label" in item:
            require_string(item.get("speaker_label"), f"{item_path}.speaker_label")


def require_mapping(value: object, path: str) -> dict[str, Any]:
    """Return a mapping value or raise a path-specific schema error.

    :param value: Value to validate.
    :type value: object
    :param path: JSON-path-like location for error reporting.
    :type path: str
    :return: Validated mapping.
    :rtype: dict[str, Any]
    :raises AWSTranscriptSchemaError: If ``value`` is not an object.
    """
    if not isinstance(value, dict):
        raise AWSTranscriptSchemaError(path, "expected object")
    return value


def require_list(value: object, path: str) -> list[Any]:
    """Return a list value or raise a path-specific schema error.

    :param value: Value to validate.
    :type value: object
    :param path: JSON-path-like location for error reporting.
    :type path: str
    :return: Validated list.
    :rtype: list[Any]
    :raises AWSTranscriptSchemaError: If ``value`` is not a list.
    """
    if not isinstance(value, list):
        raise AWSTranscriptSchemaError(path, "expected list")
    return value


def require_string(value: object, path: str) -> str:
    """Return a string value or raise a path-specific schema error.

    :param value: Value to validate.
    :type value: object
    :param path: JSON-path-like location for error reporting.
    :type path: str
    :return: Validated string.
    :rtype: str
    :raises AWSTranscriptSchemaError: If ``value`` is not a string.
    """
    if not isinstance(value, str):
        raise AWSTranscriptSchemaError(path, "expected string")
    return value


def require_numeric(value: object, path: str) -> float:
    """Return a float-compatible value or raise a path-specific schema error.

    :param value: Value to validate.
    :type value: object
    :param path: JSON-path-like location for error reporting.
    :type path: str
    :return: Validated float value.
    :rtype: float
    :raises AWSTranscriptSchemaError: If ``value`` cannot be converted to float.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise AWSTranscriptSchemaError(path, "expected numeric value") from exc
