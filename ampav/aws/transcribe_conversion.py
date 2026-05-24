"""Convert AWS Transcribe transcript JSON into AMPAV transcript schema."""

from __future__ import annotations

import logging
from typing import Any

from ampav.core.schema import ParagraphSegment, Transcript, WordSegment

from .errors import AWSTranscriptSchemaError
from .transcribe_contract import validate_aws_transcript_contract

# BDW: Overall -- you're using isinstance() way too much here.  By the time
# you get to the point you want to convert the AWS transcript to our Transcript
# all of that should have been normalized.

# YF: Agree. We can trust validated data (AWSTranscribeResult validated by Pydantic) at this stage.

def aws_transcript_to_transcript(
    aws_transcript: dict[str, Any],
    media_duration: float | None = None,
) -> Transcript:
    """Convert AWS Transcribe JSON into the AMPAV transcript schema.

    The AWS contract validator runs first so AWS output drift reports the
    missing or incompatible JSON path before conversion begins.

    :param aws_transcript: Raw AWS Transcribe transcript JSON object.
    :type aws_transcript: dict[str, Any]
    :param media_duration: Optional media duration override. When ``None``,
        duration is inferred from AWS segments or word end times.
    :type media_duration: float | None
    :return: Normalized AMPAV Transcript.
    :rtype: Transcript
    :raises AWSTranscriptSchemaError: If the AWS transcript contract is invalid
        or conversion fails.
    """
    validate_aws_transcript_contract(aws_transcript)
    try:
        results = aws_transcript["results"]
        words, words_by_item_id = aws_items_to_words(results.get("items", []))
        transcript = Transcript(
            text=aws_transcript_text(results, words),
            media_duration=media_duration if media_duration is not None else infer_transcript_duration(results, words),
            words=words,
        )
        transcript.paragraphs = aws_results_to_paragraphs(results, words, words_by_item_id)
        return transcript
    except AWSTranscriptSchemaError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise AWSTranscriptSchemaError("$", f"failed to convert AWS transcript: {exc}") from exc


def aws_transcript_text(results: dict[str, Any], words: list[WordSegment]) -> str:
    """Return transcript text from AWS transcripts or reconstruct it from words.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :param words: Converted word segments used as a fallback text source.
    :type words: list[WordSegment]
    :return: Transcript text.
    :rtype: str
    """
    transcripts = results.get("transcripts")
    if isinstance(transcripts, list) and transcripts:
        first = transcripts[0]
        if isinstance(first, dict) and isinstance(first.get("transcript"), str):
            return first["transcript"]
    return words_to_text(words)

# BDW: Here's a prime example where pre-normalizing the aws transcribe json
# can be used to to your advantage to make this smaller and easier to understand.
# this one function effectively replaces aws_items_to_words, 
# aws_pronunciation_item_to_word, attach_punctuation, and first_alternative.
# YF: Agree and adopt
from .transcribe_contract import AWSTranscribeResult
def bdw_aws_items_to_words(aws: AWSTranscribeResult) -> list[WordSegment]:
    res: list[WordSegment] = []
    for w in aws.results.items:
        if w.type == "punctuation":
            res[-1].suffix = w.alternatives[0].content
        else:
            res.append(WordSegment(start_time=w.start_time,
                                   end_time=w.end_time,
                                   confidence=w.alternatives[0].confidence,
                                   word=w.alternatives[0].content))
    return res

def aws_items_to_words(items: object) -> tuple[list[WordSegment], dict[int, WordSegment]]:
    """Convert AWS item entries into AMPAV word segments and an item-id map.

    :param items: AWS ``results.items`` value.
    :type items: object
    :return: Word segments and a map from AWS integer item id to word segment.
    :rtype: tuple[list[WordSegment], dict[int, WordSegment]]
    :raises ValueError: If ``items`` is not a list or contains invalid entries.
    """
    if not isinstance(items, list):
        raise ValueError("AWS transcript results.items must be a list when present")

    words: list[WordSegment] = []
    words_by_item_id: dict[int, WordSegment] = {}
    previous_word: WordSegment | None = None

    for item in items:
        if not isinstance(item, dict):
            raise ValueError("AWS transcript item entries must be objects")

        item_type = item.get("type")
        if item_type == "pronunciation":
            word = aws_pronunciation_item_to_word(item)
            words.append(word)
            previous_word = word
            item_id = item.get("id")
            if isinstance(item_id, int):
                words_by_item_id[item_id] = word
        elif item_type == "punctuation":
            if previous_word is not None:
                attach_punctuation(previous_word, item)
        else:
            logging.warning("Skipping unsupported AWS transcript item type: %s", item_type)

    return words, words_by_item_id

def aws_pronunciation_item_to_word(item: dict[str, Any]) -> WordSegment:
    """Convert a single AWS pronunciation item into a WordSegment.

    :param item: AWS pronunciation item object.
    :type item: dict[str, Any]
    :return: Converted AMPAV word segment.
    :rtype: WordSegment
    :raises ValueError: If the item is missing required alternative content.
    """
    alternative = first_alternative(item)
    content = alternative.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("AWS pronunciation item is missing alternative content")

    confidence = optional_float(alternative.get("confidence"))
    return WordSegment(
        word=content,
        start_time=optional_float(item.get("start_time")),
        end_time=optional_float(item.get("end_time")),
        speaker=item.get("speaker_label") if isinstance(item.get("speaker_label"), str) else None,
        tool_specific={
            "aws_item_id": item.get("id"),
            "aws_type": item.get("type"),
            "confidence": confidence,
            "alternatives": item.get("alternatives", []),
        },
    )


def attach_punctuation(word: WordSegment, item: dict[str, Any]) -> None:
    """Attach an AWS punctuation item to the previous WordSegment suffix.

    :param word: Word segment that receives the punctuation suffix.
    :type word: WordSegment
    :param item: AWS punctuation item object.
    :type item: dict[str, Any]
    :raises ValueError: If the item is missing required punctuation content.
    """
    alternative = first_alternative(item)
    content = alternative.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("AWS punctuation item is missing alternative content")

    word.suffix = f"{word.suffix or ''}{content}"
    if word.tool_specific is None:
        word.tool_specific = {}
    word.tool_specific.setdefault("aws_punctuation", []).append(
        {
            "aws_item_id": item.get("id"),
            "content": content,
            "confidence": optional_float(alternative.get("confidence")),
            "alternatives": item.get("alternatives", []),
        }
    )


def first_alternative(item: dict[str, Any]) -> dict[str, Any]:
    """Return the first AWS alternative entry for a transcript item.

    :param item: AWS transcript item object.
    :type item: dict[str, Any]
    :return: First alternative entry.
    :rtype: dict[str, Any]
    :raises ValueError: If no usable alternative exists.
    """
    # BDW: It's my understanding that there is always one alternative, so you 
    # could just write this (inline) as:
    # item['alternatives'][0]
    alternatives = item.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives or not isinstance(alternatives[0], dict):
        raise ValueError("AWS transcript item is missing alternatives")
    return alternatives[0]

# BDW: if you have the whole aws transcript you don't need words_by_item_id
# because they're just they're literally the identity of their position in
# the array (I'm 99.9% sure of this after looking at the code), so you can 
# just do an array lookup.
def aws_results_to_paragraphs(
    results: dict[str, Any],
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[ParagraphSegment]:
    """Build paragraphs from AWS audio segments, speaker labels, or word timing.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :param words: Converted word segments.
    :type words: list[WordSegment]
    :param words_by_item_id: Map from AWS integer item id to word segment.
    :type words_by_item_id: dict[int, WordSegment]
    :return: Paragraph segments.
    :rtype: list[ParagraphSegment]
    """
    audio_segments = results.get("audio_segments")
    if isinstance(audio_segments, list) and audio_segments:
        return aws_audio_segments_to_paragraphs(audio_segments)

    speaker_labels = results.get("speaker_labels")
    if isinstance(speaker_labels, dict):
        speaker_segments = speaker_labels.get("segments")
        if isinstance(speaker_segments, list) and speaker_segments:
            return aws_speaker_segments_to_paragraphs(speaker_segments, words, words_by_item_id)

    if not words:
        return []
    transcript = Transcript(words=words)
    transcript.reformat_paragraphs()
    return transcript.paragraphs


def aws_audio_segments_to_paragraphs(audio_segments: list[object]) -> list[ParagraphSegment]:
    """Convert AWS audio_segments entries into AMPAV paragraph segments.

    :param audio_segments: AWS ``results.audio_segments`` entries.
    :type audio_segments: list[object]
    :return: Paragraph segments.
    :rtype: list[ParagraphSegment]
    :raises ValueError: If an audio segment is not an object.
    """
    paragraphs: list[ParagraphSegment] = []
    for segment in audio_segments:
        if not isinstance(segment, dict):
            raise ValueError("AWS transcript audio_segments entries must be objects")
        paragraphs.append(
            ParagraphSegment(
                start_time=optional_float(segment.get("start_time")),
                end_time=optional_float(segment.get("end_time")),
                speaker=segment.get("speaker_label") if isinstance(segment.get("speaker_label"), str) else None,
                text=segment.get("transcript") if isinstance(segment.get("transcript"), str) else "",
                tool_specific={
                    "aws_segment_type": "audio_segment",
                    "aws_segment_id": segment.get("id"),
                    "aws_item_ids": segment.get("items", []),
                },
            )
        )
    return paragraphs


def aws_speaker_segments_to_paragraphs(
    speaker_segments: list[object],
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[ParagraphSegment]:
    """Convert AWS speaker label segments into AMPAV paragraph segments.

    :param speaker_segments: AWS ``speaker_labels.segments`` entries.
    :type speaker_segments: list[object]
    :param words: Converted word segments.
    :type words: list[WordSegment]
    :param words_by_item_id: Map from AWS integer item id to word segment.
    :type words_by_item_id: dict[int, WordSegment]
    :return: Paragraph segments.
    :rtype: list[ParagraphSegment]
    :raises ValueError: If a speaker segment is not an object.
    """
    paragraphs: list[ParagraphSegment] = []
    for segment in speaker_segments:
        if not isinstance(segment, dict):
            raise ValueError("AWS transcript speaker label segments must be objects")

        segment_words = speaker_segment_words(segment, words, words_by_item_id)
        segment_start_time = optional_float(segment.get("start_time"))
        segment_end_time = optional_float(segment.get("end_time"))
        if segment_words:
            text = words_to_text(segment_words)
            start_time = segment_start_time if segment_start_time is not None else segment_words[0].start_time
            end_time = segment_end_time if segment_end_time is not None else segment_words[-1].end_time
        else:
            text = ""
            start_time = segment_start_time
            end_time = segment_end_time

        paragraphs.append(
            ParagraphSegment(
                start_time=start_time,
                end_time=end_time,
                speaker=segment.get("speaker_label") if isinstance(segment.get("speaker_label"), str) else None,
                text=text,
                tool_specific={
                    "aws_segment_type": "speaker_label",
                    "aws_items": segment.get("items", []),
                },
            )
        )
    return paragraphs


def speaker_segment_words(
    segment: dict[str, Any],
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[WordSegment]:
    """Find words that belong to an AWS speaker label segment.

    :param segment: AWS speaker label segment object.
    :type segment: dict[str, Any]
    :param words: Converted word segments.
    :type words: list[WordSegment]
    :param words_by_item_id: Map from AWS integer item id to word segment.
    :type words_by_item_id: dict[int, WordSegment]
    :return: Words matched to the speaker segment.
    :rtype: list[WordSegment]
    """
    items = segment.get("items")
    if isinstance(items, list):
        matched_by_time: list[WordSegment] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_word = match_word_by_time(
                words=words,
                start_time=optional_float(item.get("start_time")),
                end_time=optional_float(item.get("end_time")),
                speaker=item.get("speaker_label") if isinstance(item.get("speaker_label"), str) else None,
            )
            if item_word is not None:
                matched_by_time.append(item_word)
        if matched_by_time:
            return dedupe_words(matched_by_time)

    segment_item_ids = segment.get("items")
    if isinstance(segment_item_ids, list):
        matched_by_id = [
            words_by_item_id[item_id]
            for item_id in segment_item_ids
            if isinstance(item_id, int) and item_id in words_by_item_id
        ]
        if matched_by_id:
            return dedupe_words(matched_by_id)

    start_time = optional_float(segment.get("start_time"))
    end_time = optional_float(segment.get("end_time"))
    speaker = segment.get("speaker_label") if isinstance(segment.get("speaker_label"), str) else None
    return [
        word
        for word in words
        if word_in_time_range(word, start_time, end_time) and (speaker is None or word.speaker == speaker)
    ]


def match_word_by_time(
    words: list[WordSegment],
    start_time: float | None,
    end_time: float | None,
    speaker: str | None,
) -> WordSegment | None:
    """Return the first word matching a speaker segment item time range.

    :param words: Candidate word segments.
    :type words: list[WordSegment]
    :param start_time: Required start time, or ``None``.
    :type start_time: float | None
    :param end_time: Required end time, or ``None``.
    :type end_time: float | None
    :param speaker: Required speaker label, or ``None`` to ignore speaker.
    :type speaker: str | None
    :return: Matching word segment or ``None``.
    :rtype: WordSegment | None
    """
    for word in words:
        if word.start_time == start_time and word.end_time == end_time and (speaker is None or word.speaker == speaker):
            return word
    return None


def word_in_time_range(word: WordSegment, start_time: float | None, end_time: float | None) -> bool:
    """Return whether a word is wholly contained in a time range.

    :param word: Word segment to test.
    :type word: WordSegment
    :param start_time: Inclusive start of the time range.
    :type start_time: float | None
    :param end_time: Inclusive end of the time range.
    :type end_time: float | None
    :return: ``True`` if the word is fully inside the range.
    :rtype: bool
    """
    if start_time is None or end_time is None or word.start_time is None or word.end_time is None:
        return False
    # BDW: python supports this structure:
    # return start_time <= word.start_time <= end_time
    return start_time <= word.start_time and word.end_time <= end_time


# BDW: I think I've used the id function twice it the years I've used python.
# equality should suffice.
def dedupe_words(words: list[WordSegment]) -> list[WordSegment]:
    """Deduplicate word object references while preserving order.

    :param words: Word segments that may contain duplicate object references.
    :type words: list[WordSegment]
    :return: Deduplicated word segments in original order.
    :rtype: list[WordSegment]
    """
    seen: set[int] = set()
    result: list[WordSegment] = []
    for word in words:
        identity = id(word)
        if identity not in seen:
            seen.add(identity)
            result.append(word)
    return result


def words_to_text(words: list[WordSegment]) -> str:
    """Render AMPAV word segments into plain transcript text.

    :param words: Word segments to render.
    :type words: list[WordSegment]
    :return: Plain transcript text.
    :rtype: str
    """
    return " ".join(word.to_str() for word in words)


def infer_transcript_duration(results: dict[str, Any], words: list[WordSegment]) -> float | None:
    """Infer media duration from AWS segment or word end times.

    :param results: AWS ``results`` object.
    :type results: dict[str, Any]
    :param words: Converted word segments.
    :type words: list[WordSegment]
    :return: Maximum observed end time, or ``None`` when unavailable.
    :rtype: float | None
    """
    candidates: list[float] = []
    audio_segments = results.get("audio_segments")
    if isinstance(audio_segments, list):
        candidates.extend(
            time_value
            for segment in audio_segments
            if isinstance(segment, dict)
            for time_value in [optional_float(segment.get("end_time"))]
            if time_value is not None
        )
    candidates.extend(word.end_time for word in words if word.end_time is not None)
    return max(candidates) if candidates else None


def optional_float(value: object) -> float | None:
    """Convert an optional AWS numeric string into a float.

    :param value: AWS numeric string, number, or ``None``.
    :type value: object
    :return: Converted float or ``None``.
    :rtype: float | None
    :raises ValueError: If ``value`` is not ``None`` and cannot be converted.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric AWS transcript value, got {value!r}") from exc
