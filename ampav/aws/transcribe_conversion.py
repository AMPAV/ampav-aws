from __future__ import annotations

import logging
from typing import Any

from ampav.core.schema import ParagraphSegment, Transcript, WordSegment


def aws_transcript_to_transcript(
    aws_transcript: dict[str, Any],
    media_duration: float | None = None,
) -> Transcript:
    results = aws_transcript.get("results")
    if not isinstance(results, dict):
        raise ValueError("AWS transcript JSON must contain a results object")

    words, words_by_item_id = aws_items_to_words(results.get("items", []))
    transcript = Transcript(
        text=aws_transcript_text(results, words),
        media_duration=media_duration if media_duration is not None else infer_transcript_duration(results, words),
        words=words,
    )
    transcript.paragraphs = aws_results_to_paragraphs(results, words, words_by_item_id)
    return transcript


def aws_transcript_text(results: dict[str, Any], words: list[WordSegment]) -> str:
    transcripts = results.get("transcripts")
    if isinstance(transcripts, list) and transcripts:
        first = transcripts[0]
        if isinstance(first, dict) and isinstance(first.get("transcript"), str):
            return first["transcript"]
    return words_to_text(words)


def aws_items_to_words(items: object) -> tuple[list[WordSegment], dict[int, WordSegment]]:
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
    alternatives = item.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives or not isinstance(alternatives[0], dict):
        raise ValueError("AWS transcript item is missing alternatives")
    return alternatives[0]


def aws_results_to_paragraphs(
    results: dict[str, Any],
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[ParagraphSegment]:
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
    for word in words:
        if word.start_time == start_time and word.end_time == end_time and (speaker is None or word.speaker == speaker):
            return word
    return None


def word_in_time_range(word: WordSegment, start_time: float | None, end_time: float | None) -> bool:
    if start_time is None or end_time is None or word.start_time is None or word.end_time is None:
        return False
    return start_time <= word.start_time and word.end_time <= end_time


def dedupe_words(words: list[WordSegment]) -> list[WordSegment]:
    seen: set[int] = set()
    result: list[WordSegment] = []
    for word in words:
        identity = id(word)
        if identity not in seen:
            seen.add(identity)
            result.append(word)
    return result


def words_to_text(words: list[WordSegment]) -> str:
    return " ".join(word.to_str() for word in words)


def infer_transcript_duration(results: dict[str, Any], words: list[WordSegment]) -> float | None:
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
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric AWS transcript value, got {value!r}") from exc
