"""Convert AWS Transcribe JSON into AMPAV transcript schema."""

from typing import Any

from ampav.core.schema import ParagraphSegment, Transcript, WordSegment

from .errors import AwsTranscriptSchemaError
from .transcribe_contract import (
    AwsAudioSegment,
    AwsPunctuationItem,
    AwsSpeakerSegment,
    AwsSpeakerSegmentItem,
    AwsTranscriptItem,
    AwsTranscribeResult,
    validate_aws_transcript_contract,
)


def aws_transcript_to_transcript(
    aws_transcript: dict[str, Any] | AwsTranscribeResult,
    media_duration: float | None = None,
) -> Transcript:
    """Convert raw or validated AWS Transcribe output into an AMPAV transcript."""
    aws = (
        aws_transcript
        if isinstance(aws_transcript, AwsTranscribeResult)
        else validate_aws_transcript_contract(aws_transcript)
    )

    try:
        words, words_by_item_id = aws_items_to_words(aws.results.items)
        transcript = Transcript(
            text=aws.results.transcripts[0].transcript if aws.results.transcripts else words_to_text(words),
            media_duration=media_duration if media_duration is not None else infer_transcript_duration(aws, words),
            words=words,
            languages=infer_languages(words),
        )
        transcript.paragraphs = aws_results_to_paragraphs(aws, words, words_by_item_id)
        return transcript
    except AwsTranscriptSchemaError:
        raise
    except (IndexError, TypeError, ValueError) as exc:
        raise AwsTranscriptSchemaError("$", f"failed to convert AWS transcript: {exc}") from exc


def aws_items_to_words(items: list[AwsTranscriptItem]) -> tuple[list[WordSegment], dict[int, WordSegment]]:
    """Convert AWS pronunciation/punctuation items into AMPAV word segments.

    Punctuation items are attached to the preceding word as suffix text and kept
    in `tool_private` for traceability. Pronunciation confidence is promoted to
    the first-class AMPAV `WordSegment.confidence` field.
    """
    words: list[WordSegment] = []
    words_by_item_id: dict[int, WordSegment] = {}
    previous_word: WordSegment | None = None

    for item in items:
        alternative = item.alternatives[0]
        if isinstance(item, AwsPunctuationItem):
            if previous_word is not None:
                previous_word.suffix = f"{previous_word.suffix or ''}{alternative.content}"
                previous_word.tool_private = previous_word.tool_private or {}
                previous_word.tool_private.setdefault("aws_punctuation", []).append(
                    {
                        "aws_item_id": item.id,
                        "content": alternative.content,
                        "confidence": alternative.confidence,
                        "alternatives": [alt.model_dump(mode="json") for alt in item.alternatives],
                    }
                )
            continue

        word = WordSegment(
            word=alternative.content,
            start_time=item.start_time,
            end_time=item.end_time,
            speaker=item.speaker_label,
            language=item.language_code,
            confidence=alternative.confidence,
            tool_private={
                "aws_item_id": item.id,
                "aws_type": item.type,
                "alternatives": [alt.model_dump(mode="json") for alt in item.alternatives],
            },
        )
        words.append(word)
        previous_word = word
        if item.id is not None:
            words_by_item_id[item.id] = word

    return words, words_by_item_id


def aws_results_to_paragraphs(
    aws: AwsTranscribeResult,
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[ParagraphSegment]:
    """Build AMPAV paragraphs using the best AWS structure available.

    AWS `audio_segments` already contain transcript text, timing, and speaker
    labels, so they are preferred. Older/alternate outputs may only contain
    `speaker_labels`; when neither structure is present, AMPAV falls back to
    paragraph formatting from words.
    """
    if aws.results.audio_segments:
        return aws_audio_segments_to_paragraphs(aws.results.audio_segments)

    if aws.results.speaker_labels and aws.results.speaker_labels.segments:
        return aws_speaker_segments_to_paragraphs(aws.results.speaker_labels.segments, words, words_by_item_id)

    if not words:
        return []
    transcript = Transcript(words=words)
    transcript.reformat_paragraphs()
    return transcript.paragraphs


def aws_audio_segments_to_paragraphs(audio_segments: list[AwsAudioSegment]) -> list[ParagraphSegment]:
    """Convert AWS audio segments directly into AMPAV paragraphs."""
    return [
        ParagraphSegment(
            start_time=segment.start_time,
            end_time=segment.end_time,
            speaker=segment.speaker_label,
            language=segment.language_code,
            text=segment.transcript,
            tool_private={
                "aws_segment_type": "audio_segment",
                "aws_segment_id": segment.id,
                "aws_item_ids": segment.items,
            },
        )
        for segment in audio_segments
    ]


def aws_speaker_segments_to_paragraphs(
    speaker_segments: list[AwsSpeakerSegment],
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[ParagraphSegment]:
    """Convert AWS speaker-label segments into AMPAV paragraphs.

    Speaker-label segments may reference words by detailed timing entries or by
    item IDs depending on the AWS output shape. The conversion preserves the raw
    segment item references in `tool_private`.
    """
    paragraphs: list[ParagraphSegment] = []
    for segment in speaker_segments:
        segment_words = speaker_segment_words(segment, words, words_by_item_id)
        text = words_to_text(segment_words) if segment_words else ""
        start_time = segment.start_time if segment.start_time is not None else segment_words[0].start_time
        end_time = segment.end_time if segment.end_time is not None else segment_words[-1].end_time

        paragraphs.append(
            ParagraphSegment(
                start_time=start_time,
                end_time=end_time,
                speaker=segment.speaker_label,
                text=text,
                tool_private={
                    "aws_segment_type": "speaker_label",
                    "aws_items": [
                        item if isinstance(item, int) else item.model_dump(mode="json") for item in segment.items
                    ],
                },
            )
        )
    return paragraphs


def speaker_segment_words(
    segment: AwsSpeakerSegment,
    words: list[WordSegment],
    words_by_item_id: dict[int, WordSegment],
) -> list[WordSegment]:
    """Find words belonging to a speaker segment.

    Matching prefers explicit timing entries, then integer item IDs, then a
    final time-range/speaker fallback. This keeps diarization useful across the
    AWS shapes observed in tests and live runs.
    """
    matched_by_time = [
        word
        for item in segment.items
        if isinstance(item, AwsSpeakerSegmentItem)
        for word in [match_word_by_time(words, item.start_time, item.end_time, item.speaker_label)]
        if word is not None
    ]
    if matched_by_time:
        return dedupe_words(matched_by_time)

    matched_by_id = [words_by_item_id[item] for item in segment.items if isinstance(item, int) and item in words_by_item_id]
    if matched_by_id:
        return dedupe_words(matched_by_id)

    return [
        word
        for word in words
        if word_in_time_range(word, segment.start_time, segment.end_time)
        and (segment.speaker_label is None or word.speaker == segment.speaker_label)
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
    return start_time <= word.start_time <= end_time and start_time <= word.end_time <= end_time


def dedupe_words(words: list[WordSegment]) -> list[WordSegment]:
    result: list[WordSegment] = []
    for word in words:
        if word not in result:
            result.append(word)
    return result


def words_to_text(words: list[WordSegment]) -> str:
    return " ".join(word.to_str() for word in words)


def infer_transcript_duration(aws: AwsTranscribeResult, words: list[WordSegment]) -> float | None:
    candidates: list[float] = []
    if aws.results.audio_segments:
        candidates.extend(segment.end_time for segment in aws.results.audio_segments)
    candidates.extend(word.end_time for word in words if word.end_time is not None)
    return max(candidates) if candidates else None


def infer_languages(words: list[WordSegment]) -> list[str] | None:
    languages = sorted({word.language for word in words if word.language})
    return languages or None
