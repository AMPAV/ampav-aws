from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from ampav.aws.transcribe import aws_transcript_to_transcript


FIXTURE = Path(__file__).parent / "fixtures" / "aws_transcript_opendoor.json"


class AWSTranscribeConversionTest(unittest.TestCase):
    def load_fixture(self) -> dict:
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_converts_observed_audio_segments_schema(self) -> None:
        transcript = aws_transcript_to_transcript(self.load_fixture())

        self.assertEqual(transcript.text, "Please open the door.")
        self.assertEqual(transcript.media_duration, 1.649)
        self.assertEqual([word.to_str() for word in transcript.words], ["Please", "open", "the", "door."])
        self.assertEqual(len(transcript.words), 4)
        self.assertEqual(transcript.words[0].speaker, "spk_0")
        self.assertEqual(transcript.words[0].tool_specific["confidence"], 0.999)
        self.assertEqual(transcript.words[-1].suffix, ".")
        self.assertEqual(transcript.words[-1].tool_specific["aws_punctuation"][0]["content"], ".")

        self.assertEqual(len(transcript.paragraphs), 1)
        paragraph = transcript.paragraphs[0]
        self.assertEqual(paragraph.text, "Please open the door.")
        self.assertEqual(paragraph.speaker, "spk_0")
        self.assertEqual(paragraph.start_time, 0.0)
        self.assertEqual(paragraph.end_time, 1.649)
        self.assertEqual(paragraph.tool_specific["aws_segment_type"], "audio_segment")

    def test_falls_back_to_speaker_label_segments(self) -> None:
        data = copy.deepcopy(self.load_fixture())
        del data["results"]["audio_segments"]

        transcript = aws_transcript_to_transcript(data)

        self.assertEqual(len(transcript.paragraphs), 1)
        paragraph = transcript.paragraphs[0]
        self.assertEqual(paragraph.text, "Please open the door.")
        self.assertEqual(paragraph.speaker, "spk_0")
        self.assertEqual(paragraph.start_time, 0.0)
        self.assertEqual(paragraph.end_time, 1.649)
        self.assertEqual(paragraph.tool_specific["aws_segment_type"], "speaker_label")


if __name__ == "__main__":
    unittest.main()
