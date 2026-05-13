from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from ampav.aws.errors import AWSTranscriptSchemaError
from ampav.aws.transcribe_contract import validate_aws_transcript_contract
from ampav.aws.transcribe_conversion import aws_transcript_to_transcript


FIXTURE = Path(__file__).parent / "fixtures" / "aws_transcript_opendoor.json"


class AWSTranscribeContractTest(unittest.TestCase):
    def load_fixture(self) -> dict:
        """Load the sanitized AWS transcript fixture."""
        return json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_accepts_observed_aws_transcript_contract(self) -> None:
        validate_aws_transcript_contract(self.load_fixture())

    def test_missing_results_reports_schema_path(self) -> None:
        data = self.load_fixture()
        del data["results"]

        with self.assertRaises(AWSTranscriptSchemaError) as caught:
            validate_aws_transcript_contract(data)

        self.assertEqual(caught.exception.path, "$.results")
        self.assertIn("expected object", str(caught.exception))

    def test_missing_transcript_text_reports_schema_path(self) -> None:
        data = self.load_fixture()
        del data["results"]["transcripts"][0]["transcript"]

        with self.assertRaises(AWSTranscriptSchemaError) as caught:
            validate_aws_transcript_contract(data)

        self.assertEqual(caught.exception.path, "$.results.transcripts[0].transcript")

    def test_invalid_word_timing_reports_schema_path(self) -> None:
        data = self.load_fixture()
        data["results"]["items"][0]["start_time"] = "not-a-time"

        with self.assertRaises(AWSTranscriptSchemaError) as caught:
            validate_aws_transcript_contract(data)

        self.assertEqual(caught.exception.path, "$.results.items[0].start_time")

    def test_conversion_runs_contract_validation_first(self) -> None:
        data = self.load_fixture()
        data["results"]["items"][0]["alternatives"] = []

        with self.assertRaises(AWSTranscriptSchemaError) as caught:
            aws_transcript_to_transcript(data)

        self.assertEqual(caught.exception.path, "$.results.items[0].alternatives")

    def test_accepts_speaker_labels_fallback_without_audio_segments(self) -> None:
        data = copy.deepcopy(self.load_fixture())
        del data["results"]["audio_segments"]

        validate_aws_transcript_contract(data)


if __name__ == "__main__":
    unittest.main()
