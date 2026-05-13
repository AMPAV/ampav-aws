from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from ampav.aws.transcribe import load_config, transcribe_file_with_config
from ampav.aws.transcribe_contract import validate_aws_transcript_contract


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_AUDIO = ROOT / "tests" / "fixtures" / "OpenDoor.wav"


@unittest.skipUnless(
    os.environ.get("AMPAV_AWS_TRANSCRIBE_LIVE_TEST") == "1"
    and os.environ.get("AMPAV_AWS_TRANSCRIBE_CONFIG"),
    "set AMPAV_AWS_TRANSCRIBE_LIVE_TEST=1 and AMPAV_AWS_TRANSCRIBE_CONFIG to run live AWS test",
)
class AWSTranscribeLiveTest(unittest.TestCase):
    def test_live_opendoor_transcription_matches_contract(self) -> None:
        """Submit OpenDoor.wav to AWS and validate the returned transcript shape."""
        config_path = Path(os.environ["AMPAV_AWS_TRANSCRIBE_CONFIG"])
        config = load_config(config_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.paths.runs_dir = Path(tmpdir) / "runs"
            output = transcribe_file_with_config(SAMPLE_AUDIO, config=config, config_path=config_path)

            raw_transcript_path = Path(output.parameters["raw_transcript_json"])
            raw_transcript = json.loads(raw_transcript_path.read_text(encoding="utf-8"))
            validate_aws_transcript_contract(raw_transcript)

        self.assertIsNotNone(output.output)
        self.assertIn("open", output.output.text.lower())
        self.assertGreaterEqual(len(output.output.words), 3)
        self.assertGreaterEqual(len(output.output.paragraphs), 1)


if __name__ == "__main__":
    unittest.main()
