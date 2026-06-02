import os
import unittest
from pathlib import Path
from typing import Any

import yaml

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings
from ampav.aws.transcribe_contract import validate_aws_transcript_contract


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_AUDIO = ROOT / "tests" / "fixtures" / "OpenDoor.wav"


@unittest.skipUnless(
    os.environ.get("AMPAV_AWS_TRANSCRIBE_LIVE_TEST") == "1"
    and os.environ.get("AMPAV_AWS_TRANSCRIBE_CONFIG"),
    "set AMPAV_AWS_TRANSCRIBE_LIVE_TEST=1 and AMPAV_AWS_TRANSCRIBE_CONFIG to run live AWS test",
)
class AwsTranscribeLiveTest(unittest.TestCase):
    def test_live_opendoor_transcription_matches_contract(self) -> None:
        config = load_yaml(Path(os.environ["AMPAV_AWS_TRANSCRIBE_CONFIG"]))
        aws_config = config.get("aws", {})
        s3_config = config.get("s3", {})
        transcription_config = config.get("transcription", {})
        polling_config = config.get("polling", {})

        output_bucket = s3_config.get("output_bucket") or s3_config.get("bucket")
        self.assertIsNotNone(output_bucket)

        client = AwsTranscribe(
            region_name=aws_config.get("region"),
            profile_name=aws_config.get("profile_name"),
            polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
            timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
        )
        output = client.process(
            SAMPLE_AUDIO,
            output_bucket=output_bucket,
            input_bucket=s3_config.get("input_bucket") or s3_config.get("bucket"),
            input_prefix=s3_config.get("input_prefix", "aws_transcribe/input"),
            output_prefix=s3_config.get("output_prefix", "aws_transcribe/output"),
            job_name_prefix=transcription_config.pop("job_name_prefix", "ampav-aws-transcribe"),
            transcription=TranscriptionSettings(**transcription_config),
        )

        validate_aws_transcript_contract(output.tool_private["raw_transcript"])
        self.assertIsNotNone(output.output)
        self.assertIn("open", output.output.text.lower())
        self.assertGreaterEqual(len(output.output.words), 3)
        self.assertGreaterEqual(len(output.output.paragraphs), 1)


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    unittest.main()
