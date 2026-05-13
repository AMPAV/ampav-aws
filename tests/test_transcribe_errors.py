from __future__ import annotations

import unittest

from ampav.aws.errors import AWSTranscribeJobError
from ampav.aws.transcribe import PollingSettings, poll_until_complete


class FailedTranscribeService:
    def get_job(self, job_name: str) -> dict:
        """Return a failed AWS Transcribe job response."""
        return {
            "TranscriptionJob": {
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "test failure",
            }
        }


class AWSTranscribeErrorTest(unittest.TestCase):
    def test_failed_job_raises_typed_error(self) -> None:
        with self.assertRaises(AWSTranscribeJobError) as caught:
            poll_until_complete(
                service=FailedTranscribeService(),
                job_name="test-job",
                polling=PollingSettings(interval_seconds=1, timeout_seconds=1),
                status_history_path=None,
            )

        self.assertEqual(caught.exception.job_name, "test-job")
        self.assertIn("test failure", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
