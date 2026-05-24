import unittest

from ampav.aws.errors import AwsTranscribeError
from ampav.aws.transcribe import AwsTranscribe, PollingSettings


class FailedTranscribeClient:
    def get_transcription_job(self, TranscriptionJobName: str) -> dict:
        return {
            "TranscriptionJob": {
                "TranscriptionJobName": TranscriptionJobName,
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "test failure",
            }
        }


class AwsTranscribeErrorTest(unittest.TestCase):
    def test_failed_job_raises_typed_error(self) -> None:
        client = AwsTranscribe(transcribe_client=FailedTranscribeClient(), s3_client=object())

        with self.assertRaises(AwsTranscribeError) as caught:
            client.wait("test-job", polling=PollingSettings(interval_seconds=1, timeout_seconds=1))

        self.assertEqual(caught.exception.job_name, "test-job")
        self.assertIn("test failure", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
