import unittest

from ampav.aws.errors import AwsTranscribeError
from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings


class FailedTranscribeClient:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.deleted: list[str] = []

    def start_transcription_job(self, **request: object) -> dict:
        self.started.append(request)
        return {"TranscriptionJob": {"TranscriptionJobName": request["TranscriptionJobName"]}}

    def get_transcription_job(self, TranscriptionJobName: str) -> dict:
        return {
            "TranscriptionJob": {
                "TranscriptionJobName": TranscriptionJobName,
                "TranscriptionJobStatus": "FAILED",
                "FailureReason": "test failure",
            }
        }

    def delete_transcription_job(self, TranscriptionJobName: str) -> None:
        self.deleted.append(TranscriptionJobName)


class AwsTranscribeErrorTest(unittest.TestCase):
    def test_failed_job_raises_typed_error(self) -> None:
        client = AwsTranscribe(transcribe_client=FailedTranscribeClient(), s3_client=object())

        with self.assertRaises(AwsTranscribeError) as caught:
            client.process(
                "s3://input/audio.wav",
                output_s3_uri="s3://out/result.json",
                job_name_suffix="test-job",
                transcription=TranscriptionSettings(media_format="wav"),
            )

        self.assertTrue(caught.exception.job_name.startswith("ampav-aws-transcribe-"))
        self.assertTrue(caught.exception.job_name.endswith("-test-job"))
        self.assertIn("test failure", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
