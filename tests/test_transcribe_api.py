import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ampav.core.async_tool import AsyncStatusCode

from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings
from ampav_aws_cli.transcribe import build_cli_parser
from ampav_aws_utils.s3_files import upload_file


FIXTURE = Path(__file__).parent / "fixtures" / "aws_transcript_opendoor.json"


class FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self) -> "FakeBody":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


class FakeTranscribeClient:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.deleted: list[str] = []

    def start_transcription_job(self, **request: object) -> dict:
        self.started.append(request)
        return {"TranscriptionJob": {"TranscriptionJobName": request["TranscriptionJobName"]}}

    def get_transcription_job(self, TranscriptionJobName: str) -> dict:
        now = datetime(2026, 5, 24, tzinfo=timezone.utc)
        return {
            "TranscriptionJob": {
                "TranscriptionJobName": TranscriptionJobName,
                "TranscriptionJobStatus": "COMPLETED",
                "CreationTime": now,
                "StartTime": now,
                "CompletionTime": now,
                "LanguageCode": "en-US",
                "MediaFormat": "wav",
                "Media": {"MediaFileUri": "s3://input/audio.wav"},
                "Transcript": {"TranscriptFileUri": "s3://out/result.json"},
            }
        }

    def list_transcription_jobs(self, **kwargs: object) -> dict:
        return {"TranscriptionJobSummaries": []}

    def delete_transcription_job(self, TranscriptionJobName: str) -> None:
        self.deleted.append(TranscriptionJobName)


class FakeS3Client:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self.transcript = FIXTURE.read_bytes()

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.uploads.append((source, bucket, key))

    def get_object(self, Bucket: str, Key: str) -> dict:
        return {"Body": FakeBody(self.transcript)}

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.deleted.append((Bucket, Key))


class AwsTranscribeApiTest(unittest.TestCase):
    def make_client(self) -> tuple[AwsTranscribe, FakeTranscribeClient, FakeS3Client]:
        transcribe = FakeTranscribeClient()
        s3 = FakeS3Client()
        return AwsTranscribe(transcribe_client=transcribe, s3_client=s3), transcribe, s3

    def test_submit_from_existing_s3_uri_does_not_upload(self) -> None:
        client, transcribe, s3 = self.make_client()

        job = client.submit(
            "s3://input/audio.wav",
            output_s3_uri="s3://out/result.json",
            job_name_suffix="test-job",
            transcription=TranscriptionSettings(media_format="wav"),
        )

        self.assertTrue(job.startswith("ampav-aws-transcribe-"))
        self.assertTrue(job.endswith("-test-job"))
        self.assertEqual(s3.uploads, [])
        self.assertEqual(transcribe.started[0]["TranscriptionJobName"], job)
        self.assertEqual(transcribe.started[0]["Media"]["MediaFileUri"], "s3://input/audio.wav")
        self.assertEqual(transcribe.started[0]["OutputBucketName"], "out")
        self.assertEqual(transcribe.started[0]["OutputKey"], "result.json")

    def test_submit_rejects_non_s3_uri(self) -> None:
        client, transcribe, _ = self.make_client()

        with self.assertRaises(ValueError):
            client.submit(
                "https://example.com/audio.wav",
                transcription=TranscriptionSettings(media_format="wav"),
            )

        self.assertEqual(transcribe.started, [])

    def test_upload_file_helper_uploads_local_file(self) -> None:
        _, _, s3 = self.make_client()
        with tempfile.TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "sample.wav"
            audio.write_bytes(b"test")

            location = upload_file(
                s3,
                audio,
                bucket="in",
                key="input/sample.wav",
            )

        self.assertEqual(location.uri, "s3://in/input/sample.wav")
        self.assertEqual(s3.uploads[0][1:], ("in", "input/sample.wav"))

    def test_get_status_maps_completed_job(self) -> None:
        client, _, _ = self.make_client()
        job = client.submit(
            "s3://input/audio.wav",
            output_s3_uri="s3://out/result.json",
            job_name_suffix="test-job",
            transcription=TranscriptionSettings(media_format="wav"),
        )

        status = client.get_status(job)

        self.assertEqual(status.status, AsyncStatusCode.SUCCEEDED)
        self.assertEqual(status.job_id, job)

    def test_process_s3_uri_returns_tool_output_with_private_raw_data(self) -> None:
        client, transcribe, _ = self.make_client()

        output = client.process(
            "s3://input/audio.wav",
            output_s3_uri="s3://out/result.json",
            job_name_suffix="test-job",
            include_tool_private=True,
            transcription=TranscriptionSettings(media_format="wav"),
        )

        self.assertEqual(output.output.text, "Please open the door.")
        self.assertEqual(output.parameters["media_format"], "wav")
        self.assertEqual(output.parameters["language_code"], "en-US")
        self.assertNotIn("content_source", output.parameters)
        self.assertNotIn("transcript_file_uri", output.parameters)
        self.assertIn("raw_transcript", output.tool_private)
        self.assertTrue(output.tool_private["aws_transcribe_job"]["name"].startswith("ampav-aws-transcribe-"))
        self.assertTrue(output.tool_private["aws_transcribe_job"]["name"].endswith("-test-job"))
        self.assertEqual(transcribe.deleted, [output.tool_private["aws_transcribe_job"]["name"]])

    def test_process_omits_tool_private_by_default(self) -> None:
        client, _, _ = self.make_client()

        output = client.process(
            "s3://input/audio.wav",
            output_s3_uri="s3://out/result.json",
            job_name_suffix="public-output",
            transcription=TranscriptionSettings(media_format="wav"),
        )

        self.assertIsNone(output.tool_private)

    def test_cleanup_deletes_requested_output_and_job(self) -> None:
        client, transcribe, s3 = self.make_client()
        job = client.submit(
            "s3://input/audio.wav",
            output_s3_uri="s3://out/result.json",
            delete_output=True,
            job_name_suffix="cleanup-job",
            transcription=TranscriptionSettings(media_format="wav"),
        )

        client.cleanup(job)

        self.assertEqual(transcribe.deleted, [job])
        self.assertEqual(s3.deleted, [("out", "result.json")])

    def test_cli_parses_s3_input_and_cleanup_flags(self) -> None:
        args = build_cli_parser().parse_args(
            [
                "s3://input/audio.wav",
                "--output-s3-uri",
                "s3://out/result.json",
                "--region",
                "us-east-2",
                "--delete-output",
                "--job-name-suffix",
                "demo",
                "--include-tool-private",
            ]
        )

        self.assertEqual(args.media, "s3://input/audio.wav")
        self.assertEqual(args.output_s3_uri, "s3://out/result.json")
        self.assertEqual(args.region, "us-east-2")
        self.assertTrue(args.delete_output)
        self.assertEqual(args.job_name_suffix, "demo")
        self.assertTrue(args.include_tool_private)


if __name__ == "__main__":
    unittest.main()
