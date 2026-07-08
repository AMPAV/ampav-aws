import io
import json
import tarfile
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from ampav.core.schema import Transcript, WordSegment

from ampav_aws_pipeline.comprehend import extract_named_entities
from ampav_aws_pipeline.transcribe import transcribe_file


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


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads: list[tuple[str, str, str]] = []
        self.deleted: list[tuple[str, str]] = []
        self.archive = build_archive(
            [
                {
                    "File": "transcript.txt",
                    "Entities": [
                        {"BeginOffset": 0, "EndOffset": 9, "Score": 0.99, "Text": "Maya Chen", "Type": "PERSON"},
                        {"BeginOffset": 14, "EndOffset": 20, "Score": 0.98, "Text": "Amazon", "Type": "ORGANIZATION"},
                    ],
                }
            ]
        )

    def upload_file(self, source: str, bucket: str, key: str) -> None:
        self.uploads.append((source, bucket, key))
        self.objects[(bucket, key)] = Path(source).read_bytes()

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key.endswith(".tar.gz"):
            return {"Body": FakeBody(self.archive)}
        if Key == "transcribe-output.json":
            return {"Body": FakeBody(FIXTURE.read_bytes())}
        return {"Body": FakeBody(self.objects[(Bucket, Key)])}

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.deleted.append((Bucket, Key))


class FakeComprehendClient:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.input_s3_uri = ""
        self.output_s3_uri = ""

    def start_entities_detection_job(self, **request: object) -> dict:
        self.started.append(request)
        self.input_s3_uri = str(request["InputDataConfig"]["S3Uri"])
        self.output_s3_uri = f"{str(request['OutputDataConfig']['S3Uri']).rstrip('/')}/job/output.tar.gz"
        return {"JobId": "comprehend-job"}

    def describe_entities_detection_job(self, JobId: str) -> dict:
        now = datetime(2026, 7, 7, tzinfo=timezone.utc)
        return {
            "EntitiesDetectionJobProperties": {
                "JobId": JobId,
                "JobName": "entities-test",
                "JobStatus": "COMPLETED",
                "SubmitTime": now,
                "InputDataConfig": {"S3Uri": self.input_s3_uri, "InputFormat": "ONE_DOC_PER_FILE"},
                "OutputDataConfig": {"S3Uri": self.output_s3_uri},
                "LanguageCode": "en",
            }
        }


class FakeTranscribeClient:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.deleted: list[str] = []

    def start_transcription_job(self, **request: object) -> dict:
        self.started.append(request)
        return {"TranscriptionJob": {"TranscriptionJobName": request["TranscriptionJobName"]}}

    def get_transcription_job(self, TranscriptionJobName: str) -> dict:
        now = datetime(2026, 7, 7, tzinfo=timezone.utc)
        return {
            "TranscriptionJob": {
                "TranscriptionJobName": TranscriptionJobName,
                "TranscriptionJobStatus": "COMPLETED",
                "CreationTime": now,
                "StartTime": now,
                "CompletionTime": now,
                "LanguageCode": "en-US",
                "MediaFormat": "wav",
                "Media": {"MediaFileUri": self.started[0]["Media"]["MediaFileUri"]},
                "Transcript": {"TranscriptFileUri": "s3://out/transcribe-output.json"},
            }
        }

    def delete_transcription_job(self, TranscriptionJobName: str) -> None:
        self.deleted.append(TranscriptionJobName)


class PipelineAdaptersTest(unittest.TestCase):
    def test_extract_named_entities_uploads_transcript_text_and_aligns_timestamps(self) -> None:
        comprehend = FakeComprehendClient()
        s3 = FakeS3Client()
        transcript = Transcript(
            words=[
                WordSegment(word="Maya", start_time=0.0, end_time=0.4),
                WordSegment(word="Chen", start_time=0.5, end_time=0.9),
                WordSegment(word="met", start_time=1.0, end_time=1.2),
                WordSegment(word="Amazon", suffix=".", start_time=1.3, end_time=1.8),
            ]
        )

        result = extract_named_entities(
            transcript,
            input_bucket="in",
            output_s3_uri="s3://out/entities",
            job_name_suffix="entities-test",
            data_access_role_arn="arn:aws:iam::123456789012:role/Comprehend",
            comprehend_client=comprehend,
            s3_client=s3,
            polling_interval=0.001,
        )

        self.assertEqual(result.output.text, "Maya Chen met Amazon.")
        self.assertEqual(result.output.spans[0].start_time, 0.0)
        self.assertEqual(result.output.spans[0].end_time, 0.9)
        self.assertEqual(result.output.spans[1].start_time, 1.3)
        self.assertEqual(result.output.spans[1].end_time, 1.8)
        self.assertEqual(result.parameters["transcript_text_source"], "transcript.words")
        input_s3_uri = comprehend.started[0]["InputDataConfig"]["S3Uri"]
        self.assertTrue(input_s3_uri.startswith("s3://in/aws_comprehend_named_entities/input/"))
        self.assertIn(b"Maya Chen met Amazon.", s3.objects.values())
        self.assertEqual(len(s3.deleted), 1)
        self.assertTrue(s3.deleted[0][1].startswith("aws_comprehend_named_entities/input/"))

    def test_transcribe_file_uploads_local_file_and_deletes_uploaded_input(self) -> None:
        transcribe = FakeTranscribeClient()
        s3 = FakeS3Client()
        with tempfile.TemporaryDirectory() as tmpdir:
            media = Path(tmpdir) / "sample.wav"
            media.write_bytes(b"audio")

            result = transcribe_file(
                media,
                input_bucket="in",
                output_s3_uri="s3://out/transcribe-output.json",
                job_name_suffix="sample",
                transcribe_client=transcribe,
                s3_client=s3,
                polling_interval=0.001,
            )

        self.assertEqual(result.output.text, "Please open the door.")
        self.assertEqual(len(s3.uploads), 1)
        self.assertEqual(transcribe.started[0]["Media"]["MediaFileUri"], f"s3://{s3.uploads[0][1]}/{s3.uploads[0][2]}")
        self.assertEqual(s3.deleted, [("in", s3.uploads[0][2])])
        self.assertEqual(transcribe.deleted, [transcribe.started[0]["TranscriptionJobName"]])


def build_archive(records: list[dict]) -> bytes:
    output = "\n".join(json.dumps(record) for record in records).encode("utf-8")
    fileobj = io.BytesIO()
    with tarfile.open(fileobj=fileobj, mode="w:gz") as archive:
        info = tarfile.TarInfo("output")
        info.size = len(output)
        archive.addfile(info, io.BytesIO(output))
    return fileobj.getvalue()


if __name__ == "__main__":
    unittest.main()
