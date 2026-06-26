import io
import json
import tarfile
import unittest
from datetime import datetime, timezone

from ampav.core.async_tool import AsyncStatusCode

from ampav.aws.comprehend import AwsComprehend, parse_output_archive
from ampav.aws.errors import AwsComprehendError
from ampav_aws_utils.s3_files import upload_text


SAMPLE_TEXT = (
    "Maya Chen from Indiana University met Rafael Ortiz at Amazon in Seattle. "
    "They discussed AMPAV research on June 1, 2026."
)


class FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def __enter__(self) -> "FakeBody":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.data


class FakeComprehendClient:
    def __init__(self, statuses: list[str] | None = None) -> None:
        self.statuses = statuses or ["COMPLETED"]
        self.started: list[dict] = []
        self.described: list[str] = []
        self.stopped: list[str] = []
        self.output_s3_uri = "s3://out/aws_comprehend/output/job/output.tar.gz"

    def start_entities_detection_job(self, **request: object) -> dict:
        self.started.append(request)
        return {"JobId": "job-123", "JobArn": "arn:aws:comprehend:us-east-2:123:entities-detection-job/job-123"}

    def describe_entities_detection_job(self, JobId: str) -> dict:
        self.described.append(JobId)
        status_index = min(len(self.described) - 1, len(self.statuses) - 1)
        status = self.statuses[status_index]
        now = datetime(2026, 6, 1, tzinfo=timezone.utc)
        job = {
            "JobId": JobId,
            "JobArn": "arn:aws:comprehend:us-east-2:123:entities-detection-job/job-123",
            "JobName": "entities-test",
            "JobStatus": status,
            "SubmitTime": now,
            "InputDataConfig": {"S3Uri": "s3://in/input.txt", "InputFormat": "ONE_DOC_PER_FILE"},
            "OutputDataConfig": {"S3Uri": self.output_s3_uri},
            "LanguageCode": "en",
        }
        if status == "FAILED":
            job["Message"] = "simulated failure"
        return {"EntitiesDetectionJobProperties": job}

    def list_entities_detection_jobs(self, **kwargs: object) -> dict:
        return {
            "EntitiesDetectionJobPropertiesList": [
                {
                    "JobId": "job-123",
                    "JobName": "ampav-aws-comprehend-20260601-sample",
                    "JobStatus": "COMPLETED",
                    "OutputDataConfig": {"S3Uri": self.output_s3_uri},
                }
            ],
            "kwargs": kwargs,
        }

    def stop_entities_detection_job(self, JobId: str) -> None:
        self.stopped.append(JobId)


class FakeS3Client:
    def __init__(self) -> None:
        self.puts: list[dict] = []
        self.deleted: list[tuple[str, str]] = []
        self.archive = build_archive(
            [
                {
                    "File": "input.txt",
                    "Entities": [
                        {"BeginOffset": 0, "EndOffset": 10, "Score": 0.99, "Text": "Maya Chen", "Type": "PERSON"},
                        {
                            "BeginOffset": 16,
                            "EndOffset": 34,
                            "Score": 0.98,
                            "Text": "Indiana University",
                            "Type": "ORGANIZATION",
                        },
                    ],
                }
            ]
        )

    def put_object(self, **request: object) -> None:
        self.puts.append(request)

    def get_object(self, Bucket: str, Key: str) -> dict:
        return {"Body": FakeBody(self.archive)}

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.deleted.append((Bucket, Key))


class AwsComprehendApiTest(unittest.TestCase):
    def make_client(
        self,
        statuses: list[str] | None = None,
    ) -> tuple[AwsComprehend, FakeComprehendClient, FakeS3Client]:
        comprehend = FakeComprehendClient(statuses=statuses)
        s3 = FakeS3Client()
        return (
            AwsComprehend(
                comprehend_client=comprehend,
                s3_client=s3,
                data_access_role_arn="arn:aws:iam::123456789012:role/AwsComprehend",
                polling_interval=0.001,
            ),
            comprehend,
            s3,
        )

    def test_upload_text_helper_writes_utf8_s3_object(self) -> None:
        client, _, s3 = self.make_client()

        location = upload_text(s3, SAMPLE_TEXT, bucket="in", key="aws_comprehend/input/sample.txt")

        self.assertEqual(location.uri, "s3://in/aws_comprehend/input/sample.txt")
        self.assertEqual(s3.puts[0]["Bucket"], "in")
        self.assertEqual(s3.puts[0]["Key"], "aws_comprehend/input/sample.txt")
        self.assertEqual(s3.puts[0]["Body"], SAMPLE_TEXT.encode("utf-8"))

    def test_submit_builds_start_entities_detection_job_request(self) -> None:
        client, comprehend, _ = self.make_client()

        job_id = client.submit(
            "s3://in/aws_comprehend/input/sample.txt",
            output_s3_uri="s3://out/aws_comprehend/output",
            job_name="entities-test",
        )

        self.assertEqual(job_id, "job-123")
        request = comprehend.started[0]
        self.assertEqual(request["InputDataConfig"]["S3Uri"], "s3://in/aws_comprehend/input/sample.txt")
        self.assertEqual(request["InputDataConfig"]["InputFormat"], "ONE_DOC_PER_FILE")
        self.assertEqual(request["OutputDataConfig"]["S3Uri"], "s3://out/aws_comprehend/output")
        self.assertEqual(request["LanguageCode"], "en")

    def test_submit_requires_role_arn(self) -> None:
        comprehend = FakeComprehendClient()
        s3 = FakeS3Client()
        client = AwsComprehend(comprehend_client=comprehend, s3_client=s3)

        with self.assertRaises(ValueError):
            client.submit("s3://in/input.txt", output_s3_uri="s3://out/output")

        self.assertEqual(comprehend.started, [])

    def test_get_status_maps_completed_job(self) -> None:
        client, _, _ = self.make_client()
        job_id = client.submit("s3://in/input.txt", output_s3_uri="s3://out/output", job_name="entities-test")

        status = client.get_status(job_id)

        self.assertEqual(status.status, AsyncStatusCode.SUCCEEDED)
        self.assertEqual(status.aws_status, "COMPLETED")
        self.assertEqual(status.output_s3_uri, "s3://out/aws_comprehend/output/job/output.tar.gz")

    def test_get_result_reads_output_archive_into_tool_private(self) -> None:
        client, _, _ = self.make_client()
        job_id = client.submit("s3://in/input.txt", output_s3_uri="s3://out/output", job_name="entities-test")

        result = client.get_result(job_id)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.output, None)
        self.assertEqual(result.parameters["archive_members"], ["output"])
        self.assertEqual(result.tool_private["raw_records"][0]["File"], "input.txt")
        self.assertEqual(result.tool_private["raw_records"][0]["Entities"][0]["Text"], "Maya Chen")

    def test_process_polls_until_completion_and_returns_tool_output(self) -> None:
        client, comprehend, _ = self.make_client(statuses=["IN_PROGRESS", "COMPLETED"])

        result = client.process("s3://in/input.txt", output_s3_uri="s3://out/output", job_name="entities-test")

        self.assertEqual(result.tool_private["raw_records"][0]["Entities"][1]["Text"], "Indiana University")
        self.assertGreaterEqual(len(comprehend.described), 3)

    def test_process_raises_on_failed_job(self) -> None:
        client, _, _ = self.make_client(statuses=["FAILED"])

        with self.assertRaises(AwsComprehendError):
            client.process("s3://in/input.txt", output_s3_uri="s3://out/output", job_name="entities-test")

    def test_cleanup_deletes_requested_output_only(self) -> None:
        client, _, s3 = self.make_client()
        job_id = client.submit(
            "s3://in/input.txt",
            output_s3_uri="s3://out/output",
            delete_output=True,
            job_name="entities-test",
        )

        client.cleanup(job_id)

        self.assertEqual(
            s3.deleted,
            [
                ("out", "aws_comprehend/output/job/output.tar.gz"),
            ],
        )

    def test_submit_derives_owned_output_when_omitted(self) -> None:
        client, _, _ = self.make_client()

        client.submit("s3://in/input.txt", job_name="entities-test")

        request = client.comprehend_client.started[0]
        self.assertEqual(request["OutputDataConfig"]["S3Uri"], "s3://in/aws_comprehend/output/entities-test")

    def test_parse_output_archive_reads_json_lines(self) -> None:
        archive = build_archive(
            [
                {"File": "a.txt", "Entities": []},
                {"File": "b.txt", "ErrorCode": "DOCUMENT_SIZE_EXCEEDED", "ErrorMessage": "too large"},
            ]
        )

        members, records = parse_output_archive(archive)

        self.assertEqual(members, ["output"])
        self.assertEqual([record["File"] for record in records], ["a.txt", "b.txt"])


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
