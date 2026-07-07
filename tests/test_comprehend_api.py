import io
import json
import tarfile
import unittest
from datetime import datetime, timezone

from ampav.core.async_tool import AsyncStatusCode
from ampav.core.schema import NamedEntities

from ampav.aws.comprehend import AwsComprehend, parse_output_archive
from ampav.aws.errors import AwsComprehendError
from ampav.aws.job import AwsJobStatus


SAMPLE_TEXT = "Maya Chen from Indiana University met Rafael Ortiz at Amazon in Seattle."


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
        output_s3_uri = request["OutputDataConfig"]["S3Uri"]
        self.output_s3_uri = f"{str(output_s3_uri).rstrip('/')}/job/output.tar.gz"
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
        self.deleted: list[tuple[str, str]] = []
        self.reads: list[tuple[str, str]] = []
        self.archive = build_archive(
            [
                {
                    "File": "input.txt",
                    "Entities": [
                        {"BeginOffset": 0, "EndOffset": 10, "Score": 0.99, "Text": "Maya Chen", "Type": "PERSON"},
                        {
                            "BeginOffset": 15,
                            "EndOffset": 33,
                            "Score": 0.98,
                            "Text": "Indiana University",
                            "Type": "ORGANIZATION",
                        },
                    ],
                }
            ]
        )

    def get_object(self, Bucket: str, Key: str) -> dict:
        self.reads.append((Bucket, Key))
        if Key.endswith(".tar.gz"):
            return {"Body": FakeBody(self.archive)}
        return {"Body": FakeBody(SAMPLE_TEXT.encode("utf-8"))}

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

    def test_submit_builds_start_entities_detection_job_request(self) -> None:
        client, comprehend, _ = self.make_client()

        job_id = client.submit(
            "s3://in/aws_comprehend/input/sample.txt",
            output_s3_uri="s3://out/aws_comprehend/output",
            job_name_suffix="entities-test",
        )

        self.assertEqual(job_id, "job-123")
        request = comprehend.started[0]
        self.assertTrue(request["JobName"].startswith("ampav-aws-comprehend-"))
        self.assertTrue(request["JobName"].endswith("-entities-test"))
        self.assertEqual(request["InputDataConfig"]["S3Uri"], "s3://in/aws_comprehend/input/sample.txt")
        self.assertEqual(request["InputDataConfig"]["InputFormat"], "ONE_DOC_PER_FILE")
        self.assertEqual(request["OutputDataConfig"]["S3Uri"], "s3://out/aws_comprehend/output")
        self.assertEqual(request["LanguageCode"], "en")
        self.assertNotIn("ClientRequestToken", request)

    def test_submit_applies_tool_level_kms_defaults(self) -> None:
        comprehend = FakeComprehendClient()
        s3 = FakeS3Client()
        client = AwsComprehend(
            comprehend_client=comprehend,
            s3_client=s3,
            data_access_role_arn="arn:aws:iam::123456789012:role/AwsComprehend",
            output_kms_key_id="arn:aws:kms:us-east-2:123456789012:key/output",
            volume_kms_key_id="arn:aws:kms:us-east-2:123456789012:key/volume",
            entity_recognizer_arn="arn:aws:comprehend:us-east-2:123456789012:entity-recognizer/test",
        )

        client.submit("s3://in/input.txt", output_s3_uri="s3://out/output")

        request = comprehend.started[0]
        self.assertEqual(request["OutputDataConfig"]["KmsKeyId"], "arn:aws:kms:us-east-2:123456789012:key/output")
        self.assertEqual(request["VolumeKmsKeyId"], "arn:aws:kms:us-east-2:123456789012:key/volume")
        self.assertEqual(
            request["EntityRecognizerArn"],
            "arn:aws:comprehend:us-east-2:123456789012:entity-recognizer/test",
        )

    def test_submit_requires_role_arn(self) -> None:
        comprehend = FakeComprehendClient()
        s3 = FakeS3Client()
        client = AwsComprehend(comprehend_client=comprehend, s3_client=s3)

        with self.assertRaises(ValueError):
            client.submit("s3://in/input.txt", output_s3_uri="s3://out/output")

        self.assertEqual(comprehend.started, [])

    def test_get_status_maps_completed_job(self) -> None:
        client, _, _ = self.make_client()
        job_id = client.submit("s3://in/input.txt", output_s3_uri="s3://out/output", job_name_suffix="entities-test")

        status = client.get_status(job_id)

        self.assertIsInstance(status, AwsJobStatus)
        self.assertEqual(status.status, AsyncStatusCode.SUCCEEDED)
        self.assertEqual(status.job_id, "job-123")
        self.assertEqual(status.job_name, "entities-test")
        self.assertEqual(status.input_s3_uri, "s3://in/input.txt")
        self.assertEqual(status.output_s3_uri, "s3://out/output/job/output.tar.gz")

    def test_get_result_converts_output_archive_to_named_entities(self) -> None:
        comprehend = FakeComprehendClient()
        s3 = FakeS3Client()
        client = AwsComprehend(
            comprehend_client=comprehend,
            s3_client=s3,
            data_access_role_arn="arn:aws:iam::123456789012:role/AwsComprehend",
            include_tool_private=True,
        )
        job_id = client.submit(
            "s3://in/input.txt",
            output_s3_uri="s3://out/output",
            job_name_suffix="entities-test",
        )

        result = client.get_result(job_id)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.text, SAMPLE_TEXT)
        self.assertEqual(result.output.languages, ["en"])
        self.assertEqual(
            [
                (entity.text, entity.entity_type, entity.confidence, entity.begin_offset, entity.end_offset, entity.language)
                for entity in result.output.spans
            ],
            [
                ("Maya Chen", "PERSON", 0.99, 0, 10, "en"),
                ("Indiana University", "ORGANIZATION", 0.98, 15, 33, "en"),
            ],
        )
        self.assertEqual(result.parameters["archive_members"], ["output"])
        self.assertEqual(result.parameters["record_count"], 1)
        self.assertEqual(result.tool_private["raw_records"][0]["File"], "input.txt")
        self.assertEqual(result.tool_private["raw_records"][0]["Entities"][0]["Text"], "Maya Chen")
        self.assertEqual(
            s3.reads,
            [
                ("in", "input.txt"),
                ("out", "output/job/output.tar.gz"),
            ],
        )

    def test_process_polls_until_completion_and_returns_tool_output(self) -> None:
        comprehend = FakeComprehendClient(statuses=["IN_PROGRESS", "COMPLETED"])
        s3 = FakeS3Client()
        client = AwsComprehend(
            comprehend_client=comprehend,
            s3_client=s3,
            data_access_role_arn="arn:aws:iam::123456789012:role/AwsComprehend",
            include_tool_private=True,
            polling_interval=0.001,
        )

        result = client.process(
            "s3://in/input.txt",
            output_s3_uri="s3://out/output",
            job_name_suffix="entities-test",
        )

        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.spans[1].text, "Indiana University")
        self.assertEqual(result.tool_private["raw_records"][0]["Entities"][1]["Text"], "Indiana University")
        self.assertGreaterEqual(len(comprehend.described), 3)

    def test_process_omits_tool_private_by_default(self) -> None:
        client, _, _ = self.make_client()

        result = client.process(
            "s3://in/input.txt",
            output_s3_uri="s3://out/output",
            job_name_suffix="entities-test",
        )

        self.assertIsNone(result.tool_private)

    def test_process_raises_on_failed_job(self) -> None:
        client, _, _ = self.make_client(statuses=["FAILED"])

        with self.assertRaises(AwsComprehendError):
            client.process("s3://in/input.txt", output_s3_uri="s3://out/output", job_name_suffix="entities-test")

    def test_process_raises_on_record_error(self) -> None:
        client, _, s3 = self.make_client()
        s3.archive = build_archive(
            [
                {
                    "File": "input.txt",
                    "ErrorCode": "DOCUMENT_SIZE_EXCEEDED",
                    "ErrorMessage": "too large",
                }
            ]
        )

        with self.assertRaisesRegex(AwsComprehendError, "DOCUMENT_SIZE_EXCEEDED"):
            client.process("s3://in/input.txt", output_s3_uri="s3://out/output", job_name_suffix="entities-test")

    def test_process_raises_on_multiple_records(self) -> None:
        client, _, s3 = self.make_client()
        s3.archive = build_archive(
            [
                {"File": "a.txt", "Entities": []},
                {"File": "b.txt", "Entities": []},
            ]
        )

        with self.assertRaisesRegex(AwsComprehendError, "expected one Comprehend output record"):
            client.process("s3://in/input.txt", output_s3_uri="s3://out/output", job_name_suffix="entities-test")

    def test_cleanup_deletes_requested_output_only(self) -> None:
        comprehend = FakeComprehendClient()
        s3 = FakeS3Client()
        client = AwsComprehend(
            comprehend_client=comprehend,
            s3_client=s3,
            data_access_role_arn="arn:aws:iam::123456789012:role/AwsComprehend",
            delete_user_owned_outputs=True,
        )
        job_id = client.submit(
            "s3://in/input.txt",
            output_s3_uri="s3://out/output",
            job_name_suffix="entities-test",
        )

        client.cleanup(job_id)

        self.assertEqual(
            s3.deleted,
            [
                ("out", "output/job/output.tar.gz"),
            ],
        )

    def test_user_owned_output_is_kept_by_default(self) -> None:
        client, _, s3 = self.make_client()
        job_id = client.submit(
            "s3://in/input.txt",
            output_s3_uri="s3://out/output",
            job_name_suffix="entities-test",
        )

        client.cleanup(job_id)

        self.assertEqual(s3.deleted, [])

    def test_submit_derives_tool_managed_output_when_omitted(self) -> None:
        client, _, _ = self.make_client()

        client.submit("s3://in/input.txt", job_name_suffix="entities-test")

        request = client.comprehend_client.started[0]
        self.assertTrue(request["OutputDataConfig"]["S3Uri"].startswith("s3://in/aws_comprehend/_ampav_tmp/ampav-aws-comprehend-"))
        self.assertTrue(request["OutputDataConfig"]["S3Uri"].endswith("-entities-test"))

    def test_tool_managed_output_is_deleted_by_default(self) -> None:
        client, _, s3 = self.make_client()
        job_id = client.submit("s3://in/input.txt", job_name_suffix="entities-test")

        client.cleanup(job_id)

        self.assertEqual(len(s3.deleted), 1)
        bucket, key = s3.deleted[0]
        self.assertEqual(bucket, "in")
        self.assertTrue(key.startswith("aws_comprehend/_ampav_tmp/ampav-aws-comprehend-"))
        self.assertTrue(key.endswith("-entities-test/job/output.tar.gz"))

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
