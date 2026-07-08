import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ampav.core.schema import NamedEntities, ToolOutput

from ampav_aws_cli import comprehend_named_entities as cli


class FakeS3Client:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.deleted: list[tuple[str, str]] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[(Bucket, Key)] = Body

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.deleted.append((Bucket, Key))


class FakeComprehendNamedEntities:
    instances: list["FakeComprehendNamedEntities"] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.s3_client = FakeS3Client()
        self.process_calls: list[dict] = []
        FakeComprehendNamedEntities.instances.append(self)

    def process(self, input_s3_uri: str, **kwargs):
        self.process_calls.append({"input_s3_uri": input_s3_uri, **kwargs})
        return ToolOutput(
            tool_name="aws_comprehend_named_entities",
            output=NamedEntities(text="Maya"),
        )


class ComprehendNamedEntitiesCliTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeComprehendNamedEntities.instances = []

    def test_cli_parser_accepts_required_settings(self) -> None:
        args = cli.build_cli_parser().parse_args(
            [
                "input.txt",
                "--input-bucket",
                "in",
                "--output-s3-uri",
                "s3://out/entities",
                "--data-access-role-arn",
                "arn:aws:iam::123456789012:role/Comprehend",
                "--region",
                "us-east-2",
                "--job-name-suffix",
                "demo",
                "--include-tool-private",
            ]
        )

        self.assertEqual(args.text_file, "input.txt")
        self.assertEqual(args.input_bucket, "in")
        self.assertEqual(args.output_s3_uri, "s3://out/entities")
        self.assertEqual(args.region, "us-east-2")
        self.assertEqual(args.job_name_suffix, "demo")
        self.assertTrue(args.include_tool_private)

    def test_main_prints_tool_output_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "input.txt"
            text_path.write_text("Maya met Amazon.", encoding="utf-8")

            original = cli.AwsComprehendNamedEntities
            cli.AwsComprehendNamedEntities = FakeComprehendNamedEntities
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main(
                        [
                            str(text_path),
                            "--input-bucket",
                            "in",
                            "--data-access-role-arn",
                            "arn:aws:iam::123456789012:role/Comprehend",
                            "--job-name-suffix",
                            "demo",
                        ]
                    )
            finally:
                cli.AwsComprehendNamedEntities = original

        self.assertEqual(exit_code, 0)
        self.assertIn("tool_name: aws_comprehend_named_entities", stdout.getvalue())
        instance = FakeComprehendNamedEntities.instances[0]
        self.assertEqual(instance.kwargs["data_access_role_arn"], "arn:aws:iam::123456789012:role/Comprehend")
        self.assertIn(b"Maya met Amazon.", instance.s3_client.objects.values())
        self.assertEqual(instance.process_calls[0]["job_name_suffix"], "demo")
        self.assertTrue(instance.process_calls[0]["input_s3_uri"].startswith("s3://in/aws_comprehend_named_entities/input/"))
        self.assertEqual(len(instance.s3_client.deleted), 1)


if __name__ == "__main__":
    unittest.main()
