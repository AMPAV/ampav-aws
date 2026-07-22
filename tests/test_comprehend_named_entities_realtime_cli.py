import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from ampav.core.schema import NamedEntities, ToolOutput

from ampav_aws_cli import comprehend_named_entities_realtime as cli


class FakeComprehendNamedEntitiesRealtime:
    instances: list["FakeComprehendNamedEntitiesRealtime"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.process_calls: list[dict[str, object]] = []
        self.instances.append(self)

    def process(self, text: str, **kwargs: object) -> ToolOutput:
        self.process_calls.append({"text": text, **kwargs})
        return ToolOutput(
            tool_name="aws_comprehend_named_entities_realtime",
            output=NamedEntities(text=text),
        )


class ComprehendNamedEntitiesRealtimeCliTest(unittest.TestCase):
    def setUp(self) -> None:
        FakeComprehendNamedEntitiesRealtime.instances = []

    def test_parser_accepts_realtime_settings(self) -> None:
        args = cli.build_cli_parser().parse_args(
            [
                "input.txt",
                "--language-code",
                "es",
                "--max-chunk-bytes",
                "50000",
                "--chunk-overlap-bytes",
                "500",
                "--profile",
                "research",
                "--region",
                "us-east-2",
                "--include-tool-private",
            ]
        )

        self.assertEqual(args.text_file, "input.txt")
        self.assertEqual(args.language_code, "es")
        self.assertEqual(args.max_chunk_bytes, 50_000)
        self.assertEqual(args.chunk_overlap_bytes, 500)
        self.assertEqual(args.profile, "research")
        self.assertEqual(args.region, "us-east-2")
        self.assertTrue(args.include_tool_private)

    def test_main_reads_text_and_prints_tool_output_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "input.txt"
            text_path.write_text("Maya Chen visited Bloomington.", encoding="utf-8")

            original = cli.AwsComprehendNamedEntitiesRealtime
            cli.AwsComprehendNamedEntitiesRealtime = (
                FakeComprehendNamedEntitiesRealtime
            )
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    exit_code = cli.main(
                        [
                            str(text_path),
                            "--language-code",
                            "en",
                            "--max-chunk-bytes",
                            "50000",
                            "--chunk-overlap-bytes",
                            "500",
                        ]
                    )
            finally:
                cli.AwsComprehendNamedEntitiesRealtime = original

        self.assertEqual(exit_code, 0)
        self.assertIn(
            "tool_name: aws_comprehend_named_entities_realtime",
            stdout.getvalue(),
        )
        instance = FakeComprehendNamedEntitiesRealtime.instances[0]
        self.assertEqual(instance.kwargs["max_chunk_bytes"], 50_000)
        self.assertEqual(instance.kwargs["chunk_overlap_bytes"], 500)
        self.assertFalse(instance.kwargs["include_tool_private"])
        self.assertEqual(
            instance.process_calls,
            [
                {
                    "text": "Maya Chen visited Bloomington.",
                    "language_code": "en",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
