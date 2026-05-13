from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ampav.aws.errors import AWSConfigError
from ampav.aws.transcribe import load_config


class AWSTranscribeConfigTest(unittest.TestCase):
    def test_runs_dir_defaults_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aws_config.yaml"
            config_path.write_text(
                """
s3:
  bucket: test-bucket
""".lstrip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertIsNone(config.paths.runs_dir)

    def test_relative_runs_dir_resolves_from_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config" / "aws_config.yaml"
            config_path.parent.mkdir()
            config_path.write_text(
                """
s3:
  bucket: test-bucket
paths:
  runs_dir: ../runs
""".lstrip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.paths.runs_dir, (config_path.parent / "../runs").resolve())

    def test_missing_s3_bucket_raises_config_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "aws_config.yaml"
            config_path.write_text("s3: {}\n", encoding="utf-8")

            with self.assertRaises(AWSConfigError) as caught:
                load_config(config_path)

        self.assertIn("Invalid AWS Transcribe config", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
