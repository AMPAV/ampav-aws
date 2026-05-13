from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
