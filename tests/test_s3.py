from __future__ import annotations

import unittest
from pathlib import Path

from ampav.aws.errors import AWSArtifactError
from ampav.aws.s3 import S3Location, upload_file


class FailingS3Client:
    def upload_file(self, source: str, bucket: str, key: str) -> None:
        """Simulate a boto3 upload failure."""
        raise RuntimeError(f"upload failed for {bucket}/{key}")


class S3HelperTest(unittest.TestCase):
    def test_upload_file_wraps_client_error(self) -> None:
        with self.assertRaises(AWSArtifactError) as caught:
            upload_file(
                FailingS3Client(),
                Path("sample.wav"),
                S3Location(bucket="test-bucket", key="input/sample.wav"),
            )

        self.assertIn("Could not upload", str(caught.exception))
        self.assertIn("s3://test-bucket/input/sample.wav", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
