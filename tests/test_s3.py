import unittest

from ampav.aws.s3 import S3Location, join_s3_key, parse_s3_uri


class S3HelperTest(unittest.TestCase):
    def test_parse_s3_uri(self) -> None:
        location = parse_s3_uri("s3://test-bucket/path/to/audio.wav")

        self.assertEqual(location, S3Location(bucket="test-bucket", key="path/to/audio.wav"))
        self.assertEqual(location.uri, "s3://test-bucket/path/to/audio.wav")

    def test_parse_s3_uri_rejects_invalid_uri(self) -> None:
        with self.assertRaises(ValueError):
            parse_s3_uri("https://example.com/audio.wav")

    def test_join_s3_key(self) -> None:
        self.assertEqual(join_s3_key("/prefix/", "/file.json"), "prefix/file.json")
        self.assertEqual(join_s3_key("", "file.json"), "file.json")


if __name__ == "__main__":
    unittest.main()
