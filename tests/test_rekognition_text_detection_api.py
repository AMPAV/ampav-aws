"""Offline API checks for the native Rekognition Video text wrapper."""

from __future__ import annotations

import unittest

from ampav.aws.rekognition_text_detection import AwsRekognitionVideoTextDetection


class FakeRekognitionClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.started: list[dict[str, object]] = []
        self.requests: list[dict[str, object]] = []

    def start_text_detection(self, **request: object) -> dict[str, object]:
        self.started.append(request)
        return {"JobId": "text-job"}

    def get_text_detection(self, **request: object) -> dict[str, object]:
        self.requests.append(request)
        return self.pages.pop(0)


class AwsRekognitionVideoTextDetectionTest(unittest.TestCase):
    def test_start_passes_existing_s3_video_without_uploading(self) -> None:
        native = FakeRekognitionClient([])
        tool = AwsRekognitionVideoTextDetection(rekognition_client=native)

        self.assertEqual(tool.start("s3://video-bucket/path/demo.mp4", job_tag="phase1"), {"JobId": "text-job"})
        self.assertEqual(native.started, [{
            "Video": {"S3Object": {"Bucket": "video-bucket", "Name": "path/demo.mp4"}},
            "JobTag": "phase1",
        }])

    def test_process_preserves_native_page_boundaries(self) -> None:
        first = {"JobStatus": "SUCCEEDED", "NextToken": "next", "TextDetections": [{"Timestamp": 0}]}
        second = {"JobStatus": "SUCCEEDED", "TextDetections": [{"Timestamp": 40}]}
        native = FakeRekognitionClient([first, second])
        tool = AwsRekognitionVideoTextDetection(rekognition_client=native)

        self.assertEqual(tool.process("s3://video-bucket/demo.mp4"), [first, second])
        self.assertEqual(native.requests, [
            {"JobId": "text-job", "MaxResults": 1000},
            {"JobId": "text-job", "MaxResults": 1000, "NextToken": "next"},
        ])

    def test_process_reports_native_failure(self) -> None:
        native = FakeRekognitionClient([{"JobStatus": "FAILED", "StatusMessage": "invalid video"}])
        tool = AwsRekognitionVideoTextDetection(rekognition_client=native)

        with self.assertRaisesRegex(RuntimeError, "invalid video"):
            tool.process("s3://video-bucket/demo.mp4")


if __name__ == "__main__":
    unittest.main()
