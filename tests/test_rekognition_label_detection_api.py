"""Offline API checks for the native Rekognition label wrapper."""

from __future__ import annotations

import unittest

from ampav.aws.rekognition_label_detection import AwsRekognitionLabelDetection


class FakeClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.started: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def start_label_detection(self, **request: object) -> dict[str, object]:
        self.started.append(request)
        return {"JobId": "label-job"}

    def get_label_detection(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return self.pages.pop(0)


class AwsRekognitionLabelDetectionTest(unittest.TestCase):
    def test_start_uses_native_video_and_confidence(self) -> None:
        native = FakeClient([])
        AwsRekognitionLabelDetection(rekognition_client=native).start("s3://videos/demo.mp4", job_tag="phase1")
        self.assertEqual(native.started, [{
            "Video": {"S3Object": {"Bucket": "videos", "Name": "demo.mp4"}},
            "MinConfidence": 50.0,
            "JobTag": "phase1",
        }])

    def test_process_preserves_native_page_boundaries(self) -> None:
        first = {"JobStatus": "SUCCEEDED", "NextToken": "next", "Labels": [{"Timestamp": 0}]}
        second = {"JobStatus": "SUCCEEDED", "Labels": [{"Timestamp": 40}]}
        native = FakeClient([first, second])
        self.assertEqual(AwsRekognitionLabelDetection(rekognition_client=native).process("s3://videos/demo.mp4"), [first, second])

    def test_process_reports_native_failure(self) -> None:
        native = FakeClient([{"JobStatus": "FAILED", "StatusMessage": "unsupported video"}])
        with self.assertRaisesRegex(RuntimeError, "unsupported video"):
            AwsRekognitionLabelDetection(rekognition_client=native).process("s3://videos/demo.mp4")
