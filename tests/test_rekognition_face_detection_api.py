"""Offline API checks for the native Rekognition face-detection wrapper."""

from __future__ import annotations

import unittest

from ampav.aws.rekognition_face_detection import AwsRekognitionFaceDetection


class FakeClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.started: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def start_face_detection(self, **request: object) -> dict[str, object]:
        self.started.append(request)
        return {"JobId": "face-job"}

    def get_face_detection(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return self.pages.pop(0)


class AwsRekognitionFaceDetectionTest(unittest.TestCase):
    def test_start_uses_existing_s3_video_without_identity_options(self) -> None:
        native = FakeClient([])
        tool = AwsRekognitionFaceDetection(rekognition_client=native)

        self.assertEqual(tool.start("s3://video-bucket/demo.mp4", job_tag="phase1"), {"JobId": "face-job"})
        self.assertEqual(native.started, [{
            "Video": {"S3Object": {"Bucket": "video-bucket", "Name": "demo.mp4"}},
            "JobTag": "phase1",
        }])

    def test_process_preserves_native_page_boundaries(self) -> None:
        first = {"JobStatus": "SUCCEEDED", "NextToken": "next", "Faces": [{"Timestamp": 0}]}
        second = {"JobStatus": "SUCCEEDED", "Faces": [{"Timestamp": 40}]}
        native = FakeClient([first, second])

        self.assertEqual(AwsRekognitionFaceDetection(rekognition_client=native).process("s3://video-bucket/demo.mp4"), [first, second])
        self.assertEqual(native.calls, [
            {"JobId": "face-job", "MaxResults": 1000},
            {"JobId": "face-job", "MaxResults": 1000, "NextToken": "next"},
        ])

    def test_process_reports_native_failure(self) -> None:
        native = FakeClient([{"JobStatus": "FAILED", "StatusMessage": "unsupported video"}])
        with self.assertRaisesRegex(RuntimeError, "unsupported video"):
            AwsRekognitionFaceDetection(rekognition_client=native).process("s3://video-bucket/demo.mp4")
