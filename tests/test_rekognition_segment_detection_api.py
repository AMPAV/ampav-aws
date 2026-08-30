"""Offline API checks for the native Rekognition segment wrapper."""

from __future__ import annotations

import unittest

from ampav.aws.rekognition_segment_detection import AwsRekognitionSegmentDetection


class FakeClient:
    def __init__(self, pages: list[dict[str, object]]) -> None:
        self.pages = pages
        self.started: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []

    def start_segment_detection(self, **request: object) -> dict[str, object]:
        self.started.append(request)
        return {"JobId": "segment-job"}

    def get_segment_detection(self, **request: object) -> dict[str, object]:
        self.calls.append(request)
        return self.pages.pop(0)


class AwsRekognitionSegmentDetectionTest(unittest.TestCase):
    def test_start_uses_native_shot_and_technical_cue_request(self) -> None:
        native = FakeClient([])
        tool = AwsRekognitionSegmentDetection(rekognition_client=native)

        tool.start("s3://videos/demo.mp4", job_tag="phase1")

        self.assertEqual(native.started, [{
            "Video": {"S3Object": {"Bucket": "videos", "Name": "demo.mp4"}},
            "SegmentTypes": ["SHOT", "TECHNICAL_CUE"],
            "JobTag": "phase1",
        }])

    def test_process_preserves_native_page_boundaries(self) -> None:
        first = {"JobStatus": "SUCCEEDED", "NextToken": "next", "Segments": [{"Type": "SHOT"}]}
        second = {"JobStatus": "SUCCEEDED", "Segments": [{"Type": "TECHNICAL_CUE"}]}
        native = FakeClient([first, second])

        self.assertEqual(AwsRekognitionSegmentDetection(rekognition_client=native).process("s3://videos/demo.mp4"), [first, second])
        self.assertEqual(native.calls, [
            {"JobId": "segment-job", "MaxResults": 1000},
            {"JobId": "segment-job", "MaxResults": 1000, "NextToken": "next"},
        ])

    def test_process_reports_native_failure(self) -> None:
        native = FakeClient([{"JobStatus": "FAILED", "StatusMessage": "unsupported video"}])
        with self.assertRaisesRegex(RuntimeError, "unsupported video"):
            AwsRekognitionSegmentDetection(rekognition_client=native).process("s3://videos/demo.mp4")
