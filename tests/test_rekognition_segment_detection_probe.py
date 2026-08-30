"""Offline checks for the native Rekognition segment-detection probe."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ampav.aws.rekognition_segment_detection import AwsRekognitionSegmentDetection
from experiments.rekognition_segment_detection import (
    DEFAULT_SEGMENT_TYPES,
    JOB_TAG,
    summarize_pages,
)


class FakeRekognitionClient:
    def __init__(self, responses: list[dict[str, object]]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []
        self.started: list[dict[str, object]] = []

    def start_segment_detection(self, **kwargs: object) -> dict[str, object]:
        self.started.append(kwargs)
        return {"JobId": "segment-job"}

    def get_segment_detection(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class RekognitionSegmentDetectionProbeTest(unittest.TestCase):
    def test_wrapper_builds_native_request_for_both_segment_types(self) -> None:
        client = FakeRekognitionClient([])
        AwsRekognitionSegmentDetection(rekognition_client=client).start("s3://media-bucket/input/demo.mp4", job_tag=JOB_TAG)
        self.assertEqual(
            client.started,
            [{
                "Video": {"S3Object": {"Bucket": "media-bucket", "Name": "input/demo.mp4"}}, "SegmentTypes": ["SHOT", "TECHNICAL_CUE"], "JobTag": JOB_TAG,
            }],
        )

    def test_waits_until_native_job_succeeds(self) -> None:
        client = FakeRekognitionClient(
            [{"JobStatus": "IN_PROGRESS"}, {"JobStatus": "SUCCEEDED", "Segments": []}]
        )
        tool = AwsRekognitionSegmentDetection(rekognition_client=client, polling_interval=0.1, timeout=10)
        with patch("ampav.aws.rekognition_segment_detection.time.sleep"):
            terminal, history = tool.wait_for_terminal_response("job-1")

        self.assertEqual(terminal["JobStatus"], "SUCCEEDED")
        self.assertEqual(len(history), 2)
        self.assertEqual(client.calls, [{"JobId": "job-1", "MaxResults": 1000}] * 2)

    def test_preserves_pagination_as_native_pages(self) -> None:
        client = FakeRekognitionClient(
            [{"JobStatus": "SUCCEEDED", "Segments": [], "NextToken": "next"}]
        )
        first = {"JobStatus": "SUCCEEDED", "Segments": [], "NextToken": "next"}
        client.responses = [{"JobStatus": "SUCCEEDED", "Segments": []}]

        pages = AwsRekognitionSegmentDetection(rekognition_client=client).get_all_pages("job-1", first_response=first)

        self.assertEqual(pages, [first, {"JobStatus": "SUCCEEDED", "Segments": []}])
        self.assertEqual(
            client.calls,
            [{"JobId": "job-1", "MaxResults": 1000, "NextToken": "next"}],
        )

    def test_summarizes_native_segment_and_cue_types_without_conversion(self) -> None:
        summary = summarize_pages(
            [
                {
                    "Segments": [
                        {"Type": "SHOT", "ShotSegment": {"Confidence": 99.1}},
                        {
                            "Type": "TECHNICAL_CUE",
                            "TechnicalCueSegment": {"Type": "BlackFrames", "Confidence": 98.2},
                        },
                    ]
                }
            ]
        )

        self.assertEqual(summary["segment_count"], 2)
        self.assertEqual(summary["segment_types"], {"SHOT": 1, "TECHNICAL_CUE": 1})
        self.assertEqual(summary["technical_cue_types"], {"BlackFrames": 1})


if __name__ == "__main__":
    unittest.main()
