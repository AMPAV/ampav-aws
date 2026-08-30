"""Offline checks for the native Rekognition Video text-detection probe."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from ampav.aws.rekognition_text_detection import AwsRekognitionVideoTextDetection
from experiments.rekognition_text_detection import summarize_pages


class FakeClient:
    def __init__(self, responses: list[dict[str, object]]):
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get_text_detection(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class RekognitionTextDetectionProbeTest(unittest.TestCase):
    def test_wrapper_waits_for_native_success(self) -> None:
        client = FakeClient([{"JobStatus": "IN_PROGRESS"}, {"JobStatus": "SUCCEEDED"}])
        tool = AwsRekognitionVideoTextDetection(rekognition_client=client, polling_interval=1, timeout=10)
        with patch("ampav.aws.rekognition_text_detection.time.sleep"):
            terminal, history = tool.wait_for_terminal_response("job")
        self.assertEqual(terminal["JobStatus"], "SUCCEEDED")
        self.assertEqual(len(history), 2)

    def test_wrapper_preserves_pagination(self) -> None:
        first = {"JobStatus": "SUCCEEDED", "NextToken": "more", "TextDetections": []}
        client = FakeClient([{"JobStatus": "SUCCEEDED", "TextDetections": []}])
        tool = AwsRekognitionVideoTextDetection(rekognition_client=client)
        self.assertEqual(tool.get_all_pages("job", first_response=first), [first, {"JobStatus": "SUCCEEDED", "TextDetections": []}])
        self.assertEqual(client.calls[0]["NextToken"], "more")

    def test_reports_repetition_and_native_fields(self) -> None:
        summary = summarize_pages([{"TextDetections": [
            {"Timestamp": 0, "TextDetection": {"DetectedText": "AMPAV", "Type": "WORD", "Confidence": 99.0, "Geometry": {}}},
            {"Timestamp": 40, "TextDetection": {"DetectedText": "AMPAV", "Type": "LINE", "Confidence": 95.0, "Geometry": {}}},
        ]}])
        self.assertEqual(summary["detection_types"], {"LINE": 1, "WORD": 1})
        self.assertEqual(summary["unique_detected_text_count"], 1)
        self.assertEqual(summary["top_repeated_detected_text"], [{"text": "ampav", "occurrences": 2}])
        self.assertEqual(summary["geometry_presence"], 2)


if __name__ == "__main__":
    unittest.main()
