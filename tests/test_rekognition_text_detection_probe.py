"""Offline checks for the native Rekognition Video text-detection probe."""

from __future__ import annotations

import unittest

from experiments.rekognition_text_detection import fetch_all_text_pages, summarize_pages, wait_for_terminal_response


class FakeClient:
    def __init__(self, responses: list[dict[str, object]]):
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def get_text_detection(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.responses.pop(0)


class RekognitionTextDetectionProbeTest(unittest.TestCase):
    def test_waits_for_native_success(self) -> None:
        client = FakeClient([{"JobStatus": "IN_PROGRESS"}, {"JobStatus": "SUCCEEDED"}])
        clock = iter([0.0, 0.0])
        terminal, history = wait_for_terminal_response(client, "job", polling_interval=1, timeout=10, sleep=lambda _: None, monotonic=lambda: next(clock))
        self.assertEqual(terminal["JobStatus"], "SUCCEEDED")
        self.assertEqual(len(history), 2)

    def test_preserves_pagination(self) -> None:
        first = {"JobStatus": "SUCCEEDED", "NextToken": "more", "TextDetections": []}
        client = FakeClient([{"JobStatus": "SUCCEEDED", "TextDetections": []}])
        self.assertEqual(fetch_all_text_pages(client, "job", first_response=first), [first, {"JobStatus": "SUCCEEDED", "TextDetections": []}])
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
