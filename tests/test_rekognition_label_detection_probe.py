"""Offline checks for Rekognition Video label-detection probe helpers."""

from __future__ import annotations

import unittest

from experiments.rekognition_label_detection import build_start_request, fetch_all_label_pages, summarize_pages


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_label_detection(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return {"JobStatus": "SUCCEEDED", "Labels": []}


class RekognitionLabelDetectionProbeTest(unittest.TestCase):
    def test_builds_native_request(self) -> None:
        self.assertEqual(build_start_request("bucket", "input.mp4"), {"Video": {"S3Object": {"Bucket": "bucket", "Name": "input.mp4"}}, "MinConfidence": 50.0, "JobTag": "ampav-aws-rekognition-label"})

    def test_preserves_next_page(self) -> None:
        client = FakeClient()
        first = {"JobStatus": "SUCCEEDED", "Labels": [], "NextToken": "next"}
        self.assertEqual(len(fetch_all_label_pages(client, "job", first_response=first)), 2)
        self.assertEqual(client.calls[0]["NextToken"], "next")

    def test_summarizes_native_taxonomy_without_aggregation(self) -> None:
        summary = summarize_pages([{"Labels": [{"Timestamp": 1000, "Label": {"Name": "Person", "Confidence": 99.0, "Instances": [{}], "Parents": [{"Name": "Human"}]}}, {"Timestamp": 2000, "Label": {"Name": "Person", "Confidence": 80.0, "Parents": [{"Name": "Human"}]}}]}])
        self.assertEqual(summary["label_record_count"], 2)
        self.assertEqual(summary["top_label_names"], [{"name": "Person", "occurrences": 2}])
        self.assertEqual(summary["parent_taxonomy"], {"Human": 2})


if __name__ == "__main__":
    unittest.main()
