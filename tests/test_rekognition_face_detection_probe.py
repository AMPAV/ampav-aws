"""Offline checks for Rekognition Video face-detection probe helpers."""

from __future__ import annotations

import unittest

from experiments.rekognition_face_detection import summarize_pages


class RekognitionFaceDetectionProbeTest(unittest.TestCase):
    def test_summarizes_native_face_fields_without_identity_inference(self) -> None:
        summary = summarize_pages([{"Faces": [
            {"Timestamp": 1000, "Face": {"Confidence": 99.0, "BoundingBox": {}, "Landmarks": [{}], "Emotions": [{}]}},
            {"Timestamp": 1000, "Face": {"Confidence": 95.0, "BoundingBox": {}, "Landmarks": []}},
        ]}])

        self.assertEqual(summary["face_record_count"], 2)
        self.assertEqual(summary["timestamps_with_faces"], 1)
        self.assertEqual(summary["max_faces_at_one_timestamp"], 2)
        self.assertEqual(summary["records_with_bounding_box"], 2)
        self.assertEqual(summary["records_with_landmarks"], 1)
        self.assertEqual(summary["records_with_emotions"], 1)
        self.assertEqual(summary["confidence"], {"minimum": 95.0, "maximum": 99.0, "median": 97.0})
