"""Focused model-free tests for the native Comprehend PII experiment."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PROBE_PATH = Path(__file__).parents[1] / "experiments" / "comprehend_pii.py"
SPEC = importlib.util.spec_from_file_location("comprehend_pii", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load experiment module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FakeTool:
    """Expose a provider-shaped response without contacting AWS."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.comprehend_client = type(
            "Client",
            (), {"meta": type("Meta", (), {"region_name": "us-east-2"})()},
        )()
        self.requests: list[tuple[str, str]] = []

    def process(self, text: str, *, language_code: str) -> dict[str, object]:
        self.requests.append((text, language_code))
        return {
            "Entities": [
                {
                    "Score": 0.99,
                    "Type": "EMAIL",
                    "BeginOffset": 8,
                    "EndOffset": 24,
                }
            ],
            "ResponseMetadata": {"RequestId": "request-id"},
        }


class ComprehendPiiProbeTest(unittest.TestCase):
    def test_probe_retains_complete_native_response_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.txt"
            output_dir = root / "run"
            input_path.write_text("Email: maya@example.edu", encoding="utf-8")
            args = argparse.Namespace(
                input=input_path,
                output_dir=output_dir,
                fixture_id="synthetic-pii",
                language_code="en",
                profile=None,
                region=None,
            )

            with patch.object(PROBE, "AwsComprehendPiiRealtime", FakeTool):
                result = PROBE.run_probe(args)

            response = json.loads(
                (output_dir / "native_pii_entities.json").read_text(encoding="utf-8")
            )
            manifest = PROBE.yaml.safe_load(
                (output_dir / "manifest.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(result["entity_count"], 1)
        self.assertEqual(result["entity_types"], ["EMAIL"])
        self.assertEqual(response["ResponseMetadata"]["RequestId"], "request-id")
        self.assertEqual(manifest["native"]["operation"], "DetectPiiEntities")
        self.assertEqual(manifest["effective_parameters"]["region"], "us-east-2")
        self.assertEqual(
            manifest["response_summary"]["score_semantics"],
            "AWS provider score; not normalized by AMPAV",
        )

    def test_probe_requires_new_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                PROBE._validate_new_output_dir(output_dir)

    def test_entity_types_rejects_malformed_native_response(self) -> None:
        with self.assertRaisesRegex(ValueError, "Entities list"):
            PROBE._entity_types({"Entities": {}})
