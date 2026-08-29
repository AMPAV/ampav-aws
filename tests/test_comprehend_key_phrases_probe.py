"""Focused model-free tests for the native Comprehend key-phrase experiment."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch


PROBE_PATH = (
    Path(__file__).parents[1] / "experiments" / "comprehend_key_phrases.py"
)
SPEC = importlib.util.spec_from_file_location("comprehend_key_phrases", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load experiment module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class FakeComprehendClient:
    """Record native calls and return fixed provider-shaped responses."""

    def __init__(self) -> None:
        self.key_phrase_requests: list[dict[str, str]] = []
        self.entity_requests: list[dict[str, str]] = []
        self.meta = SimpleNamespace(region_name="us-east-2")

    def detect_key_phrases(self, **request):
        self.key_phrase_requests.append(request)
        return {
            "KeyPhrases": [
                {
                    "Text": "the bicycle race",
                    "Score": 0.98,
                    "BeginOffset": 0,
                    "EndOffset": 16,
                }
            ],
            "ResponseMetadata": {"RequestId": "request-id"},
        }

    def detect_entities(self, **request):
        self.entity_requests.append(request)
        return {
            "Entities": [
                {
                    "Text": "bicycle race",
                    "Score": 0.8,
                    "Type": "EVENT",
                    "BeginOffset": 4,
                    "EndOffset": 16,
                }
            ]
        }


class ComprehendKeyPhraseExperimentTest(unittest.TestCase):
    """Protect native calls, limits, diagnostics, and retained metadata."""

    def test_native_calls_preserve_request_and_response_shape(self) -> None:
        client = FakeComprehendClient()
        request = {"Text": "the bicycle race", "LanguageCode": "en"}

        key_phrases = PROBE._detect_key_phrases(client, request)
        entities = PROBE._detect_entities(client, request)

        self.assertEqual(client.key_phrase_requests, [request])
        self.assertEqual(client.entity_requests, [request])
        self.assertEqual(key_phrases["ResponseMetadata"]["RequestId"], "request-id")
        self.assertEqual(entities["Entities"][0]["Type"], "EVENT")

    def test_probe_retains_complete_native_responses_and_manifest(self) -> None:
        client = FakeComprehendClient()
        session = SimpleNamespace(
            region_name="us-east-2",
            client=lambda service: client,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.txt"
            output_dir = root / "run"
            input_path.write_text("the bicycle race", encoding="utf-8")
            args = argparse.Namespace(
                input=input_path,
                output_dir=output_dir,
                fixture_id="fixture",
                language_code="en",
                profile=None,
                region=None,
                include_entities=True,
            )

            with patch.object(PROBE.boto3, "Session", return_value=session):
                result = PROBE.run_probe(args)

            key_phrases = json.loads(
                (output_dir / "native_key_phrases.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = PROBE.yaml.safe_load(
                (output_dir / "manifest.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(result["key_phrase_occurrences"], 1)
        self.assertEqual(key_phrases["ResponseMetadata"]["RequestId"], "request-id")
        self.assertEqual(manifest["effective_parameters"]["region"], "us-east-2")
        self.assertEqual(
            manifest["native_operations"],
            ["DetectKeyPhrases", "DetectEntities"],
        )

    def test_text_limit_is_exclusive_and_utf8_based(self) -> None:
        self.assertEqual(
            PROBE._validate_text("caf\N{LATIN SMALL LETTER E WITH ACUTE}"),
            b"caf\xc3\xa9",
        )
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            PROBE._validate_text("")
        with self.assertRaisesRegex(ValueError, "less than 100000"):
            PROBE._validate_text("x" * PROBE.AWS_MAX_TEXT_BYTES)

    def test_analysis_preserves_native_score_meaning_and_overlap_counts(self) -> None:
        text = "the bicycle race and a race"
        key_phrases = {
            "KeyPhrases": [
                {
                    "Text": "the bicycle race",
                    "Score": 0.9,
                    "BeginOffset": 0,
                    "EndOffset": 16,
                },
                {
                    "Text": "a race",
                    "Score": 0.7,
                    "BeginOffset": 21,
                    "EndOffset": 27,
                },
            ]
        }
        entities = {
            "Entities": [
                {
                    "Text": "bicycle race",
                    "Score": 0.8,
                    "Type": "EVENT",
                    "BeginOffset": 4,
                    "EndOffset": 16,
                }
            ]
        }

        report = PROBE.analyze_responses(
            text,
            key_phrases,
            entity_response=entities,
            reference_keywords=[{"text": "bicycle race"}],
            top_phrases=2,
        )

        summary = report["summary"]
        self.assertEqual(summary["aws_key_phrase_occurrence_count"], 2)
        self.assertEqual(summary["aws_entity_occurrences_contained_in_a_keyphrase"], 1)
        self.assertEqual(
            summary["aws_key_phrase_score"]["meaning"],
            "AWS confidence that the detected span is a noun phrase",
        )
        self.assertEqual(
            report["reference_keyword_comparison"][0][
                "aws_phrase_containing_variants"
            ],
            ["the bicycle race"],
        )

    def test_token_sequence_matching_does_not_use_substrings(self) -> None:
        self.assertTrue(PROBE._contains_token_sequence("delta chi team", "delta chi"))
        self.assertFalse(PROBE._contains_token_sequence("chief steward", "chi"))
        self.assertEqual(PROBE._literal_occurrences("chief and chi", "chi"), 1)

    def test_offset_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "offset mismatch"):
            PROBE.analyze_responses(
                "bicycle race",
                {
                    "KeyPhrases": [
                        {
                            "Text": "race",
                            "Score": 0.9,
                            "BeginOffset": 0,
                            "EndOffset": 4,
                        }
                    ]
                },
            )

    def test_reference_keyword_loader_accepts_strings_and_provider_objects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "keywords.json"
            path.write_text(
                json.dumps(
                    [
                        "bicycle race",
                        {"text": "student team", "instances": [{}, {}]},
                    ]
                ),
                encoding="utf-8",
            )

            loaded = PROBE._load_reference_keywords(path)

        self.assertEqual(
            loaded,
            [
                {"text": "bicycle race", "instance_count": None},
                {"text": "student team", "instance_count": 2},
            ],
        )

    def test_manifest_records_native_context_without_profile(self) -> None:
        args = SimpleNamespace(
            fixture_id="fixture",
            language_code="en",
            region="us-east-2",
            profile="private-profile",
        )
        timestamp = datetime(2026, 8, 29, tzinfo=timezone.utc)

        manifest = PROBE._build_manifest(
            args,
            text="bicycle race",
            text_bytes=b"bicycle race",
            sha256="checksum",
            api_count=1,
            started_at=timestamp,
            completed_at=timestamp,
            elapsed_seconds=0.25,
            effective_region="us-east-2",
            key_phrase_response={"KeyPhrases": []},
            entity_response=None,
        )

        self.assertEqual(manifest["task"], "AMPAV-63")
        self.assertEqual(manifest["native_operations"], ["DetectKeyPhrases"])
        self.assertNotIn("profile", json.dumps(manifest))
        self.assertEqual(manifest["estimated_billing"]["total_units"], 3)

    def test_existing_probe_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                PROBE._validate_new_output_dir(Path(directory))

    def test_negative_top_phrase_count_is_rejected(self) -> None:
        args = argparse.Namespace(top_phrases=-1)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            PROBE.run_analysis(args)


if __name__ == "__main__":
    unittest.main()
