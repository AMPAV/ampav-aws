import unittest

from botocore.exceptions import ClientError

from ampav.core.schema import NamedEntities

from ampav.aws import AwsComprehendNamedEntitiesRealtime, __version__
from ampav.aws.comprehend_named_entities_realtime import (
    AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
)
from ampav.aws.errors import (
    AwsComprehendNamedEntitiesError,
    AwsComprehendNamedEntitiesSchemaError,
)


class FakeComprehendRealtimeClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def detect_entities(self, **request: object) -> object:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        if isinstance(response, Exception):
            raise response
        return response


class AwsComprehendNamedEntitiesRealtimeTest(unittest.TestCase):
    def test_process_calls_detect_entities_and_returns_tool_output(self) -> None:
        text = "Maya Chen met Amazon."
        comprehend = FakeComprehendRealtimeClient(
            [
                {
                    "Entities": [
                        {
                            "Text": "Maya Chen",
                            "Type": "PERSON",
                            "Score": 0.99,
                            "BeginOffset": 0,
                            "EndOffset": 9,
                        },
                        {
                            "Text": "Amazon",
                            "Type": "ORGANIZATION",
                            "Score": 0.98,
                            "BeginOffset": 14,
                            "EndOffset": 20,
                        },
                    ]
                }
            ]
        )
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
        )

        result = tool.process(text, language_code="en")

        self.assertEqual(
            comprehend.requests,
            [{"Text": text, "LanguageCode": "en"}],
        )
        self.assertEqual(result.tool_name, "aws_comprehend_named_entities_realtime")
        self.assertEqual(result.tool_version, __version__)
        self.assertEqual(result.parameters["recognition_model"], "built_in")
        self.assertEqual(
            result.parameters["max_chunk_bytes"],
            AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
        )
        self.assertEqual(result.parameters["chunk_overlap_bytes"], 1_000)
        self.assertEqual(result.parameters["chunk_count"], 1)
        self.assertIsNone(result.tool_private)
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(result.output.text, text)
        self.assertEqual(result.output.languages, ["en"])
        self.assertEqual(
            [
                (
                    entity.text,
                    entity.entity_type,
                    entity.confidence,
                    entity.begin_offset,
                    entity.end_offset,
                    entity.language,
                )
                for entity in result.output.spans
            ],
            [
                ("Maya Chen", "PERSON", 0.99, 0, 9, "en"),
                ("Amazon", "ORGANIZATION", 0.98, 14, 20, "en"),
            ],
        )

    def test_process_uses_utf8_bytes_but_preserves_character_offsets(self) -> None:
        text = "éé aa bb"
        comprehend = FakeComprehendRealtimeClient(
            [
                {
                    "Entities": [
                        {
                            "Text": "éé",
                            "Type": "OTHER",
                            "Score": 0.9,
                            "BeginOffset": 0,
                            "EndOffset": 2,
                        }
                    ]
                },
                {
                    "Entities": [
                        {
                            "Text": "bb",
                            "Type": "OTHER",
                            "Score": 0.8,
                            "BeginOffset": 3,
                            "EndOffset": 5,
                        }
                    ]
                },
            ]
        )
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
            max_chunk_bytes=6,
            chunk_overlap_bytes=0,
        )

        result = tool.process(text)

        self.assertEqual(
            [request["Text"] for request in comprehend.requests],
            ["éé ", "aa bb"],
        )
        self.assertTrue(
            all(
                len(str(request["Text"]).encode("utf-8")) <= 6
                for request in comprehend.requests
            )
        )
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(
            [
                (entity.text, entity.begin_offset, entity.end_offset)
                for entity in result.output.spans
            ],
            [("éé", 0, 2), ("bb", 6, 8)],
        )

    def test_process_reassembles_overlap_by_midpoint_ownership(self) -> None:
        text = "one two three four"
        comprehend = FakeComprehendRealtimeClient(
            [
                {
                    "Entities": [
                        _entity("one", 0, 3),
                        _entity("two", 4, 7),
                    ]
                },
                {"Entities": [_entity("two", 4, 7)]},
                {
                    "Entities": [
                        _entity("two", 0, 3),
                        _entity("three", 4, 9),
                        _entity("four", 10, 14),
                    ]
                },
                {"Entities": [_entity("four", 0, 4)]},
            ]
        )
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
            max_chunk_bytes=14,
            chunk_overlap_bytes=4,
        )

        result = tool.process(text)

        self.assertEqual(
            [request["Text"] for request in comprehend.requests],
            ["one two ", "one two ", "two three four", "four"],
        )
        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        self.assertEqual(
            [
                (entity.text, entity.begin_offset, entity.end_offset)
                for entity in result.output.spans
            ],
            [
                ("one", 0, 3),
                ("two", 4, 7),
                ("three", 8, 13),
                ("four", 14, 18),
            ],
        )

    def test_process_includes_native_chunk_responses_when_enabled(self) -> None:
        response = {"Entities": []}
        comprehend = FakeComprehendRealtimeClient([response])
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
            include_tool_private=True,
        )

        result = tool.process("No entities here.")

        chunks = result.tool_private[
            "aws_comprehend_named_entities_realtime_chunks"
        ]
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["begin_offset"], 0)
        self.assertEqual(chunks[0]["end_offset"], len("No entities here."))
        self.assertEqual(chunks[0]["response"], response)

    def test_process_rejects_missing_entities_list(self) -> None:
        comprehend = FakeComprehendRealtimeClient([{}])
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
        )

        with self.assertRaisesRegex(
            AwsComprehendNamedEntitiesSchemaError,
            r"\$\.chunks\[0\]\.response\.Entities: expected list",
        ):
            tool.process("Maya Chen")

    def test_process_rejects_malformed_or_misaligned_entity(self) -> None:
        cases = [
            (
                {"Entities": [_entity("wrong", 0, 4)]},
                "text does not match offsets",
            ),
            (
                {
                    "Entities": [
                        {
                            "Text": "Maya",
                            "Type": "PERSON",
                            "Score": 0.9,
                            "EndOffset": 4,
                        }
                    ]
                },
                "missing required field 'BeginOffset'",
            ),
        ]
        for response, message in cases:
            with self.subTest(message=message):
                comprehend = FakeComprehendRealtimeClient([response])
                tool = AwsComprehendNamedEntitiesRealtime(
                    comprehend_client=comprehend,
                )

                with self.assertRaisesRegex(
                    AwsComprehendNamedEntitiesSchemaError,
                    message,
                ):
                    tool.process("Maya Chen")

    def test_process_wraps_aws_client_errors(self) -> None:
        error = ClientError(
            {
                "Error": {
                    "Code": "UnsupportedLanguageException",
                    "Message": "unsupported language",
                }
            },
            "DetectEntities",
        )
        comprehend = FakeComprehendRealtimeClient([error])
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
        )

        with self.assertRaisesRegex(
            AwsComprehendNamedEntitiesError,
            "DetectEntities failed with UnsupportedLanguageException",
        ):
            tool.process("Maya Chen", language_code="xx")

    def test_constructor_rejects_invalid_chunk_configuration(self) -> None:
        comprehend = FakeComprehendRealtimeClient([])
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            AwsComprehendNamedEntitiesRealtime(
                comprehend_client=comprehend,
                max_chunk_bytes=AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES + 1,
            )
        with self.assertRaisesRegex(ValueError, "positive owned capacity"):
            AwsComprehendNamedEntitiesRealtime(
                comprehend_client=comprehend,
                max_chunk_bytes=2_000,
                chunk_overlap_bytes=1_000,
            )

    def test_process_rejects_empty_text_before_calling_aws(self) -> None:
        comprehend = FakeComprehendRealtimeClient([])
        tool = AwsComprehendNamedEntitiesRealtime(
            comprehend_client=comprehend,
        )

        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            tool.process("  ")

        self.assertEqual(comprehend.requests, [])


def _entity(text: str, begin: int, end: int) -> dict[str, object]:
    return {
        "Text": text,
        "Type": "OTHER",
        "Score": 0.9,
        "BeginOffset": begin,
        "EndOffset": end,
    }


if __name__ == "__main__":
    unittest.main()
