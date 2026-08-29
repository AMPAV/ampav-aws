import unittest

from botocore.exceptions import ClientError, NoCredentialsError

from ampav.aws.comprehend_pii_realtime import AwsComprehendPiiRealtime
from ampav.aws.errors import AwsComprehendPiiError


class FakeComprehendPiiClient:
    """Record PII requests and return a configurable native response."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def detect_pii_entities(self, **request: object) -> object:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


class AwsComprehendPiiRealtimeTest(unittest.TestCase):
    def test_process_preserves_native_request_and_response(self) -> None:
        text = "Call Maya Chen at 555-0100."
        response = {
            "Entities": [
                {
                    "Score": 0.99,
                    "Type": "PHONE",
                    "BeginOffset": 22,
                    "EndOffset": 30,
                }
            ],
            "ResponseMetadata": {"RequestId": "request-id"},
        }
        client = FakeComprehendPiiClient(response)
        tool = AwsComprehendPiiRealtime(comprehend_client=client)

        result = tool.process(text, language_code="en")

        self.assertIs(result, response)
        self.assertEqual(
            client.requests,
            [{"Text": text, "LanguageCode": "en"}],
        )

    def test_process_rejects_empty_text_before_calling_aws(self) -> None:
        client = FakeComprehendPiiClient({"Entities": []})
        tool = AwsComprehendPiiRealtime(comprehend_client=client)

        with self.assertRaisesRegex(ValueError, "text must not be empty"):
            tool.process("  ")

        self.assertEqual(client.requests, [])

    def test_process_rejects_missing_language_code_before_calling_aws(self) -> None:
        client = FakeComprehendPiiClient({"Entities": []})
        tool = AwsComprehendPiiRealtime(comprehend_client=client)

        with self.assertRaisesRegex(ValueError, "language_code must not be empty"):
            tool.process("Maya Chen", language_code=" ")

        self.assertEqual(client.requests, [])

    def test_process_wraps_aws_client_errors(self) -> None:
        client = FakeComprehendPiiClient(
            ClientError(
                {
                    "Error": {
                        "Code": "UnsupportedLanguageException",
                        "Message": "unsupported language",
                    }
                },
                "DetectPiiEntities",
            )
        )
        tool = AwsComprehendPiiRealtime(comprehend_client=client)

        with self.assertRaisesRegex(
            AwsComprehendPiiError,
            "DetectPiiEntities failed with UnsupportedLanguageException",
        ):
            tool.process("Maya Chen", language_code="xx")

    def test_process_wraps_boto_setup_errors(self) -> None:
        tool = AwsComprehendPiiRealtime(
            comprehend_client=FakeComprehendPiiClient(NoCredentialsError()),
        )

        with self.assertRaisesRegex(
            AwsComprehendPiiError,
            "DetectPiiEntities request failed: Unable to locate credentials",
        ):
            tool.process("Maya Chen")

    def test_process_rejects_non_object_response(self) -> None:
        tool = AwsComprehendPiiRealtime(
            comprehend_client=FakeComprehendPiiClient([]),
        )

        with self.assertRaisesRegex(AwsComprehendPiiError, "non-object response"):
            tool.process("Maya Chen")
