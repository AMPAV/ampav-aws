"""Native AWS Comprehend real-time PII detection for experiment use."""

from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .errors import AwsComprehendPiiError


class AwsComprehendPiiRealtime:
    """Thin synchronous wrapper around AWS Comprehend ``DetectPiiEntities``.

    This provisional wrapper preserves the native response dictionary rather
    than converting PII into an AMPAV schema. It exists to support focused
    native-behavior evaluation before an integration contract is chosen.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        comprehend_client: Any | None = None,
    ) -> None:
        """Configure PII detection with boto3 settings or an injected client.

        Args:
            region_name: Optional AWS region used when creating a boto3 session.
            profile_name: Optional boto3 profile used for authentication.
            session: Optional injected boto3-compatible session.
            comprehend_client: Optional injected Comprehend client.
        """
        if session is None and comprehend_client is None:
            session = boto3.Session(
                region_name=region_name,
                profile_name=profile_name,
            )
        self.comprehend_client = comprehend_client or session.client("comprehend")

    def process(
        self,
        text: str,
        *,
        language_code: str = "en",
    ) -> dict[str, Any]:
        """Detect PII in text and return AWS's unmodified response dictionary.

        Args:
            text: Non-empty source text passed directly to AWS.
            language_code: AWS Comprehend language code.

        The provider owns text-size and language support enforcement so this
        experiment can observe its native behavior and error responses.
        """
        _validate_text(text)
        _validate_language_code(language_code)
        try:
            response = self.comprehend_client.detect_pii_entities(
                Text=text,
                LanguageCode=language_code,
            )
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = error.get("Code", "UNKNOWN")
            message = error.get("Message", str(exc))
            raise AwsComprehendPiiError(
                f"DetectPiiEntities failed with {code}: {message}",
            ) from exc
        except BotoCoreError as exc:
            raise AwsComprehendPiiError(
                f"DetectPiiEntities request failed: {exc}",
            ) from exc
        if not isinstance(response, dict):
            raise AwsComprehendPiiError(
                "DetectPiiEntities returned a non-object response",
            )
        return response


def _validate_text(text: str) -> None:
    """Reject empty input before making an AWS request."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def _validate_language_code(language_code: str) -> None:
    """Reject missing language codes before making an AWS request."""
    if not isinstance(language_code, str):
        raise TypeError("language_code must be a string")
    if not language_code.strip():
        raise ValueError("language_code must not be empty")
