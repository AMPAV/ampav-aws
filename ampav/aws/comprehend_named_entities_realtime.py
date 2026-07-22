"""AWS Comprehend real-time named-entity detection for plain text."""

from collections.abc import Sequence
import logging
from time import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from ampav.core.schema import NamedEntities, NamedEntity, ToolOutput
from ampav.core.text_chunking import (
    TextChunk,
    TextUnit,
    chunk_text,
    dechunk_text_spans,
    text_to_units,
)

from ._version import DISTRIBUTION_NAME, __version__
from .comprehend_named_entities import aws_entity_to_named_entity
from .errors import (
    AwsComprehendNamedEntitiesError,
    AwsComprehendNamedEntitiesSchemaError,
)


AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES = 100_000
DEFAULT_CHUNK_OVERLAP_BYTES = 1_000
logger = logging.getLogger(__name__)


class AwsComprehendNamedEntitiesRealtime:
    """Thin synchronous wrapper around AWS Comprehend ``DetectEntities``.

    Long input is split into byte-limited windows. Results are reassembled into
    one ``NamedEntities`` value whose offsets refer to the complete source text.
    This initial API supports only AWS's built-in entity recognition model.
    """

    distribution_name = DISTRIBUTION_NAME
    tool_name = "aws_comprehend_named_entities_realtime"
    tool_version = __version__

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        session: Any | None = None,
        comprehend_client: Any | None = None,
        max_chunk_bytes: int = AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
        chunk_overlap_bytes: int = DEFAULT_CHUNK_OVERLAP_BYTES,
        include_tool_private: bool = False,
    ) -> None:
        """Configure real-time named-entity detection.

        Args:
            region_name: Optional AWS region used when creating a boto3 session.
            profile_name: Optional boto3 profile used for authentication.
            session: Optional injected boto3-compatible session.
            comprehend_client: Optional injected Comprehend client.
            max_chunk_bytes: Maximum UTF-8 bytes sent in one request. Callers
                may lower, but not exceed, the built-in AWS provider limit.
            chunk_overlap_bytes: Best-effort context added on each side of a
                chunk's owned range. It must leave positive owned capacity.
            include_tool_private: Include native per-chunk responses and source
                windows for troubleshooting.
        """
        _validate_chunk_config(max_chunk_bytes, chunk_overlap_bytes)
        if session is None and comprehend_client is None:
            session = boto3.Session(
                region_name=region_name,
                profile_name=profile_name,
            )
        self.comprehend_client = comprehend_client or session.client("comprehend")
        self.max_chunk_bytes = max_chunk_bytes
        self.chunk_overlap_bytes = chunk_overlap_bytes
        self.include_tool_private = include_tool_private

    def process(
        self,
        text: str,
        *,
        language_code: str = "en",
    ) -> ToolOutput:
        """Detect entities in plain text and return one normalized AMPAV output.

        Args:
            text: Non-empty source text. Final entity offsets refer to this
                complete string.
            language_code: AWS language code for the built-in entity model.
        """
        _validate_text(text)
        _validate_language_code(language_code)
        units = text_to_units(text, weight_fn=_utf8_size)
        return self._process_with_units(
            text,
            units,
            language_code=language_code,
        )

    def _process_with_units(
        self,
        text: str,
        units: Sequence[TextUnit],
        *,
        language_code: str = "en",
        media_duration: float | None = None,
        extra_parameters: dict[str, Any] | None = None,
    ) -> ToolOutput:
        """Process text using canonical units supplied by an in-package adapter.

        Public direct callers should use :meth:`process`. This internal hook
        lets transcript adapters build text and units together from timestamped
        words so extraction and timestamp alignment use one coordinate system.
        """
        _validate_text(text)
        _validate_language_code(language_code)
        chunks = chunk_text(
            text,
            units,
            max_weight=self.max_chunk_bytes,
            overlap_weight=self.chunk_overlap_bytes,
        )
        parameters: dict[str, Any] = {
            "language_code": language_code,
            "recognition_model": "built_in",
            "provider_max_chunk_bytes": AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES,
            "max_chunk_bytes": self.max_chunk_bytes,
            "chunk_overlap_bytes": self.chunk_overlap_bytes,
            "chunk_count": len(chunks),
        }
        if extra_parameters is not None:
            parameters.update(extra_parameters)
        output = ToolOutput(
            tool_name=self.tool_name,
            tool_version=self.tool_version,
            parameters=parameters,
        )

        logger.debug(
            "Processing %d AWS Comprehend real-time chunk(s) with a %d-byte limit",
            len(chunks),
            self.max_chunk_bytes,
        )
        chunk_outputs: list[tuple[TextChunk, Sequence[NamedEntity]]] = []
        private_chunks: list[dict[str, Any]] = []
        output.start_time = time()
        for chunk_index, chunk in enumerate(chunks):
            logger.debug(
                "Processing AWS Comprehend chunk %d/%d at source offsets %d:%d",
                chunk_index + 1,
                len(chunks),
                chunk.begin_offset,
                chunk.end_offset,
            )
            response = self._detect_entities(
                chunk.text,
                language_code=language_code,
            )
            entities = _response_to_named_entities(
                response,
                language_code=language_code,
                path=f"$.chunks[{chunk_index}].response",
            )
            chunk_outputs.append((chunk, entities))
            if self.include_tool_private:
                private_chunks.append(_private_chunk(chunk, response))
        output.end_time = time()

        try:
            spans = dechunk_text_spans(text, chunk_outputs)
        except (TypeError, ValueError) as exc:
            raise AwsComprehendNamedEntitiesSchemaError(
                "$.chunks",
                str(exc),
            ) from exc

        output.output = NamedEntities(
            media_duration=media_duration,
            text=text,
            spans=spans,
            languages=[language_code],
        )
        if self.include_tool_private:
            output.tool_private = {
                "aws_comprehend_named_entities_realtime_chunks": private_chunks,
            }
        return output

    def _detect_entities(
        self,
        text: str,
        *,
        language_code: str,
    ) -> dict[str, Any]:
        """Call AWS Comprehend and normalize provider call failures."""
        try:
            response = self.comprehend_client.detect_entities(
                Text=text,
                LanguageCode=language_code,
            )
        except ClientError as exc:
            error = exc.response.get("Error", {})
            code = error.get("Code", "UNKNOWN")
            message = error.get("Message", str(exc))
            raise AwsComprehendNamedEntitiesError(
                None,
                f"DetectEntities failed with {code}: {message}",
            ) from exc
        if not isinstance(response, dict):
            raise AwsComprehendNamedEntitiesSchemaError(
                "$.response",
                "expected JSON object",
            )
        return response


def _response_to_named_entities(
    response: dict[str, Any],
    *,
    language_code: str,
    path: str,
) -> list[NamedEntity]:
    """Strictly convert one chunk-local ``DetectEntities`` response."""
    native_entities = response.get("Entities")
    if not isinstance(native_entities, list):
        raise AwsComprehendNamedEntitiesSchemaError(
            f"{path}.Entities",
            "expected list",
        )
    return [
        aws_entity_to_named_entity(
            entity,
            language=language_code,
            path=f"{path}.Entities[{index}]",
        )
        for index, entity in enumerate(native_entities)
    ]


def _private_chunk(chunk: TextChunk, response: dict[str, Any]) -> dict[str, Any]:
    """Describe one source window and its native chunk-local response."""
    return {
        "begin_offset": chunk.begin_offset,
        "end_offset": chunk.end_offset,
        "owned_begin_offset": chunk.owned_begin_offset,
        "owned_end_offset": chunk.owned_end_offset,
        "response": response,
    }


def _validate_chunk_config(max_chunk_bytes: int, chunk_overlap_bytes: int) -> None:
    """Validate caller-configurable chunk limits against AWS's built-in cap."""
    if isinstance(max_chunk_bytes, bool) or not isinstance(max_chunk_bytes, int):
        raise TypeError("max_chunk_bytes must be an integer")
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be greater than zero")
    if max_chunk_bytes > AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES:
        raise ValueError(
            "max_chunk_bytes must not exceed the AWS Comprehend built-in limit "
            f"of {AWS_COMPREHEND_BUILT_IN_MAX_TEXT_BYTES} bytes"
        )
    if isinstance(chunk_overlap_bytes, bool) or not isinstance(chunk_overlap_bytes, int):
        raise TypeError("chunk_overlap_bytes must be an integer")
    if chunk_overlap_bytes < 0:
        raise ValueError("chunk_overlap_bytes must be greater than or equal to zero")
    if chunk_overlap_bytes * 2 >= max_chunk_bytes:
        raise ValueError("chunk_overlap_bytes must leave positive owned capacity")


def _validate_text(text: str) -> None:
    """Reject input that AWS cannot analyze as a meaningful text document."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def _validate_language_code(language_code: str) -> None:
    """Validate the request language before calling AWS."""
    if not isinstance(language_code, str):
        raise TypeError("language_code must be a string")
    if not language_code.strip():
        raise ValueError("language_code must not be empty")


def _utf8_size(value: str) -> int:
    """Return the exact UTF-8 byte size used by AWS's text quota."""
    return len(value.encode("utf-8"))
