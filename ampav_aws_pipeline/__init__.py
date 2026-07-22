"""Blocking AWS pipeline adapters for connecting AMPAV tool outputs."""

from .comprehend import (
    extract_named_entities,
    extract_named_entities_realtime_from_transcript,
)
from .s3_files import upload_file, upload_text
from .transcribe import transcribe_file

__all__ = [
    "extract_named_entities",
    "extract_named_entities_realtime_from_transcript",
    "transcribe_file",
    "upload_file",
    "upload_text",
]
