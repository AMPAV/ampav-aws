"""Compatibility package for helpers moved to `ampav_aws_pipeline`."""

from ampav_aws_pipeline import extract_named_entities, transcribe_file, upload_file, upload_text

__all__ = [
    "extract_named_entities",
    "transcribe_file",
    "upload_file",
    "upload_text",
]
