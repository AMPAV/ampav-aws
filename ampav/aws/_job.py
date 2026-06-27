"""Internal AWS job metadata helpers."""

from dataclasses import dataclass


@dataclass
class _AwsJobMeta:
    """In-process metadata that AWS does not preserve for AMPAV cleanup/output."""

    delete_output: bool = False
    owned_output: bool = False
    include_tool_private: bool = False
