"""AWS tooling for AMPAV."""

from ._version import __version__
from .comprehend_named_entities import AwsComprehendNamedEntities
from .comprehend_named_entities_realtime import AwsComprehendNamedEntitiesRealtime
from .job import AwsJobStatus
from .transcribe import AwsTranscribe, TranscriptionSettings

__all__ = [
    "__version__",
    "AwsComprehendNamedEntities",
    "AwsComprehendNamedEntitiesRealtime",
    "AwsJobStatus",
    "AwsTranscribe",
    "TranscriptionSettings",
]
