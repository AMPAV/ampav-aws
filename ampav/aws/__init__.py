"""AWS tooling for AMPAV."""

__version__ = "0.0.9"
DISTRIBUTION_NAME = "ampav-aws"

from .comprehend_named_entities import AwsComprehendNamedEntities
from .comprehend_named_entities_realtime import AwsComprehendNamedEntitiesRealtime
from .job import AwsJobStatus
from .rekognition_face_detection import AwsRekognitionFaceDetection
from .rekognition_label_detection import AwsRekognitionLabelDetection
from .rekognition_segment_detection import AwsRekognitionSegmentDetection
from .rekognition_text_detection import AwsRekognitionVideoTextDetection
from .transcribe import AwsTranscribe, TranscriptionSettings

__all__ = [
    "__version__",
    "AwsComprehendNamedEntities",
    "AwsComprehendNamedEntitiesRealtime",
    "AwsJobStatus",
    "AwsRekognitionFaceDetection",
    "AwsRekognitionLabelDetection",
    "AwsRekognitionSegmentDetection",
    "AwsRekognitionVideoTextDetection",
    "AwsTranscribe",
    "TranscriptionSettings",
]
