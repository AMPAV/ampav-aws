"""AWS provider errors."""

from ampav.core.async_tool import ToolError


class AwsTranscribeError(ToolError):
    """Raised when the AWS Transcribe workflow fails."""

    def __init__(self, job_name: str | None, message: str):
        self.job_name = job_name
        prefix = f"AWS Transcribe job {job_name}: " if job_name else "AWS Transcribe: "
        super().__init__(prefix + message)


class AwsTranscriptSchemaError(AwsTranscribeError):
    """Raised when AWS transcript JSON does not match the consumed shape."""

    def __init__(self, path: str, message: str):
        self.path = path
        super().__init__(None, f"{path}: {message}")


class AwsComprehendError(ToolError):
    """Raised when the AWS Comprehend workflow fails."""

    def __init__(self, job_id: str | None, message: str):
        self.job_id = job_id
        prefix = f"AWS Comprehend job {job_id}: " if job_id else "AWS Comprehend: "
        super().__init__(prefix + message)
