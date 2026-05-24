"""AWS provider errors."""

from botocore.exceptions import BotoCoreError, ClientError


class AwsTranscribeError(Exception):
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


def is_aws_sdk_error(exc: BaseException) -> bool:
    """Return true when an exception came from botocore/boto3."""
    return isinstance(exc, (BotoCoreError, ClientError))
