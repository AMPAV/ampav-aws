"""AWS async job status models."""

from ampav.core.async_tool import AsyncJobStatus


class AwsJobStatus(AsyncJobStatus):
    """AWS job status with selected provider location details."""

    job_name: str | None = None
    input_s3_uri: str | None = None
    output_s3_uri: str | None = None
