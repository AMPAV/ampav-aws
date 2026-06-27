"""Client-side S3 file helpers for examples and CLI."""

from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

from ampav.aws.s3 import S3Location, join_s3_key
from ampav.aws.transcribe import safe_job_part


def upload_file(
    s3_client: object,
    source: str | PathLike[str],
    *,
    bucket: str,
    key: str | None = None,
    prefix: str = "",
    name_prefix: str = "ampav-aws",
) -> S3Location:
    """Upload a local file to S3 and return its S3 location."""
    source_path = Path(source).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {source_path}")

    if key is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_prefix = safe_job_part(name_prefix) or "ampav-aws"
        safe_stem = safe_job_part(source_path.stem) or "input"
        key = join_s3_key(prefix, f"{safe_prefix}-{timestamp}-{safe_stem}{source_path.suffix}")

    s3_client.upload_file(str(source_path), bucket, key)
    return S3Location(bucket=bucket, key=key)
