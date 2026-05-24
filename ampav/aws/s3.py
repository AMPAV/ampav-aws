"""Small AWS S3 URI helpers."""

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class S3Location:
    """Parsed S3 object location."""

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        """Return the location as an `s3://bucket/key` URI."""
        return f"s3://{self.bucket}/{self.key}"


def parse_s3_uri(uri: str) -> S3Location:
    """Parse an `s3://bucket/key` URI into bucket and key parts."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected S3 URI in s3://bucket/key form, got {uri!r}")
    return S3Location(bucket=parsed.netloc, key=parsed.path.lstrip("/"))


def join_s3_key(prefix: str | None, filename: str) -> str:
    """Join an optional S3 prefix and object filename without leading slashes."""
    clean_prefix = (prefix or "").strip("/")
    clean_filename = filename.lstrip("/")
    return f"{clean_prefix}/{clean_filename}" if clean_prefix else clean_filename
