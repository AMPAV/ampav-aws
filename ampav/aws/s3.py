from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def join_s3_key(prefix: str, filename: str) -> str:
    clean_prefix = prefix.strip("/")
    return f"{clean_prefix}/{filename}" if clean_prefix else filename


def upload_file(s3_client: Any, source: Path, destination: S3Location) -> None:
    s3_client.upload_file(str(source), destination.bucket, destination.key)


def download_file(s3_client: Any, source: S3Location, destination: Path) -> None:
    s3_client.download_file(source.bucket, source.key, str(destination))


def read_json(s3_client: Any, source: S3Location) -> Any:
    response = s3_client.get_object(Bucket=source.bucket, Key=source.key)
    with response["Body"] as body:
        return json.loads(body.read().decode("utf-8"))
