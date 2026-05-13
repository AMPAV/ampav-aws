from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AWSSettings(StrictBaseModel):
    region: str | None = None
    profile_name: str | None = None
    access_key_id: str | None = Field(default=None, alias="aws_access_key_id")
    secret_access_key: str | None = Field(default=None, alias="aws_secret_access_key")
    session_token: str | None = Field(default=None, alias="aws_session_token")

    @model_validator(mode="after")
    def validate_credentials(self) -> AWSSettings:
        explicit_keys = self.access_key_id or self.secret_access_key or self.session_token
        if self.profile_name and explicit_keys:
            raise ValueError("Use either profile_name or explicit AWS credentials, not both")
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise ValueError("access_key_id and secret_access_key must be configured together")
        return self


class S3Settings(StrictBaseModel):
    bucket: str
    input_prefix: str = "aws_transcribe/input"
    output_prefix: str = "aws_transcribe/output"


class PathSettings(StrictBaseModel):
    runs_dir: Path | None = None


def load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return raw


def resolve_path_from_config(config_path: Path, path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def redact_aws_credentials(data: dict[str, Any]) -> dict[str, Any]:
    aws_data = data.get("aws", {})
    for key in ("access_key_id", "secret_access_key", "session_token"):
        if aws_data.get(key):
            aws_data[key] = "***"
    return data
