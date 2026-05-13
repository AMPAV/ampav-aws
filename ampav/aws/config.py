"""Shared configuration models and YAML helpers for AMPAV AWS tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import AWSConfigError


class StrictBaseModel(BaseModel):
    """Base pydantic model for strict AMPAV AWS configuration sections.

    Configuration models reject unknown fields so misspelled YAML keys fail
    early instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class AWSSettings(StrictBaseModel):
    """AWS credential and region settings used to create boto3 sessions.

    Explicit access keys, a named AWS profile, or the default boto3 credential
    chain can be used. Explicit keys and profile names are mutually exclusive.
    """

    region: str | None = None
    profile_name: str | None = None
    access_key_id: str | None = Field(default=None, alias="aws_access_key_id")
    secret_access_key: str | None = Field(default=None, alias="aws_secret_access_key")
    session_token: str | None = Field(default=None, alias="aws_session_token")

    @model_validator(mode="after")
    def validate_credentials(self) -> AWSSettings:
        """Validate mutually exclusive credential configuration styles.

        :return: The validated AWS settings instance.
        :rtype: AWSSettings
        :raises ValueError: If profile and explicit keys are mixed, or if only
            one half of an access-key pair is configured.
        """
        explicit_keys = self.access_key_id or self.secret_access_key or self.session_token
        if self.profile_name and explicit_keys:
            raise ValueError("Use either profile_name or explicit AWS credentials, not both")
        if bool(self.access_key_id) != bool(self.secret_access_key):
            raise ValueError("access_key_id and secret_access_key must be configured together")
        return self


class S3Settings(StrictBaseModel):
    """S3 bucket and key-prefix settings shared by AWS tools.

    The bucket is required. Prefixes default to locations suitable for AWS
    Transcribe input and output objects.
    """

    bucket: str
    input_prefix: str = "aws_transcribe/input"
    output_prefix: str = "aws_transcribe/output"


class PathSettings(StrictBaseModel):
    """Optional local paths used by AWS tools for debug artifacts.

    When ``runs_dir`` is ``None`` local artifact persistence is disabled.
    """

    runs_dir: Path | None = None


def load_yaml_mapping(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file and require a top-level mapping.

    :param config_path: Path to the YAML config file.
    :type config_path: Path
    :return: Parsed top-level YAML mapping.
    :rtype: dict[str, Any]
    :raises AWSConfigError: If the file cannot be read, parsed, or does not
        contain a top-level mapping.
    """
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AWSConfigError(f"Could not read config file {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise AWSConfigError(f"Could not parse YAML config file {config_path}: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise AWSConfigError(f"Config file must contain a YAML mapping: {config_path}")
    return raw


def resolve_path_from_config(config_path: Path, path: Path | None) -> Path | None:
    """Resolve a config path relative to the config file directory.

    :param config_path: Path to the config file that supplied ``path``.
    :type config_path: Path
    :param path: Path value to resolve. Optional; when ``None``, ``None`` is
        returned.
    :type path: Path | None
    :return: Absolute path, original absolute path, or ``None``.
    :rtype: Path | None
    """
    if path is None or path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def redact_aws_credentials(data: dict[str, Any]) -> dict[str, Any]:
    """Redact AWS credential values from a serialized config dictionary.

    :param data: Serialized config dictionary to redact in place.
    :type data: dict[str, Any]
    :return: The same dictionary with AWS secret fields replaced by ``"***"``.
    :rtype: dict[str, Any]
    """
    aws_data = data.get("aws", {})
    for key in ("access_key_id", "secret_access_key", "session_token"):
        if aws_data.get(key):
            aws_data[key] = "***"
    return data
