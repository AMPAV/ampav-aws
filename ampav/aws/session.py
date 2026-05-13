from __future__ import annotations

from typing import Any

from .config import AWSSettings


def create_boto3_session(settings: AWSSettings) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required to run AWS tools") from exc

    kwargs: dict[str, Any] = {}
    if settings.region:
        kwargs["region_name"] = settings.region
    if settings.profile_name:
        kwargs["profile_name"] = settings.profile_name
    elif settings.access_key_id and settings.secret_access_key:
        kwargs["aws_access_key_id"] = settings.access_key_id
        kwargs["aws_secret_access_key"] = settings.secret_access_key
        if settings.session_token:
            kwargs["aws_session_token"] = settings.session_token
    return boto3.Session(**kwargs)
