"""boto3 session creation helpers for AMPAV AWS tools."""

from __future__ import annotations

from typing import Any

from .config import AWSSettings

# BDW: this makes using boto3's default configuration stuff harder to use and
# really isn't much more than a very thin wrapper around boto3.Session
def create_boto3_session(settings: AWSSettings) -> Any:
    """Create a boto3 session from explicit, profile, or default-chain settings.

    :param settings: AWS region and credential settings.
    :type settings: AWSSettings
    :return: Configured ``boto3.Session`` instance.
    :rtype: Any
    :raises RuntimeError: If boto3 is not installed.
    """
    # BDW: Import at top
    # YF: Agree and adopt.
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

    # BDW: the utility of the function notwithstanding, what you're trying to
    # do can be more easily be implemented as:
    #return boto3.Session(region_name=settings.region,
    #                     profile_name=settings.profile_name,
    #                     aws_access_key_id=settings.access_key_id,
    #                     aws_secret_access_key=settings.secret_access_key,
    #                     aws_session_token=settings.session_token)
    # YF: Agree and adopt.
    
    return boto3.Session(**kwargs)
