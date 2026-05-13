"""Small S3 helper functions shared by AMPAV AWS tools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AWSArtifactError

# BDW: This really isn't necessary.  S3 locations are really just URL strings at
# their heart, so they should be handled thus.  Also, this syntax precludes 
# using anything other than AWS S3 -- because they will use "regular" urls, but
# the syntax of the URL will vary depending on the provider. 
#
# The rest of the functions are just really thin wrappers on an s3_client and 
# I'm not sure about the utility of them rather than just having the users
# call the boto3 functions directly.
#
@dataclass(frozen=True)
class S3Location:
    """A bucket/key pair identifying an object in S3.

    :param bucket: S3 bucket name.
    :type bucket: str
    :param key: Object key within the bucket.
    :type key: str
    """

    bucket: str
    key: str

    @property
    def uri(self) -> str:
        """Return the location as an s3:// URI.

        :return: S3 URI in ``s3://bucket/key`` form.
        :rtype: str
        """
        return f"s3://{self.bucket}/{self.key}"


def join_s3_key(prefix: str, filename: str) -> str:
    """Join an S3 key prefix and filename without duplicate slashes.

    :param prefix: S3 key prefix. Empty strings are allowed.
    :type prefix: str
    :param filename: Leaf object name to append to the prefix.
    :type filename: str
    :return: Combined S3 key.
    :rtype: str
    """
    clean_prefix = prefix.strip("/")
    return f"{clean_prefix}/{filename}" if clean_prefix else filename


def upload_file(s3_client: Any, source: Path, destination: S3Location) -> None:
    """Upload a local file to an S3 destination.

    :param s3_client: boto3 S3 client.
    :type s3_client: Any
    :param source: Local file to upload.
    :type source: Path
    :param destination: Destination bucket/key.
    :type destination: S3Location
    """
    try:
        s3_client.upload_file(str(source), destination.bucket, destination.key)
    except Exception as exc:
        raise AWSArtifactError(f"Could not upload {source} to {destination.uri}: {exc}") from exc


def download_file(s3_client: Any, source: S3Location, destination: Path) -> None:
    """Download an S3 object into a local file.

    :param s3_client: boto3 S3 client.
    :type s3_client: Any
    :param source: Source bucket/key.
    :type source: S3Location
    :param destination: Local file path to write.
    :type destination: Path
    """
    try:
        s3_client.download_file(source.bucket, source.key, str(destination))
    except Exception as exc:
        raise AWSArtifactError(f"Could not download {source.uri} to {destination}: {exc}") from exc


def read_json(s3_client: Any, source: S3Location) -> Any:
    """Read an S3 object and parse it as JSON.

    :param s3_client: boto3 S3 client.
    :type s3_client: Any
    :param source: Source bucket/key.
    :type source: S3Location
    :return: Parsed JSON value.
    :rtype: Any
    :raises AWSArtifactError: If the S3 object body is not valid JSON.
    """
    try:
        response = s3_client.get_object(Bucket=source.bucket, Key=source.key)
        with response["Body"] as body:
            return json.loads(body.read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AWSArtifactError(f"Could not parse JSON from {source.uri}: {exc}") from exc
    except Exception as exc:
        raise AWSArtifactError(f"Could not read JSON from {source.uri}: {exc}") from exc
