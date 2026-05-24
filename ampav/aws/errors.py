"""Exceptions raised by AMPAV AWS helpers."""

from __future__ import annotations

# BDW: There really isn't a terrible need for most of these exceptions, let
# alone a hierarchy.  Unless you're doing something really special, you 
# likely just want to use the built-in exception classes.

# YF: Partially Agree. 
# We can reduce the granularity of Error type to be per tool (AwsTranscribeError). Considering that
# the tools are intended to be used in pipelines, it's helpful to isolate errors per tool for troubleshooing.
# In addition, we should also keep AwsTranscriptSchemaError (subclass of AwsTranscribeError),
# so it's easy to search in logs, as this is a specific one we want to watch over time.
# Note: If a paritcular error type is already defined in boto3, we don't need to wrap it.

class AmpavAWSError(Exception):
    """Base class for AMPAV AWS package errors."""


class AWSConfigError(AmpavAWSError):
    """Raised when an AWS tool configuration cannot be loaded or validated."""


class AWSTranscribeJobError(AmpavAWSError):
    """Raised when an AWS Transcribe job fails or returns an invalid job response."""

    def __init__(self, job_name: str, message: str):
        """Create an error tied to an AWS Transcribe job name.

        :param job_name: AWS Transcribe job name associated with the failure.
        :type job_name: str
        :param message: Human-readable failure details.
        :type message: str
        """
        self.job_name = job_name
        super().__init__(f"AWS Transcribe job {job_name}: {message}")


class AWSTranscriptSchemaError(AmpavAWSError):
    """Raised when AWS transcript JSON is missing fields AMPAV consumes."""

    def __init__(self, path: str, message: str):
        """Create a schema error for a JSON-path-like location.

        :param path: JSON-path-like location of the invalid field.
        :type path: str
        :param message: Human-readable schema problem.
        :type message: str
        """
        self.path = path
        super().__init__(f"{path}: {message}")


class AWSArtifactError(AmpavAWSError):
    """Raised when local or downloaded AWS artifacts cannot be read or written."""


def is_aws_sdk_error(exc: BaseException) -> bool:
    """Return whether an exception comes from botocore/boto3.

    :param exc: Exception to inspect.
    :type exc: BaseException
    :return: ``True`` if the exception is a botocore/boto3 error.
    :rtype: bool
    """
    # BDW: imports should aways happen at the top.  The only time they 
    # wouldn't be is if they're optional (like I do with numpy in core) for
    # the library.  Here, the library is actually loaded into python by boto3
    # so putting this at the top of the file is the right thing to do.
    # YF: Agree and adopt.
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return False
    return isinstance(exc, (BotoCoreError, ClientError))
