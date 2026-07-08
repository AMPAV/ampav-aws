"""Compatibility re-exports for S3 helpers moved to `ampav_aws_pipeline`."""

from ampav_aws_pipeline.s3_files import upload_file, upload_text

__all__ = ["upload_file", "upload_text"]
