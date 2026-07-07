import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ampav.core.schema import NamedEntities

from ampav.aws.comprehend import AwsComprehend
from ampav.aws.s3 import S3Location, join_s3_key


SAMPLE_TEXT = (
    "Maya Chen from Indiana University met Rafael Ortiz at Amazon in Seattle. "
    "They discussed AMPAV research on June 1, 2026."
)


@unittest.skipUnless(
    os.environ.get("AMPAV_AWS_COMPREHEND_LIVE_TEST") == "1"
    and os.environ.get("AMPAV_AWS_COMPREHEND_CONFIG"),
    "set AMPAV_AWS_COMPREHEND_LIVE_TEST=1 and AMPAV_AWS_COMPREHEND_CONFIG to run live AWS test",
)
class AwsComprehendLiveTest(unittest.TestCase):
    def test_live_entities_job_returns_raw_provider_result(self) -> None:
        config = load_yaml(Path(os.environ["AMPAV_AWS_COMPREHEND_CONFIG"]))
        aws_config = config.get("aws", {})
        s3_config = config.get("s3", {})
        comprehend_config = config.get("comprehend", {})
        polling_config = config.get("polling", {})

        bucket = s3_config.get("bucket")
        role_arn = comprehend_config.get("data_access_role_arn") or aws_config.get("role_arn")
        self.assertIsNotNone(bucket)
        self.assertIsNotNone(role_arn)

        client = AwsComprehend(
            region_name=aws_config.get("region"),
            profile_name=aws_config.get("profile_name"),
            data_access_role_arn=role_arn,
            delete_user_owned_outputs=True,
            include_tool_private=True,
            polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
            timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
        )
        job_name_suffix = comprehend_config.get("job_name_suffix", "live")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        input_location = S3Location(
            bucket=bucket,
            key=join_s3_key("aws_comprehend/input", f"ampav-aws-comprehend-{timestamp}.txt"),
        )
        client.s3_client.put_object(
            Bucket=input_location.bucket,
            Key=input_location.key,
            Body=SAMPLE_TEXT.encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        output_s3_uri = f"s3://{bucket}/{comprehend_config.get('output_prefix', 'aws_comprehend/output').strip('/')}"
        try:
            result = client.process(
                input_location.uri,
                output_s3_uri=output_s3_uri,
                language_code=comprehend_config.get("language_code", "en"),
                job_name_suffix=job_name_suffix,
            )
        finally:
            client.s3_client.delete_object(Bucket=input_location.bucket, Key=input_location.key)

        self.assertIsInstance(result.output, NamedEntities)
        assert isinstance(result.output, NamedEntities)
        entity_text = " ".join(entity.text for entity in result.output.spans)
        self.assertTrue(any(term in entity_text for term in {"Maya", "Indiana", "Amazon", "Seattle"}))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    unittest.main()
