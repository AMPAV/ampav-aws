import os
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ampav.aws.comprehend import AwsComprehend
from ampav.aws.s3 import join_s3_key
from ampav_aws_utils.s3_files import delete_object, upload_text


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
            polling_interval=polling_config.get("polling_interval", polling_config.get("interval_seconds", 30)),
            timeout=polling_config.get("timeout", polling_config.get("timeout_seconds", 7200)),
        )
        job_name_prefix = comprehend_config.get("job_name_prefix", "ampav-aws-comprehend-live")
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        input_location = upload_text(
            client.s3_client,
            SAMPLE_TEXT,
            bucket=bucket,
            key=join_s3_key(comprehend_config.get("input_prefix", "aws_comprehend/input"), f"{job_name_prefix}-{timestamp}.txt"),
        )
        output_s3_uri = f"s3://{bucket}/{comprehend_config.get('output_prefix', 'aws_comprehend/output').strip('/')}"
        try:
            result = client.process(
                input_location.uri,
                output_s3_uri=output_s3_uri,
                delete_output=True,
                language_code=comprehend_config.get("language_code", "en"),
                input_format=comprehend_config.get("input_format", "ONE_DOC_PER_FILE"),
                job_name_prefix=job_name_prefix,
            )
        finally:
            delete_object(client.s3_client, input_location)

        records = result.tool_private["raw_records"]
        self.assertGreaterEqual(len(records), 1)
        self.assertIn("Entities", records[0])
        entity_text = " ".join(entity["Text"] for entity in records[0]["Entities"])
        self.assertTrue(any(term in entity_text for term in {"Maya", "Indiana", "Amazon", "Seattle"}))


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


if __name__ == "__main__":
    unittest.main()
