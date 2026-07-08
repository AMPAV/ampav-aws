# ampav-aws

AWS tooling for AMPAV.

## AWS Transcribe

`ampav-aws` provides small AWS clients plus CLI/example code. AWS Transcribe
returns AMPAV `ToolOutput` objects whose `output` is an AMPAV `Transcript`.

The library API is job-oriented: submit a job, wait for completion, fetch the
result, and clean up AWS-side job records/resources owned by the tool. Library
methods expect provider-native inputs such as `s3://bucket/key`. Local file
upload, config loading, and artifact persistence are client concerns handled by
CLI, examples, or application code.

## Python API

Use an existing S3 media object:

```python
from ampav.aws.transcribe import AwsTranscribe, TranscriptionSettings

client = AwsTranscribe(region_name="us-east-2", profile_name="my-profile")
result = client.process(
    "s3://my-bucket/input/audio.wav",
    output_s3_uri="s3://my-bucket/output/audio.json",
    transcription_settings=TranscriptionSettings(language_code="en-US"),
)

print(result.output.text)
```

For lower-level job lifecycle control, use `AwsTranscribe` directly and call
`submit()`, `get_status()`, `list_jobs()`, `get_result()`, and `cleanup()`.
`submit()` accepts media that already exists in S3 and returns an opaque AWS job
ID string. `process()` is the high-level blocking path for provider-native
inputs.

Pass `include_tool_private=True` when constructing the tool only when you need
raw AWS job/transcript data for troubleshooting. Normal client code should use
`ToolOutput.output`.

## CLI

The CLI is a thin wrapper over the Python API:

```bash
ampav_aws_transcribe -h
ampav_aws_comprehend_named_entities -h
```

```bash
ampav_aws_transcribe s3://my-bucket/input/audio.wav \
  --output-s3-uri s3://my-bucket/output/audio.json \
  --region us-east-2
```

For local files:

```bash
ampav_aws_transcribe examples/data/AMP-Intro.m4a \
  --input-bucket my-bucket \
  --input-prefix aws_transcribe/input \
  --output-s3-uri s3://my-bucket/aws_transcribe/output/AMP-Intro.json \
  --region us-east-2
```

For Comprehend named entities from a local text file:

```bash
ampav_aws_comprehend_named_entities input.txt \
  --input-bucket my-bucket \
  --output-s3-uri s3://my-bucket/aws_comprehend_named_entities/output \
  --data-access-role-arn arn:aws:iam::123456789012:role/ComprehendDataAccess \
  --region us-east-2
```

Do not put AWS secret keys on the command line. Use boto3-native auth:

- AWS profile via `--profile`
- AWS region via `--region`
- environment variables
- `~/.aws/config` and `~/.aws/credentials`
- IAM role credentials where available

If a CLI uploads a local input file or text object, it deletes that uploaded input by
default. Pass `--keep-input` to keep it. Caller-supplied output is kept by
default; pass `--delete-user-owned-outputs` to remove it after retrieval.

The library always attempts provider job cleanup after terminal result
retrieval.

## Examples

Config loading and local artifact persistence are client concerns, not library
defaults. For local testing, copy `examples/` outside the repo, copy
`config/aws_config.example.yaml` to `config/aws_config.yaml`, update the copied
config for your AWS account, then run the copied scripts.

- `aws_transcribe_file_example.py`: upload `data/AMP-Intro.m4a`, transcribe it using copied config, and write a `ToolOutput` YAML file.
- `aws_transcribe_s3_example.py`: transcribe an existing `s3://` media URI using standard boto3 profile/region settings and write a `ToolOutput` YAML file.
- `aws_comprehend_named_entities_transcript_example.py`: read `data/AMP-Intro-Transcript.yaml`, extract named entities using copied config, and write a `ToolOutput` YAML file.
- `aws_comprehend_named_entities_s3_example.py`: extract named entities from an existing `s3://` text object using standard boto3 profile/region settings and write a `ToolOutput` YAML file.
- `config/aws_config.example.yaml`: sample shared AWS config for examples.
- `data/`: small curated inputs and checked-in example outputs.

Keep real credentials, local configs, generated logs, and ad hoc run outputs in
`.work/`, not in git.

## Tests

Routine tests are offline and deterministic:

```bash
python -m unittest discover -s tests
```

An optional live AWS smoke test is skipped by default:

```bash
AMPAV_AWS_TRANSCRIBE_LIVE_TEST=1 \
AMPAV_AWS_TRANSCRIBE_CONFIG=/path/to/aws_config.yaml \
python -m unittest discover -s tests
```
