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
    transcription=TranscriptionSettings(language_code="en-US"),
)

print(result.output.text)
```

For lower-level job lifecycle control, use `AwsTranscribe` directly and call
`submit()`, `get_status()`, `list_jobs()`, `get_result()`, and `cleanup()`.
`submit()` accepts media that already exists in S3 and returns an opaque AWS job
ID string. `process()` is the high-level blocking path for provider-native
inputs.

`ToolOutput.tool_private` contains raw AWS job/transcript data for
troubleshooting. Normal client code should use `ToolOutput.output`.

## CLI

The CLI is a thin wrapper over the Python API:

```bash
ampav_aws_transcribe -h
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

Do not put AWS secret keys on the command line. Use boto3-native auth:

- AWS profile via `--profile`
- AWS region via `--region`
- environment variables
- `~/.aws/config` and `~/.aws/credentials`
- IAM role credentials where available

If the CLI uploads a local input file, it deletes that uploaded input by
default. Pass `--keep-input` to keep it. Caller-supplied output is kept by
default; pass `--delete-output` to remove it after retrieval.

The library always attempts provider job cleanup after terminal result
retrieval.

## Examples

Config loading and local artifact persistence are client concerns, not library
defaults. See `examples/` for patterns:

- `aws_transcribe_from_s3.py`: transcribe existing `s3://` media.
- `aws_transcribe_local_file.py`: upload a local file, then transcribe it.
- `aws_transcribe_with_yaml_config.py`: keep structured config in client code.
- `aws_transcribe_save_artifacts.py`: persist selected run artifacts outside the library.
- `aws_config.example.yaml`: sample shared AWS config for examples.
- `examples/data/`: small curated example data.

Keep real credentials, local configs, generated logs, and ad hoc run outputs in
`.work/`, not in git.

## Tests

Routine tests are offline and deterministic:

```bash
/home/yingfeng/AMPAV/.venv/bin/python -m unittest discover -s tests
```

An optional live AWS smoke test is skipped by default:

```bash
AMPAV_AWS_TRANSCRIBE_LIVE_TEST=1 \
AMPAV_AWS_TRANSCRIBE_CONFIG=/path/to/aws_config.yaml \
/home/yingfeng/AMPAV/.venv/bin/python -m unittest discover -s tests
```
