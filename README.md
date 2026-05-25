# ampav-aws

AWS tooling for AMPAV.

## AWS Transcribe

`ampav-aws` provides a small AWS Transcribe client plus a CLI. It returns AMPAV
`ToolOutput` objects whose `output` is an AMPAV `Transcript`.

The library API is job-oriented: submit a job, wait for completion, fetch the
transcript, and optionally clean up AWS-side resources. It supports both existing
`s3://bucket/key` media and local files that need to be uploaded first.

## Python API

Use an existing S3 media object:

```python
from ampav.aws.transcribe import TranscriptionSettings, transcribe_uri

result = transcribe_uri(
    "s3://my-bucket/input/audio.wav",
    output_bucket="my-bucket",
    output_key="output/audio.json",
    transcription=TranscriptionSettings(language_code="en-US"),
    region_name="us-east-2",
    profile_name="my-profile",
)

print(result.output.text)
```

Use a local file:

```python
from pathlib import Path

from ampav.aws.transcribe import transcribe_file

result = transcribe_file(
    Path("tests/fixtures/OpenDoor.wav"),
    output_bucket="my-bucket",
    input_prefix="aws_transcribe/input",
    output_prefix="aws_transcribe/output",
    region_name="us-east-2",
)
```

For lower-level job lifecycle control, use `AwsTranscribe` directly and call
`submit()`, `submit_file()`, `wait()`, `get_transcription()`, and `cleanup()`.
`submit()` returns an `AwsTranscribeJob` with the AWS job name and S3 locations.

`ToolOutput.tool_private` contains raw AWS job/transcript data for
troubleshooting. Normal client code should use `ToolOutput.output`.

## CLI

The CLI is a thin wrapper over the Python API:

```bash
ampav_aws_transcribe -h
```

```bash
ampav_aws_transcribe s3://my-bucket/input/audio.wav \
  --output-bucket my-bucket \
  --output-key output/audio.json \
  --region us-east-2
```

For local files:

```bash
ampav_aws_transcribe tests/fixtures/OpenDoor.wav \
  --output-bucket my-bucket \
  --input-prefix aws_transcribe/input \
  --output-prefix aws_transcribe/output \
  --region us-east-2
```

Do not put AWS secret keys on the command line. Use boto3-native auth:

- AWS profile via `--profile`
- AWS region via `--region`
- environment variables
- `~/.aws/config` and `~/.aws/credentials`
- IAM role credentials where available

Cleanup flags are explicit and off by default:

- `--delete-job`
- `--delete-input`
- `--delete-output`

## Examples

Config loading and local artifact persistence are client concerns, not library
defaults. See `examples/` for patterns:

- `aws_transcribe_from_s3.py`: transcribe existing `s3://` media.
- `aws_transcribe_local_file.py`: upload a local file, then transcribe it.
- `aws_transcribe_with_yaml_config.py`: keep structured config in client code.
- `aws_transcribe_save_artifacts.py`: persist selected run artifacts outside the library.
- `aws_transcribe_config.example.yaml`: sample config for the YAML example.

Keep real credentials, local configs, and generated outputs outside git.

## Tests

Routine tests are offline and deterministic:

```bash
/home/yingfeng/AMPAV/.venv/bin/python -m unittest discover -s tests
```

An optional live AWS smoke test is skipped by default:

```bash
AMPAV_AWS_TRANSCRIBE_LIVE_TEST=1 \
AMPAV_AWS_TRANSCRIBE_CONFIG=/path/to/aws_transcribe_config.yaml \
/home/yingfeng/AMPAV/.venv/bin/python -m unittest discover -s tests
```
