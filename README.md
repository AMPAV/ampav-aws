# ampav-aws

AWS tooling for AMPAV.

## AWS Transcribe

AWS Transcribe support is available as both a Python API and a CLI. The library
returns an AMPAV `ToolOutput` whose `output` is an AMPAV `Transcript`.

### CLI usage

Create a real config from `examples/aws_config.example.yaml`, place it outside
the repo or in another ignored local runtime directory, then run:

```bash
ampav_aws_transcribe PATH_TO_AUDIO_FILE --config PATH_TO_LOCAL_AWS_CONFIG_YAML
```

Use `--debug` to include AWS request/response metadata in logs:

```bash
ampav_aws_transcribe tests/fixtures/OpenDoor.wav --config /path/to/aws_config.yaml --debug
```

Do not put real AWS credentials on the command line. Use a local config file, an
AWS profile, or the normal boto3 credential chain. For notebook users, a local
YAML config file is the simplest workflow; keep real configs out of git and
restrict local file permissions when possible.

### Python API

```python
from pathlib import Path

from ampav.aws.transcribe import load_config, transcribe_file_with_config

config = load_config(Path("/path/to/aws_config.yaml"))
result = transcribe_file_with_config(Path("tests/fixtures/OpenDoor.wav"), config=config)

print(result.output.text)
```

The returned `ToolOutput` includes:

- `tool_name`: `aws_transcribe`
- `parameters`: input path, AWS job name, S3 input/output locations, and optional artifact paths
- `messages`: log messages emitted during the run
- `output`: normalized AMPAV `Transcript` with text, words, paragraphs, speaker labels, and AWS confidence in word `tool_specific`

### Runtime artifacts

By default, local artifact persistence is disabled:

```yaml
paths:
  runs_dir: null
```

When `paths.runs_dir` is configured, each execution creates a timestamped run
directory containing debug artifacts such as:

- `aws_transcribe.log`
- `request.json`
- `start_response.json`
- `status_history.json`
- `transcription_job.json`
- `aws_transcript.json`
- `run_result.json`

These artifacts are for troubleshooting. External clients and notebooks should
decide when and where to persist the returned `ToolOutput`.

Runtime data belongs outside this repo in a local developer workspace when used:


```text
PATH_TO_LOCAL_RUNTIME_WORKSPACE/
  config/
  input/
  runs/
```

### Tests

Routine tests are offline and deterministic:

```bash
AMPAV/.venv/bin/python -m unittest discover -s tests
```

An optional live AWS smoke test uses `tests/fixtures/OpenDoor.wav` and is skipped by
default. Run it only when you intentionally want to submit one AWS Transcribe
job:

```bash
AMPAV_AWS_TRANSCRIBE_LIVE_TEST=1 \
AMPAV_AWS_TRANSCRIBE_CONFIG=/path/to/aws_config.yaml \
/home/yingfeng/AMPAV/.venv/bin/python -m unittest discover -s tests
```

The live test validates the current AWS transcript contract before converting
the raw AWS JSON into AMPAV schema.
