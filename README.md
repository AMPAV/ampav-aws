# ampav-aws_transcribe

AWS Transcribe tooling for AMPAV.

## Usage

Create a real config from `examples/aws_config.example.yaml`, place it in a local runtime directory outside this repo, then run:

```bash
ampav_aws_transcribe PATH_TO_AUDIO_FILE --config PATH_TO_LOCAL_AWS_CONFIG_YAML
```

Each execution creates a timestamped run directory under the configured `paths.runs_dir` containing logs, status metadata, request metadata, and the downloaded raw AWS transcript JSON.

Do not put real AWS credentials on the command line. Use a local ignored config file, an AWS profile, or the normal boto3 credential chain.

Runtime data belongs outside this repo in a local developer workspace:

```text
PATH_TO_LOCAL_RUNTIME_WORKSPACE/
  config/
  input/
  runs/
```