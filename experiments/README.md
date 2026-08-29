# Experiments

This directory contains preliminary provider-native probes. They are not part
of the `ampav.aws` public API and do not define AMPAV output schemas.

## AWS Comprehend key phrases

`comprehend_key_phrases.py` preserves the experiment for AWS Comprehend
`DetectKeyPhrases`. The provider describes each result as a noun phrase and its
`Score` as confidence in that detection. The score is not document relevance,
rank, subject confidence, or topic confidence.

The initial real-transcript comparison found high-volume noun-phrase
occurrences rather than a selective content-keyword list. Integration is
deferred. The probe remains available if a concrete noun-phrase or search-index
use case warrants reassessment.

Run the native key-phrase API and optionally the native entity API on the same
text:

```bash
../.venv/bin/python experiments/comprehend_key_phrases.py probe \
  ../.work/ampav-aws/data/input.txt \
  ../.work/ampav-aws/runs/comprehend-key-phrases-input-native \
  --fixture-id input \
  --profile PROFILE \
  --region REGION \
  --include-entities
```

The synchronous API requires non-empty text below 100,000 UTF-8 bytes. The
probe writes complete native JSON responses and a manifest to a new output
directory. Authentication uses the normal boto3 credential chain; credentials
and local configuration are not accepted as retained metadata.

Analyze retained output and optionally compare it with an external list of
provider-selected keywords:

```bash
../.venv/bin/python experiments/comprehend_key_phrases.py analyze \
  ../.work/ampav-aws/data/input.txt \
  ../.work/ampav-aws/runs/comprehend-key-phrases-input-native/native_key_phrases.json \
  ../.work/ampav-aws/runs/comprehend-key-phrases-input-analysis \
  --entities ../.work/ampav-aws/runs/comprehend-key-phrases-input-native/native_entities.json \
  --reference-keywords ../.work/ampav-aws/data/reference-keywords.json
```

Reference-keyword JSON may be a list of strings or objects containing `text`.
An optional `instances` list is used only to report an occurrence count. The
calculated counts and overlaps are structural diagnostics, not accuracy or
quality measurements.
