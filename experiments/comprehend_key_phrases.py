"""Run and evaluate native AWS Comprehend key-phrase probes.

This experiment preserves provider-native behavior without defining an AMPAV
tool API or schema. Runtime inputs, credentials, and retained outputs belong
under the caller's ``.work`` directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import shlex
import statistics
import sys
from time import perf_counter
from typing import Any

import boto3
import yaml


AWS_MAX_TEXT_BYTES = 100_000
DEFAULT_TOP_PHRASES = 30
TASK = "AMPAV-63"


def parse_args() -> argparse.Namespace:
    """Parse native-probe or retained-output analysis arguments."""
    parser = argparse.ArgumentParser(
        description="Probe native AWS Comprehend key-phrase behavior.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser(
        "probe",
        help="call native DetectKeyPhrases and retain its response",
    )
    probe.add_argument("input", type=Path, help="UTF-8 source text")
    probe.add_argument("output_dir", type=Path, help="new retained-run directory")
    probe.add_argument("--fixture-id", required=True)
    probe.add_argument("--language-code", default="en")
    probe.add_argument("--profile", help="optional boto3 profile name")
    probe.add_argument("--region", help="optional AWS region")
    probe.add_argument(
        "--include-entities",
        action="store_true",
        help="also call DetectEntities on the same text for structural comparison",
    )

    analyze = subparsers.add_parser(
        "analyze",
        help="calculate diagnostics from retained native responses",
    )
    analyze.add_argument("input", type=Path, help="UTF-8 source text")
    analyze.add_argument(
        "key_phrases",
        type=Path,
        help="complete native DetectKeyPhrases response JSON",
    )
    analyze.add_argument("output_dir", type=Path, help="analysis output directory")
    analyze.add_argument(
        "--entities",
        type=Path,
        help="optional complete native DetectEntities response JSON",
    )
    analyze.add_argument(
        "--reference-keywords",
        type=Path,
        help="optional JSON list of strings or objects containing a text field",
    )
    analyze.add_argument(
        "--top-phrases",
        type=int,
        default=DEFAULT_TOP_PHRASES,
        help=f"number of repeated native phrases to report (default: {DEFAULT_TOP_PHRASES})",
    )

    return parser.parse_args()


def main() -> None:
    """Run the selected experiment command."""
    args = parse_args()
    if args.command == "probe":
        result = run_probe(args)
    else:
        result = run_analysis(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Call native Comprehend APIs and retain complete responses and metadata."""
    _validate_new_output_dir(args.output_dir)
    text = args.input.read_text(encoding="utf-8")
    text_bytes = _validate_text(text)

    session = boto3.Session(
        region_name=args.region,
        profile_name=args.profile,
    )
    client = session.client("comprehend")
    effective_region = (
        getattr(getattr(client, "meta", None), "region_name", None)
        or args.region
        or session.region_name
    )
    request = {"Text": text, "LanguageCode": args.language_code}

    started_at = datetime.now(timezone.utc)
    started = perf_counter()
    key_phrase_response = _detect_key_phrases(client, request)
    args.output_dir.mkdir(parents=True)
    _write_json(
        args.output_dir / "native_key_phrases.json",
        key_phrase_response,
    )
    entity_response = None
    if args.include_entities:
        entity_response = _detect_entities(client, request)
        _write_json(
            args.output_dir / "native_entities.json",
            entity_response,
        )
    elapsed_seconds = perf_counter() - started
    completed_at = datetime.now(timezone.utc)

    sha256 = hashlib.sha256(text_bytes).hexdigest()
    api_count = 2 if entity_response is not None else 1
    manifest = _build_manifest(
        args,
        text=text,
        text_bytes=text_bytes,
        sha256=sha256,
        api_count=api_count,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed_seconds,
        effective_region=effective_region,
        key_phrase_response=key_phrase_response,
        entity_response=entity_response,
    )
    (args.output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (args.output_dir / "input_ref.txt").write_text(
        "\n".join(
            [
                f"fixture_id={args.fixture_id}",
                f"input={args.input.resolve()}",
                f"sha256={sha256}",
                "cleanup=No remote artifacts were created by the synchronous APIs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )

    return {
        "output_dir": str(args.output_dir.resolve()),
        "key_phrase_occurrences": len(key_phrase_response.get("KeyPhrases", [])),
        "entity_occurrences": (
            len(entity_response.get("Entities", []))
            if entity_response is not None
            else None
        ),
        "manifest": str((args.output_dir / "manifest.yaml").resolve()),
    }


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    """Analyze retained native responses and write JSON and Markdown diagnostics."""
    if args.top_phrases < 0:
        raise ValueError("top_phrases must be non-negative")
    text = args.input.read_text(encoding="utf-8")
    key_phrase_response = _load_json_mapping(args.key_phrases)
    entity_response = (
        _load_json_mapping(args.entities) if args.entities is not None else None
    )
    reference_keywords = (
        _load_reference_keywords(args.reference_keywords)
        if args.reference_keywords is not None
        else []
    )
    report = analyze_responses(
        text,
        key_phrase_response,
        entity_response=entity_response,
        reference_keywords=reference_keywords,
        top_phrases=args.top_phrases,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "analysis.json"
    markdown_path = args.output_dir / "analysis.md"
    for path in (json_path, markdown_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"analysis output already exists: {path}")
    _write_json(json_path, report)
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")

    return {
        "analysis_json": str(json_path.resolve()),
        "analysis_markdown": str(markdown_path.resolve()),
        "summary": report["summary"],
    }


def _detect_key_phrases(client: Any, request: dict[str, str]) -> dict[str, Any]:
    """Call native ``DetectKeyPhrases`` without changing its response shape."""
    response = client.detect_key_phrases(**request)
    if not isinstance(response, dict) or not isinstance(
        response.get("KeyPhrases"), list
    ):
        raise TypeError("DetectKeyPhrases response must contain a KeyPhrases list")
    return response


def _detect_entities(client: Any, request: dict[str, str]) -> dict[str, Any]:
    """Call native ``DetectEntities`` for optional structural comparison."""
    response = client.detect_entities(**request)
    if not isinstance(response, dict) or not isinstance(
        response.get("Entities"), list
    ):
        raise TypeError("DetectEntities response must contain an Entities list")
    return response


def analyze_responses(
    text: str,
    key_phrase_response: dict[str, Any],
    *,
    entity_response: dict[str, Any] | None = None,
    reference_keywords: list[dict[str, Any]] | None = None,
    top_phrases: int = DEFAULT_TOP_PHRASES,
) -> dict[str, Any]:
    """Calculate structural diagnostics without treating them as quality scores."""
    if top_phrases < 0:
        raise ValueError("top_phrases must be non-negative")
    key_phrases = _response_items(key_phrase_response, "KeyPhrases")
    entities = (
        _response_items(entity_response, "Entities")
        if entity_response is not None
        else []
    )
    reference_keywords = reference_keywords or []

    _validate_offsets(text, key_phrases, "AWS key phrases")
    _validate_offsets(text, entities, "AWS entities")

    phrase_by_text = _group_by_normalized_text(key_phrases)
    entity_by_text = _group_by_normalized_text(entities)
    phrase_spans = {_span_identity(item) for item in key_phrases}
    entity_spans = {_span_identity(item) for item in entities}
    phrase_counts = Counter(_normalize(item["Text"]) for item in key_phrases)
    display_forms = {
        _normalize(item["Text"]): item["Text"] for item in key_phrases
    }

    reference_rows = [
        _reference_comparison_row(
            keyword,
            text,
            phrase_by_text,
            entity_by_text,
        )
        for keyword in reference_keywords
    ]
    summary: dict[str, Any] = {
        "aws_key_phrase_occurrence_count": len(key_phrases),
        "aws_unique_key_phrase_text_count": len(phrase_by_text),
        "aws_key_phrase_score": _score_summary(key_phrases),
        "diagnostic_note": (
            "Counts and overlaps are structural diagnostics, not relevance or "
            "quality metrics. AWS Score is native noun-phrase detection confidence."
        ),
    }
    if entity_response is not None:
        summary.update(
            {
                "aws_entity_occurrence_count": len(entities),
                "aws_unique_entity_text_count": len(entity_by_text),
                "aws_keyphrase_entity_exact_span_overlap": len(
                    phrase_spans & entity_spans
                ),
                "aws_entity_occurrences_contained_in_a_keyphrase": sum(
                    _contained_in_any(entity, key_phrases) for entity in entities
                ),
                "aws_keyphrase_occurrences_contained_in_an_entity": sum(
                    _contained_in_any(phrase, entities) for phrase in key_phrases
                ),
                "aws_entity_types": dict(
                    sorted(
                        Counter(
                            item.get("Type", "<missing>") for item in entities
                        ).items()
                    )
                ),
            }
        )
    if reference_keywords:
        summary.update(
            {
                "reference_keyword_count": len(reference_keywords),
                "reference_keywords_found_in_source": sum(
                    row["in_source"] for row in reference_rows
                ),
                "reference_keywords_with_exact_aws_phrase": sum(
                    row["aws_phrase_exact_occurrences"] > 0
                    for row in reference_rows
                ),
                "reference_keywords_with_exact_or_containing_aws_phrase": sum(
                    row["aws_phrase_exact_occurrences"] > 0
                    or bool(row["aws_phrase_containing_variants"])
                    for row in reference_rows
                ),
                "reference_keywords_with_exact_aws_entity": sum(
                    row["aws_entity_exact_occurrences"] > 0
                    for row in reference_rows
                ),
            }
        )

    repeated = [
        {
            "text": display_forms[normalized],
            "occurrences": count,
            "mean_native_score": round(
                statistics.fmean(
                    float(item["Score"])
                    for item in phrase_by_text[normalized]
                ),
                6,
            ),
        }
        for normalized, count in phrase_counts.most_common(top_phrases)
    ]
    return {
        "summary": summary,
        "reference_keyword_comparison": reference_rows,
        "top_repeated_aws_key_phrases": repeated,
    }


def _validate_text(text: str) -> bytes:
    """Validate the documented synchronous Comprehend text boundary."""
    if not text:
        raise ValueError("input text must not be empty")
    text_bytes = text.encode("utf-8")
    if len(text_bytes) >= AWS_MAX_TEXT_BYTES:
        raise ValueError(
            f"input is {len(text_bytes)} UTF-8 bytes; DetectKeyPhrases requires "
            f"less than {AWS_MAX_TEXT_BYTES}"
        )
    return text_bytes


def _validate_new_output_dir(path: Path) -> None:
    """Reject an existing run directory so retained evidence is not overwritten."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output directory already exists: {path}")


def _build_manifest(
    args: argparse.Namespace,
    *,
    text: str,
    text_bytes: bytes,
    sha256: str,
    api_count: int,
    started_at: datetime,
    completed_at: datetime,
    elapsed_seconds: float,
    effective_region: str | None,
    key_phrase_response: dict[str, Any],
    entity_response: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build structured context for a retained native run."""
    billable_units_per_api = max(3, (len(text) + 99) // 100)
    return {
        "task": TASK,
        "experiment": "aws_comprehend_key_phrases",
        "fixture_id": args.fixture_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "native_service": "Amazon Comprehend",
        "native_operations": [
            "DetectKeyPhrases",
            *(["DetectEntities"] if entity_response is not None else []),
        ],
        "boto3_version": _package_version("boto3"),
        "botocore_version": _package_version("botocore"),
        "effective_parameters": {
            "language_code": args.language_code,
            "region": effective_region,
            "include_entities": entity_response is not None,
            "provider_max_utf8_bytes_exclusive": AWS_MAX_TEXT_BYTES,
        },
        "input": {
            "characters": len(text),
            "utf8_bytes": len(text_bytes),
            "sha256": sha256,
        },
        "estimated_billing": {
            "api_calls": api_count,
            "units_per_api": billable_units_per_api,
            "total_units": api_count * billable_units_per_api,
            "unit_definition": "100 input characters; minimum 3 units per API call",
            "currency_cost_not_recorded": "Pricing varies over time and region.",
        },
        "native_counts": {
            "key_phrase_occurrences": len(
                key_phrase_response.get("KeyPhrases", [])
            ),
            "entity_occurrences": (
                len(entity_response.get("Entities", []))
                if entity_response is not None
                else None
            ),
        },
        "cleanup": "No remote artifacts were created by the synchronous APIs.",
    }


def _response_items(response: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """Return a validated native response item list."""
    items = response.get(key)
    if not isinstance(items, list) or not all(
        isinstance(item, dict) for item in items
    ):
        raise TypeError(f"native response must contain a {key} object list")
    return items


def _validate_offsets(text: str, items: list[dict[str, Any]], label: str) -> None:
    """Verify that every native offset addresses its reported source text."""
    mismatches = []
    for index, item in enumerate(items):
        try:
            begin = item["BeginOffset"]
            end = item["EndOffset"]
            item_text = item["Text"]
        except KeyError as exc:
            raise ValueError(f"{label}[{index}] is missing {exc.args[0]}") from exc
        if (
            not isinstance(begin, int)
            or not isinstance(end, int)
            or begin < 0
            or end < begin
            or end > len(text)
            or not isinstance(item_text, str)
            or text[begin:end] != item_text
        ):
            mismatches.append(index)
    if mismatches:
        raise ValueError(
            f"{label} contains {len(mismatches)} offset mismatch(es): {mismatches[:5]}"
        )


def _group_by_normalized_text(
    items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[_normalize(item["Text"])].append(item)
    return grouped


def _span_identity(item: dict[str, Any]) -> tuple[int, int, str]:
    return (
        item["BeginOffset"],
        item["EndOffset"],
        _normalize(item["Text"]),
    )


def _contained_in_any(
    item: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> bool:
    return any(
        candidate["BeginOffset"] <= item["BeginOffset"]
        and item["EndOffset"] <= candidate["EndOffset"]
        for candidate in candidates
    )


def _score_summary(items: list[dict[str, Any]]) -> dict[str, float] | None:
    """Summarize the native ``Score`` without changing its meaning."""
    if not items:
        return None
    try:
        scores = [float(item["Score"]) for item in items]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("every AWS key phrase must have a numeric Score") from exc
    return {
        "minimum": round(min(scores), 6),
        "median": round(statistics.median(scores), 6),
        "mean": round(statistics.fmean(scores), 6),
        "maximum": round(max(scores), 6),
        "meaning": "AWS confidence that the detected span is a noun phrase",
    }


def _reference_comparison_row(
    keyword: dict[str, Any],
    text: str,
    phrase_by_text: dict[str, list[dict[str, Any]]],
    entity_by_text: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    keyword_text = keyword["text"]
    normalized = _normalize(keyword_text)
    exact_phrases = phrase_by_text.get(normalized, [])
    exact_entities = entity_by_text.get(normalized, [])
    containing_variants = sorted(
        {
            item["Text"]
            for candidate, items in phrase_by_text.items()
            if candidate != normalized
            and _contains_token_sequence(candidate, normalized)
            for item in items
        },
        key=str.casefold,
    )
    literal_occurrences = _literal_occurrences(text, keyword_text)
    return {
        "reference_keyword": keyword_text,
        "reference_instance_count": keyword.get("instance_count"),
        "in_source": literal_occurrences > 0,
        "source_literal_occurrences": literal_occurrences,
        "aws_phrase_exact_occurrences": len(exact_phrases),
        "aws_phrase_containing_variants": containing_variants,
        "aws_entity_exact_occurrences": len(exact_entities),
        "aws_entity_types": sorted(
            {item["Type"] for item in exact_entities if "Type" in item}
        ),
    }


def _contains_token_sequence(container: str, phrase: str) -> bool:
    """Match complete normalized tokens, avoiding substring matches such as chi/chief."""
    container_tokens = container.split()
    phrase_tokens = phrase.split()
    width = len(phrase_tokens)
    return bool(phrase_tokens) and any(
        container_tokens[index : index + width] == phrase_tokens
        for index in range(len(container_tokens) - width + 1)
    )


def _literal_occurrences(source: str, phrase: str) -> int:
    """Count case-insensitive literal phrases at non-word boundaries."""
    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return len(re.findall(pattern, source, flags=re.IGNORECASE))


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _load_reference_keywords(path: Path) -> list[dict[str, Any]]:
    """Load a simple external keyword list without assuming a provider schema."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise TypeError("reference keyword JSON must be a list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            text = item
            instance_count = None
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text = item["text"]
            instances = item.get("instances")
            instance_count = len(instances) if isinstance(instances, list) else None
        else:
            raise TypeError(
                f"reference keyword {index} must be a string or object with text"
            )
        if not text.strip():
            raise ValueError(f"reference keyword {index} has empty text")
        result.append({"text": text, "instance_count": instance_count})
    return result


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# AWS Comprehend Key-Phrase Diagnostics",
        "",
        "> These are structural diagnostics, not relevance or quality metrics. "
        "AWS `Score` is confidence that a detected span is a noun phrase.",
        "",
        "## Summary",
        "",
        "| Measure | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            lines.append(f"| {key.replace('_', ' ')} | {value} |")

    reference_rows = report["reference_keyword_comparison"]
    if reference_rows:
        lines.extend(
            [
                "",
                "## Reference-keyword comparison",
                "",
                "| Reference keyword | Source occurrences | AWS exact phrases | "
                "AWS containing phrases | AWS exact entities |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in reference_rows:
            variants = ", ".join(row["aws_phrase_containing_variants"]) or "-"
            entity = str(row["aws_entity_exact_occurrences"])
            if row["aws_entity_types"]:
                entity += f" ({', '.join(row['aws_entity_types'])})"
            lines.append(
                f"| {row['reference_keyword']} | "
                f"{row['source_literal_occurrences']} | "
                f"{row['aws_phrase_exact_occurrences']} | {variants} | {entity} |"
            )

    lines.extend(
        [
            "",
            "## Most repeated AWS key phrases",
            "",
            "| Phrase | Occurrences | Mean native score |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in report["top_repeated_aws_key_phrases"]:
        lines.append(
            f"| {row['text']} | {row['occurrences']} | "
            f"{row['mean_native_score']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
