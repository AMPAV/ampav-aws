"""Shared helpers for ampav-aws examples."""

from pathlib import Path
from typing import Any

import yaml


EXAMPLES_DIR = Path(__file__).resolve().parent
CONFIG_PATH = EXAMPLES_DIR / "config" / "aws_config.yaml"
DATA_DIR = EXAMPLES_DIR / "data"


def load_config() -> dict[str, Any]:
    """Load the local example config copied from config/aws_config.example.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Expected local config at {CONFIG_PATH}. "
            "Copy config/aws_config.example.yaml to config/aws_config.yaml and update it first."
        )
    data = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {CONFIG_PATH}")
    return data


def write_tool_output(filename: str, output: Any) -> Path:
    """Write a ToolOutput YAML file under examples/data."""
    path = DATA_DIR / filename
    path.write_text(output.model_dump_yaml(sort_keys=False))
    return path
