from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ArtifactWriter:
    run_dir: Path | None

    def path(self, name: str) -> Path | None:
        if self.run_dir is None:
            return None
        return self.run_dir / name

    def write_json(self, name: str, data: Any) -> Path | None:
        path = self.path(name)
        if path is not None:
            write_json(path, data)
        return path


def create_artifact_writer(runs_dir: Path | None, timestamp: str, job_name: str) -> ArtifactWriter:
    if runs_dir is None:
        return ArtifactWriter(run_dir=None)
    return ArtifactWriter(run_dir=create_run_directory(runs_dir, timestamp, job_name))


def create_run_directory(runs_dir: Path, timestamp: str, job_name: str) -> Path:
    runs_dir = runs_dir.expanduser()
    run_name = safe_path_part(f"{timestamp}_{job_name}")[:240]
    candidate = runs_dir / run_name
    suffix = 1
    while candidate.exists():
        candidate = runs_dir / f"{run_name}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
