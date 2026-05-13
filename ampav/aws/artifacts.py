"""Optional local artifact persistence for AMPAV AWS runs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import AWSArtifactError

# BDW: ArtifactWriter should just be regular class -- it's not just passing
# structured data around as a unit.  You also have a constructor that's outside
# the class.  A more streamlined implemenation may be:

class BDWArtifactWriter:
    def __init__(self, run_dir: Path | None, timestamp: str = None, job_name: str = None):
        self.run_dir: Path = run_dir
        if run_dir is not None:
            self.run_dir=create_run_directory(run_dir, timestamp, job_name)

    # there's no need for the path method, because you can easily derive it
    # via:   self.run_dir / "some_filename" and it's only used internally.

    def write_json(self, name: str, data: Any) -> Path | None:
        ...



@dataclass
class ArtifactWriter:
    """Write optional per-run debug artifacts into a local directory.

    :param run_dir: Directory for one run's artifacts. Optional; when ``None``,
        writes are disabled.
    :type run_dir: Path | None
    """

    run_dir: Path | None

    def path(self, name: str) -> Path | None:
        """Return an artifact path, or None when artifact persistence is disabled.

        :param name: Artifact filename relative to ``run_dir``.
        :type name: str
        :return: Artifact path or ``None``.
        :rtype: Path | None
        """
        if self.run_dir is None:
            return None
        return self.run_dir / name

    def write_json(self, name: str, data: Any) -> Path | None:
        """Write JSON artifact data when this writer has a run directory.

        :param name: Artifact filename relative to ``run_dir``.
        :type name: str
        :param data: JSON-serializable value to write.
        :type data: Any
        :return: Written artifact path, or ``None`` when persistence is disabled.
        :rtype: Path | None
        :raises AWSArtifactError: If the file cannot be written.
        """
        # BDW as mentioned above you don't need path so this can be written:
        #if self.path:
        #    p = selfpath / name
        #    write_json(p, data)
        #    return p
        # None is the default return value....so no need to return it explicitly        

        path = self.path(name)
        if path is not None:
            write_json(path, data)
        return path


def create_artifact_writer(runs_dir: Path | None, timestamp: str, job_name: str) -> ArtifactWriter:
    """Create a writer for a new run directory or a no-op writer.

    :param runs_dir: Parent directory for run artifacts. Optional; when
        ``None``, artifact persistence is disabled.
    :type runs_dir: Path | None
    :param timestamp: Timestamp string used in the run directory name.
    :type timestamp: str
    :param job_name: AWS job name used in the run directory name.
    :type job_name: str
    :return: Artifact writer for the run.
    :rtype: ArtifactWriter
    """
    if runs_dir is None:
        return ArtifactWriter(run_dir=None)
    return ArtifactWriter(run_dir=create_run_directory(runs_dir, timestamp, job_name))

# BDW: this is really a function that should be left to the user.  You're making
# a ton of assumptions on what the output directory should look like and be
# named.  Let the user specify where they want the data to be placed and let
# them create the directory for you.  

def create_run_directory(runs_dir: Path, timestamp: str, job_name: str) -> Path:
    """Create a unique timestamped run artifact directory.

    :param runs_dir: Parent directory for all run artifacts.
    :type runs_dir: Path
    :param timestamp: Timestamp string used in the run directory name.
    :type timestamp: str
    :param job_name: AWS job name used in the run directory name.
    :type job_name: str
    :return: Newly created run directory.
    :rtype: Path
    """
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
    """Sanitize a string for use in local path names.

    :param value: Raw string to sanitize.
    :type value: str
    :return: Sanitized path component.
    :rtype: str
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")


def write_json(path: Path, data: Any) -> None:
    """Write JSON data to a local artifact path.

    :param path: Destination JSON file.
    :type path: Path
    :param data: JSON-serializable value to write.
    :type data: Any
    :raises AWSArtifactError: If the file cannot be written.
    """
    try:
        # utf-8 is the default encoding, so no need to specify it.  No need to
        # append a newline.  
        path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        # alternately, it can be written like this which is more idiomatic:
        #with open(path, "w") as f:
        #    json.dump(data, f, indent=2)
    except OSError as exc:
        # BDW: If it were me, I'd just let the error propigate upward as-is and
        # skip the try/except block altogether.
        raise AWSArtifactError(f"Could not write artifact {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    """Read JSON data from a local artifact path.

    :param path: Source JSON file.
    :type path: Path
    :return: Parsed JSON value.
    :rtype: Any
    :raises AWSArtifactError: If the file cannot be read or parsed.
    """
    # BDW: let the individual exceptions work their way up and skip the
    # try/except block entirely
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AWSArtifactError(f"Could not read artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AWSArtifactError(f"Could not parse JSON artifact {path}: {exc}") from exc
