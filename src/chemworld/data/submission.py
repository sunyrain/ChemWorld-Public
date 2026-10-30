"""Submission manifest helpers for reproducible benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from chemworld import __version__
from chemworld.data.logging import load_jsonl
from chemworld.data.validation import validate_records

SUBMISSION_SCHEMA_VERSION = "chemworld-submission-0.1"
SUBMISSION_BUNDLE_SCHEMA_VERSION = "chemworld-submission-bundle-0.1"


@dataclass(frozen=True)
class SubmissionManifest:
    schema_version: str
    chemworld_version: str
    trajectory_path: str
    agent_manifest: dict[str, Any]
    command: list[str]
    dependency_versions: dict[str, str]
    platform: dict[str, str]
    source_digest: str | None
    created_at: str
    notes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "chemworld_version": self.chemworld_version,
            "trajectory_path": self.trajectory_path,
            "agent_manifest": self.agent_manifest,
            "command": self.command,
            "dependency_versions": self.dependency_versions,
            "platform": self.platform,
            "source_digest": self.source_digest,
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SubmissionBundleValidation:
    valid: bool
    path: str
    errors: list[str]
    trajectory_count: int
    result_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "path": self.path,
            "errors": self.errors,
            "trajectory_count": self.trajectory_count,
            "result_count": self.result_count,
        }


def installed_versions(packages: list[str] | None = None) -> dict[str, str]:
    packages = packages or [
        "chemworld-bench",
        "gymnasium",
        "numpy",
        "pandas",
        "scikit-learn",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def current_platform() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def git_commit() -> str | None:
    """Return this installation's source commit only for an actual source checkout."""
    package = Path(__file__).resolve().parents[1]
    root = package.parent.parent
    if package.parent.name != "src" or not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def source_digest(root: str | Path | None = None) -> str | None:
    """Hash installed package code/resources, excluding caches and repository documents.

    An explicit root may be a package directory or a source checkout containing
    src/chemworld. The default is the package that is actually executing.
    """
    package = Path(__file__).resolve().parents[1] if root is None else Path(root)
    if (package / "src" / "chemworld").is_dir():
        package = package / "src" / "chemworld"
    if not (package / "__init__.py").is_file():
        return None
    paths = sorted(path for path in package.rglob("*") if path.is_file()
                   and "__pycache__" not in path.relative_to(package).parts
                   and path.suffix not in {".pyc", ".pyo"})
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.relative_to(package).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
        hasher.update(b"\0")
    return hasher.hexdigest()


def _relative_dependency_path(root: Path, dependency_file: str) -> Path:
    dependency_path = Path(dependency_file)
    if dependency_path.is_absolute():
        return dependency_path
    return root / dependency_path


def _write_dependency_notes(
    root: Path,
    *,
    dependency_file: str,
    command: list[str],
) -> None:
    dependency_path = _relative_dependency_path(root, dependency_file)
    dependency_path.parent.mkdir(parents=True, exist_ok=True)
    if dependency_path.name == "pyproject.toml":
        # A wheel does not contain the repository's development project file.
        # Record an installable environment for this submitted artifact instead.
        dependency_path.write_text(
            '[project]\nname = "chemworld-submission"\nversion = "0.0.0"\n'
            f'dependencies = ["chemworld-bench=={__version__}"]\n', encoding="utf-8",
        )
        return

    dependency_lines = [
        "# Dependency Notes",
        "",
        "This file records the minimum local dependency context for reproducing this",
        "ChemWorld submission bundle.",
        "",
        "## Reproducible Command",
        "",
        "```bash",
        " ".join(command) if command else "chemworld submission example <bundle-path>",
        "```",
        "",
        "## Platform",
        "",
    ]
    for key, value in sorted(current_platform().items()):
        dependency_lines.append(f"- {key}: {value}")
    dependency_lines.extend(["", "## Installed Package Versions", ""])
    for key, value in sorted(installed_versions().items()):
        dependency_lines.append(f"- {key}: {value}")
    dependency_lines.extend(["", f"- git_commit: {git_commit()}", ""])
    dependency_path.write_text("\n".join(dependency_lines), encoding="utf-8")


def _write_bundle_readme(
    root: Path,
    *,
    task_id: str,
    agent_name: str,
    seeds: list[int],
    command: list[str],
) -> None:
    trajectory_name = f"{agent_name}_{task_id}_seed{seeds[0]}.jsonl"
    readme = f"""# ChemWorld Submission Bundle Example

This bundle is a valid local submission example for ChemWorld-Bench.

## Contents

- `manifest.json`: submission metadata, task id, seeds, dependency file, and command.
- `trajectories/*.jsonl`: replayable ChemWorld trajectory logs.
- `results/*.json`: local metric output generated from the trajectories.
- `explanations/*.json`: structured public explanation notes.
- `dependency_notes.md`: package versions and reproducibility notes.

## Reproduce

```bash
{" ".join(command)}
```

## Validate

```bash
chemworld submission validate <bundle-path>
chemworld submission summarize <bundle-path>
chemworld verify --constitution \\
  --submission <bundle-path>/trajectories/{trajectory_name}
```
"""
    (root / "README.md").write_text(readme, encoding="utf-8")


def build_submission_manifest(
    *,
    trajectory_path: str | Path,
    agent_manifest: dict[str, Any],
    command: list[str],
    notes: dict[str, Any] | None = None,
) -> SubmissionManifest:
    enriched_agent_manifest = dict(agent_manifest)
    enriched_agent_manifest.setdefault("git_commit", git_commit())
    return SubmissionManifest(
        schema_version=SUBMISSION_SCHEMA_VERSION,
        chemworld_version=__version__,
        trajectory_path=str(trajectory_path),
        agent_manifest=enriched_agent_manifest,
        command=command,
        dependency_versions=installed_versions(),
        platform=current_platform(),
        source_digest=source_digest(),
        created_at=datetime.now(UTC).isoformat(),
        notes=notes or {},
    )


def write_submission_manifest(
    *,
    path: str | Path,
    trajectory_path: str | Path,
    agent_manifest: dict[str, Any],
    command: list[str],
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = build_submission_manifest(
        trajectory_path=trajectory_path,
        agent_manifest=agent_manifest,
        command=command,
        notes=notes,
    ).to_dict()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def init_submission_bundle(
    path: str | Path,
    *,
    agent_name: str,
    agent_family: str,
    task_id: str,
    seeds: list[int],
    command: list[str],
    dependency_file: str,
    llm_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a local submission bundle skeleton and manifest."""

    root = Path(path)
    (root / "trajectories").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "explanations").mkdir(parents=True, exist_ok=True)
    _write_dependency_notes(root, dependency_file=dependency_file, command=command)
    manifest = {
        "schema_version": SUBMISSION_BUNDLE_SCHEMA_VERSION,
        "agent_name": agent_name,
        "agent_family": agent_family,
        "platform_version": __version__,
        "commit_hash": git_commit(),
        "dependency_file": dependency_file,
        "command": command,
        "task_id": task_id,
        "seeds": seeds,
        "llm_metadata": llm_metadata or {},
        "created_at": datetime.now(UTC).isoformat(),
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    _write_bundle_readme(
        root,
        task_id=task_id,
        agent_name=agent_name,
        seeds=seeds,
        command=command,
    )
    return manifest


def _load_bundle_manifest(root: Path, errors: list[str]) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        errors.append("missing manifest.json")
        return {}
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        errors.append(f"manifest.json is not valid JSON: {exc}")
        return {}
    required = {
        "schema_version",
        "agent_name",
        "agent_family",
        "platform_version",
        "commit_hash",
        "dependency_file",
        "command",
        "task_id",
        "seeds",
        "llm_metadata",
    }
    missing = sorted(required - manifest.keys())
    if missing:
        errors.append(f"manifest.json missing keys: {missing}")
    if manifest.get("schema_version") != SUBMISSION_BUNDLE_SCHEMA_VERSION:
        errors.append("manifest.json has unsupported schema_version")
    command = manifest.get("command")
    if not isinstance(command, list) or not command:
        errors.append("manifest.json command must be a non-empty list")
    seeds = manifest.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("manifest.json seeds must be a non-empty list")
    dependency_file = manifest.get("dependency_file")
    if isinstance(dependency_file, str) and dependency_file:
        dependency_path = _relative_dependency_path(root, dependency_file)
        if not dependency_path.exists():
            errors.append(f"dependency file is missing: {dependency_file}")
    elif "dependency_file" in manifest:
        errors.append("manifest.json dependency_file must be a non-empty string")
    return manifest


def validate_submission_bundle(path: str | Path) -> SubmissionBundleValidation:
    """Validate a local submission bundle structure and trajectory schemas."""

    root = Path(path)
    errors: list[str] = []
    if not root.exists():
        return SubmissionBundleValidation(False, str(root), ["bundle path does not exist"], 0, 0)
    manifest = _load_bundle_manifest(root, errors)
    if not (root / "README.md").exists():
        errors.append("missing README.md")

    trajectory_dir = root / "trajectories"
    result_dir = root / "results"
    explanation_dir = root / "explanations"
    if not trajectory_dir.exists():
        errors.append("missing trajectories directory")
        trajectory_paths: list[Path] = []
    else:
        trajectory_paths = sorted(trajectory_dir.glob("*.jsonl"))
        if not trajectory_paths:
            errors.append("trajectories directory contains no .jsonl files")
    if not result_dir.exists():
        errors.append("missing results directory")
        result_paths: list[Path] = []
    else:
        result_paths = sorted(result_dir.glob("*.json"))
        if not result_paths:
            errors.append("results directory contains no .json files")
    if not explanation_dir.exists():
        errors.append("missing explanations directory")
        explanation_paths: list[Path] = []
    else:
        explanation_paths = sorted(explanation_dir.glob("*.json"))

    if trajectory_paths and result_paths:
        trajectory_stems = {path.stem for path in trajectory_paths}
        result_stems = {path.stem for path in result_paths}
        missing_results = sorted(trajectory_stems - result_stems)
        if missing_results:
            errors.append(f"missing result files for trajectories: {missing_results}")

    for trajectory_path in trajectory_paths:
        try:
            validate_records(load_jsonl(trajectory_path))
        except ValueError as exc:
            errors.append(f"{trajectory_path.name}: {exc}")

    for result_path in result_paths:
        try:
            with result_path.open("r", encoding="utf-8") as handle:
                result = json.load(handle)
        except json.JSONDecodeError as exc:
            errors.append(f"{result_path.name}: invalid JSON: {exc}")
            continue
        if "total_score" not in result:
            errors.append(f"{result_path.name}: missing total_score")
        if manifest and result.get("benchmark_task_id") not in {None, manifest.get("task_id")}:
            errors.append(f"{result_path.name}: benchmark_task_id does not match manifest task_id")

    for explanation_path in explanation_paths:
        try:
            with explanation_path.open("r", encoding="utf-8") as handle:
                explanation = json.load(handle)
        except json.JSONDecodeError as exc:
            errors.append(f"{explanation_path.name}: invalid JSON: {exc}")
            continue
        if not isinstance(explanation, dict):
            errors.append(f"{explanation_path.name}: explanation must be a JSON object")
            continue
        for key in ("hypothesis", "learned_mechanism", "failure_analysis"):
            if key not in explanation:
                errors.append(f"{explanation_path.name}: missing {key}")

    return SubmissionBundleValidation(
        valid=not errors,
        path=str(root),
        errors=errors,
        trajectory_count=len(trajectory_paths),
        result_count=len(result_paths),
    )
