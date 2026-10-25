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
from chemworld.data.records import trajectory_value
from chemworld.data.validation import validate_records
from chemworld.tasks import get_task

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
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def source_digest(root: str | Path | None = None) -> str | None:
    """Compute a deterministic digest for source files when git metadata is absent."""

    if root is None:
        root = Path(__file__).resolve().parents[3]
    root_path = Path(root)
    if not root_path.exists():
        return None

    hasher = hashlib.sha256()
    source_roots = [root_path / "src", root_path / "tests", root_path / "docs"]
    extra_files = [root_path / "pyproject.toml", root_path / "README.md", root_path / "mkdocs.yml"]
    paths: list[Path] = []
    for source_root in source_roots:
        if source_root.exists():
            paths.extend(path for path in source_root.rglob("*") if path.is_file())
    paths.extend(path for path in extra_files if path.exists())

    for path in sorted(paths):
        relative = path.relative_to(root_path).as_posix()
        hasher.update(relative.encode())
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
        repo_pyproject = _repo_root() / "pyproject.toml"
        if repo_pyproject.exists():
            dependency_path.write_text(repo_pyproject.read_text(encoding="utf-8"), encoding="utf-8")
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


def summarize_submission_bundle(path: str | Path) -> dict[str, Any]:
    """Summarize score, risk, and seed coverage for a local submission bundle."""

    from chemworld.eval.metrics import evaluate_records

    root = Path(path)
    validation = validate_submission_bundle(root)
    if not validation.valid:
        return {"valid": False, "validation": validation.to_dict()}

    trajectory_paths = sorted((root / "trajectories").glob("*.jsonl"))
    evaluations = []
    seeds: set[int] = set()
    task_maturity: dict[str, dict[str, Any]] = {}
    for trajectory_path in trajectory_paths:
        records = load_jsonl(trajectory_path)
        evaluations.append(evaluate_records(records).to_dict())
        seeds.update(int(record["seed"]) for record in records)
        first = records[0]
        task_id = str(trajectory_value(first, 'benchmark_task_id', None) or first["task_id"])
        maturity_payload = {
            "kernel_maturity": first["kernel_maturity"],
            "physics_maturity": first["physics_maturity"],
            "proxy_allowed": bool(first["proxy_allowed"]),
        }
        existing = task_maturity.get(task_id)
        if existing is not None and existing != maturity_payload:
            return {
                "valid": False,
                "validation": validation.to_dict(),
                "errors": [f"mixed maturity metadata for benchmark task {task_id!r}"],
            }
        task_maturity[task_id] = maturity_payload

    total_scores = [float(item["total_score"]) for item in evaluations]
    risks = [float(item["mean_safety_risk"]) for item in evaluations]
    best_scores = [float(item["final_best_score"]) for item in evaluations]
    return {
        "valid": True,
        "path": str(root),
        "trajectory_count": len(trajectory_paths),
        "result_count": validation.result_count,
        "seeds": sorted(seeds),
        "mean_total_score": sum(total_scores) / len(total_scores),
        "mean_safety_risk": sum(risks) / len(risks),
        "max_final_best_score": max(best_scores),
        "task_maturity": task_maturity,
        "physics_maturity_levels": sorted(
            {item["physics_maturity"] for item in task_maturity.values()}
        ),
    }


def build_example_submission_bundle(
    path: str | Path,
    *,
    task_id: str = "reaction-to-purification",
    agent_name: str = "tool_using_llm_stub",
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Build a complete, valid local submission bundle example.

    The bundle is intentionally generated through the same runner, evaluator,
    verifier, and validator used by normal submissions.
    """

    from chemworld.eval.metrics import evaluate_records
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records
    from chemworld.legacy.agent_registry import make_agent

    task = get_task(task_id)
    resolved_seeds = list(task.seeds[:1] if seeds is None else seeds)
    root = Path(path)
    command = [
        "python",
        "-m",
        "chemworld.legacy.cli",
        "submission",
        "example",
        str(root),
        "--task-id",
        task_id,
        "--agent",
        agent_name,
        "--seeds",
        *(str(seed) for seed in resolved_seeds),
    ]
    agent_family = make_agent(agent_name).__class__.__name__
    init_submission_bundle(
        root,
        agent_name=agent_name,
        agent_family=agent_family,
        task_id=task_id,
        seeds=resolved_seeds,
        command=command,
        dependency_file="dependency_notes.md",
        llm_metadata={
            "online_model": None,
            "mode": "offline deterministic example",
            "requires_network": False,
        },
    )

    verification_results: list[dict[str, Any]] = []
    for seed in resolved_seeds:
        trajectory_path = (
            root / "trajectories" / f"{agent_name}_{task_id}_seed{seed}.jsonl"
        )
        result_path = root / "results" / f"{agent_name}_{task_id}_seed{seed}.json"
        explanation_path = (
            root / "explanations" / f"{agent_name}_{task_id}_seed{seed}.json"
        )
        run_agent(
            env_id=task.env_id,
            agent=make_agent(agent_name),
            world_split=task.world_split,
            budget=task.budget,
            objective=task.objective,
            seed=seed,
            task_id=task.task_id,
            output_path=trajectory_path,
        )
        records = load_jsonl(trajectory_path)
        validate_records(records)
        verification = verify_records(records).to_dict()
        verification_results.append(
            {
                "seed": seed,
                "trajectory_path": str(trajectory_path),
                "verified": verification["verified"],
                "checked_steps": verification["checked_steps"],
                "max_abs_error": verification["max_abs_error"],
            }
        )
        evaluation = evaluate_records(records, threshold=task.threshold).to_dict()
        evaluation.update(
            {
                "task_id": task.task_id,
                "task_contract_hash": records[0].get("task_contract_hash"),
                "runtime_profile_hash": records[0].get("runtime_profile_hash"),
                "mechanism_hash": records[0].get("mechanism_hash"),
                "scoring_contract_hash": records[0].get("scoring_contract_hash"),
                "observation_contract_hash": records[0].get("observation_contract_hash"),
                "trajectory_path": str(trajectory_path),
                "verification": verification,
            }
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2, sort_keys=True)
        explanation = {
            "schema_version": "chemworld-explanation-0.1",
            "task_id": task.task_id,
            "agent_name": agent_name,
            "seed": seed,
            "hypothesis": (
                "Moderate reaction conditions followed by phase handling should "
                "increase final assay score while keeping risk controlled."
            ),
            "learned_mechanism": (
                "The public observations are consistent with a target reaction "
                "competing against byproduct/degradation and downstream recovery losses."
            ),
            "failure_analysis": (
                "This example is a reproducibility artifact, not an optimized submission; "
                "it uses a deterministic tool-agent plan and does not tune per seed."
            ),
            "next_experiment_rationale": (
                "Compare a lower-temperature reaction with a longer extraction stage and "
                "use HPLC before final assay to update the local belief state."
            ),
            "public_evidence": {
                "steps": len(records),
                "final_best_score": evaluation["final_best_score"],
                "total_score": evaluation["total_score"],
                "invalid_action_count": evaluation["invalid_action_count"],
                "final_assay_count": evaluation["final_assay_count"],
            },
        }
        with explanation_path.open("w", encoding="utf-8") as handle:
            json.dump(explanation, handle, indent=2, sort_keys=True)

    validation = validate_submission_bundle(root)
    summary = summarize_submission_bundle(root)
    return {
        "path": str(root),
        "task_id": task_id,
        "agent_name": agent_name,
        "seeds": resolved_seeds,
        "validation": validation.to_dict(),
        "summary": summary,
        "verification": verification_results,
        "reproducible_command": command,
    }
