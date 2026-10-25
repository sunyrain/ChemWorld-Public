"""Canonical artifact I/O, Git provenance, and solver manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from time import sleep
from typing import Any
from uuid import uuid4

from chemworld import __version__
from chemworld.data.submission import git_commit
from chemworld.eval.seed_suite import private_eval_salt_policy
from chemworld.physchem.solver_backend import (
    DEFAULT_REACTION_ODE_POLICY,
    DEFAULT_REACTOR_ODE_POLICY,
    REFERENCE_REACTION_ODE_POLICY,
    RUNTIME_REACTION_KERNEL_ODE_POLICY,
)
from chemworld.tasks import get_task

PROVENANCE_SCHEMA_VERSION = "chemworld-solver-provenance-0.2"
ATOMIC_REPLACE_RETRY_DELAYS_S = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64)


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize *payload* with the repository's stable hashing contract."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    """Return the SHA-256 of canonical UTF-8 JSON."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of a materialized artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def repository_tree_sha256(
    root: Path,
    *,
    relative_roots: Iterable[str],
) -> str:
    """Hash every material source file under explicitly declared repository roots."""

    entries: list[dict[str, str]] = []
    resolved_root = root.resolve()
    for relative in sorted(set(relative_roots)):
        source = (resolved_root / relative).resolve()
        if not source.is_relative_to(resolved_root) or not source.exists():
            raise ValueError(f"invalid repository source root: {relative}")
        paths = (
            [source]
            if source.is_file()
            else sorted(path for path in source.rglob("*") if path.is_file())
        )
        for path in paths:
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            entries.append(
                {
                    "path": path.relative_to(resolved_root).as_posix(),
                    "sha256": file_sha256(path),
                }
            )
    return canonical_json_sha256(entries)


def write_json_atomic(path: Path, payload: Any, *, sort_keys: bool = True) -> None:
    """Write deterministic, human-readable JSON through an adjacent temp file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the adjacent temporary basename short for Windows MAX_PATH while
    # retaining collision safety for concurrent writers in one directory.
    temporary = path.with_name(f".atomic-{uuid4().hex[:8]}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=sort_keys,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        for retry_delay_s in (*ATOMIC_REPLACE_RETRY_DELAYS_S, None):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if retry_delay_s is None:
                    raise
                sleep(retry_delay_s)
    finally:
        temporary.unlink(missing_ok=True)


def git_source_commit(root: Path) -> str:
    """Return the repository HEAD bound to an artifact generation run."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def git_worktree_dirty(
    root: Path,
    *,
    excluded_paths: Iterable[str] = (),
    excluded_prefixes: Iterable[str] = (),
) -> bool:
    """Report tracked or untracked changes outside declared generated evidence.

    A clean-source attestation must describe the entire material repository, not
    merely Git's index. Callers must explicitly enumerate generated outputs that
    may be refreshed without invalidating source provenance.
    """

    exact = {_normalize_git_path(path) for path in excluded_paths}
    prefixes = tuple(_normalize_git_prefix(path) for path in excluded_prefixes)
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    for line in completed.stdout.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        normalized = _normalize_git_path(path)
        if normalized in exact or any(normalized.startswith(item) for item in prefixes):
            continue
        return True
    return False


def _normalize_git_path(path: str) -> str:
    return path.strip('"').replace("\\", "/")


def _normalize_git_prefix(path: str) -> str:
    normalized = _normalize_git_path(path)
    return normalized if normalized.endswith("/") else f"{normalized}/"


def build_solver_provenance_manifest(
    *,
    task_ids: list[str],
    agents: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Return a JSON-friendly manifest for reproducible benchmark artifacts."""

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "chemworld_version": __version__,
        "commit_hash": git_commit(),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "dependencies": {
            package: _version_or_unavailable(package)
            for package in (
                "gymnasium",
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "pytest",
                "mkdocs",
            )
        },
        "solver_tolerances": {
            "trajectory_verify_default_tolerance": 1.0e-5,
            "acid_base_root_xtol": 1.0e-14,
            "acid_base_root_rtol": 1.0e-12,
            "precipitation_hook_max_passes": 3,
            "equilibrium_residual_public_metric": "normalized charge-balance proxy",
        },
        "ode_solver_backend": {
            "backend_module": "chemworld.physchem.solver_backend",
            "scipy_entrypoint": "scipy.integrate.solve_ivp",
            "policies": [
                DEFAULT_REACTION_ODE_POLICY.to_dict(),
                DEFAULT_REACTOR_ODE_POLICY.to_dict(),
                RUNTIME_REACTION_KERNEL_ODE_POLICY.to_dict(),
                REFERENCE_REACTION_ODE_POLICY.to_dict(),
            ],
            "diagnostic_contract": {
                "records": [
                    "success",
                    "message",
                    "status",
                    "nfev",
                    "njev",
                    "nlu",
                    "event_count",
                    "final_time_s",
                    "policy_hash",
                ],
                "failure_policy": "callers raise RuntimeError after recording solver diagnostics",
            },
        },
        "tasks": [
            {
                "task_id": task.task_id,
                "contract_hash": task.contract_hash,
                "scenario_id": task.scenario_id,
                "world_split": task.world_split,
                "budget": task.budget,
                "episode_mode": task.episode_mode,
                "physics_maturity": task.kernel_maturity.lowest_level.value,
                "proxy_allowed": task.kernel_maturity.proxy_allowed,
            }
            for task in (get_task(task_id) for task_id in task_ids)
        ],
        "agents": list(agents),
        "seeds": list(seeds),
        "private_eval_salt_policy": private_eval_salt_policy(),
    }


def _version_or_unavailable(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unavailable"


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "build_solver_provenance_manifest",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "file_sha256",
    "git_source_commit",
    "git_worktree_dirty",
    "write_json_atomic",
]
