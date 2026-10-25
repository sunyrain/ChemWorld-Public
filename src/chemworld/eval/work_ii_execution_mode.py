"""Shared development and release-freeze execution contracts for Work II."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import (
    canonical_json_sha256,
    git_source_commit,
    git_worktree_dirty,
)
from chemworld.eval.work_ii_source_binding import work_ii_material_tree_sha256

RELEASE_MANIFEST_VERSION = "chemworld-work-ii-execution-release-manifest-0.1"
DEVELOPMENT_EVIDENCE_STATUS = "development_only"
RELEASE_EVIDENCE_STATUS = "release_candidate"


class ExecutionMode(StrEnum):
    """Work II execution modes with deliberately different provenance costs."""

    DEVELOPMENT = "development"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class WorkIIExecutionContext:
    """Validated mode and provenance binding shared by a Work II run."""

    mode: ExecutionMode
    evidence_status: str
    release_eligible: bool
    c2_admission_authorized: bool
    tested_commit: str | None
    freeze_id: str | None
    release_manifest_sha256: str | None
    execution_surface_sha256: str | None

    @property
    def execution_mode(self) -> str:
        """Return the JSON-facing execution-mode value."""

        return self.mode.value


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def release_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical self-hash for a release manifest."""

    return _self_hash(manifest, "manifest_sha256")


def _is_hex_digest(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and re.fullmatch(r"[0-9a-f]+", value) is not None
    )


def _normalize_execution_surface(
    root: Path, execution_surface: Sequence[str]
) -> tuple[str, ...]:
    resolved_root = root.resolve()
    normalized: set[str] = set()
    for raw in execution_surface:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("execution-surface paths must be non-empty strings")
        candidate = (resolved_root / raw).resolve()
        if not candidate.is_relative_to(resolved_root) or candidate == resolved_root:
            raise ValueError(f"execution-surface path escapes the repository: {raw}")
        if not candidate.exists():
            raise ValueError(f"execution-surface path does not exist: {raw}")
        normalized.add(candidate.relative_to(resolved_root).as_posix())
    if not normalized:
        raise ValueError("release freeze requires a non-empty execution surface")
    return tuple(sorted(normalized))


def _freeze_id(*, tested_commit: str, execution_surface_sha256: str) -> str:
    return canonical_json_sha256(
        {
            "schema_version": RELEASE_MANIFEST_VERSION,
            "tested_commit": tested_commit,
            "execution_surface_sha256": execution_surface_sha256,
        }
    )


def build_release_manifest(
    root: Path, *, execution_surface: Sequence[str]
) -> dict[str, Any]:
    """Freeze a clean HEAD and the smallest caller-declared execution surface."""

    root = root.resolve()
    if git_worktree_dirty(root):
        raise ValueError("release freeze requires a clean worktree")
    relative_roots = _normalize_execution_surface(root, execution_surface)
    tested_commit = git_source_commit(root)
    surface_sha256 = work_ii_material_tree_sha256(
        root, relative_roots=relative_roots
    )
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_MANIFEST_VERSION,
        "status": "frozen_release_execution_surface",
        "execution_mode": ExecutionMode.RELEASE.value,
        "formal_result": False,
        "formal_participant_outcome_count": 0,
        "tested_commit": tested_commit,
        "execution_surface": {
            "relative_roots": list(relative_roots),
            "sha256": surface_sha256,
        },
        "freeze_id": _freeze_id(
            tested_commit=tested_commit,
            execution_surface_sha256=surface_sha256,
        ),
    }
    manifest["manifest_sha256"] = release_manifest_sha256(manifest)
    return manifest


def validate_release_manifest(
    root: Path, manifest: Mapping[str, Any]
) -> list[str]:
    """Validate one release manifest against a clean, exact current HEAD."""

    root = root.resolve()
    errors: list[str] = []
    if manifest.get("schema_version") != RELEASE_MANIFEST_VERSION:
        errors.append("unexpected Work II execution release-manifest schema")
    if manifest.get("manifest_sha256") != release_manifest_sha256(manifest):
        errors.append("Work II execution release manifest self-hash mismatch")
    if (
        manifest.get("status") != "frozen_release_execution_surface"
        or manifest.get("execution_mode") != ExecutionMode.RELEASE.value
        or manifest.get("formal_result") is not False
        or manifest.get("formal_participant_outcome_count") != 0
    ):
        errors.append("Work II execution release manifest crossed its freeze boundary")

    tested_commit = manifest.get("tested_commit")
    if not _is_hex_digest(tested_commit, 40):
        errors.append("Work II execution release manifest lacks a full tested commit")

    surface = manifest.get("execution_surface")
    surface = surface if isinstance(surface, Mapping) else {}
    roots = surface.get("relative_roots")
    if not isinstance(roots, list) or not all(isinstance(item, str) for item in roots):
        errors.append("Work II execution release manifest lacks its execution surface")
        normalized_roots: tuple[str, ...] | None = None
    else:
        try:
            normalized_roots = _normalize_execution_surface(root, roots)
        except ValueError as error:
            errors.append(f"Work II execution surface is invalid: {error}")
            normalized_roots = None
        else:
            if list(normalized_roots) != roots:
                errors.append("Work II execution-surface roster is not canonical")

    surface_sha256 = surface.get("sha256")
    if not _is_hex_digest(surface_sha256, 64):
        errors.append("Work II execution release manifest lacks its surface hash")
    elif normalized_roots is not None:
        current_surface_sha256 = work_ii_material_tree_sha256(
            root, relative_roots=normalized_roots
        )
        if current_surface_sha256 != surface_sha256:
            errors.append("Work II release execution surface changed after freeze")

    if (
        isinstance(tested_commit, str)
        and _is_hex_digest(tested_commit, 40)
        and isinstance(surface_sha256, str)
        and _is_hex_digest(surface_sha256, 64)
    ):
        if manifest.get("freeze_id") != _freeze_id(
            tested_commit=tested_commit,
            execution_surface_sha256=surface_sha256,
        ):
            errors.append("Work II execution release manifest freeze ID mismatch")
    elif not _is_hex_digest(manifest.get("freeze_id"), 64):
        errors.append("Work II execution release manifest lacks a valid freeze ID")

    current_commit = git_source_commit(root)
    if _is_hex_digest(tested_commit, 40) and current_commit != tested_commit:
        errors.append("Work II release execution requires the exact frozen HEAD")
    return errors


def _load_release_manifest(
    release_manifest: Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(release_manifest, Path):
        value = json.loads(release_manifest.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("release manifest must contain a JSON object")
        return value
    return dict(release_manifest)


def prepare_execution_context(
    root: Path,
    *,
    mode: ExecutionMode | str = ExecutionMode.DEVELOPMENT,
    release_manifest: Path | Mapping[str, Any] | None = None,
) -> WorkIIExecutionContext:
    """Prepare a cheap development context or a fully validated release context."""

    resolved_mode = ExecutionMode(mode)
    if resolved_mode is ExecutionMode.DEVELOPMENT:
        if release_manifest is not None:
            raise ValueError("development mode must not bind a release manifest")
        return WorkIIExecutionContext(
            mode=resolved_mode,
            evidence_status=DEVELOPMENT_EVIDENCE_STATUS,
            release_eligible=False,
            c2_admission_authorized=False,
            tested_commit=None,
            freeze_id=None,
            release_manifest_sha256=None,
            execution_surface_sha256=None,
        )
    if release_manifest is None:
        raise ValueError("release mode requires a release manifest")
    manifest = _load_release_manifest(release_manifest)
    errors = validate_release_manifest(root, manifest)
    if errors:
        raise ValueError("invalid Work II execution release manifest: " + "; ".join(errors))
    surface = manifest["execution_surface"]
    return WorkIIExecutionContext(
        mode=resolved_mode,
        evidence_status=RELEASE_EVIDENCE_STATUS,
        release_eligible=True,
        c2_admission_authorized=True,
        tested_commit=str(manifest["tested_commit"]),
        freeze_id=str(manifest["freeze_id"]),
        release_manifest_sha256=str(manifest["manifest_sha256"]),
        execution_surface_sha256=str(surface["sha256"]),
    )


def build_execution_envelope(context: WorkIIExecutionContext) -> dict[str, object]:
    """Return the uniform JSON envelope embedded in every Work II artifact."""

    return {
        "execution_mode": context.execution_mode,
        "evidence_status": context.evidence_status,
        "release_eligible": context.release_eligible,
        "c2_admission_authorized": context.c2_admission_authorized,
        "tested_commit": context.tested_commit,
        "freeze_id": context.freeze_id,
        "release_manifest_sha256": context.release_manifest_sha256,
        "execution_surface_sha256": context.execution_surface_sha256,
    }


def validate_execution_envelope(
    root: Path,
    payload: Mapping[str, Any],
    expected_context: WorkIIExecutionContext | None = None,
) -> list[str]:
    """Validate a uniform artifact envelope and any expected run binding."""

    errors: list[str] = []
    expected_keys = set(build_execution_envelope(
        WorkIIExecutionContext(
            mode=ExecutionMode.DEVELOPMENT,
            evidence_status=DEVELOPMENT_EVIDENCE_STATUS,
            release_eligible=False,
            c2_admission_authorized=False,
            tested_commit=None,
            freeze_id=None,
            release_manifest_sha256=None,
            execution_surface_sha256=None,
        )
    ))
    if set(payload) != expected_keys:
        errors.append("Work II execution envelope field roster mismatch")
    mode = payload.get("execution_mode")
    if mode == ExecutionMode.DEVELOPMENT.value:
        expected = build_execution_envelope(
            prepare_execution_context(root, mode=ExecutionMode.DEVELOPMENT)
        )
        if dict(payload) != expected:
            errors.append("Work II development envelope has release bindings or admission")
    elif mode == ExecutionMode.RELEASE.value:
        if (
            payload.get("evidence_status") != RELEASE_EVIDENCE_STATUS
            or payload.get("release_eligible") is not True
            or payload.get("c2_admission_authorized") is not True
            or not _is_hex_digest(payload.get("tested_commit"), 40)
            or not _is_hex_digest(payload.get("freeze_id"), 64)
            or not _is_hex_digest(payload.get("release_manifest_sha256"), 64)
            or not _is_hex_digest(payload.get("execution_surface_sha256"), 64)
        ):
            errors.append("Work II release envelope lacks valid release bindings")
        if git_source_commit(root) != payload.get("tested_commit"):
            errors.append("Work II release envelope does not bind the exact current HEAD")
    else:
        errors.append("Work II execution envelope has an unsupported mode")
    if expected_context is not None and dict(payload) != build_execution_envelope(
        expected_context
    ):
        errors.append("Work II execution envelope differs from the expected context")
    return errors


def validate_release_d1_config(
    root: Path,
    config: Mapping[str, Any],
    release_manifest: Path | Mapping[str, Any],
    *,
    require_provider_authorized: bool,
) -> list[str]:
    """Validate that one D1 config descends from the exact active release freeze."""

    errors: list[str] = []
    try:
        context = prepare_execution_context(
            root,
            mode=ExecutionMode.RELEASE,
            release_manifest=release_manifest,
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"Work II release manifest validation failed: {error}"]

    execution_context = config.get("execution_context")
    if not isinstance(execution_context, Mapping):
        errors.append("Work II release D1 lacks its execution context")
    else:
        context_errors = validate_execution_envelope(root, execution_context, context)
        errors.extend(
            f"Work II release D1 execution context is invalid: {error}"
            for error in context_errors
        )
    if config.get("legacy_source_evidence") is not False:
        errors.append("Work II release D1 uses legacy source evidence")
    qualification = config.get("qualification")
    qualification = qualification if isinstance(qualification, Mapping) else {}
    if qualification.get("q2_passed") is not True:
        errors.append("Work II release D1 lacks a passing same-freeze Q2")
    if qualification.get("formal_r5_authorized") is not False:
        errors.append("Work II release D1 crossed the R5 authorization boundary")
    execution_authorized = qualification.get("execution_authorized")
    if not isinstance(execution_authorized, bool):
        errors.append("Work II release D1 execution authorization is malformed")
    elif require_provider_authorized and execution_authorized is not True:
        errors.append("Work II release D1 is not provider-authorized")
    return errors


__all__ = [
    "DEVELOPMENT_EVIDENCE_STATUS",
    "RELEASE_EVIDENCE_STATUS",
    "RELEASE_MANIFEST_VERSION",
    "ExecutionMode",
    "WorkIIExecutionContext",
    "build_execution_envelope",
    "build_release_manifest",
    "prepare_execution_context",
    "release_manifest_sha256",
    "validate_execution_envelope",
    "validate_release_d1_config",
    "validate_release_manifest",
]
