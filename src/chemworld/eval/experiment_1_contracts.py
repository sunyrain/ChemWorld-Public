"""Strict single-parent contract overlays for versioned Experiment 1 repairs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import file_sha256


class Experiment1ContractExtensionError(ValueError):
    """Raised when a repair overlay cannot be resolved exactly."""


def load_contract_document(root: Path, path: Path) -> dict[str, Any]:
    """Load a contract or a strict one-level base-plus-overrides contract."""

    raw = _load_object(path)
    extension = raw.get("extends")
    if extension is None:
        return raw
    if not isinstance(extension, Mapping):
        raise Experiment1ContractExtensionError("extends must be a path/digest binding")
    relative = Path(str(extension.get("path", "")))
    base_path = (root / relative).resolve()
    try:
        base_path.relative_to(root.resolve())
    except ValueError as exc:
        raise Experiment1ContractExtensionError(
            "extended contract escapes repository root"
        ) from exc
    if not base_path.is_file():
        raise Experiment1ContractExtensionError("extended contract path is missing")
    actual_digest = file_sha256(base_path)
    if str(extension.get("sha256", "")) != actual_digest:
        raise Experiment1ContractExtensionError("extended contract digest changed")
    base = _load_object(base_path)
    if "extends" in base:
        raise Experiment1ContractExtensionError("nested contract extension is forbidden")
    overrides = raw.get("overrides")
    if not isinstance(overrides, Mapping):
        raise Experiment1ContractExtensionError("extended contract requires object overrides")
    merged = _deep_merge(base, overrides)
    for key, value in raw.items():
        if key not in {"extends", "overrides"}:
            merged[key] = deepcopy(value)
    merged["contract_extension"] = {
        "path": relative.as_posix(),
        "sha256": actual_digest,
    }
    return merged


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Experiment1ContractExtensionError(f"{path} must contain an object")
    return value


def _deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


__all__ = ["Experiment1ContractExtensionError", "load_contract_document"]
