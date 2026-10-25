"""Work II-only material tree binding helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from chemworld.eval.provenance import canonical_json_sha256, file_sha256


def work_ii_material_tree_sha256(
    root: Path,
    *,
    relative_roots: Iterable[str],
    excluded_relative_paths: Iterable[str] = (),
) -> str:
    """Hash a Work II source surface while avoiding declared evidence-registration cycles."""

    resolved_root = root.resolve()
    excluded = {
        (resolved_root / relative).resolve() for relative in excluded_relative_paths
    }
    if any(not path.is_relative_to(resolved_root) for path in excluded):
        raise ValueError("Work II source exclusion escapes the repository")
    entries: list[dict[str, str]] = []
    for relative in sorted(set(relative_roots)):
        source = (resolved_root / relative).resolve()
        if not source.is_relative_to(resolved_root) or not source.exists():
            raise ValueError(f"invalid Work II source root: {relative}")
        paths = (
            [source]
            if source.is_file()
            else sorted(path for path in source.rglob("*") if path.is_file())
        )
        for path in paths:
            if (
                any(
                    path == excluded_path or path.is_relative_to(excluded_path)
                    for excluded_path in excluded
                )
                or "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            entries.append(
                {
                    "path": path.relative_to(resolved_root).as_posix(),
                    "sha256": file_sha256(path),
                }
            )
    return canonical_json_sha256(entries)


__all__ = ["work_ii_material_tree_sha256"]
