"""Confined relative references under a caller-supplied artifact root."""

from pathlib import Path


def repository_relative_reference(
    path: str | Path, *, repository_root: Path
) -> str:
    """Serialize a repository artifact without retaining an author-machine root."""

    root = repository_root.resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact is outside repository root: {path}") from error
