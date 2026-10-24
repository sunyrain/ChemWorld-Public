"""Shared report output and checked Git calls for evaluation entry points."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

Report = TypeVar("Report", bound=Mapping[str, Any])


def write_report_outputs(
    report: Report,
    *,
    render_markdown: Callable[[Report], str],
    output_path: str | Path,
    markdown_path: str | Path | None = None,
    allow_existing: bool = False,
) -> tuple[Path, Path]:
    """Write strict JSON and caller-rendered Markdown with overwrite protection."""

    report_path = Path(output_path)
    resolved_markdown = (
        Path(markdown_path) if markdown_path is not None else report_path.with_suffix(".md")
    )
    if report_path.resolve() == resolved_markdown.resolve():
        raise ValueError("JSON and Markdown outputs must use distinct paths")
    occupied = [path for path in (report_path, resolved_markdown) if path.exists()]
    if occupied and not allow_existing:
        rendered = ", ".join(str(path) for path in occupied)
        raise FileExistsError(f"refusing to replace or overwrite completed output: {rendered}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_markdown.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    resolved_markdown.write_text(
        render_markdown(report),
        encoding="utf-8",
        newline="\n",
    )
    return report_path, resolved_markdown


def git_output(root: Path, *args: str) -> str:
    """Run checked Git in the supplied root and return stripped text stdout."""

    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
