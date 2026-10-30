"""Shared, immutable execution sources for reference blocks (outside the product).

Create the store at the block root before launching workers. Descendant units keep
small path-to-content references; equal bytes occupy one object in the block.
Each unit has one writer. Different units may publish shared objects concurrently.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import TypedDict


class SourceBinding(TypedDict):
    sha256: str
    bytes: int


class SourceStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory).resolve()
        self.index = self.directory / "source-references.json"
        self.bindings: dict[str, SourceBinding] = {}
        if self.index.exists():
            saved = json.loads(self.index.read_text(encoding="utf-8"))
            self.root = (self.directory / saved["store"]).resolve()
            self.bindings = saved["files"]
        else:
            self.root = next(
                (
                    parent / ".source-store"
                    for parent in (self.directory, *self.directory.parents)
                    if (parent / ".source-store").is_dir()
                ),
                self.directory / ".source-store",
            )
        self.root.mkdir(parents=True, exist_ok=True)

    def object_path(self, binding: SourceBinding) -> Path:
        digest = binding["sha256"]
        return self.root / digest[:2] / digest[2:]

    def capture(self, path: Path, payload: bytes | None = None) -> tuple[Path, bool]:
        path = path.resolve()
        data = path.read_bytes() if payload is None else payload
        binding: SourceBinding = {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        key = str(path)
        previous = self.bindings.get(key)
        if previous is not None and previous != binding:
            raise ValueError("loaded source changed; preserve the earlier snapshot unchanged")
        target = self.object_path(binding)
        if previous is not None and not target.is_file():
            raise ValueError("shared execution snapshot missing; original binding retained")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".writing-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                # Publish complete bytes without replacing another worker's object.
                with suppress(FileExistsError):
                    os.link(temporary, target)
            finally:
                Path(temporary).unlink()
        if target.read_bytes() != data:
            raise ValueError("shared execution snapshot changed; original binding retained")
        if previous is None:
            self.bindings[key] = binding
            self._save()
        return target, previous is not None

    def _save(self) -> None:
        value = {
            "store": os.path.relpath(self.root, self.directory),
            "files": self.bindings,
        }
        fd, temporary = tempfile.mkstemp(dir=self.directory, prefix=".source-references-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.index)
        finally:
            Path(temporary).unlink(missing_ok=True)
