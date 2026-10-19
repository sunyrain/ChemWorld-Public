"""Small, host-owned revision store for agent-authored, on-demand notes.

One host serializes writes. Revision JSON is authoritative; the Markdown file is
a reconstructable view. This module does not advance or restore an environment.
"""

from __future__ import annotations

import difflib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class NotebookVersions:
    def __init__(self, notebook_path: Path, default_text: str) -> None:
        self.path = notebook_path
        self.directory = notebook_path.parent / "notebook_history"
        self.default_text = default_text

    def _paths(self) -> list[Path]:
        return sorted(self.directory.glob("[0-9]*.json"))

    def _load(self, revision: int | None = None) -> dict[str, Any]:
        paths = self._paths()
        if revision is None:
            if not paths:
                return {
                    "revision": 0,
                    "text": self.path.read_text(encoding="utf-8"),
                    "reviewed_through": None,
                    "public_cursor": None,
                }
            path = paths[-1]
        else:
            if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
                raise ValueError("revision must be a positive integer")
            path = self.directory / f"{revision:08d}.json"
        if not path.is_file():
            raise ValueError(f"unknown notebook revision: {revision}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _metadata(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key != "text"}

    def status(self) -> dict[str, Any]:
        return self._metadata(self._load())

    def sync_view(self) -> None:
        from chemworld.agents.experiment_documents import _atomic_write_text

        if self._paths():
            _atomic_write_text(self.path, self._load()["text"])

    @staticmethod
    def _page(text: str, offset: int, limit: int) -> dict[str, Any]:
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 32000
        ):
            raise ValueError("offset must be non-negative; limit must be 1..32000 characters")
        end = min(len(text), offset + limit)
        return {
            "text": text[offset:end],
            "offset": offset,
            "total_characters": len(text),
            "next_offset": end if end < len(text) else None,
            "truncated": offset > 0 or end < len(text),
        }

    def read(
        self, revision: int | None = None, *, offset: int = 0, limit: int = 12000
    ) -> dict[str, Any]:
        record = self._load(revision)
        return {**self._metadata(record), **self._page(record["text"], offset, limit)}

    def log(self, *, offset: int = 0, limit: int = 10) -> dict[str, Any]:
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            raise ValueError("offset must be non-negative; limit must be 1..100 revisions")
        paths = list(reversed(self._paths()))
        records = [
            self._metadata(json.loads(path.read_text(encoding="utf-8")))
            for path in paths[offset : offset + limit]
        ]
        end = offset + len(records)
        return {
            "revisions": records,
            "total": len(paths),
            "offset": offset,
            "next_offset": end if end < len(paths) else None,
        }

    def commit(
        self,
        text: str,
        *,
        public_cursor: dict[str, Any],
        reviewed_through: str | None = None,
        message: str | None = None,
        restored_from: int | None = None,
        kind: str = "write",
    ) -> dict[str, Any]:
        from chemworld.agents.experiment_documents import _atomic_write_text

        if not isinstance(text, str):
            raise TypeError("notebook text must be a string")
        if message is not None and not isinstance(message, str):
            raise TypeError("message must be a string or null")
        if reviewed_through is not None and not isinstance(reviewed_through, str):
            raise TypeError("reviewed_through must be an event ID or null")
        previous = self._load()
        # Import pre-existing free-form content at the current time, never backdate it.
        if previous["revision"] == 0 and previous["text"] != self.default_text:
            self._save(
                {
                    "revision": 1,
                    "text": previous["text"],
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "kind": "import",
                    "public_cursor": public_cursor,
                    "reviewed_through": None,
                    "message": "Imported existing notebook; original timing unknown",
                    "restored_from": None,
                }
            )
            previous = self._load()
        record = {
            "revision": previous["revision"] + 1,
            "text": text,
            "recorded_at": datetime.now(UTC).isoformat(),
            "kind": kind,
            "public_cursor": public_cursor,
            "reviewed_through": reviewed_through,
            "message": message,
            "restored_from": restored_from,
        }
        self._save(record)
        response = self._metadata(record)
        try:
            _atomic_write_text(self.path, text)
        except OSError:
            # The commit already succeeded; do not report failure and induce a retry.
            response["view_synced"] = False
        else:
            response["view_synced"] = True
        return response

    def _save(self, record: dict[str, Any]) -> None:
        from chemworld.agents.experiment_documents import _atomic_write_text

        path = self.directory / f"{record['revision']:08d}.json"
        if path.exists():
            raise FileExistsError("notebook revision already exists")
        _atomic_write_text(path, json.dumps(record, ensure_ascii=False, allow_nan=False))

    def diff(
        self,
        before: int | None = None,
        after: int | None = None,
        *,
        offset: int = 0,
        limit: int = 12000,
    ) -> dict[str, Any]:
        right = self._load(after)
        if before is None and right["revision"] <= 1:
            left: dict[str, Any] = {"revision": 0, "text": self.default_text}
        else:
            left = self._load(right["revision"] - 1 if before is None else before)
        lines = difflib.unified_diff(
            left["text"].splitlines(keepends=True),
            right["text"].splitlines(keepends=True),
            fromfile=f"v{left['revision']}",
            tofile=f"v{right['revision']}",
        )
        # Agent-authored Markdown often has no trailing newline. Preserve that
        # distinction without joining the removed and added text onto one line.
        delta = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in lines
        )
        return {
            "before": self._metadata(left),
            "after": self._metadata(right),
            **self._page(delta, offset, limit),
        }

    def restore(
        self, revision: int, *, public_cursor: dict[str, Any], message: str | None = None
    ) -> dict[str, Any]:
        old = self._load(revision)
        return self.commit(
            old["text"],
            public_cursor=public_cursor,
            reviewed_through=old.get("reviewed_through"),
            message=message,
            restored_from=revision,
            kind="restore",
        )
