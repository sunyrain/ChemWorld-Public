"""Persistent, ownership-separated documents for operation-level experiments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chemworld.data.logging import to_builtin

if TYPE_CHECKING:
    from chemworld.agents.notebook_versions import NotebookVersions

EXPERIMENT_DOCUMENTS_VERSION = "chemworld-experiment-documents-0.1"
AUTHORITATIVE_LEDGER_VERSION = "chemworld-environment-authoritative-ledger-0.1"
DEFAULT_NOTEBOOK_TEXT = "# Experiment Notebook\n\n"


class AuthoritativeDocumentIntegrityError(RuntimeError):
    """Raised when the environment-owned ledger changed outside the host API."""


class ExperimentDocumentWorkspace:
    """Own one authoritative JSONL ledger and one model-owned Markdown notebook.

    The authoritative file is append-only through :meth:`append_operation`.
    Notebook writes are allowed, but every public mutation verifies that the
    authoritative file still matches the hash retained by this host object.
    """

    def __init__(self, run_directory: str | Path, *, versioned_notebook: bool = False) -> None:
        root = Path(run_directory).expanduser().resolve()
        self.run_directory = root
        self.documents_directory = root / "experiment_documents"
        self.authoritative_path = (
            self.documents_directory / "environment_authoritative_ledger.jsonl"
        )
        self.notebook_path = self.documents_directory / "model_owned_notebook.md"
        self.versioned_notebook = versioned_notebook
        self._expected_authoritative_sha256: str | None = None
        self._last_event_id: str | None = None
        self._initialized = False
        self._lock = threading.RLock()

    def initialize(
        self,
        *,
        expected_authoritative_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Create missing documents while preserving an existing workspace."""

        with self._lock:
            self.documents_directory.mkdir(parents=True, exist_ok=True)
            if not self.authoritative_path.exists():
                _atomic_write_bytes(self.authoritative_path, b"")
            if not self.notebook_path.exists():
                _atomic_write_text(self.notebook_path, DEFAULT_NOTEBOOK_TEXT)
            fingerprint = _authoritative_fingerprint(self.authoritative_path)
            if (
                expected_authoritative_sha256 is not None
                and fingerprint["sha256"] != expected_authoritative_sha256
            ):
                raise AuthoritativeDocumentIntegrityError(
                    "authoritative ledger does not match the supplied restart hash"
                )
            if (
                self._initialized
                and self._expected_authoritative_sha256 is not None
                and fingerprint["sha256"] != self._expected_authoritative_sha256
            ):
                raise AuthoritativeDocumentIntegrityError(
                    "authoritative ledger changed outside the host append API"
                )
            self._expected_authoritative_sha256 = fingerprint["sha256"]
            self._last_event_id = fingerprint["last_event_id"]
            self._initialized = True
            if self.versioned_notebook:
                self._versions().sync_view()
            return self.manifest()

    def reset(self) -> dict[str, Any]:
        """Explicitly replace both documents with a fresh empty workspace."""

        with self._lock:
            if self.versioned_notebook and self._versions().log()["total"]:
                raise ValueError("start a new run directory to reset a versioned notebook")
            self.documents_directory.mkdir(parents=True, exist_ok=True)
            _atomic_write_bytes(self.authoritative_path, b"")
            _atomic_write_text(self.notebook_path, DEFAULT_NOTEBOOK_TEXT)
            self._expected_authoritative_sha256 = _sha256_bytes(b"")
            self._last_event_id = None
            self._initialized = True
            return self.manifest()

    def append_operation(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Append one canonical public operation event after integrity checks."""

        with self._lock:
            self._require_initialized()
            self.verify_authoritative_integrity()
            normalized = _normalize_event(event)
            event_id = normalized["event_id"]
            if event_id in _authoritative_event_ids(self.authoritative_path):
                raise ValueError("event_id must be unique in the authoritative ledger")
            encoded = (
                json.dumps(
                    normalized,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                + b"\n"
            )
            existing = self.authoritative_path.read_bytes()
            _atomic_write_bytes(self.authoritative_path, existing + encoded)
            fingerprint = _authoritative_fingerprint(self.authoritative_path)
            if fingerprint["last_event_id"] != event_id:
                raise AuthoritativeDocumentIntegrityError(
                    "authoritative append did not retain the submitted event"
                )
            self._expected_authoritative_sha256 = fingerprint["sha256"]
            self._last_event_id = event_id
            self.verify_authoritative_integrity()
            return self.manifest()

    def verify_authoritative_integrity(
        self,
        *,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Verify bytes, JSONL structure, and the retained host-side hash."""

        with self._lock:
            self._require_initialized()
            fingerprint = _authoritative_fingerprint(self.authoritative_path)
            expected = (
                expected_sha256
                if expected_sha256 is not None
                else self._expected_authoritative_sha256
            )
            if expected is None or fingerprint["sha256"] != expected:
                raise AuthoritativeDocumentIntegrityError(
                    "authoritative ledger changed outside the host append API"
                )
            if fingerprint["last_event_id"] != self._last_event_id:
                raise AuthoritativeDocumentIntegrityError(
                    "authoritative ledger event sequence changed unexpectedly"
                )
            return deepcopy(fingerprint)

    def read_notebook(self, revision: int | None = None) -> str:
        """Return the model-owned notebook after checking the host ledger."""

        with self._lock:
            self.verify_authoritative_integrity()
            if self.versioned_notebook:
                return str(self._versions()._load(revision)["text"])
            if revision is not None:
                raise ValueError("notebook versioning is not enabled")
            return self.notebook_path.read_text(encoding="utf-8")

    def write_notebook(self, text: str) -> dict[str, Any]:
        """Atomically replace model-owned notebook text without touching the ledger."""

        if not isinstance(text, str):
            raise TypeError("notebook text must be a string")
        with self._lock:
            self.verify_authoritative_integrity()
            if self.versioned_notebook:
                self.notebook_tool("write", text=text)
            else:
                _atomic_write_text(self.notebook_path, text)
            self.verify_authoritative_integrity()
            return self.manifest()

    def _versions(self) -> NotebookVersions:
        from chemworld.agents.notebook_versions import NotebookVersions

        return NotebookVersions(self.notebook_path, DEFAULT_NOTEBOOK_TEXT)

    def notebook_tool(self, operation: str, **arguments: Any) -> dict[str, Any]:
        """On-demand notebook API; no body is returned by writes, logs or restores."""
        if not self.versioned_notebook:
            raise ValueError("notebook versioning is not enabled")
        fields = {
            "write": {"text", "message", "reviewed_through"},
            "restore": {"revision", "message"},
            "read": {"revision", "offset", "limit"},
            "log": {"offset", "limit"},
            "diff": {"before", "after", "offset", "limit"},
        }
        if operation not in fields or arguments.keys() - fields[operation]:
            raise ValueError("unknown notebook operation or arguments")
        with self._lock:
            ledger = self.verify_authoritative_integrity()
            versions = self._versions()
            cursor = {"event_count": ledger["line_count"], "last_event_id": ledger["last_event_id"]}
            if operation in {"write", "restore"}:
                reviewed = arguments.get("reviewed_through")
                if reviewed is not None and reviewed not in _authoritative_event_ids(
                    self.authoritative_path
                ):
                    raise ValueError("reviewed_through must reference an existing public event")
                method = versions.commit if operation == "write" else versions.restore
                result = method(public_cursor=cursor, **arguments)
            elif operation in {"read", "log", "diff"}:
                result = getattr(versions, operation)(**arguments)
            else:
                raise ValueError("unknown notebook operation")
            self.verify_authoritative_integrity()
            return {**result, "current_public_cursor": cursor}

    def manifest(self) -> dict[str, Any]:
        """Return paths and fingerprints, never authoritative ledger contents."""

        with self._lock:
            authoritative = self.verify_authoritative_integrity()
            notebook = _text_fingerprint(self.notebook_path)
            return {
                "schema_version": EXPERIMENT_DOCUMENTS_VERSION,
                "authoritative_ledger": {
                    "schema_version": AUTHORITATIVE_LEDGER_VERSION,
                    "owner": "environment_host",
                    "agent_writable": False,
                    "relative_path": self._relative_path(self.authoritative_path),
                    **authoritative,
                },
                "model_notebook": {
                    "owner": "model",
                    "agent_writable": True,
                    "relative_path": self._relative_path(self.notebook_path),
                    **notebook,
                    **(
                        {
                            "versioning": True,
                            "latest": self._versions().status(),
                            "content_policy": "on_demand_only",
                        }
                        if self.versioned_notebook
                        else {}
                    ),
                },
            }

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.run_directory).as_posix()

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("experiment document workspace is not initialized")


def _normalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(event, Mapping):
        raise TypeError("authoritative event must be a mapping")
    _reject_non_finite_numbers(event)
    normalized = to_builtin(dict(event))
    if not isinstance(normalized, dict):
        raise TypeError("authoritative event must normalize to a JSON object")
    event_id = normalized.get("event_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("authoritative event requires a non-empty string event_id")
    normalized["event_id"] = event_id.strip()
    try:
        json.dumps(normalized, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("authoritative event must contain finite JSON values") from exc
    return normalized


def _reject_non_finite_numbers(value: Any) -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("authoritative event must contain finite JSON values")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_non_finite_numbers(item)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _reject_non_finite_numbers(item)


def _authoritative_fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AuthoritativeDocumentIntegrityError(
            "authoritative ledger is not valid UTF-8"
        ) from exc
    lines = text.splitlines()
    last_event_id: str | None = None
    seen_event_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise AuthoritativeDocumentIntegrityError(
                f"authoritative ledger contains an empty JSONL record at line {line_number}"
            )
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuthoritativeDocumentIntegrityError(
                f"authoritative ledger contains invalid JSON at line {line_number}"
            ) from exc
        if not isinstance(event, dict):
            raise AuthoritativeDocumentIntegrityError(
                f"authoritative ledger record {line_number} is not an object"
            )
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise AuthoritativeDocumentIntegrityError(
                f"authoritative ledger record {line_number} lacks event_id"
            )
        normalized_event_id = event_id.strip()
        if normalized_event_id in seen_event_ids:
            raise AuthoritativeDocumentIntegrityError(
                f"authoritative ledger repeats event_id at line {line_number}"
            )
        seen_event_ids.add(normalized_event_id)
        last_event_id = normalized_event_id
    return {
        "sha256": _sha256_bytes(data),
        "byte_count": len(data),
        "line_count": len(lines),
        "last_event_id": last_event_id,
    }


def _authoritative_event_ids(path: Path) -> set[str]:
    event_ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        event_ids.add(event["event_id"].strip())
    return event_ids


def _text_fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("model notebook is not valid UTF-8") from exc
    return {
        "sha256": _sha256_bytes(data),
        "byte_count": len(data),
        "line_count": len(text.splitlines()),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Windows readers can briefly prevent replacement of an open snapshot.
        # Retry only this already-written file commit, never the scientific action.
        deadline = time.monotonic() + 0.25
        while True:
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError as error:
                if getattr(error, "winerror", None) not in {5, 32} or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "AUTHORITATIVE_LEDGER_VERSION",
    "DEFAULT_NOTEBOOK_TEXT",
    "EXPERIMENT_DOCUMENTS_VERSION",
    "AuthoritativeDocumentIntegrityError",
    "ExperimentDocumentWorkspace",
]
