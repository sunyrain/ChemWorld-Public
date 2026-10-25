"""Bounded file IPC for one-session, tool-driven Codex experiments.

The authoritative environment and trajectory never live in this workspace.  The
workspace contains a complete public operation stream, a recent cache, a small bridge, and an
initially empty directory in which the model may create arbitrary notes.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chemworld.agents.public_history import public_history_page
from chemworld.data.logging import to_builtin

EXPERIMENT_CODEX_IPC_VERSION = "chemworld-experiment-codex-ipc-0.2"
CAMPAIGN_PROGRESS_VERSION = "chemworld-campaign-progress-0.2"
DEFAULT_MAX_TOOL_OUTPUT_BYTES = 32_768
DEFAULT_HISTORY_EVENT_LIMIT = 64
DEFAULT_HISTORY_BYTE_LIMIT = 131_072
DEFAULT_MAX_ARTIFACT_BYTES = 8_388_608
ATOMIC_REPLACE_RETRY_LIMIT = 40
ATOMIC_REPLACE_RETRY_INTERVAL_S = 0.025
SESSION_RETIRE_RETRY_LIMIT = 20
SESSION_RETIRE_RETRY_INTERVAL_S = 0.05


class ExperimentCodexIPCError(RuntimeError):
    """The isolated Codex/runner bridge violated its public protocol."""


@dataclass(frozen=True)
class IPCRequest:
    """One idempotent operation request accepted from the Codex process."""

    session_id: str
    request_id: str
    expected_step: int
    action: dict[str, Any]
    payload_sha256: str
    decision_audit: dict[str, Any] | None = None


class ExperimentCodexWorkspace:
    """Manage a bounded public cache and session-scoped request/response files."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_tool_output_bytes: int = DEFAULT_MAX_TOOL_OUTPUT_BYTES,
        history_event_limit: int = DEFAULT_HISTORY_EVENT_LIMIT,
        history_byte_limit: int = DEFAULT_HISTORY_BYTE_LIMIT,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        poll_interval_s: float = 0.05,
    ) -> None:
        resolved = Path(root).expanduser().resolve()
        if max_tool_output_bytes < 1_024:
            raise ValueError("max_tool_output_bytes must be at least 1024")
        if history_event_limit < 1:
            raise ValueError("history_event_limit must be positive")
        if history_byte_limit < max_tool_output_bytes:
            raise ValueError("history_byte_limit must be at least max_tool_output_bytes")
        if max_artifact_bytes < max_tool_output_bytes:
            raise ValueError("max_artifact_bytes must be at least max_tool_output_bytes")
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        self.root = resolved
        self.agent_directory = resolved / "agent"
        self.transport_directory = self.agent_directory / ".transport"
        self.public_directory = resolved / "public"
        self.artifacts_directory = self.public_directory / "artifacts"
        self.reference_directory = resolved / "reference"
        self.ipc_directory = resolved / ".ipc"
        self.sessions_directory = self.ipc_directory / "sessions"
        # Codex' Windows workspace sandbox will execute a program from its writable
        # working root, but can decline an executable path that traverses to the
        # parent.  This generated bridge therefore lives in agent/.  It is still
        # untrusted/non-authoritative: the host verifies its exact source hash
        # before accepting every action and after each Codex session.
        self.lab_tool_path = self.agent_directory / "lab_tool.py"
        self.current_path = self.public_directory / "current.json"
        self.history_path = self.public_directory / "history.jsonl"
        self.operation_history_path = self.public_directory / "operation_history.jsonl"
        self.active_session_path = self.ipc_directory / "active_session.json"
        self.max_tool_output_bytes = int(max_tool_output_bytes)
        self.history_event_limit = int(history_event_limit)
        self.history_byte_limit = int(history_byte_limit)
        self.max_artifact_bytes = int(max_artifact_bytes)
        self.poll_interval_s = float(poll_interval_s)
        self._initialized = False
        self._history_rows: list[dict[str, Any]] = []
        self._session_descriptors: dict[str, dict[str, Any]] = {}

    def initialize_fresh(self) -> dict[str, Any]:
        """Initialize a new cell without deleting or silently reusing old memory."""

        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"interactive Codex workspace must start empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.agent_directory.mkdir()
        self.public_directory.mkdir()
        self.artifacts_directory.mkdir()
        self.reference_directory.mkdir()
        self.sessions_directory.mkdir(parents=True)
        _atomic_write_text(self.lab_tool_path, LAB_TOOL_SOURCE)
        # Windows images may not expose a python command to a sandboxed
        # Codex subprocess. Publish a tiny local shim so the documented
        # python lab_tool.py command resolves deterministically.
        if os.name == "nt":
            _atomic_write_text(
                self.agent_directory / "python.cmd",
                _python_shim_source(),
            )
        _atomic_write_text(self.history_path, "")
        _atomic_write_text(self.operation_history_path, "")
        self._history_rows = []
        self._session_descriptors = {}
        self._initialized = True
        return self.manifest()

    def publish_material_information(
        self,
        material_information: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write the environment-owned material condition under identical topology."""

        self._require_initialized()
        normalized = _bounded_json_object(
            material_information,
            max_bytes=self.max_artifact_bytes,
            label="material information",
        )
        path = self.reference_directory / "material_information.json"
        _atomic_write_json(path, normalized)
        return {
            "relative_path": "reference/material_information.json",
            **_fingerprint(path),
        }

    def publish_task_contract(
        self,
        task_contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write the bounded public objective/lifecycle contract for this cell."""

        self._require_initialized()
        normalized = _bounded_json_object(
            task_contract,
            max_bytes=self.max_artifact_bytes,
            label="public task contract",
        )
        path = self.reference_directory / "task_contract.json"
        _atomic_write_json(path, normalized)
        return {
            "relative_path": "reference/task_contract.json",
            **_fingerprint(path),
        }

    def publish_belief_checkpoint_contract(
        self,
        contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write the host-owned Work II checkpoint contract for a campaign cell."""

        self._require_initialized()
        normalized = _bounded_json_object(
            contract,
            max_bytes=self.max_artifact_bytes,
            label="belief checkpoint contract",
        )
        path = self.reference_directory / "belief_checkpoint_contract.json"
        _atomic_write_json(path, normalized)
        return {
            "relative_path": "reference/belief_checkpoint_contract.json",
            **_fingerprint(path),
        }

    def publish_terminal_action_readout_contract(
        self,
        contract: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write a terminal-only held-out action packet for a campaign cell."""

        self._require_initialized()
        normalized = _bounded_json_object(
            contract,
            max_bytes=self.max_artifact_bytes,
            label="terminal action readout contract",
        )
        path = self.reference_directory / "terminal_action_readout_contract.json"
        _atomic_write_json(path, normalized)
        return {
            "relative_path": "reference/terminal_action_readout_contract.json",
            **_fingerprint(path),
        }

    def publish_artifact(
        self,
        *,
        artifact_id: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Publish one immutable public characterization packet for bounded inspection."""

        self._require_initialized()
        _validate_leaf_name(artifact_id, label="artifact_id")
        normalized = _bounded_json_object(
            payload,
            max_bytes=self.max_artifact_bytes,
            label="public characterization artifact",
        )
        path = self.artifacts_directory / f"{artifact_id}.json"
        if path.exists():
            existing = _read_json_object(path)
            if _canonical_json(existing) != _canonical_json(normalized):
                raise ExperimentCodexIPCError(
                    "artifact_id was reused with different public content"
                )
        else:
            _atomic_write_json(path, normalized)
        return {
            "artifact_id": artifact_id,
            "relative_path": path.relative_to(self.root).as_posix(),
            **_fingerprint(path),
        }

    def artifact_access_audit(self, session_id: str) -> list[dict[str, Any]]:
        """Return bounded characterization-access events from either experiment bridge."""

        trusted_path = self.session_root(session_id) / "artifact_access.jsonl"
        path = (
            trusted_path
            if trusted_path.exists()
            else self.transport_session_root(session_id) / "artifact_access.jsonl"
        )
        if not path.exists():
            return []
        try:
            rows = _read_jsonl(path)
        except ExperimentCodexIPCError:
            return [
                {
                    "status": "invalid_untrusted_access_audit",
                    **_fingerprint(path),
                }
            ]
        return rows[-self.history_event_limit :]

    def mcp_tool_call_audit(self, session_id: str) -> list[dict[str, Any]]:
        """Return host-owned MCP call metadata without argument or result bodies."""

        path = self.session_root(session_id) / "mcp_tool_calls.jsonl"
        if not path.exists():
            return []
        rows = _read_jsonl(path)
        return rows[-max(self.history_event_limit * 8, 64) :]

    def belief_snapshot_audit(self, session_id: str) -> list[dict[str, Any]]:
        """Return the structured Work II checkpoints committed in one Codex session."""

        root = self.session_root(session_id) / "belief_snapshots"
        if not root.exists():
            return []
        return [_read_json_object(path) for path in sorted(root.glob("*.json"))]

    def belief_snapshot_draft_audit(self, session_id: str) -> dict[str, Any]:
        """Retain participant-authored staged fragments, including incomplete drafts."""

        root = self.session_root(session_id) / "belief_snapshot_drafts"
        drafts: list[dict[str, Any]] = []
        fragment_count = 0
        if root.exists():
            for draft_root in sorted(path for path in root.iterdir() if path.is_dir()):
                manifest_path = draft_root / "manifest.json"
                fragments_root = draft_root / "fragments"
                fragments = (
                    [_read_json_object(path) for path in sorted(fragments_root.glob("*.json"))]
                    if fragments_root.is_dir()
                    else []
                )
                fragment_count += len(fragments)
                finalization_path = draft_root / "finalization.json"
                drafts.append(
                    {
                        "draft_id": draft_root.name,
                        "manifest": (
                            _read_json_object(manifest_path) if manifest_path.is_file() else None
                        ),
                        "fragments": fragments,
                        "fragment_count": len(fragments),
                        "finalization": (
                            _read_json_object(finalization_path)
                            if finalization_path.is_file()
                            else None
                        ),
                    }
                )
        return {
            "schema_version": "chemworld-work-ii-belief-snapshot-draft-audit-0.1",
            "drafts": drafts,
            "draft_count": len(drafts),
            "fragment_count": fragment_count,
        }

    def final_recommendation_audit(self, session_id: str) -> dict[str, Any] | None:
        """Return the host-owned participant recommendation committed in one session."""

        path = self.session_root(session_id) / "final_recommendation.json"
        return _read_json_object(path) if path.exists() else None

    def start_session(
        self,
        *,
        session_id: str,
        expected_step: int = 1,
        response_timeout_s: float,
        session_scope: str = "experiment",
        terminal_action_readout_required: bool = False,
    ) -> dict[str, Any]:
        """Create a new session namespace while preserving optional Agent files."""

        self._require_initialized()
        if not session_id or any(char in session_id for char in "/\\"):
            raise ValueError("session_id must be a non-path string")
        if expected_step < 1:
            raise ValueError("expected_step must be positive")
        if response_timeout_s <= 0:
            raise ValueError("response_timeout_s must be positive")
        if session_scope not in {"experiment", "campaign"}:
            raise ValueError("session_scope must be experiment or campaign")
        if terminal_action_readout_required and session_scope != "campaign":
            raise ValueError("terminal action readout requires campaign scope")
        session_root = self.session_root(session_id)
        if session_root.exists():
            raise FileExistsError(f"session already exists: {session_id}")
        (session_root / "responses").mkdir(parents=True)
        (session_root / "mcp_requests").mkdir()
        transport_root = self.transport_session_root(session_id)
        if transport_root.exists():
            raise FileExistsError(f"session transport already exists: {session_id}")
        (transport_root / "requests").mkdir(parents=True)
        descriptor = {
            "schema_version": EXPERIMENT_CODEX_IPC_VERSION,
            "session_id": session_id,
            "expected_step": int(expected_step),
            "max_tool_output_bytes": self.max_tool_output_bytes,
            "response_timeout_s": float(response_timeout_s),
            "session_scope": session_scope,
            "terminal_action_readout_required": bool(terminal_action_readout_required),
        }
        _atomic_write_json(session_root / "session.json", descriptor)
        if session_scope == "campaign":
            _atomic_write_json(
                session_root / "campaign_progress.json",
                {
                    "schema_version": CAMPAIGN_PROGRESS_VERSION,
                    "closed_batch_count": 0,
                    "completed_experiment_count": 0,
                    "completed_experiment_indices": [],
                    "completed_batch_ids": [],
                    "observed_evidence_ids": [],
                    "campaign_ended": False,
                },
            )
        _atomic_write_json(self.active_session_path, descriptor)
        self._session_descriptors[session_id] = deepcopy(descriptor)
        return deepcopy(descriptor)

    def publish_campaign_progress(
        self,
        *,
        session_id: str,
        closed_batch_count: int,
        completed_experiment_count: int,
        completed_experiment_index: int | None,
        observed_evidence_id: str | None,
        campaign_ended: bool,
    ) -> dict[str, Any]:
        """Advance the small host-owned campaign ledger independently of bounded history."""

        descriptor = self._session_descriptor(session_id)
        if descriptor.get("session_scope") != "campaign":
            raise ExperimentCodexIPCError("campaign progress requires a campaign session")
        for value, label in (
            (closed_batch_count, "closed batch count"),
            (completed_experiment_count, "completed experiment count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ExperimentCodexIPCError(f"{label} must be non-negative")
        if completed_experiment_count > closed_batch_count:
            raise ExperimentCodexIPCError(
                "completed experiment count cannot exceed closed batch count"
            )
        path = self.session_root(session_id) / "campaign_progress.json"
        current = _read_json_object(path)
        if current.get("schema_version") != CAMPAIGN_PROGRESS_VERSION:
            raise ExperimentCodexIPCError("campaign progress schema is invalid")
        prior_closed = current.get("closed_batch_count")
        prior_count = current.get("completed_experiment_count")
        prior_indices = current.get("completed_experiment_indices")
        prior_batch_ids = current.get("completed_batch_ids")
        prior_evidence = current.get("observed_evidence_ids")
        prior_ended = current.get("campaign_ended")
        if (
            isinstance(prior_closed, bool)
            or not isinstance(prior_closed, int)
            or isinstance(prior_count, bool)
            or not isinstance(prior_count, int)
            or not isinstance(prior_indices, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in prior_indices
            )
            or prior_indices != sorted(set(prior_indices))
            or len(prior_indices) != prior_count
            or not isinstance(prior_batch_ids, list)
            or prior_batch_ids != [f"batch-{index:04d}" for index in prior_indices]
            or not isinstance(prior_evidence, list)
            or any(not isinstance(item, str) or not item for item in prior_evidence)
            or len(set(prior_evidence)) != len(prior_evidence)
            or len(prior_evidence) != prior_count
            or not isinstance(prior_ended, bool)
        ):
            raise ExperimentCodexIPCError("campaign progress state is malformed")
        if closed_batch_count not in {prior_closed, prior_closed + 1}:
            raise ExperimentCodexIPCError("campaign closed batch count is not monotonic")
        if completed_experiment_count not in {prior_count, prior_count + 1}:
            raise ExperimentCodexIPCError("campaign progress count is not monotonic")
        if completed_experiment_count == prior_count + 1 and closed_batch_count != prior_closed + 1:
            raise ExperimentCodexIPCError("a completed experiment must close one new batch")
        if closed_batch_count == prior_closed and completed_experiment_count != prior_count:
            raise ExperimentCodexIPCError("completion cannot advance without a new closed batch")
        indices = list(prior_indices)
        batch_ids = list(prior_batch_ids)
        evidence = list(prior_evidence)
        if completed_experiment_count == prior_count + 1:
            if (
                isinstance(completed_experiment_index, bool)
                or not isinstance(completed_experiment_index, int)
                or completed_experiment_index != closed_batch_count
            ):
                raise ExperimentCodexIPCError(
                    "completed experiment index must identify the newly closed lifecycle batch"
                )
            expected = f"experiment-{completed_experiment_count}-final-assay"
            if observed_evidence_id != expected:
                raise ExperimentCodexIPCError("campaign progress evidence ID is invalid")
            indices.append(completed_experiment_index)
            batch_ids.append(f"batch-{completed_experiment_index:04d}")
            evidence.append(expected)
        elif observed_evidence_id is not None or completed_experiment_index is not None:
            raise ExperimentCodexIPCError(
                "campaign progress added completion identity without completion"
            )
        if prior_ended and not campaign_ended:
            raise ExperimentCodexIPCError("campaign terminal state cannot be reversed")
        state = {
            "schema_version": CAMPAIGN_PROGRESS_VERSION,
            "closed_batch_count": closed_batch_count,
            "completed_experiment_count": completed_experiment_count,
            "completed_experiment_indices": indices,
            "completed_batch_ids": batch_ids,
            "observed_evidence_ids": evidence,
            "campaign_ended": campaign_ended,
        }
        _atomic_write_json(path, state)
        return deepcopy(state)

    def publish_current(self, packet: Mapping[str, Any]) -> dict[str, Any]:
        """Publish a bounded reconstructable view for ``lab_tool status``."""

        self._require_initialized()
        normalized = _bounded_json_object(
            packet,
            max_bytes=self.max_tool_output_bytes,
            label="current public state",
        )
        _atomic_write_json(self.current_path, normalized)
        return _fingerprint(self.current_path)

    def append_public_history(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Retain the public outcome durably, then refresh its bounded recent cache."""

        self._require_initialized()
        normalized = _bounded_json_object(
            event,
            max_bytes=self.max_tool_output_bytes,
            label="public history event",
        )
        rows = deepcopy(self._history_rows)
        rows.append(normalized)
        rows = rows[-self.history_event_limit :]
        while rows and len(_encode_jsonl(rows)) > self.history_byte_limit:
            rows.pop(0)
        if not rows:
            raise ExperimentCodexIPCError(
                "one public history event exceeds the configured cache budget"
            )
        with self.operation_history_path.open("ab") as handle:
            handle.write(_encode_jsonl([normalized]))
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_write_bytes(self.history_path, _encode_jsonl(rows))
        self._history_rows = rows
        return {
            **_fingerprint(self.history_path),
            "retained_event_count": len(rows),
            "authoritative": False,
        }

    def update_expected_step(self, *, session_id: str, expected_step: int) -> None:
        """Advance the public CAS counter after one runner operation."""

        if expected_step < 1:
            raise ValueError("expected_step must be positive")
        descriptor = deepcopy(self._session_descriptor(session_id))
        descriptor["expected_step"] = int(expected_step)
        _atomic_write_json(self.session_root(session_id) / "session.json", descriptor)
        # ``active_session.json`` is an immutable pointer for the lifetime of one
        # provider turn.  The MCP process reads it frequently to resolve the
        # session namespace, so replacing it after every operation creates an
        # unnecessary Windows reader/writer sharing race.  Mutable CAS state is
        # owned by the session descriptor and the host-side descriptor cache.
        self._session_descriptors[session_id] = deepcopy(descriptor)

    def wait_for_request(
        self,
        *,
        session_id: str,
        expected_step: int,
        timeout_s: float,
        process_alive: Callable[[], bool],
        handled_request_ids: set[str],
        progress_callback: Callable[[], None] | None = None,
        progress_interval_s: float = 30.0,
    ) -> IPCRequest:
        """Wait for one well-formed, unhandled request from the active Codex turn."""

        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if progress_interval_s <= 0:
            raise ValueError("progress_interval_s must be positive")
        request_roots = (
            self.session_root(session_id) / "mcp_requests",
            self.transport_session_root(session_id) / "requests",
        )
        deadline = time.monotonic() + timeout_s
        next_progress = time.monotonic() + progress_interval_s
        while time.monotonic() < deadline:
            request_paths = sorted(
                (path for root in request_roots for path in root.glob("*.json")),
                key=lambda item: (item.name, item.parent.as_posix()),
            )
            for path in request_paths:
                raw = _read_json_object(path)
                request = _normalize_request(raw, session_id=session_id)
                if request.request_id in handled_request_ids:
                    continue
                if request.expected_step != expected_step:
                    self.write_response(
                        session_id=session_id,
                        request_id=request.request_id,
                        response={
                            "ok": False,
                            "error": "expected_step_mismatch",
                            "expected_step": expected_step,
                            "submitted_step": request.expected_step,
                        },
                    )
                    handled_request_ids.add(request.request_id)
                    continue
                return request
            if not process_alive():
                raise ExperimentCodexIPCError(
                    "Codex process exited before submitting the next operation"
                )
            if progress_callback is not None and time.monotonic() >= next_progress:
                progress_callback()
                next_progress = time.monotonic() + progress_interval_s
            time.sleep(self.poll_interval_s)
        raise TimeoutError("timed out waiting for a Codex lab operation")

    def write_response(
        self,
        *,
        session_id: str,
        request_id: str,
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Write one immutable, bounded response for a request."""

        normalized = _bounded_json_object(
            response,
            max_bytes=self.max_tool_output_bytes,
            label="lab tool response",
        )
        path = self.response_path(session_id, request_id)
        if path.exists():
            existing = _read_json_object(path)
            if _canonical_json(existing) != _canonical_json(normalized):
                raise ExperimentCodexIPCError("idempotent request already has a different response")
            return _fingerprint(path)
        _atomic_write_json(path, normalized)
        return _fingerprint(path)

    def response_path(self, session_id: str, request_id: str) -> Path:
        if not request_id or any(char in request_id for char in "/\\"):
            raise ValueError("request_id must be a non-path string")
        return self.session_root(session_id) / "responses" / f"{request_id}.json"

    def retire_session(self, session_id: str) -> None:
        """Remove completed transport/response files so they cannot become a ledger."""

        self._require_initialized()
        session_root = self.session_root(session_id).resolve()
        transport_root = self.transport_session_root(session_id).resolve()
        for path, parent in (
            (session_root, self.sessions_directory.resolve()),
            (transport_root, self.transport_directory.resolve()),
        ):
            try:
                path.relative_to(parent)
            except ValueError as error:
                raise ExperimentCodexIPCError(
                    "refusing to retire a session outside its protocol root"
                ) from error
            for attempt in range(SESSION_RETIRE_RETRY_LIMIT):
                try:
                    if path.exists():
                        shutil.rmtree(path)
                    break
                except OSError:
                    if attempt + 1 == SESSION_RETIRE_RETRY_LIMIT:
                        raise
                    time.sleep(SESSION_RETIRE_RETRY_INTERVAL_S)
        active = (
            _read_json_object(self.active_session_path) if self.active_session_path.exists() else {}
        )
        if active.get("session_id") == session_id:
            self.active_session_path.unlink(missing_ok=True)
        request_file = self.agent_directory / "request.json"
        request_file.unlink(missing_ok=True)
        self._session_descriptors.pop(session_id, None)

    def session_root(self, session_id: str) -> Path:
        _validate_leaf_name(session_id, label="session_id")
        return self.sessions_directory / session_id

    def transport_session_root(self, session_id: str) -> Path:
        """Return the Agent-writable transport namespace for one session."""

        _validate_leaf_name(session_id, label="session_id")
        return self.transport_directory / session_id

    def snapshot_agent_files(self) -> dict[str, dict[str, Any]]:
        """Return content-minimized metadata for optional Agent-owned files."""

        self._require_initialized()
        snapshot: dict[str, dict[str, Any]] = {}
        for path in sorted(self.agent_directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.agent_directory).as_posix()
            if (
                relative in {"lab_tool.py", "request.json"}
                or relative == "python.cmd"
                or relative.startswith(".transport/")
            ):
                # These are generated bridge/transport files, not Agent-authored
                # scientific memory.
                continue
            snapshot[relative] = _fingerprint(path)
        return snapshot

    def manifest(self) -> dict[str, Any]:
        self._require_initialized()
        return {
            "schema_version": EXPERIMENT_CODEX_IPC_VERSION,
            "workspace_root": str(self.root),
            "agent_directory": {
                "relative_path": "agent",
                "owner": "agent",
                "writable": True,
                "required_file": True,
                "required_files": ["lab_tool.py"],
                "file_count": len(self.snapshot_agent_files()),
                "file_count_scope": ("agent_authored_memory_excluding_bridge_and_transport"),
            },
            "transport": {
                "relative_path": "agent/.transport",
                "owner": "agent",
                "writable": True,
                "authoritative": False,
                "contains_host_responses": False,
            },
            "public_cache": {
                "relative_path": "public",
                "authoritative": False,
                "max_tool_output_bytes": self.max_tool_output_bytes,
                "history_event_limit": self.history_event_limit,
                "history_byte_limit": self.history_byte_limit,
                "complete_history": "public/operation_history.jsonl",
                "history_pagination": "history offset=0 starts at the first public operation",
                "max_artifact_bytes": self.max_artifact_bytes,
            },
            "material_information": (
                {
                    "relative_path": "reference/material_information.json",
                    **_fingerprint(self.reference_directory / "material_information.json"),
                }
                if (self.reference_directory / "material_information.json").exists()
                else None
            ),
            "task_contract": (
                {
                    "relative_path": "reference/task_contract.json",
                    **_fingerprint(self.reference_directory / "task_contract.json"),
                }
                if (self.reference_directory / "task_contract.json").exists()
                else None
            ),
            "authoritative_trajectory_in_workspace": False,
            "lab_tool": {
                "relative_path": "agent/lab_tool.py",
                "owner": "host_generated",
                "writable_by_topology": True,
                "authoritative": False,
                "integrity_policy": (
                    "exact_source_sha256_before_every_accepted_action_and_after_session"
                ),
                "expected_sha256": _lab_tool_sha256(),
                **_fingerprint(self.lab_tool_path),
            },
            "python_command_shim": (
                {
                    "relative_path": "agent/python.cmd",
                    "owner": "host_generated",
                    "authoritative": False,
                    **_fingerprint(self.agent_directory / "python.cmd"),
                }
                if (self.agent_directory / "python.cmd").is_file()
                else None
            ),
        }

    def verify_lab_tool(self) -> None:
        self._require_initialized()
        try:
            valid_regular_file = (
                self.lab_tool_path.is_file() and not self.lab_tool_path.is_symlink()
            )
            actual = str(_fingerprint(self.lab_tool_path)["sha256"]) if valid_regular_file else None
        except OSError:
            actual = None
        if actual != _lab_tool_sha256():
            raise ExperimentCodexIPCError("lab_tool.py changed during the Codex session")
        python_shim = self.agent_directory / "python.cmd"
        if python_shim.is_file() and _fingerprint(python_shim)["sha256"] != _python_shim_sha256():
            raise ExperimentCodexIPCError("python.cmd changed during the Codex session")

    def verify_file(self, *, relative_path: str, expected_sha256: str) -> None:
        """Verify one host-published workspace file immediately before an action."""

        path = (self.root / relative_path).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("relative_path escapes the workspace") from error
        if not path.is_file() or _fingerprint(path)["sha256"] != expected_sha256:
            raise ExperimentCodexIPCError(f"host-published workspace file changed: {relative_path}")

    def _session_descriptor(self, session_id: str) -> dict[str, Any]:
        try:
            return deepcopy(self._session_descriptors[session_id])
        except KeyError as error:
            raise ExperimentCodexIPCError(f"unknown IPC session: {session_id}") from error

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("interactive Codex workspace is not initialized")


def diff_agent_snapshots(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Describe optional Agent memory writes without retaining their contents."""

    before_keys = set(before)
    after_keys = set(after)
    created = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)
    modified = sorted(
        path
        for path in before_keys & after_keys
        if before[path].get("sha256") != after[path].get("sha256")
    )
    return {
        "created": created,
        "modified": modified,
        "deleted": deleted,
        "bytes_before": sum(int(item.get("byte_count", 0)) for item in before.values()),
        "bytes_after": sum(int(item.get("byte_count", 0)) for item in after.values()),
    }


def _normalize_request(raw: Mapping[str, Any], *, session_id: str) -> IPCRequest:
    if raw.get("schema_version") != EXPERIMENT_CODEX_IPC_VERSION:
        raise ExperimentCodexIPCError("unsupported IPC request schema")
    if raw.get("session_id") != session_id:
        raise ExperimentCodexIPCError("IPC request belongs to another session")
    request_id = raw.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ExperimentCodexIPCError("IPC request lacks request_id")
    expected_step = raw.get("expected_step")
    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step < 1:
        raise ExperimentCodexIPCError("IPC request expected_step must be positive")
    action = raw.get("action")
    if not isinstance(action, dict) or not isinstance(action.get("operation"), str):
        raise ExperimentCodexIPCError("IPC request requires action.operation")
    normalized_action = to_builtin(action)
    raw_decision_audit = raw.get("decision_audit")
    if raw_decision_audit is not None and not isinstance(raw_decision_audit, dict):
        raise ExperimentCodexIPCError("IPC request decision_audit must be an object or null")
    decision_audit = to_builtin(raw_decision_audit) if raw_decision_audit is not None else None
    payload_sha256 = hashlib.sha256(_canonical_json(normalized_action).encode("utf-8")).hexdigest()
    return IPCRequest(
        session_id=session_id,
        request_id=request_id.strip(),
        expected_step=expected_step,
        action=normalized_action,
        payload_sha256=payload_sha256,
        decision_audit=decision_audit,
    )


def _validate_leaf_name(value: str, *, label: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or any(char in value for char in "/\\")
    ):
        raise ValueError(f"{label} must be a non-path string")


def _bounded_json_object(
    value: Mapping[str, Any],
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    normalized = to_builtin(dict(value))
    if not isinstance(normalized, dict):
        raise TypeError(f"{label} must normalize to an object")
    encoded = _canonical_json(normalized).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ExperimentCodexIPCError(
            f"{label} is {len(encoded)} bytes, above hard cap {max_bytes}"
        )
    return normalized


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentCodexIPCError(f"invalid bridge JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ExperimentCodexIPCError(f"bridge JSON is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExperimentCodexIPCError("public history cache is invalid JSONL") from error
        if not isinstance(value, dict):
            raise ExperimentCodexIPCError("public history row is not an object")
        rows.append(value)
    return rows


def _encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(f"{_canonical_json(row)}\n" for row in rows).encode("utf-8") if rows else b""


def _fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "byte_count": len(data),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _lab_tool_sha256() -> str:
    return hashlib.sha256(LAB_TOOL_SOURCE.encode("utf-8")).hexdigest()


def _python_shim_source() -> str:
    executable = str(Path(sys.executable).resolve())
    return f'@"{executable}" %*\r\n'


def _python_shim_sha256() -> str:
    return hashlib.sha256(_python_shim_source().encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(path, _canonical_json(to_builtin(dict(value))) + "\n")


def _atomic_write_text(path: Path, value: str) -> None:
    _atomic_write_bytes(path, value.encode("utf-8"))


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _replace_with_retry(temporary: Path, path: Path) -> None:
    """Bound transient Windows sharing violations without hiding hard failures."""

    for retry in range(ATOMIC_REPLACE_RETRY_LIMIT + 1):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if retry >= ATOMIC_REPLACE_RETRY_LIMIT:
                raise
            time.sleep(ATOMIC_REPLACE_RETRY_INTERVAL_S)


LAB_TOOL_SOURCE = r'''"""Isolated ChemWorld file-IPC client.  Standard library only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

VERSION = "chemworld-experiment-codex-ipc-0.2"
ATOMIC_REPLACE_RETRY_LIMIT = 40
ATOMIC_REPLACE_RETRY_INTERVAL_S = 0.025
TOOL = Path(__file__).resolve(strict=True)
AGENT = TOOL.parent
ROOT = AGENT.parent.resolve(strict=True)
if AGENT.name != "agent" or TOOL.name != "lab_tool.py":
    raise RuntimeError("lab_tool.py must run from WORKSPACE/agent/lab_tool.py")
if AGENT != (ROOT / "agent").resolve(strict=True):
    raise RuntimeError("lab_tool.py resolved outside the expected agent root")
TRANSPORT = AGENT / ".transport"
IPC = ROOT / ".ipc"
PUBLIC = ROOT / "public"
ARTIFACTS = PUBLIC / "artifacts"


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} is not a JSON object")
    return value


def encode(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def emit(value: object, *, cap: int) -> None:
    data = encode(value)
    if len(data) > cap:
        raise ValueError(f"tool output exceeds {cap} byte cap")
    sys.stdout.buffer.write(data + b"\n")
    sys.stdout.buffer.flush()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encode(value) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace_with_retry(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def replace_with_retry(temp: Path, path: Path) -> None:
    for retry in range(ATOMIC_REPLACE_RETRY_LIMIT + 1):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if retry >= ATOMIC_REPLACE_RETRY_LIMIT:
                raise
            time.sleep(ATOMIC_REPLACE_RETRY_INTERVAL_S)


def descriptor() -> dict:
    value = read_object(IPC / "active_session.json")
    if value.get("schema_version") != VERSION:
        raise ValueError("unsupported active session")
    return value


def within_agent(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(AGENT)
    except ValueError as error:
        raise ValueError("request file must be inside agent/") from error
    return resolved


def status(_: argparse.Namespace) -> None:
    active = descriptor()
    emit(read_object(PUBLIC / "current.json"), cap=int(active["max_tool_output_bytes"]))


# PUBLIC_HISTORY_READER


def history(args: argparse.Namespace) -> None:
    active = descriptor()
    emit(
        {"schema_version": VERSION, **public_history_page(
            PUBLIC, limit=args.limit, offset=args.offset,
            max_bytes=int(active["max_tool_output_bytes"]) - 128
        )},
        cap=int(active["max_tool_output_bytes"]),
    )


def append_access(session_root: Path, value: dict) -> None:
    path = session_root / "artifact_access.jsonl"
    line = encode(value) + b"\n"
    with path.open("ab") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def inspect_artifact(args: argparse.Namespace) -> None:
    active = descriptor()
    cap = int(active["max_tool_output_bytes"])
    artifact_id = str(args.artifact_id)
    if (
        not artifact_id
        or artifact_id in {".", ".."}
        or "/" in artifact_id
        or "\\" in artifact_id
    ):
        raise ValueError("artifact_id must be a non-path string")
    path = ARTIFACTS / f"{artifact_id}.json"
    text = path.read_text(encoding="utf-8")
    offset = max(int(args.offset), 0)
    maximum = max(256, cap // 2)
    limit = max(1, min(int(args.limit), maximum))
    fragment = text[offset : offset + limit]
    session_root = TRANSPORT / str(active["session_id"])
    append_access(
        session_root,
        {
            "artifact_id": artifact_id,
            "offset": offset,
            "limit": limit,
            "returned_character_count": len(fragment),
            "total_character_count": len(text),
        },
    )
    emit(
        {
            "schema_version": VERSION,
            "artifact_id": artifact_id,
            "offset": offset,
            "next_offset": offset + len(fragment),
            "total_character_count": len(text),
            "complete": offset + len(fragment) >= len(text),
            "encoding": "utf-8-json-text-fragment",
            "data": fragment,
        },
        cap=cap,
    )


def _submit_request(args: argparse.Namespace) -> tuple[dict, Path]:
    active = descriptor()
    session_id = str(active["session_id"])
    if args.action_json is not None:
        if args.expected_step is None:
            raise ValueError("--expected-step is required with --action-json")
        action = json.loads(args.action_json)
        request = {
            "expected_step": args.expected_step,
            "action": action,
        }
        if args.request_id is not None:
            request["request_id"] = args.request_id
    else:
        if args.request_file is None:
            raise ValueError("step requires --action-json or --request-file")
        request = read_object(within_agent(Path(args.request_file)))
    expected_step = request.get("expected_step")
    action = request.get("action")
    if isinstance(expected_step, bool) or not isinstance(expected_step, int) or expected_step < 1:
        raise ValueError("request expected_step must be a positive integer")
    if not isinstance(action, dict) or not isinstance(action.get("operation"), str):
        raise ValueError("request requires action.operation")
    canonical_action = encode(action)
    request_id = request.get("request_id")
    if request_id is None:
        request_id = hashlib.sha256(
            session_id.encode("utf-8")
            + b"|"
            + str(expected_step).encode("ascii")
            + b"|"
            + canonical_action
        ).hexdigest()[:32]
    if not isinstance(request_id, str) or not request_id or "/" in request_id or "\\" in request_id:
        raise ValueError("request_id must be a non-path string")
    envelope = {
        "schema_version": VERSION,
        "session_id": session_id,
        "request_id": request_id,
        "expected_step": expected_step,
        "action": action,
    }
    request_root = TRANSPORT / session_id / "requests"
    response_root = IPC / "sessions" / session_id / "responses"
    request_path = request_root / f"{request_id}.json"
    response_path = response_root / f"{request_id}.json"
    if request_path.exists():
        if encode(read_object(request_path)) != encode(envelope):
            raise ValueError("request_id was reused with another payload")
    else:
        atomic_json(request_path, envelope)
    return envelope, response_path


def submit(args: argparse.Namespace) -> None:
    active = descriptor()
    cap = int(active["max_tool_output_bytes"])
    envelope, _ = _submit_request(args)
    emit(
        {
            "schema_version": VERSION,
            "ok": True,
            "submitted": True,
            "request_id": envelope["request_id"],
            "expected_step": envelope["expected_step"],
        },
        cap=cap,
    )


def poll(args: argparse.Namespace) -> None:
    active = descriptor()
    cap = int(active["max_tool_output_bytes"])
    request_id = str(args.request_id)
    if not request_id or request_id in {".", ".."} or "/" in request_id or "\\" in request_id:
        raise ValueError("request_id must be a non-path string")
    response_path = (
        IPC / "sessions" / str(active["session_id"]) / "responses" / f"{request_id}.json"
    )
    if response_path.exists():
        emit(read_object(response_path), cap=cap)
        return
    emit(
        {
            "schema_version": VERSION,
            "ok": True,
            "pending": True,
            "request_id": request_id,
        },
        cap=cap,
    )


def step(args: argparse.Namespace) -> None:
    active = descriptor()
    session_id = str(active["session_id"])
    cap = int(active["max_tool_output_bytes"])
    envelope, response_path = _submit_request(args)
    deadline = time.monotonic() + float(active["response_timeout_s"])
    while time.monotonic() < deadline:
        if response_path.exists():
            emit(read_object(response_path), cap=cap)
            return
        current = descriptor()
        if current.get("session_id") != session_id:
            raise RuntimeError("session changed before the operation completed")
        time.sleep(0.05)
    raise TimeoutError("runner did not return an operation outcome")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    status_parser = commands.add_parser("status")
    status_parser.set_defaults(func=status)
    history_parser = commands.add_parser("history")
    history_parser.add_argument("--limit", type=int, default=5)
    history_parser.add_argument("--offset", type=int)
    history_parser.set_defaults(func=history)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--artifact-id", required=True)
    inspect_parser.add_argument("--offset", type=int, default=0)
    inspect_parser.add_argument("--limit", type=int, default=4096)
    inspect_parser.set_defaults(func=inspect_artifact)
    step_parser = commands.add_parser("step")
    source = step_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--action-json")
    source.add_argument("--request-file")
    step_parser.add_argument("--expected-step", type=int)
    step_parser.add_argument("--request-id")
    step_parser.set_defaults(func=step)
    submit_parser = commands.add_parser("submit")
    submit_source = submit_parser.add_mutually_exclusive_group(required=True)
    submit_source.add_argument("--action-json")
    submit_source.add_argument("--request-file")
    submit_parser.add_argument("--expected-step", type=int)
    submit_parser.add_argument("--request-id")
    submit_parser.set_defaults(func=submit)
    poll_parser = commands.add_parser("poll")
    poll_parser.add_argument("--request-id", required=True)
    poll_parser.set_defaults(func=poll)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.func(args)
    except Exception as error:
        active = {}
        try:
            active = descriptor()
        except Exception:
            pass
        cap = int(active.get("max_tool_output_bytes", 32768))
        emit({"ok": False, "error": type(error).__name__}, cap=cap)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


# Keep the standalone, standard-library bridge and MCP on the same reader.
LAB_TOOL_SOURCE = LAB_TOOL_SOURCE.replace(
    "# PUBLIC_HISTORY_READER", inspect.getsource(public_history_page)
)


__all__ = [
    "DEFAULT_HISTORY_BYTE_LIMIT",
    "DEFAULT_HISTORY_EVENT_LIMIT",
    "DEFAULT_MAX_ARTIFACT_BYTES",
    "DEFAULT_MAX_TOOL_OUTPUT_BYTES",
    "EXPERIMENT_CODEX_IPC_VERSION",
    "ExperimentCodexIPCError",
    "ExperimentCodexWorkspace",
    "IPCRequest",
    "diff_agent_snapshots",
]
