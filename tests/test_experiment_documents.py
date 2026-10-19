from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from chemworld.agents.experiment_documents import (
    DEFAULT_NOTEBOOK_TEXT,
    AuthoritativeDocumentIntegrityError,
    ExperimentDocumentWorkspace,
)


def test_initialize_creates_relative_manifest_without_ledger_text(
    tmp_path: Path,
) -> None:
    workspace = ExperimentDocumentWorkspace(tmp_path / "run with spaces")

    manifest = workspace.initialize()

    authoritative = manifest["authoritative_ledger"]
    notebook = manifest["model_notebook"]
    assert authoritative["relative_path"] == (
        "experiment_documents/environment_authoritative_ledger.jsonl"
    )
    assert notebook["relative_path"] == (
        "experiment_documents/model_owned_notebook.md"
    )
    assert authoritative["line_count"] == 0
    assert authoritative["byte_count"] == 0
    assert authoritative["last_event_id"] is None
    assert authoritative["agent_writable"] is False
    assert notebook["agent_writable"] is True
    assert workspace.read_notebook() == DEFAULT_NOTEBOOK_TEXT
    assert "contents" not in json.dumps(manifest).lower()


def test_host_append_is_canonical_persistent_and_restart_verifiable(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    workspace = ExperimentDocumentWorkspace(run_directory)
    workspace.initialize()

    manifest = workspace.append_operation(
        {
            "event_id": "operation-0001",
            "action": {"volume_L": 0.02, "operation": "add_solvent"},
            "committed": True,
        }
    )

    authoritative = manifest["authoritative_ledger"]
    line = workspace.authoritative_path.read_text(encoding="utf-8")
    assert line == (
        '{"action":{"operation":"add_solvent","volume_L":0.02},'
        '"committed":true,"event_id":"operation-0001"}\n'
    )
    assert authoritative["line_count"] == 1
    assert authoritative["last_event_id"] == "operation-0001"
    restarted = ExperimentDocumentWorkspace(run_directory)
    restarted_manifest = restarted.initialize(
        expected_authoritative_sha256=authoritative["sha256"]
    )
    assert restarted_manifest["authoritative_ledger"] == authoritative


def test_model_notebook_is_writable_without_mutating_authoritative_ledger(
    tmp_path: Path,
) -> None:
    workspace = ExperimentDocumentWorkspace(tmp_path / "run")
    initial = workspace.initialize()
    authoritative_hash = initial["authoritative_ledger"]["sha256"]

    manifest = workspace.write_notebook("# Working hypothesis\n\nTry lower heat.\n")

    assert workspace.read_notebook() == "# Working hypothesis\n\nTry lower heat.\n"
    assert manifest["authoritative_ledger"]["sha256"] == authoritative_hash
    assert manifest["model_notebook"]["sha256"] == hashlib.sha256(
        workspace.notebook_path.read_bytes()
    ).hexdigest()


def test_authoritative_tampering_blocks_notebook_and_host_append(
    tmp_path: Path,
) -> None:
    workspace = ExperimentDocumentWorkspace(tmp_path / "run")
    workspace.initialize()
    workspace.append_operation({"event_id": "operation-0001", "committed": True})
    workspace.authoritative_path.write_text(
        '{"event_id":"operation-tampered","committed":false}\n',
        encoding="utf-8",
    )

    with pytest.raises(AuthoritativeDocumentIntegrityError):
        workspace.verify_authoritative_integrity()
    with pytest.raises(AuthoritativeDocumentIntegrityError):
        workspace.write_notebook("tamper must not be accepted")
    with pytest.raises(AuthoritativeDocumentIntegrityError):
        workspace.append_operation(
            {"event_id": "operation-0002", "committed": True}
        )


def test_reset_is_explicit_and_clears_both_documents(tmp_path: Path) -> None:
    workspace = ExperimentDocumentWorkspace(tmp_path / "run")
    workspace.initialize()
    workspace.append_operation({"event_id": "operation-0001", "committed": True})
    workspace.write_notebook("model notes")

    manifest = workspace.reset()

    assert workspace.authoritative_path.read_bytes() == b""
    assert workspace.read_notebook() == DEFAULT_NOTEBOOK_TEXT
    assert manifest["authoritative_ledger"]["line_count"] == 0
    assert manifest["authoritative_ledger"]["last_event_id"] is None


def test_authoritative_events_require_unique_ids_and_finite_json(
    tmp_path: Path,
) -> None:
    workspace = ExperimentDocumentWorkspace(tmp_path / "run")
    workspace.initialize()
    workspace.append_operation({"event_id": "operation-0001", "reward": 0.0})

    with pytest.raises(ValueError, match="must be unique"):
        workspace.append_operation({"event_id": "operation-0001", "reward": 0.1})
    workspace.append_operation({"event_id": "operation-0002", "reward": 0.2})
    with pytest.raises(ValueError, match="must be unique"):
        workspace.append_operation({"event_id": "operation-0001", "reward": 0.3})
    with pytest.raises(ValueError, match="finite JSON"):
        workspace.append_operation({"event_id": "operation-0003", "reward": float("nan")})
    assert workspace.manifest()["authoritative_ledger"]["line_count"] == 2
