from __future__ import annotations

import json

import pytest

from chemworld.agents.experiment_documents import ExperimentDocumentWorkspace


def workspace(path):
    result = ExperimentDocumentWorkspace(path, versioned_notebook=True)
    result.initialize()
    return result


def test_restore_retains_history_evidence_boundary_and_facts(tmp_path):
    w = workspace(tmp_path)
    w.append_operation({"event_id": "e1", "observation": 1})
    w.notebook_tool("write", text="# Question\nMaybe A.\n", reviewed_through="e1")
    w.append_operation({"event_id": "e2", "observation": 2})
    w.notebook_tool("write", text="# Question\nMaybe B.\n", reviewed_through="e2")
    ledger = w.authoritative_path.read_bytes()
    restored = w.notebook_tool("restore", revision=1)
    assert restored["revision"] == 3
    assert restored["restored_from"] == 1
    assert restored["reviewed_through"] == "e1"
    assert restored["public_cursor"]["last_event_id"] == "e2"
    assert w.authoritative_path.read_bytes() == ledger
    assert w.read_notebook() == w.read_notebook(1)
    assert "Maybe B" in w.read_notebook(2)
    delta = w.notebook_tool("diff", before=2, after=3)
    assert "-Maybe B." in delta["text"] and "+Maybe A." in delta["text"]
    assert w.notebook_tool("log")["total"] == 3
    assert "Maybe A" not in json.dumps(w.manifest())
    assert "Maybe A" not in json.dumps(w.notebook_tool("log"))


def test_commit_survives_view_failure_and_restart(tmp_path, monkeypatch):
    import chemworld.agents.experiment_documents as module

    w = workspace(tmp_path)
    original = module._atomic_write_text

    def fail_view(path, text):
        if path == w.notebook_path:
            raise OSError("injected view write failure")
        return original(path, text)

    with monkeypatch.context() as m:
        m.setattr(module, "_atomic_write_text", fail_view)
        result = w.notebook_tool("write", text="saved conclusion")
    assert result["revision"] == 1 and result["view_synced"] is False
    assert w.read_notebook() == "saved conclusion"
    restarted = workspace(tmp_path)
    assert restarted.notebook_path.read_text(encoding="utf-8") == "saved conclusion"


def test_on_demand_pagination_import_and_invalid_requests(tmp_path):
    legacy = ExperimentDocumentWorkspace(tmp_path)
    legacy.initialize()
    legacy.write_notebook("earlier unversioned notes")
    w = workspace(tmp_path)
    result = w.notebook_tool("write", text="新" * 20)
    assert result["revision"] == 2
    assert w.notebook_tool("read", revision=1)["kind"] == "import"
    page = w.notebook_tool("read", limit=7)
    assert page["text"] == "新" * 7 and page["next_offset"] == 7
    assert page["truncated"]
    assert w.notebook_tool("log", limit=1)["next_offset"] == 1
    for operation, args in [
        ("read", {"revision": "../1"}),
        ("restore", {"revision": 999}),
        ("write", {"text": "x", "public_cursor": {}}),
        ("write", {"text": "x", "reviewed_through": "future"}),
        ("read", {"offset": -1}),
        ("log", {"limit": 0}),
    ]:
        with pytest.raises((ValueError, TypeError)):
            w.notebook_tool(operation, **args)
    assert w.notebook_tool("log")["total"] == 2
    with pytest.raises(ValueError, match="new run"):
        w.reset()


def test_diff_keeps_unterminated_lines_readable_and_detects_newline_only_change(tmp_path):
    w = workspace(tmp_path)
    w.notebook_tool("write", text="H1: unresolved")
    w.notebook_tool("write", text="H1: narrower scope")
    diff = w.notebook_tool("diff")["text"]
    assert "-H1: unresolved\n\\ No newline at end of file\n" in diff
    assert "+H1: narrower scope\n\\ No newline at end of file\n" in diff
    assert "unresolved+H1" not in diff
    w.notebook_tool("write", text="H1: narrower scope\n")
    diff = w.notebook_tool("diff")["text"]
    assert "-H1: narrower scope\n\\ No newline at end of file\n" in diff
    assert "+H1: narrower scope\n" in diff
    assert diff.count("No newline") == 1
