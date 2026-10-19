"""Preserve frozen evidence while allowing current runtime development."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_historical_evidence_still_matches_its_frozen_manifest() -> None:
    manifest = json.loads((ROOT / "release/manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_version"] == "0.4.0"
    rows = [row for row in manifest["files"] if row["path"].startswith("evidence/")]
    assert rows
    for row in rows:
        content = (ROOT / row["path"]).read_bytes()
        assert len(content) == row["bytes"], row["path"]
        assert hashlib.sha256(content).hexdigest() == row["sha256"], row["path"]


def test_historical_evidence_denominators_failures_and_replay() -> None:
    spec = importlib.util.spec_from_file_location(
        "chemworld_frozen_verifier", ROOT / "scripts/verify_release.py"
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    verifier.verify_evidence()


def test_research_only_interfaces_and_inputs_stay_out_of_public_tree() -> None:
    for relative in (
        "src/chemworld/eval/paper_artifact.py",
        "src/chemworld/eval/first_paper_u05_complete_agent.py",
        "src/chemworld/physchem/abductive_worlds.py",
        "configs/current.json",
        "configs/providers",
        "workstreams",
        "paper",
    ):
        assert not (ROOT / relative).exists(), relative
    completed = subprocess.run(
        [sys.executable, "-m", "chemworld.cli", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "paper/preprint" not in completed.stdout
    assert "lab" in completed.stdout
