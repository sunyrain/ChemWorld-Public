from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_first_experiment.ipynb",
    "02_reaction_to_purification.ipynb",
    "03_controlled_world_change.ipynb",
)
PAGES = (
    "index",
    "quickstart",
    "agents",
    "student-lab",
    "one-experiment",
    "world-foundations",
    "evidence",
    "notebooks",
    "reference",
    "limitations",
)
_INTERNAL_WORKSTREAM = "workstreams" + "/arxiv_v1"
_INTERNAL_TODO = "FIRST_PAPER" + "_TODOLIST"
FORBIDDEN_TUTORIAL = re.compile(
    r"private-eval|"
    + re.escape(_INTERNAL_WORKSTREAM)
    + "|"
    + re.escape(_INTERNAL_TODO)
    + r"|/(?:root|home|mnt)/(?:[^\s\"']+)|"
    + r"(?:sk-|gh[pousr]_)[A-Za-z0-9_-]{16,}",
    re.IGNORECASE,
)


def test_bilingual_public_site_routes_are_complete() -> None:
    for page in PAGES:
        assert (ROOT / "docs" / f"{page}.md").is_file()
        assert (ROOT / "docs" / f"{page}.zh.md").is_file()
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "default: true" in config
    assert "locale: en" in config
    assert "locale: zh" in config
    assert "https://sunyrain.github.io/ChemWorld-Public/" in config


def test_notebooks_retain_only_executed_public_tutorial_outputs() -> None:
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        raw = path.read_text(encoding="utf-8")
        assert FORBIDDEN_TUTORIAL.search(raw) is None
        notebook = json.loads(raw)
        release = notebook["metadata"]["chemworld"]
        assert release["runtime_release"] == "0.3.0"
        assert release["output_scope"] == "deterministic_public_tutorial"
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        assert code_cells
        assert all(cell.get("execution_count") is not None for cell in code_cells)
        assert all(
            output.get("output_type") != "error"
            for cell in code_cells
            for output in cell.get("outputs", [])
        )


def test_colab_links_are_pinned_to_the_release_tag() -> None:
    joined = "\n".join(
        [
            (ROOT / "README.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "notebooks.md").read_text(encoding="utf-8"),
            (ROOT / "docs" / "notebooks.zh.md").read_text(encoding="utf-8"),
        ]
    )
    for name in NOTEBOOKS:
        expected = f"/blob/v0.3.0/notebooks/{name}"
        assert joined.count(expected) >= 3


def test_interactive_payload_matches_readme_visual_payload() -> None:
    derived = json.loads(
        (ROOT / "evidence" / "derived" / "representative-behavior-and-forks.json").read_text(
            encoding="utf-8"
        )
    )
    site = json.loads(
        (ROOT / "docs" / "assets" / "data" / "representative-behavior-and-forks.json").read_text(
            encoding="utf-8"
        )
    )
    assert site == derived
    assert len(site["agent_lifecycle"]["actions"]) == 15
    assert site["rollback_recovery"]["summary"]["rolled_back_actions"] == 1
    assert site["controlled_world_forks"]["summary"]["pairs"] == 6


def test_public_readme_and_documentation_keep_distinct_entry_points() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    hero_marker = "docs/assets/chemworld-readme-banner.png"
    evidence_marker = "docs/assets/representative-agent-and-world-change.svg"
    assert (ROOT / hero_marker).is_file()
    assert readme.index(hero_marker) < readme.index("## Quick start")
    assert readme.index(evidence_marker) > readme.index("### Frozen evidence")

    for suffix in ("", ".zh"):
        home = (ROOT / "docs" / f"index{suffix}.md").read_text(encoding="utf-8")
        assert "environment-authoring.md" in home
        assert "quickstart.md" in home
        assert "reference.md" in home
        assert "https://sunyrain.github.io/ChemWorld-Public/?lang=" in home
        assert "data-cw-autoplay" not in home

    script = (ROOT / "docs/assets/javascripts/experiment-explorer.js").read_text(
        encoding="utf-8"
    )
    assert "initializeAutoplay" in script
    assert 'document.querySelectorAll("[data-cw-autoplay]")' in script


def test_agent_guide_exposes_custom_offline_and_live_paths() -> None:
    for suffix in ("", ".zh"):
        guide = (ROOT / "docs" / f"agents{suffix}.md").read_text(encoding="utf-8")
        assert "BaseAgent" in guide
        assert "run_agent" in guide
        assert "LiveLLMAgent" in guide
        assert "DEEPSEEK_API_KEY" in guide
        assert "CodexSubscriptionClient" in guide
        assert "private reasoning" in guide or "隐藏推理" in guide

    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Connect an agent: agents.md" in config
