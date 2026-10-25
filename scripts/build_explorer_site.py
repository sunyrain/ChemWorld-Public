"""Build the bilingual static explorer from public documentation and test records."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    "native_foundation",
    "native_domains",
    "native_capabilities",
    "native_mechanisms",
    "native_shared_lab",
    "native_biochemical",
    "native_stochastic",
    "world-foundations",
    "operations",
    "agents",
    "student-lab",
    "environment",
    "environment-authoring",
    "legacy",
    "native_cpu",
    "limitations",
    "quickstart",
    "evidence",
    "current-foundation",
    "extension-roadmap",
    "reference",
)


def guide_index(name: str, language: str) -> dict:
    source = ROOT / "docs" / f"{name}{'.zh' if language == 'zh' else ''}.md"
    prose = source.read_text(encoding="utf-8")
    heading = next(line[2:].strip() for line in prose.splitlines() if line.startswith("# "))
    return {"id": name, "source": source.relative_to(ROOT).as_posix(), "title": heading}


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    web = ROOT / "web"
    for source in web.iterdir():
        if source.is_file() and source.suffix in {".html", ".js", ".css"}:
            shutil.copyfile(source, output / source.name)
    for source in (web / "assets").rglob("*.webp"):
        destination = output / source.relative_to(web)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    locales = {name: {lang: guide_index(name, lang) for lang in ("zh", "en")} for name in GUIDES}
    (output / "chemworld-guide-index.js").write_text(
        f"window.CHEMWORLD_DOC_INDEX = {encode(locales)};\n", encoding="utf-8"
    )
    content = {
        "guides": [
            {"id": name, **{lang: editions[lang]["title"] for lang in ("zh", "en")}}
            for name, editions in locales.items()
        ],
        "fragments": {
            lang: (web / "templates" / f"explorer.{lang}.html").read_text(encoding="utf-8")
            for lang in ("zh", "en")
        },
    }
    (output / "chemworld-site-content.json").write_text(encode(content), encoding="utf-8")
    source = "tests/fixtures/golden/core_scripted_trajectories.json"
    fixture = json.loads((ROOT / source).read_text(encoding="utf-8"))
    records = []
    for summary in fixture["summaries"]:
        steps = []
        for row in summary["steps"]:
            observations = {
                key: value
                for key, value in row["observation"].items()
                if row["observed_mask"].get(key) is True
                and isinstance(value, (int, float))
                and math.isfinite(value)
            }
            steps.append(
                {
                    "step": row["step"],
                    "action": row["action"],
                    "observation": observations,
                    "observedKeys": row["observed_keys"],
                    "status": row["transaction_status"],
                    "rollbackReason": row["rollback_reason"],
                    "sampleConsumed": row["sample_consumed"],
                    "measurementCost": row["measurement_cost"],
                    "terminated": row["terminated"],
                    "truncated": row["truncated"],
                }
            )
        records.append(
            {
                "id": summary["task_id"],
                "source": source,
                "kind": "historical-scripted-test-record",
                "agent": summary["agent_name"],
                "mode": summary["episode_mode"],
                "steps": steps,
            }
        )
    payload = {
        "source": source,
        "description": "Historical public test observations, not current scientific validation.",
        "records": records,
    }
    (output / "chemworld-site-records.json").write_text(encode(payload), encoding="utf-8")
    print(f"Indexed {len(GUIDES)} bilingual documentation links and {len(records)} examples.")
    print(f"Website output: {output}")
    print("Static records only; no experiments executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    build(parser.parse_args().output.resolve())
