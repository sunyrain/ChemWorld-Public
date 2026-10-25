"""Build the bilingual static explorer from public documentation and test records."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import markdown

from chemworld.examples.native_capabilities import capability_rows

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


def render(name: str, language: str) -> dict:
    source = ROOT / "docs" / f"{name}{'.zh' if language == 'zh' else ''}.md"
    if not source.is_file():
        raise ValueError(f"Missing {language} guide: {source.relative_to(ROOT)}")
    prose = source.read_text(encoding="utf-8")
    engine = markdown.Markdown(
        extensions=[
            "toc",
            "tables",
            "attr_list",
            "md_in_html",
            "admonition",
            "footnotes",
            "pymdownx.details",
            "pymdownx.superfences",
            "pymdownx.highlight",
            "pymdownx.tabbed",
        ],
        extension_configs={
            "toc": {"permalink": False},
            "pymdownx.highlight": {"css_class": "codehilite"},
            "pymdownx.tabbed": {"alternate_style": True},
        },
    )
    html = engine.convert(prose)
    heading = next(line[2:].strip() for line in prose.splitlines() if line.startswith("# "))
    return {
        "id": name,
        "source": source.relative_to(ROOT).as_posix(),
        "title": heading,
        "html": html,
        "toc": engine.toc_tokens,
        "markdown": prose,
        "language": language,
        "reference": False,
        "translation": "local-preview" if "中文预览译文" in prose else "source",
    }


def encode(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    web = ROOT / "web"
    for source in web.iterdir():
        if source.is_file() and source.suffix in {".html", ".js", ".css"}:
            shutil.copyfile(source, output / source.name)
    rows = capability_rows()
    registry = {
        "modules": len(rows),
        "components": len({name for row in rows for name in row["components"]}),
        "operations": len({name for row in rows for name in row["operations"]}),
        "measurements": len({name for row in rows for name in row["instruments"]}),
    }
    for name in ("chemworld-capabilities.js", "chemworld-capabilities.en.js"):
        path = output / name
        prefix, payload = path.read_text(encoding="utf-8").split(" = ", 1)
        catalog = json.loads(payload.strip().removesuffix(";"))
        catalog["registry"] = registry
        path.write_text(f"{prefix} = {encode(catalog)};\n", encoding="utf-8")
    for source in (web / "assets").rglob("*.webp"):
        destination = output / source.relative_to(web)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    locales = {name: {lang: render(name, lang) for lang in ("zh", "en")} for name in GUIDES}
    pages = {name: editions["en"] for name, editions in locales.items()}
    # Map both actual source paths to the same guide; the reader preserves locale.
    pages.update({name + ".en": editions["zh"] for name, editions in locales.items()})
    for filename, variable, value in (
        ("chemworld-guide-content.js", "CHEMWORLD_GUIDES", {"pages": pages}),
        ("chemworld-guide-localized.js", "CHEMWORLD_GUIDE_LOCALES", locales),
    ):
        (output / filename).write_text(f"window.{variable} = {encode(value)};\n", encoding="utf-8")
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
    print(f"Built {len(GUIDES)} bilingual guides and {len(records)} recorded examples at {output}.")
    print("Static records only; no experiments executed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    build(parser.parse_args().output.resolve())
