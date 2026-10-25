"""Record actual original fixture XML, then require byte equality and no namespace leak."""

import argparse
import json
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

from run_sbml_input_block import cases as input_cases
from run_sbml_trajectory_block import cases as trajectory_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in (
        "run_sbml51k_namespace.py",
        "run_sbml_input_block.py",
        "run_sbml_trajectory_block.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    before = dict(ET._namespace_map)
    sources = {row[0]: row[3] for row in input_cases()}
    sources.update({row[0]: row[3] for row in trajectory_cases()})
    after = dict(ET._namespace_map)
    (args.output / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    result = {
        "mode": "compare" if args.baseline else "capture",
        "planned": 30,
        "recorded": len(sources),
        "namespace_before": before,
        "namespace_after": after,
        "namespace_leaked": before != after,
    }
    if args.baseline:
        original = json.loads((args.baseline / "sources.json").read_text(encoding="utf-8"))
        result["different_sources"] = [key for key in sources if sources[key] != original.get(key)]
        result["passed"] = len(sources) == 30 and sources == original and before == after
    else:
        result["passed"] = len(sources) == 30
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
