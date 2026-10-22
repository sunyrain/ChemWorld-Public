"""Execute public documentation examples and render API signatures for the site."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from capability_reference_recipes import BINDINGS, RECIPES

ROOT = Path(__file__).resolve().parents[1]


def catalogue():
    text = (ROOT / "web/chemworld-capabilities.js").read_text(encoding="utf-8")
    tree = json.loads(text.split(" = ", 1)[1].strip().removesuffix(";"))["tree"]
    nodes = {}

    def walk(node, parent=None):
        nodes[node["id"]] = {**node, "parent_id": parent}
        for child in node.get("children", []):
            walk(child, node["id"])

    for branch in tree:
        walk(branch)
    return nodes


def api_reference(target):
    module, path = target.split(":")
    obj = importlib.import_module(module)
    for part in path.split("."):
        obj = getattr(obj, part)
    signature = inspect.signature(obj)
    return {
        "path": module + "." + path,
        "signature": path + str(signature),
        "doc": inspect.getdoc(obj) or "",
        "parameters": [
            {
                "name": name,
                "kind": p.kind.name,
                "type": ""
                if p.annotation is inspect.Parameter.empty
                else inspect.formatannotation(p.annotation),
                "required": p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD),
                "default": "" if p.default is inspect.Parameter.empty else repr(p.default),
            }
            for name, p in signature.parameters.items()
            if name not in {"self", "cls"}
        ],
        "returns": ""
        if signature.return_annotation is inspect.Signature.empty
        else inspect.formatannotation(signature.return_annotation),
    }


def build(*, only=(), report):
    nodes = catalogue()
    missing = [
        n["id"]
        for n in nodes.values()
        if n.get("kind") == "capability" and n.get("state") != "planned" and n["id"] not in BINDINGS
    ]
    selected = list(only or RECIPES)
    payload = {"recipes": {}, "nodes": {}, "coverage": {}}
    results = {
        "examples": len(selected),
        "passed": [],
        "failures": [],
        "missing_capabilities": missing,
    }
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chemworld-api-docs-") as temp:
        for index, key in enumerate(selected, 1):
            recipe = RECIPES[key]
            print(f"[{index}/{len(selected)}] {key}: running documented code", flush=True)
            try:
                script = Path(temp) / (key + ".py")
                script.write_text(recipe["code"], encoding="utf-8")
                process = subprocess.Popen(
                    [sys.executable, "-X", "utf8", str(script)],
                    cwd=ROOT,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
                ticks = 0
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        ticks += 1
                        print(
                            f"  {key}: alive {ticks * 30}s; {index - 1}/{len(selected)} complete",
                            flush=True,
                        )
                        if ticks >= 10:
                            process.kill()
                            process.communicate()
                            raise RuntimeError(
                                "documentation example exceeded 300 seconds"
                            ) from None
                if process.returncode:
                    raise RuntimeError(stderr.strip() or stdout)
                output = json.loads(stdout)
                apis = [api_reference(target) for target in recipe["targets"]]
                payload["recipes"][key] = {
                    **{k: v for k, v in recipe.items() if k != "targets"},
                    "output": output,
                    "apis": apis,
                }
                results["passed"].append(key)
                elapsed = time.monotonic() - started
                rate = index / elapsed
                print(
                    f"  PASS; {index}/{len(selected)} checked, {rate:.2f}/s, "
                    f"ETA {(len(selected) - index) / rate:.0f}s",
                    flush=True,
                )
            except Exception as exc:
                results["failures"].append({"example": key, "error": str(exc)})
                print(f"  FAIL: {exc}", flush=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    if results["failures"] or (missing and not only):
        print(f"Reference not published. Report: {report}; missing: {missing}")
        return 1
    if only:
        print(f"Selected examples passed; no published artifact changed. Report: {report}")
        return 0
    for key, node in nodes.items():
        if node.get("kind") not in {"capability", "method"}:
            continue
        parent = node.get("derivedFrom") or node["parent_id"]
        payload["nodes"][key] = {
            "examples": BINDINGS.get(key, BINDINGS.get(parent, [])),
            "shared_example": key not in BINDINGS and node.get("kind") == "method",
            "planned": node.get("state") == "planned",
        }
    payload["coverage"] = {
        "examples": len(RECIPES),
        "implemented_capabilities": sum(
            n.get("kind") == "capability" and n.get("state") != "planned" for n in nodes.values()
        ),
        "planned_capabilities": sum(
            n.get("kind") == "capability" and n.get("state") == "planned" for n in nodes.values()
        ),
        "execution": "Windows / Python " + sys.version.split()[0],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output = ROOT / "web/chemworld-api-reference.js"
    output.write_text(
        "// Generated by scripts/build_capability_reference.py\n"
        + "window.CHEMWORLD_API_REFERENCE = "
        + encoded
        + ";\n",
        encoding="utf-8",
    )
    print(f"Published {len(RECIPES)} executed examples to {output}; report: {report}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(build(only=args.only, report=args.report))
