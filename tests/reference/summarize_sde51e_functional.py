"""Summarize every retained functional attempt; never choose only passing runs."""

import argparse
import contextlib
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def junit(path, root):
    suites = list(ET.parse(path).getroot().iter("testsuite"))
    counts = {
        key: sum(int(s.get(key, "0")) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    cases = [case for suite in suites for case in suite.findall("testcase")]
    if not suites or counts["tests"] != len(cases):
        raise ValueError("expected a complete pytest JUnit report: " + str(path))
    failed, skipped = [], []
    for case in cases:
        identity = case.get("classname", "") + "." + case.get("name", "")
        for tag, target in (("failure", failed), ("error", failed), ("skipped", skipped)):
            node = case.find(tag)
            if node is not None:
                target.append({"test": identity, "message_excerpt": node.get("message", "")[:280]})
    if len(failed) != counts["failures"] + counts["errors"] or len(skipped) != counts["skipped"]:
        raise ValueError("JUnit denominator and failure records disagree")
    return {
        "artifact": path.relative_to(root).as_posix(),
        **counts,
        "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"],
        "seconds": sum(float(s.get("time", "0")) for s in suites),
        "failed_cases": failed,
        "skipped_cases": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--current-integrated", required=True)
    parser.add_argument("--doc-file", type=Path, default=Path("docs/native_biochemical.md"))
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    paths = sorted(
        [*root.glob("conv51e-*.xml"), *root.glob("conv51e-intervention-integration-*/junit.xml")]
    )
    records = [junit(path, root) for path in paths]
    if args.current_integrated not in {r["artifact"] for r in records}:
        raise ValueError("the explicitly selected current complete attempt is missing")
    documentation = args.doc_file.read_text(encoding="utf-8").split(
        "## Continuous reaction SDE (development)", 1
    )[1]
    examples = re.findall(r"```python\n(.*?)```", documentation, re.S)
    if not examples:
        raise ValueError("the documented SDE example is missing")
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for example in examples:
            exec(compile(example, str(args.doc_file), "exec"), scope)
    if (
        scope["result"].info["transaction_status"] != "committed"
        or len(scope["result"].observation.channels) != 2
        or any(c.unit != "mol" for c in scope["result"].observation.channels)
    ):
        raise ValueError("the current documented SDE example did not succeed")
    summary = {
        "schema": "chemworld-conv51e-functional-development-v1",
        "task_status": "CLAIMED",
        "scientific_sdk_qualification_completed": False,
        "denominator_policy": (
            "Attempts overlap and are never added. All failed attempts are retained."
        ),
        "raw_junit_location": (
            "outside Git: reference-evidence/conv51e-*.xml and named stage directories"
        ),
        "current_integrated_attempt": args.current_integrated,
        "current_intervention_attempt": "conv51e-interventions-sixth.xml",
        "documented_example": {
            "passed": True,
            "channels": 2,
            "unit": "mol",
            "code_blocks": len(examples),
            "provenance": "current native SDE examples executed by this summary command",
        },
        "attempts": records,
        "remaining": [
            "Independent SDE distributions, moments, boundary and step convergence",
            "Real installed-package qualification",
            "Remaining parent-51 methods and dynamic partition",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    current = next(r for r in records if r["artifact"] == args.current_integrated)
    print(json.dumps({"retained_attempts": len(records), "current": current}, indent=2))


if __name__ == "__main__":
    main()
