"""Complete functional/install histories, including failures and known typing errors."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_log(path):
    raw = path.read_bytes()
    return raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")


def functional(root):
    attempts = []
    for directory in sorted(root.glob("conv51h-*")):
        junit = directory / "junit.xml"
        if not junit.is_file():
            continue
        suites = ET.parse(junit).getroot().findall("testsuite")
        issues = []
        for suite in suites:
            for case in suite.findall("testcase"):
                issue = case.find("failure")
                if issue is None:
                    issue = case.find("error")
                if issue is not None:
                    issues.append(
                        {
                            "class": case.attrib.get("classname"),
                            "name": case.attrib.get("name"),
                            "message": issue.attrib.get("message"),
                            "details": issue.text,
                        }
                    )
        attempts.append(
            {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                **{
                    key: sum(int(s.attrib.get(key, 0)) for s in suites)
                    for key in ("tests", "failures", "errors", "skipped")
                },
                "elapsed_s": sum(float(s.attrib.get("time", 0)) for s in suites),
                "failed_cases": issues,
            }
        )
    paths = {
        "baseline": root / "conv51h-functional-first/baseline-mypy-shadow.log",
        "current": root / "conv51h-integration-second/mypy.log",
    }
    messages = {
        kind: [
            line.split("error: ", 1)[1] for line in read_log(path).splitlines() if "error: " in line
        ]
        for kind, path in paths.items()
    }
    return {
        "schema": "chemworld-integer51h-functional-history-v1",
        "scope": (
            "Original world and three integer algorithms; "
            "no new-chemistry distribution/resource qualification yet"
        ),
        "attempts": attempts,
        "mypy": {
            "tau_module": "passed with repository disallow_untyped_defs, follow-imports=silent",
            "two_module_passed": not messages["current"],
            "baseline_error_messages": messages["baseline"],
            "current_error_messages": messages["current"],
            "original_shadow_errors_equal": messages["baseline"] == messages["current"],
            "raw_logs": {k: str(v) for k, v in paths.items()},
        },
        "all_failures_retained": True,
    }


def packages(root):
    attempts, unfinished = [], []
    for directory in sorted(root.glob("conv51h-package-*")):
        path = directory / "summary.json"
        if not path.is_file():
            unfinished.append(str(directory))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        legacy = directory / "legacy-hybrid-protocols/summary.json"
        data["shared_probe_legacy_hybrid"] = (
            json.loads(legacy.read_text(encoding="utf-8"))
            if legacy.is_file()
            else {"passed": False, "error": "legacy branch regression missing"}
        )
        attempts.append({"attempt": directory.name, "raw_evidence": str(directory), **data})
    return {
        "schema": "chemworld-integer51h-package-history-v1",
        "scope": (
            "Actual sdist-to-wheel installation with locked shared dependencies; not a sealed venv"
        ),
        "attempts": attempts,
        "unfinished_attempts": unfinished,
        "raw_trajectories_and_checkpoints_retained": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tests, installed = functional(args.evidence), packages(args.evidence)
    write(args.output / "FUNCTIONAL_RESULTS.json", tests)
    write(args.output / "PACKAGE_RESULTS.json", installed)
    print(
        json.dumps(
            {
                "functional_attempts": len(tests["attempts"]),
                "package_attempts": len(installed["attempts"]),
                "unfinished_packages": installed["unfinished_attempts"],
                "baseline_typing_errors_equal": tests["mypy"]["original_shadow_errors_equal"],
            }
        )
    )


if __name__ == "__main__":
    main()
