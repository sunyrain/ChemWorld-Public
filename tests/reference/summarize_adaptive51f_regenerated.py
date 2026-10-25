"""All combined regenerated-reactant functional attempts and every failed node."""

import argparse
import json
from pathlib import Path
from xml.etree.ElementTree import parse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts, unfinished = [], []
    for directory in sorted(args.evidence.glob("conv51f-regenerated-functional-*")):
        if not directory.is_dir():
            continue
        if not (directory / "process.json").is_file() or not (directory / "junit.xml").is_file():
            unfinished.append(str(directory))
            continue
        suites = list(parse(directory / "junit.xml").getroot().iter("testsuite"))
        counts = {
            field: sum(int(s.get(field, "0")) for s in suites)
            for field in ("tests", "failures", "errors", "skipped")
        }
        process = json.loads((directory / "process.json").read_text(encoding="utf-8"))
        failures = []
        for suite in suites:
            for case in suite.findall("testcase"):
                for tag in ("failure", "error", "skipped"):
                    detail = case.find(tag)
                    if detail is not None:
                        failures.append(
                            {
                                "name": case.get("name"),
                                "class": case.get("classname"),
                                "kind": tag,
                                "message": detail.get("message"),
                            }
                        )
        attempts.append(
            {
                "attempt": directory.name,
                "raw_evidence": str(directory),
                **counts,
                "process": process,
                "failed_cases": failures,
                "passed": counts["tests"] == 88
                and not any(counts[k] for k in ("failures", "errors", "skipped"))
                and process["exit_code"] == 0,
            }
        )
    args.output.write_text(
        json.dumps(
            {
                "schema": "chemworld-adaptive51f-regenerated-functional-history-v1",
                "attempts": attempts,
                "unfinished_attempts": unfinished,
                "scope": "Original72 plus new16 nodes; counts overlap earlier qualifications",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"attempts": len(attempts), "unfinished": unfinished}))


if __name__ == "__main__":
    main()
