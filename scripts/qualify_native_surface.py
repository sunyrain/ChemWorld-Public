"""Run fixed surface development blocks; retain every failure, never overwrite output."""

import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

from chemworld.eval.surface_acceptance import acceptance_cases, evaluate_case, summarize_cases
from chemworld.runtime.semantics import RUNTIME_SEMANTICS_ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=("all", "rate", "isothermal", "thermal"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = [c for c in acceptance_cases() if args.block in ("all", c["block"])]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the result path before executing; do not replace a previous result.
    with args.output.open("x", encoding="utf-8") as stream:
        started, rows = perf_counter(), []
        versions = {}
        for name in ("numpy", "scipy", "cantera"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        stream.write(
            json.dumps(
                {
                    "mode": "development",
                    "runtime_semantics_id": RUNTIME_SEMANTICS_ID,
                    "versions": versions,
                    "expected_units": len(cases),
                }
            )[:-1]
            + ', "cases": [\n'
        )
        for index, case in enumerate(cases, 1):
            try:
                row = evaluate_case(case)
                row["status"] = "passed" if row["passed"] else "failed"
            except Exception as exc:
                row = {
                    "case": case,
                    "passed": False,
                    "exception": type(exc).__name__,
                    "reason": str(exc),
                    "status": "unavailable" if isinstance(exc, ImportError) else "failed",
                }
            rows.append(row)
            if index > 1:
                stream.write(",\n")
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False))
            stream.flush()
            elapsed = perf_counter() - started
            print(
                f"surface {case['block']} {index}/{len(cases)} {index / elapsed:.2f} units/s "
                f"ETA {(len(cases) - index) * elapsed / index:.1f}s "
                f"{case['case_id']} {'PASS' if row['passed'] else 'FAIL'}",
                flush=True,
            )
        summary = summarize_cases(cases, rows)
        summary["elapsed_s"] = perf_counter() - started
        stream.write('\n], "summary": ' + json.dumps(summary) + "}\n")
        print(json.dumps(summary), flush=True)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
