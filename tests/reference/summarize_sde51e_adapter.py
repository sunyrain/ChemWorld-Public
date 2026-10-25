"""Publish all retained fixed adapter attempts, distinguishing setup from simulation."""

import argparse
import json
from pathlib import Path


def summarize(directory):
    raw = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    sdk_path = directory / "copasi/trajectories.jsonl"
    sdk = [json.loads(line) for line in sdk_path.read_text(encoding="utf-8").splitlines()]
    native = [
        json.loads(line)
        for line in (directory / "native.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    # The first adapter predates the explicit stage fields. Its recorded compile
    # TypeError is before task.process; it is never counted as an executed course.
    def attempted(row):
        if "simulation_attempted" in row:
            return row["simulation_attempted"]
        if row["status"] == "ok":
            return True
        if row.get("error") == (
            "TypeError('CModel.compileIfNecessary() takes 1 positional argument but 2 were given')"
        ):
            return False
        raise ValueError("unknown historical simulation stage: " + row["id"])

    return {
        "attempt": directory.name,
        "passed": raw["passed"],
        "elapsed_s": raw["elapsed_s"],
        "native_planned": raw["native_planned"],
        "native_recorded": len(native),
        "native_completed": sum(r["status"] == "ok" for r in native),
        "sdk_planned": raw["sdk_planned"],
        "sdk_exit_code": raw["sdk_exit_code"],
        "sdk_jobs_recorded": len(sdk),
        "sdk_simulations_attempted": sum(attempted(r) for r in sdk),
        "sdk_simulations_completed": sum(r["status"] == "ok" for r in sdk),
        "units": [
            {
                "id": unit["id"],
                "passed": unit["passed"],
                "checks": len(unit["checks"]),
                "failed_checks": [c["name"] for c in unit["checks"] if not c["passed"]],
            }
            for unit in raw["units"]
        ],
        "failures": [
            {"engine": engine, "id": r["id"], "error": r["error"]}
            for engine, rows in (("native", native), ("copasi", sdk))
            for r in rows
            if r["status"] != "ok"
        ],
        "sdk_unrecorded": raw["sdk_unexecuted"],
        "versions": sorted({r["version"] for r in sdk if r["status"] == "ok"}),
        "raw_evidence": str(directory),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = sorted(p for p in args.evidence.glob("conv51e-adapter-*") if p.is_dir())
    if not directories:
        raise ValueError("no adapter attempts")
    result = {
        "schema": "chemworld-sde51e-adapter-history-v1",
        "statistical_qualification": False,
        "attempts": [summarize(p) for p in directories],
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"attempts": len(directories), "output": str(args.output)}))


if __name__ == "__main__":
    main()
