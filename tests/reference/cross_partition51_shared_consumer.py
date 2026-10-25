"""Separate actual shared-world complete snapshot continuation consumer."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from cross_partition51_cases import shared_world

from chemworld.runtime import ChemWorldRuntime

DURATIONS = (1., 2., 1.)


def continue_snapshot(payload):
    runtime = ChemWorldRuntime.from_snapshot(payload)
    committed = [runtime.execute({"operation": "wait", "duration_s": duration}).info[
        "transaction_status"] for duration in DURATIONS]
    return {"committed": committed, "state": runtime._state.to_dict(),
            "snapshot": runtime.snapshot().to_dict()}


def run(directory):
    checks = []

    def check(name, actual, expected=True):
        checks.append({"check": name, "passed": actual == expected,
                       "actual": actual, "expected": expected})

    runtime = ChemWorldRuntime(composition=shared_world(seed=17))
    runtime.reset(seed=3)
    check("select_other", runtime.execute({"operation": "select_vessel", "vessel": "second"}
                                         ).info["transaction_status"], "committed")
    check("shared_initial_wait", runtime.execute({"operation": "wait", "duration_s": 1.}
                                               ).info["transaction_status"], "committed")
    check("both_initial_clocks", sorted(runtime._state.vessel_elapsed_s.values()), [1., 1.])
    check("both_original_receipts_nonzero", all(sum(
        local.process.stochastic_continuation["partition_transfer"]["firings"].values()) > 0
        for local in (runtime._state, *runtime._state.inactive_vessels.values())))
    before = runtime._state.to_dict()
    runtime.observe()
    check("observation_is_pure", runtime._state.to_dict() == before)
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    source, child = directory / "shared-snapshot.json", directory / "child.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--resume", str(source),
                    "--output", str(child)], check=True)
    fresh = json.loads(child.read_text(encoding="utf-8"))
    in_process = json.loads(json.dumps(continue_snapshot(payload)))
    committed = [runtime.execute({"operation": "wait", "duration_s": duration}).info[
        "transaction_status"] for duration in DURATIONS]
    check("original_subsequent_actions", committed, ["committed"] * 3)
    check("restored_subsequent_actions", in_process["committed"], ["committed"] * 3)
    check("fresh_process_subsequent_actions", fresh["committed"], ["committed"] * 3)
    state = json.loads(json.dumps(runtime._state.to_dict()))
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    check("exact_complete_world_in_process", state == in_process["state"])
    check("exact_complete_runtime_in_process", snapshot == in_process["snapshot"])
    check("exact_complete_world_fresh_process", state == fresh["state"])
    check("exact_complete_runtime_fresh_process", snapshot == fresh["snapshot"])
    check("both_final_clocks", sorted(runtime._state.vessel_elapsed_s.values()), [5., 5.])
    return {"schema": "cross-partition51-shared-consumer-v1", "units_total": 1,
            "units_passed": int(all(c["passed"] for c in checks)),
            "checks_total": len(checks), "checks_passed": sum(c["passed"] for c in checks),
            "failures": [c for c in checks if not c["passed"]], "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.resume:
        result = continue_snapshot(json.loads(args.resume.read_text(encoding="utf-8")))
    else:
        args.output.parent.mkdir(parents=True, exist_ok=False)
        result = run(args.output.parent)
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}), flush=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return int(result.get("units_passed", 1) != 1)


if __name__ == "__main__":
    raise SystemExit(main())
