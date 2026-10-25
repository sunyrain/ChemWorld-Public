"""Predeclared actual-runtime feedback construction check; preserve every attempt."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Event, Thread

from chemworld.examples.thermal_feedback import thermal_feedback_world
from chemworld.runtime import ChemWorldRuntime

CHECKS = (
    "committed",
    "actual_temperature",
    "finite_heat",
    "one_device_memory",
    "no_fake_work",
    "one_clock",
    "pause_or_queue",
    "snapshot_exact",
    "fresh_process_exact",
)


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    if result.info["transaction_status"] != "committed":
        raise ValueError(json.dumps(result.info))
    return result


def reactor(state):
    return state if state.vessel_id == "reactor" else state.inactive_vessels["reactor"]


def memory(state):
    return next(
        s
        for s in reactor(state).state_slots.slots
        if s.spec.state_type == "thermal-feedback-memory-v1"
    )


def unit(identity, folder):
    row = {"id": identity, "checks": dict.fromkeys(CHECKS, False), "failures": []}
    runtime = replay = None
    try:
        shared = identity == "shared-PID"
        mode = "PI" if identity == "local-PI" else "PID"
        runtime = ChemWorldRuntime(composition=thermal_feedback_world(mode, shared=shared))
        runtime.reset(seed=33)
        if shared:
            execute(runtime, "select_vessel", vessel="reactor")
            execute(runtime, "queue_lab_heat", target_temperature_K=350.0, duration_s=2.0)
            execute(runtime, "select_vessel", vessel="witness")
            result = execute(runtime, "wait", duration_s=1.0)
        else:
            result = execute(runtime, "heat", target_temperature_K=350.0, duration_s=1.0)
        row["first_info"] = result.info
        state = runtime._state
        row["checks"].update(
            committed=True,
            actual_temperature=reactor(state).temperature_K > 300.0,
            finite_heat=0 < state.ledger.energy_jacket_J <= 30.0 + 1e-7,
            one_device_memory=(
                memory(state).spec.owner_kind == "equipment"
                and memory(state).spec.owner_id == "feedback-heater"
                and len(
                    [
                        s
                        for s in reactor(state).state_slots.slots
                        if s.spec.state_type == "thermal-feedback-memory-v1"
                    ]
                )
                == 1
            ),
            no_fake_work=state.ledger.flow_shaft_work_J == state.ledger.boundary_work_J == 0,
            one_clock=state.ledger.time_s == 1.0,
        )
        if identity == "local-PID-pause":
            before = dict(memory(state).values)
            heat = state.ledger.energy_jacket_J
            execute(runtime, "pause_control")
            execute(runtime, "wait", duration_s=0.5)
            after = memory(runtime._state).values
            row["checks"]["pause_or_queue"] = (
                after["integral_K"] == before["integral_K"]
                and after["filtered_temperature_K"] != before["filtered_temperature_K"]
                and runtime._state.ledger.energy_jacket_J == heat
            )
            execute(runtime, "resume_control")
        else:
            row["checks"]["pause_or_queue"] = (
                len(state.lab_clock.running) == 1
                if shared
                else memory(state).values["integral_K"] != 0
            )
        payload = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        source, target = folder / f"{identity}-snapshot.json", folder / f"{identity}-fresh.json"
        source.write_text(json.dumps(payload), encoding="utf8")
        replay = ChemWorldRuntime.from_snapshot(payload)
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"p=json.loads(Path({str(source)!r}).read_text(encoding='utf8')); "
            "r=ChemWorldRuntime.from_snapshot(p); "
            "z=r.execute({'operation':'wait','duration_s':1.}); "
            "assert z.info['transaction_status']=='committed',z.info; "
            f"p=Path({str(target)!r}); "
            "p.write_text(json.dumps(r._state.to_dict()),encoding='utf8'); r.close()"
        )
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        row["child"] = {
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
        }
        if child.returncode:
            raise RuntimeError("new-process continuation failed")
        for session in (runtime, replay):
            execute(session, "wait", duration_s=1.0)
        row["checks"]["snapshot_exact"] = runtime._state.to_dict() == replay._state.to_dict()
        row["checks"]["fresh_process_exact"] = json.loads(
            target.read_text(encoding="utf8")
        ) == json.loads(json.dumps(runtime._state.to_dict()))
        row["final_state"] = runtime._state.to_dict()
        if shared:
            row["checks"]["pause_or_queue"] &= (
                len(runtime._state.lab_clock.completed) == 1
                and not reactor(runtime._state)
                .equipment.equipment["feedback-heater"]
                .settings["thermal_device"]["enabled"]
            )
    except Exception as exc:
        row["failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
    finally:
        if runtime:
            runtime.close()
        if replay:
            replay.close()
    row["passed"] = all(row["checks"].values()) and not row["failures"]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start, done, rows = time.monotonic(), Event(), []

    def progress():
        print(f"33D canary units={len(rows)}/3 elapsed={time.monotonic() - start:.1f}s", flush=True)

    def heartbeat():
        while not done.wait(30):
            progress()

    Thread(target=heartbeat, daemon=True).start()
    for identity in ("local-PI", "local-PID-pause", "shared-PID"):
        rows.append(unit(identity, args.output))
        progress()
        print(identity, rows[-1]["checks"], "failure_count", len(rows[-1]["failures"]), flush=True)
        (args.output / "results.json").write_text(
            json.dumps(
                {
                    "rows": rows,
                    "elapsed_s": time.monotonic() - start,
                    "units_planned": 3,
                    "checks_planned": 27,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf8",
        )
    done.set()
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
