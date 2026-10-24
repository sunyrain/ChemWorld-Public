"""Fixed 18-unit development block; retain every row, failure and continuation."""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from chemworld.physchem.reactive_water_treatment import advance_reactive_water_treatment
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.reactive_water_treatment import unpack_reactive_treatment
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.reactive_water_treatment_world import reactive_water_treatment_world_spec
from reference.reactive_water_treatment49_fixtures import DATA, fixed_cases, inventory_case
from reference.reactive_water_treatment49_reference import independent


def world_unit(spec, initial, flow, duration, shared, directory):
    payload = reactive_water_treatment_world_spec(
        spec, *initial.amounts_mol, shared_clock=shared
    ).to_dict()
    payload["task"]["operations"].append("select_vessel")
    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    runtime.reset()

    def execute(rt, operation, **values):
        result = rt.execute({"operation": operation, **values})
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
        return result

    execute(runtime, "set_flow_rate", flow_rate_mL_min=flow * 6e7)
    execute(runtime, "select_vessel", vessel="product")
    if not shared:
        execute(runtime, "select_vessel", vessel="reactor")
    operation = "wait" if shared else "run_flow"
    execute(runtime, operation, duration_s=5.0)
    prefix = json.loads(json.dumps(runtime.snapshot().to_dict()))
    source, output = directory / "prefix.json", directory / "continued.json"
    source.write_text(json.dumps(prefix, indent=2), encoding="utf-8")
    outcome = execute(runtime, operation, duration_s=duration - 5.0)
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    replay = ChemWorldRuntime.from_snapshot(prefix)
    execute(replay, operation, duration_s=duration - 5.0)
    replay_equal = json.loads(json.dumps(replay.snapshot().to_dict())) == snapshot
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/reactive_water_treatment49_resume.py",
            str(source),
            str(output),
            operation,
            str(duration - 5.0),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path("tests").resolve())},
        timeout=50,
    )
    (directory / "child.json").write_text(
        json.dumps(
            {"returncode": child.returncode, "stdout": child.stdout, "stderr": child.stderr},
            indent=2,
        ),
        encoding="utf-8",
    )
    process_equal = (
        child.returncode == 0 and json.loads(output.read_text(encoding="utf-8")) == snapshot
    )
    current = runtime._state
    states = {current.vessel_id: current.replace(inactive_vessels={}), **current.inactive_vessels}
    anchor = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    _, _, final = unpack_reactive_treatment(anchor)
    ledger = current.ledger
    pump = ledger.flow_pump_work_J
    hydraulic = pump * spec.pump.efficiency
    motor = pump - hydraulic
    diagnostic = {
        "throughput_m3": spec.volume_m3(initial.amounts_mol[0])
        - spec.volume_m3(final.amounts_mol[0]),
        "pump_work_J": pump,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": motor,
        "bath_heat_into_liquids_J": ledger.energy_jacket_J - (ledger.heat_loss_J - motor),
    }
    stop = (
        None
        if outcome.info["operation_outcome"] == "success"
        else outcome.info["operation_outcome"]
    )
    return final, diagnostic, ledger.time_s, stop, replay_equal, process_equal, snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--unit",
        choices=[case[0] for case in fixed_cases()],
        help="retry one reporting-failed unit; retain the original attempt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="retain completed raw units and run only missing units",
    )
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=args.resume)
    start = time.monotonic()
    cases = [case for case in fixed_cases() if args.unit is None or case[0] == args.unit]
    total = len(cases)
    status = {"complete": 0, "current": "starting"}
    done = threading.Event()

    def heartbeat():
        while not done.wait(25):
            elapsed = time.monotonic() - start
            print(
                f"49B alive current={status['current']} complete={status['complete']}/{total} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    rows = []
    for index, (name, spec, flow, duration, shared) in enumerate(cases, 1):
        status["current"] = name
        unit = directory / name
        unit.mkdir(exist_ok=args.resume)
        if args.resume and (unit / "result.json").exists():
            row = json.loads((unit / "result.json").read_text(encoding="utf-8"))
            if row["unit"] != name or not row["passed"]:
                raise RuntimeError("resume cannot replace a retained failure or different unit")
            rows.append(row)
            status["complete"] = index
            print(f"49B retained {index}/{total} {name}; no physical rerun", flush=True)
            continue
        checks = []

        def numeric(label, actual, expected, atol, rtol=3e-7, checks=checks):
            checks.append(
                {
                    "name": label,
                    "actual": actual,
                    "expected": expected,
                    "passed": bool(
                        math.isfinite(actual)
                        and abs(actual - expected) <= atol + rtol * abs(expected)
                    ),
                    "absolute_error": abs(actual - expected),
                    "atol": atol,
                    "rtol": rtol,
                }
            )

        def exact(label, actual, expected, checks=checks):
            checks.append(
                {
                    "name": label,
                    "actual": actual,
                    "expected": expected,
                    "passed": bool(actual == expected),
                }
            )

        row = {"unit": name, "expected_comparisons": 26 + int(shared is not None)}
        try:
            initial = inventory_case(spec)
            reference = independent(spec, initial, flow, duration)
            if shared is None:
                result, diagnostic = advance_reactive_water_treatment(spec, initial, flow, duration)
                again, again_diagnostic = advance_reactive_water_treatment(
                    spec, initial, flow, duration
                )
                final, dt, stop = (
                    result.final_state,
                    result.actual_duration_s,
                    result.stopping_event,
                )
                replay = result == again and diagnostic == again_diagnostic
                row["solver"] = dict(result.solver_diagnostic)
            else:
                final, diagnostic, dt, stop, replay, process_equal, snapshot = world_unit(
                    spec, initial, flow, duration, shared, unit
                )
                (unit / "snapshot.json").write_text(
                    json.dumps(snapshot, indent=2), encoding="utf-8"
                )
            for i, n in enumerate(final.amounts_mol):
                for j, k in enumerate(DATA):
                    numeric(f"n:{i}:{k}", n[k], reference["amounts"][i][j], 3e-8)
            for i, h in enumerate(final.enthalpies_J):
                numeric(f"H:{i}", h, reference["enthalpies"][i], 3e-6)
            numeric("elapsed_s", dt, reference["dt"], 2e-7, 3e-8)
            numeric("throughput_m3", diagnostic["throughput_m3"], reference["throughput_m3"], 1e-12)
            for key in (
                "pump_work_J",
                "hydraulic_work_J",
                "motor_loss_J",
                "bath_heat_into_liquids_J",
            ):
                numeric(key, diagnostic[key], reference[key], 3e-6)
            numeric(
                "reactor_volume_L",
                spec.volume_m3(final.amounts_mol[1]) * 1000,
                reference["reactor_volume_L"],
                1e-9,
                0,
            )
            element_residuals = [
                sum(
                    (
                        sum(n[s.species_id] for n in final.amounts_mol)
                        - sum(n[s.species_id] for n in initial.amounts_mol)
                    )
                    * s.composition.get(e, 0)
                    for s in spec.network.species
                )
                for e in ("C", "H", "O")
            ]
            numeric("maximum_element_residual_mol", max(map(abs, element_residuals)), 0.0, 3e-8, 0)
            energy = (
                sum(final.enthalpies_J)
                - sum(initial.enthalpies_J)
                - diagnostic["bath_heat_into_liquids_J"]
                - diagnostic["hydraulic_work_J"]
            )
            numeric("energy_residual_J", energy, 0.0, 3e-6, 0)
            exact("stop_identity", stop, reference["stop"])
            exact("exact_replay", replay, True)
            if shared is not None:
                exact("new_process_full_json", process_equal, True)
            row.update(
                initial={"amounts_mol": initial.amounts_mol, "enthalpies_J": initial.enthalpies_J},
                final={"amounts_mol": final.amounts_mol, "enthalpies_J": final.enthalpies_J},
                reference=reference,
                diagnostic=diagnostic,
            )
        except Exception as exc:
            import traceback

            row.update(exception=repr(exc), traceback=traceback.format_exc())
        row.update(
            checks=checks,
            passed=len(checks) == row["expected_comparisons"] and all(c["passed"] for c in checks),
        )
        (unit / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        status["complete"] = index
        elapsed = max(time.monotonic() - start, 1e-12)
        print(
            f"49B {index}/{total} {name} passed={row['passed']} "
            f"comparisons={len(checks)}/{row['expected_comparisons']} "
            f"rate={index / elapsed:.2f}/s ETA={(total - index) * elapsed / index:.1f}s",
            flush=True,
        )
    done.set()
    checks = [c for r in rows for c in r["checks"]]
    summary = {
        "mode": "development",
        "units": total,
        "units_passed": sum(r["passed"] for r in rows),
        "comparisons": sum(r["expected_comparisons"] for r in rows),
        "executed_comparisons": len(checks),
        "comparisons_passed": sum(c["passed"] for c in checks),
        "exceptions": [
            {"unit": r["unit"], "exception": r["exception"]} for r in rows if "exception" in r
        ],
        "failures": [
            {"unit": r["unit"], **c} for r in rows for c in r["checks"] if not c["passed"]
        ],
        "elapsed_s": time.monotonic() - start,
        "passed": all(r["passed"] for r in rows)
        and len(checks) == sum(r["expected_comparisons"] for r in rows),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
