"""Fixed 18-unit, 598-comparison development block with all raw attempts retained."""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

from chemworld.physchem.aqueous_adsorption import advance_aqueous_adsorption
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_adsorption import unpack_aqueous_adsorption
from chemworld.world.aqueous_adsorption_world import aqueous_adsorption_world_spec
from reference.aqueous_adsorption49_fixtures import DATA, ENERGIES, fixed_cases, inventory_case
from reference.aqueous_adsorption49_reference import independent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def inventory_payload(inventory):
    return {
        "amounts_mol": inventory.amounts_mol,
        "enthalpies_J": inventory.enthalpies_J,
        "loading_mol": {k: inventory.sorbent_slot.values[f"loading:{k}"] for k in ("A", "B")},
        "time_s": inventory.time_s,
    }


def reference_inventory(initial, ref):
    values = dict(initial.sorbent_slot.values)
    values.update({f"loading:{k}": n for k, n in zip(("A", "B"), ref["loading"], strict=True)})
    return replace(
        initial,
        amounts_mol=tuple(dict(zip(DATA, row, strict=True)) for row in ref["amounts"]),
        enthalpies_J=tuple(ref["enthalpies"]),
        sorbent_slot=replace(initial.sorbent_slot, values=values),
        time_s=initial.time_s + ref["dt"],
    )


def cycle_reference(spec, initial, flow, duration):
    process = independent(spec, initial, flow, duration, "process")
    rinse = independent(spec, reference_inventory(initial, process), flow, duration, "rinse")
    return {
        **rinse,
        **{
            key: process[key] + rinse[key]
            for key in (
                "dt",
                "throughput_m3",
                "pump_work_J",
                "hydraulic_work_J",
                "motor_loss_J",
                "bath_heat_into_system_J",
                "internal_energy_change_J",
            )
        },
    }


def world_unit(spec, initial, flow, duration, shared, directory):
    runtime = ChemWorldRuntime(
        composition=aqueous_adsorption_world_spec(spec, initial.amounts_mol, shared_clock=shared)
    )
    runtime.reset(seed=49)

    def execute(rt, operation, **values):
        result = rt.execute({"operation": operation, **values})
        if result.info["transaction_status"] != "committed":
            raise RuntimeError(result.info)
        return result

    operation = "wait" if shared else "run_flow"
    execute(runtime, "set_flow_rate", flow_rate_mL_min=flow * 6e7, flow_path="process")
    if shared:
        execute(runtime, "select_vessel", vessel="adsorber")
    execute(runtime, operation, duration_s=duration)
    write_json(directory / "process.json", runtime.snapshot().to_dict())
    execute(runtime, "select_vessel", vessel="reactor")
    execute(runtime, "set_flow_rate", flow_rate_mL_min=flow * 6e7, flow_path="rinse")
    if shared:
        execute(runtime, "select_vessel", vessel="adsorber")
    execute(runtime, operation, duration_s=5.0)
    prefix = json.loads(json.dumps(runtime.snapshot().to_dict()))
    source, output = directory / "prefix.json", directory / "continued.json"
    write_json(source, prefix)
    outcome = execute(runtime, operation, duration_s=duration - 5.0)
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    write_json(directory / "snapshot.json", snapshot)
    replay = ChemWorldRuntime.from_snapshot(prefix)
    execute(replay, operation, duration_s=duration - 5.0)
    replay_equal = json.loads(json.dumps(replay.snapshot().to_dict())) == snapshot
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/aqueous_adsorption49_resume.py",
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
    write_json(
        directory / "child.json",
        {
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
        },
    )
    process_equal = (
        child.returncode == 0 and json.loads(output.read_text(encoding="utf-8")) == snapshot
    )
    current = runtime._state
    states = {current.vessel_id: current.replace(inactive_vessels={}), **current.inactive_vessels}
    anchor = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    final = unpack_aqueous_adsorption(anchor)[2]
    ledger = current.ledger
    pump = ledger.flow_pump_work_J
    hydraulic = pump * spec.pump.efficiency
    motor = pump - hydraulic
    diagnostic = {
        "throughput_m3": sum(
            spec.volume_m3(initial.amounts_mol[i]) - spec.volume_m3(final.amounts_mol[i])
            for i in (0, 3)
        ),
        "pump_work_J": pump,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": motor,
        "bath_heat_into_system_J": ledger.energy_jacket_J - (ledger.heat_loss_J - motor),
    }
    stop = (
        None
        if outcome.info["operation_outcome"] == "success"
        else outcome.info["operation_outcome"]
    )
    return final, diagnostic, ledger.time_s, stop, replay_equal, process_equal


def evaluate(
    spec,
    initial,
    final,
    diagnostic,
    dt,
    stop,
    replay,
    reference,
    *,
    equilibrium=False,
    process_equal=None,
):
    checks = []

    def numeric(label, actual, expected, atol, rtol=3e-7):
        actual, expected = float(actual), float(expected)
        checks.append(
            {
                "name": label,
                "actual": actual,
                "expected": expected,
                "passed": bool(
                    math.isfinite(actual) and abs(actual - expected) <= atol + rtol * abs(expected)
                ),
                "absolute_error": abs(actual - expected),
                "atol": atol,
                "rtol": rtol,
            }
        )

    def exact(label, actual, expected):
        checks.append(
            {
                "name": label,
                "actual": actual,
                "expected": expected,
                "passed": bool(actual == expected),
            }
        )

    for i, n in enumerate(final.amounts_mol):
        for j, k in enumerate(DATA):
            numeric(f"n:{i}:{k}", n[k], reference["amounts"][i][j], 3e-8)
    for j, k in enumerate(("A", "B")):
        numeric(
            f"bound:{k}", final.sorbent_slot.values[f"loading:{k}"], reference["loading"][j], 3e-8
        )
    for i, h in enumerate(final.enthalpies_J):
        numeric(f"H:{i}", h, reference["enthalpies"][i], 3e-6)
    numeric("elapsed_s", dt, reference["dt"], 2e-7, 3e-8)
    numeric("throughput_m3", diagnostic["throughput_m3"], reference["throughput_m3"], 1e-12)
    for key in ("pump_work_J", "hydraulic_work_J", "motor_loss_J", "bath_heat_into_system_J"):
        numeric(key, diagnostic[key], reference[key], 3e-6)
    numeric("contact_volume_L", spec.volume_m3(final.amounts_mol[1]) * 1000, 0.005, 1e-9, 0)
    residuals = []
    for k in DATA:
        residual = sum(n[k] for n in final.amounts_mol) - sum(n[k] for n in initial.amounts_mol)
        if k != "water":
            residual += (
                final.sorbent_slot.values[f"loading:{k}"]
                - initial.sorbent_slot.values[f"loading:{k}"]
            )
        residuals.append(residual)
    numeric("maximum_component_residual_mol", max(map(abs, residuals)), 0, 3e-8, 0)
    # Water volumes sum to a constant; zero solute/adsorbed volumes make total
    # delta(PV)=0. The original carrier stays at constant temperature.
    energy = sum(final.enthalpies_J) - sum(initial.enthalpies_J)
    for k in ("A", "B"):
        h = DATA[k][2] + DATA[k][1] * (spec.temperature_K - 298.15)
        energy += (
            final.sorbent_slot.values[f"loading:{k}"] - initial.sorbent_slot.values[f"loading:{k}"]
        ) * (h + ENERGIES[k])
    energy -= diagnostic["bath_heat_into_system_J"] + diagnostic["hydraulic_work_J"]
    numeric("energy_residual_J", energy, 0, 3e-6, 0)
    exact("stop_identity", stop, reference["stop"])
    exact("exact_replay", replay, True)
    if equilibrium:
        numeric(
            "single_langmuir_equilibrium_A_mol",
            final.sorbent_slot.values["loading:A"],
            reference["equilibrium_A_mol"],
            3e-8,
        )
    if process_equal is not None:
        exact("new_process_full_json", process_equal, True)
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cases, rows = fixed_cases(), []
    status = {"complete": 0, "current": "starting"}
    done = threading.Event()

    def heartbeat():
        while not done.wait(25):
            elapsed = time.monotonic() - start
            completed = status["complete"]
            rate = completed / elapsed
            eta = (len(cases) - completed) / rate if rate else None
            print(
                f"49C alive current={status['current']} complete={completed}/{len(cases)} "
                f"elapsed={elapsed:.1f}s rate={rate:.3f}/s ETA={eta}",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    for index, (name, spec, options, flow, duration, path, shared) in enumerate(cases, 1):
        status["current"] = name
        unit = directory / name
        unit.mkdir()
        unit_start = time.monotonic()
        row = {
            "unit": name,
            "expected_comparisons": 33 + int(name.startswith("batch-single") or shared is not None),
        }
        checks = []
        try:
            initial = inventory_case(spec, **options)
            process_equal = None
            if shared is None:
                reference = independent(spec, initial, flow, duration, path)
                result, diagnostic = advance_aqueous_adsorption(
                    spec, initial, flow, duration, flow_path=path
                )
                final, dt, stop = (
                    result.final_state,
                    result.actual_duration_s,
                    result.stopping_event,
                )
                row["solver"] = dict(result.solver_diagnostic)
                write_json(
                    unit / "physical.json",
                    {
                        "initial": inventory_payload(initial),
                        "final": inventory_payload(final),
                        "diagnostic": diagnostic,
                        "dt": dt,
                        "stop": stop,
                    },
                )
                again, again_diagnostic = advance_aqueous_adsorption(
                    spec, initial, flow, duration, flow_path=path
                )
                replay = bool(result == again and diagnostic == again_diagnostic)
            else:
                reference = cycle_reference(spec, initial, flow, duration)
                final, diagnostic, dt, stop, replay, process_equal = world_unit(
                    spec, initial, flow, duration, shared, unit
                )
            row.update(
                initial=inventory_payload(initial),
                final=inventory_payload(final),
                diagnostic=diagnostic,
                reference=reference,
            )
            checks = evaluate(
                spec,
                initial,
                final,
                diagnostic,
                dt,
                stop,
                replay,
                reference,
                equilibrium=name.startswith("batch-single"),
                process_equal=process_equal,
            )
        except Exception as exc:
            row.update(exception=repr(exc), traceback=traceback.format_exc())
        row.update(
            checks=checks,
            elapsed_s=time.monotonic() - unit_start,
            passed=bool(
                len(checks) == row["expected_comparisons"] and all(c["passed"] for c in checks)
            ),
        )
        write_json(unit / "result.json", row)
        rows.append(row)
        status["complete"] = index
        elapsed = max(time.monotonic() - start, 1e-12)
        print(
            f"49C {index}/{len(cases)} {name} passed={row['passed']} "
            f"comparisons={len(checks)}/{row['expected_comparisons']} "
            f"rate={index / elapsed:.2f}/s ETA={(len(cases) - index) * elapsed / index:.1f}s",
            flush=True,
        )
    done.set()
    checks = [c for r in rows for c in r["checks"]]
    summary = {
        "mode": "development",
        "units": len(cases),
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
        "passed": bool(all(r["passed"] for r in rows) and len(checks) == 598),
    }
    write_json(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
