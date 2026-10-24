"""Fixed 18-unit/172-comparison osmosis qualification; retains every result."""

import argparse
import json
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from chemworld.physchem.osmotic_membrane import (
    advance_osmotic_membrane,
    osmotic_membrane_inventory,
    osmotic_membrane_point,
)
from chemworld.runtime import ChemWorldRuntime
from reference.osmotic_membrane48_fixtures import inventories, spec
from reference.osmotic_membrane48_reference import (
    endpoint,
    event_time,
    point,
    trajectory,
)
from test_osmotic_membrane48 import physical, setup


class Comparisons:
    def __init__(self):
        self.rows = []

    def near(self, name, actual, expected, atol, rtol=0):
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        passed = (
            a.shape == b.shape and bool(np.all(np.isfinite(a))) and bool(np.all(np.isfinite(b)))
        )
        passed = passed and bool(np.all(np.abs(a - b) <= atol + rtol * np.abs(b)))
        self.rows.append(
            {
                "name": name,
                "passed": bool(passed),
                "actual": a.tolist(),
                "expected": b.tolist(),
                "atol": atol,
                "rtol": rtol,
            }
        )

    def exact(self, name, actual, expected):
        self.rows.append(
            {"name": name, "passed": actual == expected, "actual": actual, "expected": expected}
        )


def point_unit(c, family, mode, left=10, right=60):
    model, amounts = spec(family, mode), inventories(left, right)
    native, reference = osmotic_membrane_point(model, amounts), point(model, amounts)
    c.near(
        "chemical_potential_difference",
        native["water_chemical_potential_difference_J_mol"],
        reference["delta_mu"],
        2e-9,
        2e-8,
    )
    c.near("signed_water_rate", native["water_transfer_mol_s"], reference["flow"], 2e-12, 2e-8)
    for i, name in enumerate(("left_H_rate", "right_H_rate")):
        c.near(name, native["enthalpy_rates_W"][i], reference["enthalpy_rates_W"][i], 2e-9, 2e-8)
    c.near("bath_rate", native["bath_heat_into_liquids_W"], reference["bath_W"], 2e-9, 2e-8)
    c.near(
        "nonnegative_entropy_rate",
        [native["entropy_production_W_K"], float(native["entropy_production_W_K"] >= 0)],
        [reference["entropy_W_K"], 1],
        2e-12,
        2e-8,
    )
    for i, name in enumerate(("left_volume_m3", "right_volume_m3")):
        c.near(name, native["volumes_L"][i] / 1000, reference["volumes_L"][i] / 1000, 2e-13, 2e-8)
    for i, name in enumerate(("left_concentration", "right_concentration")):
        c.near(name, native["concentrations_mol_m3"][i], reference["concentrations"][i], 2e-9, 2e-8)
    return {"model": model.to_dict(), "amounts": amounts, "native": native, "reference": reference}


def trajectory_unit(c, family, mode):
    model, amounts = spec(family, mode), inventories()
    result, receipt = advance_osmotic_membrane(
        model,
        osmotic_membrane_inventory(model, *amounts),
        duration_s=20,
        evaluation_times_s=(0, 5, 10, 20),
    )
    refs = trajectory(model, amounts)
    coarse = trajectory(model, amounts, steps=128)
    waters = [[n["H2O"] for n in s.amounts_mol] for s in result.trajectory]
    expected = [[n["H2O"] for n in row["amounts"]] for row in refs]
    c.near("all_water_history", waters, expected, 2e-8)
    c.near(
        "all_enthalpy_history",
        [s.enthalpies_J for s in result.trajectory],
        [row["enthalpies_J"] for row in refs],
        3e-6,
    )
    for i, name in enumerate(("left_final_volume", "right_final_volume")):
        c.near(name, receipt["final"]["volumes_L"][i], refs[-1]["volumes_L"][i], 3e-10)
    c.exact(
        "all_original_salt_exact",
        [[n[k] for n in s.amounts_mol for k in ("Na+", "Cl-")] for s in result.trajectory],
        [[n[k] for n in amounts for k in ("Na+", "Cl-")]] * len(result.trajectory),
    )
    c.near(
        "total_water_balance",
        [sum(n["H2O"] for n in s.amounts_mol) for s in result.trajectory],
        [0.1] * len(result.trajectory),
        2e-10,
    )
    c.near(
        "heat_and_enthalpy",
        [receipt["bath_heat_into_liquids_J"], receipt["enthalpy_change_J"]],
        [refs[-1]["heat_J"]] * 2,
        5e-6,
    )
    c.near("pressure_boundary_work", receipt["boundary_work_J"], refs[-1]["work_J"], 5e-6)
    c.near(
        "internal_energy_and_first_law",
        [receipt["internal_energy_change_J"], receipt["internal_energy_closure_J"]],
        [refs[-1]["internal_energy_change_J"], 0],
        5e-6,
    )
    c.near(
        "entropy_from_same_gibbs", receipt["entropy_production_J_K"], refs[-1]["entropy_J_K"], 2e-8
    )
    c.near("actual_duration", result.actual_duration_s, 20, 2e-7)
    c.near(
        "independent_128_256_step_convergence",
        [[n["H2O"] for n in row["amounts"]] for row in coarse],
        expected,
        2e-10,
    )
    return {
        "native": [
            {"time_s": s.time_s, "amounts": s.amounts_mol, "enthalpies_J": s.enthalpies_J}
            for s in result.trajectory
        ],
        "receipt": receipt,
        "solver": result.solver_diagnostic,
        "reference": refs,
        "reference_128": coarse,
    }


def resources(runtime):
    ledger = runtime._state.ledger.to_dict()
    return {k: v for k, v in ledger.items() if k not in {"cost", "risk"}}


def world_unit(c, number, out):
    mode = {11: "FO", 12: "PRO", 13: "FO", 14: "reverse"}[number]
    model, amounts = spec(mode=mode), inventories()
    if number == 12:
        model = replace(model, minimum_water_mol=0.045)
    if number == 13:
        model = replace(
            model, capacities_L=(0.003, endpoint(model, amounts, 0.005)["volumes_L"][1])
        )
    runtime = setup(model, shared=number != 11)
    prefix = None
    if number == 13:
        prefix = runtime.execute({"operation": "wait", "duration_s": 5})
        prefix_time = runtime._state.ledger.time_s
    before = json.loads(json.dumps(runtime.snapshot().to_dict()))
    duration = {11: 20, 12: 120, 13: 60, 14: 20}[number]
    result = runtime.execute({"operation": "wait", "duration_s": duration})
    commit = result.info["transaction_status"] == "committed"
    if prefix is not None:
        commit = commit and prefix.info["transaction_status"] == "committed" and prefix_time == 5
    c.exact("actual_transaction_and_prefix", commit, True)
    expected_time = event_time(model, amounts) if number in {12, 13} else 20
    reference = (
        endpoint(model, amounts, 0.005) if number in {12, 13} else trajectory(model, amounts)[-1]
    )
    c.near(
        "actual_stop_time",
        runtime._state.ledger.time_s,
        expected_time,
        2e-4 if number in {12, 13} else 2e-7,
    )
    tanks = (runtime._state, runtime._state.inactive_vessels["draw"])
    for i, name in enumerate(("left_water", "right_water")):
        c.near(name, tanks[i].species_amounts["H2O"], reference["amounts"][i]["H2O"], 2e-8)
    for i, name in enumerate(("left_enthalpy", "right_enthalpy")):
        c.near(
            name,
            sum(slot.values["enthalpy_J"] for slot in tanks[i].state_slots.slots),
            reference["enthalpies_J"][i],
            3e-6,
        )
    c.exact(
        "salt_in_each_original_owner",
        [[tank.species_amounts[k] for k in ("Na+", "Cl-")] for tank in tanks],
        [[n[k] for k in ("Na+", "Cl-")] for n in amounts],
    )
    ledger = runtime._state.ledger
    c.near(
        "original_bath_ledger",
        ledger.energy_jacket_J - ledger.heat_loss_J,
        reference["heat_J"],
        5e-6,
    )
    c.near("original_boundary_work_ledger", ledger.boundary_work_J, reference["work_J"], 5e-6)
    c.exact(
        "no_pump_shaft_or_electrical_work",
        [ledger.flow_pump_work_J, ledger.flow_shaft_work_J, ledger.flow_electrical_work_J],
        [0, 0, 0],
    )
    c.near(
        "all_original_clocks",
        [tank.ledger.time_s for tank in tanks] + list(runtime._state.vessel_elapsed_s.values()),
        [ledger.time_s] * (2 + len(runtime._state.vessel_elapsed_s)),
        2e-7,
    )
    expected_snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    if number == 13:
        source, target = out / "unit13-prefix.json", out / "unit13-fresh-process.json"
        source.write_text(json.dumps(before), encoding="utf-8")
        process = subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                str(Path(__file__).with_name("osmotic_membrane48_resume.py")),
                str(source),
                str(target),
                str(duration),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        (out / "unit13-process.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
        if process.returncode:
            raise RuntimeError(f"fresh process failed: {process.returncode}: {process.stderr}")
        actual_snapshot = json.loads(target.read_text(encoding="utf-8"))["snapshot"]
    else:
        restored = ChemWorldRuntime.from_snapshot(before)
        restored.execute({"operation": "wait", "duration_s": duration})
        actual_snapshot = json.loads(json.dumps(restored.snapshot().to_dict()))
    # Full snapshots are retained separately; a true comparison is recorded here.
    equal = actual_snapshot == expected_snapshot
    c.exact("full_snapshot_replay_exact", equal, True)
    (out / f"unit{number}-final.json").write_text(json.dumps(expected_snapshot), encoding="utf-8")
    (out / f"unit{number}-replay.json").write_text(json.dumps(actual_snapshot), encoding="utf-8")
    return {
        "info": result.info,
        "prefix_info": prefix.info if prefix else None,
        "reference": reference,
        "expected_time_s": expected_time,
    }


def rejection_unit(c, number):
    model = spec(mode="PRO")
    if number == 15:
        model = replace(model, bath=replace(model.bath, maximum_power_W=1e-10))
    runtime = setup(model)
    before, ledger = physical(runtime), resources(runtime)
    error, refused = "", False
    if number == 15:
        result = runtime.execute({"operation": "wait", "duration_s": 20})
        refused = result.info["transaction_status"] != "committed"
        error = str(result.info["outcome_diagnostics"])
    else:
        try:
            if number == 16:
                replace(model, maximum_pressure_difference_Pa=1)
            elif number == 17:
                payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
                target = payload["world_state"]["inactive_vessels"]["draw"]
                target["phases"]["liquid"]["species_amounts_mol"]["Na+"] *= 2
                target["species_amounts"]["Na+"] *= 2
                ChemWorldRuntime.from_snapshot(payload)
            else:
                replace(model, bath=replace(model.bath, setpoint_K=350, maximum_setpoint_K=360))
        except ValueError as exc:
            refused, error = True, str(exc)
    tokens = {
        15: ("bath power rating",),
        16: ("pressure-difference",),
        17: ("salt basis", "electroneutral"),
        18: ("domain", "range", "outside"),
    }[number]
    c.exact("rejected", refused, True)
    c.exact("diagnostic_reason", any(token in error for token in tokens), True)
    c.exact("physical_state_unchanged", physical(runtime), before)
    c.exact("physical_resources_and_clock_unchanged", resources(runtime), ledger)
    return {"error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    jobs = [
        (
            f"point-{family}-{mode}-{left}-{right}",
            10,
            lambda c, family=family, mode=mode, left=left, right=right: point_unit(
                c, family, mode, left, right
            ),
        )
        for family, mode, left, right in (
            ("ideal", "FO", 10, 60),
            ("ideal", "PRO", 10, 60),
            ("ideal", "reverse", 10, 60),
            ("ideal", "FO", 30, 30),
            ("ideal", "FO", 0, 0),
            ("Davies", "FO", 10, 60),
        )
    ]
    jobs += [
        (
            f"finite-{family}-{mode}",
            12,
            lambda c, family=family, mode=mode: trajectory_unit(c, family, mode),
        )
        for family, mode in (
            ("ideal", "FO"),
            ("ideal", "PRO"),
            ("ideal", "reverse"),
            ("Davies", "FO"),
        )
    ]
    jobs += [(f"world-{n}", 12, lambda c, n=n: world_unit(c, n, args.out)) for n in range(11, 15)]
    jobs += [(f"rejection-{n}", 4, lambda c, n=n: rejection_unit(c, n)) for n in range(15, 19)]
    units = []
    for index, (name, expected, job) in enumerate(jobs, 1):
        local_started, counter, stop = time.monotonic(), Comparisons(), threading.Event()

        def heartbeat(
            stop=stop, index=index, name=name, local_started=local_started,
            counter=counter, expected=expected,
        ):
            while not stop.wait(25):
                print(
                    f"48D unit={index}/18 {name} "
                    f"active elapsed={time.monotonic() - local_started:.1f}s "
                    f"comparisons={len(counter.rows)}/{expected}",
                    flush=True,
                )

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        row = {"unit": index, "name": name, "expected_comparisons": expected}
        try:
            row["details"] = job(counter)
        except Exception:
            row["exception"] = traceback.format_exc()
        finally:
            stop.set()
            thread.join()
        row["comparisons"] = counter.rows
        row["executed_comparisons"] = len(counter.rows)
        row["passed_comparisons"] = sum(check["passed"] for check in counter.rows)
        row["passed"] = (
            "exception" not in row
            and len(counter.rows) == expected
            and all(check["passed"] for check in counter.rows)
        )
        row["elapsed_s"] = time.monotonic() - local_started
        units.append(row)
        (args.out / f"unit-{index:02}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        elapsed = time.monotonic() - started
        rate = f"{index / elapsed:.2f}" if elapsed > 0 else "pending-clock-tick"
        print(
            f"48D {index}/18 {name}: {row['passed_comparisons']}/{expected}; "
            f"{rate} units/s; ETA={elapsed / index * (18 - index):.1f}s",
            flush=True,
        )
    summary = {
        "status": "pass" if all(u["passed"] for u in units) else "fail",
        "mode": "development",
        "units": 18,
        "passed_units": sum(u["passed"] for u in units),
        "expected_comparisons": 172,
        "executed_comparisons": sum(u["executed_comparisons"] for u in units),
        "passed_comparisons": sum(u["passed_comparisons"] for u in units),
        "unexecuted_comparisons": 172 - sum(u["executed_comparisons"] for u in units),
        "exceptions": sum("exception" in u for u in units),
        "elapsed_s": time.monotonic() - started,
        "unit_results": [{k: v for k, v in u.items() if k != "details"} for u in units],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "unit_results"}), flush=True)
    raise SystemExit(0 if summary["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
