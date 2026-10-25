"""Fixed 32B twenty-unit/two-hundred-comparison block; no adaptive coverage."""

from __future__ import annotations

import argparse
import json
import subprocess
import traceback
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np

from chemworld.examples.molecular_cascade import molecular_cascade_example
from chemworld.physchem.molecular_cascade import solve_molecular_cascade
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_cascade import cascade_record, unpack_cascade
from chemworld.world.molecular_cascade_world import build_molecular_cascade_world
from reference.molecular_cascade32_reference import finite, steady
from reference.molecular_cascade32_report import assessed_checks

STEADY_CHECKS = (
    "success_and_complete_phase_checks",
    "stage_flows",
    "compositions",
    "densities",
    "molar_enthalpies",
    "stage_duties",
    "component_balance",
    "pressure",
    "chemical_potential",
    "truthful_budget_and_unchanged_declaration",
)
WORLD_CHECKS = (
    "expected_outcome",
    "actual_time",
    "four_species_endpoints",
    "four_H_endpoints",
    "component_conservation",
    "H_heat_balance",
    "U_heat_pressure_work",
    "EOS_volume_PH",
    "actual_resources_and_earliest_event_or_atomic_rejection",
    "exact_continuation_or_original_state_consistency",
)


def checks(names):
    return {name: {"passed": False, "detail": "calculation unavailable"} for name in names}


def check(rows, name, passed, detail=None):
    rows[name] = {"passed": bool(passed), "detail": detail}


def maximum(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def steady_unit(temperatures, ratio):
    spec, _, _, feeds = molecular_cascade_example(temperatures_K=temperatures, flow_ratio=ratio)
    before = spec.to_dict()
    reference = steady(temperatures, ratio)
    actual = solve_molecular_cascade(spec.cascade, *feeds)
    rows = checks(STEADY_CHECKS)
    diag = actual.diagnostics
    check(
        rows,
        STEADY_CHECKS[0],
        actual.success and reference["success"] and len(diag["phase_checks"]) == len(temperatures),
        actual.failure_reason,
    )
    check(
        rows,
        STEADY_CHECKS[9],
        spec.to_dict() == before
        and diag["eos_evaluations"] > 0
        and diag["root_policy"]["maximum_iterations"] == 100
        and diag.get("root", {}).get("iterations", 101) <= 100,
        {"eos_evaluations": diag["eos_evaluations"], "root": diag.get("root")},
    )
    if actual.success and reference["success"]:
        pairs = [
            (a[side], b, k)
            for a, b in zip(diag["stages"], reference["stages"], strict=True)
            for k, side in enumerate(("vapor", "liquid"))
        ]
        errors = [
            max(abs(a["molar_rate_mol_s"] - b["rates"][k]) for a, b, k in pairs),
            max(maximum(a["state"]["mole_fractions"], b["compositions"][k]) for a, b, k in pairs),
            max(
                abs(a["state"]["molar_density_mol_m3"] / b["densities"][k] - 1) for a, b, k in pairs
            ),
            max(abs(a["state"]["enthalpy_J_mol"] - b["enthalpies"][k]) for a, b, k in pairs),
            max(
                abs(a["heat_duty_W"] - b["Q"])
                for a, b in zip(diag["stages"], reference["stages"], strict=True)
            ),
            diag["component_residual_mol_s"],
            diag["pressure_relative_error"],
            diag["chemical_potential_error_RT"],
        ]
        for name, error, tolerance in zip(
            STEADY_CHECKS[1:9],
            errors,
            (1e-8, 2e-5, 3e-5, 0.02, 5e-5, 1e-10, 2e-7, 2e-7),
            strict=True,
        ):
            check(rows, name, error <= tolerance, {"error": error, "tolerance": tolerance})
    return {
        "checks": rows,
        "native": actual.to_dict(),
        "reference": reference,
        "declaration": before,
    }


def world_unit(kind, output, *, ratio=1.25):
    # Explicit fixed ratio, independent of the user example's feasible default.
    spec, amounts, temperatures, _ = molecular_cascade_example(4, flow_ratio=ratio)
    reference = steady((300.0,) * 4, ratio)
    duration = 1.0
    if kind == "source":
        amounts = (
            {k: v * ((1e-5 + 0.001 * 2) / 0.02) for k, v in amounts[0].items()},
            *amounts[1:],
        )
        duration = 3.0
    elif kind == "receiver_resume":
        spec = replace(
            spec, receiver_capacities_mol=(0.02 + reference["stages"][0]["rates"][0] * 1.5, 0.1)
        )
        duration = 2.0
    elif kind == "power":
        spec = replace(
            spec,
            cascade=replace(
                spec.cascade,
                devices=tuple(
                    replace(d, maximum_power_W=1e-6, maximum_cooling_power_W=1e-6)
                    for d in spec.cascade.devices
                ),
            ),
        )
    declaration = spec.to_dict()
    rows = checks(WORLD_CHECKS)
    data = {"checks": rows, "reference": reference, "declaration": declaration}
    runtime = None
    try:
        world = build_molecular_cascade_world(
            spec, amounts, temperatures, formulas={"propane": "C3H8", "butane": "C4H10"}
        )
        runtime = ChemWorldRuntime(composition=world)
        runtime.reset(seed=32)
        initial = unpack_cascade(runtime._state)[2]
        snapshot = None
        if kind == "receiver_resume":
            first = runtime.execute({"operation": "wait", "duration_s": 0.5})
            data["first_info"] = first.info
            data["first_time_s"] = unpack_cascade(runtime._state)[2].time_s
            snapshot = output / "unit19-initial-snapshot.json"
            snapshot.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
        outcome = runtime.execute({"operation": "wait", "duration_s": duration})
        data["info"] = outcome.info
        final = unpack_cascade(runtime._state)[2]
        data["final_inventory"] = {
            "amounts": final.amounts_mol,
            "enthalpies": final.enthalpies_J,
            "time_s": final.time_s,
        }
        if kind == "power":
            unchanged = final == initial
            exact_reason = "insufficient_stage_heating_or_cooling_power" in str(
                outcome.info["outcome_diagnostics"]
            )
            check(
                rows,
                WORLD_CHECKS[0],
                exact_reason and outcome.info["operation_outcome"] == "numerical_failure",
                outcome.info["outcome_diagnostics"],
            )
            for name in WORLD_CHECKS[1:8]:
                check(
                    rows,
                    name,
                    unchanged,
                    "negative control: actual four n/H states and time unchanged",
                )
            check(
                rows,
                WORLD_CHECKS[8],
                unchanged
                and exact_reason
                and runtime._state.ledger.flow_heat_J == 0
                and runtime._state.ledger.boundary_work_J == 0,
                "atomic rejection requires insufficient-duty reason",
            )
            check(
                rows,
                WORLD_CHECKS[9],
                declaration == cascade_record(runtime._state).settings["physical_model"],
            )
            return data
        expected_time = {"normal": 1.0, "source": 2.0, "receiver_resume": 1.5}[kind]
        if outcome.info["transaction_status"] != "committed":
            data["failure_reason"] = "expected finite operation was not executed"
            return data
        target = finite([list(n.values()) for n in amounts], (300.0,) * 4, ratio, expected_time)
        if not target["success"]:
            data["failure_reason"] = "independent model has no positive two-phase flow solution"
            return data
        check(
            rows,
            WORLD_CHECKS[0],
            outcome.info["operation_outcome"] == ("success" if kind == "normal" else "stopped"),
        )
        check(rows, WORLD_CHECKS[1], abs(final.time_s - expected_time) <= 2e-7)
        check(
            rows,
            WORLD_CHECKS[2],
            maximum([list(n.values()) for n in final.amounts_mol], target["amounts"]) <= 2e-9,
        )
        check(rows, WORLD_CHECKS[3], maximum(final.enthalpies_J, target["enthalpies"]) <= 2e-4)
        check(
            rows,
            WORLD_CHECKS[4],
            maximum(
                np.sum([list(n.values()) for n in final.amounts_mol], axis=0),
                np.sum([list(n.values()) for n in initial.amounts_mol], axis=0),
            )
            <= 1e-10,
        )
        heat = runtime._state.ledger.flow_heat_J
        check(
            rows,
            WORLD_CHECKS[5],
            abs(sum(final.enthalpies_J) - sum(initial.enthalpies_J) - heat) <= 2e-5,
        )
        actual_states = {
            runtime._state.vessel_id: runtime._state,
            **runtime._state.inactive_vessels,
        }
        dv = sum(actual_states[n].volume_L / 1000 for n in spec.vessel_ids) - sum(
            p["V"] for p in target["initial"]
        )
        du = sum(final.enthalpies_J) - sum(initial.enthalpies_J) - 6e5 * dv
        check(rows, WORLD_CHECKS[6], abs(du - heat + 6e5 * dv) <= 2e-5)
        check(
            rows,
            WORLD_CHECKS[7],
            all(
                abs(actual_states[n].volume_L / (1000 * p["V"]) - 1) <= 3e-6
                and abs(actual_states[n].temperature_K - p["T"]) <= 2e-4
                for n, p in zip(spec.vessel_ids, target["final"], strict=True)
            ),
        )
        check(
            rows,
            WORLD_CHECKS[8],
            runtime._state.ledger.flow_pump_work_J == 0
            and abs(final.time_s - expected_time) <= 2e-7,
        )
        consistent = declaration == cascade_record(runtime._state).settings["physical_model"]
        if snapshot is not None:
            resumed = output / "unit19-resumed.json"
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).with_name("molecular_cascade32_resume.py")),
                    str(snapshot),
                    str(resumed),
                    "2",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            data["resume_stderr"] = process.stderr
            consistent = process.returncode == 0 and json.loads(
                resumed.read_text(encoding="utf-8")
            )["snapshot"] == json.loads(json.dumps(runtime.snapshot().to_dict()))
            consistent = (
                consistent
                and data["first_info"]["transaction_status"] == "committed"
                and abs(data["first_time_s"] - 0.5) <= 2e-7
            )
        check(rows, WORLD_CHECKS[9], consistent)
    except (ValueError, RuntimeError, ArithmeticError) as exc:
        data["failure_reason"] = str(exc)
        data["traceback"] = traceback.format_exc()
    finally:
        if runtime is not None:
            runtime.close()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units = []
    for n in (1, 2, 4, 6):
        for program in ("uniform", "linear"):
            ts = (
                (300.0,) * n
                if program == "uniform"
                else tuple(np.linspace(300, 302, n))
                if n > 1
                else (302.0,)
            )
            ts = tuple(map(float, ts))
            for ratio in (0.75, 1.25):
                units.append(
                    (
                        f"N{n}-{program}-ratio{ratio}",
                        lambda ts=ts, ratio=ratio: steady_unit(ts, ratio),
                    )
                )
    for kind in ("normal", "source", "receiver_resume", "power"):
        units.append(("world-" + kind, lambda kind=kind: world_unit(kind, args.output)))
    start, completed = monotonic(), []
    live = {"stage": "starting", "completed": 0}
    stop = Event()

    def heartbeat():
        while not stop.wait(30):
            elapsed = monotonic() - start
            print(
                f"32B stage={live['stage']} completed={live['completed']}/20 "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for i, (name, execute) in enumerate(units, 1):
            live["stage"] = name
            print(f"32B start {i}/20 {name}", flush=True)
            tick = monotonic()
            try:
                result = execute()
            except Exception as exc:
                result = {
                    "checks": checks(STEADY_CHECKS if i <= 16 else WORLD_CHECKS),
                    "failure_reason": str(exc),
                    "traceback": traceback.format_exc(),
                }
            result.update(id=name, elapsed_s=monotonic() - tick)
            result["comparisons_total"] = len(result["checks"])
            result["comparisons_passed"] = sum(c["passed"] for c in result["checks"].values())
            if result["comparisons_total"] != 10:
                raise RuntimeError("fixed comparison denominator changed")
            result["passed"] = result["comparisons_passed"] == 10
            result["comparison_availability"] = assessed_checks(i, result)
            result["comparisons_assessed"] = sum(
                c["assessed"] for c in result["comparison_availability"].values()
            )
            result["comparisons_unexecuted_unavailable"] = 10 - result["comparisons_assessed"]
            (args.output / f"unit{i:02d}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            completed.append(result)
            live["completed"] = i
            elapsed = monotonic() - start
            print(
                f"32B completed={i}/20 comparisons={result['comparisons_passed']}/10 "
                f"rate={i / elapsed:.3f} units/s ETA={(20 - i) * elapsed / i:.1f}s",
                flush=True,
            )
    finally:
        stop.set()
        thread.join()
    summary = {
        "task": "32B",
        "mode": "development",
        "units_executed": len(completed),
        "units_total": 20,
        "units_passed": sum(r["passed"] for r in completed),
        "comparison_slots_filled": 10 * len(completed),
        "comparisons_assessed": sum(r["comparisons_assessed"] for r in completed),
        "comparisons_unexecuted_unavailable": sum(
            r["comparisons_unexecuted_unavailable"] for r in completed
        ),
        "comparisons_total": 200,
        "comparisons_passed": sum(r["comparisons_passed"] for r in completed),
        "elapsed_s": monotonic() - start,
        "rows": [
            {
                "id": r["id"],
                "passed": r["passed"],
                "comparisons_passed": r["comparisons_passed"],
                "failed_comparisons": [k for k, c in r["checks"].items() if not c["passed"]],
                "failure_reason": r.get(
                    "failure_reason", r.get("native", {}).get("failure_reason", "")
                ),
            }
            for r in completed
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
