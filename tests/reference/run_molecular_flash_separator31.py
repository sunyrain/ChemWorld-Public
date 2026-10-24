"""Fixed 31C twelve-unit block; actual comparisons and unavailable slots stay distinct."""

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

from chemworld.examples.molecular_flash_separator import molecular_flash_separator_example
from chemworld.physchem.molecular_flash_separator import evaluate_molecular_flash_separator
from chemworld.physchem.molecular_flash_separator_inventory import (
    MolecularFlashSeparatorInventorySpec,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_flash_separator import separator_record, unpack_separator
from chemworld.world.molecular_flash_separator_world import build_molecular_flash_separator_world
from reference.molecular_flash_separator31_reference import finite, steady

STEADY_CHECKS = (
    "actual_phase_labels_and_complete_search",
    "phase_rates",
    "compositions",
    "densities",
    "molar_enthalpies",
    "heat_duty",
    "component_balance",
    "pressure",
    "phase_equilibrium",
    "truthful_budget_and_unchanged_declaration",
)
WORLD_CHECKS = (
    "expected_outcome_and_reason",
    "actual_time",
    "species_endpoints",
    "enthalpy_endpoints",
    "component_balance",
    "H_heat_balance",
    "original_U_heat_pressure_work",
    "PH_T_and_volume",
    "actual_resources_earliest_event_or_atomicity",
    "exact_continuation_and_original_ownership",
)


def checks(names):
    return {
        name: {"assessed": False, "passed": False, "detail": "calculation unavailable"}
        for name in names
    }


def check(rows, name, passed, detail=None):
    rows[name] = {"assessed": True, "passed": bool(passed), "detail": detail}


def maximum(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def error_check(rows, name, error, tolerance):
    check(rows, name, error <= tolerance, {"error": float(error), "tolerance": tolerance})


def steady_unit(temperature, z, hot):
    rows = checks(STEADY_CHECKS)
    data = {"checks": rows}
    try:
        spec, _, _, inlet = molecular_flash_separator_example(
            temperature_K=temperature, z=z, hot_feed=hot
        )
        declaration = spec.to_dict()
        data["declaration"] = declaration
        reference = steady(temperature, z, hot)
        actual = evaluate_molecular_flash_separator(spec.separator, inlet)
        data.update(native=actual.to_dict(), reference=reference)
        flash = actual.diagnostics.get("flash", {})
        stabilities = [flash.get("stability", {}), *flash.get("post_split_stability", [])]
        labels = actual.diagnostics["outlet_phase_labels"]
        check(
            rows,
            STEADY_CHECKS[0],
            actual.success
            and labels == reference["labels"]
            and len(actual.outlets) == len(labels)
            and all(s.get("complete", False) for s in stabilities),
            actual.failure_reason,
        )
        attempts = [a for s in stabilities for a in s.get("starts", [])]
        roots = flash.get("solver_attempts", [])
        check(
            rows,
            STEADY_CHECKS[9],
            spec.to_dict() == declaration
            and MolecularFlashSeparatorInventorySpec.from_dict(
                json.loads(json.dumps(declaration))
            ).to_dict()
            == declaration
            and bool(attempts)
            and all("success" in a and "iterations" in a for a in attempts)
            and all(0 <= a["iterations"] <= 200 for a in attempts)
            and all(0 <= a.get("root", {}).get("iterations", 0) <= 200 for a in roots),
            {"tpd_attempts": len(attempts), "split_attempts": len(roots), "budget": 200},
        )
        if actual.success and labels == reference["labels"]:
            pairs = list(zip(actual.outlets, reference["phases"], strict=True))
            errors = (
                max(abs(a.molar_rate_mol_s - b["rate"]) for a, b in pairs),
                max(maximum(a.state.mole_fractions, b["x"]) for a, b in pairs),
                max(abs(a.state.molar_density_mol_m3 / b["rho"] - 1) for a, b in pairs),
                max(abs(a.state.enthalpy_J_mol - b["h"]) for a, b in pairs),
                abs(actual.heat_duty_W - reference["Q"]),
                max(abs(v) for v in actual.diagnostics["component_residuals_mol_s"].values()),
                max(abs(a.state.pressure_Pa / 6e5 - 1) for a, _ in pairs),
            )
            for name, error, tolerance in zip(
                STEADY_CHECKS[1:8], errors, (1e-8, 2e-5, 3e-5, 0.02, 5e-5, 1e-10, 2e-7), strict=True
            ):
                error_check(rows, name, error, tolerance)
            if len(actual.outlets) == 2:
                error_check(rows, STEADY_CHECKS[8], flash["chemical_potential_error_RT"], 2e-7)
            else:
                s = flash["stability"]
                check(
                    rows,
                    STEADY_CHECKS[8],
                    s["complete"] and not s["unstable"] and s["minimum_tpd_RT"] >= -1e-8,
                    {
                        "minimum_tpd_RT": s["minimum_tpd_RT"],
                        "finite_search_complete": s["complete"],
                    },
                )
    except Exception as exc:
        data.update(failure_reason=str(exc), traceback=traceback.format_exc())
    return data


def physical(tank):
    return {
        "amounts": dict(tank.species_amounts),
        "temperature": tank.temperature_K,
        "volume": tank.volume_L,
        "pressure": tank.pressure_Pa,
        "phases": tank.phases.to_dict(),
        "slots": tank.state_slots.to_dict(),
    }


def tanks(runtime):
    return {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}


def world_unit(kind, output):
    rows = checks(WORLD_CHECKS)
    data = {"checks": rows}
    runtime = None
    try:
        z = 0.4 if kind == "source" else 0.6
        spec, amounts, temperatures, _ = molecular_flash_separator_example(z=z)
        if kind == "source":
            amounts = (
                {k: v * ((1e-5 + 0.002) / 0.02) for k, v in amounts[0].items()},
                *amounts[1:],
            )
        elif kind == "receiver_resume":
            rate = next(p["rate"] for p in steady(300.0, z)["phases"] if p["label"] == "gas")
            spec = replace(spec, receiver_capacities_mol=(0.02 + rate * 1.5, 0.1))
        elif kind == "power":
            spec = replace(
                spec,
                separator=replace(
                    spec.separator,
                    device=replace(
                        spec.separator.device, maximum_power_W=1e-6, maximum_cooling_power_W=1e-6
                    ),
                ),
            )
        declaration = spec.to_dict()
        data["declaration"] = declaration
        runtime = ChemWorldRuntime(
            composition=build_molecular_flash_separator_world(
                spec, amounts, temperatures, formulas={"propane": "C3H8", "butane": "C4H10"}
            )
        )
        runtime.reset(seed=31)
        initial = unpack_separator(runtime._state)[2]
        original = {n: physical(t) for n, t in tanks(runtime).items()}
        snapshot = None
        if kind == "receiver_resume":
            first = runtime.execute({"operation": "wait", "duration_s": 0.5})
            data["first_info"] = first.info
            snapshot = output / "unit11-initial-snapshot.json"
            snapshot.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
        duration = {"normal": 1.0, "source": 3.0, "receiver_resume": 2.0, "power": 1.0}[kind]
        outcome = runtime.execute({"operation": "wait", "duration_s": duration})
        data["info"] = outcome.info
        final = unpack_separator(runtime._state)[2]
        data["final_inventory"] = {
            "amounts": final.amounts_mol,
            "enthalpies": final.enthalpies_J,
            "time_s": final.time_s,
        }
        actual_tanks = tanks(runtime)
        current = {n: physical(t) for n, t in actual_tanks.items()}
        if kind == "power":
            exact_reason = "insufficient_separator_heating_or_cooling_power" in str(
                outcome.info["outcome_diagnostics"]
            )
            unchanged = final == initial and original == current
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
                    "negative control: original three physical n/H/T/V/P states and time unchanged",
                )
            check(
                rows,
                WORLD_CHECKS[8],
                unchanged
                and exact_reason
                and runtime._state.ledger.flow_heat_J == 0
                and runtime._state.ledger.boundary_work_J == 0
                and runtime._state.ledger.flow_pump_work_J == 0
                and all(t.ledger.time_s == 0 for t in actual_tanks.values()),
            )
            check(
                rows,
                WORLD_CHECKS[9],
                unchanged
                and declaration == separator_record(runtime._state).settings["physical_model"],
            )
            return data
        expected_time = {"normal": 1.0, "source": 2.0, "receiver_resume": 1.5}[kind]
        event = {
            "normal": None,
            "source": "source_reserve:source",
            "receiver_resume": "receiver_capacity:reactor",
        }[kind]
        diagnostic = outcome.info["outcome_diagnostics"]
        check(
            rows,
            WORLD_CHECKS[0],
            outcome.info["transaction_status"] == "committed"
            and outcome.info["operation_outcome"] == ("success" if event is None else "stopped")
            and (event is None or event in str(diagnostic)),
            diagnostic,
        )
        if outcome.info["transaction_status"] != "committed":
            data["failure_reason"] = "expected finite operation did not execute"
            return data

        def ordered(ns):
            return [[n[k] for k in spec.properties.component_ids] for n in ns]

        reference = finite(ordered(amounts), 300.0, z, False, expected_time)
        data["reference"] = reference
        error_check(rows, WORLD_CHECKS[1], abs(final.time_s - expected_time), 2e-7)
        error_check(
            rows, WORLD_CHECKS[2], maximum(ordered(final.amounts_mol), reference["amounts"]), 2e-9
        )
        error_check(
            rows, WORLD_CHECKS[3], maximum(final.enthalpies_J, reference["enthalpies"]), 2e-4
        )
        error_check(
            rows,
            WORLD_CHECKS[4],
            maximum(
                np.sum(ordered(final.amounts_mol), axis=0),
                np.sum(ordered(initial.amounts_mol), axis=0),
            ),
            1e-10,
        )
        heat, work = runtime._state.ledger.flow_heat_J, runtime._state.ledger.boundary_work_J
        error_check(
            rows,
            WORLD_CHECKS[5],
            abs(sum(final.enthalpies_J) - sum(initial.enthalpies_J) - heat),
            2e-5,
        )
        # Original caloric U, independently evaluated at each actual tank endpoint.
        initial_u = sum(
            sum(n.values()) * spec.at_tp(i, n, temperatures[i]).internal_energy_J_mol
            for i, n in enumerate(initial.amounts_mol)
        )
        final_u = sum(
            sum(final.amounts_mol[i].values())
            * spec.at_tp(
                i, final.amounts_mol[i], actual_tanks[name].temperature_K
            ).internal_energy_J_mol
            for i, name in enumerate(spec.vessel_ids)
        )
        dv = sum(actual_tanks[n].volume_L - original[n]["volume"] for n in spec.vessel_ids) / 1000
        error_check(rows, WORLD_CHECKS[6], abs(final_u - initial_u - heat + 6e5 * dv), 2e-5)
        t_error = max(
            abs(actual_tanks[n].temperature_K - p["T"])
            for n, p in zip(spec.vessel_ids, reference["final"], strict=True)
        )
        v_error = max(
            abs(actual_tanks[n].volume_L / (1000 * p["V"]) - 1)
            for n, p in zip(spec.vessel_ids, reference["final"], strict=True)
        )
        check(
            rows,
            WORLD_CHECKS[7],
            t_error <= 2e-4 and v_error <= 3e-6,
            {"temperature_error_K": t_error, "volume_relative_error": v_error},
        )
        check(
            rows,
            WORLD_CHECKS[8],
            runtime._state.ledger.flow_pump_work_J == 0
            and abs(heat - reference["Q"]) <= 2e-5
            and abs(work - 6e5 * dv) <= 2e-5
            and abs(final.time_s - expected_time) <= 2e-7
            and (event is None or event in str(diagnostic))
            and all(abs(t.ledger.time_s - expected_time) <= 2e-7 for t in actual_tanks.values()),
            {
                "actual_heat_J": heat,
                "expected_heat_J": reference["Q"],
                "boundary_work_J": work,
                "actual_PdV_J": 6e5 * dv,
                "expected_event": event,
            },
        )
        consistent = declaration == separator_record(runtime._state).settings["physical_model"]
        detail = {"declaration_unchanged": consistent}
        if kind == "source":
            consistent = consistent and original[spec.vessel_ids[1]] == current[spec.vessel_ids[1]]
            detail["absent_receiver_unchanged"] = consistent
        if snapshot is not None:
            resumed = output / "unit11-resumed.json"
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).with_name("molecular_flash_separator31_resume.py")),
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
            fresh = (
                json.loads(resumed.read_text(encoding="utf-8")) if process.returncode == 0 else {}
            )
            exact = fresh.get("snapshot") == json.loads(json.dumps(runtime.snapshot().to_dict()))
            consistent = consistent and exact and fresh.get("status") == "committed"
            detail["exact_complete_fresh_process_snapshot"] = exact
        else:
            restored = ChemWorldRuntime.from_snapshot(
                json.loads(json.dumps(runtime.snapshot().to_dict()))
            )
            consistent = (
                consistent and {n: physical(t) for n, t in tanks(restored).items()} == current
            )
            restored.close()
        check(rows, WORLD_CHECKS[9], consistent, detail)
    except Exception as exc:
        data.update(failure_reason=str(exc), traceback=traceback.format_exc())
    finally:
        if runtime is not None:
            runtime.close()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units = [
        (f"T{t}-z{z}-{'hot' if hot else 'cold'}", lambda t=t, z=z, hot=hot: steady_unit(t, z, hot))
        for t in (300.0, 302.0)
        for z in (0.4, 0.6)
        for hot in (False, True)
    ]
    units.extend(
        ("world-" + kind, lambda kind=kind: world_unit(kind, args.output))
        for kind in ("normal", "source", "receiver_resume", "power")
    )
    start, completed = monotonic(), []
    live, stop = {"stage": "starting", "completed": 0}, Event()

    def progress():
        elapsed = monotonic() - start
        n = live["completed"]
        rate = n / elapsed if elapsed > 0 else 0.0
        eta = f"{(12 - n) / rate:.1f}s" if n else "pending first unit"
        print(
            f"31C stage={live['stage']} completed={n}/12 elapsed={elapsed:.1f}s "
            f"rate={rate:.3f} units/s ETA={eta}",
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for i, (name, execute) in enumerate(units, 1):
            live["stage"] = name
            progress()
            tick = monotonic()
            result = execute()
            result.update(id=name, elapsed_s=monotonic() - tick)
            result["comparisons_total"] = len(result["checks"])
            result["comparisons_assessed"] = sum(c["assessed"] for c in result["checks"].values())
            result["comparisons_passed"] = sum(c["passed"] for c in result["checks"].values())
            result["comparisons_unavailable"] = 10 - result["comparisons_assessed"]
            result["passed"] = result["comparisons_passed"] == 10
            if result["comparisons_total"] != 10:
                raise RuntimeError("fixed comparison denominator changed")
            (args.output / f"unit{i:02d}.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
            completed.append(result)
            live["completed"] = i
            progress()
    finally:
        stop.set()
        thread.join()
    assessed = sum(r["comparisons_assessed"] for r in completed)
    passed = sum(r["comparisons_passed"] for r in completed)
    summary = {
        "task": "31C",
        "mode": "development",
        "units_total": 12,
        "units_attempted": len(completed),
        "units_passed": sum(r["passed"] for r in completed),
        "comparisons_total": 120,
        "comparisons_assessed": assessed,
        "comparisons_passed": passed,
        "comparisons_measured_failed": assessed - passed,
        "comparisons_unexecuted_unavailable": 120 - assessed,
        "elapsed_s": monotonic() - start,
        "rows": [
            {
                "id": r["id"],
                "passed": r["passed"],
                "comparisons_assessed": r["comparisons_assessed"],
                "comparisons_passed": r["comparisons_passed"],
                "failed_comparisons": [k for k, c in r["checks"].items() if not c["passed"]],
                "failure_reason": r.get("failure_reason", ""),
            }
            for r in completed
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
