"""Fixed 40-unit/320-comparison molecular caloric and finite-device block."""

from __future__ import annotations

import argparse
import json
from math import fsum
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_caloric_flash import solve_molecular_caloric_flash
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.molecular_flash_thermal_world import build_molecular_flash_thermal_world
from reference.molecular28_caloric_cases import caloric_spec, closure_cases, seed, thermal_spec
from reference.molecular28_caloric_reference import lle_state, pr_state, thermal_history


def comparison(name, error, limit):
    return {"name": name, "absolute_error": float(error), "limit": limit, "passed": error <= limit}


def closure_unit(case):
    spec = caloric_spec(case["kind"], case["policy"])
    ref = (
        pr_state(case["z"], temperature=case["temperature_K"])
        if case["kind"] == "PR"
        else lle_state(case["z"], case["temperature_K"])
    )
    ids = spec.model.component_ids
    feed = dict(zip(ids, (case["z"], 1 - case["z"]), strict=True))
    result = solve_molecular_caloric_flash(
        spec,
        feed,
        coordinate=case["coordinate"],
        value=ref[case["coordinate"]],
        seed=seed(spec, feed),
    )
    if not result.success or ref.get("phase_status", "two_phase") != "two_phase":
        return {
            "case": case,
            "success": False,
            "failure_reason": result.failure_reason or "oracle_outside_declared_two_phase_branch",
            "native": result.to_dict(),
            "reference": ref,
            "comparisons": [],
        }
    comparisons = [
        comparison("temperature_K", abs(result.temperature_K - ref["T"]), 2e-4),
        comparison(
            "phase_amount_mol",
            max(
                abs(p.component_amounts_mol[k] - ref["amounts"][i][j])
                for i, p in enumerate(result.phases)
                for j, k in enumerate(ids)
            ),
            2e-6,
        ),
        comparison(
            "density_relative",
            max(
                abs(p.state.molar_density_mol_m3 / ref["densities"][i] - 1)
                for i, p in enumerate(result.phases)
            ),
            2e-5,
        ),
        comparison("enthalpy_J", abs(result.enthalpy_J - ref["H"]), 0.02),
        comparison("entropy_J_K", abs(result.entropy_J_K - ref["S"]), 2e-4),
        comparison("chemical_potential_RT", result.chemical_potential_error_RT, 2e-7),
        comparison("pressure_relative", result.pressure_relative_error, 2e-7),
        comparison("material_mol", result.material_balance_error_mol, 2e-10),
    ]
    return {
        "case": case,
        "success": all(c["passed"] for c in comparisons),
        "native": result.to_dict(),
        "reference": ref,
        "comparisons": comparisons,
    }


def thermal_unit(mode, method):
    spec = thermal_spec(method)
    refs = thermal_history(mode)
    feed = {"propane": 0.006, "butane": 0.004}
    device = ThermalDeviceSpec(
        conductance_W_K=0.001 if mode == "proportional" else 1.0,
        maximum_power_W=0.02,
        maximum_cooling_power_W=0.02,
        setpoint_K=301.0 if mode == "proportional" else 285.0 if mode == "cooler" else 315.0,
        minimum_setpoint_K=280.0,
        maximum_setpoint_K=320.0,
    )
    composition = build_molecular_flash_thermal_world(
        spec,
        feed,
        enthalpy_J=refs[0]["H"],
        seed=seed(spec.equilibrium, feed),
        formulas={"propane": "C3H8", "butane": "C4H10"},
        device=device,
    )
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=28)
    records, errors = [], []
    ids = spec.equilibrium.model.component_ids
    try:

        def capture(ref):
            state = session._state
            phases = [state.phases.phases[p.phase_id] for p in spec.phase_targets]
            densities = [
                fsum(p.species_amounts_mol.values()) / (p.volume_L * 0.001) for p in phases
            ]
            calorics = [
                spec.equilibrium.model.state(
                    state.temperature_K,
                    rho,
                    tuple(
                        p.species_amounts_mol[k] / fsum(p.species_amounts_mol.values()) for k in ids
                    ),
                )
                for p, rho in zip(phases, densities, strict=True)
            ]
            u = fsum(
                fsum(p.species_amounts_mol.values()) * s.internal_energy_J_mol
                for p, s in zip(phases, calorics, strict=True)
            )
            u += 0.1 * (state.temperature_K - 298.15)
            original_u = u if not records else records[0]["U"]
            q = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
            work = state.ledger.boundary_work_J
            row = {
                "T": state.temperature_K,
                "time": state.ledger.time_s,
                "U": u,
                "amounts": [dict(p.species_amounts_mol) for p in phases],
                "densities": densities,
                "heat": q,
                "work": work,
                "first_law_residual_J": u - original_u - q + work,
            }
            records.append(row)
            errors.append(
                [
                    abs(row["T"] - ref["T"]),
                    max(
                        abs(p.species_amounts_mol[k] - ref["amounts"][i][j])
                        for i, p in enumerate(phases)
                        for j, k in enumerate(ids)
                    ),
                    max(abs(rho / ref["densities"][i] - 1) for i, rho in enumerate(densities)),
                    abs(q - ref["heat"]),
                    abs(work - ref["work"]),
                    abs(row["first_law_residual_J"]),
                    max(abs(state.species_amounts[k] - feed[k]) for k in ids),
                    abs(row["time"] - ref["time"]),
                ]
            )

        capture(refs[0])
        for index in range(1, 6):
            if mode == "pause" and index in (3, 4):
                operation = "pause_control" if index == 3 else "resume_control"
                action = session.execute({"operation": operation})
                if action.info["transaction_status"] != "committed":
                    raise ValueError("finite device control rejected: " + str(action.info))
            action = session.execute({"operation": "wait", "duration_s": 40.0})
            if action.info["transaction_status"] != "committed":
                raise ValueError("finite-device wait rejected: " + str(action.info))
            capture(refs[index])
        comparisons = [
            comparison(name, max(row[i] for row in errors), limit)
            for i, (name, limit) in enumerate(
                zip(
                    (
                        "temperature_K",
                        "phase_amount_mol",
                        "density_relative",
                        "heat_J",
                        "work_J",
                        "first_law_J",
                        "material_mol",
                        "clock_s",
                    ),
                    (0.002, 2e-7, 2e-5, 2e-5, 2e-5, 2e-6, 2e-10, 1e-9),
                    strict=True,
                )
            )
        ]
        return {
            "case": {"id": f"thermal-{mode}-{method}", "mode": mode, "method": method},
            "success": all(c["passed"] for c in comparisons),
            "comparisons": comparisons,
            "native": records,
            "reference": refs,
        }
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("qualification results are immutable; choose a new output path")
    started, stopped, completed = monotonic(), Event(), [0]

    def heartbeat():
        while not stopped.wait(30):
            elapsed = monotonic() - started
            rate = completed[0] / elapsed
            print(
                f"28B progress {completed[0]}/40 units, {rate:.3f} unit/s, "
                f"ETA={(40 - completed[0]) / rate if rate else -1:.1f}s",
                flush=True,
            )

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    rows = []
    try:
        jobs = [(c["id"], lambda c=c: closure_unit(c)) for c in closure_cases()]
        jobs += [
            (
                f"thermal-{mode}-{method}",
                lambda mode=mode, method=method: thermal_unit(mode, method),
            )
            for mode in ("heater", "cooler", "proportional", "pause")
            for method in ("BDF", "Radau")
        ]
        for name, run in jobs:
            try:
                row = run()
            except (ValueError, ArithmeticError, RuntimeError, TypeError) as exc:
                row = {
                    "case": {"id": name},
                    "success": False,
                    "failure_reason": str(exc),
                    "comparisons": [],
                }
            rows.append(row)
            completed[0] += 1
            print(
                f"28B {completed[0]}/40 {name}: {'PASS' if row['success'] else 'FAIL'}", flush=True
            )
        result = {
            "label": "development",
            "units_scheduled": 40,
            "units_recorded": len(rows),
            "units_passed": sum(r["success"] for r in rows),
            "comparisons_scheduled": 320,
            "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
            "failures": [r["case"]["id"] for r in rows if not r["success"]],
            "elapsed_s": monotonic() - started,
            "results": rows,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
