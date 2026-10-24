"""Fixed 24-unit/192-comparison original CALPHAD thermal development block."""

from __future__ import annotations

import argparse
import json
from math import fsum
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.physchem.calphad_caloric_equilibrium import solve_calphad_ph
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.calphad_thermal import calphad_inventory_enthalpy_j
from chemworld.world.calphad_thermal_world import build_calphad_thermal_world
from reference.calphad42_thermal_cases import (
    caloric_spec,
    device,
    ph_cases,
    thermal_cases,
    thermal_spec,
)
from reference.calphad42_thermal_reference import binary_state, history, pure_state


def check(name, error, tolerance):
    return {
        "name": name,
        "absolute_error": float(error),
        "tolerance": tolerance,
        "passed": bool(error <= tolerance),
    }


def native_phases(result):
    return [
        {
            "name": p.phase_name,
            "n": p.amount_mol_atoms,
            "x": p.element_fractions,
            "sites": list(p.site_fractions),
        }
        for p in result.phases
    ]


def phase_errors(actual, expected):
    elements = sorted({e for p in (*actual, *expected) for e in p["x"]})
    a = sorted(actual, key=lambda p: (p["name"], tuple(p["x"].get(e, 0.0) for e in elements)))
    b = sorted(expected, key=lambda p: (p["name"], tuple(p["x"].get(e, 0.0) for e in elements)))
    if len(a) != len(b) or [p["name"] for p in a] != [p["name"] for p in b]:
        return 1.0, 1.0, 1.0, 1.0
    amount = max(abs(p["n"] - q["n"]) for p, q in zip(a, b, strict=True))
    fraction = amount / max(fsum(p["n"] for p in b), 1e-20)
    composition = max(
        abs(p["x"].get(e, 0) - q["x"].get(e, 0))
        for p, q in zip(a, b, strict=True)
        for e in set(p["x"]) | set(q["x"])
    )
    site = max(
        abs(s[k] - t[k])
        for p, q in zip(a, b, strict=True)
        for s, t in zip(p["sites"], q["sites"], strict=True)
        for k in s
    )
    elemental_amount = max(
        abs(p["n"] * p["x"].get(e, 0.0) - q["n"] * q["x"].get(e, 0.0))
        for p, q in zip(a, b, strict=True)
        for e in elements
    )
    return 0.0, fraction, max(composition, site), elemental_amount


def ph_unit(case):
    reference = (
        pure_state(temperature=case["T"], beta=case["beta"])
        if case["kind"] == "pure"
        else binary_state(case["z"], temperature=case["T"])
    )
    result = solve_calphad_ph(
        caloric_spec(case["kind"], case["policy"]), case["feed"], reference["H"]
    )
    if not result.success:
        return {
            "case": case,
            "success": False,
            "failure_reason": result.failure_reason,
            "native": result.to_dict(),
            "reference": reference,
            "comparisons": [],
        }
    phases = native_phases(result)
    identity, fraction, constitution, _ = phase_errors(
        [p for p in phases if p["n"] > 1e-6], [p for p in reference["phases"] if p["n"] > 1e-6]
    )
    post = result.diagnostics["attempts"][-1]["post_tangent"]
    stable = all(p["complete"] and p["minimum_distance_J_mol"] >= -1e-4 for p in post)
    comparisons = [
        check("temperature_K", abs(result.temperature_K - reference["T"]), 2e-4),
        check("phase_identity", identity, 0.0),
        check("phase_fraction", fraction, 2e-6),
        check("constitution", constitution, 2e-6),
        check("enthalpy_J_mol", abs(result.enthalpy_J - reference["H"]), 0.02),
        check("entropy_J_mol_K", abs(result.entropy_J_K - reference["S"]), 2e-4),
        check("elements_mol_mol", result.element_residual_mol, 2e-10),
        check("post_tangent_complete", 0 if stable else 1, 0.0),
    ]
    return {
        "case": case,
        "success": all(c["passed"] for c in comparisons),
        "comparisons": comparisons,
        "native": result.to_dict(),
        "reference": reference,
    }


def thermal_unit(case):
    spec = thermal_spec(case["kind"], case["method"])
    refs = history(case)
    feed = {"FE": 0.001} if case["kind"] == "pure" else {"FE": 0.0003, "NI": 0.0007}
    world = build_calphad_thermal_world(spec, feed, enthalpy_J=refs[0]["H"], device=device(case))
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset(seed=42)
    rows, errors, diagnostics = [], [], []
    try:

        def capture(reference):
            state = runtime._state
            h = calphad_inventory_enthalpy_j(state, spec)
            q = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
            work = state.ledger.boundary_work_J
            u = h - spec.pressure_Pa * state.volume_L * 0.001
            phases = []
            for p in state.phases.phases.values():
                n = fsum(p.species_amounts_mol.values())
                phases.append(
                    {
                        "name": p.metadata["tdb_phase"],
                        "n": n,
                        "x": {e: v / n for e, v in p.species_amounts_mol.items()},
                        "sites": p.metadata["site_fractions"],
                    }
                )
            row = {
                "T": state.temperature_K,
                "H": h,
                "U": u,
                "phases": phases,
                "heat": q,
                "work": work,
                "time": state.ledger.time_s,
            }
            _, _, _, phase_amount = phase_errors(phases, reference["phases"])
            firstlaw = abs(u - (rows[0]["U"] if rows else u) - q + work)
            closure = abs(h - (rows[0]["H"] if rows else h) - q)
            errors.append(
                [
                    abs(row["T"] - reference["T"]),
                    phase_amount,
                    abs(q - reference["heat"]),
                    abs(work - reference["work"]),
                    firstlaw,
                    max(abs(state.species_amounts[e] - feed[e]) for e in feed),
                    abs(row["time"] - reference["time"]),
                    closure,
                ]
            )
            rows.append(row)

        capture(refs[0])
        for index, (start, end) in enumerate(
            zip(case["times"][:-1], case["times"][1:], strict=True), 1
        ):
            if case["id"] == "binary-pause" and start in (80.0, 120.0):
                action = runtime.execute(
                    {"operation": "pause_control" if start == 80 else "resume_control"}
                )
                if action.info["transaction_status"] != "committed":
                    raise ValueError("CALPHAD device control failed")
            action = runtime.execute({"operation": "wait", "duration_s": end - start})
            diagnostics.append(action.info["outcome_diagnostics"])
            if action.info["transaction_status"] != "committed":
                diagnostics.append(action.info)
                raise ValueError("CALPHAD wait not committed")
            capture(refs[index])
        comparisons = [
            check(name, max(row[i] for row in errors), limit)
            for i, (name, limit) in enumerate(
                zip(
                    (
                        "temperature_K",
                        "phase_element_mol",
                        "heat_J",
                        "work_J",
                        "first_law_J",
                        "elements_mol",
                        "clock_s",
                        "PH_latent_J",
                    ),
                    (0.002, 2e-7, 2e-5, 2e-5, 2e-6, 2e-10, 1e-9, 2e-5),
                    strict=True,
                )
            )
        ]
        return {
            "case": case,
            "success": all(c["passed"] for c in comparisons),
            "comparisons": comparisons,
            "native": rows,
            "reference": refs,
            "advance_diagnostics": diagnostics,
        }
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        return {
            "case": case,
            "success": False,
            "failure_reason": str(exc),
            "comparisons": [],
            "native": rows,
            "reference": refs,
            "advance_diagnostics": diagnostics,
        }
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise ValueError("fixed evidence is immutable; choose a new output")
    started, stopped, completed = monotonic(), Event(), [0]

    def heartbeat():
        while not stopped.wait(30):
            elapsed = monotonic() - started
            rate = completed[0] / elapsed
            print(
                f"42B progress {completed[0]}/24; {rate:.3f} unit/s; "
                f"ETA {(24 - completed[0]) / rate if rate else -1:.1f}s",
                flush=True,
            )

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    rows = []
    try:
        jobs = [(c["id"], lambda c=c: ph_unit(c)) for c in ph_cases()]
        jobs += [
            (c["id"] + "-" + c["method"], lambda c=c: thermal_unit(c)) for c in thermal_cases()
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
                f"42B {completed[0]}/24 {name}: {'PASS' if row['success'] else 'FAIL'}", flush=True
            )
        result = {
            "label": "development",
            "units_scheduled": 24,
            "units_recorded": len(rows),
            "units_passed": sum(r["success"] for r in rows),
            "comparisons_scheduled": 192,
            "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
            "observations_scheduled": 84,
            "observations_recorded": sum(len(r.get("native", [])) for r in rows[12:]),
            "failures": [r["case"] for r in rows if not r["success"]],
            "elapsed_s": monotonic() - started,
            "results": rows,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
