"""Run all fixed 40B original-inventory comparisons, retaining every outcome."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from aqueous40_mineral_cases import cases, owned_inventory
from aqueous40_mineral_reference import invariants, reference, saturation

from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime.aqueous_mineral_equilibrium import equilibrate_aqueous_mineral_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    output = parser.parse_args().output
    if output.exists():
        raise ValueError("retain completed block and choose a new output")
    started, completed = monotonic(), 0
    stop = Event()

    def heartbeat():
        while not stop.wait(30):
            print(f"40B {completed}/24 units, elapsed={monotonic() - started:.1f}s", flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    rows = []
    keys = (
        "species",
        "minerals",
        "water",
        "complementarity",
        "charge",
        "invariants",
        "heat",
        "work",
    )
    tolerances = (2e-6, 2e-6, 1e-7, 2e-7, 1e-10, 1e-9, 2e-4, 1e-7)
    try:
        for case in cases():
            for tight in (False, True):
                checks = dict.fromkeys(keys, False)
                row = {"condition": case["id"], "tight": tight, "checks": checks}
                try:
                    ref = reference(case)
                    spec, ledger, vessel, solid_ids = owned_inventory(case)
                    spec = replace(
                        spec,
                        root_policy=RootPolicy(
                            tolerance=1e-11 if tight else 1e-9,
                            maximum_iterations=150 if tight else 100,
                        ),
                    )
                    result = equilibrate_aqueous_mineral_inventory(
                        ledger,
                        spec,
                        "water",
                        solid_ids,
                        vessel=vessel,
                        temperature_K=case["temperature_K"],
                        pressure_Pa=case["pressure_Pa"],
                    )
                    row.update(reference=ref, native=result.to_dict())
                    if result.success:
                        n = result.equilibrium.amounts_mol
                        scale = max(sum(v for k, v in ref["initial"].items() if k != "H2O"), 1e-6)
                        initial_inv, final_inv = (
                            invariants(case, ref["initial"]),
                            invariants(case, n),
                        )
                        si = saturation(case, n)
                        errors = {
                            "species": max(
                                abs(n[k] - v) / scale for k, v in ref["amounts_mol"].items()
                            ),
                            "minerals": max(
                                abs(n[k] - ref["amounts_mol"][k]) / scale for k in solid_ids
                            ),
                            "water": abs(n["H2O"] - ref["amounts_mol"]["H2O"]),
                            "complementarity": max(
                                abs(v) if n[k] > 0 else max(v, 0.0) for k, v in si.items()
                            ),
                            "charge": abs(n["Na+"] + n.get("K+", 0.0) - n["Cl-"]),
                            "invariants": max(
                                abs(final_inv[k] - v) for k, v in initial_inv.items()
                            ),
                            "heat": abs(result.heat_duty_J - ref["heat_J"]),
                            "work": abs(result.boundary_work_J - ref["work_J"]),
                        }
                        row["errors"] = errors
                        checks.update(
                            {k: errors[k] <= tol for k, tol in zip(keys, tolerances, strict=True)}
                        )
                except Exception as exc:
                    row["error"] = repr(exc)
                rows.append(row)
                completed += 1
                elapsed = monotonic() - started
                print(
                    f"40B {completed}/24 {case['id']} tight={tight}: {sum(checks.values())}/8 "
                    f"rate={completed / elapsed:.2f}/s "
                    f"ETA={(24 - completed) * elapsed / completed:.1f}s",
                    flush=True,
                )
        summary = {
            "units": 24,
            "successful": sum(r.get("native", {}).get("success", False) for r in rows),
            "comparisons": 192,
            "passed": sum(sum(r["checks"].values()) for r in rows),
            "failures": [
                {
                    "condition": r["condition"],
                    "tight": r["tight"],
                    "checks": [k for k, v in r["checks"].items() if not v],
                    "error": r.get("error"),
                }
                for r in rows
                if not all(r["checks"].values())
            ],
            "elapsed_s": monotonic() - started,
        }
        payload = {"mode": "development", "fixed_cases": cases(), "summary": summary, "runs": rows}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        stop.set()
        thread.join()


if __name__ == "__main__":
    main()
