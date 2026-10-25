"""Prospectively fixed two-phase qualification and original-domain adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.physchem.molecular_flash import molecular_tp_flash
from reference.molecular28_caloric_cases import caloric_spec, closure_cases
from reference.molecular28_caloric_reference import pr_state
from reference.run_molecular28_caloric import closure_unit, comparison, thermal_unit


def adjudication_unit():
    spec = caloric_spec()
    feed = {"propane": 0.5, "butane": 0.5}
    ref = pr_state(0.5, temperature=298.0)
    native = molecular_tp_flash(
        spec.model.residual_model, 298.0, 6e5, feed, basins=spec.basins,
        max_iterations=200, tpd_tolerance=1e-8,
    )
    checks = [comparison("root_success", float(not native.success), 0)]
    if native.success:
        phase = native.phases[0]
        state = spec.model.state(298.0, phase.state.molar_density_mol_m3, (0.5, 0.5))
        checks += [
            comparison("single_phase_classification", float(
                native.phase_status != ref["phase_status"] or len(native.phases) != 1), 0),
            comparison("stable_liquid_basin", float(
                phase.basin != "liquid" or not native.stability.complete
                or native.stability.unstable), 0),
            comparison("density_relative", abs(
                phase.state.molar_density_mol_m3 / ref["densities"][0] - 1), 2e-5),
            comparison("enthalpy_J", abs(state.enthalpy_J_mol - ref["H"]), 0.02),
            comparison("entropy_J_K", abs(state.entropy_J_mol_K - ref["S"]), 2e-4),
            comparison("material_mol", max(abs(phase.component_amounts_mol[k] - feed[k])
                for k in feed), 2e-10),
            comparison("pressure_relative", native.pressure_relative_error, 2e-7),
        ]
    return {"case": {"id": "original-298-0.5-TP-adjudication"},
            "success": len(checks) == 8 and all(c["passed"] for c in checks),
            "native": native.to_dict(), "reference": ref, "comparisons": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    if output.exists():
        raise ValueError("choose a new immutable result path")
    cases = []
    for original in closure_cases():
        case = dict(original)
        if case["kind"] == "PR" and case["temperature_K"] == 298:
            case.update(temperature_K=300.0, id=case["id"].replace("298", "300"))
        cases.append(case)
    jobs = [(c["id"], lambda c=c: closure_unit(c)) for c in cases]
    jobs += [(f"thermal-{mode}-{method}", lambda mode=mode, method=method:
              thermal_unit(mode, method))
             for mode in ("heater", "cooler", "proportional", "pause")
             for method in ("BDF", "Radau")]
    jobs += [("original-298-0.5-TP-adjudication", adjudication_unit)]
    rows, stopped, started = [], Event(), monotonic()

    def heartbeat():
        while not stopped.wait(30):
            elapsed = monotonic() - started
            rate = len(rows) / elapsed
            print(f"28B completion {len(rows)}/41 units; {rate:.3f}/s; "
                  f"ETA={(41-len(rows))/rate if rate else -1:.1f}s", flush=True)

    worker = Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        for name, run in jobs:
            try:
                row = run()
            except (ValueError, ArithmeticError, RuntimeError, TypeError, AttributeError) as exc:
                row = {"case": {"id": name}, "success": False,
                       "failure_reason": str(exc), "comparisons": []}
            rows.append(row)
            print(f"28B {len(rows)}/41 {name}: {row['success']}", flush=True)
        result = {"label": "development", "units_scheduled": 41,
                  "units_recorded": len(rows), "units_passed": sum(r["success"] for r in rows),
                  "comparisons_scheduled": 328,
                  "comparisons_executed": sum(len(r["comparisons"]) for r in rows),
                  "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
                  "failures": [r["case"]["id"] for r in rows if not r["success"]],
                  "elapsed_s": monotonic() - started, "results": rows}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
    finally:
        stopped.set()
        worker.join()


if __name__ == "__main__":
    main()
