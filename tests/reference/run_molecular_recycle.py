"""Fixed 18-unit analytic recycle and original finite receiver block."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from chemworld.foundation import WorldState
from chemworld.physchem.molecular_recycle import MolecularRecyclePath
from chemworld.runtime.molecular_process_path import advance_molecular_process_receiver


def run(output: Path, summary: Path) -> bool:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from reference.molecular_recycle_cases import expected, path, receiver

    if output.exists() or summary.exists():
        raise ValueError("retain previous attempts; choose fresh output and summary paths")
    records, started = [], monotonic()
    finished = Event()

    def heartbeat():
        while not finished.wait(30):
            elapsed = monotonic() - started
            count = len(records)
            print(
                f"stage=recycle units={count}/18 elapsed_s={elapsed:.1f} "
                f"throughput={count / max(elapsed, 1e-9):.3f}/s",
                flush=True,
            )

    Thread(target=heartbeat, daemon=True).start()

    def unit(name, parameters, operation):
        row = {"unit": name, "parameters": parameters, "comparisons": [], "results": []}

        def check(key, actual, target, *, atol=2e-8, rtol=2e-7):
            passed = (
                actual == target
                if isinstance(target, bool)
                else (abs(actual - target) <= atol + rtol * abs(target))
            )
            row["comparisons"].append(
                {"name": key, "actual": actual, "expected": target, "passed": bool(passed)}
            )

        try:
            operation(row, check)
        except Exception as error:
            row["error"] = {"type": type(error).__name__, "message": str(error)}
            check("completed", False, True)
        records.append(row)
        elapsed = monotonic() - started
        print(
            f"stage=recycle units={len(records)}/18 unit={name} "
            f"throughput={len(records) / elapsed:.2f}/s "
            f"eta_s={(18 - len(records)) * elapsed / len(records):.1f}",
            flush=True,
        )

    def ideal_unit(parameters, row, check):
        model = path(**parameters)
        result = model.evaluate()
        row["results"].append(result.to_dict())
        check("solved", result.success, True)
        if not result.success:
            return
        exact = expected(parameters["x"], parameters["recycle"], parameters["pressure_cycle"])
        check("tear_flow", result.diagnostics["tear_molar_rate_mol_s"], exact["tear_flow"])
        check("product_flow", result.outlet.molar_rate_mol_s, exact["product_flow"])
        check("outlet_T", result.outlet.state.temperature_K, exact["outlet_T"])
        check("mixer_T", result.diagnostics["mixer"]["temperature_K"], exact["mixer_T"])
        check("heat", result.heat_duty_W, exact["heat"])
        check("work", result.shaft_work_W, exact["work"])
        check("energy", result.diagnostics["energy_residual_W"], 0.0, atol=1e-5)
        for i, key in enumerate(model.domain.model.component_ids):
            check(
                f"product_fraction:{key}", result.outlet.state.mole_fractions[i], parameters["x"][i]
            )
            check(
                f"tear_component:{key}",
                result.diagnostics["tear"]["mole_fractions"][i]
                * result.diagnostics["tear_molar_rate_mol_s"],
                exact["tear_flow"] * parameters["x"][i],
            )
            check(
                f"component_balance:{key}", result.diagnostics["component_balance_mol_s"][key], 0.0
            )

    for i, (x, recycle, cycle, start) in enumerate(
        product(((0.25, 0.75), (0.6, 0.4)), (0.2, 0.5), (False, True), (350.0, 450.0))
    ):
        parameters = {"x": x, "recycle": recycle, "pressure_cycle": cycle, "start": start}
        unit(
            f"analytic-{i + 1:02}",
            parameters,
            lambda row, check, p=parameters: ideal_unit(p, row, check),
        )

    def consumer(x, row, check):
        model = path(x=x, real=True, recycle=0.3)
        state = receiver(model)
        before = state.to_dict(include_hidden=True)
        row["declaration"] = model.to_dict()

        def advance(current, selected, duration):
            result = advance_molecular_process_receiver(
                current, selected, duration, phase_id="bulk", maximum_amount_mol=0.3
            )
            row["results"].append({"success": result.success, "diagnostics": result.diagnostics})
            check("receiver_advance", result.success, True)
            if not result.success:
                raise ValueError("original receiver failed")
            check("path_energy", result.diagnostics["path_energy_residual_J"], 0.0, atol=2e-5)
            check(
                "receiver_energy", result.diagnostics["receiver_energy_residual_J"], 0.0, atol=2e-5
            )
            return result.state

        full = advance(state, model, 0.2)
        first = advance(state, model, 0.1)
        decoded = MolecularRecyclePath.from_dict(json.loads(json.dumps(model.to_dict())))
        restored = WorldState.from_dict(json.loads(json.dumps(first.to_dict(include_hidden=True))))
        resumed = advance(restored, decoded, 0.1)
        check("input_unchanged", state.to_dict(include_hidden=True) == before, True)
        for i, key in enumerate(model.domain.model.component_ids):
            check(
                f"actual_product:{key}",
                full.species_amounts[key],
                state.species_amounts[key] + 0.01 * x[i],
            )
            check(f"continued:{key}", resumed.species_amounts[key], full.species_amounts[key])
        check("continued_temperature", resumed.temperature_K, full.temperature_K)
        check("continued_pressure", resumed.pressure_Pa, full.pressure_Pa)
        check("time", full.ledger.time_s, 0.2)
        check("heat", full.ledger.flow_heat_J, 1.0, atol=2e-5)
        check("work", full.ledger.flow_shaft_work_J, 0.0, atol=2e-5)
        check("no_local_jacket_debit", full.ledger.energy_jacket_J, 0.0)
        check("continued_heat", resumed.ledger.flow_heat_J, full.ledger.flow_heat_J, atol=2e-5)

    for i, x in enumerate(((0.25, 0.75), (0.6, 0.4))):
        unit(
            f"native-consumer-{i + 1}",
            {"composition": x},
            lambda row, check, x=x: consumer(x, row, check),
        )
    finished.set()
    checks = [c for r in records for c in r["comparisons"]]
    failures = [
        {"unit": r["unit"], **c} for r in records for c in r["comparisons"] if not c["passed"]
    ]
    header = {
        "mode": "development",
        "design": "33A/EXPERIMENT.md",
        "planned_units": 18,
        "executed_units": len(records),
        "passed_units": sum(all(c["passed"] for c in r["comparisons"]) for r in records),
        "comparisons": len(checks),
        "passed_comparisons": sum(c["passed"] for c in checks),
        "failures": failures,
        "elapsed_s": monotonic() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump({**header, "units": records}, handle, indent=2, allow_nan=False)
    compact = {
        **header,
        "full_artifact": str(output.resolve()),
        "units": [
            {
                "unit": r["unit"],
                "parameters": r["parameters"],
                "comparisons": r["comparisons"],
                "error": r.get("error"),
                "root_trials": [
                    {
                        "success": item.get("success"),
                        "root": item["diagnostics"].get("root"),
                        "trial_evaluations": item["diagnostics"].get("trial_evaluations"),
                        "failed_trials": [
                            t for t in item["diagnostics"].get("trials", []) if not t.get("success")
                        ],
                    }
                    for item in r["results"]
                ],
            }
            for r in records
        ],
    }
    with summary.open("x", encoding="utf-8") as handle:
        json.dump(compact, handle, indent=2, allow_nan=False)
    print(json.dumps(header), flush=True)
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.summary))
