"""Execute the prospective 38D 16-unit / 160-comparison development block."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import energy_grain_qsd38_cases as inputs
import energy_grain_qsd38_reference as oracle
import numpy as np

from chemworld.physchem.energy_grain_binding import energy_grain_world
from chemworld.physchem.energy_grain_reduction import energy_grain_qsd
from chemworld.physchem.energy_grain_specs import EnergyGrainMasterEquationSpec, R
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.energy_grain_analysis import analyze_energy_grain_world_state


def numeric(name, actual, expected, *, atol=1e-9, rtol=3e-7):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return {
        "name": name,
        "passed": bool(np.all(np.isfinite(a)) and np.allclose(a, e, atol=atol, rtol=rtol)),
        "actual": a.tolist(),
        "expected": e.tolist(),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": float(np.max(np.abs(a - e))),
    }


def exact(name, conditions):
    return {"name": name, "passed": all(conditions.values()), "conditions": conditions}


def unit(model, temperature, baths, record):
    names = [g.species_id for g in model.wells[0].grains]
    products = [s.species_id for s in model.sinks]
    amounts = {g.species_id: 0.0 for g in model.molecular_states} | baths
    amounts[names[0]], amounts[products[0]] = 0.002, 0.001
    grains = {name: amounts[name] for name in names}
    report = energy_grain_qsd(
        model, temperature_K=temperature, bath_concentrations_mol_L=baths, grain_amounts_mol=grains
    )
    reference = oracle.solve(model.to_dict(), temperature, baths)
    q = np.asarray([report["qsd_fractions"][s] for s in names])
    left = np.asarray([report["left_mode_weights"][s] for s in names])
    rate = report["effective_rate_s_inverse"]
    product_rates = np.asarray([report["product_rates_s_inverse"][s] for s in products])
    branch = np.asarray([report["qsd_product_branching"][s] for s in products])
    gap = report["spectral_gap_s_inverse"]
    checks = record["comparisons"]
    record["source_report"] = report
    checks.append(numeric("right-qsd", q, reference["q"]))
    checks.append(numeric("total-decay", rate, reference["rate"]))
    checks.append(
        numeric(
            "products-and-branches",
            np.r_[product_rates, branch],
            np.r_[reference["product_rates"], reference["product_rates"] / reference["rate"]],
        )
    )
    checks.append(
        numeric(
            "second-and-gap",
            [report["second_decay_rate_s_inverse"], gap],
            [reference["second"], reference["second"] - reference["rate"]],
        )
    )
    checks.append(numeric("left-mode", left, reference["left"]))
    checks.append(
        numeric(
            "normalization-and-positivity",
            [sum(q), left @ q, sum(branch), min(q) > 0],
            [1, 1, 1, 1],
            atol=1e-12,
            rtol=0,
        )
    )
    checks.append(
        numeric(
            "qsd-exact-exponential",
            oracle.qsd_trajectory(reference, q, rate),
            np.r_[q * np.exp(-0.6), branch * (1 - np.exp(-0.6))],
        )
    )
    initial = np.zeros(len(q))
    initial[0] = 1
    checks.append(
        numeric(
            "arbitrary-initial-asymptote",
            oracle.scaled_long_time(reference, initial, rate, gap),
            q * (left @ initial),
        )
    )
    reactor = GasReactorNetworkSpec((GasReactorNodeSpec("cell", 1),), (), temperature, "cell")
    composition = energy_grain_world(model, reactor, {"cell": amounts})
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=38)
        before = runtime._state.to_dict()
        current = analyze_energy_grain_world_state(runtime._state, composition)
        record["world_report"] = current
        expected_pressure = sum(amounts.values()) * 1000 * R * temperature
        actual_pressure = current["world_query"]["total_gas_pressure_Pa"]
        checks.append(
            exact(
                "original-world-read-only-and-pressures",
                {
                    "same_query": {k: v for k, v in current.items() if k != "world_query"}
                    == report,
                    "state_unchanged": before == runtime._state.to_dict(),
                    "actual_total_pressure": bool(
                        np.isclose(actual_pressure, expected_pressure, atol=1e-7, rtol=1e-12)
                    ),
                    "bath_pressure_is_distinct": actual_pressure > current["bath_pressure_Pa"],
                    "actual_bath_inventory": current["world_query"]["bath_amounts_mol"] == baths,
                },
            )
        )
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
        restored = ChemWorldRuntime.from_snapshot(snapshot)
        source_restored = EnergyGrainMasterEquationSpec.from_dict(
            json.loads(json.dumps(model.to_dict()))
        )
        checks.append(
            exact(
                "source-and-world-json-replay",
                {
                    "world_query_replay": analyze_energy_grain_world_state(
                        restored._state, composition
                    )
                    == current,
                    "source_query_replay": energy_grain_qsd(
                        source_restored,
                        temperature_K=temperature,
                        bath_concentrations_mol_L=baths,
                        grain_amounts_mol=grains,
                    )
                    == report,
                    "report_json": json.loads(json.dumps(current, allow_nan=False)) == current,
                },
            )
        )
    finally:
        runtime.close()
        if restored:
            restored.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, started = [], time.perf_counter()
    cases = inputs.cases()
    for index, (name, model, temperature, baths) in enumerate(cases, 1):
        begin = time.perf_counter()
        record = {
            "unit": name,
            "comparisons": [],
            "input": {"model": model.to_dict(), "temperature_K": temperature, "baths_mol_L": baths},
        }
        try:
            unit(model, temperature, baths, record)
        except Exception as exc:
            record["exception"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        while len(record["comparisons"]) < 10:
            record["comparisons"].append(
                {"name": f"unresolved-slot-{len(record['comparisons']) + 1}", "passed": False}
            )
        record["elapsed_s"] = time.perf_counter() - begin
        record["passed"] = all(c["passed"] for c in record["comparisons"])
        records.append(record)
        (args.output / f"{index:02}-{name}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.perf_counter() - started
        print(
            f"QSD comparison {index}/{len(cases)}; passed={record['passed']}; "
            f"rate={index / elapsed:.2f} unit/s; ETA={(len(cases) - index) * elapsed / index:.1f}s",
            flush=True,
        )
    summary = {
        "mode": "development",
        "units": len(records),
        "comparisons": sum(len(r["comparisons"]) for r in records),
        "units_passed": sum(r["passed"] for r in records),
        "comparisons_passed": sum(c["passed"] for r in records for c in r["comparisons"]),
        "elapsed_s": time.perf_counter() - started,
        "failures": [
            {"unit": r["unit"], "comparison": c["name"], "exception": r.get("exception")}
            for r in records
            for c in r["comparisons"]
            if not c["passed"]
        ],
        "scope": (
            "fixed-condition single-well finite-generator QSD; "
            "no dynamic reduction or continuum claim"
        ),
        "reference": (
            "independent matrix assembly; analytic two-state roots/general nonsymmetric eig; expm"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
