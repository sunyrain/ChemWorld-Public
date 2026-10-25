"""Seven fixed independent teqp source-parameter recovery units; no source fitting in teqp."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import teqp
from run_multifluid_core import R, source_input

from chemworld.data.multifluid import load_multifluid_model
from chemworld.data.multifluid_calibration import MultifluidCalibration, MultifluidFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.coupled_system import CoupledStateView
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.molecular_eos import molecular_density_from_pressure
from chemworld.physchem.molecular_mixture import MolecularIsochoricThermodynamics
from chemworld.physchem.multifluid import MultifluidModel

UNITS = (
    ("pure-amplitude", ("Methane",), ("pure:n",)),
    ("binary-temperature-reducing", ("Methane", "Ethane"), ("gamma_T",)),
    ("binary-density-reducing", ("Methane", "Ethane"), ("gamma_v",)),
    ("binary-departure-multiplier", ("Methane", "Ethane"), ("F",)),
    ("binary-departure-amplitude", ("Methane", "Ethane"), ("departure:n",)),
    ("ternary-two-reducing", ("Methane", "Ethane", "Nitrogen"), ("beta_T", "gamma_v")),
    (
        "quaternary-reducing-departure",
        ("Methane", "Ethane", "Nitrogen", "n-Propane"),
        ("gamma_T", "F"),
    ),
)
COMPOSITIONS = {
    1: (((1.0,),), ((1.0,),)),
    2: (((0.25, 0.75), (0.75, 0.25)), ((0.4, 0.6), (0.6, 0.4))),
    3: (((0.2, 0.3, 0.5), (0.6, 0.3, 0.1)), ((0.3, 0.5, 0.2), (0.5, 0.2, 0.3))),
    4: (
        ((0.1, 0.2, 0.3, 0.4), (0.4, 0.3, 0.2, 0.1)),
        ((0.15, 0.35, 0.25, 0.25), (0.35, 0.15, 0.15, 0.35)),
    ),
}


def parameters(model, selected):
    pair = next(
        (p for p in model.binary_pairs if set(p.component_ids) == set(model.component_ids[:2])),
        None,
    )
    declarations, truths = [], {}
    for name in selected:
        if name == "pure:n":
            truth = model.fluids[0].residual_terms[0].coefficients["n"][0]
            scale = max(abs(truth), 0.1)
            p = MultifluidFitParameter(
                "pure_residual",
                (model.component_ids[0],),
                "n",
                (truth - 0.4 * scale, truth + 0.4 * scale),
                truth + 0.12 * scale,
                term_index=0,
                coefficient_index=0,
            )
        elif name == "departure:n":
            truth = pair.departure_terms[0].n
            scale = max(abs(truth), 0.1)
            p = MultifluidFitParameter(
                "departure",
                pair.component_ids,
                "n",
                (truth - 0.4 * scale, truth + 0.4 * scale),
                truth + 0.12 * scale,
                term_index=0,
            )
        else:
            truth = getattr(pair, name)
            p = (
                MultifluidFitParameter(
                    "binary", pair.component_ids, name, (truth - 0.4, truth + 0.4), truth + 0.12
                )
                if name == "F"
                else MultifluidFitParameter(
                    "binary",
                    pair.component_ids,
                    name,
                    (0.8 * truth, 1.2 * truth),
                    1.08 * truth,
                    coordinate="log",
                )
            )
        declarations.append(p)
        truths[p.parameter_id] = truth
    return tuple(declarations), truths


def table_for(model, reference, unit_id):
    columns = (
        QuantityColumn("temperature", "K", "condition", (0.0, 2000.0), "absolute temperature"),
        QuantityColumn("molar_density", "mol/m^3", "condition", (0.0, 1e6), "molar density"),
        *(
            QuantityColumn("x:" + k, "dimensionless", "condition", (0.0, 1.0), "mole fraction")
            for k in model.component_ids
        ),
        QuantityColumn("pressure_Pa", "Pa", "observation", (-1e10, 1e10), "homogeneous pressure"),
        QuantityColumn(
            "residual_enthalpy_J_mol",
            "J/mol",
            "observation",
            (-1e10, 1e10),
            "residual enthalpy relative to ideal at same T/rho/composition",
        ),
    )
    rows = []
    for split, compositions, states in (
        (
            "train",
            COMPOSITIONS[len(model.fluids)][0],
            tuple(itertools.product((300.0, 400.0), (200.0, 1200.0))),
        ),
        ("validation", COMPOSITIONS[len(model.fluids)][1], ((350.0, 650.0), (450.0, 900.0))),
    ):
        for xs, (temperature, density) in itertools.product(compositions, states):
            xarray = np.asarray(xs)
            a01 = float(reference.get_Ar01(temperature, density, xarray))
            a10 = float(reference.get_Ar10(temperature, density, xarray))
            pressure = density * R * temperature * (1 + a01)
            enthalpy = R * temperature * (a10 + a01)
            values = {
                "temperature": InputValue(temperature, "K", 0.0, "exact synthetic condition"),
                "molar_density": InputValue(density, "mol/m^3", 0.0, "exact synthetic condition"),
                **{
                    "x:" + k: InputValue(x, "dimensionless", 0.0, "exact synthetic condition")
                    for k, x in zip(model.component_ids, xs, strict=True)
                },
                "pressure_Pa": InputValue(
                    pressure,
                    "Pa",
                    max(0.05, abs(pressure) * 1e-6),
                    "fixed synthetic weighting, not measurement precision",
                ),
                "residual_enthalpy_J_mol": InputValue(
                    enthalpy, "J/mol", 0.001, "fixed synthetic weighting, not measurement precision"
                ),
            }
            index = str(len(rows))
            rows.append(ScientificRow(index, "teqp", index, index, split, values))
    source = InputSource(
        DataSourceProvenance(
            "teqp",
            0,
            "teqp independently evaluated multifluid equations "
            "with bundled CoolProp fluid/pair data",
            version=teqp.__version__,
            license="teqp MIT; bundled CoolProp data MIT",
        ),
        "synthetic",
    )
    return ScientificTable(unit_id, "measurements", columns, (source,), tuple(rows))


def comparison(kind, actual, expected, *, atol, rtol, **identity):
    actual, expected = float(actual), float(expected)
    passed = (
        math.isfinite(actual)
        and math.isfinite(expected)
        and abs(actual - expected) <= atol + rtol * abs(expected)
    )
    return {
        "kind": kind,
        **identity,
        "actual": actual,
        "reference": expected,
        "absolute_error": abs(actual - expected),
        "atol": atol,
        "rtol": rtol,
        "passed": passed,
    }


def compact_summary(result, full_record_path):
    """Mechanically retain every unit/check/failure; keep repeated raw traces external."""
    units = []
    for original in result["units"]:
        unit = {key: value for key, value in original.items() if key not in {"table", "fit"}}
        if "table" in original:
            unit["table_definition"] = {
                key: value for key, value in original["table"].items() if key != "rows"
            }
            unit["table_row_count"] = len(original["table"]["rows"])
        if "fit" in original:
            fit = original["fit"]
            unit["fit"] = {
                key: value
                for key, value in fit.items()
                if key not in {"input_model", "fitted_model", "runs", "identifiability"}
            }
            if "identifiability" in fit:
                unit["fit"]["identifiability"] = {
                    key: value
                    for key, value in fit["identifiability"].items()
                    if key != "weighted_jacobian"
                }
            # Failure details survive verbatim, even when a whole unit failed.
            unit["fit"]["forward_failure_records"] = [
                run for run in fit["runs"] if run["failure"] is not None
            ]
            unit["fit"]["forward_record_count"] = len(fit["runs"])
            unit["model_source"] = fit.get("input_model", {}).get("source")
            unit["fitted_model_source"] = fit.get("fitted_model", {}).get("source")
        units.append(unit)
    return {
        **{key: value for key, value in result.items() if key != "units"},
        "schema": "chemworld-multifluid-regression-summary-1",
        "full_record": {
            "path": str(Path(full_record_path).resolve()),
            "schema": result["schema"],
            "storage": "outside Git; original complete execution record retained unchanged",
        },
        "extraction": {
            "method": "mechanical field projection; no unit or numeric comparison selection",
            "retained": (
                "every unit, parameter truth/estimate, numeric comparison, training/holdout "
                "measurement and metric, denominator, resource count, rank/singular values, "
                "exclusion and failure including every failed forward record"
            ),
            "external_only": (
                "repeated successful forward traces, complete source/fitted model payloads, "
                "raw table rows and weighted Jacobian arrays"
            ),
            "fixed_input_reconstruction": (
                "tests/reference/run_multifluid_regression.py UNITS/COMPOSITIONS/parameters/"
                "table_for; teqp 0.23.2 bundled source_input; experiment note retains all rules"
            ),
        },
        "units": units,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", help="optional compact summary for the repository")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing to overwrite an earlier attempt")
    summary_output = Path(args.summary_output) if args.summary_output else None
    if summary_output is not None and (summary_output.exists() or summary_output == output):
        raise SystemExit("summary output must be a distinct new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    records = []
    for index, (unit_id, names, selected) in enumerate(UNITS):
        record = {
            "unit_id": unit_id,
            "components": list(names),
            "selected": list(selected),
            "comparisons": [],
            "errors": [],
            "passed": False,
        }
        records.append(record)
        print(f"unit {index + 1}/{len(UNITS)} start: {unit_id}", flush=True)
        last_progress = time.monotonic()

        def progress(value, unit_index=index, label=unit_id):
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress >= 20:
                print(
                    f"unit {unit_index + 1}/{len(UNITS)} {label}: {value['stage']}, "
                    f"forward_states={value['completed_forward_groups']} "
                    f"budget={value['forward_limit']} elapsed={now - start:.1f}s",
                    flush=True,
                )
                last_progress = now

        try:
            model = load_multifluid_model(
                source_input(names),
                source="teqp 0.23.2 bundled source; explicit CODATA R; regression truth",
            )
            reference = teqp.build_multifluid_model(
                list(names), teqp.get_datapath(), flags={"Rmodel": "CODATA"}
            )
            declarations, truth = parameters(model, selected)
            table = table_for(model, reference, unit_id)
            record.update(table=table.to_dict(), parameter_truths=truth)
            fit = MultifluidCalibration(
                table,
                model,
                declarations,
                fitted_source="synthetic teqp recovery: " + unit_id,
                max_forward_groups=10000,
                progress=progress,
            )
            report = fit.fit(max_nfev=40, diff_step=1e-3, rank_rtol=1e-5, rank_atol=1e-8)
            record["fit"] = report
            if report["status"] != "fitted":
                record["errors"].append("fit status: " + report["status"])
                continue
            if report["identifiability"]["status"] != "full_rank_local":
                record["errors"].append("rank is not full local column rank")
            for key, target in truth.items():
                record["comparisons"].append(
                    comparison(
                        "coefficient",
                        report["estimate"][key],
                        target,
                        atol=2e-5 * max(1, abs(target)),
                        rtol=0.0,
                        parameter_id=key,
                    )
                )
            for reading in report["validation"]["measurements"]:
                record["comparisons"].append(
                    comparison(
                        "heldout",
                        reading["predicted"],
                        reading["measured"],
                        atol=1e-7,
                        rtol=2e-6,
                        row_id=reading["row_id"],
                        quantity=reading["quantity_id"],
                    )
                )
            fitted = MultifluidModel.from_dict(json.loads(json.dumps(report["fitted_model"])))
            record["codec_roundtrip"] = fitted.to_dict() == report["fitted_model"]
            if not record["codec_roundtrip"]:
                record["errors"].append("original model codec roundtrip differs")
            xs = COMPOSITIONS[len(names)][1][0]
            state = fitted.caloric_state(350.0, 650.0, xs)
            density = molecular_density_from_pressure(
                fitted,
                350.0,
                state.pressure_Pa,
                xs,
                initial_density_mol_m3=600.0,
                density_bounds_mol_m3=(300.0, 1000.0),
            )
            record["comparisons"].append(
                comparison(
                    "density_consumer", density.molar_density_mol_m3, 650.0, atol=0.0, rtol=2e-7
                )
            )
            volume = 2.0
            amounts = {
                key: 650.0 * 0.001 * volume * x
                for key, x in zip(fitted.component_ids, xs, strict=True)
            }
            thermal = MolecularIsochoricThermodynamics(fitted.caloric_model(), volume)
            view = CoupledStateView(amounts, 350.0, volume)
            record["comparisons"].append(
                comparison(
                    "inventory_consumer",
                    thermal.internal_energy_j(view),
                    sum(amounts.values()) * state.internal_energy_J_mol,
                    atol=1e-7,
                    rtol=2e-10,
                )
            )
            record["passed"] = not record["errors"] and all(
                c["passed"] for c in record["comparisons"]
            )
        except Exception as exc:
            record["errors"].append(f"{type(exc).__name__}: {exc}")
        finally:
            elapsed = time.monotonic() - start
            completed = index + 1
            print(
                f"completed {completed}/{len(UNITS)}: passed={record['passed']} "
                f"errors={record['errors']} elapsed={elapsed:.1f}s "
                f"throughput={completed / max(elapsed, 1e-9):.4f}units/s "
                f"eta={elapsed / completed * (len(UNITS) - completed):.1f}s",
                flush=True,
            )
    all_checks = [c for r in records for c in r["comparisons"]]
    expected = {"coefficient": 9, "heldout": 52, "density_consumer": 7, "inventory_consumer": 7}
    summary = {
        "units_expected": 7,
        "units_completed": len(records),
        "units_passed": sum(r["passed"] for r in records),
        "train_states_expected": 52,
        "train_observations_expected": 104,
        "heldout_states_expected": 26,
        "heldout_observations_expected": 52,
        "codec_roundtrips": {
            "expected": 7,
            "passed": sum(r.get("codec_roundtrip", False) for r in records),
        },
        "checks": {
            kind: {
                "expected": total,
                "executed": sum(c["kind"] == kind for c in all_checks),
                "passed": sum(c["kind"] == kind and c["passed"] for c in all_checks),
            }
            for kind, total in expected.items()
        },
        "forward_groups": sum(
            r.get("fit", {}).get("resources", {}).get("forward_groups", 0) for r in records
        ),
        "all_failures": [
            {"unit_id": r["unit_id"], "error": e} for r in records for e in r["errors"]
        ]
        + [
            {"unit_id": r["unit_id"], "failed_comparison": c}
            for r in records
            for c in r["comparisons"]
            if not c["passed"]
        ],
    }
    result = {
        "schema": "chemworld-multifluid-regression-reference-1",
        "evidence_mode": "development",
        "boundary": (
            "synthetic same-equation recovery, not measured-data or global parameter qualification"
        ),
        "reference": {"package": "teqp", "version": teqp.__version__, "R_J_mol_K": R},
        "elapsed_s": time.monotonic() - start,
        "summary": summary,
        "units": records,
    }
    rendered = json.dumps(result, indent=2, allow_nan=False)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
        stream.write("\n")
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("x", encoding="utf-8") as stream:
            json.dump(compact_summary(result, output), stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["units_passed"] == 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
