"""Execute the predeclared 23A block; retain every case and every failure."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import thermo
from activity_reference_cases import (
    COMPOSITIONS,
    MODELS,
    TEMPERATURES,
    expected_report,
    high_precision_report,
    model_case,
    upstream,
)

from chemworld.data.activity_calibration import ActivityCalibration, ActivityFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.activity_thermodynamics import (
    activity_infinite_dilution,
    activity_thermodynamic_report,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance


def report_values(r):
    return {
        "GE": r.gibbs_excess_J_mol,
        "HE": r.enthalpy_excess_J_mol,
        "SE": r.entropy_excess_J_mol_K,
        "CpE": r.heat_capacity_excess_J_mol_K,
        "dGE_dT": r.dGE_dT,
        "d2GE_dT2": r.d2GE_dT2,
        "d3GE_dT3": r.d3GE_dT3,
        "dGE_dxs": r.dGE_dxs,
        "d2GE_dxixjs": r.d2GE_dxixjs,
        "d2GE_dTdxs": r.d2GE_dTdxs,
        "gamma": list(r.activity_coefficients.values()),
        "ln_gamma": list(r.log_activity_coefficients.values()),
        "dln_gamma_dT": list(r.dln_gamma_dT.values()),
        "dln_gamma_dns": r.dln_gamma_dns,
        "partial_GE": list(r.partial_molar_gibbs_excess_J_mol.values()),
        "partial_HE": list(r.partial_molar_enthalpy_excess_J_mol.values()),
        "partial_SE": list(r.partial_molar_entropy_excess_J_mol_K.values()),
    }


def flatten(value, prefix=""):
    if isinstance(value, dict):
        return {
            k: v
            for name, item in value.items()
            for k, v in flatten(item, prefix + name + "/").items()
        }
    if isinstance(value, (list, tuple)):
        return {
            k: v
            for i, item in enumerate(value)
            for k, v in flatten(item, prefix + str(i) + "/").items()
        }
    return {prefix.rstrip("/"): float(value)}


def compare(record, actual, expected, *, absolute=False):
    left, right = flatten(actual), flatten(expected)
    if left.keys() != right.keys():
        raise ValueError("reference and native response coordinates differ")
    for name, value in left.items():
        target = right[name]
        error = abs(value - target)
        tolerance = 2e-8 + (0 if absolute else 2e-9 * abs(target))
        record["comparisons"] += 1
        record["max_absolute_error"] = max(record["max_absolute_error"], error)
        if not math.isfinite(value) or error > tolerance:
            record["failures"].append(
                {
                    "quantity": name,
                    "actual": value,
                    "expected": target,
                    "absolute_error": error,
                    "tolerance": tolerance,
                }
            )


def new_record(kind, model, n, **kwargs):
    return {
        "kind": kind,
        "model": model,
        "components": n,
        **kwargs,
        "comparisons": 0,
        "max_absolute_error": 0.0,
        "failures": [],
    }


def fit_inputs(model):
    truth = model_case(model, 2)
    prefix = "lambda" if model == "wilson" else "tau"
    names = (f"{prefix}_b:A|B", f"{prefix}_b:B|A")
    initials = {
        name: truth.parameters[name] + delta
        for name, delta in zip(names, (45.0, -35.0), strict=True)
    }
    initial = replace(
        truth, model_id=truth.model_id + "-seed", parameters={**truth.parameters, **initials}
    )
    parameters = tuple(
        ActivityFitParameter(name, (-500.0, 500.0), initials[name]) for name in names
    )
    source = InputSource(
        DataSourceProvenance(
            "thermo-activity-reference",
            0,
            "Fixed thermo 0.6.0 synthetic same-equation activities",
            version="0.6.0",
            license="MIT upstream software; authored synthetic points",
        ),
        "synthetic",
    )
    columns = (
        QuantityColumn("temperature", "K", "condition", (200, 500), "absolute temperature"),
        *(
            QuantityColumn("x:" + key, "dimensionless", "condition", (0, 1), "mole fraction")
            for key in "AB"
        ),
        *(
            QuantityColumn(
                "ln_gamma:" + key,
                "dimensionless",
                "observation",
                (-100, 100),
                "pure component liquid reference",
            )
            for key in "AB"
        ),
    )
    rows = []
    for split, Ts, xs in (
        ("train", (280.0, 340.0), (0.15, 0.35, 0.65, 0.85)),
        ("validation", (310.0, 370.0), (0.25, 0.75)),
    ):
        for T in Ts:
            for x in xs:
                index = len(rows)
                logs = upstream(truth, T, (x, 1 - x)).lngammas()
                values = {
                    "temperature": InputValue(T, "K", 0, "exact synthetic input"),
                    "x:A": InputValue(x, "dimensionless", 0, "exact synthetic input"),
                    "x:B": InputValue(1 - x, "dimensionless", 0, "exact synthetic input"),
                    **{
                        "ln_gamma:" + key: InputValue(
                            value, "dimensionless", 0.01, "declared synthetic sigma"
                        )
                        for key, value in zip("AB", logs, strict=True)
                    },
                }
                rows.append(
                    ScientificRow(
                        str(index),
                        "thermo-activity-reference",
                        f"g{index}",
                        f"s{index}",
                        split,
                        values,
                    )
                )
    return (
        truth,
        ScientificTable("fixed-23-" + model, "measurements", columns, (source,), tuple(rows)),
        initial,
        parameters,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", choices=("thermo", "high_precision"), default="thermo")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = {
        "block": "native-conv23-unified-activity",
        "evidence_mode": "development",
        "reference": {
            "thermo": thermo.__version__,
            "mpmath_decimal_digits": 80,
            "state_reference": args.reference,
        },
        "expected_state_units": 90,
        "expected_boundary_units": 25,
        "expected_dilution_units": 25,
        "expected_scientific_comparisons": 4390,
        "expected_fit_units": 3,
        "expected_fit_criteria": 15,
        "units": [],
        "fits": [],
    }
    completed = 0
    for model in MODELS:
        for n in (2, 3):
            spec = model_case(model, n)
            for T in TEMPERATURES:
                for xs in COMPOSITIONS[n]:
                    rec = new_record("state", model, n, temperature_K=T, composition=list(xs))
                    result["units"].append(rec)
                    try:
                        native = activity_thermodynamic_report(
                            spec, dict(zip(spec.component_ids, xs, strict=True)), temperature_K=T
                        )
                        reference = (
                            high_precision_report
                            if args.reference == "high_precision"
                            else expected_report
                        )
                        compare(rec, report_values(native), reference(spec, T, xs))
                        mat = native.dln_gamma_dns
                        closures = {
                            "partial_GE": sum(
                                xs[i] * native.partial_molar_gibbs_excess_J_mol[k]
                                for i, k in enumerate(spec.component_ids)
                            )
                            - native.gibbs_excess_J_mol,
                            "partial_HE": sum(
                                xs[i] * native.partial_molar_enthalpy_excess_J_mol[k]
                                for i, k in enumerate(spec.component_ids)
                            )
                            - native.enthalpy_excess_J_mol,
                            "partial_SE": sum(
                                xs[i] * native.partial_molar_entropy_excess_J_mol_K[k]
                                for i, k in enumerate(spec.component_ids)
                            )
                            - native.entropy_excess_J_mol_K,
                            "gibbs_duhem": max(
                                abs(sum(xs[i] * mat[i][j] for i in range(n))) for j in range(n)
                            ),
                            "symmetry": max(
                                abs(mat[i][j] - mat[j][i]) for i in range(n) for j in range(n)
                            ),
                            "thermal_gibbs_duhem": sum(
                                xs[i] * native.dln_gamma_dT[k]
                                for i, k in enumerate(spec.component_ids)
                            )
                            + native.enthalpy_excess_J_mol / (8.31446261815324 * T * T),
                        }
                        compare(rec, closures, dict.fromkeys(closures, 0.0), absolute=True)
                    except Exception as exc:
                        rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
                    completed += 1
                    if completed % 10 == 0:
                        elapsed = time.monotonic() - start
                        rate = completed / max(elapsed, 1e-9)
                        (args.output / "partial.json").write_text(
                            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
                        )
                        print(
                            f"Activity states {completed}/90; {rate:.2f}/s; "
                            f"ETA {(90 - completed) * elapsed / completed:.1f}s",
                            flush=True,
                        )
            for i, key in enumerate(spec.component_ids):
                xs = [float(j == i) for j in range(n)]
                rec = new_record("pure_endpoint", model, n, pure_component=key)
                result["units"].append(rec)
                try:
                    r = activity_thermodynamic_report(
                        spec, dict(zip(spec.component_ids, xs, strict=True)), temperature_K=320.0
                    )
                    compare(
                        rec,
                        {
                            "GE": r.gibbs_excess_J_mol,
                            "HE": r.enthalpy_excess_J_mol,
                            "SE": r.entropy_excess_J_mol_K,
                            "CpE": r.heat_capacity_excess_J_mol_K,
                            "pure_log_gamma": r.log_activity_coefficients[key],
                        },
                        dict.fromkeys(("GE", "HE", "SE", "CpE", "pure_log_gamma"), 0.0),
                        absolute=True,
                    )
                    compare(
                        rec,
                        {
                            "finite_gradient": float(all(math.isfinite(v) for v in r.dGE_dxs)),
                            "finite_hessian": float(
                                all(math.isfinite(v) for row in r.d2GE_dxixjs for v in row)
                            ),
                            "finite_response": float(
                                all(math.isfinite(v) for v in flatten(report_values(r)).values())
                            ),
                        },
                        dict.fromkeys(
                            ("finite_gradient", "finite_hessian", "finite_response"), 1.0
                        ),
                    )
                except Exception as exc:
                    rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
                rec = new_record("infinite_dilution", model, n, solute=key)
                result["units"].append(rec)
                try:
                    r = activity_infinite_dilution(
                        spec,
                        dict.fromkeys(spec.component_ids, 1.0),
                        temperature_K=320.0,
                        solute_id=key,
                    )
                    xs = [0.0 if j == i else 1 / (n - 1) for j in range(n)]
                    ref = high_precision_report(spec, 320.0, xs)
                    compare(
                        rec,
                        {
                            "gamma": r.activity_coefficients[key],
                            "ln_gamma": r.log_activity_coefficients[key],
                        },
                        {"gamma": ref["gamma"][i], "ln_gamma": ref["ln_gamma"][i]},
                    )
                except Exception as exc:
                    rec["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
    for model in ("nrtl", "wilson", "uniquac"):
        record = {"model": model, "criteria": 0, "failures": []}
        result["fits"].append(record)
        try:
            truth, table, initial, parameters = fit_inputs(model)
            fit = ActivityCalibration(table, initial, parameters, fitted_model_id="fitted-" + model)
            fitted = fit.fit(max_nfev=100)
            (args.output / (model + "-fit.json")).write_text(
                json.dumps(fitted, indent=2, allow_nan=False), encoding="utf-8"
            )
            record.update(
                status=fitted["status"],
                estimate=fitted.get("estimate"),
                resources=fitted["resources"],
            )
            if fitted["status"] != "fitted":
                raise ValueError(f"fit not completed: {fitted['failures']}")
            record.update(
                training_weighted_rmse=fitted["training"]["weighted_rmse"],
                validation_weighted_rmse=fitted["validation"]["weighted_rmse"],
                rank=fitted["identifiability"]["rank"],
            )
            checks = {
                "parameters": all(
                    math.isclose(
                        fitted["estimate"][p.name],
                        truth.parameters[p.name],
                        rel_tol=1e-6,
                        abs_tol=1e-5,
                    )
                    for p in parameters
                ),
                "training": fitted["training"]["weighted_rmse"] < 1e-6,
                "validation": fitted["validation"]["weighted_rmse"] < 1e-6,
                "rank": fitted["identifiability"]["rank"] == 2,
                "holdout_isolation": all(r["split"] == "train" for r in fitted["jacobian_runs"])
                and all(r["split"] == "train" for r in fitted["runs"] if r["stage"] != "heldout"),
            }
            record["criteria"] = len(checks)
            record["failures"].extend(
                {"criterion": key} for key, passed in checks.items() if not passed
            )
        except Exception as exc:
            record["failures"].append({"exception": f"{type(exc).__name__}: {exc}"})
        print(
            f"Activity fits {len(result['fits'])}/3 {model}: {len(record['failures'])} failures",
            flush=True,
        )
    result["scientific_comparisons"] = sum(r["comparisons"] for r in result["units"])
    result["fit_criteria"] = sum(r["criteria"] for r in result["fits"])
    result["failures"] = [
        {"unit_index": i, "kind": r.get("kind", "fit"), "model": r["model"], **failure}
        for i, r in enumerate([*result["units"], *result["fits"]])
        for failure in r["failures"]
    ]
    result["elapsed_seconds"] = time.monotonic() - start
    result["passed"] = (
        not result["failures"]
        and result["scientific_comparisons"] == 4390
        and result["fit_criteria"] == 15
    )
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("scientific_comparisons", "fit_criteria", "passed", "elapsed_seconds")
            }
        ),
        flush=True,
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
