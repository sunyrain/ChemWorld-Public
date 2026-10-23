"""Fixed 12 full-public-column pressure/viscosity fits from the G2 fourth block."""

import argparse
import json
import time
from math import cos
from pathlib import Path

import chemicals
import mpmath as mp
import numpy as np
from chemicals import vapor_pressure, viscosity
from run_pressure_methods_block import CASES, JACOBIANS
from scipy.optimize import least_squares

from chemworld.data.correlation_calibration import CorrelationCalibration, CorrelationFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_equations import evaluate_correlation
from chemworld.physchem.specs import PropertyCorrelation

FAMILIES = (
    *(
        (equation, method, parameters, temperatures, JACOBIANS[method])
        for equation, method, parameters, temperatures in CASES
        if method in JACOBIANS
    ),
    (
        "viscosity_mu_yaws",
        "mu_Yaws",
        {"A": -6.0, "B": 800.0, "C": -0.001, "D": 1e-6},
        (300.0, 400.0, 500.0),
        tuple("ABCD"),
    ),
)


def reference_value(method, temperature, parameters):
    module = viscosity if method == "mu_Yaws" else vapor_pressure
    return getattr(module, method)(temperature, **parameters)


def mp_expression(method, t, p):
    """Independent expressions of the public equations, without native math imports."""
    ln10 = mp.log(10)
    if method == "Antoine":
        return mp.exp(mp.log(p["base"]) * (p["A"] - p["B"] / (t + p["C"])))
    if method == "TRC_Antoine_extended":
        x = (t - p["to"] - mp.mpf("273.15")) / p["Tc"]
        exponent = p["A"] - p["B"] / (t + p["C"])
        if x > 0:
            exponent += mp.mpf(".43429") * x ** p["n"] + p["E"] * x**8 + p["F"] * x**12
        return mp.exp(ln10 * exponent)
    if method in {"Wagner", "Wagner_original"}:
        tr = t / p["Tc"]
        powers = (
            (1, mp.mpf("1.5"), 3, 6)
            if method == "Wagner_original"
            else (1, mp.mpf("1.5"), mp.mpf("2.5"), 5)
        )
        exponent = (
            sum(p[name] * (1 - tr) ** power for name, power in zip("abcd", powers, strict=True))
            / tr
        )
        return p["Pc"] * mp.exp(exponent)
    if method == "Yaws_Psat":
        return mp.exp(
            ln10 * (p["A"] + p["B"] / t + p["D"] * t + p["E"] * t * t) + p["C"] * mp.log(t)
        )
    if method == "mu_Yaws":
        return mp.exp(ln10 * (p["A"] + p["B"] / t + p["C"] * t + p["D"] * t * t))
    raise ValueError(method)


def inputs(family, variant):
    equation, method, coefficients, temperatures, names = family
    truth = {
        name: value * (0.9 if variant and name in names else 1.0)
        for name, value in coefficients.items()
    }
    low, high = min(temperatures), max(temperatures)
    train = tuple(float(v) for v in np.linspace(low, high, 12))
    heldout = tuple((train[i] + train[i + 1]) / 2 for i in (0, 2, 4, 6, 8, 10))
    unit, prop = ("Pa*s", "liquid_viscosity") if method == "mu_Yaws" else ("Pa", "vapor_pressure")
    card = PropertyCorrelation(
        f"{equation}-{variant}",
        equation,
        truth,
        {"temperature": "K"},
        unit,
        prop,
        {"temperature": (low, high)},
        "Synthetic public-column fitting qualification",
    )
    parameters = tuple(
        CorrelationFitParameter(
            name,
            tuple(sorted((truth[name] * 0.7, truth[name] * 1.3))),
            truth[name] * (1.08 if i % 2 == 0 else 0.92),
        )
        for i, name in enumerate(names)
    )
    source = InputSource(
        DataSourceProvenance(
            "synthetic:remaining-fit",
            0,
            "chemicals 1.5.2 public equations with fixed synthetic perturbation",
            version="1",
            license="MIT (synthetic measurements)",
        ),
        "synthetic",
    )
    columns = (
        QuantityColumn("temperature", "K", "condition", (low, high), "absolute temperature"),
        QuantityColumn("value", unit, "observation", (0, 1e200), "synthetic observation"),
    )
    rows = []
    for split, points in (("train", train), ("validation", heldout)):
        for i, t in enumerate(points):
            value = reference_value(method, t, truth)
            sigma = 0.01 * max(abs(value), 1e-12)
            row_id = f"{split}-{i}"
            rows.append(
                ScientificRow(
                    row_id,
                    source.provenance.source_id,
                    row_id,
                    row_id,
                    split,
                    {
                        "temperature": InputValue(t, "K", 0.0, "exact synthetic temperature"),
                        "value": InputValue(
                            value + (0.15 * sigma * cos(i) if split == "train" else 0),
                            unit,
                            sigma,
                            "declared 1 percent synthetic standard uncertainty",
                        ),
                    },
                )
            )
    return (
        ScientificTable(card.correlation_id, "measurements", columns, (source,), tuple(rows)),
        card,
        parameters,
    )


def reference_fit(table, card, parameters, method, progress):
    names = tuple(p.name for p in parameters)
    lower, upper = np.asarray([p.bounds for p in parameters]).T
    initial = np.asarray([p.initial for p in parameters])
    rows = table.rows_for("train")
    evaluations = 0

    def coefficients(x):
        return {**card.coefficients, **dict(zip(names, lower + x * (upper - lower), strict=True))}

    def residual(x):
        nonlocal evaluations
        p = coefficients(x)
        evaluations += 1
        progress({"stage": "reference_objective", "evaluations": evaluations})
        return np.asarray(
            [
                (
                    reference_value(method, row.values["temperature"].value, p)
                    - row.values["value"].value
                )
                / row.values["value"].uncertainty
                for row in rows
            ]
        )

    result = least_squares(
        residual,
        (initial - lower) / (upper - lower),
        bounds=(0, 1),
        jac="3-point",
        diff_step=1e-3,
        max_nfev=1500,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    return {
        "success": bool(result.success),
        "message": result.message,
        "estimate": coefficients(result.x),
        "sum_squared_weighted_residuals": float(result.fun @ result.fun),
        "nfev": result.nfev,
        "njev": result.njev,
        "objective_evaluations": evaluations,
    }


def qualify(family, variant, progress):
    table, card, parameters = inputs(family, variant)
    method = family[1]
    row = {
        "id": f"{method}/{variant}",
        "table": table.to_dict(),
        "initial_correlation": card.to_dict(),
        "checks": [],
        "failures": [],
    }

    def check(name, passed, **details):
        item = {"check": name, "passed": bool(passed), **details}
        row["checks"].append(item)
        if not passed:
            row["failures"].append(item)

    fit = CorrelationCalibration(
        table, card, parameters, fitted_correlation_id=f"fit-{method}-{variant}", progress=progress
    )
    try:
        native = fit.fit(max_nfev=1500)
        row["native"] = native
        check("native_converged", native["status"] == "fitted", status=native["status"])
    except Exception as exc:
        native = None
        row["native_partial"] = {"runs": fit.runs, "jacobian_runs": fit.jacobian_runs}
        check("native_converged", False, failure=f"{type(exc).__name__}: {exc}")
    try:
        ref = reference_fit(table, card, parameters, method, progress)
        row["reference_fit"] = ref
        check("reference_converged", ref["success"], message=ref["message"])
    except Exception as exc:
        ref = None
        check("reference_converged", False, failure=f"{type(exc).__name__}: {exc}")
    if native is None or native["status"] != "fitted" or ref is None:
        return {**row, "passed": False}
    for key in ("training", "validation"):
        check(
            key + "_weighted_rmse",
            native[key]["weighted_rmse"] <= 0.75,
            actual=native[key]["weighted_rmse"],
            maximum=0.75,
        )
    actual, expected = (
        native["training"]["sum_squared_weighted_residuals"],
        ref["sum_squared_weighted_residuals"],
    )
    check(
        "training_objective",
        abs(actual - expected) <= 0.002 * abs(expected) + 1e-5,
        actual=actual,
        expected=expected,
    )
    fitted = PropertyCorrelation.from_dict(native["fitted_correlation"])
    for observation in table.rows:
        t = observation.values["temperature"].value
        sigma = observation.values["value"].uncertainty
        a = evaluate_correlation(fitted, temperature_K=t, validity_policy="raise").value
        e = reference_value(method, t, ref["estimate"])
        check(
            "prediction/" + observation.row_id,
            abs(a - e) <= 0.1 * sigma,
            actual=a,
            expected=e,
            maximum_absolute_error=0.1 * sigma,
        )
    mp.mp.dps = 80
    p = {name: mp.mpf(str(value)) for name, value in fitted.coefficients.items()}
    weighted = native["identifiability"]["weighted_jacobian"]
    reference_jac = []
    for i, observation in enumerate(table.rows_for("train")):
        t = mp.mpf(str(observation.values["temperature"].value))
        sigma = observation.values["value"].uncertainty
        ref_row = []
        for j, parameter in enumerate(parameters):
            name = parameter.name
            derivative = mp.diff(
                lambda v, name=name, t=t: mp_expression(method, t, {**p, name: v}), p[name]
            )
            e = float(derivative) * (parameter.bounds[1] - parameter.bounds[0]) / sigma
            a = weighted[i][j]
            ref_row.append(e)
            check(
                f"weighted_jacobian/{i}/{name}",
                abs(a - e) <= 2e-8 * abs(e) + 1e-8,
                actual=a,
                expected=e,
            )
        reference_jac.append(ref_row)
    singular = np.linalg.svd(reference_jac, compute_uv=False)
    rank = int(np.count_nonzero(singular > max(1e-8, 1e-5 * singular[0])))
    check(
        "local_rank",
        rank == native["identifiability"]["rank"],
        actual=native["identifiability"]["rank"],
        expected=rank,
        reference_singular_values=singular.tolist(),
    )
    check(
        "train_only_objective",
        all(r["split"] == "train" for r in native["runs"] if r["stage"] == "fit")
        and all(r["split"] == "train" for r in native["jacobian_runs"]),
    )
    return {**row, "passed": not row["failures"]}


def run(output):
    if output.exists():
        raise ValueError("new attempt directory required; previous results are never overwritten")
    output.mkdir(parents=True)
    assert chemicals.__version__ == "1.5.2" and len(FAMILIES) == 6
    started = time.perf_counter()
    last_progress = started
    results = []

    def progress(event):
        nonlocal last_progress
        now = time.perf_counter()
        if now - last_progress >= 30:
            print(json.dumps({"completed": len(results), "total": 12, **event}), flush=True)
            last_progress = now

    for family in FAMILIES:
        for variant in (0, 1):
            start = time.perf_counter()
            try:
                row = qualify(family, variant, progress)
            except Exception as exc:
                row = {
                    "id": f"{family[1]}/{variant}",
                    "passed": False,
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            row["seconds"] = time.perf_counter() - start
            results.append(row)
            (output / f"{family[1]}-{variant}.json").write_text(
                json.dumps(row, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "total": 12,
                        "id": row["id"],
                        "passed": row["passed"],
                        "fits_per_second": len(results) / elapsed,
                        "eta_seconds": elapsed / len(results) * (12 - len(results)),
                    }
                ),
                flush=True,
            )
    report = {
        "schema": "chemworld-remaining-correlation-fits-1",
        "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__, "mpmath": mp.__version__},
        "expected": 12,
        "completed": len(results),
        "passed_count": sum(r["passed"] for r in results),
        "cases": results,
        "failures": [r["id"] for r in results if not r["passed"]],
        "seconds": time.perf_counter() - started,
        "boundary": "All public fitting columns; synthetic regression, not material accuracy.",
    }
    report["passed"] = report["passed_count"] == 12
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: report[key] for key in ("completed", "passed_count", "failures", "seconds")}
        ),
        flush=True,
    )
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
