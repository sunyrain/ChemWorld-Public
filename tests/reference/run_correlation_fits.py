"""Fixed 28 full-coefficient fits from NATIVE_CONV22G2_EXPERIMENT.md."""

import argparse
import json
import time
from math import cos, isfinite
from pathlib import Path

import chemicals
import mpmath as mp
import numpy as np
from chemicals import dippr
from run_dippr_derivatives import mp_expression
from scipy.optimize import least_squares
from test_native_dippr_reference import CASES, correlation

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

TRAIN = tuple(range(230, 451, 20))
HELDOUT = tuple(range(240, 441, 40))


def inputs(number, parameter_set):
    card = correlation(number, parameter_set)
    parameters = tuple(CorrelationFitParameter(name, tuple(sorted((value*.7, value*1.3))),
        value*(1.08 if i % 2 == 0 else .92)) for i, (name, value) in
        enumerate(card.coefficients.items()))
    source = InputSource(DataSourceProvenance("synthetic:dippr-fit", 0,
        "chemicals 1.5.2 documented DIPPR equation; authored fixed perturbation",
        version="1", license="MIT (synthetic observations)"), "synthetic")
    columns = (QuantityColumn("temperature", "K", "condition", (220, 450), "absolute T"),
               QuantityColumn("value", card.output_unit, "observation", (-1e12, 1e12),
                              "synthetic value in source output unit"))
    rows = []
    for split, temperatures in (("train", TRAIN), ("validation", HELDOUT)):
        for i, T in enumerate(temperatures):
            value = getattr(dippr, "EQ"+number)(T, **card.coefficients)
            sigma = .01*max(abs(value), 1e-12)
            row_id = f"{split}-{i}"
            rows.append(ScientificRow(row_id, source.provenance.source_id, row_id, row_id, split,
                {"temperature": InputValue(float(T), "K", 0., "exact synthetic input"),
                 "value": InputValue(value + (.15*sigma*cos(i) if split == "train" else 0),
                    card.output_unit, sigma, "declared 1 percent synthetic standard uncertainty")}))
    table = ScientificTable(f"dippr-{number}-{parameter_set}", "measurements", columns,
                              (source,), tuple(rows))
    return table, card, parameters


def reference_fit(table, card, parameters):
    names = tuple(p.name for p in parameters)
    lower, upper = np.asarray([p.bounds for p in parameters]).T
    initial = np.asarray([p.initial for p in parameters])
    rows = table.rows_for("train")

    def coefficients(x):
        return dict(zip(names, lower+x*(upper-lower), strict=True))

    def residual(x):
        p = coefficients(x)
        function = getattr(dippr, "EQ"+card.equation_id.removeprefix("dippr"))
        return np.asarray([(function(row.values["temperature"].value, **p)
                            - row.values["value"].value)/row.values["value"].uncertainty
                           for row in rows])

    result = least_squares(residual, (initial-lower)/(upper-lower), bounds=(0, 1),
        jac="3-point", diff_step=1e-3, max_nfev=1500, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    return {"success": bool(result.success), "message": result.message,
            "estimate": coefficients(result.x), "sum_squared_weighted_residuals":
                float(result.fun@result.fun), "nfev": result.nfev, "njev": result.njev}


def qualify(number, parameter_set, progress):
    table, card, parameters = inputs(number, parameter_set)
    row = {"id": f"{number}/{parameter_set}", "table": table.to_dict(),
           "initial_correlation": card.to_dict(), "checks": [], "failures": []}
    fit = CorrelationCalibration(table, card, parameters,
        fitted_correlation_id=f"fitted-{number}-{parameter_set}", progress=progress)
    try:
        native = fit.fit(max_nfev=1500)
    except Exception as exc:
        row.update(passed=False, failure=f"{type(exc).__name__}: {exc}",
                   native_partial={"runs": fit.runs, "jacobian_runs": fit.jacobian_runs})
        return row
    row["native"] = native
    try:
        ref = reference_fit(table, card, parameters)
    except Exception as exc:
        row.update(passed=False, failure=f"Reference {type(exc).__name__}: {exc}")
        return row
    row["reference_fit"] = ref

    def check(name, passed, **details):
        record = {"check": name, "passed": bool(passed), **details}
        row["checks"].append(record)
        if not passed:
            row["failures"].append(record)

    check("native_converged", native["status"] == "fitted", status=native["status"])
    check("reference_converged", ref["success"], message=ref["message"])
    if native["status"] != "fitted":
        row["passed"] = False
        return row
    for split, key in (("train", "training"), ("validation", "validation")):
        rmse = native[key]["weighted_rmse"]
        check(f"{split}_weighted_rmse", rmse <= .75, actual=rmse, maximum=.75)
    actual_ss = native["training"]["sum_squared_weighted_residuals"]
    expected_ss = ref["sum_squared_weighted_residuals"]
    check("training_objective", abs(actual_ss-expected_ss) <= 2e-3*abs(expected_ss)+1e-5,
          actual=actual_ss, expected=expected_ss)
    fitted = PropertyCorrelation.from_dict(native["fitted_correlation"])
    ref_function = getattr(dippr, "EQ"+number)
    for observation in table.rows:
        T = observation.values["temperature"].value
        sigma = observation.values["value"].uncertainty
        actual = evaluate_correlation(fitted, temperature_K=T, validity_policy="raise").value
        expected = ref_function(T, **ref["estimate"])
        check("prediction/"+observation.row_id, abs(actual-expected) <= .1*sigma,
              actual=actual, expected=expected, maximum_absolute_error=.1*sigma)
    mp.mp.dps = 80
    final_coeffs = {name: mp.mpf(str(value)) for name, value in native["estimate"].items()}
    actual_jac = np.asarray(native["identifiability"]["weighted_jacobian"])
    reference_jac = []
    for i, observation in enumerate(table.rows_for("train")):
        T = mp.mpf(observation.values["temperature"].value)
        sigma = observation.values["value"].uncertainty
        reference_row = []
        for j, parameter in enumerate(parameters):
            name = parameter.name
            derivative = mp.diff(lambda value, name=name, T=T:
                mp_expression(number, T, {**final_coeffs, name: value}), final_coeffs[name])
            expected = float(derivative)*(parameter.bounds[1]-parameter.bounds[0])/sigma
            actual = float(actual_jac[i, j])
            reference_row.append(expected)
            check(f"weighted_jacobian/{i}/{name}",
                  abs(actual-expected) <= 2e-8*abs(expected)+1e-8,
                  actual=actual, expected=expected)
        reference_jac.append(reference_row)
    singular_values = np.linalg.svd(reference_jac, compute_uv=False)
    threshold = max(1e-8, 1e-5*singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    check("local_rank", rank == native["identifiability"]["rank"],
          actual=native["identifiability"]["rank"], expected=rank,
          reference_singular_values=singular_values.tolist())
    check("train_only_objective", all(r["split"] == "train" for r in native["runs"]
          if r["stage"] == "fit") and all(r["split"] == "train" for r in native["jacobian_runs"]))
    row["passed"] = not row["failures"]
    return row


def run(output):
    if output.exists():
        raise ValueError("new attempt directory required; no previous results are overwritten")
    output.mkdir(parents=True)
    assert chemicals.__version__ == "1.5.2"
    started = time.perf_counter()
    last_progress = started
    cases = []

    def progress(event):
        nonlocal last_progress
        now = time.perf_counter()
        if now-last_progress >= 30:
            print(json.dumps({"completed_fits": len(cases), "total": 28, **event}), flush=True)
            last_progress = now

    for number in CASES:
        for parameter_set in (0, 1):
            case_start = time.perf_counter()
            try:
                case = qualify(number, parameter_set, progress)
            except Exception as exc:
                case = {"id": f"{number}/{parameter_set}", "passed": False,
                        "failure": f"{type(exc).__name__}: {exc}"}
            case["seconds"] = time.perf_counter()-case_start
            cases.append(case)
            # Terminal per-case output remains available even if the next case fails.
            (output/f"fit-{number}-{parameter_set}.json").write_text(
                json.dumps(case, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            elapsed = time.perf_counter()-started
            print(json.dumps({"completed": len(cases), "total": 28, "id": case["id"],
                "passed": case["passed"], "fits_per_second": len(cases)/elapsed,
                "eta_seconds": elapsed/len(cases)*(28-len(cases))}), flush=True)
    report = {"schema": "chemworld-conv22g2-fits-1", "mode": "development",
        "reference_versions": {"chemicals": chemicals.__version__, "mpmath": mp.__version__},
        "expected": 28, "completed": len(cases), "passed_count": sum(c["passed"] for c in cases),
        "cases": cases, "failures": [c["id"] for c in cases if not c["passed"]],
        "seconds": time.perf_counter()-started,
        "boundary": "synthetic full-coefficient regression; local rank is not global uniqueness"}
    report["passed"] = report["passed_count"] == 28 and isfinite(report["seconds"])
    (output/"summary.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                                       encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("completed", "passed_count", "failures", "seconds")}), flush=True)
    return report["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(0 if run(parser.parse_args().output) else 1)
