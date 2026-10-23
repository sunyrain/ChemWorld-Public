"""One fitting framework, explicit parameters, heldout isolation and failures."""

import json
from dataclasses import replace

import pytest

from chemworld.data.activity_calibration import ActivityCalibration, ActivityFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.equilibrium import ActivityModelSpec, activity_coefficients


def inputs():
    source = InputSource(
        DataSourceProvenance(
            "margules-synthetic",
            0,
            "Authored standard three-suffix analytic log activity",
            version="1",
            license="MIT",
        ),
        "synthetic",
    )
    columns = (
        QuantityColumn("temperature", "K", "condition", (200, 500), "absolute T"),
        *(
            QuantityColumn("x:" + i, "dimensionless", "condition", (0, 1), "mole fraction")
            for i in "AB"
        ),
        *(
            QuantityColumn(
                "ln_gamma:" + i, "dimensionless", "observation", (-10, 10), "pure liquid reference"
            )
            for i in "AB"
        ),
    )
    rows = []
    for index, x in enumerate((0.1, 0.3, 0.6, 0.9, 0.2, 0.75)):
        y = 1 - x
        values = {
            "temperature": InputValue(300.0, "K", 0, "exact synthetic input"),
            "x:A": InputValue(x, "dimensionless", 0, "exact synthetic input"),
            "x:B": InputValue(y, "dimensionless", 0, "exact synthetic input"),
            "ln_gamma:A": InputValue(
                y * y * (1.2 + 2 * (0.8 - 1.2) * x),
                "dimensionless",
                0.01,
                "declared synthetic uncertainty",
            ),
            "ln_gamma:B": InputValue(
                x * x * (0.8 + 2 * (1.2 - 0.8) * y),
                "dimensionless",
                0.01,
                "declared synthetic uncertainty",
            ),
        }
        rows.append(
            ScientificRow(
                str(index),
                "margules-synthetic",
                f"g{index}",
                f"s{index}",
                "train" if index < 4 else "validation",
                values,
            )
        )
    model = ActivityModelSpec("initial", ("A", "B"), "margules", {"A:A|B": 0.7, "A:B|A": 1.1})
    params = (
        ActivityFitParameter("A:A|B", (0.1, 2), 0.7),
        ActivityFitParameter("A:B|A", (0.1, 2), 1.1, "log"),
    )
    return (
        ScientificTable("margules-fixed", "measurements", columns, (source,), tuple(rows)),
        model,
        params,
    )


def test_shared_fit_exports_consumable_model_and_never_uses_heldout_for_training():
    table, model, params = inputs()
    fit = ActivityCalibration(table, model, params, fitted_model_id="fitted")
    result = fit.fit(max_nfev=100)
    assert result["status"] == "fitted"
    assert result["estimate"] == pytest.approx({"A:A|B": 1.2, "A:B|A": 0.8}, abs=1e-9)
    assert result["identifiability"]["status"] == "full_rank_local"
    assert result["training"]["n_valid"] == 8 and result["validation"]["n_valid"] == 4
    assert result["validation"]["weighted_rmse"] < 1e-8
    assert all(r["split"] == "train" for r in result["jacobian_runs"])
    assert all(r["split"] == "train" for r in result["runs"] if r["stage"] != "heldout")
    exported = result["fitted_model"]
    restored = ActivityModelSpec(
        exported["model_id"],
        tuple(exported["component_ids"]),
        exported["model"],
        exported["parameters"],
    )
    assert activity_coefficients(restored, {"A": 0.2, "B": 0.8}, temperature_K=300)["A"] > 1
    assert model.parameters == {"A:A|B": 0.7, "A:B|A": 1.1}
    json.dumps(result, allow_nan=False)
    altered = replace(
        table,
        rows=tuple(
            replace(
                row,
                values={**row.values, "ln_gamma:A": replace(row.values["ln_gamma:A"], value=5.0)},
            )
            if row.split == "validation"
            else row
            for row in table.rows
        ),
    )
    other = ActivityCalibration(altered, model, params, fitted_model_id="other").fit(max_nfev=100)
    assert other["estimate"] == result["estimate"]
    assert other["training"] == result["training"]
    assert other["validation"]["weighted_rmse"] > 100


@pytest.mark.parametrize("failure", ("budget", "iterations"))
def test_failure_retains_attempts_does_not_export_and_never_reads_holdout(failure):
    table, model, params = inputs()
    result = ActivityCalibration(
        table,
        model,
        params,
        fitted_model_id="failed",
        max_forward_groups=1 if failure == "budget" else 10000,
    ).fit(max_nfev=1 if failure == "iterations" else 100)
    assert result["status"] != "fitted" and result["failures"]
    assert result["runs"] and "fitted_model" not in result and "validation" not in result
    assert all(r["split"] == "train" for r in result["runs"])
    json.dumps(result, allow_nan=False)


def test_missing_response_is_excluded_once_and_whitelist_is_enforced():
    table, model, params = inputs()
    first = table.rows[0]
    table = replace(
        table,
        rows=(
            replace(
                first,
                values={
                    **first.values,
                    "ln_gamma:A": InputValue(
                        None, "dimensionless", None, "missing measurement", "missing"
                    ),
                },
            ),
            *table.rows[1:],
        ),
    )
    result = ActivityCalibration(table, model, params, fitted_model_id="missing").fit(max_nfev=100)
    assert result["training"]["n_valid"] == 7 and result["training"]["n_excluded"] == 1
    with pytest.raises(ValueError, match="active explicit"):
        ActivityCalibration(
            table, model, (replace(params[0], name="unknown"),), fitted_model_id="bad"
        )
