"""Actual shared fitting, units, holdout isolation, rank and failure accounting."""

import json
from dataclasses import replace
from math import log

import numpy as np
import pytest

from chemworld.data.correlation_calibration import CorrelationCalibration, CorrelationFitParameter
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.enthalpy import caloric_correlation_change
from chemworld.physchem.specs import PropertyCorrelation
from chemworld.physchem.volume_correlations import volume_correlation


def inputs(*, units=False, confounded=False, log_parameter=False):
    source = InputSource(DataSourceProvenance("fit-test", 0, "Authored linear synthetic Cp",
                                            version="1", license="MIT"), "synthetic")
    columns = (
        QuantityColumn("temperature", "degC" if units else "K", "condition", (-100, 1000),
                       "absolute temperature"),
        QuantityColumn("value", "kJ/(mol*K)" if units else "J/(mol*K)", "observation",
                       (0, 1000), "synthetic ideal gas Cp"),
    )
    rows = []
    for i, T in enumerate((250., 300., 350., 400., 275., 375.)):
        T = 300. if confounded else T
        rows.append(ScientificRow(str(i), "fit-test", f"group-{i}", f"sample-{i}",
            "train" if i < 4 else "validation", {
                "temperature": InputValue(T, "K", 0, "exact synthetic input"),
                "value": InputValue(30+.02*T, "J/(mol*K)", .1, "declared synthetic sigma"),
            }))
    table = ScientificTable("linear-cp", "measurements", columns, (source,), tuple(rows))
    curve = PropertyCorrelation("seed", "dippr100", {"A": 32., "B": .01},
        {"temperature": "K"}, "J/(mol*K)", "ideal_gas_heat_capacity",
        {"temperature": (240, 450)}, "Synthetic starting curve",
        {"source": {"source_id": "original-source"}})
    parameters = (CorrelationFitParameter("A", (10, 60), 32),
                  CorrelationFitParameter("B", (.001, .1), .01,
                                          "log" if log_parameter else "linear"))
    return table, curve, parameters


def calibration(*, options=None, **kwargs):
    return CorrelationCalibration(*inputs(**kwargs), fitted_correlation_id="fitted-cp",
                                  **(options or {}))


def test_fit_survives_units_json_and_enters_original_heat_consumer():
    fit = calibration(units=True)
    before = fit.correlation.to_dict()
    result = fit.fit(max_nfev=100)
    assert result["status"] == "fitted"
    assert result["estimate"] == pytest.approx({"A": 30., "B": .02}, abs=1e-8)
    assert result["optimizer"]["jacobian"] == "analytic source coefficients"
    assert result["validation"]["n_valid"] == 2
    assert result["validation"]["weighted_rmse"] < 1e-8
    assert result["identifiability"]["status"] == "full_rank_local"
    assert result["resources"]["runtime_sessions"] == 0
    assert result["resources"]["jacobian_calls"] > 0
    assert fit.correlation.to_dict() == before
    curve = PropertyCorrelation.from_dict(json.loads(json.dumps(result["fitted_correlation"])))
    assert curve.metadata["parent_correlation"] == before
    assert curve.metadata["source"]["source_id"] == "fit:fitted-cp"
    energy = caloric_correlation_change(curve, initial_temperature_K=250, final_temperature_K=400)
    assert energy.value == pytest.approx(5475., rel=1e-10)
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="fresh"):
        fit.fit()


def test_log_coordinate_and_observation_uncertainty_scale_the_analytic_jacobian():
    fit = calibration(units=True, log_parameter=True)
    coordinates = [p.encode(p.initial) for p in fit.parameters]
    actual = fit.training_jacobian(coordinates)
    expected = [[500., T*.01*log(100)/.1] for T in (250, 300, 350, 400)]
    assert actual == pytest.approx(np.asarray(expected), rel=1e-12)
    result = calibration(units=True, log_parameter=True).fit(max_nfev=100)
    assert result["estimate"] == pytest.approx({"A": 30., "B": .02}, abs=1e-8)


def test_changing_holdout_responses_does_not_change_training_or_consume_them_during_fit():
    table, curve, parameters = inputs()
    altered = replace(table, rows=tuple(replace(r, values={**r.values,
        "value": replace(r.values["value"], value=500.)}) if r.split == "validation" else r
        for r in table.rows))
    result = CorrelationCalibration(table, curve, parameters,
                                     fitted_correlation_id="a").fit(max_nfev=100)
    other = CorrelationCalibration(altered, curve, parameters,
                                    fitted_correlation_id="b").fit(max_nfev=100)
    assert result["estimate"] == other["estimate"]
    assert result["training"] == other["training"]
    assert other["validation"]["weighted_rmse"] > 1000
    assert all(r["split"] == "train" for r in result["jacobian_runs"])
    assert all(r["split"] == "train" for r in result["runs"] if r["stage"] != "heldout")


def test_rank_deficiency_is_not_hidden_by_a_small_residual():
    result = calibration(confounded=True).fit(max_nfev=100)
    assert result["status"] == "fitted"
    assert result["training"]["weighted_rmse"] < 1e-7
    assert result["identifiability"]["rank"] == 1
    assert result["identifiability"]["status"] == "rank_deficient"


def test_missing_data_is_explicitly_excluded_and_never_filled():
    table, curve, parameters = inputs()
    row = table.rows[0]
    table = replace(table, rows=(replace(row, values={**row.values,
        "value": InputValue(None, "J/(mol*K)", None, "missing observation", "missing")}),
        *table.rows[1:]))
    result = CorrelationCalibration(table, curve, parameters,
                                     fitted_correlation_id="missing-fit").fit(max_nfev=100)
    assert result["training"]["n_valid"] == 3 and result["training"]["n_excluded"] == 1
    assert result["excluded_observations"][0]["quality"] == "missing"
    assert all(v["row_id"] != "0" for r in result["runs"] for v in r["readouts"])


@pytest.mark.parametrize("kind", ("budget", "domain", "nonconvergence"))
def test_failed_fits_keep_attempts_and_cannot_export_or_consume_holdout(kind):
    table, curve, parameters = inputs()
    if kind == "domain":
        curve = replace(curve, validity_ranges={"temperature": (260, 450)})
    fit = CorrelationCalibration(table, curve, parameters, fitted_correlation_id="failure",
                                  max_forward_groups=1 if kind == "budget" else 300)
    result = fit.fit(max_nfev=1 if kind == "nonconvergence" else 100)
    assert result["status"] == ("optimizer_not_converged" if kind == "nonconvergence" else
                                "forward_failed")
    assert result["failures"] and result["runs"]
    assert "validation" not in result and "fitted_correlation" not in result
    assert not any(r["split"] == "validation" for r in result["runs"])
    json.dumps(result, allow_nan=False)


def test_numerical_derivative_is_explicit_and_never_reported_as_analytic():
    table, original, parameters = inputs()
    curve = replace(original, equation_id="cp_polynomial", coefficients={"a":32., "b":.01})
    parameters = tuple(replace(p, name=p.name.lower()) for p in parameters)
    with pytest.raises(ValueError, match="explicitly requested"):
        CorrelationCalibration(table, curve, parameters, fitted_correlation_id="unsupported")
    result = CorrelationCalibration(table, curve, parameters, fitted_correlation_id="numerical",
        derivative_mode="3-point").fit(max_nfev=100)
    assert result["optimizer"]["jacobian"] == "3-point" and not result["jacobian_runs"]
    assert result["estimate"] == pytest.approx({"a": 30., "b": .02}, abs=1e-8)


@pytest.mark.parametrize("family", ("pressure", "viscosity", "volume"))
def test_existing_analytic_families_use_source_units_and_explicit_pressure(family):
    original, _, _ = inputs()
    pressure_column = None
    if family == "pressure":
        curve = PropertyCorrelation("antoine-seed", "antoine", {"A": 6.1, "B": 800, "C": 293.15},
            {"temperature": "degC"}, "kPa", "vapor_pressure", {"temperature": (-33.15, 176.85)})
        parameter = CorrelationFitParameter("A", (5, 7), 6.1)
        unit, expected_parameter = "Pa", 6.
    elif family == "viscosity":
        curve = PropertyCorrelation("yaws-seed", "viscosity_mu_yaws",
            {"A": -3.1, "B": 500, "C": .001, "D": -1e-6}, {"temperature": "K"},
            "Pa*s", "liquid_viscosity", {"temperature": (240, 450)})
        parameter = CorrelationFitParameter("A", (-4, -2), -3.1)
        unit, expected_parameter = "Pa*s", -3.
    else:
        curve = volume_correlation("Tait", parameters={"P_ref": 1e5, "rho_ref": 810,
            "B": 1e8, "C": .1}, temperature_range_K=(280, 320), pressure_range_Pa=(1e5, 1e8),
            output="density", source_note="synthetic fit test")
        parameter = CorrelationFitParameter("rho_ref", (600, 1000), 810)
        unit, expected_parameter, pressure_column = "kg/m^3", 800., "pressure"
    columns = (original.columns[0], replace(original.columns[1], unit=unit,
                                            valid_range=(0, 1e12)))
    if pressure_column:
        columns += (QuantityColumn("pressure", "Pa", "condition", (1e5, 1e8), "absolute pressure"),)
    rows = []
    for i, old in enumerate(original.rows):
        T = old.values["temperature"].value
        values = {"temperature": old.values["temperature"]}
        if family == "pressure":
            value = 1000*10**(6-800/(T+20))
        elif family == "viscosity":
            value = 10**(-3+500/T+.001*T-1e-6*T*T)
        else:
            P = (1e5, 1e6, 1e7, 2e7, 3e7, 5e7)[i]
            value = 800/(1-.1*log((1e8+P)/(1e8+1e5)))
            values["temperature"] = replace(old.values["temperature"], value=300.)
            values["pressure"] = InputValue(P/1e6, "MPa", 0., "synthetic exact pressure")
        values["value"] = InputValue(value, unit, .01*value, "synthetic 1 percent uncertainty")
        rows.append(replace(old, values=values))
    table = replace(original, columns=columns, rows=tuple(rows))
    fit = CorrelationCalibration(table, curve, (parameter,), pressure_column=pressure_column,
                                  fitted_correlation_id="fitted-"+family)
    result = fit.fit(max_nfev=100)
    assert result["status"] == "fitted"
    assert result["estimate"][parameter.name] == pytest.approx(expected_parameter, abs=1e-8)
    assert result["validation"]["weighted_rmse"] < 1e-8
    assert result["identifiability"]["rank"] == 1
