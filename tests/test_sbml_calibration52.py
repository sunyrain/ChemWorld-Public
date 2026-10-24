"""Actual native calibration, analytic tangents, scope and immutable controls."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.sbml import SBMLModel
from chemworld.data.sbml_calibration import SBMLCalibration
from chemworld.examples.sbml_calibration import TIMES, fixed_inputs, pool_model
from chemworld.physchem.sbml_sensitivity import sbml_trajectory_sensitivities
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import validate_model_state
from reference.sbml_calibration52_reference import reference


@pytest.mark.parametrize(
    "kind,scaled", [("decay", False), ("reversible", False), ("decay", True), ("reversible", True)]
)
def test_original_mathml_tangents_match_closed_forms_and_native_packets(kind, scaled):
    table, parameters, experiments = fixed_inputs(kind=kind, scaled=scaled)
    fit = SBMLCalibration(table, parameters, experiments)
    values = {p.parameter_id: {"k1": 0.55, "k2": 0.23}[p.parameter_id] for p in parameters}
    analysis = fit.sensitivity(experiments[0], values)
    packets = fit.forward(values, "train")
    for index, entry in enumerate(analysis["records"]):
        expected = reference(
            entry["time_s"],
            kind=kind,
            initial_b=0.2 if kind == "reversible" else 0,
            source_seconds=60 if scaled else 1,
        )
        for channel, reading in entry["readouts"].items():
            value, derivatives = expected[channel]
            assert reading["value"] == pytest.approx(value, abs=2e-8, rel=2e-6)
            assert reading["derivative"] == pytest.approx(derivatives, abs=2e-8, rel=2e-6)
            assert packets[(f"pools-0-{index}", channel)] == pytest.approx(value, abs=2e-8)
    json.dumps(analysis, allow_nan=False)


def test_weighted_log_jacobian_uses_original_source_and_observation_units():
    table, parameters, experiments = fixed_inputs(kind="reversible", scaled=True)
    fit = SBMLCalibration(table, parameters, experiments)
    x = np.array([p.encode(p.initial) for p in parameters])
    actual = fit.training_jacobian(x)
    differences = []
    for column in range(len(parameters)):
        dx = np.zeros(len(parameters))
        dx[column] = 1e-4
        positive = fit.residuals(fit.forward(fit.values(x + dx), "train"), "train")
        negative = fit.residuals(fit.forward(fit.values(x - dx), "train"), "train")
        differences.append((positive - negative) / (2e-4))
    np.testing.assert_allclose(actual, np.asarray(differences).T, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("kind", ["decay", "reversible", "confounded"])
def test_actual_public_measurement_histories_fit_whole_group_holdout(kind):
    result = SBMLCalibration(*fixed_inputs(kind=kind)).fit()
    assert result["status"] == "fitted"
    assert result["training"]["weighted_rmse"] < 0.01
    assert result["validation"]["weighted_rmse"] < 0.01
    assert result["identifiability"]["rank"] == (
        1 if kind == "confounded" else len(result["estimate"])
    )
    if kind != "confounded":
        for name, value in result["estimate"].items():
            assert value == pytest.approx({"k1": 0.55, "k2": 0.23}[name], rel=2e-4)
    assert result["resources"]["jacobian_calls"] > 0
    assert all(
        run["steps"][0]["action"]["operation"] == "set_model_parameters" for run in result["runs"]
    )
    assert all(run["stage"] == "heldout" for run in result["runs"] if run["split"] == "validation")
    assert all(
        group["group_id"] != "pools-2" for run in result["jacobian_runs"] for group in run["groups"]
    )
    json.dumps(result, allow_nan=False)


def test_bounded_fit_reports_active_endpoint():
    result = SBMLCalibration(*fixed_inputs(boundary=True)).fit()
    assert result["status"] == "fitted"
    assert result["estimate"]["k1"] == pytest.approx(0.4, abs=2e-5)
    assert result["active_bounds"]["k1"] == 1


def test_control_preserves_source_initial_assignments_and_original_world():
    inputs = fixed_inputs(initial_assignment=True)
    fit = SBMLCalibration(*inputs)
    experiment = inputs[2][0]
    before = experiment.composition.to_dict()
    model, initial = fit._model_initial(experiment)
    analysis = fit.sensitivity(experiment, {"k1": 0.8})
    assert analysis["initial_coordinates_after_control"]["N:A"] == pytest.approx(0.8)
    assert initial["P:k1"] == pytest.approx(0.3)
    assert analysis["initial_coordinates_after_control"]["P:k1"] == 0.8
    assert analysis["records"][0]["readouts"]["amount_A"]["derivative"]["k1"] == 0
    spec, actions = fit._candidate(experiment, {"k1": 0.8})
    runtime = ChemWorldRuntime(composition=spec)
    try:
        runtime.reset(seed=0)
        outcome = runtime.execute(actions[0])
        assert outcome.info["transaction_status"] == "committed"
        state = runtime._state
        _, controlled, _ = validate_model_state(state, model)
        assert controlled == analysis["initial_coordinates_after_control"]
    finally:
        runtime.close()
    assert experiment.composition.to_dict() == before


def test_zero_observable_has_explicitly_undefined_normalized_response():
    model = SBMLSystemSpec(
        pool_model(),
        observables=(SBMLObservable("amount", "B"),),
        controlled_parameters=("k1",),
        readout_relative_sd=0.0,
    )
    result = sbml_trajectory_sensitivities(model, model.frame.initial_values, {"k1": 0.55}, TIMES)
    zero = result["records"][0]["readouts"]["amount_B"]
    assert zero == {
        "value": 0.0,
        "unit": "mol",
        "derivative": {"k1": 0.0},
        "normalized": {"k1": None},
    }


@pytest.mark.parametrize(
    "defect",
    [
        "noise",
        "second_control",
        "missing_control",
        "wrong_unit",
        "unknown_parameter",
        "duplicate_source",
        "time",
        "meter",
    ],
)
def test_unsupported_control_histories_are_explicit(defect):
    table, parameters, experiments = fixed_inputs()
    experiment = experiments[0]
    if defect == "noise":
        physics = experiment.composition.physics
        model = physics.models_of("sbml")["formal_workspace"]
        composition = replace(
            experiment.composition,
            physics=replace(
                physics,
                models={
                    **physics.models,
                    "sbml": {"formal_workspace": replace(model, readout_relative_sd=0.01)},
                },
            ),
        )
        experiments = (replace(experiment, composition=composition), *experiments[1:])
    elif defect in {"second_control", "missing_control", "meter"}:
        actions = [dict(action) for action in experiment.actions]
        if defect == "second_control":
            actions[2] = actions[0]
        elif defect == "missing_control":
            actions[0] = {"operation": "wait", "duration_s": 1.0}
        else:
            actions[1]["instrument"] = "gas_pressure"
        experiments = (replace(experiment, actions=tuple(actions)), *experiments[1:])
    elif defect == "wrong_unit":
        parameters = (replace(parameters[0], column=replace(parameters[0].column, unit="K")),)
    elif defect == "unknown_parameter":
        parameters = (replace(parameters[0], source_id="unknown"),)
    elif defect == "duplicate_source":
        parameters = (
            *parameters,
            replace(parameters[0], column=replace(parameters[0].column, quantity_id="other")),
        )
    else:
        row = table.rows[0]
        table = replace(
            table,
            rows=(
                replace(row, values={**row.values, "time": replace(row.values["time"], value=0.1)}),
                *table.rows[1:],
            ),
        )
    with pytest.raises(ValueError):
        SBMLCalibration(table, parameters, experiments)


@pytest.mark.parametrize("defect", ["time", "event", "rate_rule", "nonsmooth"])
def test_unsupported_source_sensitivity_semantics_are_rejected(defect):
    original = pool_model()
    xml = original.to_xml()
    if defect == "time":
        xml = xml.replace(
            "<ci>k1</ci><ci>A</ci>",
            "<ci>k1</ci><ci>A</ci><apply><exp/>"
            '<apply><times/><ci>k1</ci><csymbol encoding="text" '
            'definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol></apply></apply>',
        )
    elif defect == "event":
        xml = xml.replace(
            "</model>",
            '<listOfEvents><event id="event" useValuesFromTriggerTime="true">'
            '<trigger initialValue="false" persistent="true"><math '
            'xmlns="http://www.w3.org/1998/Math/MathML"><true/></math></trigger>'
            "</event></listOfEvents></model>",
        )
    elif defect == "rate_rule":
        xml = xml.replace(
            'id="k2" value="0.2" units="rate" constant="true"',
            'id="k2" value="0.2" units="rate" constant="false"',
        )
        xml = xml.replace(
            "</model>",
            '<listOfRules><rateRule variable="k2"><math '
            'xmlns="http://www.w3.org/1998/Math/MathML"><apply><times/><ci>k2</ci><ci>k2</ci>'
            "</apply></math></rateRule></listOfRules></model>",
        )
    else:
        xml = xml.replace("<ci>A</ci>", "<apply><abs/><ci>A</ci></apply>")
    with pytest.raises(ValueError):
        model = SBMLSystemSpec(
            SBMLModel(xml, original.source), controlled_parameters=("k1",), readout_relative_sd=0.0
        )
        sbml_trajectory_sensitivities(model, model.frame.initial_values, {"k1": 0.5}, TIMES)


def test_missing_observations_and_forward_budget_remain_in_report():
    table, parameters, experiments = fixed_inputs()
    row = table.rows[1]
    table = replace(
        table,
        rows=(
            table.rows[0],
            replace(
                row,
                values={
                    **row.values,
                    "amount_A": replace(row.values["amount_A"], quality="missing", value=None),
                },
            ),
            *table.rows[2:],
        ),
    )
    result = SBMLCalibration(table, parameters, experiments, max_forward_groups=1).fit()
    assert result["status"] == "forward_failed"
    assert result["resources"]["forward_groups"] == 1
    assert result["resources"]["committed_operations"] > 0
    assert len(result["excluded_observations"]) == 1
    assert "budget" in result["failures"][0]


def test_failed_tangent_retains_attempted_group_and_actual_rhs_work(monkeypatch):
    import chemworld.physchem.sbml_sensitivity as module

    def fail_after_work(rhs, initial, **kwargs):
        rhs(0.0, initial)
        raise ValueError("declared synthetic tangent failure after one RHS call")

    monkeypatch.setattr(module, "solve_ode", fail_after_work)
    result = SBMLCalibration(*fixed_inputs()).fit()
    assert result["status"] == "forward_failed"
    assert result["resources"]["tangent_groups"] == 1
    assert result["resources"]["tangent_rhs_evaluations"] == 1
    analysis = result["jacobian_runs"][0]["groups"][0]["analysis"]
    assert analysis["status"] == "failed"
    assert len(analysis["records"]) == 1
    assert "synthetic tangent failure" in analysis["failure"]
    json.dumps(result, allow_nan=False)


def test_declared_linear_parameter_coordinate_has_correct_weighted_chain_rule():
    table, parameters, experiments = fixed_inputs(scaled=True)
    parameters = (replace(parameters[0], coordinate="linear"),)
    calibration = SBMLCalibration(table, parameters, experiments)
    x = [parameters[0].encode(0.3)]
    jacobian = calibration.training_jacobian(x)
    for index, (row_id, quantity) in enumerate(calibration.pairs["train"]):
        row = calibration.rows[row_id]
        initial_a = 0.8 if row.group_id == "pools-0" else 0.45
        slope = reference(row.values["time"].value, initial_a=initial_a, k1=0.3)[quantity][1]["k1"]
        expected = slope * (1.5 - 0.04) / row.values[quantity].uncertainty
        assert jacobian[index, 0] == pytest.approx(expected, abs=2e-7, rel=2e-6)
