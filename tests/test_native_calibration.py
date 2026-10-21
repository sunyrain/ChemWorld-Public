"""Boundary input, immutable configuration and actual failed forward execution."""

from dataclasses import replace

import pytest

from chemworld.data.calibration import (
    CalibrationParameter,
    ForwardEvaluationError,
    NativeCalibration,
    parameterized_world,
)
from chemworld.data.scientific_inputs import QuantityColumn
from chemworld.examples.native_calibration import fixed_inputs


@pytest.mark.parametrize("kind", ["thermal", "battery"])
def test_parameters_change_only_whitelisted_configuration_and_preserve_initial_inventory(kind):
    table, parameters, experiments = fixed_inputs(kind)
    experiment = experiments[0]
    before = experiment.composition.to_dict()
    candidate = {p.column.quantity_id: p.initial for p in parameters}
    configured = parameterized_world(experiment, parameters, candidate).to_dict()
    for parameter in parameters:
        old, new = before, configured
        for part in parameter.path[:-1]:
            old, new = old[part], new[part]
        assert new[parameter.path[-1]] == candidate[parameter.column.quantity_id]
        # Restore just the allowed parameter and compare the entire structure.
        new[parameter.path[-1]] = old[parameter.path[-1]]
    assert configured == before
    assert experiment.composition.to_dict() == before
    assert len(NativeCalibration(table, parameters, experiments).experiments) == 3


@pytest.mark.parametrize("path", [
    ("physics", "initial_states", "reactor", "temperature_K"),
    ("physics", "initial_states", "reactor", "state_slots", "inventory"),
    ("task", "budget"),
    ("physics", "network", "reactions"),
])
def test_observations_task_targets_and_initial_inventory_are_not_parameter_paths(path):
    with pytest.raises(ValueError, match="model"):
        CalibrationParameter(
            QuantityColumn("forbidden", "K", "parameter", (1, 1000), "absolute"), path, 350,
        )


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -.001, 1.001])
def test_parameter_coordinates_never_clip_invalid_candidates(coordinate):
    _, parameters, _ = fixed_inputs("thermal")
    with pytest.raises(ValueError):
        parameters[0].decode(coordinate)


@pytest.mark.parametrize("value", [.009, .201, float("nan"), float("inf")])
def test_parameter_candidates_must_remain_inside_declared_range(value):
    _, parameters, experiments = fixed_inputs("thermal")
    with pytest.raises(ValueError):
        parameterized_world(experiments[0], parameters, {"heat_loss": value})


def test_unknown_parameters_and_integer_model_dimensions_cannot_be_optimized():
    _, parameters, experiments = fixed_inputs("battery")
    with pytest.raises(ValueError, match="whitelist"):
        parameterized_world(experiments[0], parameters, {"other": 1})
    grid = replace(parameters[0], path=(*parameters[0].path[:-1], "cells"))
    with pytest.raises(ValueError, match="continuous"):
        parameterized_world(experiments[0], (grid,), {"negative_diffusivity": 1.5e-14})


@pytest.mark.parametrize("uncertainty", [None, 0.0])
def test_unknown_or_zero_uncertainty_does_not_silently_become_a_weight(uncertainty):
    table, parameters, experiments = fixed_inputs("thermal")
    row = table.rows[0]
    changed = replace(
        row, values={**row.values, "pressure": replace(
            row.values["pressure"], uncertainty=uncertainty,
        )},
    )
    with pytest.raises(ValueError, match="positive declared"):
        NativeCalibration(replace(table, rows=(changed, *table.rows[1:])), parameters, experiments)


def test_invalid_data_is_retained_in_denominator_with_explicit_exclusion():
    table, parameters, experiments = fixed_inputs("thermal")
    row = table.rows[0]
    changed = replace(
        row, values={**row.values, "pressure": replace(
            row.values["pressure"], value=None, quality="invalid",
        )},
    )
    calibration = NativeCalibration(
        replace(table, rows=(changed, *table.rows[1:])), parameters, experiments,
    )
    assert len(calibration.table.rows) == 18
    assert len(calibration.pairs["train"]) == 11
    assert calibration.excluded == [{
        "row_id": row.row_id, "quantity_id": "pressure", "split": "train", "quality": "invalid",
    }]


@pytest.mark.parametrize("defect", ["duplicate_parameter", "duplicate_path", "group", "readout"])
def test_ambiguous_parameter_or_measurement_assignments_are_rejected(defect):
    table, parameters, experiments = fixed_inputs("thermal")
    if defect == "duplicate_parameter":
        parameters = (*parameters, parameters[0])
    elif defect == "duplicate_path":
        parameters = (*parameters, replace(
            parameters[0], column=replace(parameters[0].column, quantity_id="other"),
        ))
    elif defect == "group":
        experiments = (*experiments, experiments[0])
    else:
        experiments = (
            replace(experiments[0], readouts=experiments[0].readouts[:-1]), *experiments[1:],
        )
    with pytest.raises(ValueError):
        NativeCalibration(table, parameters, experiments)


def test_residuals_use_canonical_uncertainty_and_never_include_holdout_rows():
    table, parameters, experiments = fixed_inputs("thermal")
    calibration = NativeCalibration(table, parameters, experiments)
    predicted = {
        key: calibration.rows[key[0]].values[key[1]].value + 10.0
        for key in calibration.pairs["train"]
    }
    assert calibration.residuals(predicted, "train") == pytest.approx([.5] * 12)
    assert calibration.metrics(predicted, "train")["weighted_rmse"] == .5
    assert len(calibration.pairs["validation"]) == 6
    # No validation prediction is available, and none is requested for training.
    assert not any(key in predicted for key in calibration.pairs["validation"])


def test_prepared_experiments_do_not_follow_external_action_mutations():
    table, parameters, experiments = fixed_inputs("thermal")
    calibration = NativeCalibration(table, parameters, experiments)
    experiments[0].actions[1]["duration_s"] = -1
    assert calibration.experiments[0].actions[1]["duration_s"] == 19


def test_rejected_native_history_retains_failure_resources_and_does_not_evaluate_holdout():
    table, parameters, experiments = fixed_inputs("thermal")
    actions = list(experiments[0].actions)
    actions[1] = {"operation": "wait", "duration_s": -1}
    experiments = (replace(experiments[0], actions=tuple(actions)), *experiments[1:])
    calibration = NativeCalibration(table, parameters, experiments)
    result = calibration.fit()
    assert result["status"] == "forward_failed"
    assert result["failures"] and result["runs"][0]["failure"]
    assert result["resources"]["forward_groups"] == 1
    assert result["resources"]["attempted_operations"] == 2
    assert result["resources"]["committed_operations"] == 1
    assert len(result["runs"][0]["steps"]) == 2
    assert result["runs"][0]["steps"][-1]["transaction_status"] != "committed"
    assert result["resources"]["actual_duration_s"] == 0
    assert "validation" not in result and "estimate" not in result
    with pytest.raises(ValueError, match="fresh"):
        calibration.fit()


def test_sampling_time_mismatch_is_not_interpolated_or_relabelled():
    table, parameters, experiments = fixed_inputs("thermal")
    actions = list(experiments[0].actions)
    actions[1] = {**actions[1], "duration_s": 18}
    experiments = (replace(experiments[0], actions=tuple(actions)), *experiments[1:])
    calibration = NativeCalibration(table, parameters, experiments)
    with pytest.raises(ForwardEvaluationError, match="sampling time"):
        calibration.forward({"heat_loss": .06}, "train")
    record = calibration.runs[0]
    assert len(record["steps"]) == 3
    assert record["steps"][-1]["measurement_status"] == "valid"
    assert record["steps"][-1]["cost_delta"] == .02
    assert record["readouts"][0]["native"]["sampled_at"]["time_s"] == 19
    assert record["readouts"][0]["value"] is None
    assert "sampling time" in record["failure"]


def test_declared_forward_limit_is_hard_and_retains_completed_groups():
    table, parameters, experiments = fixed_inputs("thermal")
    calibration = NativeCalibration(table, parameters, experiments, max_forward_groups=1)
    with pytest.raises(ForwardEvaluationError, match="budget"):
        calibration.forward({"heat_loss": .06}, "train")
    assert len(calibration.runs) == 1
    assert calibration.runs[0]["failure"] is None
    assert len(calibration.runs[0]["readouts"]) == 6
