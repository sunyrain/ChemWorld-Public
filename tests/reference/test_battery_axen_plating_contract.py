"""Preparation contracts, excluded from the fixed Axen/plating scientific block."""

from copy import deepcopy

import numpy as np
import pytest
from battery_axen_plating_cases import REFUSALS, active_electrodes, case_data
from battery_axen_plating_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 13))
def test_actual_source_build_keeps_original_h_domains_and_scales(index):
    data = case_data(index)
    simulation = build_reference(data, 8)
    simulation.build()
    model = simulation.built_model
    variables = [v for v in model.rhs if "hysteresis" in v.name]
    assert len(variables) == len(active_electrodes(data))
    assert all(model.rhs[v].shape == (5, 1) for v in variables)
    assert all(float(v.scale.evaluate()) == 1.0 for v in variables)
    metals = [
        v
        for v in model.rhs
        if "lithium plating concentration" in v.name or "dead lithium" in v.name
    ]
    assert len(metals) == 2
    assert all(model.rhs[v].shape == (1, 1) for v in metals)
    assert all(float(v.scale.evaluate()) == 1000.0 for v in metals)


def short_data(index):
    data = deepcopy(case_data(index))
    data.update(
        waveform=((0.2, *data["waveform"][0][1:]),), samples_per_segment=3, expected="complete"
    )
    return data


@pytest.mark.parametrize("index", (1, 2, 3, 4, 5, 6, 7, 9, 10))
def test_source_short_observables(index):
    from battery_axen_plating_assess import axen_metrics, integrity

    data = short_data(index)
    trace = reference_time(data, 8)
    checks = integrity(trace, data)
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    metrics = axen_metrics(trace, data)
    assert metrics["h_rate_limit_ratio"] <= 1
    assert metrics["raw_h_copy_error"] <= 1e-12
    assert metrics["OCP_mixture_error_V"] <= 1e-7


@pytest.mark.parametrize("index", (1, 2, 3, 4, 5, 6, 7, 9, 10))
def test_native_short_observables(index):
    from battery_axen_plating_assess import axen_metrics, integrity
    from battery_axen_plating_traces import native_time

    data = short_data(index)
    trace = native_time(data, 8)
    checks = integrity(trace, data)
    assert all(checks.values()), [k for k, v in checks.items() if not v]
    metrics = axen_metrics(trace, data)
    assert metrics["h_rate_limit_ratio"] <= 1
    assert metrics["OCP_mixture_error_V"] <= 1e-7
    assert "rhs_boundary_extensions_mol" in trace["segments"][0]["solver"]
    if index == 6:
        assert metrics["wrong_terminal_drive_rate_error_s_inv"] > 1e-5


@pytest.mark.parametrize("name", REFUSALS)
def test_named_refusals(name):
    from battery_axen_plating_traces import refusal

    result = refusal(case_data(13), name)
    assert result["rejected"], result
    assert result["input_state_unchanged"]


@pytest.mark.parametrize("index", (1, 2))
def test_baseline_short_complete_and_physical_match(index):
    from battery_axen_plating_assess import compare, compare_h, integrity
    from battery_axen_plating_traces import native_time

    data = short_data(index)
    for function in (native_time, reference_time):
        old = function(data, 8, baseline=True)
        new = function(data, 8)
        checks = integrity(old, data)
        assert all(checks.values()), [k for k, v in checks.items() if not v]
        assert compare(new, old, old_limit=True)["passed"]
        assert compare_h(new, old, old_limit=True)["passed"]


def test_declared_denominators():
    from run_battery_axen_plating_block import tasks

    assert sum(len(tasks(i, case_data(i))) for i in range(1, 14)) == 70
    assert sum(len(tasks(i, case_data(i))) for i in range(1, 13)) == 56


def test_required_history_fields_and_copy_shape_fail_closed():
    from battery_axen_plating_assess import integrity

    data = short_data(3)
    trace = reference_time(data, 8)
    for field in ("termination", "hysteresis", "raw_profiles_theta", "rows"):
        bad = deepcopy(trace)
        del bad["segments"][0][field]
        assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["segments"][0]["hysteresis"]["negative"]["raw-h"] = [[0.0] * 3]
    assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["segments"][0]["hysteresis"]["negative"]["raw-h"][0][0] = np.nan
    assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["segments"][0]["times_s"][-1] = 0.1
    assert not all(integrity(bad, data).values())


def test_native_missing_fields_or_h_reset_are_rejected():
    from battery_axen_plating_assess import axen_metrics, integrity
    from battery_axen_plating_traces import native_time

    data = short_data(5)
    data["waveform"] = ((0.2, -0.6, "A"), (0.2, 0.0, "A"))
    trace = native_time(data, 8)
    for field in ("outcome", "raw_amounts_mol", "raw_coordinates", "solver"):
        bad = deepcopy(trace)
        del bad["segments"][0][field]
        assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["segments"][1]["hysteresis"]["negative"]["h"][0] += 0.1
    assert axen_metrics(bad, data)["h_stage_jump"] > 0.09
    bad = deepcopy(trace)
    del bad["segments"][-1]
    assert not all(integrity(bad, data).values())


def test_route_binding_cannot_hide_h_or_substitute_another_grid():
    from battery_axen_plating_assess import route_binding

    data = case_data(2)
    valid = {
        "implementation": "native SPM Axen/plating",
        "radial_cells": 40,
        "baseline": False,
        "has_plating": True,
        "h_electrodes": ("negative", "positive"),
    }
    assert all(route_binding("native_coarse", valid, data).values())
    for changes in (
        {"baseline": True, "h_electrodes": ()},
        {"radial_cells": 80},
        {"implementation": "actual PyBaMM25.4.2 SPM Axen/plating"},
        {"has_plating": False},
    ):
        assert not all(route_binding("native_coarse", {**valid, **changes}, data).values())


def test_all_source_scales_are_required_and_solver_metadata_is_explicit():
    from battery_axen_plating_assess import integrity

    data = short_data(3)
    trace = reference_time(data, 8)
    for key in trace["source_state_scales"]:
        bad = deepcopy(trace)
        del bad["source_state_scales"][key]
        assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["source_state_scales"]["fake-metal"] = 1000.0
    assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["tolerances"] = {}
    assert not all(integrity(bad, data).values())


def test_native_raw_identity_and_required_diagnostics_are_not_optional():
    from battery_axen_plating_assess import integrity
    from battery_axen_plating_traces import native_time

    data = short_data(3)
    trace = native_time(data, 8)
    for key, delta in (("discharged_charge_C", 0.1), ("delivered_work_J", 0.1)):
        bad = deepcopy(trace)
        bad["segments"][0]["raw_coordinates"][1][key] += delta
        assert not all(integrity(bad, data).values())
    for key in ("negative:Li:0", "plating:Li", "dead:Li"):
        bad = deepcopy(trace)
        bad["segments"][0]["raw_amounts_mol"][1][key] += 1e-6
        assert not all(integrity(bad, data).values())
    for solver in (
        {},
        {**trace["segments"][0]["solver"], "residual": {}},
        {**trace["segments"][0]["solver"], "new_counter": np.nan},
    ):
        bad = deepcopy(trace)
        bad["segments"][0]["solver"] = solver
        assert not all(integrity(bad, data).values())
    bad = deepcopy(trace)
    bad["segments"][0]["roundoff_corrections_mol"] = {"negative:Li:0": np.inf}
    assert not all(integrity(bad, data).values())
