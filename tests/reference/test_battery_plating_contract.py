"""Short preparation contracts; never counted as the fixed scientific block."""

from copy import deepcopy

import numpy as np
import pytest
from battery_plating_cases import RESPONSES, case_data
from battery_plating_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 13))
def test_actual_source_builds(index):
    simulation = build_reference(case_data(index), 8)
    simulation.build()
    assert simulation.built_model.len_rhs > 16
    assert simulation.built_model.len_alg >= 2
    for key in (
        "qualification:plated_mol",
        "qualification:dead_mol",
        "qualification:stripping_current_A",
    ):
        assert key in simulation.built_model.variables


@pytest.mark.parametrize("index", (2, 4, 6, 9, 10))
def test_source_short_history_and_same_kinetics(index):
    data = deepcopy(case_data(index))
    data["waveform"] = ((0.2, *data["waveform"][0][1:]),)
    data["samples_per_segment"] = 3
    trace = reference_time(data, 8)
    segment = trace["segments"][0]
    values = np.asarray(segment["rows"])
    assert values.shape == (3, len(RESPONSES))
    assert np.isfinite(values).all()
    assert segment["times_s"][0] == 0
    assert segment["end_time_s"] == pytest.approx(0.2)
    assert values[0, 2:4] == pytest.approx([0.0, 0.0])
    q, cpl, potential = values[:, 4], values[:, 8], values[:, 7]
    f, r = trace["F_C_mol"], trace["R_J_mol_K"]
    exponent = f * potential / (r * data["temperature_K"])
    area = 3 * data["electrodes"][0]["active_volume_m3"] / data["electrodes"][0]["radius_m"]
    expected = -area * f * data["k_m_s"] * data["ce_mol_m3"] * np.exp(-data["alpha"] * exponent)
    if data["mode"] == "reversible":
        expected += area * f * data["k_m_s"] * cpl * np.exp((1 - data["alpha"]) * exponent)
    assert q == pytest.approx(expected, abs=1e-9, rel=1e-8)
    assert values[:, 5] == pytest.approx(cpl * data["negative_electrode_volume_m3"])
    assert np.asarray(segment["raw_profiles_theta"]).shape == (2, 3, 8)


def test_old_source_zero_rate_build():
    simulation = build_reference(case_data(1), 8, old_spm=True)
    simulation.build()
    assert simulation.built_model.len_alg >= 2


@pytest.mark.parametrize("index", (1, 2, 4, 6, 9, 10))
def test_native_short_history_and_inventory_mapping(index):
    from battery_plating_traces import history_metrics, native_time

    data = deepcopy(case_data(index))
    data["waveform"] = ((0.2, *data["waveform"][0][1:]),)
    data["samples_per_segment"] = 3
    trace = native_time(data, 8)
    assert trace["segments"][0]["end_time_s"] == pytest.approx(0.2)
    assert trace["segments"][0]["outcome"] == "success"
    assert isinstance(trace["segments"][0]["roundoff_corrections_mol"], dict)
    assert trace["segments"][0]["solver"]["accepted_nonnegative_relative_tolerance"] == 1e-10
    assert "rhs_boundary_extensions_mol" in trace["segments"][0]["solver"]
    metrics = history_metrics(trace, data)
    assert metrics["total_Li_error_mol"] < 1e-10
    assert metrics["independent_kinetic_limit_ratio"] < 1
    if index == 1:
        original = native_time(data, 8, old_spm=True)
        assert np.asarray(trace["segments"][0]["rows"]) == pytest.approx(
            np.asarray(original["segments"][0]["rows"]), abs=1e-7
        )


def test_integrity_rejects_incomplete_histories_and_wrong_initial_state():
    from battery_plating_assess import integrity

    data = deepcopy(case_data(2))
    data["waveform"] = ((0.2, -0.6, "A"),)
    data["samples_per_segment"] = 3
    trace = reference_time(data, 8)
    assert all(integrity(trace, data).values())
    assert not all(integrity({"segments": []}, data).values())
    for kind in ("shortened", "no_t0", "initial_inventory", "bad_outcome", "nonfinite_profile"):
        altered = deepcopy(trace)
        segment = altered["segments"][0]
        if kind == "shortened":
            segment["end_time_s"] -= 0.1
        elif kind == "no_t0":
            segment["times_s"][0] += 0.01
        elif kind == "initial_inventory":
            segment["rows"][0][12] += 1e-4
        elif kind == "bad_outcome":
            segment["outcome"] = "arbitrary-success-name"
        else:
            segment["raw_profiles_theta"][0][0][0] = float("nan")
        assert not all(integrity(altered, data).values()), kind


def test_route_denominators_and_no_selected_subset():
    from run_battery_plating_block import tasks

    assert sum(len(tasks(i, case_data(i))) for i in range(1, 14)) == 64
    assert sum(len(tasks(i, case_data(i))) for i in range(1, 13)) == 52
