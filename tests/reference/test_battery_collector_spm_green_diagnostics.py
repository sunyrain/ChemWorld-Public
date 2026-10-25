"""Synthetic wiring/coverage regressions for the read-only electrical diagnostic."""

from copy import deepcopy

import numpy as np
import pytest
from battery_collector_spm_cases import case_data
from battery_collector_spm_green import (
    solve_static_collector,
    static_inputs_from_case,
)
from battery_collector_spm_green_diagnostics import diagnose_electrical, diagnose_segment


def fixture(cells=2):
    data, grid = case_data("CC02"), {"z": cells, "r": 12}
    data["waveform"] = ((1.0, 0.03, "A"),)
    layout = {"negative_tab": "left", "positive_tab": "right"}
    kwargs = static_inputs_from_case(data, grid, layout)
    answer = solve_static_collector(**kwargs, target=0.03)
    profiles = {}
    for name in ("negative", "positive"):
        profiles[name] = {
            "theta": np.full((2, cells, 12), data[name]["initial_fraction"]).tolist(),
            "surface_fraction": [answer[name]["surface_fraction"].tolist()] * 2,
            "reaction_current_A": [
                ((1 if name == "negative" else -1) * answer["through_current_A"]).tolist()
            ]
            * 2,
            "collector_potential_V": [answer["collectors"][name]["fv_centre_V"].tolist()] * 2,
            "local_voltage_V": [answer["collectors"]["local_voltage_V"]["fv_centre_V"].tolist()]
            * 2,
        }
    segment = {
        "times_s": [0.0, 1.0],
        "values": [[answer["voltage_V"], 0.03, 0.0, 0.0]] * 2,
        "raw_profiles": profiles,
        "source_boundary_voltage_V": [answer["collectors"]["source_boundary_voltage_V"]] * 2,
    }
    return data, grid, layout, segment


def test_all_samples_checked_and_potential_corruption_is_not_hidden_by_refitting():
    data, grid, layout, segment = fixture()
    segment["raw_profiles"]["negative"]["collector_potential_V"][1][0] += 0.02
    frozen = deepcopy(segment)
    result = diagnose_segment(segment, "native", data, grid, layout, 0)
    assert result["saved_samples"] == result["linear_samples"] == result["static_samples"] == 2
    assert not result["failures"]
    assert result["metrics"]["negative_FV_potential_residual_V"]["max_abs"] == pytest.approx(0.02)
    assert result["metrics"]["static_q_minus_recorded_q_A"]["max_abs"] < 1e-10
    assert segment == frozen


def test_missing_segment_is_retained_and_does_not_hide_later_segments():
    data, grid, layout, segment = fixture()
    broken = deepcopy(segment)
    del broken["values"]
    trace = {"grid": grid, "layout": layout, "segments": [broken, segment]}
    # The second segment has no matching authored waveform; its linear samples
    # are still reported and its static exceptions remain attached per sample.
    report = {
        "units": [
            {"id": "CC02", "inputs": data, "routes": {"history_opposite_coarse": {"native": trace}}}
        ]
    }
    result = diagnose_electrical(report)
    assert len(result["failures"]) == 1
    records = result["units"][0]["routes"]["history_opposite_coarse"]["native"]["segments"]
    assert "unavailable" in records[0]
    assert records[1]["linear_samples"] == 2
    assert records[1]["static_samples"] == 0
    assert len(records[1]["failures"]) == 2


def test_failed_implementation_partial_segments_recover_frozen_route_geometry():
    data, _grid, _layout, segment = fixture(8)
    route = {
        "partial_history": {
            "stage": "native",
            "failed_implementation_context": {
                "completed_segments": [segment],
            },
        }
    }
    report = {
        "units": [{"id": "CC02", "inputs": data, "routes": {"history_opposite_coarse": route}}]
    }
    result = diagnose_electrical(report)
    assert not result["failures"]
    record = result["units"][0]["routes"]["history_opposite_coarse"]["native"]["segments"][0]
    assert record["linear_samples"] == record["static_samples"] == 2
    assert not record["failures"]


def test_source_tab_inference_uses_port_current_and_reports_reaction_current_gap():
    data, grid, layout, segment = fixture()
    answer = solve_static_collector(**static_inputs_from_case(data, grid, layout), target=0.03)
    profiles = segment["raw_profiles"]
    fields = {
        name.capitalize() + " current collector potential [V]": profiles[name][
            "collector_potential_V"
        ]
        for name in ("negative", "positive")
    }
    fields["Local voltage [V]"] = profiles["negative"]["local_voltage_V"]
    for name in ("negative", "positive"):
        profiles[name]["z_edges_m"] = np.linspace(0, data["height_m"], 3).tolist()
        profiles[name]["open_circuit_potential_V"] = [answer[name]["ocv_V"].tolist()] * 2
        profiles[name]["kinetic_overpotential_V"] = [
            answer[name]["kinetic_overpotential_V"].tolist()
        ] * 2
    segment["source_fields"] = fields
    segment["values"] = [[answer["voltage_V"], 0.031, 0.0, 0.0]] * 2
    result = diagnose_segment(segment, "source", data, grid, layout, 0)
    conductance = data["sigma_cc_S_m"] * data["collector_thickness_m"] * data["width_m"]
    expected = 0.001 * data["height_m"] / 2 / (2 * conductance)
    metrics = result["metrics"]
    assert result["source_identity_samples"] == result["linear_samples"] == 2
    assert result["static_samples"] == 0
    assert not result["failures"]
    assert metrics["reaction_sum_minus_port_half_cell_voltage_V"]["max_abs"] == pytest.approx(
        expected
    )
    assert metrics["positive_FV_potential_residual_V"]["max_abs"] == pytest.approx(expected)
    assert metrics["source_raw_OCP_plus_eta_identity_residual_V"]["max_abs"] < 1e-10
