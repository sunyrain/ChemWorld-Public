"""Build/discretise contracts for the 45-COLLECTOR-SPM source adapter."""

from itertools import pairwise

import numpy as np
import pytest
from battery_collector_spm_cases import (
    EXPECTED_ROUTES,
    GRIDS,
    HISTORY_NAMES,
    LAYOUTS,
    NAMES,
    REFUSAL_NAMES,
    case_data,
    route_plan,
)
from battery_collector_spm_pybamm import build_reference, source_contract_snapshot
from battery_collector_spm_traces import refusal


def _short_case(name="CC02", *, target=0.01, unit="A"):
    """A 0.1 ms adapter fixture, never a replacement qualification route."""
    data = case_data(name)
    data["waveform"] = ((1e-4, target, unit),)
    data["samples_per_segment"] = 3
    return data


def test_source_short_trace_has_actual_mesh_current_and_work():
    from battery_collector_spm_traces import source_time

    data = _short_case()
    trace = source_time(data, {"z": 4, "r": 6}, LAYOUTS["opposite"])
    segment = trace["segments"][0]
    values = np.asarray(segment["values"])
    assert values[0, 2:] == pytest.approx([0, 0])
    assert values[-1, 2] == pytest.approx(1e-6, abs=1e-10)
    assert values[-1, 3] == pytest.approx(1e-6 * values[0, 0], rel=1e-4)
    for name, sign in (("negative", 1), ("positive", -1)):
        profile = segment["raw_profiles"][name]
        assert len(profile["r_nodes_m"]) == 6
        assert len(profile["r_edges_m"]) == 7
        assert profile["z_nodes_m"] == pytest.approx([0.0125, 0.0375, 0.0625, 0.0875])
        q = np.asarray(profile["reaction_current_A"])
        j = np.asarray(profile["reaction_current_density_A_m2"])
        area = 3 * data[name]["active_volume_m3"] / data[name]["radius_m"] / 4
        assert q == pytest.approx(j * area, abs=1e-9)
        assert q.sum(axis=1) == pytest.approx(sign * values[:, 1], abs=1e-8)
        assert np.asarray(profile["theta"]).shape == (3, 6, 4)
        assert np.asarray(profile["average_fraction"]).shape == (3, 4)
        surface = np.asarray(profile["surface_fraction"])
        j0 = np.asarray(profile["exchange_current_density_A_m2"])
        eta = np.asarray(profile["kinetic_overpotential_V"])
        assert j0 == pytest.approx(
            data[name]["exchange_current_scale_A_m2"] * np.sqrt(surface * (1 - surface)),
            abs=1e-10,
        )
        assert j == pytest.approx(
            2
            * j0
            * np.sinh(trace["faraday_C_mol"] * eta / (2 * 8.314462618 * data["temperature_K"])),
            abs=1e-8,
        )
        u0, slope = data[name]["ocv"]
        assert np.asarray(profile["open_circuit_potential_V"]) == pytest.approx(
            u0 + slope * surface, abs=1e-10
        )
        state_info = next(
            item
            for item in trace["source_build"]["states"]
            if item["name"] == f"X-averaged {name} particle concentration [mol.m-3]"
        )
        first, last = state_info["slices"][0]
        actual = np.asarray(segment["source_state_vectors"])[:, first:last]
        reconstructed = (
            (actual * state_info["scale"] + state_info["reference"])
            .reshape(3, 4, 6)
            .transpose(0, 2, 1)
        )
        assert reconstructed == pytest.approx(
            np.asarray(profile["concentration_mol_m3"]), abs=1e-10
        )
    qn = np.asarray(segment["raw_profiles"]["negative"]["reaction_current_A"])
    qp = np.asarray(segment["raw_profiles"]["positive"]["reaction_current_A"])
    assert qn == pytest.approx(-qp, abs=1e-8)
    assert qn == pytest.approx(np.asarray(segment["source_fields"]["through_current_A"]), abs=1e-8)


@pytest.mark.parametrize("name", ("CC05", "CC06"))
def test_source_authored_z_profile_is_actual_initial_state(name):
    from battery_collector_spm_cases import _z_fractions
    from battery_collector_spm_traces import source_time

    data = _short_case(name, target=0.0)
    trace = source_time(data, {"z": 4, "r": 4}, LAYOUTS["same_end"])
    for electrode in ("negative", "positive"):
        profile = trace["segments"][0]["raw_profiles"][electrode]
        theta = np.asarray(profile["theta"])[0]
        expected = _z_fractions(data["initial_z_fractions"][electrode], 4)
        assert theta == pytest.approx(np.tile(expected, (4, 1)), abs=1e-12)


@pytest.mark.parametrize(
    "target,limit,expected_mode",
    ((3.78, 1.0, "CV"), (3.4, 0.001, "positive-limit"), (4.1, 0.001, "negative-limit")),
)
def test_source_finite_voltage_control_has_real_algebraic_closure(target, limit, expected_mode):
    from battery_collector_spm_traces import source_time

    data = _short_case(target=target, unit="V")
    data["maximum_current_A"] = limit
    trace = source_time(data, {"z": 4, "r": 4}, LAYOUTS["same_end"])
    values = np.asarray(trace["segments"][0]["values"])
    v, current = values[:, 0], values[:, 1]
    residual = current - np.clip(current + limit * (v - target), -limit, limit)
    assert residual == pytest.approx(np.zeros(3), abs=1e-8)
    if expected_mode == "CV":
        assert v == pytest.approx(np.full(3, target), abs=1e-7)
        assert np.max(np.abs(current)) < limit
    else:
        assert current == pytest.approx(
            np.full(3, limit if expected_mode == "positive-limit" else -limit), abs=1e-8
        )


def test_source_initial_voltage_rejection_keeps_consistent_state_and_events():
    from battery_collector_spm_traces import source_time

    with pytest.raises(Exception, match="Minimum voltage") as caught:
        source_time(_short_case("CC01", target=0.3), GRIDS["coarse"], LAYOUTS["opposite"])
    diagnostic = caught.value.partial_history["consistent_initialization"]
    assert diagnostic["time_s"] == 0
    assert diagnostic["variables"]["Voltage [V]"][0] < 2.5
    assert diagnostic["events"]["Minimum voltage [V]"][0] < 0
    assert np.all(np.isfinite(diagnostic["state_vector"]))
    assert diagnostic["algebraic_residual"]
    assert caught.value.partial_history["completed_segments"] == []


def test_paired_failure_retains_completed_trace_and_inner_context(monkeypatch):
    import battery_collector_spm_traces as traces

    native = {"implementation": "native", "segments": [{"raw": [1, 2, 3]}]}
    inner = {"stage": "extract", "completed_segments": [{"source_raw": [4, 5]}]}

    def fail(*args):
        error = ValueError("fixture extract failure")
        error.partial_history = inner
        raise error

    monkeypatch.setattr(traces, "native_time", lambda *args: native)
    monkeypatch.setattr(traces, "source_time", fail)
    with pytest.raises(ValueError, match="fixture extract") as caught:
        traces.paired_time(case_data("CC02"), GRIDS["coarse"], LAYOUTS["opposite"])
    assert caught.value.partial_history["completed_traces"]["native"] == native
    assert caught.value.partial_history["failed_implementation_context"] == inner


def test_source_extract_failure_keeps_completed_raw_solution(monkeypatch):
    import battery_collector_spm_traces as traces

    def fail(*args):
        raise ValueError("fixture postprocessing failure")

    monkeypatch.setattr(traces, "_source_segment", fail)
    with pytest.raises(ValueError, match="postprocessing") as caught:
        traces.source_time(_short_case(), {"z": 4, "r": 4}, LAYOUTS["same_end"])
    context = caught.value.partial_history
    assert context["stage"] == "extract"
    assert context["pending_solution"]["termination"] == "final time"
    assert len(context["pending_solution"]["times_s"]) == 3
    assert len(context["pending_solution"]["source_state_vectors"]) == 3
    assert context["pending_solution"]["inputs"] == {"Current function [A]": 0.01}


def test_source_cc_rest_reverse_retains_global_inventory_qw_continuity():
    from battery_collector_spm_traces import source_time

    data = _short_case()
    data["waveform"] = ((1e-4, 0.01, "A"), (1e-4, 0.0, "A"), (1e-4, -0.01, "A"))
    trace = source_time(data, {"z": 4, "r": 6}, LAYOUTS["same_end"])
    segments = trace["segments"]
    assert len(segments) == 3
    for previous, following in pairwise(segments):
        assert following["times_s"][0] == pytest.approx(previous["times_s"][-1], abs=1e-15)
        assert following["values"][0][2:] == pytest.approx(previous["values"][-1][2:], abs=1e-12)
        for name in ("negative", "positive"):
            assert np.asarray(following["raw_profiles"][name]["shell_amounts_mol"])[
                0
            ] == pytest.approx(
                np.asarray(previous["raw_profiles"][name]["shell_amounts_mol"])[-1], abs=1e-12
            )
    assert segments[-1]["values"][-1][2] == pytest.approx(0.0, abs=1e-10)
    for segment, (_, current, _) in zip(segments, data["waveform"], strict=True):
        rows = np.asarray(segment["values"])
        assert rows[-1, 2] - rows[0, 2] == pytest.approx(current * 1e-4, abs=1e-10)
        assert rows[-1, 3] - rows[0, 3] == pytest.approx(current * 1e-4 * rows[0, 0], abs=1e-10)


def test_fixed_matrix_is_bound_before_any_scientific_run():
    assert len(NAMES) == 12
    assert len(HISTORY_NAMES) == 10
    assert len(REFUSAL_NAMES) == 2
    assert EXPECTED_ROUTES == 48
    assert tuple(GRIDS) == ("coarse", "fine")
    assert tuple(LAYOUTS) == ("opposite", "same_end")
    assert all(case_data(name)["samples_per_segment"] == 31 for name in NAMES)


@pytest.mark.parametrize("grid_name", ("coarse", "fine"))
@pytest.mark.parametrize("layout_name", ("opposite", "same_end"))
def test_actual_pybamm_source_builds_and_discretises(grid_name, layout_name):
    data = case_data("CC01")
    built = build_reference(data, GRIDS[grid_name], LAYOUTS[layout_name])
    snapshot = source_contract_snapshot(built)
    assert snapshot["pybamm_version"] == "25.4.2"
    assert snapshot["casadi_version"] == "3.6.7"
    assert snapshot["rhs_count"] == 6
    assert snapshot["algebraic_count"] == 4
    assert snapshot["initial_condition_count"] == 10
    assert snapshot["tabs"]["positive tab"] == "right"
    assert snapshot["tabs"]["negative tab"] == LAYOUTS[layout_name]["negative_tab"]
    assert len(snapshot["z_nodes_m"]) == GRIDS[grid_name]["z"]
    assert len(snapshot["z_edges_m"]) == GRIDS[grid_name]["z"] + 1
    assert snapshot["variables"]["Positive current collector potential [V]"]["shape"] == [
        GRIDS[grid_name]["z"],
        1,
    ]
    assert snapshot["variables"]["Negative electrode current density [A.m-2]"]["shape"] == [
        GRIDS[grid_name]["z"],
        1,
    ]
    assert snapshot["variables"]["Voltage [V]"]["shape"] == [1, 1]
    assert snapshot["source_boundary_voltage"]["boundary_extrapolation"].startswith("finite-volume")


def test_source_events_and_local_field_contract_are_retained():
    built = build_reference(case_data("CC01"), GRIDS["coarse"], LAYOUTS["opposite"])
    snapshot = source_contract_snapshot(built)
    names = {event["name"] for event in snapshot["events"]}
    assert {"Minimum voltage [V]", "Maximum voltage [V]"} <= names
    for field in (
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Current collector current density [A.m-2]",
        "Negative electrode current density [A.m-2]",
        "Positive electrode current density [A.m-2]",
        "Local voltage [V]",
    ):
        assert snapshot["variables"][field]["domain"]


def test_refusal_case_is_authored_and_not_silently_run_as_a_history():
    data = case_data("CC12")
    assert data["expected"] == "refusals"
    assert len(data["refusal_cases"]) == 8
    assert data["waveform"] == ((0.0, 0.0, "A"),)


def test_route_plan_preserves_the_fixed_40_history_8_refusal_denominator():
    history_routes = sum(len(route_plan(name)) for name in HISTORY_NAMES)
    refusal_routes = sum(len(route_plan(name)) for name in REFUSAL_NAMES)
    assert history_routes == 40
    assert refusal_routes == 8
    assert history_routes + refusal_routes == EXPECTED_ROUTES
    for name in HISTORY_NAMES:
        assert {item[1] for item in route_plan(name)} == {"history"}
    for name in REFUSAL_NAMES:
        assert {item[1] for item in route_plan(name)} == {"refusal"}


def test_archive_is_complete_lossless_json(tmp_path):
    import gzip
    import json

    from run_battery_collector_spm_block import archive_verified, compact_write

    raw, gz = tmp_path / "raw.json", tmp_path / "raw.json.gz"
    value = {"complete": [1.0, 2.0], "failure": {"error": "kept"}, "unicode": "锂"}
    compact_write(raw, value)
    info = archive_verified(raw, gz)
    assert info["byte_identity"] is True
    assert gzip.decompress(gz.read_bytes()) == raw.read_bytes()
    assert json.loads(gzip.decompress(gz.read_bytes())) == value


@pytest.mark.parametrize("unit", REFUSAL_NAMES)
def test_refusal_route_executes_all_authored_subcases_without_integration(unit):
    route = refusal(case_data(unit), GRIDS["coarse"], LAYOUTS["opposite"])
    assert route["implementation"] == "refusal"
    assert route["rejected"] is True
    assert [item["name"] for item in route["refusals"]] == list(case_data(unit)["refusal_cases"])
    assert all(item["rejected"] and item["state_unchanged"] for item in route["refusals"])
