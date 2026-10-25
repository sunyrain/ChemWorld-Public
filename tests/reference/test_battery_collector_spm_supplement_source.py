"""Small preparation contracts, not the supplement's scientific histories."""

from copy import deepcopy

import numpy as np
import pytest
from battery_collector_spm_cases import LAYOUTS, case_data
from battery_collector_spm_pybamm import build_reference as build_original
from battery_collector_spm_supplement_source import (
    EVENT_PREFIX,
    PHYSICAL_V,
    PHYSICAL_W,
    build_reference,
    source_contract_snapshot,
    source_time,
    source_uniform_time,
)

GRID = {"z": 4, "r": 6}


def fixture(*, mode="A", target=0.01, limit=1.0, duration=1e-4):
    data = case_data("CC02")
    data["waveform"] = ((duration, target, mode),)
    data["samples_per_segment"] = 3
    data["maximum_current_A"] = limit
    return data


def test_build_retains_source_equations_events_and_adds_real_work_state():
    data = fixture()
    original = build_original(data, GRID, LAYOUTS["opposite"])
    built = build_reference(data, GRID, LAYOUTS["opposite"])
    model = built["model"]
    before = {e.name for e in original["model"].events}
    after = {e.name for e in model.events}
    assert before <= after
    assert len(after - before) == 6
    assert {v.name for v in original["model"].rhs} | {PHYSICAL_W} == {v.name for v in model.rhs}
    snapshot = source_contract_snapshot(built)
    assert snapshot["rhs_size"] == original["model"].concatenated_rhs.size + 1
    assert snapshot["algebraic_size"] == original["model"].concatenated_algebraic.size
    assert set(snapshot["original_events"]) == before


@pytest.mark.parametrize("layout", LAYOUTS.values())
def test_physical_port_is_last_centre_minus_actual_neumann_drop(layout):
    data = fixture()
    trace = source_time(data, GRID, layout)
    segment = trace["segments"][0]
    raw = np.asarray(segment["values"])
    physical = np.asarray(segment["physical_values"])
    positive = np.asarray(segment["source_fields"]["Positive current collector potential [V]"])
    k = data["sigma_cc_S_m"] * data["collector_thickness_m"] * data["width_m"]
    assert physical[:, 0] == pytest.approx(
        positive[:, -1] - raw[:, 1] * data["height_m"] / (2 * GRID["z"] * k), abs=1e-12
    )
    assert raw[:, 0] == pytest.approx(
        positive[:, -1] + 0.5 * (positive[:, -1] - positive[:, -2]), abs=1e-12
    )
    assert np.max(abs(raw[:, 0] - physical[:, 0])) > 1e-6
    assert physical[0, 2:] == pytest.approx([0, 0])
    assert physical[-1, 3] == pytest.approx(0.01 * 1e-4 * physical[0, 0], rel=1e-5)
    assert raw[-1, 3] == pytest.approx(0.01 * 1e-4 * raw[0, 0], rel=1e-5)
    assert segment["independent_physical_tab_voltage_V"] is None


@pytest.mark.parametrize(
    "target,limit,branch", [(3.78, 1.0, "CV"), (3.40, 0.001, "positive"), (4.10, 0.001, "negative")]
)
def test_actual_cv_uses_physical_port_and_both_limits(target, limit, branch):
    trace = source_time(fixture(mode="V", target=target, limit=limit), GRID, LAYOUTS["same_end"])
    s = trace["segments"][0]
    v = np.asarray(s["physical_values"])
    assert s["controlled_port"] == "physical-tab"
    if branch == "CV":
        assert v[:, 0] == pytest.approx(target, abs=1e-8)
        assert np.max(abs(v[:, 1])) < limit
        assert v[:, 3] == pytest.approx(target * v[:, 2], abs=1e-9)
        assert np.max(abs(np.asarray(s["values"])[:, 0] - target)) > 1e-6
    else:
        sign = 1 if branch == "positive" else -1
        assert v[:, 1] == pytest.approx(sign * limit, abs=1e-8)
        assert np.all(sign * (v[:, 0] - target) > 0)


def test_added_work_is_an_unscaled_observer_without_feedback():
    data = fixture()
    trace = source_time(data, GRID, LAYOUTS["opposite"])
    built = build_reference(data, GRID, LAYOUTS["opposite"])
    model = built["model"]
    state_info = next(x for x in trace["source_build"]["states"] if x["name"] == PHYSICAL_W)
    assert state_info["scale"] == 1 and state_info["reference"] == 0
    start, stop = state_info["slices"][0]
    assert stop - start == 1
    y = np.asarray(trace["segments"][0]["source_state_vectors"])[0, :, None]
    inputs = {"Current function [A]": 0.01}

    def read(expression, state):
        return np.asarray(expression.evaluate(t=0, y=state, inputs=inputs)).reshape(-1)

    rhs = read(model.concatenated_rhs, y)
    expected = 0.01 * read(model.variables[PHYSICAL_V], y)[0]
    assert rhs[start] == pytest.approx(expected, abs=1e-12)
    changed = y.copy()
    changed[start] = 1234
    assert read(model.concatenated_rhs, changed) == pytest.approx(rhs, abs=0)
    assert read(model.concatenated_algebraic, changed) == pytest.approx(
        read(model.concatenated_algebraic, y), abs=0
    )
    assert read(model.variables["Voltage [V]"], changed) == pytest.approx(
        read(model.variables["Voltage [V]"], y), abs=0
    )
    check = source_contract_snapshot(built)["physical_work_no_feedback"]
    assert check["no_feedback"] is True
    assert check["physical_work_state_indices"] == [start]
    # Deliberately contaminate this in-memory model: the computed report must detect it.
    variable = next(v for v in model.rhs if v.name != PHYSICAL_W)
    model.rhs[variable] += model.variables[PHYSICAL_W]
    check = source_contract_snapshot(built)["physical_work_no_feedback"]
    assert check["no_feedback"] is False
    assert check["checked_expressions"]["rhs:" + variable.name] == [start]


def test_duplicate_named_events_keep_order_and_detect_feedback_in_either_copy():
    import pybamm

    built = build_reference(fixture(), GRID, LAYOUTS["opposite"])
    model = built["model"]
    original_events = tuple(model.events)
    # Two equal names are legal upstream: each initial-SOC Interpolant adds
    # its own pair of events. Event names are labels, not unique identities.
    model.events.extend(
        [pybamm.Event("same-name observer contract", pybamm.Scalar(1.0)) for _ in range(2)]
    )
    identities = [
        {"ordinal": i, "type": e.event_type.name, "name": e.name}
        for i, e in enumerate(model.events)
    ]
    keys = [f"event:{i}:{e.event_type.name}:{e.name}" for i, e in enumerate(model.events)]
    first = model.events[-2]
    model.events[-2] = pybamm.Event(first.name, model.variables[PHYSICAL_W])
    # Regression for the actual overwrite defect, before requiring the new
    # metadata: the old dictionary falsely reported this model feedback-free.
    assert source_contract_snapshot(built)["physical_work_no_feedback"]["no_feedback"] is False
    model.events[-2] = first
    clean = source_contract_snapshot(built)["physical_work_no_feedback"]
    assert clean["event_identities"] == identities
    assert [k for k in clean["checked_expressions"] if k.startswith("event:")] == keys
    assert all(clean["checked_expressions"][key] == [] for key in keys)
    assert clean["no_feedback"] is True
    for relative_index in (-1, -2):
        # Check the second duplicate explicitly, then the first (which the
        # old name-keyed dictionary silently overwrote with the clean second).
        slot = len(model.events) + relative_index
        saved = model.events[slot]
        model.events[slot] = pybamm.Event(saved.name, model.variables[PHYSICAL_W])
        contaminated = source_contract_snapshot(built)["physical_work_no_feedback"]
        assert contaminated["event_identities"] == identities
        assert contaminated["no_feedback"] is False
        assert contaminated["checked_expressions"][keys[slot]] == contaminated[
            "physical_work_state_indices"
        ]
        other = len(model.events) - (1 if relative_index == -2 else 2)
        assert contaminated["checked_expressions"][keys[other]] == []
        model.events[slot] = saved
    assert tuple(model.events[: len(original_events)]) == original_events


def test_cc_rest_reverse_preserves_full_differential_state_and_both_work_accounts():
    data = fixture()
    data["waveform"] = ((1e-4, 0.01, "A"), (1e-4, 0.0, "A"), (1e-4, -0.01, "A"))
    trace = source_time(data, GRID, LAYOUTS["opposite"])
    assert len(trace["segments"]) == 3
    n = trace["source_build"]["rhs_size"]
    for a, b in zip(trace["segments"], trace["segments"][1:], strict=False):
        assert np.asarray(a["source_state_vectors"])[-1, :n] == pytest.approx(
            np.asarray(b["source_state_vectors"])[0, :n], abs=1e-11
        )
        assert np.asarray(a["values"])[-1, 2:] == pytest.approx(
            np.asarray(b["values"])[0, 2:], abs=1e-11
        )
        assert np.asarray(a["physical_values"])[-1, 2:] == pytest.approx(
            np.asarray(b["physical_values"])[0, 2:], abs=1e-11
        )
    assert trace["segments"][-1]["values"][-1][2] == pytest.approx(0, abs=1e-10)


def test_source_original_and_added_events_remain_actual_at_endpoint():
    trace = source_time(fixture(), GRID, LAYOUTS["opposite"])
    events = trace["segments"][0]["source_event_values_at_endpoint"]
    assert "Minimum voltage [V]" in events
    assert EVENT_PREFIX + "minimum voltage" in events
    assert EVENT_PREFIX + "negative minimum surface" in events
    raw = np.asarray(trace["segments"][0]["raw_profiles"]["negative"]["surface_fraction"])
    assert events[EVENT_PREFIX + "negative minimum surface"] == pytest.approx(
        [raw[-1].min() - 0.01]
    )


def test_illegal_initial_voltage_preserves_source_and_physical_rejection_context():
    data = fixture()
    data["minimum_voltage_V"] = 4.0
    with pytest.raises(Exception) as caught:
        source_time(data, GRID, LAYOUTS["opposite"])
    context = caught.value.partial_history
    initial = context["consistent_initialization"]
    assert PHYSICAL_V in initial["variables"] and PHYSICAL_W in initial["variables"]
    assert EVENT_PREFIX + "minimum voltage" in initial["events"]
    assert context["completed_segments"] == []


def test_uniform_baseline_is_actual_dim0_spm_with_original_surface_and_work():
    trace = source_uniform_time(fixture(), GRID)
    assert trace["source_build"]["uniform"] is True
    assert trace["grid"] == {"z": 1, "r": 6}
    assert trace["requested_grid"] == GRID
    assert trace["actual_grid"] == {"z": 1, "r": 6}
    s = trace["segments"][0]
    assert np.asarray(s["physical_values"]) == pytest.approx(np.asarray(s["values"]), abs=1e-10)
    for name, sign in [("negative", 1), ("positive", -1)]:
        p = s["raw_profiles"][name]
        assert np.asarray(p["theta"]).shape == (3, 6, 1)
        assert np.asarray(p["surface_fraction"]).shape == (3, 1)
        assert np.asarray(p["reaction_current_A"])[:, 0] == pytest.approx(sign * 0.01, abs=1e-8)


def test_uniform_cannot_silently_average_nonuniform_initial_soc():
    with pytest.raises(ValueError, match="cannot average"):
        source_uniform_time(case_data("CC05"), GRID)


def test_series_resistance_scope_is_explicit_and_input_not_mutated():
    data = fixture()
    data["series_resistance_ohm"] = 0.1
    before = deepcopy(data)
    with pytest.raises(ValueError, match="zero series"):
        source_time(data, GRID, LAYOUTS["opposite"])
    assert data == before


def test_finite_surface_event_is_true_raw_source_surface_with_original_events_retained():
    # Small, expressly functional fixture. It is not an authored qualification unit.
    data = fixture(duration=0.5)
    data["sigma_cc_S_m"] = 1e8
    for name in ("negative", "positive"):
        data[name]["active_volume_m3"] /= 100
        data[name]["diffusivity_m2_s"] = 1e-11
        data[name]["exchange_current_scale_A_m2"] = 40
    data["negative"]["initial_fraction"] = 0.0102
    data["initial_z_fractions"]["negative"] = (0.0102,) * 4
    trace = source_time(data, GRID, LAYOUTS["opposite"])
    s = trace["segments"][0]
    assert s["outcome"] == "stopped"
    assert 0 < s["end_time_s"] < 0.5
    assert EVENT_PREFIX + "negative minimum surface" in s["termination"]
    surface = np.asarray(s["raw_profiles"]["negative"]["surface_fraction"])
    assert surface[0].min() > 0.01
    assert surface[-1].min() == pytest.approx(0.01, abs=1e-7)
    assert "Minimum voltage [V]" in s["source_event_values_at_endpoint"]


def test_failed_later_stage_preserves_completed_history_and_current_context():
    data = fixture()
    data["waveform"] = ((1e-4, 0.01, "A"), (1e-4, 10.0, "A"))
    with pytest.raises(Exception) as caught:
        source_time(data, GRID, LAYOUTS["opposite"])
    context = caught.value.partial_history
    assert len(context["completed_segments"]) == 1
    assert context["active_segment"]["index"] == 1
    assert context["completed_segments"][0]["physical_work_J"][-1] > 0
    assert context["consistent_initialization"]["variables"][PHYSICAL_V][0] < 2.5


def test_finite_physical_voltage_event_keeps_the_distinct_source_event_value():
    data = fixture(duration=3.0, target=0.03)
    data["sigma_cc_S_m"] = 1e8
    data["minimum_voltage_V"] = 3.71
    for name in ("negative", "positive"):
        data[name]["active_volume_m3"] /= 100
        data[name]["diffusivity_m2_s"] = 1e-11
        data[name]["exchange_current_scale_A_m2"] = 40
    trace = source_time(data, GRID, LAYOUTS["opposite"])
    segment = trace["segments"][0]
    assert segment["outcome"] == "stopped"
    assert 0 < segment["end_time_s"] < 3.0
    assert EVENT_PREFIX + "minimum voltage" in segment["termination"]
    physical = np.asarray(segment["physical_values"])
    original = np.asarray(segment["values"])
    assert physical[0, 0] > 3.71
    assert physical[-1, 0] == pytest.approx(3.71, abs=1e-7)
    assert original[-1, 0] > physical[-1, 0]
    assert segment["source_event_values_at_endpoint"]["Minimum voltage [V]"][0] > 0
