"""Source build/short contracts only; never counted as the six-unit block."""

from copy import deepcopy

import numpy as np
import pytest
from battery_parameters45_cases import CASE_IDS, RESPONSE_ORDER, case_data
from battery_parameters45_pybamm import (
    build_reference,
    finite_control_residual,
    initial_state_record,
    reference_time,
    source_contract,
)


@pytest.mark.parametrize("identity", CASE_IDS)
def test_released_source_builds_without_any_material_parameter_update(identity):
    built = build_reference(case_data(identity), 4)
    contract = source_contract(built)
    manifest = contract["source_parameter_manifest"]
    assert len(manifest) == 94
    assert sum(v["kind"] == "number" for v in manifest.values()) == 86
    assert sum(v["kind"] == "function" for v in manifest.values()) == 6
    assert contract["parameter_updates"] == {}
    assert manifest["Nominal cell capacity [A.h]"]["value"] == 5
    assert manifest["Current function [A]"]["value"] == 5
    assert contract["external_source_adapter"]["maximum_current_A"] == 10
    assert contract["work_adapter"]["signed"] is True
    assert contract["work_adapter"]["source_state"] == "Discharge energy [W.h]"
    assert contract["rhs_size"] == 2 * 4 + 4
    assert contract["algebraic_size"] == 1
    assert {tuple(x.values()) for x in contract["original_events"]} <= {
        tuple(x.values()) for x in contract["all_events"]}
    assert len(contract["all_events"]) == len(contract["original_events"]) + 4
    for electrode, radius in (("negative", 5.86e-6), ("positive", 5.22e-6)):
        mesh = contract["mesh"][electrode]
        assert mesh["class"] == "Uniform1DSubMesh"
        assert mesh["cells"] == 4
        assert mesh["edges_m"][0] == 0
        assert mesh["edges_m"][-1] == radius
        assert np.diff(mesh["edges_m"]) == pytest.approx(np.full(4, radius / 4))


@pytest.mark.parametrize("identity,stage_index", (("P01", 0), ("P02", 0), ("P06", -1)))
def test_true_t0_uses_original_initial_inventory_and_source_u_barrier(identity, stage_index):
    data = case_data(identity)
    built = build_reference(data, 4)
    stage = data["stages"][stage_index]
    before = initial_state_record(built, stage)
    row = dict(zip(RESPONSE_ORDER, before["row"], strict=True))
    assert before["time_s"] == 0
    assert row["charge_C"] == row["work_J"] == 0
    assert before["differential_state_unchanged_by_boundary_root"]
    assert max(abs(x) for x in before["source_algebraic"]) < 1e-8
    assert row["negative_Li_mol"] == pytest.approx(.065 * 1.58 * 8.52e-5 * .75 * 29866)
    assert row["positive_Li_mol"] == pytest.approx(.065 * 1.58 * 7.56e-5 * .665 * 17038)
    parameters = built["parameter_values"]
    for electrode in ("negative", "positive"):
        theta = row[electrode + "_surface_fraction"]
        raw = parameters[electrode.capitalize() + " electrode OCP [V]"](theta)
        expected = float(np.asarray(raw).ravel()[0]) + 1e-6 * (
            1 / theta + 1 / (theta - 1))
        assert row[electrode + "_ocp_V"] == pytest.approx(expected, abs=1e-12)
    if stage["unit"] == "A":
        assert row["current_A"] == pytest.approx(stage["target"], abs=1e-9)
    else:
        assert abs(row["current_A"]) < data["maximum_current_A"]
        assert row["voltage_V"] == pytest.approx(stage["target"], abs=1e-8)


@pytest.mark.parametrize("current,voltage,target", ((10, 4.2, 4), (-10, 3.8, 4), (2, 4, 4)))
def test_finite_source_equation_keeps_both_limits_and_the_cv_interior(current, voltage, target):
    assert finite_control_residual(current, voltage, voltage_mode=1, current_target=0,
                                   voltage_target=target, maximum_current_A=10, gain_A_V=1) == 0
    assert finite_control_residual(11, 4, voltage_mode=1, current_target=0,
                                   voltage_target=4, maximum_current_A=10, gain_A_V=1) != 0


@pytest.mark.parametrize("target,current", ((3., 10.), (5., -10.)))
def test_actual_source_boundary_root_obeys_both_hardware_limits(target, current):
    data = case_data("P06")
    built = build_reference(data, 4)
    stage = dict(data["stages"][-1], target=target)
    root = initial_state_record(built, stage)
    assert root["row"][1] == pytest.approx(current, abs=1e-8)
    assert max(abs(x) for x in root["source_algebraic"]) < 1e-9
    assert root["differential_state_unchanged_by_boundary_root"]
    # These are algebraic hardware contracts, not valid dynamic histories.
    assert root["row"][2:4] == [0, 0]


def test_true_initial_voltage_event_keeps_failed_boundary_without_advancing():
    from pybamm import SolverError

    data = case_data("P04")
    data["purpose"] = "true source initial-event contract, not qualification"
    data["stages"] = ({"duration_s": .01, "target": -5., "unit": "A", "samples": 2,
                       "expected": "maximum-voltage-at-initial-state"},)
    with pytest.raises(SolverError, match="Maximum voltage") as caught:
        reference_time(data, 4)
    partial = caught.value.partial_history
    assert partial["segments"] == []
    assert partial["last_completed_time_s"] == 0
    assert partial["failed_boundary_t0"]["time_s"] == 0
    assert partial["failed_boundary_t0"]["row"][0] > 4.2
    assert partial["failed_boundary_t0"]["row"][2:4] == [0, 0]
    assert partial["failed_boundary_t0"]["differential_state_unchanged_by_boundary_root"]


def test_short_actual_source_retains_full_states_and_signed_energy_without_extra_observer():
    data = case_data("P02")
    data["purpose"] = "short functional contract, not qualification"
    data["stages"] = ({"duration_s": .05, "target": 1., "unit": "A", "samples": 3,
                       "expected": "complete"},)
    trace = reference_time(data, 4)
    segment = trace["segments"][0]
    assert segment["termination"] == "final time"
    assert segment["actual_start_time_s"] == 0
    assert segment["actual_end_time_s"] == .05
    assert segment["boundary_t0"]["time_s"] == 0
    assert len(segment["records"]) == 3
    assert len(segment["records"][0]["source_y"]) == 13
    rows = np.asarray([r["row"] for r in segment["records"]])
    assert rows[0, 2:4] == pytest.approx([0, 0], abs=1e-12)
    assert rows[-1, 2] == pytest.approx(.05, abs=1e-10)
    power = rows[:, 0] * rows[:, 1]
    independent_trapezoid = np.dot(np.diff(segment["original_reported_times_s"]),
                                   (power[1:] + power[:-1]) / 2)
    assert rows[-1, 3] == pytest.approx(independent_trapezoid, abs=2e-7)
    assert np.asarray(segment["original_all_ys"][0]).shape[0] == 13
    assert all(len(r["profiles"]["negative"]["fraction"]) == 4
               for r in segment["records"])
    assert all(len(r["source_events"]) == len(trace["contract"]["all_events"])
               for r in segment["records"])


def test_short_cc_rest_cv_keeps_exact_boundary_state_and_unmodified_source_times():
    data = case_data("P06")
    data["purpose"] = "short functional contract, not qualification"
    data["stages"] = tuple({"duration_s": .05, "target": target, "unit": unit,
                            "samples": 3, "expected": "complete"}
                           for target, unit in ((1., "A"), (0., "A"), (4.05, "V")))
    trace = reference_time(data, 4)
    assert trace["actual_stages"] == 3
    for previous, following in zip(trace["segments"], trace["segments"][1:], strict=False):
        boundary = following["boundary_t0"]
        assert boundary["time_s"] == previous["actual_end_time_s"]
        assert boundary["differential_state_unchanged_by_boundary_root"]
        assert boundary["source_y"][:12] == previous["records"][-1]["source_y"][:12]
        assert following["original_reported_times_s"][0] >= boundary["time_s"]
        assert following["original_reported_times_s"][0] <= np.nextafter(boundary["time_s"], np.inf)
    last = trace["segments"][-1]
    for record in (last["boundary_t0"], *last["records"]):
        assert record["row"][0] == pytest.approx(4.05, abs=1e-8)
        assert abs(record["row"][1]) < 10
        assert max(abs(x) for x in record["source_algebraic"]) < 1e-9


def test_source_failure_preserves_actual_completed_prefix_and_failed_boundary(monkeypatch):
    data = case_data("P02")
    data["stages"] = (deepcopy(data["stages"][0]), deepcopy(data["stages"][0]))
    import battery_parameters45_pybamm as adapter

    calls = []
    original = adapter.initial_state_record

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("injected boundary failure before any second-stage integration")
        return original(*args, **kwargs)

    # A tiny development interval avoids executing the fixed 1800 s unit.
    for stage in data["stages"]:
        stage["duration_s"], stage["samples"] = .01, 2
    data["purpose"] = "short failure-retention contract, not qualification"
    monkeypatch.setattr(adapter, "initial_state_record", fail_second)
    with pytest.raises(RuntimeError, match="injected boundary") as caught:
        adapter.reference_time(data, 4)
    partial = caught.value.partial_history
    assert partial["failed_stage"] == 1
    assert len(partial["segments"]) == 1
    assert partial["segments"][0]["actual_end_time_s"] == .01
    assert partial["last_completed_time_s"] == .01
