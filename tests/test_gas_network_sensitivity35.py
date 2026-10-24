"""Finite-network numerical response and read-only world boundaries."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.gas_network_sensitivity import (
    GasSensitivityParameter,
    GasSensitivityPolicy,
    gas_network_trajectory_sensitivities,
)
from chemworld.physchem.gas_reactor_network import pack_gas_reactor_network_state
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_network_sensitivity import analyze_gas_network_world_state
from chemworld.world.gas_reactor_network import gas_reactor_network_world


@pytest.fixture(scope="module")
def cases():
    path = str(Path(__file__).parent / "reference")
    sys.path.insert(0, path)
    yield importlib.import_module("gas_network_sensitivity35_cases")
    sys.path.remove(path)


def run(cases, index=3, *, times=(0.0, 0.1, 0.3), policy=None, raw=None):
    raw = raw or cases.card(index)
    spec, network, device = cases.native(raw)
    state = pack_gas_reactor_network_state(spec, network, raw["amounts"])
    parameters = tuple(GasSensitivityParameter(*binding) for binding in cases.bindings(index))
    result = gas_network_trajectory_sensitivities(
        spec, network, state, parameters, times, thermal_device=device, policy=policy
    )
    return result


@pytest.mark.parametrize("index", (1, 2, 3, 6, 7, 8))
def test_actual_network_coefficients_match_independent_reference(cases, index):
    reference = importlib.import_module("gas_network_sensitivity35_reference")
    times = (0.0, 0.1, 0.3)
    result = run(cases, index, times=times)
    expected = reference.reference(cases.card(index), cases.bindings(index), times)
    assert result["status"] == "complete_finite_difference"
    for response, derivatives in zip(result["responses"], expected["derivatives"], strict=True):
        for row, slope in zip(response["records"], derivatives, strict=True):
            for key, value in row["coefficients"].items():
                assert value["fine_dy_dlnp"] == pytest.approx(
                    slope[key],
                    abs={"mol": 2e-7, "K": 0.02, "Pa": 0.2, "J": 0.003}[result["units"][key]],
                    rel=0.003,
                )
    assert result["work"]["reported_nfev"] > 0
    json.dumps(result, allow_nan=False)


def test_normalization_preserves_undefined_zero_and_signed_coefficients(cases):
    result = run(cases)
    row = result["responses"][0]["records"][0]
    assert row["coefficients"]["node:source:bath"]["normalized"] is None
    assert row["coefficients"]["node:source:amount:A"]["normalized"] == 0
    final = result["responses"][0]["records"][-1]["coefficients"]
    assert final["node:source:amount:A"]["fine_dy_dlnp"] < 0
    assert final["node:source:amount:B"]["fine_dy_dlnp"] > 0
    for binding, response in zip(result["parameters"], result["responses"], strict=True):
        for row in response["records"]:
            for value in row["coefficients"].values():
                assert value["dy_dp"] == value["fine_dy_dlnp"] / binding["baseline_value"]
                assert value["unit_dy_dp"]


def test_valve_outputs_do_not_certify_path_smoothness(cases):
    result = run(cases, 5)
    assert result["status"] == "piecewise_finite_difference_uncertified"
    assert all(
        row["status"] == "finite_difference_piecewise_uncertified"
        for row in result["responses"][0]["records"]
    )


def test_events_do_not_substitute_endpoint_for_requested_time(cases):
    raw = cases.card(2)
    raw["nodes"][0]["minimum_mol"] = 0.015
    result = run(cases, 2, times=(0.0, 4.0, 5.0, 6.0), raw=raw)
    assert result["status"] == "incomplete_finite_difference"
    assert result["responses"][0]["records"][1]["coefficients"] is not None
    assert all(row["coefficients"] is None for row in result["responses"][0]["records"][2:])
    assert all(candidate["stopping_event"] for candidate in result["trajectories"])
    assert result["trajectories"][0]["actual_end_time_s"] == pytest.approx(5, abs=2e-6)


def test_whole_analysis_budget_keeps_candidate_time_denominators(cases):
    result = run(cases, 3, policy=GasSensitivityPolicy(max_advances=1))
    assert result["work"]["planned_trajectories"] == 9
    assert result["work"]["planned_time_slots"] == 27
    assert result["work"]["attempted_advances"] == 1
    assert len(result["responses"]) == 2
    assert all(len(response["records"]) == 3 for response in result["responses"])
    assert result["status"] == "incomplete_finite_difference"


def test_readonly_selected_member_world_and_queue_refusal(cases):
    raw = cases.card(8)
    spec, network, _ = cases.native(raw)
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    composition = gas_reactor_network_world(
        spec,
        network,
        raw["amounts"],
        anchor_thermal_device=ThermalDeviceSpec(
            setpoint_K=900, minimum_setpoint_K=200, maximum_setpoint_K=1500, enabled=False
        ),
    )
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=7)
        runtime.execute({"operation": "wait", "duration_s": 0.25})
        runtime.execute({"operation": "select_vessel", "vessel": "source"})
        parameters = (GasSensitivityParameter("flow", "feed", "coefficient"),)
        before = runtime._state.to_dict()
        report = analyze_gas_network_world_state(
            runtime._state, composition, parameters, (0.0, 0.1)
        )
        assert runtime._state.to_dict() == before
        assert report["world_consumer"]["anchor_vessel"] == "reactor"
        assert report["start_time_s"] == 0.25
        runtime.execute({"operation": "select_vessel", "vessel": "reactor"})
        runtime.execute(
            {"operation": "queue_lab_heat", "duration_s": 1, "target_temperature_K": 900}
        )
        queued = runtime._state.to_dict()
        with pytest.raises(ValueError, match="scheduled-history"):
            analyze_gas_network_world_state(runtime._state, composition, parameters, (0.0, 0.1))
        assert runtime._state.to_dict() == queued
    finally:
        runtime.close()


@pytest.mark.parametrize("times", [(), (1, 0.5), (0, 0), (-1,), (float("nan"),), (True,)])
def test_invalid_schedules_are_refused(cases, times):
    with pytest.raises(ValueError):
        run(cases, times=times)


@pytest.mark.parametrize("step", [0, -0.1, 0.2, float("inf"), True])
def test_invalid_difference_step(step):
    with pytest.raises(ValueError):
        GasSensitivityPolicy(log_step=step)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_work_budget(budget):
    with pytest.raises(ValueError):
        GasSensitivityPolicy(max_advances=budget)


def test_zero_unknown_and_duplicate_parameters(cases):
    raw = cases.card(2)
    spec, network, _ = cases.native(raw)
    state = pack_gas_reactor_network_state(spec, network, raw["amounts"])
    flow = GasSensitivityParameter("flow", "feed", "coefficient")
    for parameters in ((flow, flow), (GasSensitivityParameter("flow", "missing", "coefficient"),)):
        with pytest.raises(ValueError):
            gas_network_trajectory_sensitivities(spec, network, state, parameters, (0.0, 0.1))
    zero = replace(spec, connections=(replace(spec.connections[0], coefficient=0),))
    with pytest.raises(ValueError, match="positive"):
        gas_network_trajectory_sensitivities(zero, network, state, (flow,), (0.0, 0.1))


def test_failed_endpoint_check_keeps_solver_work_without_double_count(cases, monkeypatch):
    import chemworld.physchem.gas_network_sensitivity as module

    original = module.advance_gas_reactor_network

    def incomplete(*args, **kwargs):
        result, diagnostic = original(*args, **kwargs)
        return replace(
            result, final_state=replace(result.final_state, time_s=result.final_state.time_s - 0.01)
        ), diagnostic

    monkeypatch.setattr(module, "advance_gas_reactor_network", incomplete)
    result = run(cases, 2, times=(0.0, 0.1))
    assert result["work"]["attempted_advances"] == 5
    assert result["work"]["failed_advances"] == 5
    assert result["work"]["completed_advances"] == 0
    assert result["work"]["reported_nfev"] > 0
    assert result["work"]["unknown_work_failures"] == 0
