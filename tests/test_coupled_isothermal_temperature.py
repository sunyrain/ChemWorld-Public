"""Fixed external temperature survives solver scaling in every consumer view."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.examples.gas_network_sensitivity import example_inputs
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.gas_network_sensitivity import GasSensitivityParameter
from chemworld.physchem.solver_backend import ODESolverPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_network_sensitivity import analyze_gas_network_world_state
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


@pytest.mark.parametrize("method", ("BDF", "Radau"))
@pytest.mark.parametrize("numerical_form", ("ode", "residual"))
def test_fixed_temperature_is_exact_in_rates_events_and_snapshots(method, numerical_form):
    temperature = 900.0
    assert temperature / 350 * 350 != temperature
    seen = {"rates": [], "events": []}

    def rates(state):
        seen["rates"].append(state.temperature_K)
        assert state.temperature_K == temperature
        rate = 0.4 * state.amounts_mol["A"]
        return ProcessContribution({"A": -rate, "B": rate})

    def event(state):
        seen["events"].append(state.temperature_K)
        assert state.temperature_K == temperature
        return 0.035 - state.time_s

    system = CoupledSystem(
        ConstantCvThermodynamics(
            {"A": SpeciesInternalEnergy(0, 10), "B": SpeciesInternalEnergy(-1000, 10)},
            1,
        ),
        (CoupledProcess("conversion", rates),),
        events=(CoupledEvent("fixed-time-stop", event),),
        isothermal=True,
        numerical_form=numerical_form,
        policy=ODESolverPolicy("exact-isothermal", method=method),
    )
    start = CoupledState({"A": 0.01, "B": 0}, temperature, 1)
    result = system.advance(start, 0.1, evaluation_times_s=(0, 0.02, 0.1))
    assert all(seen.values())
    assert result.stopping_event == "fixed-time-stop"
    assert result.actual_duration_s == pytest.approx(0.035, abs=1e-10)
    assert all(point.temperature_K == temperature for point in result.trajectory)
    assert result.final_state.amounts_mol["A"] == pytest.approx(0.01 * exp(-0.4 * 0.035), rel=2e-7)
    assert result.final_state.boundary_heat_J < 0
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-8)
    assert start.temperature_K == temperature


def test_actual_900k_gas_world_and_snapshot_continue_without_temperature_rollback():
    spec, network, amounts = example_inputs()
    composition = gas_reactor_network_world(replace(spec, temperature_K=900), network, amounts)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=7)
        first = runtime.execute({"operation": "wait", "duration_s": 0.2})
        assert first.info["transaction_status"] == "committed"
        assert first.info["actual_duration_s"] == 0.2
        assert all(local.temperature_K == 900 for local in local_states(runtime._state).values())
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for consumer in (runtime, restored):
            result = consumer.execute({"operation": "wait", "duration_s": 0.3})
            assert result.info["transaction_status"] == "committed"
            assert result.info["actual_duration_s"] == 0.3
        assert restored._state.to_dict() == runtime._state.to_dict()
        assert runtime._state.ledger.time_s == 0.5
        assert all(local.temperature_K == 900 for local in local_states(runtime._state).values())
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def test_35c_readonly_analysis_preserves_fixed_temperature_after_actual_wait():
    spec, network, amounts = example_inputs()
    composition = gas_reactor_network_world(replace(spec, temperature_K=900), network, amounts)
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=7)
        result = runtime.execute({"operation": "wait", "duration_s": 0.2})
        assert result.info["transaction_status"] == "committed"
        before = runtime._state.to_dict()
        report = analyze_gas_network_world_state(
            runtime._state,
            composition,
            (GasSensitivityParameter("reaction", network.reactions[0].reaction_id, "A"),),
            (0, 0.1, 0.3),
        )
        assert report["work"]["completed_advances"] == 10
        assert report["work"]["failed_advances"] == 0
        for candidate in report["trajectories"]:
            for row in candidate["records"]:
                assert row["values"]["node:source:temperature"] == 900
                assert row["values"]["node:receiver:temperature"] == 900
        assert runtime._state.to_dict() == before
        json.dumps(report, allow_nan=False)
    finally:
        runtime.close()
