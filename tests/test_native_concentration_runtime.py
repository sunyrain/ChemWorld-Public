"""Real concentration transactions: ports, clocks, continuation and ownership."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.native_concentration_research import (
    initial_concentration_state,
    synthetic_concentration_spec,
)
from chemworld.physchem.concentration_units import simulate_vacuum_concentration
from chemworld.physchem.coupled_concentration import CoupledConcentrationRequest
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.concentration import read_concentration, validate_outlet_execution
from chemworld.world.concentration_world import concentration_world_spec


def execute(session, **action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", (
        result.info.get("rollback_reason"),
        result.info.get("outcome_diagnostics"),
    )
    return result


def make_session(spec=None):
    spec = synthetic_concentration_spec(30000) if spec is None else spec
    session = ChemWorldRuntime(
        composition=concentration_world_spec(spec, initial_concentration_state(spec))
    )
    session.reset(seed=21)
    return spec, session


def test_native_and_standalone_use_the_same_actual_contribution_and_owned_state():
    spec, session = make_session()
    try:
        result = simulate_vacuum_concentration(
            CoupledConcentrationRequest(
                spec,
                read_concentration(session._state, spec),
                5,
                ThermalDeviceSpec(maximum_power_W=10, setpoint_K=430),
            )
        )
        execute(session, operation="concentrate", duration_s=5)
        state = session._state
        current = read_concentration(state, spec)
        for actual, expected in zip(
            current.amounts_mol, result.final_state.amounts_mol, strict=True
        ):
            assert actual == pytest.approx(expected, abs=1e-12)
        assert current.enthalpies_J == pytest.approx(result.final_state.enthalpies_J, abs=1e-8)
        assert state.ledger.time_s == 5
        assert state.ledger.energy_jacket_J == pytest.approx(result.final_state.heater_energy_J)
        assert all(set(s.values) == {"enthalpy_J"} for s in state.state_slots.slots)
        assert "state" not in state.metadata["last_concentration_advance"]
        assert current.amounts_mol[0]["target"] == 0.002
        assert current.outlet_amounts_mol["light"] > 0
        assert (
            sum(p.species_amounts_mol["light"] for p in state.phases.phases.values())
            == (state.species_amounts["light"])
        )
    finally:
        session.close()


def test_measure_wait_and_exact_snapshot_replay_consume_each_boundary_once():
    spec, session = make_session()
    other = None
    try:
        execute(session, operation="concentrate", duration_s=5)
        other = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(session.snapshot().to_dict())))
        for action in (
            {"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600},
            {"operation": "measure", "instrument": "phase_state"},
        ):
            execute(session, **action)
            execute(other, **action)
        assert session._state.to_dict() == other._state.to_dict()
        assert session._state.ledger.time_s == 8
        assert session.observe().channels[0].sampled_at.time_s == 8
        inventory = read_concentration(session._state, spec)
        history = session._state.metadata["removed_phase_inventory_history"]
        assert [r["operation"] for r in history] == ["concentrate", "wait", "measure"]
        for key in spec.properties.component_ids:
            assert sum(
                r["inventories"]["concentrator_vent"]["species_amounts_mol"][key] for r in history
            ) == pytest.approx(inventory.outlet_amounts_mol[key])
    finally:
        session.close()
        if other is not None:
            other.close()


def test_collector_root_commits_actual_time_then_passive_acquisition_is_valid():
    spec = replace(synthetic_concentration_spec(30000), collector_capacity_mol=1e-4)
    _, session = make_session(spec)
    try:
        result = execute(session, operation="concentrate", duration_s=30)
        assert result.info["operation_outcome"] == "stopped"
        assert result.info["outcome_diagnostics"]["stopping_event"] == "collector_capacity"
        assert 0 < session._state.ledger.time_s < 30
        current = read_concentration(session._state, spec)
        assert sum(current.amounts_mol[1].values()) == pytest.approx(1e-4, abs=1e-12)
        assert not session._state.metadata["last_concentration_advance"]["endpoint_met"]
        before = session._state
        execute(session, operation="measure", instrument="phase_state")
        assert session._state.ledger.time_s == pytest.approx(before.ledger.time_s + 1)
        assert session._state.species_amounts == before.species_amounts
        assert session._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J
    finally:
        session.close()


def test_material_reset_retains_apparatus_heat_and_replaces_fluid_and_outlet_counters():
    spec, session = make_session()
    try:
        execute(session, operation="concentrate", duration_s=5)
        before = session._state
        body = sum(
            s.values["enthalpy_J"]
            for s in before.state_slots.slots
            if s.spec.owner_kind == "vessel"
        )
        template = concentration_world_spec(spec, initial_concentration_state(spec))
        fresh = template.physics.initial_states["concentrator"]
        fluid = sum(
            s.values["enthalpy_J"]
            for s in fresh.state_slots.slots
            if s.spec.owner_kind == "material"
        )
        session.reset(options={"scope": "material", "vessel_ids": ["concentrator"]})
        state = session._state
        assert state.ledger == before.ledger
        assert state.equipment == before.equipment
        assert state.species_amounts == pytest.approx(fresh.species_amounts)
        assert sum(s.values["enthalpy_J"] for s in state.state_slots.slots) == pytest.approx(
            body + fluid, abs=1e-7
        )
        assert not read_concentration(state, spec).outlet_amounts_mol
        assert "removed_phase_inventory_history" not in state.metadata
    finally:
        session.close()


def test_solver_failure_rolls_back_material_enthalpy_time_and_receipts(monkeypatch):
    _, session = make_session()
    try:
        before = session._state

        def fail(*args, **kwargs):
            raise RuntimeError("injected finite concentration failure")

        monkeypatch.setattr("chemworld.runtime.concentration.advance_coupled_concentration", fail)
        result = session.execute({"operation": "concentrate", "duration_s": 5})
        assert result.info["transaction_status"] == "rolled_back"
        assert result.info["operation_outcome"] == "numerical_failure"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        assert session._state.ledger.time_s == before.ledger.time_s
        assert "removed_phase_inventory_history" not in session._state.metadata
    finally:
        session.close()


def test_native_outlet_producer_rejects_a_forged_amount():
    spec, session = make_session()
    try:
        before = session._state
        execute(session, operation="concentrate", duration_s=5)
        receipt = json.loads(
            json.dumps(session._state.metadata["removed_phase_inventory_history"][-1])
        )
        receipt["inventories"]["concentrator_vent"]["species_amounts_mol"]["light"] *= 2
        with pytest.raises(ValueError, match="actual phase transport"):
            validate_outlet_execution(before, session._state, spec, receipt)
    finally:
        session.close()
