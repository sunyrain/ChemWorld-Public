"""Native sorbent contact, finite liquid routing and retained carrier energy."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.native_sorbent_research import (
    synthetic_sorbent_spec,
    synthetic_sorbent_thermodynamics,
)
from chemworld.physchem.sorbent_kinetics import loading_field
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.thermal_material_routing import _energy
from chemworld.world.sorbent_world import sorbent_world_spec


def session_for(*, mobile=False):
    model = synthetic_sorbent_spec()
    world = sorbent_world_spec(
        model, synthetic_sorbent_thermodynamics(model), {"A": "C2H6O", "B": "C3H8O"}, mobile=mobile
    )
    session = ChemWorldRuntime(composition=world)
    session.reset(seed=22)
    return world, session


def execute(session, **action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", (
        result.info.get("error_message"),
        result.info.get("outcome_diagnostics"),
    )
    return result


def test_dry_continues_actual_loading_and_snapshot_uses_same_shared_clock():
    _, session = session_for()
    replay = None
    try:
        execute(session, operation="dry")
        first = session._state.state_slots.slots[0]
        assert first.values[loading_field("B")] > 0
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for current in (session, replay):
            execute(current, operation="dry")
        assert session._state.to_dict() == replay._state.to_dict()
        assert session._state.ledger.time_s == 600
        final = session._state.state_slots.slots[0]
        assert final.spec.owner_id == first.spec.owner_id
        assert final.values["sites_mol"] == first.values["sites_mol"]
        assert final.values[loading_field("B")] != first.values[loading_field("B")]
        for key, initial in (("A", 0.02), ("B", 0.01)):
            assert session._state.species_amounts[key] + final.values[loading_field(key)] == (
                pytest.approx(initial, abs=1e-10)
            )
        assert "removed_phase_inventory_history" not in session._state.metadata
    finally:
        session.close()
        if replay is not None:
            replay.close()


def test_finite_wash_retains_fixed_loading_and_heat_then_releases_to_real_liquid():
    _, session = session_for()
    try:
        execute(session, operation="dry")
        loaded = session._state.state_slots.slots[0]
        execute(
            session,
            operation="route_material",
            connection="reactor-to-waste",
            transfer_fraction=1.0,
            mixing=1,
        )
        assert session._state.volume_L == 0
        assert session._state.state_slots.slots[0].values == loaded.values
        execute(session, operation="wait", duration_s=1, stirring_speed_rpm=600)
        assert session._state.state_slots.slots[0].values == loaded.values
        execute(session, operation="add_solvent", solvent=0, volume_L=0.01)
        assert session._state.volume_L == 0.01
        assert session._state.metadata["carrier_packet_history"][-1]["volume_L"] == 0.01
        execute(
            session,
            operation="heat",
            target_temperature_K=350,
            duration_s=120,
            stirring_speed_rpm=600,
        )
        after = session._state.state_slots.slots[0]
        assert after.values[loading_field("B")] < loaded.values[loading_field("B")]
        assert session._state.species_amounts["B"] > 0
        assert session._state.ledger.time_s == 421
    finally:
        session.close()


def test_mobile_carrier_moves_with_bound_molecules_and_signed_energy_packet():
    world, session = session_for(mobile=True)
    try:
        execute(session, operation="dry")
        before = session._state
        expected = {
            name: before.state_slots.slots[0].values[name] * 0.4
            for name in before.state_slots.slots[0].values
        }
        initial_energy = _energy(before, world.physics.models_of('thermal')["reactor"]) + _energy(
            before.inactive_vessels["waste"], world.physics.models_of('thermal')["waste"]
        )
        execute(
            session,
            operation="route_material",
            connection="reactor-to-waste",
            transfer_fraction=0.4,
            mixing=1,
        )
        moved = session._state.inactive_vessels["waste"].state_slots.slots[0]
        assert moved.values == pytest.approx(expected, abs=1e-12)
        final_energy = _energy(
            session._state, world.physics.models_of("thermal")["reactor"]
        ) + _energy(
            session._state.inactive_vessels["waste"], world.physics.models_of("thermal")["waste"]
        )
        assert final_energy == pytest.approx(initial_energy, abs=1e-6)
    finally:
        session.close()


def test_material_reset_preserves_fixed_carrier_loading_and_mixes_retained_energy():
    world, session = session_for()
    try:
        execute(session, operation="dry")
        before = session._state
        slot = before.state_slots.slots[0]
        rise = before.temperature_K - 298.15
        retained = (20 + slot.values["heat_capacity_J_K"]) * rise
        for key, cv, slope, energy in (("A", 50, 0.02, -10000), ("B", 70, 0.03, -20000)):
            retained += slot.values[loading_field(key)] * (
                energy + cv * rise + 0.5 * slope * rise**2
            )
        session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        state = session._state
        assert state.state_slots.slots[0].values == slot.values
        assert state.ledger == before.ledger
        assert state.equipment == before.equipment
        assert 298.15 < state.temperature_K < before.temperature_K
        assert _energy(state, world.physics.models_of('thermal')["reactor"]) == pytest.approx(
            retained, abs=1e-6
        )
    finally:
        session.close()


def test_finite_carrier_source_exhaustion_has_no_external_fallback():
    _, session = session_for()
    try:
        execute(session, operation="add_solvent", solvent=0, volume_L=0.03)
        before = session._state
        assert before.inactive_vessels["wash_supply"].volume_L == 0
        assert before.solvent_accounting.added.volume_L == 0
        result = session.execute({"operation": "add_solvent", "solvent": 0, "volume_L": 0.001})
        assert result.info["transaction_status"] == "validation_failed"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        assert session._state.inactive_vessels == before.inactive_vessels
        assert session._state.ledger.time_s == before.ledger.time_s
        assert session._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J
        assert session._state.ledger.heat_loss_J == before.ledger.heat_loss_J
        session.reset(options={"scope": "material"})
        assert session._state.inactive_vessels["wash_supply"].volume_L == 0
    finally:
        session.close()


def test_failed_shared_sorbent_advance_rolls_back_loading_fluid_heat_and_clock(monkeypatch):
    from chemworld.physchem.coupled_system import CoupledSystem

    _, session = session_for()
    try:
        before = session._state
        def fail(*args, **kwargs):
            raise RuntimeError("injected common solver failure")
        monkeypatch.setattr(CoupledSystem, "advance", fail)
        result = session.execute({"operation": "dry"})
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        assert session._state.temperature_K == before.temperature_K
        assert session._state.ledger.time_s == before.ledger.time_s
        assert session._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J
        assert session._state.ledger.heat_loss_J == before.ledger.heat_loss_J
        assert session._state.inactive_vessels == before.inactive_vessels
    finally:
        session.close()


def test_warm_finite_supply_carries_nonzero_energy_once():
    from chemworld.runtime.condensed_material_routing import background_source_energy

    world, original = session_for()
    original.close()
    physics = world.physics
    states = dict(physics.initial_states)
    states["wash_supply"] = states["wash_supply"].replace(temperature_K=310)
    models = {name: replace(model, condensed_background=replace(
        model.condensed_background, feed_temperature_K=310))
        for name,model in physics.models_of('thermal').items()}
    payload = world.to_dict()
    physics = replace(physics, initial_states=states, models={**physics.models, 'thermal': models})
    payload["physics"] = physics.to_dict()
    session = ChemWorldRuntime(composition=type(world).from_dict(payload))
    session.reset(seed=22)
    try:
        before = session._state
        source_energy = background_source_energy(before.inactive_vessels["wash_supply"],
                                                 physics)
        execute(session, operation="add_solvent", solvent=0, volume_L=0.01)
        after = session._state
        receipt = after.metadata["carrier_packet_history"][-1]
        assert receipt["inlet_energy_J"] == pytest.approx(600 * 0.01 * (310 - 298.15))
        assert _energy(after, models["reactor"]) - _energy(before, models["reactor"]) == (
            pytest.approx(receipt["inlet_energy_J"], abs=1e-6))
        assert source_energy - background_source_energy(
            after.inactive_vessels["wash_supply"], physics) == (
                pytest.approx(receipt["inlet_energy_J"], abs=1e-6))
    finally:
        session.close()
