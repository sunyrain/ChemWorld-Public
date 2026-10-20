"""Phase-owned finite surface geometry, native kinetics and exact continuation."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.aqueous_mineral import carbonate_feed, element_amounts
from chemworld.physchem.mineral_kinetics import MineralGeometry
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_mineral import close_aqueous, committed_speciation
from chemworld.runtime.mineral_kinetics import geometry_slot, mineral_geometry
from chemworld.world.mineral_world import mineral_world_spec


def execute(session, **action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info.get("error_message")
    if action["operation"] == "measure":
        assert result.info["measurement_status"] == "valid"
    return result


def global_elements(state):
    amounts = dict(state.species_amounts)
    for other in state.inactive_vessels.values():
        for k, n in other.species_amounts.items():
            amounts[k] += n
    return element_amounts(amounts)


@pytest.mark.parametrize("mode", ["sequential", "keyed"])
def test_native_phase_geometry_feed_kinetic_wait_measure_and_exact_snapshot(mode):
    session = ChemWorldRuntime(composition=mineral_world_spec(), observation_noise_mode=mode)
    session.reset(seed=33)
    replay = None
    try:
        initial = global_elements(session._state)
        execute(session, operation="wait", duration_s=100)
        execute(session, operation="measure", instrument="aqueous_ph")
        before = session._state.species_amounts["Calcite"]
        execute(session, operation="add_component", component="HCl-solution", amount_mol=0.01)
        assert session._state.species_amounts["Calcite"] == before
        execute(session, operation="wait", duration_s=300)
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        removed = None
        for action in (
            {
                "operation": "route_material",
                "connection": 0,
                "transfer_fraction": 0.25,
                "mixing": 1,
            },
            {"operation": "route_material", "connection": 1, "transfer_fraction": 0.1, "mixing": 1},
            {"operation": "select_vessel", "vessel": "receiver"},
            {"operation": "wait", "duration_s": 1000},
            {"operation": "sample", "sample_volume_L": 0.001},
            {"operation": "measure", "instrument": "aqueous_ph"},
        ):
            before = session._state
            geo = mineral_geometry(before)
            if action["operation"] == "sample":
                liquid = before.phases.phases["aqueous"]
                removed = element_amounts(
                    {k: n * 0.001 / liquid.volume_L for k, n in liquid.species_amounts_mol.items()}
                )
            left, right = execute(session, **action), execute(replay, **action)
            assert left.info == right.info
            assert left.observation.native_record() == right.observation.native_record()
            assert session._state.to_dict() == replay._state.to_dict()
            state = session._state
            if action["operation"] == "route_material" and action["connection"] == 0:
                receiver = state.inactive_vessels["receiver"]
                assert state.species_amounts["Calcite"] == pytest.approx(
                    before.species_amounts["Calcite"] * 0.75
                )
                assert receiver.species_amounts["Calcite"] == pytest.approx(
                    before.species_amounts["Calcite"] * 0.25
                )
                assert committed_speciation(receiver).pH == pytest.approx(7, abs=1e-10)
                assert mineral_geometry(state).reference_area_m2 == pytest.approx(
                    geo.reference_area_m2 * 0.75
                )
                assert mineral_geometry(receiver).reference_amount_mol == pytest.approx(
                    geo.reference_amount_mol * 0.25
                )
                assert state.equipment == before.equipment
            elif action["operation"] in {"sample", "route_material"}:
                assert mineral_geometry(state) == geo
                assert state.species_amounts["Calcite"] == before.species_amounts["Calcite"]
            elements = global_elements(state)
            for k, n in initial.items():
                assert abs(elements[k] + (removed[k] if removed else 0) - n) <= 1e-10 * max(
                    n, 1e-12
                )
            assert state.temperature_K == 298.15
            assert state.pressure_Pa == 101325
        assert session._state.ledger.time_s == 1402
        assert session._state.vessel_elapsed_s["reactor"] == 401
        assert session._state.vessel_elapsed_s["receiver"] == 1001
        assert session._state.ledger.sample_consumed_L == 0.001
    finally:
        session.close()
        if replay:
            replay.close()


def test_native_mineral_exhaustion_removes_surface_and_holds_full_clock():
    session = ChemWorldRuntime(
        composition=mineral_world_spec(
            amounts=carbonate_feed(water_kg=0.02, calcite_mol=1e-9),
            geometry=MineralGeometry(0.04, 0.01),
        )
    )
    session.reset(seed=33)
    try:
        execute(session, operation="wait", duration_s=10000)
        assert session._state.species_amounts["Calcite"] == 0
        assert session._state.species_amounts["Ca++"] == pytest.approx(1e-9, abs=1e-19)
        assert mineral_geometry(session._state) == MineralGeometry(0, 0)
        assert not session._state.state_slots.material().slots
        assert session._state.ledger.time_s == 10000
    finally:
        session.close()


@pytest.mark.parametrize("failure", ["temperature", "numerical", "overstock"])
def test_native_mineral_failures_preserve_all_physical_state(failure, monkeypatch):
    from chemworld.physchem import solver_backend

    session = ChemWorldRuntime(composition=mineral_world_spec())
    session.reset(seed=33)
    try:
        action = {"operation": "wait", "duration_s": 100}
        if failure == "temperature":
            session._state = session._state.replace(temperature_K=310)
        elif failure == "overstock":
            action = {"operation": "add_component", "component": "HCl-solution", "amount_mol": 1}
        else:

            def fail(*args, **kwargs):
                raise RuntimeError("injected finite mineral solve failure")

            monkeypatch.setattr(solver_backend, "solve_ode", fail)
        before = session._state
        result = session.execute(action)
        assert result.info["transaction_status"] != "committed"
        state = session._state
        assert state.phases == before.phases
        assert state.state_slots == before.state_slots
        assert state.equipment == before.equipment
        assert state.inactive_vessels == before.inactive_vessels
        assert state.ledger.time_s == before.ledger.time_s
        assert state.temperature_K == before.temperature_K
        assert state.pressure_Pa == before.pressure_Pa
    finally:
        session.close()


def test_independently_evolving_mineral_families_cannot_be_silently_mixed():
    world = mineral_world_spec()
    receiver = world.physics.initial_states["receiver"]
    amount = 0.00001
    phases = PhaseLedger(
        {
            k: replace(p, species_amounts_mol={"Calcite": amount}) if k == "calcite" else p
            for k, p in receiver.phases.phases.items()
        }
    )
    receiver = close_aqueous(
        receiver.replace(
            phases=phases,
            species_amounts={**receiver.species_amounts, "Calcite": amount},
            state_slots=StateSlots((geometry_slot("receiver", MineralGeometry()),)),
            species=replace(
                receiver.species,
                initial_amounts_mol={**receiver.species.initial_amounts_mol, "Calcite": amount},
            ),
        )
    )
    payload = world.to_dict()
    payload["physics"]["initial_states"]["receiver"] = receiver.to_dict()
    session = ChemWorldRuntime(composition=payload)
    session.reset(seed=33)
    try:
        before = session._state
        result = session.execute(
            {"operation": "route_material", "connection": 0, "transfer_fraction": 0.25, "mixing": 1}
        )
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        assert session._state.inactive_vessels == before.inactive_vessels
        assert session._state.ledger.time_s == before.ledger.time_s
    finally:
        session.close()


def test_nonempty_mineral_requires_owned_geometry_at_world_construction():
    world = mineral_world_spec()
    state = world.physics.initial_states["reactor"].replace(state_slots=StateSlots())
    payload = world.to_dict()
    payload["physics"]["initial_states"]["reactor"] = state.to_dict()
    with pytest.raises(ValueError, match="owned reference geometry"):
        ChemWorldRuntime(composition=payload)


def test_split_lot_with_different_kinetic_histories_cannot_gain_area_on_recombination():
    session = ChemWorldRuntime(composition=mineral_world_spec())
    session.reset(seed=33)
    try:
        execute(session, operation="route_material", connection=0, transfer_fraction=0.25, mixing=1)
        execute(session, operation="select_vessel", vessel="receiver")
        execute(session, operation="wait", duration_s=1000)
        execute(session, operation="select_vessel", vessel="reactor")
        before = session._state
        result = session.execute(
            {"operation": "route_material", "connection": 0, "transfer_fraction": 0.1, "mixing": 1}
        )
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._state.state_slots == before.state_slots
        assert session._state.inactive_vessels == before.inactive_vessels
        assert session._state.ledger.time_s == before.ledger.time_s
    finally:
        session.close()


def test_same_progress_mineral_split_and_return_restores_its_geometry_and_stock():
    session = ChemWorldRuntime(composition=mineral_world_spec())
    session.reset(seed=33)
    try:
        original = session._state
        geometry = mineral_geometry(original)
        execute(session, operation="route_material", connection=0, transfer_fraction=0.25, mixing=1)
        execute(session, operation="select_vessel", vessel="receiver")
        execute(session, operation="route_material", connection=2, transfer_fraction=1, mixing=1)
        assert session._state.species_amounts["Calcite"] == 0
        assert mineral_geometry(session._state) == MineralGeometry(0, 0)
        execute(session, operation="select_vessel", vessel="reactor")
        assert session._state.species_amounts == pytest.approx(
            original.species_amounts, rel=1e-12, abs=1e-18
        )
        assert mineral_geometry(session._state) == geometry
        assert session._state.ledger.time_s == 0
    finally:
        session.close()


def test_direct_heat_rejection_returns_the_required_provider_contract():
    from chemworld.runtime.mineral_kinetics import MineralKineticsRuntime

    result = MineralKineticsRuntime().advance(None, {"duration_s": 10}, heat=True)
    assert result.outcome == "rejected"
    assert result.failure_reason == "mineral_temperature_data_unavailable"
    assert result.state is None
    assert result.diagnostics == {}
