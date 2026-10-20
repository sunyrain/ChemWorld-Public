"""Actual shared thermal operations and existing finite crystal material ports."""

import json
from copy import deepcopy

import pytest

from chemworld.examples.native_crystal_research import (
    synthetic_crystal_model,
    synthetic_crystal_thermodynamics,
)
from chemworld.foundation import equipment_settings
from chemworld.physchem.elements import molecular_weight
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.crystal_thermal import crystal_inventory_energy, population_from_state
from chemworld.world.crystal_world import crystal_world_spec


def session_for():
    model = synthetic_crystal_model(molecular_weight({"C": 6, "H": 12, "O": 1}) / 1000)
    world = crystal_world_spec(
        model, synthetic_crystal_thermodynamics(), {"P": "C6H12O", "I": "C3H8O"}
    )
    session = ChemWorldRuntime(composition=world)
    session.reset(seed=23)
    return world, session


def execute(session, **action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info.get("error_message")
    return result


def test_native_cold_warm_and_snapshot_continue_one_clock_without_future_allocation():
    world, session = session_for()
    replay = None
    try:
        execute(session, operation="cool_crystallize", target_temperature_K=278.15, duration_s=60)
        before = session._state
        assert 278.15 < before.temperature_K < 298.15
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict(), sort_keys=True))
        )
        for current in (session, replay):
            execute(
                current,
                operation="heat",
                target_temperature_K=350,
                duration_s=1,
                stirring_speed_rpm=600,
            )
        after = session._state
        assert after.ledger.time_s == 61
        assert after.pressure_Pa == before.pressure_Pa == 101325
        assert after.temperature_K < before.temperature_K + 0.1
        assert after.phases.phases["solid"].species_amounts_mol["P"] > 0.001
        assert after.to_dict() == replay._state.to_dict()
        model, thermal = (
            world.physics.crystal_models["reactor"],
            world.physics.thermal_models["reactor"],
        )
        assert crystal_inventory_energy(after, model, thermal) - crystal_inventory_energy(
            before, model, thermal
        ) == pytest.approx(
            after.ledger.energy_jacket_J
            - before.ledger.energy_jacket_J
            - after.ledger.heat_loss_J
            + before.ledger.heat_loss_J,
            abs=1e-6,
        )
    finally:
        session.close()
        if replay is not None:
            replay.close()


def test_native_filter_resuspend_preserves_historical_filtrate_and_reconciles_energy():
    world, session = session_for()
    try:
        execute(session, operation="filter_crystals")
        before = session._state
        assert before.ledger.time_s == 480
        assert before.phases.phases["cake_liquor"].volume_L == pytest.approx(0.0016)
        execute(session, operation="resuspend_crystals", solvent=0, volume_L=0.01)
        sample = deepcopy(session._state.samples.to_dict())
        receipt = session._state.metadata["last_crystal_material_packet"]
        assert abs(receipt["energy_balance_residual_J"]) < 1e-6
        execute(
            session,
            operation="heat",
            target_temperature_K=350,
            duration_s=60,
            stirring_speed_rpm=600,
        )
        assert session._state.samples.to_dict() == sample
        population_from_state(session._state, world.physics.crystal_models["reactor"])
    finally:
        session.close()


def test_actual_container_port_carries_count_size_and_molecular_constituents():
    world, session = session_for()
    try:
        execute(session, operation="create_container", container="container-01", capacity_L=0.05)
        before = equipment_settings(session._state.equipment, "crystallizer")
        execute(
            session,
            operation="transfer_material",
            source_container="active",
            destination_container="container-01",
            transfer_fraction=0.4,
            mixing=0,
        )
        stored = session._state.samples.samples["container-01"].contents.properties["crystallizer"]
        for actual, original in zip(
            stored["population_cohorts"], before["population_cohorts"], strict=True
        ):
            assert actual == pytest.approx([original[0] * 0.4, original[1]])
        for actual, original in zip(
            stored["population_constituents"], before["population_constituents"], strict=True
        ):
            assert actual == pytest.approx([n * 0.4 for n in original])
        population_from_state(session._state, world.physics.crystal_models["reactor"])
    finally:
        session.close()


def test_native_common_solver_failure_rolls_back_crystal_phases_population_temperature_and_clock(
    monkeypatch,
):
    from chemworld.physchem import solver_backend

    _, session = session_for()
    try:
        before = session._state

        def fail(*args, **kwargs):
            raise RuntimeError("injected crystal integration failure")

        monkeypatch.setattr(solver_backend, "solve_ode", fail)
        result = session.execute(
            {
                "operation": "heat",
                "target_temperature_K": 350,
                "duration_s": 60,
                "stirring_speed_rpm": 600,
            }
        )
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._state.equipment == before.equipment
        assert session._state.temperature_K == before.temperature_K
        assert session._state.ledger.time_s == before.ledger.time_s
        assert session._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J
    finally:
        session.close()


def test_material_reset_mixes_retained_body_heat_with_fresh_crystal_inventory():
    world, session = session_for()
    try:
        execute(
            session,
            operation="heat",
            target_temperature_K=350,
            duration_s=10,
            stirring_speed_rpm=600,
        )
        before = session._state
        expected = -20 + 20 * (before.temperature_K - 298.15)
        session.reset(options={"scope": "material"})
        state = session._state
        assert 298.15 < state.temperature_K < before.temperature_K
        assert state.ledger == before.ledger
        assert state.phases.phases["solid"].species_amounts_mol["P"] == 0.001
        assert crystal_inventory_energy(
            state, world.physics.crystal_models["reactor"], world.physics.thermal_models["reactor"]
        ) == pytest.approx(expected, abs=1e-6)
        population_from_state(state, world.physics.crystal_models["reactor"])
    finally:
        session.close()


def test_empty_crystal_apparatus_can_heat_without_creating_solvent_or_particles():
    _, session = session_for()
    try:
        execute(session, operation="create_container", container="container-01", capacity_L=0.05)
        execute(
            session,
            operation="transfer_material",
            source_container="active",
            destination_container="container-01",
            transfer_fraction=1.0,
            mixing=0,
        )
        assert session._state.volume_L == 0
        execute(
            session,
            operation="heat",
            target_temperature_K=350,
            duration_s=1,
            stirring_speed_rpm=600,
        )
        state = session._state
        assert state.volume_L == 0
        assert not any(state.species_amounts.values())
        assert not equipment_settings(state.equipment, "crystallizer")["population_cohorts"]
        assert 298.15 < state.temperature_K < 298.41
    finally:
        session.close()


def test_native_resource_card_without_reagent_operation_enforces_actual_seed_stock():
    from chemworld.campaign_resources import CampaignResourceCard

    world, original = session_for()
    original.close()

    card = CampaignResourceCard(
        "crystal-seed-limit", 20, 1, 0, 4, {"seed_g": 0.001, "solvent_L": 0.02}
    )
    session = ChemWorldRuntime(composition=world, campaign_resource_card=card)
    session.reset(seed=23)
    try:
        execute(session, operation="seed_crystals", seed_mass_g=0.001)
        before = session._state
        result = session.execute({"operation": "seed_crystals", "seed_mass_g": 0.001})
        assert result.info["transaction_status"] != "committed"
        assert session._state.phases == before.phases
        assert session._campaign_resource_ledger.stocks_used["seed_g"] == 0.001
    finally:
        session.close()


def test_seed_input_preserves_a_seed_excluded_native_recovery_reference():
    _, session = session_for()
    try:
        execute(session, operation="seed_crystals", seed_mass_g=0.001)
        # Initial P=.021 includes .0003 solid and .0002 dissolved seed.
        assert session._state.process.metrics["pre_separation_product_mol"] == pytest.approx(0.0205)
        execute(session, operation="wait", duration_s=1, stirring_speed_rpm=600)
        solid = session._state.phases.phases["solid"].species_amounts_mol["P"]
        seed = equipment_settings(session._state.equipment, "crystallizer")["seed_target_mol"]
        assert session._state.process.metrics["seed_excluded_recovery"] == pytest.approx(
            (solid - seed) / 0.0205
        )
    finally:
        session.close()
