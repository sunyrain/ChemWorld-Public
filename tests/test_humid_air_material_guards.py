"""Identity and physical limits of the actual humid-air packet endpoints."""

from dataclasses import replace

import pytest
from tests.reference.humid_air_material_fixtures import world

from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.humid_air_material import route_humid_air_packet
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.composition import WorldCompositionSpec

ACTION = {"operation": "route_material", "connection": 0, "transfer_fraction": 0.2, "mixing": 1}


@pytest.mark.parametrize("phase", ("liquid", "ice"))
def test_snapshot_immediately_after_material_packet_saves_and_replays(phase, tmp_path):
    payload, _, _ = world(phase, "gas")
    runtime, resumed = ChemWorldRuntime(composition=payload), None
    try:
        runtime.reset(seed=25)
        assert runtime.execute(ACTION).info["transaction_status"] == "committed"
        path = tmp_path / (phase+"-after-packet.json")
        runtime.snapshot().save(path)
        resumed = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
        for run in (runtime, resumed):
            assert run.execute(ACTION).info["transaction_status"] == "committed"
        assert runtime._state.to_dict() == resumed._state.to_dict()
    finally:
        runtime.close()
        if resumed is not None:
            resumed.close()


@pytest.mark.parametrize("change", ("identity", "weight"))
def test_incompatible_endpoint_declaration_rejected_before_inventory_change(change):
    payload, _, _ = world("liquid", "gas")
    spec = WorldCompositionSpec.from_dict(payload)
    runtime = ChemWorldRuntime(composition=spec)
    try:
        runtime.reset(seed=25)
        old = runtime._state.to_dict()
        models = dict(spec.physics.models_of("thermal"))
        thermal = models["receiver"].thermodynamics
        c = thermal.nitrogen
        c = replace(
            c,
            **(
                {"identifier": "different_nitrogen"}
                if change == "identity"
                else {"molecular_weight_g_mol": c.molecular_weight_g_mol * 1.001}
            ),
        )
        models["receiver"] = replace(
            models["receiver"], thermodynamics=replace(thermal, nitrogen=c)
        )
        physics = replace(spec.physics, models={**spec.physics.models, "thermal": models})
        with pytest.raises(ValueError, match="identities and weights"):
            route_humid_air_packet(runtime._state, physics, ACTION)
        assert runtime._state.to_dict() == old
    finally:
        runtime.close()


def test_receiving_pressure_limit_rolls_back_both_finite_vessels():
    payload, _, _ = world("liquid", "gas")
    spec = WorldCompositionSpec.from_dict(payload)
    models = dict(spec.physics.models_of("thermal"))
    models["receiver"] = replace(
        models["receiver"],
        maximum_pressure_Pa=(spec.physics.initial_states["receiver"].pressure_Pa * 1.001),
    )
    spec = replace(
        spec, physics=replace(spec.physics, models={**spec.physics.models, "thermal": models})
    )
    runtime = ChemWorldRuntime(composition=spec)
    try:
        runtime.reset(seed=25)
        before = runtime._state
        result = runtime.execute(ACTION)
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.species_amounts == before.species_amounts
        assert runtime._state.temperature_K == before.temperature_K
        assert runtime._state.inactive_vessels == before.inactive_vessels
        assert runtime._state.ledger.time_s == before.ledger.time_s
        assert runtime._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J
    finally:
        runtime.close()
