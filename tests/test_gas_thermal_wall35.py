"""Wall ownership, signed radiation and explicit authoring boundaries."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.gas_thermal_wall import example_inputs
from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    gas_accounting_coordinates,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_thermal_wall import GAS_WALL_TYPE, wall_key
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import GasReactorNetworkRuntime, unpack_gas_network
from chemworld.world.gas_reactor_network import gas_reactor_network_world


@pytest.mark.parametrize("temperature", (250.0, 300.0, 800.0))
def test_environment_radiation_is_signed_and_matches_grey_law(temperature):
    spec, _, _ = example_inputs()
    wall = spec.thermal_wall
    conduction, radiation = wall.environment_losses(temperature)
    assert conduction == pytest.approx(0.02 * (temperature - 300))
    assert radiation == pytest.approx(0.8 * 5.670374419e-8 * 0.001 * (temperature**4 - 300**4))


def test_wall_codec_is_explicit_and_legacy_codec_does_not_gain_a_field():
    spec, _, _ = example_inputs()
    assert spec.to_dict() == json.loads(json.dumps(spec.to_dict()))
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    old = replace(spec, thermal_wall=None)
    assert "thermal_wall" not in old.to_dict()
    assert old.to_dict()["model"] == "finite-thermal-gas-network-v1"


def test_wall_temperature_has_one_owner_and_receipts_are_not_temperatures():
    spec, network, amounts = example_inputs()
    state = gas_reactor_network_world(spec, network, amounts).physics.initial_states["reactor"]
    slot = next(s for s in state.state_slots.slots if s.spec.state_type == GAS_WALL_TYPE)
    assert slot.spec.batch_reset == "preserve"
    assert slot.spec.heat_capacity_field is None
    assert slot.values == {"temperature_K": 800.0}
    assert all(c.unit != "K" for c in gas_accounting_coordinates(spec, network))
    packed = pack_gas_reactor_network_state(spec, network, amounts, wall_temperature_K=650.0)
    assert packed.coordinates[wall_key("wall", "temperature_K")] == 650.0
    assert packed.temperature_K == 450.0


@pytest.mark.parametrize("defect", ("duplicate", "out-of-domain", "wrong-initial", "foreign-owner"))
def test_existing_runtime_refuses_corrupt_wall_ownership(defect):
    spec, network, amounts = example_inputs()
    state = gas_reactor_network_world(spec, network, amounts).physics.initial_states["reactor"]
    slots = list(state.state_slots.slots)
    i = next(i for i, s in enumerate(slots) if s.spec.state_type == GAS_WALL_TYPE)
    if defect == "duplicate":
        slots.append(replace(slots[i], spec=replace(slots[i].spec, slot_id="duplicate")))
    elif defect == "out-of-domain":
        slots[i] = replace(slots[i], values={"temperature_K": 1500.0})
    elif defect == "wrong-initial":
        slots[i] = replace(slots[i], initial_values={"temperature_K": 600.0})
    else:
        slots[i] = replace(slots[i], spec=replace(slots[i].spec, owner_id="elsewhere"))
    with pytest.raises(ValueError):
        unpack_gas_network(state.replace(state_slots=StateSlots(tuple(slots))), network)


def test_wall_cannot_replace_the_original_network_equipment_identity():
    spec, _, _ = example_inputs()
    with pytest.raises(ValueError, match="thermal wall"):
        replace(spec, thermal_wall=replace(spec.thermal_wall, owner_id="reactor:gas-network"))


def test_undeclared_wall_temperature_cannot_enter_a_legacy_pack():
    spec, network, amounts = example_inputs()
    with pytest.raises(ValueError, match="requires a declared"):
        pack_gas_reactor_network_state(
            replace(spec, thermal_wall=None), network, amounts, wall_temperature_K=800
        )


def test_unrelated_gas_system_keeps_its_separate_wall_owner():
    spec, network, amounts = example_inputs()
    own = gas_reactor_network_world(spec, network, amounts).physics.initial_states["reactor"]
    other_spec = replace(
        spec,
        nodes=(replace(spec.nodes[0], vessel_id="other"),),
        anchor_vessel_id="other",
        energy_nodes=(replace(spec.energy_nodes[0], vessel_id="other"),),
        thermal_wall=replace(
            spec.thermal_wall,
            owner_id="other-wall",
            contacts=(replace(spec.thermal_wall.contacts[0], vessel_id="other"),),
        ),
    )
    other = gas_reactor_network_world(
        other_spec, network, {"other": amounts["reactor"]}
    ).physics.initial_states["other"]
    unpack_gas_network(own.replace(inactive_vessels={"other": other}), network)
    unpack_gas_network(other.replace(inactive_vessels={"reactor": own}), network)


@pytest.mark.parametrize("parameter", ("heat_capacity_J_K", "emissivity", "gas-support"))
def test_saved_wall_model_and_live_advance_are_bound_to_original_declaration(parameter):
    composition = gas_reactor_network_world(*example_inputs())
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=35)
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        model = snapshot["world_state"]["equipment"]["reactor:gas-network"]["settings"][
            "physical_model"
        ]
        if parameter == "gas-support":
            model["energy_nodes"][0]["support_heat_capacity_J_K"] = 2.0
        else:
            model["thermal_wall"][parameter] *= 0.5
        with pytest.raises(ValueError, match="original composition"):
            ChemWorldRuntime.from_snapshot(snapshot)
        records = dict(runtime._state.equipment.equipment)
        record = records["reactor:gas-network"]
        records[record.equipment_id] = replace(record, settings={"physical_model": deepcopy(model)})
        changed = runtime._state.replace(equipment=EquipmentLedger(records))
        result = GasReactorNetworkRuntime(composition).advance(changed, {"duration_s": 0.1})
        assert not result.success
        assert "original composition" in result.diagnostics["solver_error"]
        assert runtime._state.ledger.time_s == 0.0
    finally:
        runtime.close()
