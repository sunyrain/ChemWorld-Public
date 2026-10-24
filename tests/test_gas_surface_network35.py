"""Real finite gas/surface consumers retain original ownership and heat."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.gas_surface_network import example_inputs, example_world
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.catalysis_network import catalytic_inventory_key
from chemworld.physchem.gas_network_sensitivity import (
    GasSensitivityParameter,
    gas_network_trajectory_sensitivities,
)
from chemworld.physchem.gas_network_surface import GasSurfaceContext, gas_phase_network
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    advance_gas_reactor_network,
    gas_inventory_key,
    gas_reactor_network_system,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.reaction_network import RateLawSpec, ReactionSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import gas_surface_slots, unpack_gas_network
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def inputs(**kwargs):
    _, spec, network, gas, initial = example_inputs(**kwargs)
    slots = tuple(a.make_slot(initial[a.owner_id]) for a in spec.surfaces)
    state = pack_gas_reactor_network_state(spec, network, gas, surface_slots=slots)
    return spec, network, slots, state


@pytest.mark.parametrize("size", (1, 2, 3))
@pytest.mark.parametrize("thermal", (False, True))
def test_original_finite_slots_gas_heat_and_sites_evolve_together(size, thermal):
    spec, network, slots, start = inputs(size=size, thermal=thermal, both=True)
    result, detail = advance_gas_reactor_network(spec, network, start, 0.1, surface_slots=slots)
    assert result.actual_duration_s == 0.1
    assert abs(detail["energy_residual_J"]) < 2e-6
    for residual in result.conservation_residuals.values():
        assert abs(residual) < 1e-10
    for attachment, slot in zip(spec.surfaces, slots, strict=True):
        final = result.final_state
        assert final.amounts_mol[catalytic_inventory_key(slot, "HA")] != slot.values["HA"]
        occupied = sum(
            final.amounts_mol[catalytic_inventory_key(slot, s)] * weight
            for s, weight in attachment.mechanism.site_occupancies.items()
        )
        assert occupied == pytest.approx(1e-6, abs=1e-14)
        assert detail["nodes"][attachment.vessel_id]["energy_residual_J"] == pytest.approx(
            0, abs=2e-6
        )
    assert set(detail["connections"]["feed"]["amounts_mol"]) == {"A", "B"}
    assert GasReactorNetworkSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    assert start.time_s == 0


@pytest.mark.parametrize("thermal", (False, True))
def test_actual_world_selection_snapshot_and_original_slot_projection(thermal):
    composition = example_world(size=2, thermal=thermal, both=True)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=3)
        before = {n: local.state_slots.slots for n, local in local_states(runtime._state).items()}
        result = runtime.execute({"operation": "wait", "duration_s": 0.1})
        assert result.info["transaction_status"] == "committed"
        selected = runtime.execute({"operation": "select_vessel", "vessel": "source"})
        assert selected.info["transaction_status"] == "committed"
        assert runtime._state.vessel_id == "source"
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for consumer in (runtime, restored):
            result = consumer.execute({"operation": "wait", "duration_s": 0.2})
            assert result.info["transaction_status"] == "committed"
        assert restored._state.to_dict() == runtime._state.to_dict()
        for name, local in local_states(runtime._state).items():
            assert set(local.species_amounts) == {"A", "B"}
            catalyst = next(
                s for s in local.state_slots.slots if s.spec.owner_id == name + ":catalyst"
            )
            original = next(s for s in before[name] if s.spec.owner_id == name + ":catalyst")
            assert catalyst.values["HA"] != original.values["HA"]
            assert catalyst.initial_values == original.initial_values
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def test_gas_chemistry_flag_leaves_independently_attached_surface_active():
    spec, network, slots, start = inputs()
    reaction = ReactionSpec.from_equation(
        reaction_id="gas-conversion",
        equation="A => B",
        rate_law=RateLawSpec("gas", "mass_action", {"k": 0.4}),
    )
    network = replace(network, reactions=(reaction,))
    disabled = replace(spec, nodes=tuple(replace(n, chemistry_enabled=False) for n in spec.nodes))
    on = gas_reactor_network_system(spec, network, surface_slots=slots).processes[0].evaluate(start)
    off = (
        gas_reactor_network_system(disabled, network, surface_slots=slots)
        .processes[0]
        .evaluate(start)
    )
    for node in spec.nodes:
        key = gas_inventory_key(node.vessel_id, "A")
        assert off.amount_rates_mol_s.get(key, 0) - on.amount_rates_mol_s.get(
            key, 0
        ) == pytest.approx(0.4 * start.amounts_mol[key])
    adsorbed = catalytic_inventory_key(slots[0], "HA")
    assert off.amount_rates_mol_s[adsorbed] == on.amount_rates_mol_s[adsorbed] != 0


@pytest.mark.parametrize("defect", ("missing", "moved", "area", "owner", "capacity"))
def test_declared_slot_contract_cannot_be_silently_changed(defect):
    spec, network, slots, _ = inputs()
    slot = slots[0]
    if defect == "missing":
        slots = ()
    elif defect == "moved":
        slots = (replace(slot, location_id="receiver"),)
    elif defect == "area":
        spec = replace(spec, surfaces=(replace(spec.surfaces[0], area_m2=0.2),))
    elif defect == "owner":
        slots = (replace(slot, spec=replace(slot.spec, owner_id="other")),)
    else:
        slots = (replace(slot, values={**slot.values, "heat_capacity_J_K": 999}),)
    with pytest.raises(ValueError):
        gas_reactor_network_system(spec, network, surface_slots=slots)


def test_real_world_rejects_missing_surface_inventory_without_mutating_snapshot():
    composition = example_world()
    states = composition.physics.initial_states
    anchor = states["source"].replace(inactive_vessels={"receiver": states["receiver"]})
    broken = anchor.replace(
        state_slots=StateSlots(
            tuple(s for s in anchor.state_slots.slots if s.spec.owner_id != "source:catalyst")
        )
    )
    before = broken.to_dict()
    with pytest.raises(ValueError, match="surface slots"):
        unpack_gas_network(broken, composition.physics.network)
    assert broken.to_dict() == before


def test_wrong_gas_thermochemistry_and_surface_sensitivity_are_explicitly_refused():
    spec, network, slots, start = inputs()
    wrong = replace(network, species=(replace(network.species[0], formula="H"), network.species[1]))
    with pytest.raises(ValueError, match="identical identities"):
        GasSurfaceContext(spec, wrong, slots)
    with pytest.raises(ValueError, match="surface-network sensitivity"):
        gas_network_trajectory_sensitivities(
            spec,
            network,
            start,
            (GasSensitivityParameter("flow", "feed", "coefficient"),),
            (0, 0.1),
        )


def test_whole_identity_table_and_real_slots_have_one_gas_projection():
    composition = example_world(size=3, both=True)
    whole = composition.physics.network
    assert {s.phase for s in whole.species} == {"gas", "surface"}
    assert set(gas_phase_network(whole).species_ids) == {"A", "B"}
    states = composition.physics.initial_states
    anchor = states["source"].replace(inactive_vessels={"receiver": states["receiver"]})
    _, spec, packed = unpack_gas_network(anchor, whole)
    slots = gas_surface_slots(anchor, spec)
    context = GasSurfaceContext(spec, whole, slots)
    assert set(packed.amounts_mol) == set(context.inventory_keys)
    assert len(slots) == 2
    with pytest.raises(ValueError, match="cover the declared"):
        gas_reactor_network_world(spec, whole, {n: s.species_amounts for n, s in states.items()})
