"""Chemical identities and old battery ownership across the LOQS addition.

These are small public-interface contracts.  The atom matrix is independent
of the physical reaction implementation; no PyBaMM qualification is repeated.
"""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    ParticleSpec,
    battery_chemical_energy_j,
    battery_chemical_potentials,
    battery_dissipation_w,
    battery_sei_affinity_j_mol,
    battery_spec_from_dict,
)
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_newman_tobias import BatteryNewmanTobiasSpec
from chemworld.physchem.battery_porous_electrical import PorousElectrodeGeometrySpec
from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec

KEYS = (
    "negative:Pb", "negative:PbSO4", "positive:PbO2", "positive:PbSO4",
    "electrolyte:H2SO4", "electrolyte:H2O",
)
SPECIES = (
    "leadacid:Pb", "leadacid:PbSO4", "leadacid:PbO2", "leadacid:PbSO4",
    "leadacid:H2SO4", "leadacid:H2O",
)
FORMULAS = {
    "leadacid:Pb": "Pb", "leadacid:PbSO4": "PbSO4", "leadacid:PbO2": "PbO2",
    "leadacid:H2SO4": "H2SO4", "leadacid:H2O": "H2O",
}
# Rows Pb, S, H, O; columns retain the two separately owned sulfate stores.
ATOMS = np.array([
    [1, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 2, 2], [0, 4, 2, 4, 4, 1],
])
FARADAY = 96485.33212


def lead_spec():
    from chemworld.physchem.battery_lead_acid import BatteryLeadAcidSpec

    return BatteryLeadAcidSpec()


def lithium_spec(family):
    particles = {"negative": ParticleSpec(cells=3), "positive": ParticleSpec(cells=4)}
    if family == "SPM":
        return BatterySPMSpec(**particles)
    if family == "MPM":
        return BatteryMPMSpec(**particles)
    if family == "DFN":
        return BatteryDFNSpec(**particles, electrolyte=BatteryElectrolyteSpec(cells=(2, 2, 2)))
    return BatteryNewmanTobiasSpec(geometry=PorousElectrodeGeometrySpec(cells=(2, 2, 2)))


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.fixture
def runtime():
    instance = ChemWorldRuntime(composition=battery_world_spec(lead_spec()))
    instance.reset(seed=9)
    try:
        yield instance
    finally:
        instance.close()


@pytest.mark.parametrize("family", ["SPM", "MPM", "DFN", "NT"])
def test_old_lithium_codec_device_and_slot_declaration_are_unchanged(family):
    original = lithium_spec(family)
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    restored = battery_spec_from_dict(payload)
    assert type(restored) is type(original) and restored == original
    assert restored.to_dict() == original.to_dict()
    world = battery_world_spec(restored)
    state = world.physics.initial_states["reactor"]
    record, slot, decoded, physical, _ = unpack_battery(state)
    assert record.equipment_type == "single_particle_battery"
    assert slot.spec.state_type == "battery-lithium-and-charge-v1"
    assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
    assert slot.spec.batch_reset == "preserve"
    assert decoded == original and physical.amounts_mol == original.initial_state().amounts_mol
    assert set(slot.values) == set(original.inventory_keys) | {
        "discharged_charge_C", "delivered_work_J"}
    assert all(not field.name.startswith("leadacid") for field in slot.spec.fields)
    assert all(not key.startswith("leadacid") for key in state.species_amounts)
    assert {component.kind for component in world.components} == {"battery_spm", "observation"}


def test_lead_acid_identity_has_six_stores_five_species_and_no_housing_copy(runtime):
    record, slot, spec, physical, _ = unpack_battery(runtime._state)
    assert tuple(spec.inventory_keys) == KEYS
    assert spec.inventory_species == dict(zip(KEYS, SPECIES, strict=True))
    assert spec.species_formulas == FORMULAS
    assert tuple(physical.coordinates) == ("discharged_charge_C", "delivered_work_J")
    assert set(slot.values) == set(KEYS) | set(physical.coordinates)
    assert record.equipment_type == "electrochemical_battery"
    assert slot.spec.state_type == "battery-lead-acid-inventory-and-charge-v1"
    assert slot.spec.owner_kind == "equipment" and slot.spec.owner_id == record.equipment_id
    assert slot.spec.batch_reset == "preserve"
    assert {f.name: f.species_id for f in slot.spec.fields if f.species_id} == (
        dict(zip(KEYS, SPECIES, strict=True)))
    assert all(f.unit == "mol" for f in slot.spec.fields if f.species_id)
    assert all(value == 0 for value in runtime._state.species_amounts.values())
    assert all(not phase.species_amounts_mol for phase in runtime._state.phases.phases.values())
    owned = StateSlots((slot,)).species_amounts()
    total = runtime.constitution.material_inventory(runtime._state)
    assert total == owned
    assert total["leadacid:PbSO4"] == (
        physical.amounts_mol["negative:PbSO4"] + physical.amounts_mol["positive:PbSO4"])
    assert "Li" not in total
    elements = runtime.constitution.element_totals(total)
    expected = ATOMS @ np.array([physical.amounts_mol[key] for key in KEYS])
    assert elements == pytest.approx(dict(zip(("Pb", "S", "H", "O"), expected, strict=True)),
                                     rel=0, abs=2e-15)
    assert {component.kind for component in battery_world_spec(spec).components} == {
        "battery_cell", "observation"}


def test_declared_custom_inventory_closes_geometric_and_liquid_volume_without_hidden_water():
    from chemworld.physchem.battery_lead_acid import (
        BatteryLeadAcidSpec,
        LeadAcidGeometrySpec,
        LeadAcidMaterialSpec,
        lead_acid_fields,
        validate_lead_acid_state,
    )

    geometry = LeadAcidGeometrySpec(
        area_m2=.01, lengths_m=(.001, .001, .001),
        inactive_solid_volumes_m3=(0., 0.), separator_porosity=.6,
        reaction_area_densities_m_inv=(2e5, 2e6),
    )
    material = LeadAcidMaterialSpec(
        molar_volumes_m3_mol=(1.8e-5, 2.5e-5, 4.8e-5, 4.4e-5, 1.75e-5),
        provenance="explicit closed-volume synthetic compatibility material",
    )
    pore0, acid0, sulfate = 1.6e-5, .064, .02
    initial = ((5e-6 - sulfate * 4.8e-5) / 1.8e-5, sulfate,
               (5e-6 - sulfate * 4.8e-5) / 2.5e-5, sulfate,
               acid0, (pore0 - acid0 * 4.4e-5) / 1.75e-5)
    spec = BatteryLeadAcidSpec(geometry=geometry, material=material, initial_amounts_mol=initial)
    encoded = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    assert battery_spec_from_dict(encoded) == spec
    assert battery_spec_from_dict(encoded).to_dict() == encoded
    # Construct one stoichiometrically valid state, independently of the ODE.
    extent = .001
    n = np.asarray(initial) + extent * np.array([-1, 1, -1, 1, -2, 2])
    state = replace(spec.initial_state(), amounts_mol=dict(zip(KEYS, n, strict=True)),
                    coordinates={"discharged_charge_C": 2 * FARADAY * extent,
                                 "delivered_work_J": 0.})
    validate_lead_acid_state(spec, state)
    fields = lead_acid_fields(spec, state, current_A=0.)
    pore = pore0 + (1.8e-5 + 2.5e-5 - 2 * 4.8e-5) * extent
    assert fields["total_pore_volume_m3"] == pytest.approx(pore, rel=0, abs=2e-20)
    assert fields["acid_concentration_mol_m3"] == pytest.approx(n[4] / pore, rel=2e-14)
    assert fields["water_concentration_mol_m3"] == pytest.approx(n[5] / pore, rel=2e-14)
    assert n[4] * 4.4e-5 + n[5] * 1.75e-5 == pytest.approx(pore, rel=0, abs=2e-20)
    np.testing.assert_allclose(ATOMS @ n, ATOMS @ initial, rtol=0, atol=2e-15)
    before = state.amounts_mol.copy()
    invalid = replace(state, amounts_mol={**before, "electrolyte:H2O": n[5] + .001})
    with pytest.raises(ValueError, match=r"(volume|Faraday)"):
        validate_lead_acid_state(spec, invalid)
    assert state.amounts_mol == before


def test_real_wait_preserves_each_element_without_collapsing_the_two_sulfate_stores(runtime):
    initial = deepcopy(runtime._state)
    _, _, _, before, _ = unpack_battery(initial)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .1})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    _, _, _, after, _ = unpack_battery(runtime._state)
    n0 = np.array([before.amounts_mol[k] for k in KEYS])
    n1 = np.array([after.amounts_mol[k] for k in KEYS])
    np.testing.assert_allclose(ATOMS @ n1, ATOMS @ n0, rtol=0, atol=2e-12)
    # Independently apply Pb + PbO2 + 2 H2SO4 -> 2 PbSO4 + 2 H2O.
    extent = .1 / (2 * FARADAY)
    np.testing.assert_allclose(n1 - n0, extent * np.array([-1, 1, -1, 1, -2, 2]),
                               rtol=0, atol=2e-12)
    assert runtime.constitution.check_material_conservation(initial, runtime._state).passed
    assert set(after.amounts_mol) == set(KEYS)
    assert not any(runtime._state.species_amounts.values())


@pytest.mark.parametrize("analysis", [
    "energy", "potentials", "dissipation", "sei", "impedance_system", "impedance", "soh", "soc",
])
def test_undefined_lithium_analyses_reject_explicitly_before_accessing_particle_fields(analysis):
    spec = lead_spec()
    state = spec.initial_state()
    calls = {
        "energy": lambda: battery_chemical_energy_j(spec, state),
        "potentials": lambda: battery_chemical_potentials(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, 0.),
        "sei": lambda: battery_sei_affinity_j_mol(spec, state, 0.),
        "impedance_system": lambda: battery_impedance_system(spec, state),
        "impedance": lambda: battery_impedance(spec, [1.], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "soc": lambda: battery_spec_at_soc(spec, None, .5),
    }
    with pytest.raises(ValueError, match=r"(?i)(lead.acid|LOQS)"):
        calls[analysis]()


def test_same_chemistry_equipment_replacement_has_one_fresh_inventory_and_retains_time(runtime):
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .1})
    execute(runtime, {"operation": "wait", "duration_s": 1.})
    record, old_slot, spec, old_physical, _ = unpack_battery(runtime._state)
    before_time = runtime._state.ledger.time_s
    history = deepcopy(runtime._lifecycle_history)
    replacement = replace(record, equipment_id="lead-acid-replacement")
    fresh = initial_battery_slot(replacement, spec)
    runtime.reset(options={
        "scope": "equipment", "equipment_id": record.equipment_id,
        "replacement": replacement, "slots": StateSlots((fresh,)),
    })
    actual, slot, new_spec, physical, _ = unpack_battery(runtime._state)
    assert actual == replacement and new_spec == spec and slot == fresh
    assert physical.time_s == before_time == old_physical.time_s
    assert physical.amounts_mol == spec.initial_state().amounts_mol
    assert physical.amounts_mol != old_physical.amounts_mol
    assert physical.coordinates == {"discharged_charge_C": 0., "delivered_work_J": 0.}
    assert len(runtime._state.state_slots.slots) == 1
    assert old_slot.spec.slot_id not in runtime._state.state_slots.to_dict()
    assert record.equipment_id not in runtime._state.equipment.equipment
    assert runtime._lifecycle_history[:-1] == history
    assert runtime._lifecycle_history[-1]["introduced_slots"] == StateSlots((fresh,)).to_dict()
    assert runtime.constitution.material_inventory(runtime._state) == (
        StateSlots((fresh,)).species_amounts())


@pytest.mark.parametrize("source", ["lithium", "lead-acid"])
def test_cross_chemistry_equipment_replacement_is_atomic_rejection(source):
    source_spec = lithium_spec("SPM") if source == "lithium" else lead_spec()
    target_spec = lead_spec() if source == "lithium" else lithium_spec("SPM")
    instance = ChemWorldRuntime(composition=battery_world_spec(source_spec))
    instance.reset(seed=9)
    try:
        record = unpack_battery(instance._state)[0]
        target_world = battery_world_spec(target_spec)
        target_record = unpack_battery(target_world.physics.initial_states["reactor"])[0]
        replacement = replace(target_record, equipment_id="cross-chemistry-replacement")
        fresh = initial_battery_slot(replacement, target_spec)
        before = instance.snapshot().to_dict()
        history = deepcopy(instance._lifecycle_history)
        with pytest.raises(ValueError, match="equipment type"):
            instance.reset(options={
                "scope": "equipment", "equipment_id": record.equipment_id,
                "replacement": replacement, "slots": StateSlots((fresh,)),
            })
        assert instance.snapshot().to_dict() == before
        assert instance._lifecycle_history == history
    finally:
        instance.close()
