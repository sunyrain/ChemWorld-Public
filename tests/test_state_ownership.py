"""Ownership is exercised through real sample and component-port routing."""

import json
from dataclasses import asdict, replace

import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation import WorldState, equipment_settings, upsert_equipment_record
from chemworld.foundation.constitution import PhysicalConstitution
from chemworld.foundation.samples import StoredSample
from chemworld.foundation.solvents import SolventInventory
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, ProcessLedger
from chemworld.foundation.state_ownership import (
    CapacityConstraint,
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.runtime.component_network import _withdraw
from chemworld.runtime.material_routing import _active, _install, _merge, _portion
from chemworld.runtime.transactions import StatePatch, TransactionManager
from chemworld.world.state_factory import initial_chemworld_state
from test_component_network import request
from test_material_routing import create, make_env, transfer
from test_recrystallization_cycles import cycle_actions


def support(
    owner="particles",
    *,
    fixed=False,
    capacity=1.0,
    bound=0.2,
    heat_capacity=10.0,
    phase="reactor_liquid",
    location="batch_reactor",
    mechanism="single-site",
):
    spec = StateSlotSpec(
        slot_id=owner,
        owner_kind="equipment" if fixed else "material",
        owner_id="batch_reactor" if fixed else owner,
        state_type="catalyst_sites",
        mechanism_id=mechanism,
        phase_id=None if fixed else phase,
        fields=(
            StateFieldSpec("area", "m2", conserved_quantity="support_area"),
            StateFieldSpec("sites", "mol", conserved_quantity="single-site:sites"),
            StateFieldSpec("adsorbate", "mol", species_id="P"),
            StateFieldSpec("heat_capacity", "J/K", conserved_quantity="support_heat_capacity"),
        ),
        batch_reset="preserve" if fixed else "clear",
        heat_capacity_field="heat_capacity",
        capacities=(CapacityConstraint("sites", (("adsorbate", 1.0),)),),
    )
    return StateSlot(
        spec,
        location,
        {
            "area": capacity * 2,
            "sites": capacity,
            "adsorbate": bound,
            "heat_capacity": heat_capacity,
        },
    )


def state(*slots, temperature=300.0, volume=0.01):
    base = initial_chemworld_state(species_ids=("P",))
    return base.replace(
        species_amounts={"P": 0.01},
        volume_L=volume,
        temperature_K=temperature,
        phases=PhaseLedger(
            {
                "reactor_liquid": PhaseRecord(
                    "reactor_liquid",
                    base.vessel_id,
                    "liquid",
                    volume,
                    {"P": 0.01},
                    solvents=SolventInventory.pure(0, volume),
                )
            }
        ),
        state_slots=StateSlots(slots),
        quenched=True,
    )


@pytest.mark.parametrize("fraction", [0.0, 0.15, 0.5, 1.0])
def test_split_and_reunion_preserve_all_material_quantities(fraction):
    original = _active(state(support()))
    a = _portion(original, fraction, "vial")
    b = _portion(original, 1 - fraction, "vial")
    joined = _merge(a, b)
    assert joined.species_amounts_mol == pytest.approx(original.species_amounts_mol)
    assert joined.contents.state_slots.conserved() == pytest.approx(
        original.contents.state_slots.conserved()
    )
    assert joined.contents.state_slots.species_amounts() == pytest.approx({"P": 0.2})
    assert joined.contents.state_slots.coverages("catalyst_sites", "sites") == pytest.approx(
        {"adsorbate": 0.2, "vacant": 0.8}
    )
    assert joined.contents.state_slots.slots[0].spec.owner_id == "particles"
    assert joined.contents.state_slots.slots[0].location_id == "vial"


def test_unequal_carriers_mix_inventory_not_coverages():
    small = _portion(_active(state(support("small", capacity=1, bound=0.9))), 1, "vial")
    large = _portion(_active(state(support("large", capacity=9, bound=0.9))), 1, "vial")
    joined = _merge(small, large)
    coverage = joined.contents.state_slots.coverages("catalyst_sites", "sites")
    assert coverage["adsorbate"] == pytest.approx(0.18)
    assert {s.spec.owner_id for s in joined.contents.state_slots.slots} == {"small", "large"}


def test_mixing_temperature_uses_mobile_heat_capacity():
    a = _active(state(support("a", heat_capacity=10), temperature=300))
    b = _active(state(support("b", heat_capacity=90), temperature=400))
    joined = _merge(a, b)
    ca, cb = 41.8 + 10, 41.8 + 90
    assert joined.temperature_K == pytest.approx((ca * 300 + cb * 400) / (ca + cb))


def test_stationary_support_stays_and_thermalizes_with_new_material():
    old = state(support("fixed", fixed=True, heat_capacity=100), temperature=300, volume=0)
    incoming = _active(state(temperature=400))
    after = _install(old, incoming)
    assert after.state_slots == old.state_slots
    assert after.temperature_K == pytest.approx((100 * 300 + 41.8 * 400) / 141.8)


@pytest.mark.parametrize("port", ["out", "reactor_liquid"])
def test_component_outlet_leaves_fixed_support_and_moves_only_selected_material(port):
    before = state(support("fixed", fixed=True), support("mobile"))
    after, sample = _withdraw(before, port, 0.25, "receiver")
    fixed = next(s for s in after.state_slots.slots if s.spec.owner_kind == "equipment")
    assert fixed == before.state_slots.slots[0]
    assert all(s.spec.owner_kind == "material" for s in sample.contents.state_slots.slots)
    assert sample.contents.state_slots.species_amounts() == pytest.approx({"P": 0.05})
    assert after.state_slots.species_amounts() == pytest.approx({"P": 0.35})


def test_liquid_outlet_does_not_carry_solid_owned_state():
    before = state()
    before = before.replace(
        phases=PhaseLedger(
            {
                **before.phases.phases,
                "solid": PhaseRecord("solid", before.vessel_id, "solid", 0, {}),
            }
        ),
        state_slots=StateSlots((support("solid-carrier", phase="solid"),)),
    )
    after, sample = _withdraw(before, "reactor_liquid", 0.5, "receiver")
    assert after.state_slots == before.state_slots
    assert not sample.contents.state_slots.slots


def test_transfer_preserves_calibration_and_invalidates_observation_cache():
    before = state(support())
    equipment = upsert_equipment_record(
        before.equipment,
        equipment_id="batch_reactor",
        equipment_type="batch_reactor",
        attached_vessel_id=before.vessel_id,
        settings={"calibration_mol": 42, "stirring_speed_rpm": 731},
    )
    equipment = upsert_equipment_record(
        equipment,
        equipment_id="instrument:assay",
        equipment_type="instrument",
        attached_vessel_id=before.vessel_id,
        status="completed",
        settings={"calibration": 1.2, "use_count": 7},
    )
    before = before.replace(
        equipment=equipment,
        process=ProcessLedger(
            last_observation={"purity": 0.9}, last_observed_mask={"purity": True}
        ),
    )
    sample = _portion(_active(before), 0.5, "receiver")
    assert "calibration_mol" not in str(sample.contents.properties)
    receiver = state().replace(
        equipment=upsert_equipment_record(
            state().equipment,
            equipment_id="batch_reactor",
            equipment_type="batch_reactor",
            attached_vessel_id="batch_reactor",
            settings={"calibration_mol": 17, "stirring_speed_rpm": 400},
        )
    )
    installed = _install(receiver, sample)
    assert equipment_settings(installed.equipment, "batch_reactor")["calibration_mol"] == 17
    after = _install(before, sample)
    assert equipment_settings(after.equipment, "instrument:assay") == {
        "calibration": 1.2,
        "use_count": 7,
    }
    assert not after.process.last_observation
    assert before.process.last_observation == {"purity": 0.9}


def test_mechanism_and_phase_incompatibility_rejected():
    a = _active(state(support()))
    b = _active(state(support("b", mechanism="another-mechanism")))
    with pytest.raises(ValueError, match="mechanisms"):
        _merge(a, b)
    b = replace(
        a,
        contents=replace(
            a.contents,
            phases=PhaseLedger(
                {
                    "different": replace(
                        a.contents.phases.phases["reactor_liquid"], phase_id="different"
                    )
                }
            ),
        ),
    )
    with pytest.raises(ValueError, match="topology"):
        _merge(a, b)
    a = _active(state().replace(metadata={"mechanism_hash": "one"}))
    b = _active(state().replace(metadata={"mechanism_hash": "two"}))
    with pytest.raises(ValueError, match="mechanisms"):
        _merge(a, b)


def test_state_constraints_and_reset_rules():
    with pytest.raises(ValueError, match="exceeds"):
        support(capacity=1, bound=1.01)
    with pytest.raises(ValueError, match="Only material"):
        support(fixed=True).portion(0.5, "other")
    fixed = support("fixed", fixed=True)
    used = replace(fixed, values={**fixed.values, "adsorbate": 0.4})
    ledger = StateSlots((used, support()))
    assert ledger.reset().slots == (used,)
    assert ledger.reset(full=True).slots[0] == fixed
    moved = fixed.move("other-vessel")
    assert moved.spec.owner_id == fixed.spec.owner_id
    assert moved.values == fixed.values
    assert "adsorbate" not in json.dumps(ledger.to_dict(include_hidden=False))
    with pytest.raises(ValueError, match="owner is absent"):
        state(replace(fixed, spec=replace(fixed.spec, owner_id="missing")))


def test_explicit_intensive_mixing_and_public_projection():
    spec = StateSlotSpec(
        "load",
        "material",
        "lot",
        "inert_carrier",
        (
            StateFieldSpec("mass", "g"),
            StateFieldSpec(
                "specific_heat", "J/g/K", mixing="weighted", weight_field="mass", public=True
            ),
        ),
        batch_reset="clear",
    )
    a = StateSlot(spec, "vial", {"mass": 1, "specific_heat": 1})
    b = StateSlot(spec, "vial", {"mass": 9, "specific_heat": 3})
    result = a.merge(b)
    assert result.values["specific_heat"] == pytest.approx(2.8)
    assert result.to_dict(include_hidden=False) == {
        "specific_heat": {"value": 2.8, "unit": "J/g/K"}
    }


def test_world_snapshot_roundtrip_preserves_slots_samples_and_subsequent_move():
    before = state(support("fixed", fixed=True), support())
    outgoing = _portion(_active(before), 0.3, "container-01")
    before = _install(before, _portion(_active(before), 0.7, "active")).replace(
        samples=replace(
            before.samples,
            samples={outgoing.sample_id: outgoing},
            active_lineage=("lot-a",),
            containers_used=1,
            transfer_tools_used=1,
        )
    )
    snapshot = json.loads(json.dumps(before.to_dict()))
    restored = WorldState.from_dict(snapshot)
    assert json.loads(json.dumps(restored.to_dict())) == snapshot
    assert (
        _portion(_active(restored), 0.2, "next").to_dict()
        == _portion(_active(before), 0.2, "next").to_dict()
    )
    with pytest.raises(ValueError, match="full hidden"):
        WorldState.from_dict(before.to_dict(include_hidden=False))


def test_legacy_sample_read_converts_equipment_only_at_boundary():
    sample = _active(state())
    old = asdict(sample)
    old["contents"].pop("properties")
    old["contents"].pop("state_slots")
    old["contents"]["equipment"] = {
        "equipment": {
            "batch_reactor": {
                "equipment_id": "batch_reactor",
                "equipment_type": "batch_reactor",
                "attached_vessel_id": "original",
                "settings": {"solvent": 0, "calibration_mol": 23},
            },
            "instrument:assay": {
                "equipment_id": "instrument:assay",
                "equipment_type": "instrument",
                "attached_vessel_id": "original",
                "settings": {"use_count": 8},
            },
        }
    }
    restored = StoredSample.from_dict(json.loads(json.dumps(old)))
    assert restored.contents.properties == {"batch_reactor": {"solvent": 0}}
    assert "equipment" not in restored.to_dict()["contents"]


def test_real_gym_transfer_conserves_carriers_and_bound_species_and_rejects_creation():
    env = make_env()
    try:
        for action in cycle_actions()[1:8]:
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed"
        before = env.unwrapped._state
        # Use a registered species so the same elemental checker also counts the
        # bound inventory. This is a material-routing fixture, not a new reaction.
        species_id = next(iter(before.species_amounts))
        slots = []
        for slot in (support("fixed", fixed=True), support("mobile", phase="solid")):
            fields = tuple(
                replace(f, species_id=species_id) if f.species_id else f for f in slot.spec.fields
            )
            slots.append(replace(slot, spec=replace(slot.spec, fields=fields)))
        before = before.replace(state_slots=StateSlots(tuple(slots)))
        env.unwrapped._state = before
        inventory = PhysicalConstitution.material_inventory(before)
        owned = PhysicalConstitution.owned_inventory(before)
        for action in (
            create(1),
            transfer("active", "container-01", 0.25),
            transfer("container-01", "active", mixing=1),
        ):
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed", info.get("rollback_reason")
            after = env.unwrapped._state
            assert PhysicalConstitution.material_inventory(after) == pytest.approx(inventory)
            assert PhysicalConstitution.owned_inventory(after) == pytest.approx(owned)
        assert after.state_slots.coverages("catalyst_sites", "sites")["adsorbate"] == pytest.approx(
            0.2
        )
        changed = replace(
            after.state_slots.slots[0], values={**after.state_slots.slots[0].values, "sites": 2}
        )
        bad = after.replace(state_slots=StateSlots((changed, *after.state_slots.slots[1:])))
        result = TransactionManager(env.unwrapped.constitution).commit(
            state=after,
            operation_type="wait",
            events=(),
            patches=(StatePatch("replace_state", ("state_slots",), bad),),
        )
        assert result.transaction_status == "rolled_back"
        assert result.state.state_slots == after.state_slots
    finally:
        env.close()


def test_real_network_routing_and_restored_state_take_the_same_next_step():
    env = ChemWorldEnv(composition=request())
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "solvent": 0, "volume_L": 0.03})
        env.step({"operation": "add_reagent", "amount_mol": 0.01})
        before = env._state
        species = next(iter(before.species_amounts))
        slots = [support("mobile", location="a"), support("fixed", fixed=True, location="a")]
        slots = [
            replace(
                s,
                spec=replace(
                    s.spec,
                    fields=tuple(
                        replace(f, species_id=species) if f.species_id else f for f in s.spec.fields
                    ),
                ),
            )
            for s in slots
        ]
        before = before.replace(state_slots=StateSlots(tuple(slots)))
        restored = WorldState.from_dict(json.loads(json.dumps(before.to_dict())))
        action = {
            "operation": "route_material",
            "connection": 0,
            "transfer_fraction": 0.5,
            "mixing": 0,
        }
        first = env.runtime.apply_transaction(before, action)
        second = env.runtime.apply_transaction(restored, action)
        assert first.kernel_result.transaction_status == "committed"
        assert second.kernel_result.transaction_status == "committed"
        after = first.kernel_result.state
        assert after.to_dict() == second.kernel_result.state.to_dict()
        assert after.state_slots.stationary() == before.state_slots.stationary()
        assert after.inactive_vessels["b"].state_slots.stationary() == (
            before.inactive_vessels["b"].state_slots.stationary())
        assert after.inactive_vessels["b"].state_slots.species_amounts() == pytest.approx(
            {species: 0.1}
        )
    finally:
        env.close()
