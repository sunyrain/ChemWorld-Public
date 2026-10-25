"""Empirical reaction budgets cannot bypass original inventory transactions."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from chemworld.foundation.balances import (
    MaterialExchange,
    TransitionBalance,
    inventory_balance_exclusions,
)
from chemworld.foundation.empirical_inventory import (
    EMPIRICAL_BALANCE_SCHEMA,
    check_empirical_transitions,
    validate_empirical_basis,
    validate_empirical_transition,
)
from chemworld.foundation.state_ledgers import EquipmentLedger, EquipmentRecord
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.foundation.units import convert_value


def basis():
    return {
        "schema": EMPIRICAL_BALANCE_SCHEMA,
        "mechanism_id": "declared-test",
        "member_vessel_ids": ["reactor"],
        "quantities": {k: {"unit": "kg", "basis": b}
                       for k, b in (("substrate", "N"), ("product", "N"), ("oxygen", "O2"))},
        "reactions": {"growth": {"unit": "kg", "basis": "N", "stoichiometry": {
            "substrate": -1., "product": 1., "oxygen": -2.}}},
        "inventory_fields": {"empirical_material": {
            "substrate": "substrate", "product": "product", "dissolved": "oxygen"},
            "empirical_utility": {"supply": "oxygen", "offgas": "oxygen"}},
        "extent_state_type": "empirical_progress",
        "invariants": {"nitrogen": {"unit": "kg", "coefficients": {
            "substrate": 1., "product": 1.}}},
    }


def state(*, final=False):
    definitions = (
        ("material", "material", "lot", "empirical_material",
         {"substrate": .75 if final else 1., "product": .25 if final else 0., "dissolved": 0.}),
        ("utility", "material", "gas-budget", "empirical_utility",
         {"supply": 1.5 if final else 2., "offgas": 0.}),
        ("progress", "equipment", "bioreactor", "empirical_progress",
         {"growth": .25 if final else 0.}),
    )
    slots = tuple(StateSlot(StateSlotSpec(
        identity, owner_kind, owner, kind, tuple(StateFieldSpec(k, "kg") for k in values),
        mechanism_id="declared-test"), "reactor", values)
        for identity, owner_kind, owner, kind, values in definitions)
    records = EquipmentLedger({"bioreactor": EquipmentRecord(
        "bioreactor", "empirical_reactor", attached_vessel_id="reactor",
        settings={"empirical_reaction_balance": basis()})})
    return SimpleNamespace(vessel_id="reactor", phase="liquid", inactive_vessels={},
                           state_slots=StateSlots(slots), equipment=records)


def change_slot(world, index, **values):
    slots = list(world.state_slots.slots)
    slots[index] = replace(slots[index], values={**slots[index].values, **values})
    world.state_slots = StateSlots(tuple(slots))


def test_reaction_and_internal_oxygen_transfer_close_without_fake_molecular_inventory():
    before, after = state(), state(final=True)
    change_slot(after, 0, dissolved=.2)
    change_slot(after, 1, supply=1.2, offgas=.1)
    check_empirical_transitions(before, after, TransitionBalance())
    assert all(not f.species_id for s in after.state_slots.slots for f in s.spec.fields)


def test_empirical_receipt_names_unmodeled_science_even_from_an_inactive_vessel():
    empirical = state()
    spectator = SimpleNamespace(vessel_id="spectator", phase="liquid",
                                equipment=EquipmentLedger(),
                                inactive_vessels={"reactor": empirical})
    exclusions = inventory_balance_exclusions(spectator)
    assert exclusions == ("empirical_full_element_inventory", "empirical_full_charge_inventory",
                          "empirical_biochemical_reaction_energy")
    receipt = TransitionBalance(unmodeled_quantities=exclusions).to_dict()
    assert set(exclusions) <= set(receipt["not_covered"])
    assert "elements_mol" in receipt["quantities"]  # Actual water still has its original ledger.
    spectator.inactive_vessels = {}
    assert inventory_balance_exclusions(spectator) == ()


@pytest.mark.parametrize("index,field", ((0, "substrate"), (0, "product"), (1, "supply")))
def test_missing_empirical_material_is_detected_at_the_real_slot_boundary(index, field):
    before, after = state(), state(final=True)
    change_slot(after, index, **{field: after.state_slots.slots[index].values[field] + 1e-4})
    with pytest.raises(ValueError, match="does not close"):
        check_empirical_transitions(before, after, TransitionBalance())


def test_external_entity_receipt_cannot_disguise_a_reaction_source():
    exchange = MaterialExchange("fake", "reservoir", "vessel:reactor", "in",
                                (("substrate", .1),), "kg", "entity")
    with pytest.raises(ValueError, match="disguised"):
        check_empirical_transitions(state(), state(final=True), TransitionBalance((exchange,)))


def test_source_basis_cannot_be_swapped_during_commit():
    before, after = state(), state(final=True)
    record = after.equipment.equipment["bioreactor"]
    changed = deepcopy(record.settings)
    changed["empirical_reaction_balance"]["reactions"]["growth"]["stoichiometry"]["oxygen"] = -1.
    after.equipment = EquipmentLedger({"bioreactor": replace(record, settings=changed)})
    with pytest.raises(ValueError, match="source declaration changed"):
        check_empirical_transitions(before, after, TransitionBalance())


def test_progress_cannot_move_backward_even_when_material_would_close():
    with pytest.raises(ValueError, match="progress cannot decrease"):
        check_empirical_transitions(state(final=True), state(), TransitionBalance())


def test_unknown_units_or_nonconserving_source_matrix_are_rejected():
    broken = basis()
    broken["quantities"]["oxygen"]["unit"] = "invented_oxygen_mol"
    with pytest.raises(ValueError):
        validate_empirical_basis(broken)
    broken = basis()
    broken["reactions"]["growth"]["stoichiometry"]["product"] = .9
    with pytest.raises(ValueError, match="linear invariant"):
        validate_empirical_basis(broken)


@pytest.mark.parametrize("value", (True, -1., float("nan")))
def test_progress_receipts_require_finite_nonnegative_numeric_extents(value):
    amounts = {"substrate": 1., "product": 0., "oxygen": 2.}
    with pytest.raises(ValueError, match="complete finite nonnegative"):
        validate_empirical_transition(amounts, amounts, {"growth": 0.}, {"growth": value}, basis())


def test_material_field_cannot_silently_change_units_or_owner():
    before, after = state(), state(final=True)
    slots = list(after.state_slots.slots)
    slots[0] = replace(slots[0], spec=replace(slots[0].spec, fields=tuple(
        replace(f, unit="mol") for f in slots[0].spec.fields)))
    after.state_slots = StateSlots(tuple(slots))
    with pytest.raises(ValueError, match="unit differs"):
        check_empirical_transitions(before, after, TransitionBalance())


def test_equivalents_cannot_be_reinterpreted_as_molecular_amount():
    assert convert_value(2., "mol_eq", "mol_eq") == 2.
    with pytest.raises(ValueError):
        convert_value(2., "mol_eq", "mol")
