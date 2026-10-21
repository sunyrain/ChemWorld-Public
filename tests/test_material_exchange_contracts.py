"""Executed amounts, selected material owners and common connection integrity."""

from dataclasses import replace

import pytest

from chemworld.foundation.balances import MaterialPacket
from chemworld.foundation.state_ownership import StateSlots
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.material_balance import ServiceMaterialBoundary
from chemworld.world.mineral_world import mineral_world_spec
from test_material_routing import make_env
from test_state_ownership import support


def test_molecular_packet_keeps_trace_inventory_and_rejects_even_tiny_overdraft():
    source = {"trace": 1e-30, "bulk": 1.0}
    packet = MaterialPacket("feed", "reactor", (("trace", 1e-30), ("bulk", 0.25)))
    left, right = packet.apply(source, {"trace": 0.0, "bulk": 0.0})
    assert left["trace"] == 0 and right["trace"] == 1e-30
    assert source == {"trace": 1e-30, "bulk": 1.0}
    assert MaterialPacket.from_dict(packet.to_dict()) == packet
    with pytest.raises(ValueError, match="exceeds"):
        MaterialPacket("feed", "reactor", (("trace", 2e-30),)).apply(source, {})


@pytest.mark.parametrize("amount", [True, -1, float("nan"), float("inf")])
def test_packet_rejects_nonphysical_or_boolean_quantities(amount):
    with pytest.raises(ValueError):
        MaterialPacket("feed", "reactor", (("A", amount),))


@pytest.mark.parametrize("fraction", [0.0, 0.25, 1.0])
def test_phase_split_and_return_preserve_unselected_material_and_stationary_owners(fraction):
    slots = StateSlots((
        support("fixed", fixed=True), support("mobile"), support("solid", phase="solid"),
    ))
    retained, moved = slots.split_material(
        fraction, "batch_reactor", "receiver", phase_id="reactor_liquid"
    )
    assert [s.spec.slot_id for s in moved.slots] == ["mobile"]
    unchanged = {s.spec.slot_id: s for s in retained.slots}
    assert unchanged["fixed"] == slots.slots[0]
    assert unchanged["solid"] == slots.slots[2]
    _, returned = moved.split_material(1.0, "receiver", "batch_reactor")
    joined = retained.receive_material(returned)
    assert joined.conserved() == pytest.approx(slots.conserved(), rel=1e-14, abs=0)
    assert joined.species_amounts() == pytest.approx(slots.species_amounts(), rel=1e-14, abs=0)
    assert {s.spec.slot_id for s in joined.slots} == {"fixed", "mobile", "solid"}


def test_real_sampling_removes_mobile_bound_inventory_and_retains_fixed_carrier_and_history():
    env = make_env()
    try:
        _, _, _, _, info = env.step(
            {"operation": "add_solvent", "solvent": 0, "volume_L": 0.03}
        )
        assert info["transaction_status"] == "committed"
        base = env.unwrapped
        before = base._state
        species = next(iter(before.species_amounts))
        slots = []
        for slot in (support("fixed", fixed=True), support("mobile")):
            fields = tuple(replace(f, species_id=species) if f.species_id else f
                           for f in slot.spec.fields)
            slots.append(replace(slot, spec=replace(slot.spec, fields=fields)))
        before = before.replace(state_slots=StateSlots(tuple(slots)))
        base._state = before
        _, _, _, _, info = env.step({"operation": "sample", "sample_volume_L": 0.001})
        assert info["transaction_status"] == "committed", info
        after = base._state
        indexed = {s.spec.slot_id: s for s in after.state_slots.slots}
        assert indexed["fixed"] == slots[0]
        assert indexed["mobile"].values["sites"] == pytest.approx(1 - 0.001 / 0.03)
        assert after.samples.to_dict() == before.samples.to_dict()
        balance = base._last_operation_record.material_balance
        assert base.constitution.check_transition(before, after, balance).passed
        owned = {key: amount for e in balance.exchanges if e.inventory_kind == "entity"
                 for key, amount in e.amounts}
        assert owned["single-site:sites"] == pytest.approx(0.001 / 0.03)
    finally:
        env.close()


def test_finite_feed_accounting_consumes_packet_without_reexecuting_source_law(monkeypatch):
    runtime = ChemWorldRuntime(composition=mineral_world_spec())
    try:
        runtime.reset(seed=7)
        before = runtime._state
        services = runtime.runtime.domain_services
        action = {"operation": "add_component", "component": "HCl-solution", "amount_mol": 0.01}
        after = services.material_routing.add_finite_feed(before, action).state

        def unexpected(*args, **kwargs):
            raise AssertionError("accounting must consume the executed material packet")

        monkeypatch.setattr(services.primitive, "feed_additions", unexpected)
        balance = ServiceMaterialBoundary(services).resolve(before, after, action)
        packet = MaterialPacket.from_dict(after.metadata["last_material_packet"])
        assert dict(balance.stock_consumption)["reagent_mol"] == packet.total_mol
        assert not balance.exchanges
        for bad in ({}, {**packet.to_dict(), "source": "invented"},
                    {**packet.to_dict(), "amounts_mol": {"H2O": 0.01}}):
            candidate = after.replace(metadata={**after.metadata, "last_material_packet": bad})
            with pytest.raises(ValueError):
                ServiceMaterialBoundary(services).resolve(before, candidate, action)
    finally:
        runtime.close()
