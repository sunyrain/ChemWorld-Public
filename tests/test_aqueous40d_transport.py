"""Finite spatial chemistry, ownership and numerical boundaries must remain real."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.native_aqueous_reactive_transport import example_inputs
from chemworld.physchem.aqueous_electrodiffusion import zero_current_aqueous_face
from chemworld.physchem.aqueous_reactive_transport import (
    AqueousReactiveTransportSpec,
    advance_aqueous_reactive_transport,
    aqueous_transport_faces,
    prepare_aqueous_transport_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_reactive_transport import AqueousReactiveTransportRuntime
from chemworld.world.aqueous_reactive_transport_world import build_aqueous_reactive_transport_world


def test_unequal_mobility_flux_is_ambipolar_with_zero_current_and_dissipation():
    row = zero_current_aqueous_face(
        {"p": 20.0, "m": 20.0},
        {"p": 10.0, "m": 10.0},
        {"p": 1, "m": -1},
        {"p": 2e-9, "m": 1e-9},
        distance_m=0.001,
        area_m2=0.001,
        temperature_K=298.15,
    )
    assert row["amount_rates_mol_s"]["p"] == pytest.approx(4e-9 / 3 * 10, rel=1e-12)
    assert row["amount_rates_mol_s"]["p"] == pytest.approx(
        row["amount_rates_mol_s"]["m"], rel=1e-12
    )
    assert abs(row["current_molar_charge_rate_mol_s"]) < 1e-22
    assert row["dissipation_W"] > 0
    assert row["potential_difference_V"] != 0


def test_codec_and_reverse_rate_use_common_thermodynamics():
    spec, _ = example_inputs("slow", 1)
    restored = AqueousReactiveTransportSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()
    assert restored.reverse_rates()["precursor"] == pytest.approx(0.016)


@pytest.mark.parametrize("kind", ["salt", "fast", "slow"])
def test_actual_spatial_exchange_has_fixed_water_and_same_source_heat(kind):
    spec, amounts = example_inputs(kind)
    result = advance_aqueous_reactive_transport(spec, amounts, 2)
    final = result.final
    assert final["amounts_by_vessel"] != amounts
    assert final["boundary_amount_rates_mol_s"]["left"] == dict.fromkeys(amounts["cell-00"], 0)
    assert abs(sum(r["transported_enthalpy_J"] for r in final["cells"].values())) < 1e-13
    assert final["bath_heat_J"] == pytest.approx(
        sum(r["internal_energy_change_J"] for r in final["cells"].values()), abs=1e-12
    )
    for name in spec.cell_vessel_ids:
        assert final["amounts_by_vessel"][name]["H2O"] == amounts[name]["H2O"]
    assert result.diagnostics["rhs_evaluations"] > 0


@pytest.mark.parametrize("mode", ["nonneutral", "oneside", "allzero", "volume"])
def test_invalid_real_inventory_is_not_repaired(mode):
    spec, n = example_inputs("salt")
    original = json.dumps(n, sort_keys=True)
    if mode == "nonneutral":
        n["cell-00"]["M+"] *= 2
    elif mode == "oneside":
        n["cell-00"]["M+"] = n["cell-00"]["A-"] = 0
    elif mode == "allzero":
        for cell in n.values():
            cell["M+"] = cell["A-"] = 0
    else:
        n["cell-00"]["H2O"] *= 2
    changed = json.dumps(n, sort_keys=True)
    assert changed != original
    with pytest.raises(ValueError):
        prepare_aqueous_transport_state(spec, n)
    assert json.dumps(n, sort_keys=True) == changed


def test_world_wait_and_exact_json_replay_use_original_cell_phases():
    spec, n = example_inputs("slow")
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
    restored = None
    try:
        runtime.reset(seed=40)
        first = runtime.execute({"operation": "wait", "duration_s": 1})
        assert first.info["transaction_status"] == "committed", first.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for target in (runtime, restored):
            step = target.execute({"operation": "wait", "duration_s": 1})
            assert step.info["transaction_status"] == "committed", step.info
        assert runtime._state.to_dict() == restored._state.to_dict()
        assert runtime._state.ledger.time_s == 2
        locals_ = {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}
        for name in spec.cell_vessel_ids:
            assert (
                locals_[name].phases.phases["aqueous"].species_amounts_mol
                == locals_[name].species_amounts
            )
            assert locals_[name].species_amounts["H2O"] == n[name]["H2O"]
            assert not locals_[name].state_slots.slots
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_provider_failure_rolls_back_physical_state(monkeypatch):
    import chemworld.physchem.aqueous_reactive_transport as transport

    spec, n = example_inputs("slow")
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
    try:
        runtime.reset(seed=40)
        original = runtime._state

        def failure(*args, **kwargs):
            raise ValueError("injected fast solver failure")

        monkeypatch.setattr(transport, "solve_aqueous_reaction_equilibrium", failure)
        step = runtime.execute({"operation": "wait", "duration_s": 1})
        assert step.info["transaction_status"] != "committed"
        assert runtime._state.species_amounts == original.species_amounts
        assert runtime._state.inactive_vessels == original.inactive_vessels
        assert runtime._state.ledger.time_s == original.ledger.time_s
        assert runtime._state.ledger.energy_jacket_J == original.ledger.energy_jacket_J
    finally:
        runtime.close()


def test_no_owner_does_not_interfere_with_other_worlds():
    from types import SimpleNamespace

    AqueousReactiveTransportRuntime(SimpleNamespace(physics=None, components=())).validate_world()
    AqueousReactiveTransportRuntime(
        SimpleNamespace(physics=SimpleNamespace(initial_states={}), components=())
    ).validate_world()


def test_declared_t_domain_and_nonzero_solute_volume_are_rejected():
    spec, _ = example_inputs()
    with pytest.raises(ValueError):
        replace(spec, temperature_K=400)
    db = spec.equilibrium.database
    changed = replace(
        db,
        species=tuple(
            replace(s, standard_state=replace(s.standard_state, molar_volume_m3_mol=1e-6))
            if s.species_id == "HA"
            else s
            for s in db.species
        ),
    )
    with pytest.raises(ValueError, match="zero solute"):
        replace(spec, equilibrium=replace(spec.equilibrium, database=changed))


def test_spatial_orientation_changes_flux_sign_without_inventing_charge():
    spec, n = example_inputs("salt", 1)
    first = aqueous_transport_faces(spec, n)["faces"]
    reverse = {
        name: n[other]
        for name, other in zip(spec.cell_vessel_ids, reversed(spec.cell_vessel_ids), strict=True)
    }
    second = aqueous_transport_faces(spec, reverse)["faces"]
    for a, b in zip(first.values(), reversed(list(second.values())), strict=True):
        assert a["potential_difference_V"] == pytest.approx(-b["potential_difference_V"])
        assert a["amount_rates_mol_s"]["M+"] == pytest.approx(-b["amount_rates_mol_s"]["M+"])


def test_nonanchor_cells_reject_external_packet_and_routing():
    from chemworld.runtime.aqueous_reactive_transport import aqueous_transport_members
    from chemworld.runtime.thermal_material_routing import exchange_packet, route_gas_packet

    spec, n = example_inputs("salt")
    physics = build_aqueous_reactive_transport_world(spec, n).physics
    assert aqueous_transport_members(physics) == set(spec.cell_vessel_ids)
    cell = physics.initial_states["cell-01"]
    outside = cell.replace(vessel_id="outside")
    for source, target in ((cell, outside), (outside, cell)):
        with pytest.raises(ValueError, match="sealed aqueous"):
            exchange_packet(source, target, physics, {"M+": 1e-8, "A-": 1e-8})
    with pytest.raises(ValueError, match="sealed aqueous"):
        route_gas_packet(cell, physics, {})


def test_public_session_rejects_solvent_and_material_mutations_for_members():
    from chemworld.world.domain_modules import DomainModuleRegistry

    spec, n = example_inputs("salt")
    composition = build_aqueous_reactive_transport_world(spec, n)
    requested = (
        "wait",
        "add_solvent",
        "route_material",
        "transfer_material",
        "select_vessel",
        "terminate",
    )
    for name in spec.cell_vessel_ids:
        assert composition.physics.available_operations(
            requested, DomainModuleRegistry.default(), name
        ) == ("wait", "select_vessel", "terminate")
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=40)
        for operation in ("add_solvent", "route_material", "transfer_material"):
            before = runtime._state
            step = runtime.execute({"operation": operation, "solvent": 0, "volume_L": 1e-8})
            assert step.info["transaction_status"] != "committed"
            assert runtime._state.species_amounts == before.species_amounts
            assert runtime._state.inactive_vessels == before.inactive_vessels
            assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("change", ["temperature", "water", "owner"])
def test_snapshot_checks_current_complete_mesh_when_anchor_is_inactive(change):
    from chemworld.foundation.state_ledgers import EquipmentLedger
    from chemworld.runtime.aqueous_reactive_transport import install_aqueous_cell

    spec, n = example_inputs("salt")
    runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
    try:
        runtime.reset(seed=40)
        step = runtime.execute({"operation": "select_vessel", "vessel": 1})
        assert step.info["transaction_status"] == "committed"
        good = runtime.snapshot().to_dict()
        restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(good)))
        try:
            assert restored._state.to_dict() == runtime._state.to_dict()
        finally:
            restored.close()
        state = runtime._state
        if change == "temperature":
            state = state.replace(temperature_K=297.0)
        elif change == "water":
            amounts = {**state.species_amounts, "H2O": state.species_amounts["H2O"] * 1.1}
            state = install_aqueous_cell(state, amounts, state.volume_L)
        else:
            owner = state.inactive_vessels[spec.anchor_vessel_id]
            key, record = next(iter(owner.equipment.equipment.items()))
            settings = json.loads(json.dumps(record.settings))
            settings["physical_model"]["coefficients"][0]["value_m2_s"] *= 2
            owner = owner.replace(
                equipment=EquipmentLedger({key: replace(record, settings=settings)})
            )
            state = state.replace(
                inactive_vessels={**state.inactive_vessels, spec.anchor_vessel_id: owner}
            )
        good["world_state"] = state.to_dict()
        with pytest.raises(ValueError):
            ChemWorldRuntime.from_snapshot(good)
    finally:
        runtime.close()
