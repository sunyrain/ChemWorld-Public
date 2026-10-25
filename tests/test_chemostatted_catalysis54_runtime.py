"""Original installed actions, owned inventories, source binding and atomic failures."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.chemostatted_catalysis import example_world
from chemworld.foundation.state_ledgers import EquipmentLedger, PhaseLedger
from chemworld.foundation.state_ownership import StateSlots
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.chemostatted_catalysis import (
    ELECTRICAL_STATE_TYPE,
    ChemostattedCatalysisRuntime,
    unpack_chemostatted_catalysis,
)
from reference.chemostatted_catalysis54_reference import independent


def runtime(kind="her"):
    session = ChemWorldRuntime(composition=example_world(kind=kind, potential=-0.2))
    session.reset()
    return session


@pytest.mark.parametrize("kind", ("volmer", "her", "hydroxylation"))
def test_installed_actions_retain_real_inventory_and_owned_receipts(kind):
    session = runtime(kind)
    initial = session._state
    assert initial.species.initial_amounts_mol == initial.species_amounts
    assert initial.solvent_accounting.initial.volume_L == 0.01
    for action in (
        {"operation": "configure_electrical_control", "potential_V": -0.25},
        {"operation": "wait", "duration_s": 20},
        {"operation": "measure", "instrument": "electrical_meter"},
        {"operation": "measure", "instrument": "chemostatted_coverage"},
    ):
        result = session.execute(action)
        assert result.info["transaction_status"] == "committed", result.info
    final = session._state
    assert final.phases.phases["electrolyte"] == initial.phases.phases["electrolyte"]
    assert len(final.state_slots.slots) == 3
    assert final.ledger.time_s == 22
    _, _, _, _, physical, _, _ = unpack_chemostatted_catalysis(final)
    expected = independent(kind, 298.15, 2, -0.25, 22)
    assert physical.coordinates["faradaic_charge_C"] == pytest.approx(
        expected["faradaic_charge_C"], rel=3e-7, abs=1e-10
    )
    assert physical.exchanged_mol["reservoir:proton:in"] - physical.exchanged_mol[
        "reservoir:proton:out"
    ] == pytest.approx(expected["proton_net_in_mol"], rel=3e-7, abs=3e-15)
    assert final.ledger.flow_electrical_work_J == physical.coordinates["electrical_work_J"]
    assert final.ledger.energy_jacket_J == pytest.approx(
        expected["bath_heat_J"], rel=3e-7, abs=1e-9
    )
    coverage_key = "coverage_hydroxyl_bound" if kind == "hydroxylation" else "coverage_H_ads"
    assert result.observation.channel(coverage_key).value == pytest.approx(
        expected["theta"], rel=3e-7, abs=3e-10
    )
    encoded = json.loads(json.dumps(session.snapshot().to_dict()))
    clone = ChemWorldRuntime.from_snapshot(encoded)
    assert json.loads(json.dumps(clone.snapshot().to_dict())) == encoded


@pytest.mark.parametrize(
    "fields",
    ({"potential_V": -0.1, "ramp_duration_s": 5}, {"current_A": 0.001}, {"potential_V": 0.2}),
)
def test_unsupported_control_leaves_physical_state_atomic(fields):
    session = runtime()
    before = session._state
    result = session.execute({"operation": "configure_electrical_control", **fields})
    assert result.info["transaction_status"] != "committed"
    after = session._state
    assert after.species_amounts == before.species_amounts
    assert after.phases == before.phases
    assert after.state_slots == before.state_slots
    assert after.equipment == before.equipment
    assert after.ledger.time_s == before.ledger.time_s
    assert after.ledger.flow_electrical_work_J == before.ledger.flow_electrical_work_J
    assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J


def corrupt(state, kind):
    if kind == "bulk":
        return state.replace(species_amounts={**state.species_amounts, "proton": 2e-4})
    if kind == "liquid":
        liquid = state.phases.phases["electrolyte"]
        return state.replace(
            phases=PhaseLedger(
                {
                    **state.phases.phases,
                    "electrolyte": replace(
                        liquid, species_amounts_mol={"proton": 2e-4, "chloride": 1e-4}
                    ),
                }
            )
        )
    if kind == "slot_owner":
        return state.replace(
            state_slots=StateSlots(
                tuple(
                    replace(s, spec=replace(s.spec, owner_id="heater"))
                    if s.spec.state_type == ELECTRICAL_STATE_TYPE
                    else s
                    for s in state.state_slots.slots
                )
            )
        )
    if kind == "capacity":
        return state.replace(
            state_slots=StateSlots(
                tuple(
                    replace(s, values={**s.values, "capacity:metal": 2e-6})
                    if "capacity:metal" in s.values
                    else s
                    for s in state.state_slots.slots
                )
            )
        )
    if kind in {"source", "bath", "thermal_feedback", "bath_role"}:
        name = "working-electrode" if kind == "source" else "heater"
        equipment = state.equipment.equipment[name]
        settings = json.loads(json.dumps(dict(equipment.settings)))
        if kind == "source":
            settings["physical_model"]["support_heat_capacity_J_K"] += 1
        elif kind == "bath":
            settings["thermal_device"]["enabled"] = False
        elif kind == "thermal_feedback":
            settings["thermal_feedback"] = {}
        else:
            settings["lab_role"] = "movable"
        return state.replace(
            equipment=EquipmentLedger(
                {**state.equipment.equipment, name: replace(equipment, settings=settings)}
            )
        )
    raise AssertionError(kind)


@pytest.mark.parametrize(
    "kind",
    ("bulk", "liquid", "slot_owner", "capacity", "source", "bath", "thermal_feedback", "bath_role"),
)
def test_invalid_original_state_is_rejected_without_material_or_ledger_mutation(kind):
    world = example_world()
    original = world.physics.initial_states["reactor"]
    invalid = corrupt(original, kind)
    assert invalid.to_dict() != original.to_dict()
    before = invalid.to_dict()
    result = ChemostattedCatalysisRuntime(world).advance(
        invalid, {"operation": "wait", "duration_s": 2}
    )
    assert not result.success
    assert result.state is None
    assert result.failure_reason == "chemostatted_catalysis_failed"
    if kind == "thermal_feedback":
        assert "does not support thermal feedback" in result.diagnostics["solver_error"]
    assert invalid.to_dict() == before


def test_direct_heating_cannot_override_declared_isothermal_half_cell():
    world = example_world()
    initial = world.physics.initial_states["reactor"]
    result = ChemostattedCatalysisRuntime(world).advance(
        initial, {"operation": "wait", "duration_s": 2}, heat=True
    )
    assert result.outcome == "rejected"
    assert result.failure_reason == "chemostatted_catalysis_requires_declared_isotherm"
    assert result.state is None
