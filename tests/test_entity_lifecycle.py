"""Entity lifetime and serialized continuations through the actual session."""

import json
import os
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.campaign_resources import CampaignResourceCard
from chemworld.foundation.lifecycle import initialize_entities
from chemworld.foundation.state_ledgers import EquipmentLedger
from chemworld.foundation.state_ownership import (
    StateFieldSpec,
    StateSlot,
    StateSlots,
    StateSlotSpec,
)
from chemworld.runtime import ChemWorldRuntime, SessionSnapshot
from chemworld.world.domain_modules import DomainModuleRegistry
from chemworld.world.surface_world import surface_world_spec


def session_for(*, mobile=False, terminal_assay=False, campaign=False, noise="sequential"):
    spec = surface_world_spec(mobile=mobile, terminal_assay=terminal_assay)
    if campaign:
        spec = replace(spec, task=replace(spec.task, episode_mode="campaign"))
    session = ChemWorldRuntime(composition=spec, observation_noise_mode=noise)
    session.reset(seed=12)
    return session


def execute(session, **action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def feed_and_measure(session):
    for name, amount in (("CO-feed", 2e-5), ("O2-feed", 1e-5)):
        execute(session, operation="add_component", component=name, amount_mol=amount)
    return execute(session, operation="measure", instrument="gas_composition")


@pytest.mark.parametrize("assay", [False, True])
def test_native_campaign_retains_fixed_sites_devices_and_finite_supplies(assay):
    session = session_for(campaign=True, terminal_assay=assay)
    feed_and_measure(session)
    slots = session._state.state_slots
    supplies = session._state.inactive_vessels["co_supply"].species_amounts
    if assay:
        execute(session, operation="terminate")
        result = execute(session, operation="measure", instrument="gas_final_assay")
    else:
        result = execute(session, operation="terminate")
    assert result.info["experiment_completed"] and result.info["next_experiment_ready"]
    assert not result.terminated
    assert session._state.state_slots == slots
    assert session._state.inactive_vessels["co_supply"].species_amounts == supplies
    instrument = session._state.equipment.equipment["instrument:gas_composition"]
    assert instrument.status == "idle" and instrument.settings["use_count"] == 1
    assert session._state.samples.batch_generation == 1
    assert session._state.process.last_observation_record is None
    next_reading = feed_and_measure(session).observation
    assert next_reading.channels[0].sampled_at.clock_id == "vessel:reactor:batch:1"
    assert session._state.equipment.equipment[instrument.equipment_id].settings["use_count"] == 2
    assert session._state.inactive_vessels["co_supply"].species_amounts["CO"] == pytest.approx(6e-5)
    if assay:
        execute(session, operation="terminate")
        assert execute(session, operation="measure", instrument="gas_final_assay").info[
            "experiment_completed"
        ]


def test_mobile_carrier_is_discarded_by_its_declared_batch_rule():
    session = session_for(mobile=True, campaign=True)
    feed_and_measure(session)
    assert session._state.state_slots.material().has_inventory()
    execute(session, operation="terminate")
    assert not session._state.state_slots.material().has_inventory()
    assert session._lifecycle_history[-1]["vessels"]["reactor"]["previous_slots"]
    assert session._lifecycle_history[-1]["vessels"]["reactor"]["initialized_slots"] == {}


def test_partial_initialization_preserves_evidence_budget_and_stationary_state():
    session = session_for()
    feed_and_measure(session)
    before = session._state
    count, seed = session._operation_id, session.seed
    session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert session._operation_id == count and session.seed == seed
    assert session._state.ledger == before.ledger
    assert session._state.state_slots == before.state_slots
    assert session._state.inactive_vessels["co_supply"] == before.inactive_vessels["co_supply"]
    assert session._state.process.last_observation_record is None
    instrument = session._state.equipment.equipment["instrument:gas_composition"]
    assert instrument.settings["use_count"] == 1
    assert session._lifecycle_history[-1]["kind"] == "experiment_boundary_initialization"
    with pytest.raises(ValueError, match="cannot change"):
        session.reset(seed=2, options={"scope": "material"})


def test_vessel_replacement_resets_vessel_slots_but_not_device_slots():
    session = session_for()
    initial = session._state
    vessel_slot = StateSlot(
        StateSlotSpec("wall", "vessel", "reactor", "wall_state",
                      (StateFieldSpec("usage", "s"),)),
        "reactor", {"usage": 0},
    )
    initial = initial.replace(state_slots=StateSlots((*initial.state_slots.slots, vessel_slot)))
    old = initial.replace(state_slots=StateSlots((
        *initial.state_slots.slots[:-1], replace(vessel_slot, values={"usage": 20}),
    )))
    batch = initialize_entities(old, initial, scope="batch").state
    vessel = initialize_entities(old, initial, scope="vessel").state
    assert batch.state_slots.slots[-1].values["usage"] == 20
    assert vessel.state_slots.slots[-1].values["usage"] == 0
    assert batch.state_slots.slots[0] == vessel.state_slots.slots[0]


def test_replacement_device_has_new_identity_and_does_not_reappear_on_next_batch():
    session = session_for(campaign=True)
    before = session._state
    old = before.equipment.equipment["reactor:heater"]
    replacement = replace(old, equipment_id="replacement:heater")
    session.reset(options={
        "scope": "equipment", "equipment_id": old.equipment_id,
        "replacement": replacement.to_dict(),
    })
    assert session._state.species_amounts == before.species_amounts
    assert old.equipment_id not in session._state.equipment.equipment
    assert replacement.equipment_id in session._state.equipment.equipment
    execute(session, operation="heat", target_temperature_K=380, duration_s=1,
            stirring_speed_rpm=600)
    assert before.temperature_K < session._state.temperature_K < 380
    feed_and_measure(session)
    execute(session, operation="terminate")
    assert old.equipment_id not in session._state.equipment.equipment
    assert replacement.equipment_id in session._state.equipment.equipment
    session.reset(seed=12)
    assert old.equipment_id in session._state.equipment.equipment
    assert replacement.equipment_id not in session._state.equipment.equipment
    assert not session._lifecycle_history


def test_calibration_is_preserved_while_legacy_batch_material_settings_reset():
    session = ChemWorldRuntime(task_id="reaction-to-assay", episode_mode_override="campaign")
    session.reset(seed=2)
    execute(session, operation="add_solvent", volume_L=.02, solvent=0)
    execute(session, operation="add_reagent", amount_mol=.01)
    execute(session, operation="measure", instrument="hplc")
    records = dict(session._state.equipment.equipment)
    inst = records["instrument:hplc"]
    records[inst.equipment_id] = replace(inst, settings={**inst.settings, "gain": 1.05})
    session._state = session._state.replace(equipment=EquipmentLedger(records))
    execute(session, operation="terminate")
    execute(session, operation="measure", instrument="final_assay")
    settings = session._state.equipment.equipment[inst.equipment_id].settings
    assert settings["gain"] == 1.05 and settings["use_count"] == 1
    assert session._state.species_amounts == session._fresh_initial_state().species_amounts


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_json_snapshot_restores_across_processes_with_exact_future_reading(tmp_path, noise):
    session = session_for(campaign=True, noise=noise)
    feed_and_measure(session)
    execute(session, operation="terminate")
    feed_and_measure(session)
    path, output = tmp_path / "session.json", tmp_path / "continued.json"
    snapshot = session.snapshot()
    snapshot.save(path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["continuation"]["_lifecycle_history"]
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
    action = {"operation": "measure", "instrument": "gas_composition"}
    expected = session.execute(action)
    assert restored.execute(action) == expected
    code = (
        "import json,sys; from chemworld.runtime import ChemWorldRuntime,SessionSnapshot; "
        "s=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1])); "
        "r=s.execute({'operation':'measure','instrument':'gas_composition'}); "
        "from pathlib import Path; Path(sys.argv[2]).write_text(json.dumps("
        "{'observation':r.observation.native_record(),'state':s._state.to_dict()},allow_nan=False))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(path), str(output)],
        cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    actual = json.loads(output.read_text())
    assert actual == json.loads(json.dumps({
        "observation": expected.observation.native_record(), "state": session._state.to_dict(),
    }))


def test_snapshot_rejects_changed_module_contract_and_unknown_version():
    session = session_for()
    snapshot = session.snapshot()
    modules = DomainModuleRegistry(tuple(
        replace(module, version="changed") for module in session.domain_modules.modules
    ))
    with pytest.raises(ValueError, match="execution contracts changed"):
        ChemWorldRuntime.from_snapshot(snapshot, domain_modules=modules)
    payload = snapshot.to_dict()
    payload["schema_version"] = "unrecognized-version"
    with pytest.raises(ValueError, match="explicit migration"):
        SessionSnapshot.from_dict(payload)


def test_serialized_resource_events_are_restored_without_double_charge(tmp_path):
    card = CampaignResourceCard(
        card_id="persisted-ledger", operation_attempt_limit=10, vessel_start_limit=2,
        final_assay_limit=2, nonfinal_instrument_use_limit=2,
        stock_limits={"solvent_L": .08, "reagent_mol": .04},
    )
    session = ChemWorldRuntime(
        task_id="reaction-to-assay", episode_mode_override="campaign",
        campaign_resource_card=card,
    )
    session.reset(seed=3)
    execute(session, operation="add_solvent", solvent=0, volume_L=.02)
    execute(session, operation="add_reagent", amount_mol=.01)
    path = tmp_path / "resources.json"
    session.snapshot().save(path)
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
    assert restored.campaign_resource_snapshot() == session.campaign_resource_snapshot()
    for action in [
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]:
        assert restored.execute(action) == session.execute(action)
    assert restored.campaign_resource_snapshot() == session.campaign_resource_snapshot()
