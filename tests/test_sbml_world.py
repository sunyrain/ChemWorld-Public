"""Actual existing world, wait coordinator and native pending-queue snapshot."""

import json
import subprocess
import sys

import pytest

from chemworld.physchem.sbml_system import SBMLObservable
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_models import model_record
from chemworld.world.sbml_world import sbml_world
from reference.run_sbml_trajectory_block import cases
from test_sbml import model


def test_actual_native_world_wait_uses_sole_clock_and_formal_entity_owner():
    runtime = ChemWorldRuntime(composition=sbml_world(model(), maximum_step_s=.02))
    runtime.reset(seed=4)
    before = runtime._state
    assert before.volume_L == 0 and before.species_amounts == {}
    assert before.liquid_heat_capacity_J_L_K is None
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "success"
    assert runtime._state.ledger.time_s == 3.
    record = model_record(runtime._state)
    assert record.settings["coordinates"]["P:p"] > 5.
    assert record.settings["continuation"]["events"]["next_sequence"] == 1
    assert runtime._state.species_amounts == {} and runtime._state.volume_L == 0.


@pytest.mark.parametrize("noise", [.01, .05])
@pytest.mark.parametrize("mode", ["sequential", "keyed"])
def test_declared_gauge_uses_original_model_views_without_advancing_truth(noise, mode):
    observables = (SBMLObservable("amount", "A"), SBMLObservable("concentration", "B"),
                   SBMLObservable("volume", "c1"))
    runtime = ChemWorldRuntime(composition=sbml_world(model(), maximum_step_s=.02,
        observables=observables, readout_relative_sd=noise), observation_noise_mode=mode)
    runtime.reset(seed=8)
    runtime.execute({"operation": "wait", "duration_s": 2.25})
    owned = model_record(runtime._state).settings
    result = runtime.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
    assert result.info["operation_outcome"] == "success"
    assert runtime._state.ledger.time_s == 2.25
    assert model_record(runtime._state).settings == owned
    reading = runtime.observe()
    assert {c.channel_id for c in reading.channels} == {
        "amount_A", "concentration_B", "volume_c1"}
    assert {c.unit for c in reading.channels} == {"mol", "mol/L", "L"}
    assert all(c.uncertainty > 0 for c in reading.channels)
    assert runtime._state.ledger.sample_consumed_L == 0
    assert result.info["transaction_status"] == "committed"


def test_pending_native_world_snapshot_resumes_in_a_new_process(tmp_path):
    runtime = ChemWorldRuntime(composition=sbml_world(model(), maximum_step_s=.02))
    runtime.reset(seed=4)
    runtime.execute({"operation": "wait", "duration_s": 2.25})
    assert len(model_record(runtime._state).settings["continuation"]["events"]["pending"]) == 1
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict(), allow_nan=False))
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    action = {"operation": "wait", "duration_s": .75}
    assert runtime.execute(action).info["operation_outcome"] == "success"
    assert restored.execute(action).info["operation_outcome"] == "success"
    assert runtime._state.ledger.time_s == 3.
    assert restored._state.to_dict() == runtime._state.to_dict()
    source, output = tmp_path/"snapshot.json", tmp_path/"state.json"
    source.write_text(json.dumps(snapshot), encoding="utf-8")
    program = '''import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
runtime=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
assert runtime.execute({"operation":"wait","duration_s":.75}).info["operation_outcome"]=="success"
Path(sys.argv[2]).write_text(json.dumps(runtime._state.to_dict(),allow_nan=False),encoding="utf-8")
'''
    subprocess.run([sys.executable, "-c", program, str(source), str(output)], check=True)
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        json.dumps(runtime._state.to_dict()))


def test_event_cascade_budget_failure_rolls_back_native_coordinates_queue_and_clock():
    text = list(cases())[15][3]
    payload = sbml_world(model(text), maximum_step_s=.02).to_dict()
    payload["physics"]["models"]["sbml"]["bindings"]["formal_workspace"]["maximum_events"] = 1
    runtime = ChemWorldRuntime(composition=payload)
    runtime.reset(seed=4)
    before = model_record(runtime._state).to_dict()
    result = runtime.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert model_record(runtime._state).to_dict() == before
    assert runtime._state.ledger.time_s == 0.


def test_public_state_and_transaction_do_not_expose_formal_latents_or_queue():
    runtime = ChemWorldRuntime(composition=sbml_world(model(), maximum_step_s=.02,
        observables=(SBMLObservable("amount", "A"),)))
    runtime.reset(seed=3)
    waited = runtime.execute({"operation": "wait", "duration_s": 2.25})
    public = runtime._state.to_dict(include_hidden=False)
    owner = public["equipment"]["formal_model"]
    assert set(owner["settings"]) == {"format", "model_binding"}
    assert "private_settings" not in owner
    assert "N:A" not in json.dumps(waited.info)
    exclusions = runtime._last_operation_record.material_balance.unmodeled_quantities
    assert {"formal_entity_element_inventory", "formal_entity_charge_inventory",
            "formal_entity_energy", "formal_compartment_carrier_mass"} <= set(exclusions)
    measured = runtime.execute({"operation": "measure", "instrument": "formal_quantity_meter"})
    assert measured.info["operation_outcome"] == "success"
    assert [c.channel_id for c in runtime.observe().channels] == ["amount_A"]
    assert "N:A" not in json.dumps(measured.info)
    full = runtime._state.to_dict()
    assert set(full["equipment"]["formal_model"]["settings"]["coordinates"]) == {
        "N:A", "N:B", "V:c1", "V:c2", "P:p"}
    assert runtime._state.state_slots.species_amounts() == {}


@pytest.mark.parametrize("mutation", ["extra_clock", "missing_coordinate", "negative_amount",
    "boolean_amount", "unknown_root", "overdue", "public_latents", "duplicate_owner"])
def test_saved_native_model_state_is_checked_before_restoring_any_continuation(mutation):
    runtime = ChemWorldRuntime(composition=sbml_world(model(), maximum_step_s=.02))
    runtime.reset(seed=2)
    runtime.execute({"operation": "wait", "duration_s": 2.25})
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    # SessionSnapshot stores a closed payload, including the one native world.
    payload = snapshot.get("payload", snapshot)
    equipment = payload["world_state"]["equipment"]
    record = equipment["formal_model"]
    settings = record["settings"]
    if mutation == "extra_clock":
        settings["time_s"] = 2.25
    elif mutation == "missing_coordinate":
        settings["coordinates"].pop("N:A")
    elif mutation == "negative_amount":
        settings["coordinates"]["N:A"] = -1.
    elif mutation == "boolean_amount":
        settings["coordinates"]["N:A"] = True
    elif mutation == "unknown_root":
        settings["continuation"]["root_values"] = {"invented": 0.}
    elif mutation == "overdue":
        settings["continuation"]["events"]["pending"][0]["due_time_s"] = 1.
    elif mutation == "public_latents":
        record["private_settings"] = []
    else:
        equipment["second_owner"] = {**record, "equipment_id": "second_owner"}
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(snapshot)


def test_initial_true_source_event_is_queued_before_the_first_native_wait():
    runtime = ChemWorldRuntime(composition=sbml_world(model(list(cases())[12][3]),
                                                        maximum_step_s=.02))
    runtime.reset(seed=1)
    memory = model_record(runtime._state).settings["continuation"]["events"]
    assert len(memory["pending"]) == 1
    assert memory["pending"][0]["due_time_s"] == .5
    result = runtime.execute({"operation": "wait", "duration_s": .75})
    assert result.info["operation_outcome"] == "success"
    assert model_record(runtime._state).settings["coordinates"]["P:p"] == pytest.approx(5.025)


def test_material_preparation_does_not_reset_the_formal_entity_or_its_pending_event():
    runtime = ChemWorldRuntime(composition=sbml_world(model(), maximum_step_s=.02))
    runtime.reset(seed=7)
    runtime.execute({"operation": "wait", "duration_s": 2.25})
    owned = model_record(runtime._state).to_dict()
    runtime.reset(options={"scope": "material", "vessel_ids": ["formal_workspace"]})
    assert model_record(runtime._state).to_dict() == owned
    assert runtime._state.ledger.time_s == 2.25
    assert runtime.execute({"operation": "wait", "duration_s": .75}).info[
        "operation_outcome"] == "success"
    assert runtime._state.ledger.time_s == 3.
