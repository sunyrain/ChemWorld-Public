"""Actual SPM sessions retain metal deactivation through the existing lifecycle."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.battery_partial_plating import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import F, battery_control_current, battery_readout
from chemworld.physchem.battery_plating import plating_fields
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def start():
    runtime = ChemWorldRuntime(composition=battery_world_spec(default_spec()))
    runtime.reset(seed=4)
    return runtime


def test_meter_time_moves_dead_lithium_without_an_extra_terminal_charge():
    runtime = start()
    try:
        _, _, spec, initial, _ = unpack_battery(runtime._state)
        execute(runtime, {"operation": "configure_electrical_control", "current_A": -.5})
        execute(runtime, {"operation": "wait", "duration_s": 3.})
        before_meter = unpack_battery(runtime._state)[3]
        measured = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
        record, slot, _, final, _ = unpack_battery(runtime._state)
        assert final.time_s == 4.
        assert final.coordinates["discharged_charge_C"] == pytest.approx(-2., abs=1e-9)
        assert initial.amounts_mol["dead:Li"] < before_meter.amounts_mol["dead:Li"]
        assert before_meter.amounts_mol["dead:Li"] < final.amounts_mol["dead:Li"]
        assert sum(final.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-11)
        negative = (*spec.negative.keys("negative"), "plating:Li", "dead:Li")
        assert sum(final.amounts_mol[k]-initial.amounts_mol[k] for k in negative) == (
            pytest.approx(2./F, abs=1e-11))
        fields = plating_fields(spec, final, current_A=-.5)
        assert fields["deactivation_rate_mol_s"] > 0
        assert fields["negative_intercalation_current_A"] + fields["stripping_current_A"] == (
            pytest.approx(-.5, abs=1e-9))
        assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
            -final.coordinates["delivered_work_J"], abs=1e-8)
        assert {c.channel_id for c in measured.observation.channels} == {
            "terminal_potential", "electrical_current"}
        assert record.equipment_type == "single_particle_battery"
        assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
        assert runtime._state.phases.phases["housing"].species_amounts_mol == {}
    finally:
        runtime.close()


def test_voltage_control_keeps_one_work_receipt_with_metal_deactivation():
    runtime = start()
    try:
        _, _, spec, initial, _ = unpack_battery(runtime._state)
        target = battery_readout(spec, initial, -.2).voltage_V
        execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
        execute(runtime, {"operation": "wait", "duration_s": 3.})
        _, _, _, final, program = unpack_battery(runtime._state)
        current = battery_control_current(spec, final, program)
        assert -spec.maximum_current_A < current < 0.
        assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=2e-6)
        assert final.coordinates["delivered_work_J"] == pytest.approx(
            target*final.coordinates["discharged_charge_C"], abs=1e-7)
        assert final.amounts_mol["dead:Li"] > initial.amounts_mol["dead:Li"]
    finally:
        runtime.close()


def test_pending_charge_rest_reverse_survives_fresh_process_and_material_reset(tmp_path):
    runtime = start()
    try:
        for current, duration in ((-.5, 3.), (0., 2.), (.3, 4.)):
            execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                              "stage_duration_s": duration})
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        saved, output = tmp_path/"prefix.json", tmp_path/"continued.json"
        saved.write_text(json.dumps(runtime.snapshot().to_dict()))
        program = """
import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
runtime = ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()))
try:
    result = runtime.execute({'operation':'wait','duration_s':9.})
    assert result.info['transaction_status'] == 'committed', result.info
    Path(sys.argv[2]).write_text(json.dumps(runtime._state.to_dict()))
finally:
    runtime.close()
"""
        subprocess.run([sys.executable, "-c", program, str(saved), str(output)],
                       check=True, capture_output=True, text=True)
        execute(runtime, {"operation": "wait", "duration_s": 9.})
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
        _, slot, _, state, _ = unpack_battery(runtime._state)
        assert state.time_s == 9.
        assert state.coordinates["discharged_charge_C"] == pytest.approx(-.3, abs=1e-8)
        assert state.amounts_mol["dead:Li"] > default_spec().plating.initial_dead_mol
        before = dict(slot.values)
        runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert dict(unpack_battery(runtime._state)[1].values) == before
    finally:
        runtime.close()


def test_equipment_replacement_installs_both_new_metal_stocks_once():
    runtime = start()
    try:
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        record, _, spec, _, _ = unpack_battery(runtime._state)
        new = replace(spec, plating=replace(
            spec.plating, initial_plated_mol=1e-5, initial_dead_mol=2e-5))
        replacement = replace(record, equipment_id="fresh-partial-cell", settings={
            **record.settings, "physical_model": new.to_dict()})
        slot = initial_battery_slot(replacement, new)
        runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                               "replacement": replacement, "slots": StateSlots((slot,))})
        _, actual, decoded, state, _ = unpack_battery(runtime._state)
        assert actual == slot and decoded == new
        assert state.amounts_mol == new.initial_state().amounts_mol
        assert state.time_s == 2.
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        final = unpack_battery(runtime._state)[3]
        assert final.time_s == 3. and final.amounts_mol["dead:Li"] > 2e-5
    finally:
        runtime.close()


def test_failed_advance_rolls_back_both_metals_pending_stages_and_time(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = start()
    try:
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": -.3,
                          "stage_duration_s": 3.})
        before = deepcopy(runtime._state)

        def fail(*args, **kwargs):
            raise RuntimeError("injected partial plating failure")

        monkeypatch.setattr(adapter, "advance_battery", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 2.})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.equipment == before.equipment
        assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_partial_workflow_replays_each_actual_step(tmp_path, noise):
    report = run(tmp_path/f"partial-plating-{noise}.jsonl", noise=noise)
    assert report["steps"] == report["replay_steps"] == 11
    assert report["mode"] == "partially reversible" and report["gamma_s_inv"] == .02


def test_shared_clock_advances_deactivation_once_and_restores_exactly():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=default_spec()))
    runtime.reset(seed=4)
    restored = None
    try:
        before = local_states(runtime._state)
        initial = unpack_battery(before["cell"])[3]
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, 2.)
        final = unpack_battery(local_states(runtime._state)["cell"])[3]
        assert sum(final.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-11)
        assert final.amounts_mol["dead:Li"] > initial.amounts_mol["dead:Li"]
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict())))
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        if restored is not None:
            restored.close()
        runtime.close()
