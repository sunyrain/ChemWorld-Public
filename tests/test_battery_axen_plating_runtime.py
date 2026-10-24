"""Real sessions preserve coupled metal and Axen memory in the original cell."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.battery_axen_plating import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import battery_control_current, battery_readout
from chemworld.physchem.battery_ocp import hysteresis_key
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


def test_idle_meter_advances_main_current_memory_without_external_charge():
    runtime = start()
    try:
        _, _, spec, initial, _ = unpack_battery(runtime._state)
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        before_meter = unpack_battery(runtime._state)[3]
        measured = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
        record, slot, _, final, _ = unpack_battery(runtime._state)
        hn, hp = hysteresis_key("negative"), hysteresis_key("positive")
        assert final.time_s == 3.
        assert final.coordinates["discharged_charge_C"] == pytest.approx(0., abs=1e-12)
        assert final.coordinates["delivered_work_J"] == pytest.approx(0., abs=1e-12)
        assert initial.coordinates[hn] > before_meter.coordinates[hn] > final.coordinates[hn]
        assert final.coordinates[hp] == pytest.approx(initial.coordinates[hp], abs=1e-12)
        assert initial.amounts_mol["plating:Li"] > final.amounts_mol["plating:Li"] > 0
        assert final.amounts_mol["dead:Li"] == initial.amounts_mol["dead:Li"]
        assert sum(final.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-11)
        fields = plating_fields(spec, final, current_A=0.)
        assert fields["stripping_current_A"] > 0
        assert fields["negative_intercalation_current_A"] == pytest.approx(
            -fields["stripping_current_A"], abs=1e-10)
        assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(0., abs=1e-10)
        assert {c.channel_id for c in measured.observation.channels} == {
            "terminal_potential", "electrical_current"}
        assert record.equipment_type == "single_particle_battery"
        assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
        assert set(spec.hysteresis_keys) <= set(slot.values)
        assert runtime._state.phases.phases["housing"].species_amounts_mol == {}
    finally:
        runtime.close()


def test_finite_voltage_control_couples_memory_with_one_terminal_work_receipt():
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
        assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
            -final.coordinates["delivered_work_J"], abs=1e-8)
        assert final.coordinates[hysteresis_key("negative")] < initial.coordinates[
            hysteresis_key("negative")]
        assert final.amounts_mol["plating:Li"] != initial.amounts_mol["plating:Li"]
    finally:
        runtime.close()


def test_pending_stages_and_both_memories_resume_in_a_fresh_process(tmp_path):
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
    Path(sys.argv[2]).write_text(json.dumps(runtime.snapshot().to_dict()))
finally:
    runtime.close()
"""
        subprocess.run([sys.executable, "-c", program, str(saved), str(output)],
                       check=True, capture_output=True, text=True)
        execute(runtime, {"operation": "wait", "duration_s": 9.})
        assert json.loads(output.read_text()) == json.loads(
            json.dumps(runtime.snapshot().to_dict()))
        _, slot, _, state, _ = unpack_battery(runtime._state)
        assert state.time_s == 9.
        assert state.coordinates["discharged_charge_C"] == pytest.approx(-.3, abs=1e-8)
        before = dict(slot.values)
        runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert dict(unpack_battery(runtime._state)[1].values) == before
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        after_idle = unpack_battery(runtime._state)[3]
        assert after_idle.time_s == 10.
        assert after_idle.coordinates[hysteresis_key("negative")] != state.coordinates[
            hysteresis_key("negative")]
    finally:
        runtime.close()


def test_replacement_installs_new_metal_and_new_memory_exactly_once():
    runtime = start()
    try:
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        record, _, spec, _, _ = unpack_battery(runtime._state)
        table = spec.negative_ocp.parameters
        row = table.rows[0]
        values = {**row.values, "initial_h": replace(row.values["initial_h"], value=-.4)}
        memory = replace(spec.negative_ocp,
                         parameters=replace(table, rows=(replace(row, values=values),)))
        new = replace(spec, negative_ocp=memory, plating=replace(
            spec.plating, initial_plated_mol=1e-5, initial_dead_mol=2e-5))
        replacement = replace(record, equipment_id="fresh-Axen-plating-cell", settings={
            **record.settings, "physical_model": new.to_dict()})
        slot = initial_battery_slot(replacement, new)
        runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                               "replacement": replacement, "slots": StateSlots((slot,))})
        _, actual, decoded, state, _ = unpack_battery(runtime._state)
        assert actual == slot and decoded == new
        assert state.amounts_mol == new.initial_state().amounts_mol
        assert state.coordinates == new.initial_state().coordinates
        assert state.time_s == 2.
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        final = unpack_battery(runtime._state)[3]
        assert final.time_s == 3.
        assert final.coordinates[hysteresis_key("negative")] != -.4
    finally:
        runtime.close()


def test_failed_advance_keeps_memory_metal_queue_and_clock_atomic(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = start()
    try:
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": -.3,
                          "stage_duration_s": 3.})
        before = deepcopy(runtime._state)

        def fail(*args, **kwargs):
            raise RuntimeError("injected coupled memory/plating failure")

        monkeypatch.setattr(adapter, "advance_battery", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 2.})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.equipment == before.equipment
        assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_coupled_workflow_replays_every_actual_action(tmp_path, noise):
    report = run(tmp_path/f"axen-plating-{noise}.jsonl", noise=noise)
    assert report["steps"] == report["replay_steps"] == 11
    assert report["mode"] == "reversible" and len(report["hysteresis_coordinates"]) == 2


def test_shared_clock_moves_main_current_memory_once_and_restores_exactly():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=default_spec()))
    runtime.reset(seed=4)
    restored = None
    try:
        before = local_states(runtime._state)
        _, _, spec, initial, control = unpack_battery(before["cell"])
        current = control.value_at(initial.time_s)
        assert current == .02  # the shared-lab fixture declares this finite source
        assert plating_fields(spec, initial, current_A=current)[
            "negative_intercalation_current_A"] > 0
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, 2.)
        final = unpack_battery(local_states(runtime._state)["cell"])[3]
        assert sum(final.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-11)
        assert final.coordinates[hysteresis_key("negative")] > initial.coordinates[
            hysteresis_key("negative")]
        assert final.coordinates["discharged_charge_C"] == pytest.approx(2*current, abs=1e-10)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict())))
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        if restored is not None:
            restored.close()
        runtime.close()
