"""Real metal Li persists through original electrical and lifecycle actions."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.battery_plating import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import F, battery_control_current, battery_readout
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def start(spec=None):
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec or default_spec()))
    runtime.reset(seed=4)
    return runtime


@pytest.mark.parametrize("mode", ("reversible", "irreversible"))
def test_meter_advances_real_metal_li_and_terminal_work_once(mode):
    runtime = start(default_spec(mode=mode))
    try:
        record, slot, spec, initial, _ = unpack_battery(runtime._state)
        execute(runtime, {"operation": "configure_electrical_control", "current_A": -.5})
        execute(runtime, {"operation": "wait", "duration_s": 3.})
        measured = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
        final = unpack_battery(runtime._state)[3]
        assert final.time_s == 4.
        assert final.coordinates["discharged_charge_C"] == pytest.approx(-2., abs=1e-9)
        assert sum(final.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-11)
        positive = spec.positive.keys("positive")
        assert sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in positive) == (
            pytest.approx(-2./F, abs=1e-11))
        if mode == "reversible":
            assert final.amounts_mol["plating:Li"] > initial.amounts_mol["plating:Li"]
            assert final.amounts_mol["dead:Li"] == initial.amounts_mol["dead:Li"]
        else:
            assert final.amounts_mol["dead:Li"] > initial.amounts_mol["dead:Li"]
            assert final.amounts_mol["plating:Li"] == initial.amounts_mol["plating:Li"]
        assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
            -final.coordinates["delivered_work_J"], abs=1e-8)
        assert runtime._state.ledger.heat_reaction_J == 0.
        assert runtime._state.ledger.energy_jacket_J == 0.
        assert {c.channel_id for c in measured.observation.channels} == {
            "terminal_potential", "electrical_current"}
        assert record.equipment_type == "single_particle_battery"
        assert slot.spec.state_type == "battery-lithium-and-charge-v1"
        assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
        assert runtime._state.phases.phases["housing"].species_amounts_mol == {}
    finally:
        runtime.close()


def test_rest_reintercalates_metal_without_external_charge_or_work():
    runtime = start()
    try:
        _, _, spec, initial, _ = unpack_battery(runtime._state)
        execute(runtime, {"operation": "wait", "duration_s": 3.})
        final = unpack_battery(runtime._state)[3]
        recovered = initial.amounts_mol["plating:Li"] - final.amounts_mol["plating:Li"]
        assert recovered > 0.
        assert sum(final.amounts_mol[k] - initial.amounts_mol[k]
                   for k in spec.negative.keys("negative")) == pytest.approx(recovered, abs=1e-11)
        assert final.coordinates["discharged_charge_C"] == 0.
        assert final.coordinates["delivered_work_J"] == 0.
        assert final.amounts_mol["dead:Li"] == 0.
    finally:
        runtime.close()


def test_cv_keeps_original_finite_source_and_work_receipt():
    runtime = start()
    try:
        _, _, spec, initial, _ = unpack_battery(runtime._state)
        target = float(battery_readout(spec, initial, -.2).voltage_V)
        execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
        execute(runtime, {"operation": "wait", "duration_s": 3.})
        _, _, _, final, program = unpack_battery(runtime._state)
        current = battery_control_current(spec, final, program)
        assert -spec.maximum_current_A < current < 0.
        assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=2e-6)
        assert final.coordinates["delivered_work_J"] == pytest.approx(
            target * final.coordinates["discharged_charge_C"], abs=1e-7)
    finally:
        runtime.close()


def test_midqueue_fresh_process_and_material_reset_preserve_metal(tmp_path):
    runtime = start()
    restored = None
    try:
        for current, duration in ((-.5, 3.), (0., 2.), (.3, 4.)):
            execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                              "stage_duration_s": duration})
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
        saved, output = tmp_path / "midqueue.json", tmp_path / "continued.json"
        saved.write_text(json.dumps(payload))
        program = """
import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()))
try:
    result=r.execute({'operation':'wait','duration_s':9.})
    assert result.info['transaction_status']=='committed',result.info
    Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict(),allow_nan=False))
finally:
    r.close()
"""
        subprocess.run([sys.executable, "-c", program, str(saved), str(output)],
                       check=True, capture_output=True, text=True)
        restored = ChemWorldRuntime.from_snapshot(payload)
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 9.})
        assert runtime._state.to_dict() == restored._state.to_dict()
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
        record, slot, _, state, _ = unpack_battery(runtime._state)
        assert state.time_s == 9.
        assert state.coordinates["discharged_charge_C"] == pytest.approx(-.3, abs=1e-8)
        assert record.settings["electrical_control"]["stage_index"] == 3
        before = dict(slot.values)
        runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert dict(unpack_battery(runtime._state)[1].values) == before
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        assert unpack_battery(runtime._state)[3].time_s == 10.
    finally:
        if restored is not None:
            restored.close()
        runtime.close()


def test_equipment_replacement_installs_new_metal_once():
    runtime = start()
    try:
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        record, _, spec, previous, _ = unpack_battery(runtime._state)
        new = replace(spec, plating=replace(spec.plating, initial_plated_mol=1e-5,
                                           initial_dead_mol=2e-5))
        replacement = replace(record, equipment_id="fresh-cell", settings={
            **record.settings, "physical_model": new.to_dict()})
        slot = initial_battery_slot(replacement, new)
        runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                               "replacement": replacement, "slots": StateSlots((slot,))})
        _, actual_slot, actual_spec, state, _ = unpack_battery(runtime._state)
        assert actual_slot == slot and actual_spec == new
        assert state.amounts_mol == new.initial_state().amounts_mol
        assert state.amounts_mol != previous.amounts_mol and state.time_s == 2.
    finally:
        runtime.close()


def test_failed_solver_preserves_metal_queue_and_clock(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = start()
    try:
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": -.3,
                          "stage_duration_s": 3.})
        before = deepcopy(runtime._state)

        def fail(*args, **kwargs):
            raise RuntimeError("injected plating solver failure")

        monkeypatch.setattr(adapter, "advance_battery", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 2.})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.equipment == before.equipment
        assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_actual_plating_workflow_replays_both_noise_modes(tmp_path, noise):
    report = run(tmp_path / f"plating-{noise}.jsonl", noise=noise)
    assert report["steps"] == report["replay_steps"] == 11


def test_shared_clock_preserves_cell_owner_and_refuses_undefined_heating():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=default_spec()))
    runtime.reset(seed=4)
    restored = None
    try:
        before = local_states(runtime._state)
        initial_li = sum(unpack_battery(before["cell"])[3].amounts_mol.values())
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, 2.)
        cell = local_states(runtime._state)["cell"]
        assert sum(unpack_battery(cell)[3].amounts_mol.values()) == (
            pytest.approx(initial_li, abs=1e-11))
        execute(runtime, {"operation": "select_vessel", "vessel": "cell"})
        saved = runtime._state
        rejected = runtime.execute({"operation": "queue_lab_heat", "target_temperature_K": 310.,
                                    "duration_s": 2.})
        assert rejected.info["transaction_status"] != "committed"
        assert runtime._state.lab_clock == saved.lab_clock
        assert runtime._state.state_slots == saved.state_slots
        assert runtime._state.ledger.time_s == saved.ledger.time_s
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict())))
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        if restored is not None:
            restored.close()
        runtime.close()
