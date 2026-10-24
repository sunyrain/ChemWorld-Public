"""The original cell session advances all size bins and one metal-Li pair."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.examples.battery_mpm_plating import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import F, battery_control_current, battery_readout
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.battery_plating import plating_fields
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
def test_original_meter_advances_all_size_bins_and_one_metal_pair(mode):
    runtime = start(default_spec(mode=mode))
    try:
        _, _, spec, initial, _ = unpack_battery(runtime._state)
        execute(runtime, {"operation": "configure_electrical_control", "current_A": -.5})
        execute(runtime, {"operation": "wait", "duration_s": 3.})
        measured = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
        record, slot, _, final, _ = unpack_battery(runtime._state)
        assert final.time_s == 4.
        assert final.coordinates["discharged_charge_C"] == pytest.approx(-2., abs=1e-9)
        assert sum(final.amounts_mol.values()) == pytest.approx(
            sum(initial.amounts_mol.values()), abs=1e-11)
        negative = [k for n, _, _, keys, _ in spec.particle_bins if n == "negative" for k in keys]
        metal = ("plating:Li", "dead:Li")
        assert sum(final.amounts_mol[k]-initial.amounts_mol[k] for k in (*negative, *metal)) == (
            pytest.approx(2./F, abs=1e-11))
        assert len(final.amounts_mol) == 5*spec.negative.cells+2
        assert all(any(final.amounts_mol[k] != initial.amounts_mol[k] for k in keys)
                   for _, _, _, keys, _ in spec.particle_bins)
        changed = "plating:Li" if mode == "reversible" else "dead:Li"
        fixed = "dead:Li" if mode == "reversible" else "plating:Li"
        assert final.amounts_mol[changed] > initial.amounts_mol[changed]
        assert final.amounts_mol[fixed] == initial.amounts_mol[fixed]
        fields = plating_fields(spec, final, current_A=-.5)
        assert abs(sum(fields["reaction_current_A"][:3]) + fields["stripping_current_A"]+.5) < 1e-8
        assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
            -final.coordinates["delivered_work_J"], abs=1e-8)
        assert {c.channel_id for c in measured.observation.channels} == {
            "terminal_potential", "electrical_current"}
        assert record.equipment_type == "single_particle_battery"
        assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"
        assert runtime._state.phases.phases["housing"].species_amounts_mol == {}
    finally:
        runtime.close()


def test_original_cv_tracks_shared_voltage_with_real_bin_currents():
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
    finally:
        runtime.close()


def test_staged_bin_and_metal_state_survives_real_fresh_process(tmp_path):
    runtime = start()
    try:
        for current, duration in ((-.5, 3.), (0., 2.), (.3, 4.)):
            execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                              "stage_duration_s": duration})
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
        saved, output = tmp_path/"prefix.json", tmp_path/"continued.json"
        saved.write_text(json.dumps(payload))
        program = """
import json,sys
from pathlib import Path
from chemworld.runtime import ChemWorldRuntime
r=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()))
try:
    x=r.execute({'operation':'wait','duration_s':9.})
    assert x.info['transaction_status']=='committed',x.info
    Path(sys.argv[2]).write_text(json.dumps(r._state.to_dict(),allow_nan=False))
finally:
    r.close()
"""
        subprocess.run([sys.executable, "-c", program, str(saved), str(output)],
                       check=True, capture_output=True, text=True)
        execute(runtime, {"operation": "wait", "duration_s": 9.})
        assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))
        _, slot, _, state, _ = unpack_battery(runtime._state)
        assert state.time_s == 9.
        assert state.coordinates["discharged_charge_C"] == pytest.approx(-.3, abs=1e-8)
        before = dict(slot.values)
        runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
        assert dict(unpack_battery(runtime._state)[1].values) == before
    finally:
        runtime.close()


def test_replacing_cell_installs_new_distribution_and_metal_exactly_once():
    runtime = start()
    try:
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        record, _, spec, previous, _ = unpack_battery(runtime._state)
        distribution = ParticleSizeDistribution((4e-6, 8e-6), (.7, .3),
                                               spec.negative_distribution.source)
        new = replace(spec, negative_distribution=distribution,
                      plating=replace(spec.plating, initial_plated_mol=1e-5, initial_dead_mol=2e-5))
        replacement = replace(record, equipment_id="fresh-distribution", settings={
            **record.settings, "physical_model": new.to_dict()})
        slot = initial_battery_slot(replacement, new)
        runtime.reset(options={"scope": "equipment", "equipment_id": record.equipment_id,
                               "replacement": replacement, "slots": StateSlots((slot,))})
        _, actual, decoded, state, _ = unpack_battery(runtime._state)
        assert actual == slot and decoded == new and decoded.counts == (2, 2)
        assert state.amounts_mol == new.initial_state().amounts_mol
        assert set(state.amounts_mol) != set(previous.amounts_mol) and state.time_s == 2.
        execute(runtime, {"operation": "wait", "duration_s": 1.})
        assert unpack_battery(runtime._state)[3].time_s == 3.
    finally:
        runtime.close()


def test_failed_advance_preserves_every_size_and_metal_stock(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = start()
    try:
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": -.3,
                          "stage_duration_s": 3.})
        before = deepcopy(runtime._state)

        def fail(*args, **kwargs):
            raise RuntimeError("injected MPM plating failure")

        monkeypatch.setattr(adapter, "advance_battery", fail)
        result = runtime.execute({"operation": "wait", "duration_s": 2.})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert runtime._state.state_slots == before.state_slots
        assert runtime._state.equipment == before.equipment
        assert runtime._state.ledger.time_s == before.ledger.time_s
    finally:
        runtime.close()


@pytest.mark.parametrize("noise", ("keyed", "sequential"))
def test_actual_distribution_workflow_and_exact_replay(tmp_path, noise):
    report = run(tmp_path/f"mpm-plating-{noise}.jsonl", noise=noise)
    assert report["steps"] == report["replay_steps"] == 11
    assert report["particle_counts"] == [3, 2]
    assert report["model"] == "mpm-plating-isothermal-v1"


def test_shared_clock_keeps_all_bins_and_one_cell_owner():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(composition=reactive_closed_lab(battery_spec=default_spec()))
    runtime.reset(seed=4)
    restored = None
    try:
        before = local_states(runtime._state)
        li = sum(unpack_battery(before["cell"])[3].amounts_mol.values())
        execute(runtime, {"operation": "wait", "duration_s": 2.})
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, 2.)
        cell = local_states(runtime._state)["cell"]
        assert sum(unpack_battery(cell)[3].amounts_mol.values()) == pytest.approx(li, abs=1e-11)
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict())))
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 1.})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        if restored is not None:
            restored.close()
        runtime.close()
