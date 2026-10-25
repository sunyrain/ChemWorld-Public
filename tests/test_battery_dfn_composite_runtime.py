"""Independent materials retain one cell lifetime and one shared clock."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.battery_dfn_composite import default_spec, run
from chemworld.foundation.state_ownership import StateSlots
from chemworld.physchem.battery_models import (
    F,
    battery_control_current,
    battery_readout,
    battery_spec_from_dict,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import initial_battery_slot, unpack_battery
from chemworld.world.battery_world import battery_world_spec


def session(spec=None):
    runtime = ChemWorldRuntime(
        composition=battery_world_spec(
            default_spec(radial_cells=4, cells=(3, 3, 3)) if spec is None else spec
        )
    )
    runtime.reset(seed=4)
    return runtime


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def phase_amounts(spec, state):
    return {
        (p.electrode, p.phase_id): p.inventory_mol
        for p in battery_readout(spec, state, 0.0).phase_readouts
    }


def test_zero_terminal_current_really_redistributes_between_materials():
    runtime = session()
    _, slot, spec, initial, _ = unpack_battery(runtime._state)
    assert not hasattr(spec, "negative") and not hasattr(spec, "positive")
    assert len(slot.values) == len(spec.inventory_keys) + 2
    assert len(spec.inventory_keys) == len(set(spec.inventory_keys))
    assert all(":phase:" in key for key in spec.inventory_keys if key not in spec.electrolyte.keys)
    assert battery_spec_from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    before = phase_amounts(spec, initial)
    currents = {
        (p.electrode, p.phase_id): p.reaction_current_A
        for p in battery_readout(spec, initial, 0.0).phase_readouts
    }
    assert abs(currents["negative", "alpha"]) > 1e-5
    assert currents["negative", "alpha"] == pytest.approx(-currents["negative", "beta"])
    execute(runtime, {"operation": "wait", "duration_s": 3.0})
    _, _, _, final, _ = unpack_battery(runtime._state)
    after = phase_amounts(spec, final)
    moved = after["negative", "alpha"] - before["negative", "alpha"]
    assert abs(moved) > 1e-8
    assert after["negative", "beta"] - before["negative", "beta"] == pytest.approx(
        -moved, abs=1e-11
    )
    assert after["positive", "positive"] == pytest.approx(before["positive", "positive"], abs=1e-11)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-11
    )
    assert final.coordinates["discharged_charge_C"] == pytest.approx(0.0, abs=1e-9)
    assert final.time_s == 3.0
    assert slot.spec.owner_kind == "equipment" and slot.spec.batch_reset == "preserve"


def test_meter_uses_summed_terminal_current_and_actual_shared_time():
    runtime = session()
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    before = phase_amounts(spec, initial)
    execute(runtime, {"operation": "configure_electrical_control", "current_A": 0.12})
    execute(runtime, {"operation": "wait", "duration_s": 3.0})
    result = execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, _ = unpack_battery(runtime._state)
    after = phase_amounts(spec, final)
    assert final.time_s == 4.0
    assert final.coordinates["discharged_charge_C"] == pytest.approx(0.48, abs=1e-8)
    for electrode, sign in (("negative", -1), ("positive", 1)):
        change = sum(after[k] - before[k] for k in before if k[0] == electrode)
        assert change == pytest.approx(sign * 0.48 / F, abs=1e-11)
    salt_before = [initial.amounts_mol[k] for k in spec.electrolyte.keys]
    salt_after = [final.amounts_mol[k] for k in spec.electrolyte.keys]
    assert sum(salt_after) == pytest.approx(sum(salt_before), abs=1e-11)
    assert salt_after != salt_before
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-8
    )
    reading = battery_readout(spec, final, 0.12)
    assert sum(
        p.reaction_current_A for p in reading.phase_readouts if p.electrode == "negative"
    ) == pytest.approx(0.12, abs=1e-9)
    assert {c.channel_id for c in result.observation.channels} == {
        "terminal_potential",
        "electrical_current",
    }


def test_composite_voltage_control_uses_actual_phase_closure():
    runtime = session()
    _, _, spec, initial, _ = unpack_battery(runtime._state)
    target = battery_readout(spec, initial, 0.12).voltage_V
    execute(runtime, {"operation": "configure_electrical_control", "potential_V": target})
    execute(runtime, {"operation": "wait", "duration_s": 2.0})
    execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    _, _, _, final, program = unpack_battery(runtime._state)
    current = battery_control_current(spec, final, program)
    assert 0 < current < spec.maximum_current_A
    assert battery_readout(spec, final, current).voltage_V == pytest.approx(target, abs=2e-6)
    assert final.time_s == 3.0


def test_midqueue_snapshot_and_material_reset_preserve_each_phase():
    runtime = session()
    for current, duration in ((0.12, 3.0), (0.0, 2.0), (-0.1, 4.0)):
        execute(
            runtime,
            {
                "operation": "queue_electrical_stage",
                "current_A": current,
                "stage_duration_s": duration,
            },
        )
    execute(runtime, {"operation": "wait", "duration_s": 1.0})
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(runtime.snapshot().to_dict())))
    for active in (runtime, restored):
        execute(active, {"operation": "wait", "duration_s": 9.0})
    assert runtime._state.to_dict() == restored._state.to_dict()
    record, slot, spec, physical, programme = unpack_battery(runtime._state)
    assert physical.time_s == pytest.approx(9.0)
    assert physical.coordinates["discharged_charge_C"] == pytest.approx(-0.04, abs=1e-8)
    assert programme.value_at(physical.time_s) == 0.0
    assert record.settings["electrical_control"]["stage_index"] == 3
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before
    execute(runtime, {"operation": "wait", "duration_s": 1.0})
    after = unpack_battery(runtime._state)[3]
    assert after.time_s == 10.0
    assert after.coordinates["discharged_charge_C"] == pytest.approx(-0.04, abs=1e-8)
    # Rest evolves the material inventories even though the external charge stops.
    assert (
        abs(
            phase_amounts(spec, after)["negative", "alpha"]
            - phase_amounts(spec, physical)["negative", "alpha"]
        )
        > 1e-9
    )


def test_replacement_installs_fresh_independent_inventories_once():
    runtime = session()
    execute(runtime, {"operation": "wait", "duration_s": 3.0})
    record, _, old, previous, _ = unpack_battery(runtime._state)
    new = replace(
        old,
        negative_phases=(
            replace(old.negative_phases[0], initial_fraction=0.5),
            old.negative_phases[1],
        ),
    )
    replacement = replace(
        record,
        equipment_id="new-composite",
        settings={**record.settings, "physical_model": new.to_dict()},
    )
    fresh = initial_battery_slot(replacement, new)
    runtime.reset(
        options={
            "scope": "equipment",
            "equipment_id": record.equipment_id,
            "replacement": replacement,
            "slots": StateSlots((fresh,)),
        }
    )
    _, slot, spec, physical, _ = unpack_battery(runtime._state)
    assert slot == fresh and spec == new
    assert physical.amounts_mol == new.initial_state().amounts_mol
    assert physical.amounts_mol != previous.amounts_mol


def test_composite_solver_failure_rolls_back_every_phase_and_clock(monkeypatch):
    import chemworld.runtime.battery as adapter

    runtime = session()
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected composite DFN solver failure")

    monkeypatch.setattr(adapter, "advance_battery", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 3.0})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_composite_world_rejects_undefined_thermal_coupling():
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    with pytest.raises(ValueError, match="composite"):
        battery_world_spec(
            default_spec(radial_cells=4, cells=(3, 3, 3)), thermal_device=ThermalDeviceSpec()
        )


@pytest.mark.parametrize(
    "operation",
    [
        "impedance",
        "soh",
        "energy",
        "dissipation",
        "chemical_potentials",
        "sei_affinity",
        "soc_configuration",
    ],
)
def test_undefined_analysis_refuses_without_mutating_material_state(operation):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import (
        battery_chemical_energy_j,
        battery_chemical_potentials,
        battery_dissipation_w,
        battery_sei_affinity_j_mol,
    )
    from chemworld.physchem.battery_soh import battery_spec_at_soc, soh_from_battery

    spec = default_spec(radial_cells=4, cells=(3, 3, 3))
    state = spec.initial_state()
    original = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    calls = {
        "impedance": lambda: battery_impedance(spec, [1.0], state=state),
        "soh": lambda: soh_from_battery(spec, state),
        "energy": lambda: battery_chemical_energy_j(spec, state),
        "dissipation": lambda: battery_dissipation_w(spec, state, 0.1),
        "chemical_potentials": lambda: battery_chemical_potentials(spec, state),
        "sei_affinity": lambda: battery_sei_affinity_j_mol(spec, state, 0.1),
        "soc_configuration": lambda: battery_spec_at_soc(spec, None, 0.5),
    }
    # No single-phase analysis fallback may reinterpret spatial phase inventories.
    error, message = ValueError, "composite DFN"
    with pytest.raises(error, match=message):
        calls[operation]()
    assert (dict(state.amounts_mol), dict(state.coordinates), state.time_s) == original


@pytest.mark.parametrize("noise", ["keyed", "sequential"])
def test_real_composite_example_replays_both_noise_modes(tmp_path, noise):
    result = run(
        tmp_path / f"composite-{noise}.jsonl",
        spec=default_spec(radial_cells=4, cells=(3, 3, 3)),
        noise=noise,
    )
    assert result["steps"] == result["replay_steps"] == 11


def test_full_json_continuation_in_fresh_process_preserves_phase_salt_queue_and_noise(tmp_path):
    import subprocess
    import sys

    runtime = session()
    for current, duration in ((0.08, 3.0), (-0.06, 4.0), (0.0, 2.0)):
        execute(
            runtime,
            {
                "operation": "queue_electrical_stage",
                "current_A": current,
                "stage_duration_s": duration,
            },
        )
    execute(runtime, {"operation": "wait", "duration_s": 2.0})
    saved = runtime.snapshot().to_dict()
    source, destination = tmp_path / "before.json", tmp_path / "after.json"
    source.write_text(json.dumps(saved))
    child = r"""
import json,sys
from chemworld.runtime import ChemWorldRuntime
runtime=ChemWorldRuntime.from_snapshot(json.load(open(sys.argv[1])))
for action in ({"operation":"wait","duration_s":7.},
               {"operation":"measure","instrument":"battery_meter"}):
    result=runtime.execute(action)
    assert result.info['transaction_status']=='committed',result.info
assert 'pybamm' not in sys.modules
json.dump(runtime.snapshot().to_dict(),open(sys.argv[2],'w'),allow_nan=False)
"""
    subprocess.run([sys.executable, "-c", child, str(source), str(destination)], check=True)
    execute(runtime, {"operation": "wait", "duration_s": 7.0})
    execute(runtime, {"operation": "measure", "instrument": "battery_meter"})
    # Snapshot dictionaries contain original immutable tuple declarations;
    # JSON consumers represent the same arrays as lists. Canonicalize both sides
    # through JSON, preserving every number, key, diagnostic and receipt.
    assert json.loads(destination.read_text()) == json.loads(
        json.dumps(runtime.snapshot().to_dict(), allow_nan=False)
    )
    with pytest.raises(FileExistsError):
        run(source, spec=default_spec(radial_cells=4, cells=(3, 3, 3)))


def test_shared_clock_advances_all_phase_and_salt_stocks_and_preserves_owners():
    from chemworld.examples.closed_shared_labs import reactive_closed_lab
    from chemworld.runtime.scheduling import local_states

    runtime = ChemWorldRuntime(
        composition=reactive_closed_lab(battery_spec=default_spec(radial_cells=4, cells=(3, 3, 3)))
    )
    runtime.reset(seed=4)
    restored = None
    try:
        before = local_states(runtime._state)
        advancer = runtime.runtime.domain_services.reaction_thermal.advancer
        expected = {
            key: advancer.batch(local, {"operation": "wait", "duration_s": 2.0}, heat=False)
            for key, local in before.items()
        }
        execute(runtime, {"operation": "wait", "duration_s": 2.0})
        assert runtime._state.vessel_elapsed_s == dict.fromkeys(before, 2.0)
        for key, local in local_states(runtime._state).items():
            assert expected[key].success
            for field in (
                "species_amounts",
                "temperature_K",
                "pressure_Pa",
                "volume_L",
                "phases",
                "state_slots",
                "equipment",
            ):
                assert getattr(local, field) == getattr(expected[key].state, field)
        execute(runtime, {"operation": "select_vessel", "vessel": "cell"})
        saved = runtime._state
        rejected = runtime.execute(
            {"operation": "queue_lab_heat", "target_temperature_K": 310.0, "duration_s": 2.0}
        )
        assert rejected.info["transaction_status"] != "committed"
        assert runtime._state.lab_clock == saved.lab_clock
        assert runtime._state.state_slots == saved.state_slots
        assert runtime._state.temperature_K == saved.temperature_K
        assert runtime._state.ledger.time_s == saved.ledger.time_s
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for active in (runtime, restored):
            execute(active, {"operation": "wait", "duration_s": 1.0})
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        if restored is not None:
            restored.close()
        runtime.close()
