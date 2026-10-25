"""Real source projections use the existing equipment lifetime and shared clock."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.battery_models import F, battery_readout, battery_system
from chemworld.physchem.battery_parameter_sets import battery_parameter_set
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_source_cell_queue_meter_json_and_material_reset_keep_bound_inventory():
    parameters = battery_parameter_set()
    spec = parameters.build_spm(radial_cells=6, maximum_current_A=8.0)
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=5)
    assert runtime._state.volume_L == pytest.approx(.0242)
    _, slot, decoded, initial, _ = unpack_battery(runtime._state)
    assert decoded.parameter_binding == spec.parameter_binding
    assert slot.spec.mechanism_id.startswith("source-SPM:Chen2020@PyBaMM-25.4.2:")
    for electrode in ("negative", "positive"):
        particle = getattr(spec, electrode)
        expected = particle.capacity_mol * getattr(spec, "initial_" + electrode + "_fraction")
        total = sum(initial.amounts_mol[k] for k in particle.keys(electrode))
        assert total == pytest.approx(expected)
    for current, duration in ((1.0, 2.0), (0.0, 1.0), (-.1, 2.0)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.0})
    restored = ChemWorldRuntime.from_snapshot(
        json.loads(json.dumps(runtime.snapshot().to_dict(), sort_keys=True)))
    for active in (runtime, restored):
        completed = execute(active, {"operation": "wait", "duration_s": 5.0})
        # The queued five-second protocol stops at its end, before the requested
        # wait ends. The terminal meter then advances one second at zero current.
        assert completed.info["operation_outcome"] == "stopped"
        assert completed.info["actual_duration_s"] == pytest.approx(4.0)
        assert completed.info["process_control"]["status"] == "completed"
        assert unpack_battery(active._state)[3].time_s == pytest.approx(5.0)
        measured = execute(active, {"operation": "measure", "instrument": "battery_meter"})
        assert measured.info["actual_duration_s"] == pytest.approx(1.0)
    assert runtime._state.to_dict() == restored._state.to_dict()
    _, slot, decoded, final, _ = unpack_battery(runtime._state)
    assert decoded.parameter_binding == spec.parameter_binding
    assert final.time_s == pytest.approx(6.0)
    charge = final.coordinates["discharged_charge_C"]
    for electrode, sign in (("negative", -1), ("positive", 1)):
        change = sum(final.amounts_mol[k] - initial.amounts_mol[k]
                     for k in getattr(spec, electrode).keys(electrode))
        assert change == pytest.approx(sign * charge / F, abs=1e-11)
    assert runtime._state.ledger.flow_electrical_work_J == pytest.approx(
        -final.coordinates["delivered_work_J"], abs=1e-8)
    before = dict(slot.values)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert dict(unpack_battery(runtime._state)[1].values) == before


@pytest.mark.parametrize("change", [{"temperature_K": 310.0}, {"volume_L": .025}])
def test_source_readout_rejects_unbound_temperature_or_envelope(change):
    spec = battery_parameter_set().build_spm(radial_cells=4, maximum_current_A=8.0)
    with pytest.raises(ValueError, match="declared fixed temperature and volume"):
        battery_readout(spec, replace(spec.initial_state(), **change), 1.0)


def test_source_projection_rejects_external_thermal_reinterpretation():
    spec = battery_parameter_set().build_spm(radial_cells=4, maximum_current_A=8.0)
    for build in (lambda: battery_world_spec(spec, thermal_device=object()),
                  lambda: battery_system(spec, ScalarControlProgram(((0., 0.),), "A"),
                                         thermal_device=object())):
        with pytest.raises(ValueError, match="thermal-device"):
            build()


def test_source_near_full_cell_stops_when_queued_charge_reaches_voltage_limit():
    spec = battery_parameter_set().build_spm(radial_cells=6, maximum_current_A=8.0)
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=5)
    for current, duration in ((1.0, 2.0), (0.0, 1.0), (-.5, 2.0)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    execute(runtime, {"operation": "wait", "duration_s": 1.0})
    result = execute(runtime, {"operation": "wait", "duration_s": 5.0})
    assert result.info["operation_outcome"] == "stopped"
    assert result.info["actual_duration_s"] == pytest.approx(2.0)
    _, _, restored, final, _ = unpack_battery(runtime._state)
    assert final.time_s == pytest.approx(3.0)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(2.0, abs=1e-9)
    assert battery_readout(restored, final, -.5).voltage_V > spec.maximum_voltage_V
