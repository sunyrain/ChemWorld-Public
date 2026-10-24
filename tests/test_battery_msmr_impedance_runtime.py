"""A physical frequency analysis cannot consume an experiment or alter its future."""

import json
import subprocess
import sys

import numpy as np
import pytest

from chemworld.examples.battery_msmr_impedance import default_spec, run
from chemworld.physchem.battery_impedance import (
    BatteryImpedance,
    battery_impedance,
    battery_impedance_system,
)
from chemworld.physchem.battery_models import F, battery_readout, battery_spec_from_dict
from chemworld.physchem.residual_system import ResidualDomainError
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
from chemworld.world.battery_world import battery_world_spec


def session(spec):
    runtime = ChemWorldRuntime(composition=battery_world_spec(spec))
    runtime.reset(seed=13)
    return runtime


def execute(runtime, action):
    result = runtime.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_real_equipment_analysis_keeps_queue_clock_noise_and_future_exact():
    runtime = session(default_spec(cells=6))
    for current, duration in ((.1, 2.), (-.08, 3.)):
        execute(runtime, {"operation": "queue_electrical_stage", "current_A": current,
                          "stage_duration_s": duration})
    before = runtime.snapshot().to_dict()
    twin = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(before)))
    _, _, spec, state, _ = unpack_battery(runtime._state)
    spectrum = battery_impedance(spec, [1e-4, 1e-3, 1e-2], state=state)
    assert isinstance(spectrum, BatteryImpedance) and spectrum.success
    assert runtime.snapshot().to_dict() == before
    for active in (runtime, twin):
        execute(active, {"operation": "wait", "duration_s": 5.})
        execute(active, {"operation": "measure", "instrument": "battery_meter"})
    assert runtime.snapshot().to_dict() == twin.snapshot().to_dict()
    assert unpack_battery(runtime._state)[3].time_s == 6.


def test_worked_particle_refuses_without_relaxing_or_modifying_real_session():
    runtime = session(default_spec(cells=6))
    execute(runtime, {"operation": "configure_electrical_control", "current_A": .2})
    execute(runtime, {"operation": "wait", "duration_s": 2.})
    execute(runtime, {"operation": "configure_electrical_control", "current_A": 0.})
    before = runtime.snapshot().to_dict()
    _, _, spec, state, _ = unpack_battery(runtime._state)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        battery_impedance(spec, [1e-3], state=state)
    assert runtime.snapshot().to_dict() == before


def test_public_analysis_from_json_equipment_matches_in_fresh_process(tmp_path):
    runtime = session(default_spec(cells=6))
    before = runtime.snapshot().to_dict()
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(before))
    result_path = tmp_path / "response.json"
    child = r'''
import json,sys
from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.battery import unpack_battery
runtime=ChemWorldRuntime.from_snapshot(json.load(open(sys.argv[1])))
before=runtime.snapshot().to_dict()
_,_,spec,state,_=unpack_battery(runtime._state)
result=battery_impedance(spec,[1e-4,1e-3,1e-2],state=state)
assert result.success
assert runtime.snapshot().to_dict()==before
assert 'pybamm' not in sys.modules
json.dump(result.response.to_dict(),open(sys.argv[2],'w'),allow_nan=False)
'''
    subprocess.run([sys.executable, "-c", child, str(snapshot), str(result_path)], check=True)
    _, _, spec, state, _ = unpack_battery(runtime._state)
    expected = battery_impedance(spec, [1e-4, 1e-3, 1e-2], state=state).response.to_dict()
    assert json.loads(result_path.read_text()) == expected
    assert runtime.snapshot().to_dict() == before


def test_example_serializes_complete_response_and_sourced_reaction_identities(tmp_path):
    spec = default_spec(cells=6)
    spec = battery_spec_from_dict(json.loads(json.dumps(spec.to_dict())))
    output = tmp_path / "msmr-eis.json"
    report = run(output, spec=spec)
    assert report["spectrum"]["success"] and report["state_unchanged"]
    assert json.loads(output.read_text()) == report
    system, _ = battery_impedance_system(spec, spec.initial_state())
    assert [v["name"] for v in report["full_response"]["variables"]] == [
        v.name for v in system.variables
    ]
    for electrode in ("negative", "positive"):
        material = getattr(spec, electrode+"_material")
        assert report["reaction_rows"][electrode] == list(material.reaction_row_ids)
        assert report["model"][electrode+"_material"] == material.to_dict()
    voltage = battery_readout(spec, spec.initial_state(), 0.).voltage_V
    for point in report["inventory_responses"]:
        values = {key: complex(value["real"], value["imag"])
                  for key, value in point["responses"].items()}
        iw = 2j * np.pi * point["frequency_Hz"]
        np.testing.assert_allclose(iw * F * values["negative_Li_mol_per_A"], -1., atol=1e-10)
        np.testing.assert_allclose(iw * F * values["positive_Li_mol_per_A"], 1., atol=1e-10)
        np.testing.assert_allclose(iw * values["signed_charge_C_per_A"], 1., atol=1e-10)
        np.testing.assert_allclose(iw * values["signed_work_J_per_A"], voltage, atol=1e-10)
    with pytest.raises(FileExistsError):
        run(output, spec=spec)


def test_standalone_example_cli_outputs_all_declared_frequencies(tmp_path):
    output = tmp_path / "cli.json"
    process = subprocess.run([
        sys.executable, "-m", "chemworld.examples.battery_msmr_impedance",
        "--cells", "4", "--output", str(output),
    ], check=True, text=True, capture_output=True)
    assert json.loads(process.stdout)["frequencies"] == 9
    data = json.loads(output.read_text())
    assert data["spectrum"]["success"]
    assert len(data["spectrum"]["points"]) == len(data["full_response"]["points"]) == 9
