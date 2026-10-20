"""Independent RO limits, finite balances, native lifecycle and replay."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.membranes import (
    MembraneSpec,
    R,
    advance_membrane,
    membrane_at_pressure,
    membrane_inventory,
    membrane_operating_point,
    salt_concentration,
)
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.membrane_world import membrane_world_spec


def session(spec=None, **kwargs):
    runtime = ChemWorldRuntime(composition=membrane_world_spec(spec, **kwargs))
    runtime.reset(seed=34)
    return runtime


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_water_and_imperfect_membrane_zero_flux_have_independent_limits():
    spec = MembraneSpec()
    pure = membrane_at_pressure(spec, 0, 200000)
    assert pure["water_flux_m_s"] == pytest.approx(spec.water_permeability_m_Pa_s * 200000)
    zero = membrane_at_pressure(spec, 10, 0)
    assert zero["water_flux_m_s"] == zero["salt_flux_mol_m2_s"] == 0
    assert zero["permeate_concentration_mol_m3"] == 10
    assert zero["stopping_event"] == "zero_net_drive"
    negative = membrane_at_pressure(spec, 10, -1)
    assert negative["stopping_event"] == "unsupported_reverse_drive"
    perfect = replace(spec, salt_permeability_m_s=0, mass_transfer_m_s=1e6)
    p = 2 * R * spec.temperature_K * 10
    assert membrane_at_pressure(perfect, 10, p)["stopping_event"] == "zero_net_drive"
    actual = membrane_at_pressure(perfect, 10, 200000)
    assert actual["salt_flux_mol_m2_s"] == 0
    assert actual["water_flux_m_s"] == pytest.approx(spec.water_permeability_m_Pa_s * (200000 - p))


@pytest.mark.parametrize("pressure", [0, 200000, 800000])
def test_actual_drive_limits_and_isothermal_balances(pressure):
    spec = MembraneSpec(pump=PumpSpec(pressure, 0.2, 0.7))
    start = membrane_inventory()
    result, diagnostic = advance_membrane(spec, start, 20 / 6e7, 3600)
    final = result.final_state
    for key in ("water", "nacl"):
        assert sum(n[key] for n in final.amounts_mol) == pytest.approx(
            sum(n[key] for n in start.amounts_mol), abs=1e-11
        )
    assert diagnostic["shaft_work_J"] == diagnostic["isothermal_heat_out_J"]
    assert diagnostic["final"]["pressure_rise_Pa"] <= pressure * (1 + 1e-8)
    assert diagnostic["final"]["shaft_power_W"] <= 0.2 * (1 + 1e-8)
    assert result.enthalpy_residual_J == 0
    if pressure:
        assert salt_concentration(final.amounts_mol[0]) > 10
        assert final.amounts_mol[1]["water"] > 0
    else:
        assert result.stopping_event == "no_forward_pump_drive"
        assert result.actual_duration_s == 0


def test_blocked_perfect_membrane_does_not_fabricate_available_pump_head():
    spec = MembraneSpec(salt_permeability_m_s=0, pump=PumpSpec(1000, 0.2, 0.7))
    initial = membrane_inventory()
    point = membrane_operating_point(spec, initial.amounts_mol[0], 20 / 6e7)
    assert point["actual_flow_m3_s"] == 0
    assert point["pressure_rise_Pa"] == 1000
    assert point["net_driving_pressure_Pa"] < 0
    result, _ = advance_membrane(spec, initial, 20 / 6e7, 60)
    assert result.actual_duration_s == 0


def test_interface_domain_and_physical_inventory_stops_keep_actual_time():
    spec = MembraneSpec()
    for model, initial, event in (
        (spec, membrane_inventory(0.001, 0), "feed_pickup_exhausted"),
        (replace(spec, collector_capacity_L=0.001), membrane_inventory(), "collector_full"),
        (spec, membrane_inventory(0.02, 60), "concentration_domain"),
    ):
        result, _ = advance_membrane(model, initial, 20 / 6e7, 3600)
        assert result.stopping_event == event
        assert result.actual_duration_s < 3600
        assert min(v for n in result.final_state.amounts_mol for v in n.values()) >= 0


def test_step_refinement_split_and_serialization_preserve_variable_work():
    spec = MembraneSpec()
    initial = membrane_inventory()
    outputs = [
        advance_membrane(spec, initial, 20 / 6e7, 3600, max_step_s=s)[0].final_state
        for s in (60, 30, 15)
    ]
    half, _ = advance_membrane(spec, initial, 20 / 6e7, 1800)
    split, _ = advance_membrane(
        MembraneSpec.from_dict(spec.to_dict()), half.final_state, 20 / 6e7, 1800
    )
    for value in (*outputs, split.final_state):
        assert value.heater_energy_J == pytest.approx(outputs[-1].heater_energy_J, rel=1e-6)
        assert np.allclose(
            list(value.amounts_mol[0].values()),
            list(outputs[-1].amounts_mol[0].values()),
            rtol=1e-6,
        )
    initial_power = membrane_operating_point(spec, initial.amounts_mol[0], 20 / 6e7)[
        "shaft_power_W"
    ]
    assert outputs[-1].heater_energy_J < initial_power * 3600


def test_native_actual_stock_meter_and_lifecycle_are_shared():
    runtime = session()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    execute(runtime, "run_flow", duration_s=600)
    before = runtime._state
    reading = execute(runtime, "measure", instrument="membrane_meter").observation
    assert reading.channel("feed_salinity").value > 10
    assert (
        reading.channel("collected_salinity").value
        != reading.channel("permeate_outlet_salinity").value
    )
    assert reading.channel("water_flux").unit == "m/s"
    assert before.ledger.flow_pump_work_J == before.ledger.heat_loss_J
    runtime.reset(options={"scope": "batch"})
    assert runtime._state.species_amounts == before.species_amounts
    assert (
        runtime._state.inactive_vessels["permeate"].species_amounts
        == before.inactive_vessels["permeate"].species_amounts
    )
    assert runtime._state.vessel_elapsed_s == before.vessel_elapsed_s
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert salt_concentration(runtime._state.species_amounts) == pytest.approx(10)
    assert runtime._state.equipment.equipment["membrane"] == before.equipment.equipment["membrane"]
    assert (
        runtime._state.inactive_vessels["permeate"].species_amounts
        == before.inactive_vessels["permeate"].species_amounts
    )
    execute(runtime, "wait", duration_s=60)


def test_numerical_failure_rolls_back_and_temperature_control_is_rejected(monkeypatch):
    import chemworld.runtime.membranes as module

    runtime = session()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    before = runtime._state
    rejected = runtime.execute(
        {"operation": "run_flow", "duration_s": 60, "target_temperature_K": 310}
    )
    assert rejected.info["operation_outcome"] == "rejected"

    def fail(*args, **kwargs):
        raise RuntimeError("injected solve failure")

    monkeypatch.setattr(module, "advance_membrane", fail)
    failed = runtime.execute({"operation": "wait", "duration_s": 60})
    assert failed.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.inactive_vessels == before.inactive_vessels
    assert runtime._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_agent_membrane_history_replays_actual_stock_and_work(tmp_path, noise):
    from chemworld.agents.base import BaseAgent
    from chemworld.data.logging import load_jsonl
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records

    class Sequence(BaseAgent):
        def act(self, history):
            return (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
                {"operation": "wait", "duration_s": 600},
                {"operation": "measure", "instrument": "membrane_meter"},
                {"operation": "terminate"},
            )[len(history)]

    path = tmp_path / "membrane.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=Sequence(),
        composition=membrane_world_spec().to_dict(),
        world_split="public-dev",
        budget=100,
        objective="balanced",
        seed=34,
        observation_noise_mode=noise,
        output_path=path,
    )
    rows = load_jsonl(path)
    assert len(rows) == 4
    report = verify_records(rows, tolerance=1e-8)
    assert report.verified, report.mismatches


def test_native_gym_cli_and_new_process_membrane_continuation(tmp_path, capsys):
    import json
    import subprocess
    import sys

    from chemworld.cli import main
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.runtime.session_snapshot import SessionSnapshot

    spec = membrane_world_spec()
    native = session()
    gym = ChemWorldEnv(composition=spec)
    gym.reset(seed=34)
    actions = (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
        {"operation": "run_flow", "duration_s": 600},
        {"operation": "measure", "instrument": "membrane_meter"},
    )
    paths = []
    for i, action in enumerate(actions):
        execute(native, **action)
        assert gym.step(action)[4]["transaction_status"] == "committed"
        path = tmp_path / f"action-{i}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    assert gym._state.to_dict() == native._state.to_dict()
    world = tmp_path / "world.json"
    snapshot = tmp_path / "snapshot.json"
    world.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    main(
        [
            "render",
            "--composition",
            str(world),
            "--seed",
            "34",
            "--actions",
            *paths,
            "--format",
            "json",
            "--snapshot",
            str(snapshot),
        ]
    )
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 3
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snapshot))
    assert restored._state.to_dict() == native._state.to_dict()
    target = tmp_path / "continued.json"
    code = """import json,sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
r=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1]))
v=r.execute({'operation':'wait','duration_s':600})
assert v.info['transaction_status']=='committed',v.info
with open(sys.argv[2],'w') as f:json.dump(r._state.to_dict(),f)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(snapshot), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    execute(restored, "wait", duration_s=600)
    assert json.loads(target.read_text()) == json.loads(json.dumps(restored._state.to_dict()))
