"""Independent hydraulic limits, n/H balances and actual finite operation paths."""

from dataclasses import replace
from math import pi

import numpy as np
import pytest

from chemworld.physchem.transport import (
    FluidState,
    PipeSpec,
    PumpSpec,
    pipe_pressure_drop,
    pump_operating_point,
)
from chemworld.physchem.transport_devices import (
    FiniteTransportSpec,
    advance_transport,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


def session(spec=None, **kwargs):
    runtime = ChemWorldRuntime(composition=transport_world_spec(spec, **kwargs))
    runtime.reset(seed=24)
    return runtime


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", {
        "error": result.info.get("error_message"),
        "outcome": result.info.get("outcome_diagnostics"),
        "checks": [c for c in result.info.get("constitution_checks", []) if not c["passed"]],
    }
    return result


def test_pump_laminar_head_and_power_have_independent_roots():
    fluid, pipe = FluidState(1000, 0.001), PipeSpec(2, 0.001)
    coefficient = 128 * fluid.viscosity_Pa_s * pipe.length_m / (pi * pipe.diameter_m**4)
    for pump in (PumpSpec(1000, 10, 0.7), PumpSpec(1e6, 1e-4, 0.7)):
        point = pump_operating_point(
            pump,
            1e-7,
            lambda q: (
                pipe_pressure_drop(pipe, fluid, volumetric_flow_m3_s=q).pressure_drop_total_Pa
            ),
        )
        expected = min(
            1e-7,
            pump.maximum_pressure_rise_Pa / coefficient,
            np.sqrt(pump.maximum_shaft_power_W * pump.efficiency / coefficient),
        )
        assert point["actual_flow_m3_s"] == pytest.approx(expected, rel=1e-10)
        assert point["pressure_rise_Pa"] <= pump.maximum_pressure_rise_Pa * (1 + 1e-8)
        assert point["shaft_power_W"] <= pump.maximum_shaft_power_W * (1 + 1e-8)
    assert (
        pump_operating_point(PumpSpec(), 1e-7, lambda q: 30000 + coefficient * q)[
            "actual_flow_m3_s"
        ]
        == 0
    )


def test_native_flow_debits_source_and_commits_actual_energy_once():
    runtime = session()
    initial = runtime._state
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    result = execute(runtime, "run_flow", duration_s=60)
    final = runtime._state
    diagnostic = result.info["outcome_diagnostics"]["transport"]
    assert diagnostic["actual_flow_m3_s"] < 20 / 6e7
    assert final.inactive_vessels["feed"].volume_L < initial.inactive_vessels["feed"].volume_L
    _, spec, start = unpack_transport(initial)
    _, _, end = unpack_transport(final)
    for key in spec.properties.component_ids:
        assert sum(n[key] for n in end.amounts_mol) == pytest.approx(
            sum(n[key] for n in start.amounts_mol), abs=1e-10
        )
    assert sum(end.enthalpies_J) - sum(start.enthalpies_J) == pytest.approx(
        diagnostic["hydraulic_work_J"] + diagnostic["utility_heat_J"], abs=1e-6
    )
    assert final.ledger.time_s - initial.ledger.time_s == pytest.approx(60)
    assert final.ledger.flow_pump_work_J == pytest.approx(diagnostic["shaft_work_J"])
    observed = execute(runtime, "measure", instrument="flow_meter")
    assert observed.observation.channel("flow_rate").unit == "mL/min"


@pytest.mark.parametrize("temperature", [288.15, 298.15, 318.15])
def test_properties_follow_feed_temperature_and_composition(temperature):
    properties = synthetic_transport_properties()
    for amounts in ({"water": 1.0, "ethanol": 0.0}, {"water": 0.8, "ethanol": 0.2}):
        fluid, h, volume = properties.evaluate(amounts, temperature)
        assert fluid.viscosity_Pa_s > 0 and fluid.thermal_conductivity_W_m_K > 0
        assert volume > 0
        expected_cp = amounts["water"] * 75 + amounts["ethanol"] * 110
        assert h == pytest.approx(expected_cp * (temperature - 298.15), abs=1e-9)


def test_split_transport_preserves_n_h_and_finite_source_stop():
    runtime = session(feed_volume_L=0.0002)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    result = execute(runtime, "wait", duration_s=120)
    assert result.info["operation_outcome"] == "stopped"
    assert result.info["actual_duration_s"] < 120
    assert runtime._state.inactive_vessels["feed"].volume_L == pytest.approx(1e-6, abs=1e-10)


def test_numerical_split_and_step_refinement():
    runtime = session()
    _, spec, state = unpack_transport(runtime._state)
    results = [
        advance_transport(spec, state, 20 / 6e7, 120, max_step_s=h)[0].final_state
        for h in (20, 10, 5)
    ]
    half = advance_transport(spec, state, 20 / 6e7, 60)[0].final_state
    split = advance_transport(spec, half, 20 / 6e7, 60)[0].final_state
    for final in (*results, split):
        assert np.allclose(final.enthalpies_J, results[-1].enthalpies_J, atol=1e-6)


def test_zero_request_and_insulated_path_keep_true_balances():
    spec = FiniteTransportSpec(synthetic_transport_properties())
    spec = replace(spec, path=replace(spec.path, exchanger_area_m2=0))
    runtime = session(spec)
    _, _, state = unpack_transport(runtime._state)
    unchanged, point = advance_transport(spec, state, 0, 20)
    assert unchanged.final_state.amounts_mol == state.amounts_mol
    assert point["shaft_work_J"] == point["utility_heat_J"] == 0
    result, point = advance_transport(spec, state, 5 / 6e7, 120)
    assert sum(result.final_state.enthalpies_J) - sum(state.enthalpies_J) == pytest.approx(
        point["hydraulic_work_J"], abs=1e-7
    )


def test_serialized_spec_is_the_same_physics():
    spec = FiniteTransportSpec(synthetic_transport_properties())
    assert FiniteTransportSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict()


def test_pump_outlet_property_limit_stops_without_spending_or_discarding_energy():
    runtime = session(feed_temperature_K=339.99999)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    before = runtime._state
    result = execute(runtime, "run_flow", duration_s=30)
    assert result.info["operation_outcome"] == "stopped"
    assert result.info["outcome_diagnostics"]["stopping_event"] == "pump_outlet_property_limit"
    assert result.info["actual_duration_s"] == 0
    assert runtime._state.inactive_vessels["feed"].species_amounts == (
        before.inactive_vessels["feed"].species_amounts
    )
    assert runtime._state.ledger.flow_pump_work_J == before.ledger.flow_pump_work_J
    assert runtime._state.ledger.time_s == before.ledger.time_s
    meter = execute(runtime, "measure", instrument="flow_meter")
    assert meter.observation.channel("flow_rate").value == 0


def test_receiver_capacity_stops_and_idle_wait_does_not_pump():
    spec = FiniteTransportSpec(synthetic_transport_properties(), receiver_capacity_L=0.006)
    runtime = session(spec)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=5)
    result = execute(runtime, "wait", duration_s=120)
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.volume_L == pytest.approx(0.006, abs=1e-10)
    meter = execute(runtime, "measure", instrument="flow_meter")
    assert meter.observation.channel("flow_rate").value == 0
    execute(runtime, "set_flow_rate", flow_rate_mL_min=0)
    before = runtime._state
    execute(runtime, "wait", duration_s=10)
    assert runtime._state.ledger.time_s == before.ledger.time_s + 10
    assert runtime._state.species_amounts == before.species_amounts


def test_packed_bed_load_reuses_same_pump_and_composition_changes_flow():
    runtime = session()
    _, spec, initial = unpack_transport(runtime._state)
    bed = replace(spec, path=replace(spec.path, bed_length_m=0.2, particle_diameter_m=0.0002))
    plain, a = advance_transport(spec, initial, 20 / 6e7, 30)
    loaded, b = advance_transport(bed, initial, 20 / 6e7, 30)
    assert b["actual_flow_m3_s"] < a["actual_flow_m3_s"]
    assert loaded.enthalpy_residual_J == pytest.approx(0, abs=1e-7)
    assert plain.actual_duration_s == loaded.actual_duration_s == 30


def test_reset_retains_finite_feed_and_recloses_material_with_warm_container():
    runtime = session()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=5)
    execute(runtime, "wait", duration_s=60)
    before = runtime._state
    body = next(s for s in before.state_slots.slots if s.spec.owner_kind == "vessel")
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert runtime._state.inactive_vessels["feed"] == before.inactive_vessels["feed"]
    assert runtime._state.temperature_K > 298.15
    _, _, after = unpack_transport(runtime._state)
    expected = body.values["enthalpy_J"]  # Replacement liquid has H=0 at the authored datum.
    assert after.enthalpies_J[1] == pytest.approx(expected, abs=1e-9)
    execute(runtime, "wait", duration_s=30)
    remaining = runtime._state.inactive_vessels["feed"].species_amounts
    feed_age = runtime._state.vessel_elapsed_s["feed"]
    runtime.reset(options={"scope": "batch"})
    assert runtime._state.inactive_vessels["feed"].species_amounts == remaining
    assert runtime._state.vessel_elapsed_s["feed"] == feed_age
    assert runtime._state.vessel_elapsed_s["reactor"] == 0
    execute(runtime, "wait", duration_s=10)
    assert runtime._state.vessel_elapsed_s["feed"] == feed_age + 10
    assert runtime._state.vessel_elapsed_s["reactor"] == 10


def test_invalid_utility_and_numerical_failure_keep_real_inventory(monkeypatch):
    import chemworld.runtime.transport_devices as module

    runtime = session()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=5)
    before = runtime._state
    result = runtime.execute(
        {"operation": "run_flow", "duration_s": 30, "target_temperature_K": 400}
    )
    assert result.info["operation_outcome"] == "rejected"
    assert runtime._state.species_amounts == before.species_amounts

    def failure(*args, **kwargs):
        raise RuntimeError("injected numerical failure")

    monkeypatch.setattr(module, "advance_transport", failure)
    result = runtime.execute({"operation": "wait", "duration_s": 30})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.inactive_vessels == before.inactive_vessels
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_agent_transport_history_replays_actual_sources(tmp_path, noise):
    from chemworld.agents.base import BaseAgent
    from chemworld.data.logging import load_jsonl
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records

    class Sequence(BaseAgent):
        def act(self, history):
            return (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 5},
                {"operation": "run_flow", "duration_s": 60},
                {"operation": "measure", "instrument": "flow_meter"},
                {"operation": "terminate"},
            )[len(history)]

    path = tmp_path / "transport.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=Sequence(),
        composition=transport_world_spec().to_dict(),
        world_split="public-dev",
        budget=60,
        objective="balanced",
        seed=24,
        observation_noise_mode=noise,
        output_path=path,
    )
    records = load_jsonl(path)
    assert len(records) == 4
    report = verify_records(records, tolerance=1e-8)
    assert report.verified, report.mismatches


def test_native_gym_cli_and_new_process_restore_agree(tmp_path, capsys):
    import json
    import subprocess
    import sys

    from chemworld.cli import main
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.runtime.session_snapshot import SessionSnapshot

    composition = transport_world_spec()
    native, gym = session(), ChemWorldEnv(composition=composition)
    gym.reset(seed=24)
    actions = (
        {"operation": "set_flow_rate", "flow_rate_mL_min": 5},
        {"operation": "run_flow", "duration_s": 60},
        {"operation": "measure", "instrument": "flow_meter"},
    )
    paths = []
    for i, action in enumerate(actions):
        execute(native, **action)
        assert gym.step(action)[4]["transaction_status"] == "committed"
        path = tmp_path / f"action-{i}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    assert gym._state.state_slots == native._state.state_slots
    comp, snap = tmp_path / "world.json", tmp_path / "snapshot.json"
    comp.write_text(json.dumps(composition.to_dict()), encoding="utf-8")
    main(
        [
            "render",
            "--composition",
            str(comp),
            "--seed",
            "24",
            "--actions",
            *paths,
            "--format",
            "json",
            "--snapshot",
            str(snap),
        ]
    )
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 3
    resumed = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snap))
    assert resumed._state.state_slots == native._state.state_slots
    output = tmp_path / "continued.json"
    code = """import json,sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
r=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1]))
result=r.execute({'operation':'wait','duration_s':60})
assert result.info['transaction_status']=='committed',result.info
payload={'slots':r._state.state_slots.to_dict(),'source':r._state.inactive_vessels['feed'].to_dict()}
with open(sys.argv[2],'w') as f:json.dump(payload,f)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(snap), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    execute(resumed, "wait", duration_s=60)
    actual = json.loads(output.read_text())
    assert actual == json.loads(
        json.dumps(
            {
                "slots": resumed._state.state_slots.to_dict(),
                "source": resumed._state.inactive_vessels["feed"].to_dict(),
            }
        )
    )
