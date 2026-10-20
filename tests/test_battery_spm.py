"""Finite SPM conservation, cutoff, continuation and native lifetime semantics."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    ParticleSpec,
    advance_battery,
    battery_readout,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram


def program(current):
    return ScalarControlProgram(((0, current),), "A")


def check_inventory(spec, initial, final):
    assert sum(initial.amounts_mol.values()) == pytest.approx(
        sum(final.amounts_mol.values()), abs=1e-12
    )
    q = final.coordinates["discharged_charge_C"] - initial.coordinates["discharged_charge_C"]
    for name, particle, sign in (("negative", spec.negative, -1), ("positive", spec.positive, 1)):
        delta = sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in particle.keys(name))
        assert delta == pytest.approx(sign * q / F, abs=1e-12)
        for key, volume in zip(particle.keys(name), particle.mesh.cell_volumes_m3, strict=True):
            assert 0 <= final.amounts_mol[key] <= volume * particle.maximum_concentration_mol_m3


def test_uniform_rest_retains_inventory_and_exact_ocv():
    spec = BatterySPMSpec()
    initial = spec.initial_state()
    result = advance_battery(spec, initial, program(0), 10000)
    assert result.outcome == "success"
    assert result.final_state.amounts_mol == pytest.approx(initial.amounts_mol, abs=1e-15)
    read = battery_readout(spec, result.final_state, 0)
    assert read.voltage_V == pytest.approx((4.2 - 0.8 * 0.3) - (0.2 - 0.1 * 0.7), abs=1e-12)
    assert result.final_state.coordinates == initial.coordinates


@pytest.mark.parametrize("rate", [0.1, 0.5, 1])
def test_discharge_has_real_profiles_faradaic_closure_and_surface_cutoff(rate):
    spec = BatterySPMSpec()
    current = spec.current_at_c_rate(rate)
    initial = spec.initial_state()
    result = advance_battery(spec, initial, program(current), 3960 / rate)
    assert result.outcome == "stopped"
    assert result.actual_duration_s < 3960 / rate
    final = result.final_state
    check_inventory(spec, initial, final)
    assert final.coordinates["discharged_charge_C"] == pytest.approx(
        current * result.actual_duration_s, rel=1e-9
    )
    read = battery_readout(spec, final, current)
    assert read.negative_surface_fraction == pytest.approx(0.01, abs=1e-8)
    assert read.positive_surface_fraction == pytest.approx(0.99, abs=1e-8)
    assert read.negative_average_fraction > read.negative_surface_fraction
    assert final.coordinates["delivered_work_J"] > 0


def test_rest_relaxes_profile_without_resetting_inventory_or_charge():
    spec = BatterySPMSpec()
    discharged = advance_battery(
        spec, spec.initial_state(), program(spec.current_at_c_rate(1)), 600
    ).final_state
    rested = advance_battery(spec, discharged, program(0), 30000).final_state
    check_inventory(spec, discharged, rested)
    assert rested.coordinates == discharged.coordinates
    for name, particle in (("negative", spec.negative), ("positive", spec.positive)):
        profile = (
            np.array([rested.amounts_mol[k] for k in particle.keys(name)])
            / particle.mesh.cell_volumes_m3
        )
        assert np.ptp(profile) < 1e-3


def test_charge_reverses_actual_lithium_exchange():
    spec = BatterySPMSpec()
    initial = spec.initial_state()
    result = advance_battery(spec, initial, program(-spec.current_at_c_rate(0.5)), 200)
    check_inventory(spec, initial, result.final_state)
    assert result.final_state.coordinates["discharged_charge_C"] < 0
    assert result.final_state.coordinates["delivered_work_J"] < 0


def test_segments_continue_same_current_and_inventory():
    spec = BatterySPMSpec()
    drive = program(spec.current_at_c_rate(0.5))
    whole = advance_battery(spec, spec.initial_state(), drive, 900).final_state
    first = advance_battery(spec, spec.initial_state(), drive, 400).final_state
    final = advance_battery(spec, first, drive, 500).final_state
    assert final.amounts_mol == pytest.approx(whole.amounts_mol, rel=1e-7, abs=1e-13)
    assert final.coordinates == pytest.approx(whole.coordinates, rel=1e-7, abs=1e-8)


def test_voltage_limit_stops_at_actual_trajectory_endpoint():
    spec = BatterySPMSpec(minimum_voltage_V=3.6)
    current = spec.current_at_c_rate(0.5)
    result = advance_battery(spec, spec.initial_state(), program(current), 5000)
    assert result.stopping_event == "battery:minimum-voltage"
    assert 0 < result.actual_duration_s < 5000
    assert battery_readout(spec, result.final_state, current).voltage_V == pytest.approx(
        3.6, abs=1e-8
    )


def test_invalid_or_nonisothermal_model_rejected():
    with pytest.raises(ValueError):
        ParticleSpec(cells=True)
    with pytest.raises(ValueError):
        BatterySPMSpec(initial_negative_fraction=1)
    spec = BatterySPMSpec()
    with pytest.raises(ValueError, match="isothermal"):
        advance_battery(spec, replace(spec.initial_state(), temperature_K=310), program(0), 1)
    with pytest.raises(ValueError, match="current programme"):
        advance_battery(spec, spec.initial_state(), ScalarControlProgram(((0, 4),), "K"), 1)


def degradation_spec(enabled):
    from chemworld.physchem.battery_degradation import SEISpec

    return BatterySPMSpec(thermal=BatteryThermalSpec(), sei=SEISpec()) if enabled else None


def make_session(spec=None):
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.battery_world import battery_world_spec

    session = ChemWorldRuntime(composition=battery_world_spec(spec))
    session.reset(seed=4)
    return session


def step(session, action):
    result = session.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def test_native_current_wait_and_measurement_share_inventory_and_one_clock():
    from chemworld.runtime.battery import unpack_battery

    session = make_session()
    initial = unpack_battery(session._state)[3]
    step(session, {"operation": "configure_electrical_control", "current_A": 0.3})
    assert session._state.ledger.time_s == 0
    step(session, {"operation": "wait", "duration_s": 500})
    result = step(session, {"operation": "measure", "instrument": "battery_meter"})
    assert session._state.ledger.time_s == 501
    channels = {c.channel_id: c for c in result.observation.channels}
    assert set(channels) == {"terminal_potential", "electrical_current"}
    assert channels["terminal_potential"].unit == "V"
    assert channels["electrical_current"].value == 0.3
    assert channels["electrical_current"].unit == "A"
    _, _, spec, current, _ = unpack_battery(session._state)
    check_inventory(spec, initial, current)
    assert current.coordinates["discharged_charge_C"] == pytest.approx(0.3 * 501)
    assert not session._state.state_slots.to_dict(include_hidden=False)


@pytest.mark.parametrize("with_sei", [False, True])
def test_material_reset_preserves_battery_soc_and_replacement_has_new_inventory(with_sei):
    from chemworld.foundation.state_ownership import StateSlots
    from chemworld.runtime.battery import initial_battery_slot, unpack_battery

    session = make_session(degradation_spec(with_sei))
    step(session, {"operation": "configure_electrical_control", "current_A": 0.5})
    step(session, {"operation": "wait", "duration_s": 200})
    record, slot, spec, _, _ = unpack_battery(session._state)
    session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert unpack_battery(session._state)[1] == slot
    replacement = replace(record, equipment_id="second-battery")
    new = initial_battery_slot(replacement, spec)
    session.reset(
        options={
            "scope": "equipment",
            "equipment_id": record.equipment_id,
            "replacement": replacement,
            "slots": StateSlots((new,)),
        }
    )
    assert unpack_battery(session._state)[1] == new
    assert unpack_battery(session._state)[0].equipment_id == "second-battery"
    step(session, {"operation": "wait", "duration_s": 2})


def test_native_cutoff_keeps_actual_consumption_and_can_rest_afterward():
    from chemworld.runtime.battery import unpack_battery

    session = make_session()
    _, _, spec, _, _ = unpack_battery(session._state)
    current = spec.current_at_c_rate(1)
    step(session, {"operation": "configure_electrical_control", "current_A": current})
    result = step(session, {"operation": "wait", "duration_s": 4000})
    assert result.info["operation_outcome"] == "stopped"
    elapsed = session._state.ledger.time_s
    assert 0 < elapsed < 4000
    q = unpack_battery(session._state)[3].coordinates["discharged_charge_C"]
    assert q == pytest.approx(current * elapsed, rel=1e-8)
    step(session, {"operation": "configure_electrical_control", "current_A": 0})
    step(session, {"operation": "wait", "duration_s": 100})
    assert session._state.ledger.time_s == pytest.approx(elapsed + 100)
    assert unpack_battery(session._state)[3].coordinates["discharged_charge_C"] == q


@pytest.mark.parametrize("with_sei", [False, True])
def test_battery_numerical_failure_rolls_back_actual_inventory(monkeypatch, with_sei):
    import chemworld.runtime.battery as battery

    session = make_session(degradation_spec(with_sei))
    before = session._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected SPM numerical failure")

    monkeypatch.setattr(battery, "advance_battery", fail)
    result = session.execute({"operation": "wait", "duration_s": 2})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert session._state.state_slots == before.state_slots
    assert session._state.ledger.time_s == before.ledger.time_s


@pytest.mark.parametrize(
    "action",
    [
        {"current_A": 0.1, "potential_V": 4},
        {"potential_V": 4.6},
        {},
        {"current_A": 3},
        {"current_A": float("nan")},
        {"current_A": 0.1, "ramp_duration_s": -1},
    ],
)
def test_electrical_mode_and_device_limits_reject_without_physical_changes(action):
    session = make_session()
    before = session._state
    result = session.execute({"operation": "configure_electrical_control", **action})
    assert result.info["operation_outcome"] == "rejected"
    assert session._state.state_slots == before.state_slots
    assert session._state.equipment == before.equipment
    assert session._state.ledger.time_s == 0


def test_current_ramp_and_saved_session_continue_actual_battery(tmp_path):
    from chemworld.runtime.session_snapshot import SessionSnapshot

    session = make_session()
    step(
        session,
        {"operation": "configure_electrical_control", "current_A": 0.4, "ramp_duration_s": 200},
    )
    step(session, {"operation": "wait", "duration_s": 100})
    path = tmp_path / "battery.json"
    session.snapshot().save(path)
    restored = make_session()
    restored.restore(SessionSnapshot.load(path))
    action = {"operation": "wait", "duration_s": 200}
    step(session, action)
    step(restored, action)
    assert restored._state.state_slots == session._state.state_slots
    assert restored._state.ledger == session._state.ledger


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
@pytest.mark.parametrize("with_sei", [False, True])
def test_battery_agent_history_replays_native_entries(tmp_path, noise, with_sei):
    from chemworld.agents.base import BaseAgent
    from chemworld.data.logging import load_jsonl
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records
    from chemworld.world.battery_world import battery_world_spec

    class Sequence(BaseAgent):
        def act(self, history):
            return (
                {
                    "operation": "queue_electrical_stage",
                    "potential_V": 3.7,
                    "stage_duration_s": 200,
                },
                {"operation": "wait", "duration_s": 100},
                {"operation": "measure", "instrument": "battery_meter"},
                {"operation": "terminate"},
            )[len(history)]

    path = tmp_path / "history.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=Sequence(),
        composition=battery_world_spec(degradation_spec(with_sei)).to_dict(),
        world_split="public-dev",
        budget=80,
        objective="balanced",
        seed=4,
        observation_noise_mode=noise,
        output_path=path,
    )
    records = load_jsonl(path)
    assert len(records) == 4
    report = verify_records(records, tolerance=1e-8)
    assert report.verified, report.mismatches


@pytest.mark.parametrize("with_sei", [False, True])
def test_battery_gym_cli_and_cross_process_snapshot_share_physics(tmp_path, capsys, with_sei):
    import json
    import subprocess
    import sys

    from chemworld.cli import main
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.runtime.session_snapshot import SessionSnapshot
    from chemworld.world.battery_world import battery_world_spec

    spec = battery_world_spec(degradation_spec(with_sei))
    native = make_session(degradation_spec(with_sei))
    gym = ChemWorldEnv(composition=spec)
    gym.reset(seed=4)
    actions = (
        {"operation": "queue_electrical_stage", "potential_V": 3.7, "stage_duration_s": 200},
        {"operation": "wait", "duration_s": 100},
        {"operation": "measure", "instrument": "battery_meter"},
    )
    paths = []
    for i, action in enumerate(actions):
        step(native, action)
        assert gym.step(action)[4]["transaction_status"] == "committed"
        path = tmp_path / f"action-{i}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    assert gym._state.state_slots == native._state.state_slots
    assert gym._state.ledger.time_s == native._state.ledger.time_s
    comp, snap = tmp_path / "composition.json", tmp_path / "snapshot.json"
    comp.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    main(
        [
            "render",
            "--composition",
            str(comp),
            "--seed",
            "4",
            "--actions",
            *paths,
            "--format",
            "json",
            "--snapshot",
            str(snap),
        ]
    )
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 3
    from chemworld.runtime import ChemWorldRuntime

    resumed = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snap))
    assert resumed._state.state_slots == native._state.state_slots
    output = tmp_path / "continued.json"
    code = """import json,sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
snapshot=SessionSnapshot.load(sys.argv[1])
session=ChemWorldRuntime.from_snapshot(snapshot)
session.execute({'operation':'wait','duration_s':100})
with open(sys.argv[2],'w') as stream:json.dump(session._state.state_slots.to_dict(),stream)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(snap), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    step(resumed, {"operation": "wait", "duration_s": 100})
    assert json.loads(output.read_text()) == json.loads(
        json.dumps(resumed._state.state_slots.to_dict())
    )
    gym.close()
