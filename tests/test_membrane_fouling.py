"""Finite film transfer, real cleaning inventory and shared-session behavior."""

from dataclasses import replace
from math import exp

import numpy as np
import pytest

from chemworld.physchem.membranes import (
    FOULANT_MOLAR_MASS_KG,
    MembraneFoulingSpec,
    MembraneSpec,
    advance_membrane,
    effective_water_permeability,
    film_mass,
    membrane_at_pressure,
    membrane_inventory,
    membrane_operating_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.membranes import fouling_slot, unpack_membrane
from chemworld.world.membrane_world import membrane_world_spec


def session(spec=None, **kwargs):
    runtime = ChemWorldRuntime(
        composition=membrane_world_spec(
            spec or MembraneSpec(fouling=MembraneFoulingSpec()), **kwargs
        )
    )
    runtime.reset(seed=35)
    return runtime


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def inventory(runtime):
    return unpack_membrane(runtime._state)[2]


def with_film(state, mass=2e-5):
    n = [dict(row) for row in state.amounts_mol]
    n[0]["foulant"] -= mass / FOULANT_MOLAR_MASS_KG
    n[4]["foulant"] += mass / FOULANT_MOLAR_MASS_KG
    return replace(state, amounts_mol=tuple(n))


def test_clean_limit_reuses_ro_and_resistance_has_independent_pressure_solution():
    clean = MembraneSpec()
    spec = replace(clean, fouling=MembraneFoulingSpec(deposition_fraction=0))
    start = membrane_inventory(fouling=True)
    plain, _ = advance_membrane(clean, membrane_inventory(), 20 / 6e7, 900)
    zero, _ = advance_membrane(spec, start, 20 / 6e7, 900)
    assert film_mass(spec, zero.final_state) == 0
    for i in (0, 1):
        for k in ("water", "nacl"):
            assert zero.final_state.amounts_mol[i][k] == pytest.approx(
                plain.final_state.amounts_mol[i][k], rel=1e-6, abs=1e-12
            )
    mass = 2e-5
    expected = 1 / (
        1 / clean.water_permeability_m_Pa_s
        + 0.001 * spec.fouling.specific_resistance_m_kg * mass / spec.area_m2
    )
    assert effective_water_permeability(spec, mass) == expected
    assert membrane_at_pressure(spec, 0, 200000, mass)["water_flux_m_s"] == pytest.approx(
        expected * 200000
    )


def test_zero_flow_desorption_is_analytic_and_returns_to_feed():
    spec = MembraneSpec(fouling=MembraneFoulingSpec())
    start = with_film(membrane_inventory(fouling=True))
    result, diagnostic = advance_membrane(spec, start, 0, 900)
    assert film_mass(spec, result.final_state) == pytest.approx(2e-5 * exp(-0.009), rel=1e-6)
    assert result.final_state.amounts_mol[0]["foulant"] > start.amounts_mol[0]["foulant"]
    assert result.final_state.amounts_mol[1:4] == start.amounts_mol[1:4]
    assert diagnostic["shaft_work_J"] == 0


def test_finite_rinse_has_exponential_release_and_does_not_touch_feed_or_permeate():
    spec = MembraneSpec(fouling=MembraneFoulingSpec())
    start = with_film(membrane_inventory(fouling=True))
    result, diag = advance_membrane(spec, start, 20 / 6e7, 60, flow_path="rinse")
    end = result.final_state
    assert film_mass(spec, end) == pytest.approx(2e-5 * exp(-0.6), rel=1e-6)
    assert end.amounts_mol[:2] == start.amounts_mol[:2]
    assert end.amounts_mol[3]["foulant"] * FOULANT_MOLAR_MASS_KG == pytest.approx(
        2e-5 * (1 - exp(-0.6)), rel=1e-6
    )
    assert max(map(abs, result.component_residuals_mol.values())) < 1e-12
    assert diag["shaft_work_J"] == diag["isothermal_heat_out_J"] > 0
    assert (
        membrane_operating_point(spec, end.amounts_mol[0], 20 / 6e7, film_mass(spec, end))[
            "actual_flow_m3_s"
        ]
        > membrane_operating_point(spec, start.amounts_mol[0], 20 / 6e7, 2e-5)["actual_flow_m3_s"]
    )
    idle, _ = advance_membrane(spec, start, 0, 60, flow_path="rinse")
    assert idle.final_state.amounts_mol == start.amounts_mol


@pytest.mark.parametrize("event", ["wash_pickup_exhausted", "waste_full"])
def test_rinse_stops_keep_real_water_film_and_elapsed_time(event):
    foul = MembraneFoulingSpec(waste_capacity_L=0.001 if event == "waste_full" else 0.2)
    spec = MembraneSpec(fouling=foul)
    start = with_film(
        membrane_inventory(
            fouling=True, wash_volume_L=0.001 if event == "wash_pickup_exhausted" else 0.1
        )
    )
    result, _ = advance_membrane(spec, start, 20 / 6e7, 60, flow_path="rinse")
    assert result.stopping_event == event
    assert 0 < result.actual_duration_s < 60
    assert film_mass(spec, result.final_state) == pytest.approx(
        2e-5 * exp(-0.01 * result.actual_duration_s), rel=1e-6
    )
    repeated, _ = advance_membrane(spec, result.final_state, 20 / 6e7, 60, flow_path="rinse")
    assert repeated.actual_duration_s == 0
    assert repeated.final_state.amounts_mol == result.final_state.amounts_mol


def test_coupled_fouling_refinement_and_split_advance():
    spec = MembraneSpec(fouling=MembraneFoulingSpec(deposition_fraction=0.5))
    start = membrane_inventory(fouling=True)
    finals = [
        advance_membrane(spec, start, 20 / 6e7, 900, max_step_s=s)[0].final_state
        for s in (60, 30, 15)
    ]
    half, _ = advance_membrane(spec, start, 20 / 6e7, 450)
    finals.append(advance_membrane(spec, half.final_state, 20 / 6e7, 450)[0].final_state)
    for end in finals:
        assert np.allclose(
            [list(n.values()) for n in end.amounts_mol],
            [list(n.values()) for n in finals[2].amounts_mol],
            rtol=1e-3,
            atol=1e-10,
        )
        assert film_mass(spec, end) > 0
        assert end.amounts_mol[0]["foulant"] < start.amounts_mol[0]["foulant"]
        assert end.amounts_mol[1]["foulant"] == 0


def test_native_flow_routing_lifecycle_and_equipment_replacement():
    from chemworld.foundation.state_ownership import StateSlots

    runtime = session()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    execute(runtime, "run_flow", duration_s=900)
    initial_mass = film_mass(unpack_membrane(runtime._state)[1], inventory(runtime))
    assert initial_mass > 0
    before = runtime._state
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert film_mass(unpack_membrane(runtime._state)[1], inventory(runtime)) == initial_mass
    assert runtime._state.inactive_vessels == before.inactive_vessels
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20, flow_path="rinse")
    execute(runtime, "wait", duration_s=60)
    read = execute(runtime, "measure", instrument="membrane_meter").observation
    assert read.channel("water_flux").value == 0
    assert read.channel("waste_foulant_mass").value > 0
    assert read.channel("wash_volume").value == pytest.approx(0.08)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=0)
    assert (
        runtime._state.equipment.equipment["membrane"].settings["flow_control"]["flow_path"]
        == "rinse"
    )
    after = runtime._state
    runtime.reset(options={"scope": "batch"})
    assert runtime._state.state_slots == after.state_slots
    assert runtime._state.species_amounts == after.species_amounts
    old = runtime._state.equipment.equipment["membrane"]
    replacement = replace(old, equipment_id="new_membrane")
    runtime.reset(
        options={
            "scope": "equipment",
            "equipment_id": "membrane",
            "replacement": replacement.to_dict(),
            "slots": StateSlots((fouling_slot("reactor", "new_membrane"),)).to_dict(),
        }
    )
    assert film_mass(unpack_membrane(runtime._state)[1], inventory(runtime)) == 0
    assert runtime._state.species_amounts == after.species_amounts


def test_solver_failure_preserves_all_fluids_and_film(monkeypatch):
    import chemworld.runtime.membranes as module

    runtime = session()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    execute(runtime, "wait", duration_s=900)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20, flow_path="rinse")
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected fouling solver failure")

    monkeypatch.setattr(module, "advance_membrane", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 60})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.state_slots == before.state_slots
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.inactive_vessels == before.inactive_vessels
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_full_permeate_collector_does_not_block_independent_finite_rinse():
    runtime = session(MembraneSpec(collector_capacity_L=0.001, fouling=MembraneFoulingSpec()),
                      wash_volume_L=0.001)
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20)
    process = execute(runtime, "wait", duration_s=600)
    assert process.info["outcome_diagnostics"]["stopping_event"] == "collector_full"
    execute(runtime, "set_flow_rate", flow_rate_mL_min=20, flow_path="rinse")
    reading = execute(runtime, "measure", instrument="membrane_meter").observation
    assert reading.channel("rinse_flow").value == 20
    before = runtime._state
    rinse = execute(runtime, "run_flow", duration_s=60)
    dt = rinse.info["actual_duration_s"]
    assert 0 < dt < 60
    assert rinse.info["outcome_diagnostics"]["stopping_event"] == "wash_pickup_exhausted"
    assert runtime._state.ledger.time_s - before.ledger.time_s == pytest.approx(dt)
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.inactive_vessels["permeate"].species_amounts == (
        before.inactive_vessels["permeate"].species_amounts
    )
    reading = execute(runtime, "measure", instrument="membrane_meter").observation
    assert reading.channel("rinse_flow").value == 0


def test_unsupported_rinse_route_is_rejected_not_silently_ignored():
    from chemworld.world.transport_world import transport_world_spec

    for world in (membrane_world_spec(), transport_world_spec()):
        runtime = ChemWorldRuntime(composition=world)
        runtime.reset(seed=35)
        result = runtime.execute(
            {"operation": "set_flow_rate", "flow_rate_mL_min": 20, "flow_path": "rinse"}
        )
        assert result.info["transaction_status"] != "committed", result.info


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_fouling_agent_actions_replay_through_same_public_entry(noise, tmp_path):
    from chemworld.agents.base import BaseAgent
    from chemworld.data.logging import load_jsonl
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records

    class Sequence(BaseAgent):
        def act(self, history):
            return (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
                {"operation": "wait", "duration_s": 900},
                {"operation": "set_flow_rate", "flow_rate_mL_min": 20, "flow_path": "rinse"},
                {"operation": "run_flow", "duration_s": 60},
                {"operation": "measure", "instrument": "membrane_meter"},
                {"operation": "terminate"},
            )[len(history)]

    path = tmp_path / "fouling.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=Sequence(),
        composition=membrane_world_spec(MembraneSpec(fouling=MembraneFoulingSpec())).to_dict(),
        world_split="public-dev",
        budget=100,
        objective="balanced",
        seed=35,
        observation_noise_mode=noise,
        output_path=path,
    )
    rows = load_jsonl(path)
    assert len(rows) == 6
    report = verify_records(rows, tolerance=1e-8)
    assert report.verified, report.mismatches


def test_native_gym_cli_snapshot_and_continued_cleaning(tmp_path, capsys):
    import json

    from chemworld.cli import main
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.runtime.session_snapshot import SessionSnapshot

    world = membrane_world_spec(MembraneSpec(fouling=MembraneFoulingSpec()))
    native, gym = session(), ChemWorldEnv(composition=world)
    gym.reset(seed=35)
    actions = [
        {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
        {"operation": "wait", "duration_s": 900},
        {"operation": "set_flow_rate", "flow_rate_mL_min": 20, "flow_path": "rinse"},
    ]
    paths = []
    for i, action in enumerate(actions):
        execute(native, **action)
        assert gym.step(action)[4]["transaction_status"] == "committed"
        path = tmp_path / f"action-{i}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    assert gym._state.to_dict() == native._state.to_dict()
    path = tmp_path / "world.json"
    path.write_text(json.dumps(world.to_dict()), encoding="utf-8")
    saved = tmp_path / "snapshot.json"
    main(
        [
            "render",
            "--composition",
            str(path),
            "--seed",
            "35",
            "--actions",
            *paths,
            "--format",
            "json",
            "--snapshot",
            str(saved),
        ]
    )
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 3
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(saved))
    execute(restored, "wait", duration_s=60)
    execute(native, "wait", duration_s=60)
    assert restored._state.to_dict() == native._state.to_dict()
