"""Stable finite solids, shared heat transfer, ownership and real entry points."""

from dataclasses import replace
from math import exp, log

import numpy as np
import pytest
from scipy.optimize import brentq

from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.solid_solutions import (
    SolidPhaseSpec,
    SolidSolutionSpec,
    ideal_two_phase_spec,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.solid_solutions import SolidSolutionRuntime, solid_enthalpy, validate_solid
from chemworld.world.solid_world import solid_world_spec


def session(spec=None, **kwargs):
    runtime = ChemWorldRuntime(composition=solid_world_spec(spec, **kwargs))
    runtime.reset(seed=40)
    return runtime


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", result.info
    return result


@pytest.mark.parametrize("temperature", [300, 500, 800])
def test_regular_binodal_and_independent_lever_rule(temperature):
    model = SolidSolutionSpec()
    result = model.at_tp({"A": 0.3, "B": 0.7}, temperature)
    if temperature < 10000 / (2 * R):
        lo = brentq(
            lambda x: R * temperature * log(x / (1 - x)) + 10000 * (1 - 2 * x), 1e-10, 0.499999
        )
        assert [p.x_A for p in result.phases] == pytest.approx([lo, 1 - lo], abs=1e-9)
        assert result.phases[0].fraction == pytest.approx((1 - lo - 0.3) / (1 - 2 * lo))
    else:
        assert len(result.phases) == 1
        assert result.phases[0].x_A == pytest.approx(0.3)
    assert sum(p.fraction * p.x_A for p in result.phases) == pytest.approx(0.3, abs=1e-12)


def test_ideal_two_phase_has_analytic_common_tangent():
    model = ideal_two_phase_spec()
    temperature = 500
    lo = 1 / (1 + exp(5000 / (R * temperature)))
    result = model.at_tp({"A": 0.5, "B": 0.5}, temperature)
    assert [p.x_A for p in result.phases] == pytest.approx([lo, 1 - lo], abs=1e-10)
    assert [p.phase_name for p in result.phases] == ["beta", "alpha"]


@pytest.mark.parametrize("L1", [-3000, 3000])
def test_asymmetric_rk_global_certificate_and_caloric_derivative(L1):
    model = SolidSolutionSpec(phases=(SolidPhaseSpec(L1_J_mol=L1),), support_heat_capacity_J_K=0)
    amounts = {"A": 0.4, "B": 0.6}
    result = model.at_tp(amounts, 450)
    p, q = result.phases
    phase = model.phases[0]
    slope = phase.slope(p.x_A, 450)
    intercept = phase.gibbs(p.x_A, 450) - slope * p.x_A
    grid = np.linspace(0, 1, 10001)
    assert min(phase.gibbs(grid, 450) - slope * grid - intercept) > -1e-7
    assert phase.slope(q.x_A, 450) == pytest.approx(slope, abs=1e-5)
    assert result.support_deficit_J_mol < 1e-7 * R * 450
    eps = 0.001
    derivative = (
        model.at_tp(amounts, 450 + eps).gibbs_J - model.at_tp(amounts, 450 - eps).gibbs_J
    ) / (2 * eps)
    assert result.enthalpy_J == pytest.approx(result.gibbs_J - 450 * derivative, abs=1e-5)
    assert model.at_ph(0, amounts, result.enthalpy_J).temperature_K == pytest.approx(450)


def test_pure_ideal_empty_and_critical_limits():
    model = SolidSolutionSpec()
    for z in (0, 1):
        result = model.at_tp({"A": z, "B": 1 - z}, 300)
        assert result.gibbs_J == 0
        assert len(result.phases) == 1
    critical = model.at_tp({"A": 0.5, "B": 0.5}, 10000 / (2 * R))
    assert len(critical.phases) == 1
    ideal = replace(model, phases=(SolidPhaseSpec(L0_J_mol=0),))
    result = ideal.at_tp({"A": 0.5, "B": 0.5}, 300)
    assert result.gibbs_J == pytest.approx(-R * 300 * log(2))
    assert model.at_ph(0, {"A": 0, "B": 0}, 10).temperature_K == pytest.approx(350)
    assert SolidSolutionSpec.from_dict(model.to_dict()) == model


@pytest.mark.parametrize(
    "amounts, temperature",
    [
        ({"A": -1, "B": 1}, 300),
        ({"A": float("nan"), "B": 1}, 300),
        ({"A": 1, "B": 1}, float("nan")),
        ({"A": 1, "B": 1}, 900),
    ],
)
def test_nonphysical_input_is_rejected(amounts, temperature):
    with pytest.raises(ValueError):
        SolidSolutionSpec().at_tp(amounts, temperature)


def test_heat_measure_disable_and_material_replacement_preserve_actual_heat():
    model = SolidSolutionSpec()
    runtime = session(model)
    initial = solid_enthalpy(runtime._state, model)
    execute(runtime, "heat", target_temperature_K=800, duration_s=60)
    hot = runtime._state
    assert hot.temperature_K > 700
    assert len(hot.phases.phases) == 1
    assert hot.species_amounts == {"A": 0.0005, "B": 0.0005}
    assert solid_enthalpy(hot, model) - initial == pytest.approx(
        hot.ledger.energy_jacket_J, abs=1e-6
    )
    measurement = execute(runtime, "measure", instrument="solid_phase_meter").observation
    assert measurement.channel("temperature").unit == "K"
    assert measurement.channel("temperature").value == hot.temperature_K
    execute(runtime, "pause_control")
    execute(runtime, "wait", duration_s=5)
    assert runtime._state.temperature_K == pytest.approx(hot.temperature_K, abs=1e-7)
    runtime.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    expected_H = initial + 0.2 * (hot.temperature_K - 300)
    assert solid_enthalpy(runtime._state, model) == pytest.approx(expected_H, abs=1e-7)
    assert 300 < runtime._state.temperature_K < hot.temperature_K
    execute(runtime, "resume_control")
    execute(runtime, "wait", duration_s=5)
    runtime.reset(options={"scope": "vessel", "vessel_ids": ["reactor"]})
    assert runtime._state.temperature_K == 300


@pytest.mark.parametrize("temperature,target", [(290, 310), (850, 830)])
def test_temperature_boundary_allows_inward_heat(temperature, target):
    runtime = session(temperature_K=temperature)
    execute(runtime, "heat", target_temperature_K=target, duration_s=10)
    assert min(temperature, target) < runtime._state.temperature_K < max(temperature, target)
    assert runtime._state.ledger.time_s == 10


def test_numerical_rollback_control_rejection_and_corrupted_phase(monkeypatch):
    import chemworld.runtime.solid_solutions as module

    runtime = session()
    before = runtime._state
    result = runtime.execute({"operation": "heat", "target_temperature_K": 900, "duration_s": 10})
    assert result.info["operation_outcome"] == "rejected"
    rejected = SolidSolutionRuntime().advance(
        before, {"duration_s": 1}, jacket_program=lambda t: 500
    )
    assert rejected.outcome == "rejected"
    phase = next(iter(before.phases.phases.values()))
    corrupted = before.replace(
        phases=replace(
            before.phases,
            phases={phase.phase_id: replace(phase, species_amounts_mol={"A": 0.1, "B": 0.1})},
        )
    )
    with pytest.raises(ValueError):
        validate_solid(corrupted, SolidSolutionSpec())

    def fail(*args, **kwargs):
        raise RuntimeError("injected integration failure")

    monkeypatch.setattr(module, "integrate_phase_inventory", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 10})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.phases == before.phases
    assert runtime._state.temperature_K == before.temperature_K
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_heat_split_refinement():
    whole, split = session(), session()
    execute(whole, "heat", target_temperature_K=800, duration_s=40)
    execute(split, "heat", target_temperature_K=800, duration_s=20)
    execute(split, "wait", duration_s=20)
    assert whole._state.temperature_K == pytest.approx(
        split._state.temperature_K, rel=1e-6, abs=1e-4
    )
    assert whole._state.ledger.energy_jacket_J == pytest.approx(
        split._state.ledger.energy_jacket_J, rel=1e-6
    )


def test_solid_native_gym_cli_and_independent_process_continuation(tmp_path, capsys):
    import json
    import subprocess
    import sys

    from chemworld.cli import main
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.runtime.session_snapshot import SessionSnapshot

    spec = solid_world_spec()
    native = session()
    gym = ChemWorldEnv(composition=spec)
    gym.reset(seed=40)
    actions = [
        {"operation": "heat", "target_temperature_K": 800, "duration_s": 10},
        {"operation": "measure", "instrument": "solid_phase_meter"},
    ]
    paths = []
    for i, action in enumerate(actions):
        execute(native, **action)
        assert gym.step(action)[4]["transaction_status"] == "committed"
        path = tmp_path / f"action-{i}.json"
        path.write_text(json.dumps(action), encoding="utf-8")
        paths.append(str(path))
    assert gym._state.to_dict() == native._state.to_dict()
    world, snapshot = tmp_path / "world.json", tmp_path / "snapshot.json"
    world.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    main(
        [
            "render",
            "--composition",
            str(world),
            "--seed",
            "40",
            "--actions",
            *paths,
            "--format",
            "json",
            "--snapshot",
            str(snapshot),
        ]
    )
    assert len(json.loads(capsys.readouterr().out)["steps"]) == 2
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(snapshot))
    assert restored._state.to_dict() == native._state.to_dict()
    target = tmp_path / "continued.json"
    code = """import json,sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
r=ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1]))
v=r.execute({'operation':'wait','duration_s':10})
assert v.info['transaction_status']=='committed',v.info
with open(sys.argv[2],'w') as f:json.dump(r._state.to_dict(),f)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(snapshot), str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    execute(restored, "wait", duration_s=10)
    assert json.loads(target.read_text()) == json.loads(json.dumps(restored._state.to_dict()))


def test_solid_agent_history_exact_replay(tmp_path):
    from chemworld.agents.base import BaseAgent
    from chemworld.data.logging import load_jsonl
    from chemworld.eval.runner import run_agent
    from chemworld.eval.verify import verify_records

    class Sequence(BaseAgent):
        def act(self, history):
            return (
                {"operation": "heat", "target_temperature_K": 800, "duration_s": 10},
                {"operation": "measure", "instrument": "solid_phase_meter"},
                {"operation": "terminate"},
            )[len(history)]

    path = tmp_path / "solid.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=Sequence(),
        composition=solid_world_spec().to_dict(),
        world_split="public-dev",
        budget=100,
        objective="balanced",
        seed=40,
        output_path=path,
    )
    records = load_jsonl(path)
    report = verify_records(records, tolerance=1e-8)
    assert len(records) == 3
    assert report.verified, report.mismatches
