"""Requested RI5 endpoints receive physical acceptance before committing material."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).with_name("reference")))
from sde51e_scalar_formula import scalar_endpoint
from sde51e_statistics import cases, native_model, r3

from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.spaces import to_observation
from chemworld.eval.verify import verify_records
from chemworld.foundation.dimensions import DimensionVector as D
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem import (
    ReactionFlux,
    ReactionNoise,
    ReactionSDE,
    RI5Draw,
    SDEControl,
    SDEParameter,
    StochasticSDESpec,
    advance_sde,
    initial_sde_continuation,
)
from chemworld.physchem.stochastic_kinetics import EventBudgetError
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_network
from chemworld.world.stochastic_sde_world import stochastic_sde_world

CASES = {case["id"]: case for case in cases()}
BOUNDARY_FAILURES = (
    [("S13", seed) for seed in (36, 47, 452)]
    + [("S14", seed) for seed in (276, 316, 451, 487, 696, 785, 908)]
    + [("S17", seed) for seed in (595, 743, 820)]
)


def setup(**changes):
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = StochasticSDESpec(
        **{"volume_L": 1e-15, "step_s": 0.125, "output_policy": "ri5_truncated", **changes}
    )
    amounts = {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO}
    return network, spec, amounts, initial_sde_continuation(network, spec, amounts, seed=123)


def test_policy_roundtrip_has_its_own_binding_and_no_pending_output():
    network, spec, amounts, memory = setup()
    assert StochasticSDESpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    result = advance_sde(network, spec, amounts, memory, 0.03125)
    assert result.continuation["pending"] is None
    assert result.continuation["time_s"] == result.continuation["mesh_time_s"] == 0.03125
    for policy in ("linear", "ri5_fractional"):
        assert (
            ReactionSDE(network, replace(spec, output_policy=policy)).signature
            != memory["model_signature"]
        )
        with pytest.raises(ValueError):
            advance_sde(network, replace(spec, output_policy=policy), amounts, memory, 0.03125)


@pytest.mark.parametrize("name", ["S01", "S05"])
@pytest.mark.parametrize("hat", [-math.sqrt(3), 0.0, math.sqrt(3)])
@pytest.mark.parametrize("tilde", [-1, 1])
def test_actual_short_step_matches_independent_scalar_formula(name, hat, tilde, monkeypatch):
    from chemworld.physchem import stochastic_sde

    case = {**CASES[name], "output_policy": "ri5_truncated"}
    network, spec = native_model(case)
    amounts = {key: value / AVOGADRO for key, value in case["initial"].items()}
    memory = initial_sde_continuation(network, spec, amounts, seed=17)
    draws = []

    def draw(_rng, _dimension):
        draws.append(1)
        return RI5Draw((hat,), (tilde,))

    monkeypatch.setattr(stochastic_sde, "draw_ri5", draw)
    duration = 0.21875
    result = advance_sde(network, spec, amounts, memory, duration)
    expected = scalar_endpoint({**case, "step_s": duration}, hat)
    np.testing.assert_allclose(
        [result.amounts_mol[key] * AVOGADRO for key in ("A", "B")],
        [expected, sum(case["initial"].values()) - expected],
        atol=1e-10,
        rtol=1e-10,
    )
    assert len(draws) == result.diagnostics["completed_steps"] == 1
    assert result.continuation["pending"] is None


@pytest.mark.parametrize("name,seed", BOUNDARY_FAILURES)
def test_every_recorded_negative_fractional_seed_has_a_physical_requested_endpoint(name, seed):
    case = {**CASES[name], "output_policy": "ri5_truncated"}
    network, spec = native_model(case)
    amounts = {key: value / AVOGADRO for key, value in case["initial"].items()}
    memory = initial_sde_continuation(network, spec, amounts, seed=seed)
    saved = deepcopy((amounts, memory))
    result = advance_sde(network, spec, amounts, memory, case["duration_s"])
    assert all(math.isfinite(value) and value >= 0 for value in result.amounts_mol.values())
    assert result.continuation["pending"] is None
    assert result.continuation["time_s"] == case["duration_s"]
    assert (amounts, memory) == saved


def test_final_interval_shorter_than_minimum_is_one_real_step():
    network, spec, amounts, memory = setup(noises=(), max_evaluations_per_advance=3)
    result = advance_sde(network, spec, amounts, memory, 1e-15)
    assert result.actual_duration_s == result.continuation["time_s"] == 1e-15
    assert result.diagnostics["completed_steps"] == 1
    assert result.diagnostics["evaluations"] == 3
    assert result.continuation["pending"] is None


def test_unrepresentable_request_never_changes_stock_or_stream():
    network, spec, amounts, memory = setup(noises=())
    result = advance_sde(network, spec, amounts, memory, 1.0)
    saved = deepcopy(result)
    with pytest.raises(ValueError, match="finite clock"):
        advance_sde(
            network,
            spec,
            result.amounts_mol,
            result.continuation,
            float.fromhex("0x0.0000000000001p-1022"),
        )
    assert result == saved


def test_only_the_actual_short_endpoint_callbacks_are_charged(monkeypatch):
    network, spec, amounts, memory = setup(max_evaluations_per_advance=12)
    actual = []
    for name in ("drift", "diffusion"):
        original = getattr(ReactionSDE, name)

        def callback(self, *args, _original=original, _name=name):
            actual.append(_name)
            return _original(self, *args)

        monkeypatch.setattr(ReactionSDE, name, callback)
    result = advance_sde(network, spec, amounts, memory, 0.03125)
    assert len(actual) == result.diagnostics["evaluations"] == 12


def test_callback_budget_never_partially_commits_a_candidate():
    network, spec, amounts, memory = setup(max_evaluations_per_advance=11)
    saved = deepcopy((amounts, memory))
    with pytest.raises(EventBudgetError):
        advance_sde(network, spec, amounts, memory, 0.03125)
    assert (amounts, memory) == saved


def test_requested_boundaries_change_the_declared_numerical_grid():
    case = {**CASES["S01"], "output_policy": "ri5_truncated"}
    network, spec = native_model(case)
    spec = replace(spec, noises=())
    amounts = {key: value / AVOGADRO for key, value in case["initial"].items()}
    memory = initial_sde_continuation(network, spec, amounts, seed=17)
    whole = advance_sde(network, spec, amounts, memory, 1.0)
    first = advance_sde(network, spec, amounts, memory, 0.25)
    split = advance_sde(network, spec, first.amounts_mol, first.continuation, 0.75)
    assert whole.diagnostics["completed_steps"] == 2
    assert split.continuation["completed_steps"] == 3
    assert whole.amounts_mol != split.amounts_mol
    assert whole.amounts_mol["A"] * AVOGADRO == pytest.approx(50 + r3(-0.5) ** 2, abs=1e-11)
    assert split.amounts_mol["A"] * AVOGADRO == pytest.approx(
        50 + r3(-0.25) ** 2 * r3(-0.5), abs=1e-11
    )


def test_pending_dense_output_cannot_be_forged_for_an_endpoint_policy():
    network, spec, amounts, memory = setup()
    forged = deepcopy(memory)
    forged["pending"] = {}
    with pytest.raises(ValueError, match="cannot carry pending"):
        advance_sde(network, spec, amounts, forged, 0)


def test_reject_policy_checks_the_requested_endpoint_before_accepting_a_long_step(monkeypatch):
    from chemworld.physchem import stochastic_sde

    network, _ = native_model(CASES["S01"])

    def mathml(body):
        return '<math xmlns="http://www.w3.org/1998/Math/MathML">' + body + "</math>"

    spec = StochasticSDESpec(
        1e-15,
        step_s=16,
        output_policy="ri5_truncated",
        boundary_policy="reject",
        fluxes=(
            ReactionFlux("forward", mathml("<ci>net</ci>"), (SDEParameter("net", -1, D(time=-1)),)),
        ),
        noises=(
            ReactionNoise(
                "forward", mathml("<ci>sigma</ci>"), (SDEParameter("sigma", 2, D(time=-0.5)),)
            ),
        ),
    )
    amounts = {"A": 1 / AVOGADRO, "B": 20 / AVOGADRO}
    memory = initial_sde_continuation(network, spec, amounts, seed=17)
    monkeypatch.setattr(
        stochastic_sde, "draw_ri5", lambda _rng, _dimension: RI5Draw((-math.sqrt(3),), (-1,))
    )
    full = advance_sde(network, spec, amounts, memory, 16)
    assert all(value >= 0 for value in full.amounts_mol.values())
    saved = deepcopy((amounts, memory))
    with pytest.raises(ValueError, match="negative candidate"):
        advance_sde(network, spec, amounts, memory, 1)
    assert (amounts, memory) == saved


@pytest.mark.parametrize("entry,cls", [("native", ChemWorldRuntime), ("gym", ChemWorldEnv)])
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_real_intervention_protocol_replays_and_resumes_in_a_fresh_process(
    entry, cls, noise, tmp_path
):
    network, spec, amounts, _ = setup(
        minimum_volume_L=0.75e-15,
        controls=(SDEControl("forward_scale", "forward", "rate_multiplier", 0, 4),),
    )
    world = stochastic_sde_world(network, spec, amounts, seed=123)
    session = cls(composition=world, observation_noise_mode=noise)
    session.reset(seed=7)
    path = tmp_path / "trajectory.jsonl"
    actions = [
        {"operation": "wait", "duration_s": 0.0625},
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}},
        {"operation": "measure", "instrument": "sde_aliquot_meter"},
        {"operation": "wait", "duration_s": 0.03125},
        {"operation": "measure", "instrument": "sde_amount_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "set_sde_parameters", "values": {"forward_scale": 0}},
        {"operation": "measure", "instrument": "sde_aliquot_meter"},
        {"operation": "wait", "duration_s": 0.125},
        {"operation": "terminate"},
    ]
    try:
        with TrajectoryLogger(path) as logger:
            for index, action in enumerate(actions, 1):
                if index == 6:
                    session.reset(options={"scope": "material"})
                if entry == "native":
                    result = session.execute(action)
                    observation = to_observation(result.observation.values)
                    reward, terminated, truncated, info = (
                        result.reward,
                        result.terminated,
                        result.truncated,
                        result.info,
                    )
                else:
                    observation, reward, terminated, truncated, info = session.step(action)
                assert info["transaction_status"] == "committed", info
                logger.log(
                    task_info={**session.task_info(), **session.evaluator_provenance()},
                    step=index,
                    action=action,
                    observation=observation_to_json(observation),
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                    agent_metadata={"name": "truncated-ri5-functional"},
                )
                if index == 4:
                    checkpoint = tmp_path / "checkpoint.json"
                    checkpoint.write_text(
                        json.dumps(session.snapshot().to_dict()), encoding="utf-8"
                    )
                    restored = ChemWorldRuntime.from_snapshot(json.loads(checkpoint.read_text()))
                    try:
                        result = restored.execute({"operation": "wait", "duration_s": 0.375})
                        assert result.info["transaction_status"] == "committed", result.info
                        expected = json.loads(json.dumps(restored._state.to_dict()))
                    finally:
                        restored.close()
                    output = tmp_path / "state.json"
                    code = (
                        "import json,sys;from pathlib import Path;"
                        "from chemworld.runtime import ChemWorldRuntime;"
                        "s=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()));"
                        "r=s.execute({'operation':'wait','duration_s':.375});"
                        "assert r.info['transaction_status']=='committed';"
                        "Path(sys.argv[2]).write_text(json.dumps(s._state.to_dict()));s.close()"
                    )
                    child = subprocess.run(
                        [
                            sys.executable,
                            "-I",
                            "-X",
                            "utf8",
                            "-c",
                            code,
                            str(checkpoint),
                            str(output),
                        ],
                        cwd=tmp_path,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    (tmp_path / "fresh.stdout").write_text(child.stdout, encoding="utf-8")
                    (tmp_path / "fresh.stderr").write_text(child.stderr, encoding="utf-8")
                    assert child.returncode == 0, child.stdout + child.stderr
                    assert json.loads(output.read_text()) == expected
        replay = verify_records(load_jsonl(path), tolerance=0)
        assert replay.verified, replay.mismatches
        assert replay.checked_steps == 10
        assert len(load_jsonl(path)[5]["initialization_replay"]) == 1
    finally:
        session.close()
