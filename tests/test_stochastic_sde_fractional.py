"""Declared fractional weak output, exact saved maps and legacy linear identity."""

import json
import math
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

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
from chemworld.physchem.stochastic_sde import validate_sde_continuation
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.biochemical_worlds import biochemical_network
from chemworld.world.stochastic_sde_world import stochastic_sde_world


def mathml(body):
    return '<math xmlns="http://www.w3.org/1998/Math/MathML">' + body + "</math>"


def setup(**changes):
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = StochasticSDESpec(
        **{"volume_L": 1e-15, "step_s": 0.125, "output_policy": "ri5_fractional", **changes}
    )
    amounts = {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO}
    return network, spec, amounts, initial_sde_continuation(network, spec, amounts, seed=123)


def world():
    network, spec, amounts, _ = setup(
        minimum_volume_L=0.75e-15,
        controls=(SDEControl("forward_scale", "forward", "rate_multiplier", 0, 4),),
    )
    return stochastic_sde_world(network, spec, amounts, seed=123)


def test_legacy_codec_is_exactly_the_old_field_set_and_default():
    original = StochasticSDESpec(1e-15, step_s=0.125)
    payload = original.to_dict()
    expected_fields = {
        "schema",
        "population_basis",
        "noise_unit",
        "volume_L",
        "temperature_K",
        "step_s",
        "min_step_s",
        "max_steps_per_advance",
        "max_evaluations_per_advance",
        "max_rejections_per_step",
        "max_root_iterations",
        "boundary_policy",
        "noises",
        "fluxes",
        "controls",
        "minimum_volume_L",
    }
    assert set(payload) == expected_fields
    restored = StochasticSDESpec.from_dict(json.loads(json.dumps(payload)))
    assert restored.output_policy == "linear" and restored.to_dict() == payload
    network = replace(biochemical_network("reversible"), stochastic=None)
    assert ReactionSDE(network, original).signature == ReactionSDE(network, restored).signature


@pytest.mark.parametrize("policy", [None, True, [], "unknown", ""])
def test_unknown_output_policy_is_rejected(policy):
    with pytest.raises(ValueError, match="output policy"):
        StochasticSDESpec(1e-15, output_policy=policy)


def test_new_policy_roundtrip_and_distinct_saved_packet():
    network, spec, amounts, memory = setup()
    payload = spec.to_dict()
    assert payload["output_policy"] == "ri5_fractional"
    assert StochasticSDESpec.from_dict(json.loads(json.dumps(payload))) == spec
    result = advance_sde(network, spec, amounts, memory, 0.03125)
    assert result.continuation["pending"]["schema"] == "ri5-fractional-dense-packet-v1"
    with pytest.raises(ValueError):
        advance_sde(
            network,
            replace(spec, output_policy="linear"),
            result.amounts_mol,
            result.continuation,
            0.03125,
        )


@pytest.mark.parametrize("kind", ["ou", "gbm"])
@pytest.mark.parametrize("hat", [-math.sqrt(3), 0.0, math.sqrt(3)])
@pytest.mark.parametrize("tilde", [-1, 1])
def test_fractional_output_matches_independent_scalar_polynomial(kind, hat, tilde, monkeypatch):
    from chemworld.physchem import stochastic_sde

    original = biochemical_network("reversible")
    network = replace(
        original,
        stochastic=None,
        reactions=(
            replace(
                original.reactions[0],
                rate_law=replace(original.reactions[0].rate_law, parameters={"k": 0.0}),
            ),
        ),
    )
    mu = 0.2 if kind == "gbm" else 1.0
    expression = (
        "<apply><minus/><apply><times/><ci>mu</ci><ci>n0</ci></apply></apply>"
        if kind == "gbm"
        else "<apply><times/><ci>mu</ci><apply><minus/><ci>n0</ci><ci>center</ci></apply></apply>"
    )
    flux_parameters = (SDEParameter("mu", mu, D(time=-1)),)
    if kind == "ou":
        flux_parameters += (SDEParameter("center", 50, D()),)
    noise = ReactionNoise(
        "forward",
        mathml(
            "<apply><times/><ci>sigma</ci><ci>n0</ci></apply>"
            if kind == "gbm"
            else "<ci>sigma</ci>"
        ),
        (SDEParameter("sigma", 0.25 if kind == "gbm" else 2, D(time=-0.5)),),
    )
    spec = StochasticSDESpec(
        1e-15,
        step_s=0.5,
        output_policy="ri5_fractional",
        fluxes=(ReactionFlux("forward", mathml(expression), flux_parameters),),
        noises=(noise,),
    )
    initial = {"A": 1.0, "B": 9999.0} if kind == "gbm" else {"A": 51.0, "B": 49.0}
    amounts = {k: n / AVOGADRO for k, n in initial.items()}
    memory = initial_sde_continuation(network, spec, amounts, seed=17)
    monkeypatch.setattr(stochastic_sde, "draw_ri5", lambda _rng, _m: RI5Draw((hat,), (tilde,)))
    t = 0.21875
    z = (0.2 if kind == "gbm" else -1) * t
    r3 = 1 + z + z * z / 2 + z * z * z / 6
    expected = (
        r3 + 0.25 * math.sqrt(t) * hat * (1 + z - z * z / 18) + 0.5 * 0.25**2 * t * (hat * hat - 1)
        if kind == "gbm"
        else 50 + r3 + 2 * math.sqrt(t) * hat * (1 + z / 2 - z * z / 18)
    )
    result = advance_sde(network, spec, amounts, memory, t)
    np.testing.assert_allclose(
        [result.amounts_mol[k] * AVOGADRO for k in ("A", "B")],
        [expected, sum(initial.values()) - expected],
        atol=1e-10,
        rtol=1e-10,
    )


def test_every_feasible_wait_split_and_json_has_the_same_full_future_state():
    network, spec, amounts, memory = setup()
    full = advance_sde(network, spec, amounts, memory, 0.546875)
    current, continuation = amounts, memory
    for _ in range(35):
        result = advance_sde(network, spec, current, json.loads(json.dumps(continuation)), 0.015625)
        current, continuation = result.amounts_mol, result.continuation
    assert current == full.amounts_mol and continuation == full.continuation


def test_fractional_evaluations_are_charged_once_with_no_extra_process_draw(monkeypatch):
    network, spec, amounts, memory = setup(max_evaluations_per_advance=36)
    originals = {name: getattr(ReactionSDE, name) for name in ("drift", "diffusion")}
    actual = []
    for name, function in originals.items():

        def measured(self, t, values, _function=function, _name=name):
            actual.append(_name)
            return _function(self, t, values)

        monkeypatch.setattr(ReactionSDE, name, measured)
    first = advance_sde(network, spec, amounts, memory, 0.03125)
    assert len(actual) == first.diagnostics["evaluations"] == 24
    actual.clear()
    second = advance_sde(network, spec, first.amounts_mol, first.continuation, 0.03125)
    assert len(actual) == second.diagnostics["evaluations"] == 36
    assert first.continuation["rng_state"] == second.continuation["rng_state"]
    actual.clear()
    validate_sde_continuation(
        network, spec, second.amounts_mol, second.continuation, verify_endpoint=False
    )
    assert not actual


def test_fractional_callback_budget_rolls_back_whole_candidate():
    network, spec, amounts, memory = setup(max_evaluations_per_advance=23)
    saved = deepcopy((amounts, memory))
    with pytest.raises(EventBudgetError):
        advance_sde(network, spec, amounts, memory, 0.03125)
    assert (amounts, memory) == saved


def test_feasible_endpoint_does_not_authorize_a_negative_fractional_output(monkeypatch):
    from chemworld.physchem import stochastic_sde

    original = biochemical_network("reversible")
    network = replace(original, stochastic=None, reactions=(original.reactions[0],))
    spec = StochasticSDESpec(
        1e-15,
        step_s=16,
        output_policy="ri5_fractional",
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
        stochastic_sde, "draw_ri5", lambda _rng, _m: RI5Draw((-math.sqrt(3),), (-1,))
    )
    saved = deepcopy((amounts, memory))
    full = advance_sde(network, spec, amounts, memory, 16)
    assert all(n >= 0 for n in full.amounts_mol.values())
    with pytest.raises(ValueError, match="fractional RI5 output"):
        advance_sde(network, spec, amounts, memory, 1)
    assert (amounts, memory) == saved


def test_forged_saved_fractional_packet_rejected_before_advance():
    network, spec, amounts, memory = setup()
    first = advance_sde(network, spec, amounts, memory, 0.03125)
    for field, value in (
        ("schema", "ri5-linear-dense-packet-v1"),
        ("right_populations", [100.0, 20.75]),
    ):
        forged = deepcopy(first.continuation)
        forged["pending"][field] = value
        with pytest.raises(ValueError):
            advance_sde(network, spec, first.amounts_mol, forged, 0.03125)


@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_actual_world_full_json_and_fresh_process_after_control_sampling(noise, tmp_path):
    session = ChemWorldRuntime(composition=world(), observation_noise_mode=noise)
    session.reset(seed=7)
    try:
        for action in (
            {"operation": "wait", "duration_s": 0.0625},
            {"operation": "set_sde_parameters", "values": {"forward_scale": 0.5}},
            {"operation": "measure", "instrument": "sde_aliquot_meter"},
            {"operation": "wait", "duration_s": 0.03125},
        ):
            assert session.execute(action).info["transaction_status"] == "committed"
        payload = json.loads(json.dumps(session.snapshot().to_dict()))
        restored = ChemWorldRuntime.from_snapshot(payload)
        action = {"operation": "wait", "duration_s": 0.375}
        try:
            assert session.execute(action).info["transaction_status"] == "committed"
            assert restored.execute(action).info["transaction_status"] == "committed"
            expected = json.loads(json.dumps(session._state.to_dict()))
            assert json.loads(json.dumps(restored._state.to_dict())) == expected
        finally:
            restored.close()
        source, destination = tmp_path / "checkpoint.json", tmp_path / "state.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        code = (
            "import json,sys;from pathlib import Path;"
            "from chemworld.runtime import ChemWorldRuntime;"
            "e=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')));"
            "r=e.execute({'operation':'wait','duration_s':.375});"
            "assert r.info['transaction_status']=='committed';"
            "Path(sys.argv[2]).write_text(json.dumps(e._state.to_dict()),encoding='utf-8');e.close()"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", code, str(source), str(destination)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(destination.read_text(encoding="utf-8")) == expected
    finally:
        session.close()


@pytest.mark.parametrize("entry", ["native", "gym"])
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
def test_actual_intervention_lifecycle_ten_step_replay(entry, noise, tmp_path):
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.envs.spaces import to_observation
    from chemworld.eval.verify import verify_records

    cls = ChemWorldRuntime if entry == "native" else ChemWorldEnv
    session = cls(composition=world(), observation_noise_mode=noise)
    session.reset(seed=7)
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
    path = tmp_path / "fractional.jsonl"
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
                    agent_metadata={"name": "sde-fractional-qualification"},
                )
        records = load_jsonl(path)
        result = verify_records(records, tolerance=0)
        assert result.verified and result.checked_steps == 10, result.mismatches
        assert len(records[5]["initialization_replay"]) == 1
        assert "sample_amount_A" in path.read_text(encoding="utf-8")
    finally:
        session.close()
