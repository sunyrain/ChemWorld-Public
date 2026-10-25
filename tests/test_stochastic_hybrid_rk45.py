"""Explicit shared RK45 integration, strict old codecs and real native continuations."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.spaces import to_observation
from chemworld.eval.verify import verify_records
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.stochastic_hybrid import (
    advance_stochastic_hybrid,
    initial_hybrid_continuation,
)
from chemworld.physchem.stochastic_kinetics import EventBudgetError
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world
from reference.cross_partition51_cases import CASES as CROSS_CASES
from reference.cross_partition51_cases import author as cross_author
from reference.cross_partition51_cases import shared_world as cross_world
from reference.cross_partition51_reference import simulate as cross_reference
from reference.stochastic_hybrid51_cases import CASES as FEEDBACK_CASES
from reference.stochastic_hybrid51_cases import author as feedback_author
from reference.stochastic_hybrid51_cases import shared_world as feedback_world
from reference.stochastic_hybrid51_reference import simulate as feedback_reference

AUTHORS = {"feedback": feedback_author, "cross": cross_author}
WORLDS = {"feedback": feedback_world, "cross": cross_world}


def authored(kind, case=None, **changes):
    network, spec, amounts = AUTHORS[kind](case)
    return network, replace(spec, ode_method="RK45", **changes), amounts


def candidate(kind, case=None, *, seed=17, **changes):
    network, spec, amounts = authored(kind, case, **changes)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=seed)
    return advance_stochastic_hybrid(network, spec, amounts, memory, 3.0)


def shared_world(kind):
    original = WORLDS[kind](seed=17)
    payload = original.to_dict()
    for index, identity in enumerate(("reactor", "second")):
        spec = replace(original.physics.models_of("stochastic_hybrid")[identity], ode_method="RK45")
        payload["physics"]["models"]["stochastic_hybrid"]["bindings"][identity] = spec.to_dict()
        payload["physics"]["initial_states"][identity]["process"]["stochastic_continuation"] = (
            initial_hybrid_continuation(
                original.physics.network,
                spec,
                original.physics.initial_states[identity].species_amounts,
                seed=17 + index,
            )
        )
    return WorldCompositionSpec.from_dict(payload)


@pytest.mark.parametrize("kind", AUTHORS)
def test_old_exact_field_set_and_explicit_new_binding(kind):
    network, old, amounts = AUTHORS[kind]()
    expected = {
        "schema",
        "discrete_species",
        "stochastic_reactions",
        "volume_L",
        "temperature_K",
        "max_events_per_advance",
        "rtol",
        "atol",
        "max_step_s",
    }
    assert set(old.to_dict()) == expected
    assert type(old).from_dict(json.loads(json.dumps(old.to_dict()))) == old
    new = replace(old, ode_method="RK45")
    assert new.to_dict()["ode_method"] == "RK45"
    assert type(new).from_dict(json.loads(json.dumps(new.to_dict()))) == new
    assert (
        initial_hybrid_continuation(network, old, amounts, seed=17)["signature"]
        != (initial_hybrid_continuation(network, new, amounts, seed=17)["signature"])
    )
    with pytest.raises(ValueError, match="fields"):
        type(new).from_dict({**new.to_dict(), "unknown": 1})


@pytest.mark.parametrize("kind", AUTHORS)
@pytest.mark.parametrize("method", [None, True, [], "unknown", "LSODA"])
def test_unknown_method_rejected_without_solver_fallback(kind, method):
    with pytest.raises(ValueError, match="integration method"):
        replace(AUTHORS[kind]()[1], ode_method=method)


@pytest.mark.parametrize("kind", AUTHORS)
def test_saved_numerical_continuation_cannot_cross_methods(kind):
    network, old, amounts = AUTHORS[kind]()
    memory = initial_hybrid_continuation(network, old, amounts, seed=17)
    with pytest.raises(ValueError, match=r"hybrid|cross-partition"):
        advance_stochastic_hybrid(network, replace(old, ode_method="RK45"), amounts, memory, 0)


@pytest.mark.parametrize("kind", AUTHORS)
def test_real_shared_backend_receives_rk45_for_every_numerical_segment(kind, monkeypatch):
    from chemworld.physchem import stochastic_cross_partition, stochastic_hybrid

    module = stochastic_hybrid if kind == "feedback" else stochastic_cross_partition
    original = module.solve_ode
    methods = []

    def solve(*args, **kwargs):
        methods.append(kwargs["policy"].method)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "solve_ode", solve)
    result = candidate(kind)
    assert methods and set(methods) == {"RK45"}
    assert len(methods) == result.work["segments"]
    assert result.work["rhs_evaluations"] > 0


@pytest.mark.parametrize("case", FEEDBACK_CASES)
@pytest.mark.parametrize("seed", [1, 17, 37])
def test_feedback_matches_independent_closed_form(case, seed):
    result, expected = candidate("feedback", case, seed=seed), feedback_reference(case, seed)
    assert [e["reaction_id"] for e in result.events] == [
        e["reaction_id"] for e in expected["events"]
    ]
    assert [e["time_s"] for e in result.events] == pytest.approx(
        [e["time_s"] for e in expected["events"]], abs=2e-7
    )
    assert result.amounts_mol["A"] / case["A0_mol"] == pytest.approx(
        expected["fraction_A"], abs=2e-7
    )
    assert round(result.amounts_mol["E1"] * AVOGADRO) == expected["E1"]
    assert result.continuation["rng_state"] == expected["rng_state"]
    assert result.continuation["pending"]["targets"] == expected["targets"]
    assert result.continuation["pending"]["hazards"] == pytest.approx(expected["hazards"], abs=2e-7)


@pytest.mark.parametrize("case", CROSS_CASES)
def test_cross_partition_matches_independent_flow_availability_and_transfers(case):
    result, expected = candidate("cross", case), cross_reference(case, 17)
    assert [e["reaction_id"] for e in result.events] == [
        e["reaction_id"] for e in expected["events"]
    ]
    assert [e["time_s"] for e in result.events] == pytest.approx(
        [e["time_s"] for e in expected["events"]], abs=2e-7
    )
    for key in ("D", "A", "B"):
        assert result.amounts_mol[key] * AVOGADRO == pytest.approx(expected[key], abs=2e-7)
    assert result.continuation["rng_state"] == expected["rng_state"]
    assert result.continuation["pending"]["targets"] == expected["targets"]
    assert result.continuation["pending"]["hazards"] == pytest.approx(expected["hazards"], abs=2e-7)
    assert sum(result.continuation["partition_transfer"]["firings"].values()) == len(result.events)


@pytest.mark.parametrize("kind", AUTHORS)
def test_candidate_event_budget_keeps_original_stock_and_poisson_stream(kind):
    network, spec, amounts = authored(kind, max_events_per_advance=1)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    before = deepcopy((amounts, memory))
    with pytest.raises(EventBudgetError):
        advance_stochastic_hybrid(network, spec, amounts, memory, 3.0)
    assert (amounts, memory) == before


@pytest.mark.parametrize("kind", AUTHORS)
def test_real_budget_failure_rolls_back_material_clock_and_continuation(kind):
    network, spec, amounts = authored(kind, max_events_per_advance=1)
    session = ChemWorldRuntime(composition=stochastic_hybrid_world(network, spec, amounts, seed=17))
    session.reset(seed=7)
    try:
        before = session._state
        result = session.execute({"operation": "wait", "duration_s": 3.0})
        assert result.info["transaction_status"] == "rolled_back"
        assert session._state.species_amounts == before.species_amounts
        assert session._state.phases == before.phases
        assert (
            session._state.process.stochastic_continuation == before.process.stochastic_continuation
        )
        assert session._state.ledger == before.ledger.with_updates(
            cost=before.ledger.cost + 0.03, risk=min(1.0, before.ledger.risk + 0.08)
        )
    finally:
        session.close()


@pytest.mark.parametrize("kind", AUTHORS)
@pytest.mark.parametrize("noise", ["sequential", "keyed"])
@pytest.mark.parametrize("entry,cls", [("native", ChemWorldRuntime), ("gym", ChemWorldEnv)])
def test_real_shared_protocol_json_new_process_and_exact_logged_replay(
    kind, noise, entry, cls, tmp_path
):
    session = cls(composition=shared_world(kind), observation_noise_mode=noise)
    session.reset(seed=7)
    path = tmp_path / "trajectory.jsonl"
    actions = [
        {"operation": "wait", "duration_s": 1.0},
        {"operation": "wait", "duration_s": 2.0},
        {"operation": "wait", "duration_s": 1.0},
        {"operation": "terminate"},
    ]
    try:
        with TrajectoryLogger(path) as logger:
            for index, action in enumerate(actions, 1):
                if entry == "native":
                    result = session.execute(action)
                    observation = to_observation(result.observation.values)
                    reward, ended, truncated, info = (
                        result.reward,
                        result.terminated,
                        result.truncated,
                        result.info,
                    )
                else:
                    observation, reward, ended, truncated, info = session.step(action)
                assert info["transaction_status"] == "committed", {
                    key: info.get(key)
                    for key in ("error_message", "transaction_status", "outcome_diagnostics")
                }
                logger.log(
                    task_info={**session.task_info(), **session.evaluator_provenance()},
                    step=index,
                    action=action,
                    observation=observation_to_json(observation),
                    reward=reward,
                    terminated=ended,
                    truncated=truncated,
                    info=info,
                    agent_metadata={"name": "rk45-controlled-hybrid"},
                )
                if index != 1:
                    continue
                source = tmp_path / "checkpoint.json"
                source.write_text(json.dumps(session.snapshot().to_dict()), encoding="utf-8")
                restored = ChemWorldRuntime.from_snapshot(
                    json.loads(source.read_text(encoding="utf-8"))
                )
                try:
                    assert all(
                        spec.ode_method == "RK45"
                        for spec in restored.compiled_composition.spec.physics.models_of(
                            "stochastic_hybrid"
                        ).values()
                    )
                    result = restored.execute({"operation": "wait", "duration_s": 2.0})
                    assert result.info["transaction_status"] == "committed"
                    expected = json.loads(
                        json.dumps({"state": restored._state.to_dict(), "info": result.info})
                    )
                finally:
                    restored.close()
                output = tmp_path / "future.json"
                code = (
                    "import json,sys;from pathlib import Path;"
                    "from chemworld.runtime import ChemWorldRuntime;"
                    "s=ChemWorldRuntime.from_snapshot(json.loads(Path(sys.argv[1]).read_text()));"
                    "r=s.execute({'operation':'wait','duration_s':2.});"
                    "assert r.info['transaction_status']=='committed';"
                    "Path(sys.argv[2]).write_text(json.dumps({'state':s._state.to_dict(),'info':r.info}));"
                    "s.close()"
                )
                child = subprocess.run(
                    [sys.executable, "-I", "-X", "utf8", "-c", code, str(source), str(output)],
                    cwd=tmp_path,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=60,
                )
                (tmp_path / "fresh.stdout").write_text(child.stdout, encoding="utf-8")
                (tmp_path / "fresh.stderr").write_text(child.stderr, encoding="utf-8")
                assert child.returncode == 0, child.stdout + child.stderr
                assert json.loads(output.read_text(encoding="utf-8")) == expected
                (tmp_path / "expected-future.json").write_text(
                    json.dumps(expected), encoding="utf-8"
                )
        replay = verify_records(load_jsonl(path), tolerance=0)
        assert replay.verified, replay.mismatches
        assert replay.checked_steps == 4
    finally:
        session.close()
