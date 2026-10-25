"""Integer strategy contracts and actual native session consumers."""

import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_kinetics import (
    AVOGADRO,
    MAX_COUNT,
    EventBudgetError,
    advance_stochastic,
    counts_to_mol,
    initial_continuation,
    mol_to_counts,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.stochastic_world import stochastic_world


def network(*, reversible=True, k=0.1, max_events=1000000):
    equations = [("forward", "A => B", k)]
    if reversible:
        equations.append(("reverse", "B => A", 0.2))
    return ReactionNetworkSpec(
        "synthetic-ab",
        (SpeciesSpec("A", "H2"), SpeciesSpec("B", "H2")),
        tuple(
            ReactionSpec.from_equation(
                reaction_id=name,
                equation=eq,
                rate_law=RateLawSpec(name, "mass_action", {"k": rate}),
            )
            for name, eq, rate in equations
        ),
        stochastic={"volume_L": 0.01, "max_events_per_advance": max_events},
    )


@pytest.mark.parametrize("n", [0, 1, 2, 100, 1000, MAX_COUNT])
def test_integer_roundtrip(n):
    assert mol_to_counts(counts_to_mol({"A": n})) == {"A": n}


@pytest.mark.parametrize("n", [-1, 0.1, 1.5, float("inf"), float("nan"), True])
def test_noninteger_inventory_rejected(n):
    with pytest.raises(ValueError):
        counts_to_mol({"A": n})
    with pytest.raises(ValueError):
        mol_to_counts({"A": n / AVOGADRO if not isinstance(n, bool) else n})


def test_second_order_conversion_includes_factorial():
    n = ReactionNetworkSpec(
        "dimer",
        (SpeciesSpec("A", "H"), SpeciesSpec("B", "H2")),
        (
            ReactionSpec.from_equation(
                reaction_id="dimer",
                equation="2 A => B",
                rate_law=RateLawSpec("k", "mass_action", {"k": 3.0}),
            ),
        ),
        stochastic={"volume_L": 1e-15},
    )
    channel = n.stochastic_channels[0]
    assert channel.constant == pytest.approx(6 / (AVOGADRO * 1e-15))
    assert channel.propensity({"A": 5}) == pytest.approx(3 * 5 * 4 / (AVOGADRO * 1e-15))
    assert channel.propensity({"A": 1}) == 0


@pytest.mark.parametrize(
    "mutation", ["fractional", "reversible", "activity", "orders", "arrhenius", "negative"]
)
def test_unsupported_kinetics_fail_explicitly(mutation):
    payload = network().to_dict()
    reaction = payload["reactions"][0]
    if mutation == "fractional":
        reaction["equation"] = "0.5 A => 0.5 B"
        reaction["stoichiometry"] = {"A": -0.5, "B": 0.5}
    elif mutation == "reversible":
        reaction["equation"] = "A <=> B"
        reaction["reversible"] = True
    elif mutation == "activity":
        reaction["rate_law"]["uses_activities"] = True
    elif mutation == "orders":
        # 2 A => A + B is now admitted as elementary partial regeneration.
        # Retain an explicit unsupported-order check above molecularity three.
        reaction["forward_orders"] = {"A": 4}
    elif mutation == "arrhenius":
        reaction["rate_law"].update(equation_id="arrhenius", parameters={"A": 1, "Ea_J_per_mol": 0})
    else:
        reaction["rate_law"]["parameters"]["k"] = -1
    with pytest.raises(ValueError):
        ReactionNetworkSpec.from_dict(payload)


def test_network_roundtrip_and_no_implicit_ode_switch():
    n = network()
    assert (
        ReactionNetworkSpec.from_dict(json.loads(json.dumps(n.to_dict()))).to_dict() == n.to_dict()
    )
    with pytest.raises(ValueError, match="implicit ODE"):
        n.integrate_batch({"A": 1}, volume_L=1, temperature_K=298, duration_s=1)
    assert (
        replace(n, stochastic=None)
        .integrate_batch({"A": 1}, volume_L=1, temperature_K=298, duration_s=1)
        .final_amounts_mol["A"]
        < 1
    )


def test_pending_event_is_exact_across_interval_splits():
    n, counts, initial = network(), {"A": 100, "B": 0}, initial_continuation(21)
    whole = advance_stochastic(n, counts, initial, 20)
    events = []
    continuation = initial
    for end in range(1, 21):
        result = advance_stochastic(n, counts, continuation, end - continuation["time_s"])
        counts, continuation = result.counts, json.loads(json.dumps(result.continuation))
        events.extend(result.events)
    assert tuple(events) == whole.events
    assert counts == whole.counts
    assert continuation == whole.continuation
    assert sum(counts.values()) == 100
    assert initial == initial_continuation(21)


def test_zero_propensity_and_zero_duration_do_not_draw_randomness():
    n, initial = network(k=0, reversible=False), initial_continuation(8)
    result = advance_stochastic(n, {"A": 100}, initial, 100)
    assert result.events == ()
    assert result.continuation["rng_state"] == initial["rng_state"]
    assert result.continuation["time_s"] == 100
    result = advance_stochastic(network(), {"A": 100}, initial, 0)
    assert result.continuation["rng_state"] == initial["rng_state"]


def test_event_limit_leaves_inputs_untouched():
    n, initial, counts = network(max_events=1), initial_continuation(2), {"A": 100}
    old = copy.deepcopy(initial)
    with pytest.raises(EventBudgetError):
        advance_stochastic(n, counts, initial, 10)
    assert initial == old and counts == {"A": 100}


def test_native_wait_reads_snapshot_and_public_redaction(tmp_path):
    e = ChemWorldRuntime(composition=stochastic_world(network(), {"A": 100}, seed=12))
    e.reset(seed=4)
    result = e.execute({"operation": "wait", "duration_s": 10})
    assert result.info["transaction_status"] == "committed"
    assert result.info["actual_duration_s"] == e._state.ledger.time_s == 10
    assert result.info["outcome_diagnostics"]["advance_strategy"] == "gillespie-direct"
    before = e._state.to_dict()
    for _ in range(10):
        e.observe()
    assert e._state.to_dict() == before
    assert "rng_state" not in json.dumps(result.info)
    assert "stochastic_continuation" not in e._state.to_dict(include_hidden=False)["process"]
    payload = json.loads(json.dumps(e.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(payload)
    action = {"operation": "wait", "duration_s": 5}
    e.execute(action)
    restored.execute(action)
    assert e._state.to_dict() == restored._state.to_dict()
    # Independent process reads saved continuation, not same-process memory.
    source, output = tmp_path / "snapshot.json", tmp_path / "state.json"
    source.write_text(json.dumps(payload))
    code = (
        "import json; from pathlib import Path; from chemworld.runtime import ChemWorldRuntime; "
        f"e=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        f"e.execute({action!r}); Path({str(output)!r}).write_text(json.dumps(e._state.to_dict()))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert json.loads(output.read_text()) == json.loads(json.dumps(e._state.to_dict()))


def test_actual_runtime_event_budget_is_numerical_rollback():
    e = ChemWorldRuntime(composition=stochastic_world(network(max_events=1), {"A": 100}))
    e.reset()
    before = e._state.to_dict()
    result = e.execute({"operation": "wait", "duration_s": 10})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert e._state.species_amounts == before["species_amounts"]
    assert e._state.process.stochastic_continuation == before["process"]["stochastic_continuation"]
    assert e._state.ledger.time_s == 0


@pytest.mark.parametrize(
    "mutation", ["thermal", "reaction", "fractional", "temperature", "no_continuation"]
)
def test_native_compile_rejects_unsupported_composition(mutation):
    payload = stochastic_world(network(), {"A": 100}).to_dict()
    state = payload["physics"]["initial_states"]["reactor"]
    if mutation in {"thermal", "reaction"}:
        payload["components"].append({"kind": mutation, "parameters": {}})
    elif mutation == "fractional":
        state["species_amounts"]["A"] = 100.5 / AVOGADRO
    elif mutation == "temperature":
        state["temperature_K"] += 1
    else:
        state["process"].pop("stochastic_continuation")
    with pytest.raises(ValueError):
        ChemWorldRuntime(composition=payload)


def test_actual_logged_replay(tmp_path):
    from chemworld.agent_interface import agent_view_bundle
    from chemworld.data.logging import TrajectoryLogger, load_jsonl, observation_to_json
    from chemworld.envs.chemworld_env import ChemWorldEnv
    from chemworld.eval.verify import verify_records

    e = ChemWorldEnv(composition=stochastic_world(network(), {"A": 100}, seed=7))
    e.reset(seed=0)
    task = {**e.task_info(), **e.evaluator_provenance()}
    path = tmp_path / "trajectory.jsonl"
    with TrajectoryLogger(path) as logger:
        for i, duration in enumerate((1, 2, 3), 1):
            action = {"operation": "wait", "duration_s": duration}
            observation, reward, terminated, truncated, info = e.step(action)
            logger.log(
                task_info=task,
                step=i,
                action=action,
                observation=observation_to_json(observation),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                agent_metadata={"name": "ssa-contract-test"},
                agent_view=agent_view_bundle(e, observation, info),
            )
    assert verify_records(load_jsonl(path)).verified


@pytest.mark.parametrize("latency", [0, 1])
def test_measurement_noise_and_frequency_do_not_change_process(latency):
    from chemworld.world.domain_modules import DomainModuleRegistry, DomainModuleSpec
    from chemworld.world.electrode_instruments import (
        electrical_instrument_spec,
        electrical_observation,
    )

    # A synthetic noisy voltage source tests the measurement infrastructure,
    # not a biochemical assay. The actual process always uses SSA.
    def observer(state, action, rng):
        return electrical_observation("test_source", float(rng.normal()), 0)

    instrument = electrical_instrument_spec("test_source", observer)
    instrument = replace(
        instrument,
        contract=replace(
            instrument.contract,
            latency_s=latency,
            clock_semantics="in_situ_end" if latency else "instantaneous_snapshot",
        ),
    )
    modules = DomainModuleRegistry(
        (
            *DomainModuleRegistry.default().modules,
            DomainModuleSpec("test-noisy-source", instruments=(instrument,)),
        )
    )
    payload = stochastic_world(network(), {"A": 100}, seed=19).to_dict()
    payload["components"].append(
        {"kind": "observation", "parameters": {"instruments": ["test_source"]}}
    )
    payload["task"].update(operations=["wait", "measure", "terminate"], instruments=["test_source"])
    measured = ChemWorldRuntime(composition=payload, domain_modules=modules)
    plain = ChemWorldRuntime(composition=payload, domain_modules=modules)
    measured.reset(seed=3)
    plain.reset(seed=3)
    for _ in range(3):
        result = measured.execute({"operation": "measure", "instrument": "test_source"})
        assert result.info["transaction_status"] == "committed"
        result = measured.execute({"operation": "wait", "duration_s": 1})
        assert result.info["transaction_status"] == "committed"
    plain.execute({"operation": "wait", "duration_s": 3 * (1 + latency)})
    assert measured._state.ledger.time_s == plain._state.ledger.time_s
    assert measured._state.species_amounts == plain._state.species_amounts
    assert (
        measured._state.process.stochastic_continuation
        == plain._state.process.stochastic_continuation
    )
    assert measured._state.ledger.cost - plain._state.ledger.cost == pytest.approx(0.03)


def test_fractional_destructive_instrument_not_silently_installed():
    payload = stochastic_world(network(), {"A": 100}).to_dict()
    payload["components"].append({"kind": "observation", "parameters": {"instruments": ["uvvis"]}})
    payload["task"].update(operations=["wait", "measure", "terminate"], instruments=["uvvis"])
    with pytest.raises(ValueError, match="integer sampler"):
        ChemWorldRuntime(composition=payload)


def test_strategy_rejects_heat_without_fabricating_a_process_failure():
    from chemworld.runtime.stochastic import StochasticRuntime

    composition = stochastic_world(network(), {"A": 100})
    state = composition.physics.initial_states["reactor"]
    result = StochasticRuntime(composition).advance(state, {"duration_s": 1}, heat=True)
    assert result.outcome == "rejected" and result.state is None
    assert result.actual_duration_s == 0
