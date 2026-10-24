"""Original-inventory chemical PDMP, exact saved state and explicit boundaries."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.foundation.balances import TransitionBalance
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.stochastic_hybrid import (
    HYBRID_ALGORITHM,
    StochasticHybridSpec,
    advance_stochastic_hybrid,
    compile_hybrid,
    initial_hybrid_continuation,
)
from chemworld.physchem.stochastic_kinetics import EventBudgetError
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world
from reference.stochastic_hybrid51_cases import CASES, author, shared_world
from reference.stochastic_hybrid51_reference import simulate


def advance(case=None, seed=17, duration=3., max_events=10000):
    network, spec, amounts = author(case, max_events=max_events)
    initial = initial_hybrid_continuation(network, spec, amounts, seed=seed)
    return advance_stochastic_hybrid(network, spec, amounts, initial, duration)


def runtime(*, max_events=10000, seed=17):
    network, spec, amounts = author(max_events=max_events)
    instance = ChemWorldRuntime(composition=stochastic_hybrid_world(
        network, spec, amounts, seed=seed))
    instance.reset(seed=7)
    return instance


@pytest.mark.parametrize("case", CASES)
def test_same_system_feedback_matches_independent_closed_form(case):
    actual, expected = advance(case), simulate(case, 17)
    assert [e["reaction_id"] for e in actual.events] == [
        e["reaction_id"] for e in expected["events"]]
    assert [e["time_s"] for e in actual.events] == pytest.approx(
        [e["time_s"] for e in expected["events"]], abs=2e-7)
    assert actual.amounts_mol["A"] / case["A0_mol"] == pytest.approx(
        expected["fraction_A"], abs=2e-7)
    assert round(actual.amounts_mol["E1"] * AVOGADRO) == expected["E1"]
    assert actual.continuation["rng_state"] == expected["rng_state"]
    assert actual.continuation["pending"]["targets"] == expected["targets"]
    assert actual.continuation["pending"]["hazards"] == pytest.approx(expected["hazards"], abs=2e-7)


def test_closed_continuous_and_exact_integer_conservation_and_input_purity():
    network, spec, amounts = author()
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    original = copy.deepcopy((amounts, memory))
    result = advance_stochastic_hybrid(network, spec, amounts, memory, 3)
    assert (amounts, memory) == original
    assert result.amounts_mol["A"] + result.amounts_mol["B"] == pytest.approx(
        amounts["A"], rel=1e-12, abs=0)
    assert sum(round(result.amounts_mol[k] * AVOGADRO) for k in ("E0", "E1")) == 1
    assert result.work["rhs_evaluations"] > 0
    assert result.work["random_draws"] == len(result.events) + 2


def test_no_ode_or_jump_limit_and_zero_duration_does_not_draw():
    case = {**CASES[0], "alpha": 0., "beta": 0., "k1": 0.}
    result = advance(case)
    assert not result.events
    assert result.amounts_mol["A"] / case["A0_mol"] == pytest.approx(math.exp(-.45), abs=1e-9)
    case.update(k0=0.)
    result = advance(case)
    assert result.amounts_mol == author(case)[2]
    assert result.work["random_draws"] == 2
    assert result.continuation["pending"]["hazards"] == [0., 0.]
    result = advance(duration=0)
    assert result.continuation["pending"] is None and result.work["random_draws"] == 0


def test_finite_integrated_hazard_can_leave_a_nonzero_no_jump_probability():
    case = {**CASES[0], "k0": 1., "k1": 0., "alpha": .1, "beta": 0.}
    result = advance(case, duration=30.)
    assert result.continuation["pending"]["targets"][0] > .1
    assert result.events == ()
    assert result.continuation["pending"]["hazards"][0] == pytest.approx(.1, abs=1e-10)
    assert math.exp(-.1) > .9


def test_arbitrary_split_is_numerical_agreement_not_a_claim_of_bitwise_ode_equivalence():
    network, spec, amounts = author()
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    whole = advance_stochastic_hybrid(network, spec, amounts, memory, 3)
    events = []
    for end in (.123, .51, 1.25, 2.2, 3.):
        part = advance_stochastic_hybrid(network, spec, amounts, memory, end - memory["time_s"])
        amounts, memory = part.amounts_mol, json.loads(json.dumps(part.continuation))
        events.extend(part.events)
    assert [e["reaction_id"] for e in events] == [e["reaction_id"] for e in whole.events]
    assert [e["time_s"] for e in events] == pytest.approx(
        [e["time_s"] for e in whole.events], abs=2e-7)
    assert amounts["A"] / CASES[0]["A0_mol"] == pytest.approx(
        whole.amounts_mol["A"] / CASES[0]["A0_mol"], abs=2e-7)
    assert memory["rng_state"] == whole.continuation["rng_state"]


@pytest.mark.parametrize("field,value", [
    ("discrete_species", ("E0",)), ("stochastic_reactions", ("basal",)),
    ("stochastic_reactions", ("activate", "deactivate", "basal", "catalysed")),
])
def test_wrong_partition_rejected(field, value):
    network, spec, _ = author()
    with pytest.raises(ValueError, match="partition"):
        compile_hybrid(network, replace(spec, **{field: value}))


@pytest.mark.parametrize("change", [
    {"rtol": True}, {"volume_L": 0}, {"max_events_per_advance": 0},
    {"discrete_species": ("E0", "E0")}, {"stochastic_reactions": "activate"},
])
def test_invalid_model_fields_rejected(change):
    _, spec, _ = author()
    with pytest.raises(ValueError):
        replace(spec, **change)


def test_strict_codec_and_original_network_has_no_ssa_declaration():
    network, spec, _ = author()
    assert network.stochastic is None
    assert StochasticHybridSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    with pytest.raises(ValueError, match="schema"):
        StochasticHybridSpec.from_dict({**spec.to_dict(), "schema": "other"})
    with pytest.raises(ValueError, match="fields"):
        StochasticHybridSpec.from_dict({**spec.to_dict(), "unknown": 1})


@pytest.mark.parametrize("field,value", [
    ("algorithm", "gillespie-direct-v1"), ("discrete_species", ["E0"]),
    ("signature", None), ("events_committed", -1), ("pending", None),
])
def test_modified_continuation_rejected(field, value):
    network, spec, _ = author()
    result = advance(duration=.1)
    memory = copy.deepcopy(result.continuation)
    memory[field] = value
    with pytest.raises(ValueError, match="hybrid"):
        advance_stochastic_hybrid(network, spec, result.amounts_mol, memory, 0)


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("channel_ids", ["deactivate", "activate"]),
    ("hazards", [-1., 0.]), ("targets", [0., 1.]),
])
def test_invalid_pending_risks_rejected(field, value):
    network, spec, _ = author()
    result = advance(duration=.1)
    memory = copy.deepcopy(result.continuation)
    memory["pending"][field] = value
    with pytest.raises(ValueError, match="pending"):
        advance_stochastic_hybrid(network, spec, result.amounts_mol, memory, 0)


def test_noninteger_inventory_not_rounded():
    network, spec, amounts = author()
    with pytest.raises(ValueError, match="noninteger"):
        initial_hybrid_continuation(network, spec, {**amounts, "E0": .5 / AVOGADRO}, seed=1)


def test_event_budget_has_failed_prefix_and_does_not_change_inputs():
    network, spec, amounts = author(max_events=1)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    original = copy.deepcopy(memory)
    with pytest.raises(EventBudgetError) as caught:
        advance_stochastic_hybrid(network, spec, amounts, memory, 3)
    assert len(caught.value.events) == 1 and caught.value.work["segments"] == 2
    assert memory == original


def test_original_world_wait_observe_and_json_continuation():
    instance = runtime()
    result = instance.execute({"operation": "wait", "duration_s": 1.})
    assert result.info["transaction_status"] == "committed"
    assert result.info["outcome_diagnostics"]["solver"] == HYBRID_ALGORITHM
    assert result.info["outcome_diagnostics"]["advance_strategy"] == "gillespie-direct"
    before = instance._state.to_dict()
    instance.observe()
    assert instance._state.to_dict() == before
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(instance.snapshot().to_dict())))
    for duration in (1.2, 1.3, 1.):
        action = {"operation": "wait", "duration_s": duration}
        assert instance.execute(action).info["transaction_status"] == "committed"
        assert restored.execute(action).info["transaction_status"] == "committed"
    assert instance._state.to_dict() == restored._state.to_dict()


def test_fresh_process_exact_snapshot_continuation(tmp_path):
    instance = runtime()
    assert instance.execute({"operation": "wait", "duration_s": 1.}).info[
        "transaction_status"] == "committed"
    source, output = tmp_path / "snapshot.json", tmp_path / "result.json"
    source.write_text(json.dumps(instance.snapshot().to_dict()), encoding="utf-8")
    action = {"operation": "wait", "duration_s": 2.}
    assert instance.execute(action).info["transaction_status"] == "committed"
    code = ("import json; from pathlib import Path; "
        "from chemworld.runtime import ChemWorldRuntime; "
        f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        f"r.execute({action!r}); Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))")
    subprocess.run([sys.executable, "-c", code], check=True)
    assert json.loads(output.read_text()) == json.loads(json.dumps(instance._state.to_dict()))


def test_original_shared_clock_advances_both_hybrid_vessels():
    instance = ChemWorldRuntime(composition=shared_world())
    instance.reset(seed=7)
    result = instance.execute({"operation": "wait", "duration_s": 1.})
    assert result.info["transaction_status"] == "committed"
    assert set(instance._state.vessel_elapsed_s.values()) == {1.}
    for local in (instance._state, *instance._state.inactive_vessels.values()):
        assert local.process.stochastic_continuation["time_s"] == 1.


def test_original_world_rolls_back_failed_inventory_rng_hazards_and_clock():
    instance = runtime(max_events=1)
    before = instance._state.to_dict()
    result = instance.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    after = instance._state.to_dict()
    assert after["species_amounts"] == before["species_amounts"]
    assert (after["process"]["stochastic_continuation"]
            == before["process"]["stochastic_continuation"])
    assert after["ledger"]["time_s"] == 0.


@pytest.mark.parametrize("change", ["integer_loss", "continuous_loss", "partition"])
def test_forged_candidate_cannot_hide_loss_in_the_mixed_inventory(change):
    instance = runtime()
    state = instance._state
    amounts = dict(state.species_amounts)
    process = state.process
    if change == "integer_loss":
        amounts["E0"] = 0.
    elif change == "continuous_loss":
        amounts["A"] *= .5
    else:
        continuation = copy.deepcopy(process.stochastic_continuation)
        continuation["discrete_species"] = ["E0"]
        process = replace(process, stochastic_continuation=continuation)
    candidate = state.replace(species_amounts=amounts, process=process)
    assert not instance.constitution.check_transition(state, candidate, TransitionBalance()).passed
    with pytest.raises(ValueError, match="hybrid"):
        instance.runtime.domain_services.reaction_thermal.advancer.validate_snapshot_state(candidate)


def test_partial_lifecycle_reset_is_rejected_but_fresh_session_reset_works():
    instance = runtime()
    assert instance.execute({"operation": "wait", "duration_s": 1.}).info[
        "transaction_status"] == "committed"
    before = instance._state.to_dict()
    with pytest.raises(ValueError, match="hybrid"):
        instance.reset(options={"scope": "material"})
    assert instance._state.to_dict() == before
    instance.reset(seed=7)
    assert instance._state.ledger.time_s == 0
    assert instance._state.process.stochastic_continuation["pending"] is None


@pytest.mark.parametrize("field,value", [
    ("discrete_species", ["E0"]), ("algorithm", "gillespie-direct-v1"),
    ("pending", None), ("signature", None),
])
def test_original_snapshot_restore_rejects_forged_continuation(field, value):
    instance = runtime()
    assert instance.execute({"operation": "wait", "duration_s": 1.}).info[
        "transaction_status"] == "committed"
    payload = json.loads(json.dumps(instance.snapshot().to_dict()))
    payload["world_state"]["process"]["stochastic_continuation"][field] = value
    with pytest.raises(ValueError, match="hybrid"):
        ChemWorldRuntime.from_snapshot(payload)


@pytest.mark.parametrize("boundary", ["dose", "sample"])
def test_material_and_consuming_sampler_are_rejected_at_authoring(boundary):
    network, spec, amounts = author()
    payload = stochastic_hybrid_world(network, spec, amounts).to_dict()
    if boundary == "dose":
        payload["components"].append({"kind": "material_input", "parameters": {}})
    else:
        payload["components"].append({"kind": "observation",
            "parameters": {"instruments": ["molecule_capture"]}})
        payload["task"].update(instruments=["molecule_capture"],
                               operations=["wait", "measure", "terminate"])
    with pytest.raises(ValueError, match="hybrid"):
        ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
