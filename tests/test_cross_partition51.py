"""Real mixed-partition transfers, analytic PDMP and original consumers."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest

from chemworld.foundation.discrete_inventory import AVOGADRO, CROSS_PARTITION_ALGORITHM
from chemworld.physchem.stochastic_cross_partition import (
    CrossPartitionHybridSpec,
    advance_cross_partition,
    initial_cross_continuation,
)
from chemworld.physchem.stochastic_hybrid import (
    StochasticHybridSpec,
    advance_stochastic_hybrid,
    initial_hybrid_continuation,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.stochastic_hybrid_world import stochastic_hybrid_world
from reference.cross_partition51_cases import CASES, author, quantum_author, shared_world
from reference.cross_partition51_reference import simulate


def native(case=None, seed=17, duration=3., max_events=10000):
    network, spec, amounts = author(case, max_events=max_events)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=seed)
    return advance_stochastic_hybrid(network, spec, amounts, memory, duration)


def runtime(max_events=10000):
    network, spec, amounts = author(max_events=max_events)
    instance = ChemWorldRuntime(composition=stochastic_hybrid_world(
        network, spec, amounts, seed=17))
    instance.reset(seed=3)
    return instance


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("seed", [0, 17, 41, 103])
def test_complete_exchange_agrees_with_closed_form_and_integrated_hazards(case, seed):
    actual, expected = native(case, seed=seed), simulate(case, seed)
    assert [e["reaction_id"] for e in actual.events] == [
        e["reaction_id"] for e in expected["events"]]
    assert [e["time_s"] for e in actual.events] == pytest.approx(
        [e["time_s"] for e in expected["events"]], abs=2e-7)
    for key in ("D", "A", "B"):
        assert actual.amounts_mol[key] * AVOGADRO == pytest.approx(expected[key], abs=2e-7)
    assert actual.continuation["rng_state"] == expected["rng_state"]
    assert actual.continuation["pending"]["targets"] == expected["targets"]
    assert actual.continuation["pending"]["hazards"] == pytest.approx(expected["hazards"], abs=2e-7)
    assert actual.work["random_draws"] == 2 + len(actual.events)


def test_original_world_and_complete_json_canary():
    instance = runtime()
    result = instance.execute({"operation": "wait", "duration_s": 1.})
    assert result.info["transaction_status"] == "committed", result.info
    assert result.info["outcome_diagnostics"]["solver"] == CROSS_PARTITION_ALGORITHM
    restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(instance.snapshot().to_dict())))
    for runtime_ in (instance, restored):
        result = runtime_.execute({"operation": "wait", "duration_s": 2.})
        assert result.info["transaction_status"] == "committed", result.info
    assert instance._state.to_dict() == restored._state.to_dict()


def test_shared_clock_wait_on_other_vessel_canary():
    instance = ChemWorldRuntime(composition=shared_world())
    instance.reset(seed=3)
    selected = instance.execute({"operation": "select_vessel", "vessel": "second"})
    assert selected.info["transaction_status"] == "committed", selected.info
    result = instance.execute({"operation": "wait", "duration_s": 1.})
    assert result.info["transaction_status"] == "committed", result.info
    assert set(instance._state.vessel_elapsed_s.values()) == {1.}
    for local in (instance._state, *instance._state.inactive_vessels.values()):
        memory = local.process.stochastic_continuation
        assert memory["time_s"] == 1.
        assert sum(memory["partition_transfer"]["firings"].values()) > 0


def test_fresh_process_json_continuation_canary(tmp_path):
    instance = runtime()
    assert instance.execute({"operation": "wait", "duration_s": 1.}).info[
        "transaction_status"] == "committed"
    source, output = tmp_path / "snapshot.json", tmp_path / "state.json"
    source.write_text(json.dumps(instance.snapshot().to_dict()), encoding="utf-8")
    code = ("import json; from pathlib import Path; "
        "from chemworld.runtime import ChemWorldRuntime; "
        f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        "result=r.execute({'operation':'wait','duration_s':2.}); "
        "assert result.info['transaction_status']=='committed',result.info; "
        f"Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))")
    subprocess.run([sys.executable, "-c", code], check=True)
    assert instance.execute({"operation": "wait", "duration_s": 2.}).info[
        "transaction_status"] == "committed"
    assert json.loads(output.read_text()) == json.loads(json.dumps(instance._state.to_dict()))


def test_new_codec_does_not_relax_old_partition():
    network, spec, amounts = author()
    assert CrossPartitionHybridSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    with pytest.raises(ValueError, match="schema"):
        StochasticHybridSpec.from_dict(spec.to_dict())
    old = StochasticHybridSpec(spec.discrete_species, spec.stochastic_reactions, spec.volume_L)
    with pytest.raises(ValueError, match="partition"):
        initial_hybrid_continuation(network, old, amounts, seed=17)
    for call in (lambda: initial_cross_continuation(network, old, amounts, seed=17),
                 lambda: advance_cross_partition(network, old, amounts, {}, 0)):
        with pytest.raises(ValueError, match="explicit model"):
            call()


def test_event_budget_rolls_back_inventory_rng_receipt_and_clock():
    instance = runtime(max_events=1)
    before = copy.deepcopy(instance._state.to_dict())
    result = instance.execute({"operation": "wait", "duration_s": 3.})
    assert result.info["operation_outcome"] == "numerical_failure"
    after = instance._state.to_dict()
    for key in ("species_amounts", "phases"):
        assert before[key] == after[key]
    assert (before["process"]["stochastic_continuation"]
            == after["process"]["stochastic_continuation"])
    assert after["ledger"]["time_s"] == 0.


def test_availability_roots_keep_original_poisson_targets_and_real_fractional_stock():
    depleted = native(CASES[2])
    activated = native(CASES[1])
    assert depleted.work["availability_crossings"] == 1
    assert depleted.events == ()
    assert 0 < depleted.amounts_mol["A"] * AVOGADRO < 1
    assert depleted.continuation["pending"]["hazards"][1] == pytest.approx(
        .05 * (2.25 - 1) / 2, abs=2e-9)
    assert activated.work["availability_crossings"] > 0
    assert activated.events[0]["reaction_id"] == "capture"
    for result in (depleted, activated):
        assert result.work["random_draws"] == 2 + len(result.events)


def test_arbitrary_split_agrees_numerically_without_claiming_bitwise_ode_equivalence():
    network, spec, amounts = author()
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    whole = advance_stochastic_hybrid(network, spec, amounts, memory, 3.)
    events = []
    for end in (.123, .51, 1.25, 2.2, 3.):
        part = advance_stochastic_hybrid(network, spec, amounts, memory, end - memory["time_s"])
        amounts, memory = part.amounts_mol, json.loads(json.dumps(part.continuation))
        events.extend(part.events)
    assert [e["reaction_id"] for e in events] == [e["reaction_id"] for e in whole.events]
    assert [e["time_s"] for e in events] == pytest.approx(
        [e["time_s"] for e in whole.events], abs=2e-7)
    for key in amounts:
        assert amounts[key] * AVOGADRO == pytest.approx(whole.amounts_mol[key] * AVOGADRO, abs=2e-7)
    assert memory["rng_state"] == whole.continuation["rng_state"]
    assert memory["partition_transfer"] == whole.continuation["partition_transfer"]


@pytest.mark.parametrize("field,value", [
    ("algorithm", "integrated-hazard-pdmp-v1"), ("signature", None),
    ("events_committed", 0), ("partition_transfer", None), ("pending", None),
    ("discrete_species", ["A"]),
])
def test_snapshot_rejects_changed_continuation(field, value):
    instance = runtime()
    instance.execute({"operation": "wait", "duration_s": 1.})
    payload = json.loads(json.dumps(instance.snapshot().to_dict()))
    payload["world_state"]["process"]["stochastic_continuation"][field] = value
    with pytest.raises((ValueError, TypeError)):
        ChemWorldRuntime.from_snapshot(payload)


@pytest.mark.parametrize("kind", ["wrong_stoichiometry", "one_molecule_loss", "invented_events"])
def test_original_runtime_checks_source_bound_receipt_history(kind):
    from chemworld.physchem.stochastic_hybrid import _signature

    instance = runtime()
    instance.execute({"operation": "wait", "duration_s": 1.})
    state = instance._state
    amounts = dict(state.species_amounts)
    memory = copy.deepcopy(state.process.stochastic_continuation)
    if kind == "wrong_stoichiometry":
        memory["partition_transfer"]["channel_stoichiometry"]["release"]["A"] = 2
    elif kind == "one_molecule_loss":
        amounts["B"] -= 1 / AVOGADRO
        network, spec, _ = author()
        memory["signature"] = _signature(network, spec, amounts)
    else:
        memory["partition_transfer"]["firings"]["release"] += 1
        memory["events_committed"] += 1
    candidate = state.replace(species_amounts=amounts,
        process=replace(state.process, stochastic_continuation=memory))
    with pytest.raises(ValueError):
        instance.runtime.domain_services.reaction_thermal.advancer.validate_snapshot_state(candidate)


def test_zero_duration_is_pure_and_unresolved_continuous_quantum_fails_at_authoring():
    result = native(duration=0.)
    assert result.continuation["pending"] is None
    assert result.work["random_draws"] == 0
    network, spec, amounts = author()
    with pytest.raises(ValueError, match="numerically resolved"):
        initial_hybrid_continuation(network, spec, {**amounts, "A": 1.}, seed=17)
    with pytest.raises(ValueError, match="noninteger"):
        initial_hybrid_continuation(network, spec, {**amounts, "D": .5 / AVOGADRO}, seed=17)


@pytest.mark.parametrize("quantum", [2, 3])
def test_multiple_quantum_consumption_and_discrete_falling_factorials(quantum):
    import numpy as np

    network, spec, amounts = quantum_author(quantum)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    result = advance_stochastic_hybrid(network, spec, amounts, memory, 3.)
    generator = np.random.Generator(np.random.PCG64(17))
    targets = [-math.log1p(-float(generator.random())) for _ in range(2)]
    hazards = [0., 0.]
    time, channel, expected = 0., 0, []
    while time < 3.:
        rate = .4 * math.factorial(quantum) if channel == 0 else .3 * quantum**quantum
        dt = (targets[channel] - hazards[channel]) / rate
        if time + dt > 3.:
            break
        time += dt
        hazards[channel] = targets[channel]
        targets[channel] += -math.log1p(-float(generator.random()))
        expected.append(time)
        channel = 1 - channel
    assert [e["time_s"] for e in result.events] == pytest.approx(expected, abs=2e-7)
    assert result.continuation["pending"]["targets"] == targets
    assert result.work["random_draws"] == 2 + len(expected)
    network, spec, amounts = quantum_author(quantum, fractional=True)
    memory = initial_hybrid_continuation(network, spec, amounts, seed=17)
    result = advance_stochastic_hybrid(network, spec, amounts, memory, 3.)
    assert result.events == ()
    assert result.continuation["pending"]["hazards"] == [0., 0.]
    assert result.amounts_mol == amounts
