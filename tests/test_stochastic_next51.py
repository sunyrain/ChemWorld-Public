"""Exact next-reaction clocks on the existing discrete inventory and world."""

import copy
import json
import subprocess
import sys

import numpy as np
import pytest

from chemworld.physchem.stochastic_kinetics import (
    EventBudgetError,
    advance_stochastic,
    initial_continuation,
)
from chemworld.physchem.stochastic_next_reaction import channel_dependencies
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.stochastic_world import stochastic_world
from reference.stochastic_next51_cases import DIRECT, NEXT, make_network, shared_world


def initial(seed=17):
    return initial_continuation(seed, algorithm=NEXT)


@pytest.mark.parametrize("kind", ["irreversible", "reversible", "multiscale", "dimer"])
def test_exact_split_json_and_input_purity(kind):
    network, counts, continuation = make_network(kind), {"A": 30}, initial()
    before = copy.deepcopy(continuation)
    whole = advance_stochastic(network, counts, continuation, 4)
    events = []
    for end in (0.125, .5, 1., 2., 4.):
        part = advance_stochastic(network, counts, continuation, end - continuation["time_s"])
        counts = part.counts
        continuation = json.loads(json.dumps(part.continuation, allow_nan=False))
        events.extend(part.events)
    assert tuple(events) == whole.events
    assert counts == whole.counts and continuation == whole.continuation
    assert before == initial()
    assert all(type(n) is int and n >= 0 for n in counts.values())


def test_dependency_graph_only_tracks_changed_reactants():
    network = make_network("sparse", pairs=3)
    assert channel_dependencies(network.stochastic_channels) == (
        (0, 1), (0, 1), (2, 3), (2, 3), (4, 5), (4, 5),
    )


def test_method_explicit_and_legacy_initial_continuation_unchanged():
    assert initial_continuation(9) == {
        "algorithm": DIRECT, "time_s": 0., "rng_state": np.random.PCG64(9).state,
        "pending": None, "signature": None, "events_committed": 0,
    }
    with pytest.raises(ValueError, match="invalid SSA continuation"):
        advance_stochastic(make_network(), {"A": 20}, initial_continuation(9), 1)
    with pytest.raises(ValueError, match="unsupported stochastic algorithm"):
        initial_continuation(9, algorithm="automatic")


def test_zero_duration_and_zero_propensities_do_not_draw():
    before = initial()
    result = advance_stochastic(make_network(), {"A": 20}, before, 0)
    assert result.continuation["rng_state"] == before["rng_state"]
    assert result.continuation["pending"] is None and result.work["random_draws"] == 0
    result = advance_stochastic(make_network(), {}, before, 10)
    assert result.events == () and result.work["random_draws"] == 0
    assert result.continuation["rng_state"] == before["rng_state"]
    assert result.continuation["pending"]["times_s"] == [None, None]


def test_disabled_channel_preserves_remaining_internal_hazard():
    # Two channels consume the same lone A. After one fires, the other channel
    # must retain its unspent Poisson threshold while propensity is zero.
    from chemworld.physchem.reaction_network import (
        RateLawSpec,
        ReactionNetworkSpec,
        ReactionSpec,
        SpeciesSpec,
    )

    network = ReactionNetworkSpec(
        "disabled-clock", tuple(SpeciesSpec(k, "H2") for k in ("A", "B", "C")),
        tuple(ReactionSpec.from_equation(
            reaction_id=name, equation=eq, rate_law=RateLawSpec(name, "mass_action", {"k": k}),
        ) for name, eq, k in (("ab", "A => B", 1), ("ac", "A => C", 2),
                             ("ba", "B => A", .5), ("ca", "C => A", .5))),
        stochastic={"volume_L": 1e-15, "algorithm": NEXT},
    )
    prepared = advance_stochastic(network, {"A": 1}, initial(), 1e-10)
    times = prepared.continuation["pending"]["times_s"]
    first = min((t, i) for i, t in enumerate(times) if t is not None)
    stopped = advance_stochastic(
        network, prepared.counts, prepared.continuation, first[0] - 1e-10,
    )
    other = 1 - first[1]
    pending = stopped.continuation["pending"]
    assert pending["times_s"][other] is None
    expected = (1 if other == 0 else 2) * (times[other] - first[0])
    assert pending["dormant_hazards"][other] == expected
    revival = min(t for t in pending["times_s"] if t is not None)
    revived = advance_stochastic(
        network, stopped.counts, stopped.continuation, revival - first[0],
    )
    new_time = revived.continuation["pending"]["times_s"][other]
    assert new_time == revival + expected / (1 if other == 0 else 2)
    assert revived.work["random_draws"] == 1  # Only the previously fired A channel is new.


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("channel_ids", ["reverse", "forward"]),
    ("propensities", [0., 0.]), ("times_s", [None, None]),
    ("dormant_hazards", [-1, None]),
])
def test_invalid_pending_is_rejected(field, value):
    network = make_network()
    state = advance_stochastic(network, {"A": 20}, initial(), 1e-10)
    broken = copy.deepcopy(state.continuation)
    broken["pending"][field] = value
    with pytest.raises(ValueError, match="next-reaction"):
        advance_stochastic(network, state.counts, broken, 0)


def test_inventory_intervention_clears_clocks_without_reseeding():
    network = make_network()
    before = advance_stochastic(network, {"A": 10}, initial(), .2)
    counts = {**before.counts, "A": before.counts["A"] + 5}
    explicit = copy.deepcopy(before.continuation)
    explicit.update(pending=None, signature=None)
    automatic = advance_stochastic(network, counts, before.continuation, 1)
    restarted = advance_stochastic(network, counts, explicit, 1)
    assert automatic == restarted
    assert before.continuation["pending"] is not None


def test_budget_failure_preserves_inputs_and_actual_world():
    network = make_network(max_events=1)
    counts, continuation = {"A": 100}, initial()
    with pytest.raises(EventBudgetError):
        advance_stochastic(network, counts, continuation, 10)
    assert counts == {"A": 100} and continuation == initial()
    runtime = ChemWorldRuntime(composition=stochastic_world(network, counts, seed=17))
    runtime.reset(seed=8)
    before = runtime._state.to_dict()
    result = runtime.execute({"operation": "wait", "duration_s": 10})
    assert result.info["operation_outcome"] == "numerical_failure"
    after = runtime._state.to_dict()
    assert after["species_amounts"] == before["species_amounts"]
    assert (after["process"]["stochastic_continuation"]
            == before["process"]["stochastic_continuation"])
    assert after["ledger"]["time_s"] == 0


def test_original_world_snapshot_process_and_diagnostics(tmp_path):
    runtime = ChemWorldRuntime(composition=stochastic_world(make_network(), {"A": 100}, seed=17))
    runtime.reset(seed=8)
    result = runtime.execute({"operation": "wait", "duration_s": 1})
    assert result.info["transaction_status"] == "committed"
    diagnostics = result.info["outcome_diagnostics"]
    assert diagnostics["solver"] == NEXT
    assert diagnostics["advance_strategy"] == "gillespie-direct"  # Legacy dispatcher identity.
    assert diagnostics["stochastic_work"]["propensity_evaluations"] > 0
    assert "rng_state" not in json.dumps(result.info)
    before = runtime._state.to_dict()
    runtime.observe()
    assert runtime._state.to_dict() == before
    source, output = tmp_path / "snapshot.json", tmp_path / "result.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    action = {"operation": "wait", "duration_s": 1.5}
    runtime.execute(action)
    code = (
        "import json; from pathlib import Path; from chemworld.runtime import ChemWorldRuntime; "
        f"r=ChemWorldRuntime.from_snapshot(json.loads(Path({str(source)!r}).read_text())); "
        f"r.execute({action!r}); Path({str(output)!r}).write_text(json.dumps(r._state.to_dict()))"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
    assert json.loads(output.read_text()) == json.loads(json.dumps(runtime._state.to_dict()))


def test_original_shared_clock_owns_all_vessel_times_and_replay():
    runtime = ChemWorldRuntime(composition=shared_world(19))
    runtime.reset(seed=7)
    result = runtime.execute({"operation": "wait", "duration_s": .5})
    assert result.info["transaction_status"] == "committed"
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    action = {"operation": "wait", "duration_s": .75}
    runtime.execute(action)
    restored.execute(action)
    assert restored._state.to_dict() == runtime._state.to_dict()
    assert runtime._state.ledger.time_s == 1.25
    assert set(runtime._state.vessel_elapsed_s.values()) == {1.25}
    continuation = runtime._state.process.stochastic_continuation
    assert continuation["time_s"] == 1.25 and continuation["algorithm"] == NEXT


def test_sparse_work_counter_records_dependency_savings():
    results = [advance_stochastic(
        make_network("sparse", algorithm=algorithm), {f"A{i}": 20 for i in range(16)},
        initial_continuation(17, algorithm=algorithm), 1,
    ) for algorithm in (DIRECT, NEXT)]
    direct, nxt = [r.work["propensity_evaluations"] / len(r.events) for r in results]
    assert nxt < direct / 4
