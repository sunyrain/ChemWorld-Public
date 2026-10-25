"""Actual reaction/batch boundaries in the original kernel, preserving its full future."""

import copy
from dataclasses import asdict

import pytest

from chemworld.physchem.stochastic_kinetics import (
    StochasticSpec,
    advance_event_channels,
    initial_continuation,
)
from reference.stochastic_tau51_cases import CASES, DURATION, make_network

METHODS = ("gillespie-direct-v1", "gibson-bruck-next-reaction-v1", "adaptive-explicit-tau-leap-v1")


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", (17, 37))
def test_returning_each_original_boundary_preserves_the_full_path_and_pending_future(
    case, method, seed
):
    network = make_network(case, algorithm=method)
    spec = StochasticSpec(**network.stochastic)
    channels = network.stochastic_channels
    counts = {key: case["initial"].get(key, 0) for key in network.species_ids}
    initial = initial_continuation(seed, algorithm=method)
    original = copy.deepcopy((counts, initial))
    full = advance_event_channels(channels, network.species_ids, spec, counts, initial, DURATION)
    explicit = advance_event_channels(
        channels, network.species_ids, spec, counts, initial, DURATION, stop_after_boundary=False
    )
    assert asdict(explicit) == asdict(full)
    assert (counts, initial) == original
    memory, events, batches = initial, [], []
    calls = 0
    while memory["time_s"] < DURATION:
        start = memory["time_s"]
        step = advance_event_channels(
            channels,
            network.species_ids,
            spec,
            counts,
            memory,
            DURATION - start,
            stop_after_boundary=True,
        )
        assert 0 < step.actual_duration_s <= DURATION - start
        assert len(step.events) + len(step.batches) <= 1
        assert step.continuation["time_s"] == start + step.actual_duration_s
        events.extend(step.events)
        batches.extend(step.batches)
        counts, memory = step.counts, step.continuation
        calls += 1
        assert calls < 100000
    assert counts == full.counts and memory == full.continuation
    assert tuple(events) == full.events and tuple(batches) == full.batches


@pytest.mark.parametrize("flag", (None, 0, 1, "true"))
def test_boundary_policy_cannot_be_selected_by_implicit_coercion(flag):
    network = make_network()
    with pytest.raises(ValueError, match="Boolean"):
        advance_event_channels(
            network.stochastic_channels,
            network.species_ids,
            StochasticSpec(**network.stochastic),
            CASES[1]["initial"],
            initial_continuation(17, algorithm=METHODS[-1]),
            1.0,
            stop_after_boundary=flag,
        )


@pytest.mark.parametrize("method", METHODS)
def test_zero_duration_does_not_draw_a_new_process_event(method):
    network = make_network(algorithm=method)
    memory = initial_continuation(17, algorithm=method)
    result = advance_event_channels(
        network.stochastic_channels,
        network.species_ids,
        StochasticSpec(**network.stochastic),
        CASES[1]["initial"],
        memory,
        0.0,
        stop_after_boundary=True,
    )
    assert result.actual_duration_s == 0 and result.events == result.batches == ()
    assert result.continuation["rng_state"] == memory["rng_state"]
    assert result.work["random_draws"] == 0
