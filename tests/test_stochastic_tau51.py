"""Tau batching, original-world consumers and backwards-compatible exact methods."""

import copy
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from chemworld.examples.stochastic_tau_leap import run_example
from chemworld.physchem.stochastic_kinetics import (
    TAU_ALGORITHM,
    EventBudgetError,
    StochasticSpec,
    advance_stochastic,
    initial_continuation,
)
from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy, _species_factors
from reference.stochastic_tau51_cases import CASES, POLICIES, make_network
from reference.stochastic_tau51_reference import Law


def initial(seed=71):
    return initial_continuation(seed, algorithm=TAU_ALGORITHM)


def test_policy_is_immutable_and_exact_codec_unchanged():
    for algorithm in ("gillespie-direct-v1", "gibson-bruck-next-reaction-v1"):
        assert StochasticSpec(1., algorithm=algorithm).to_dict() == {
            "volume_L": 1., "temperature_K": 298.15, "max_events_per_advance": 1000000,
            "algorithm": algorithm}
    source = POLICIES[1].to_dict()
    spec = StochasticSpec(1., algorithm=TAU_ALGORITHM, tau_policy=source)
    source["epsilon"] = .5
    assert spec.tau_policy.epsilon == .03
    with pytest.raises(FrozenInstanceError):
        spec.tau_policy.epsilon = .4
    assert StochasticSpec(**json.loads(json.dumps(spec.to_dict()))) == spec


@pytest.mark.parametrize("kwargs", ({"epsilon": 0}, {"epsilon": 1.1}, {"epsilon": True},
    {"max_tau_s": float("inf")}, {"critical_threshold": 0}, {"critical_threshold": True},
    {"ssa_threshold": 0}, {"max_steps_per_advance": 1.5}, {"max_rejections_per_leap": 0}))
def test_bad_policy(kwargs):
    with pytest.raises(ValueError):
        TauLeapPolicy(**kwargs)


def test_explicit_policy_and_algorithm_selection():
    with pytest.raises(ValueError):
        StochasticSpec(1., algorithm=TAU_ALGORITHM)
    with pytest.raises(ValueError):
        StochasticSpec(1., tau_policy={})
    with pytest.raises(ValueError):
        TauLeapPolicy.from_dict({"other": 1})


def test_actual_poisson_batches_and_original_event_meaning():
    network = make_network()
    result = advance_stochastic(network, {"A": 1000}, initial(), 1.)
    firings = sum(n for b in result.batches for _, n in b.reaction_counts)
    assert result.batches and firings > len(result.batches)
    assert result.counts["B"] == firings + len(result.events)
    assert result.continuation["events_committed"] == firings + len(result.events)
    assert result.work["leap_firings"] == firings
    assert result.work["rejection_unbiased"] is False
    assert all(n >= 0 and type(n) is int for b in result.batches for _, n in b.reaction_counts)


def test_non_grid_split_exactly_preserves_batches_events_and_rng():
    network, counts, state = make_network(), {"A": 1000}, initial()
    whole = advance_stochastic(network, counts, state, 3.)
    events, batches = [], []
    for end in (.017, .311, .973, 1.007, 2.319, 3.):
        result = advance_stochastic(network, counts, state, end-state["time_s"])
        counts, state = result.counts, json.loads(json.dumps(result.continuation))
        events.extend(result.events)
        batches.extend(result.batches)
    assert counts == whole.counts and state == whole.continuation
    assert tuple(events) == whole.events and tuple(batches) == whole.batches


def test_pending_packet_is_not_resampled_or_recounted():
    network = make_network()
    first = advance_stochastic(network, {"A": 1000}, initial(), .001)
    second = advance_stochastic(network, first.counts, first.continuation, .001)
    assert second.continuation["pending"] == first.continuation["pending"]
    assert second.continuation["rng_state"] == first.continuation["rng_state"]
    assert second.work["firings_planned"] == second.work["random_draws"] == 0
    assert second.work["firings_committed"] == 0
    assert second.work["pending_firings"] > 0
    assert second.work["output_hold_s"] == .002


@pytest.mark.parametrize("field,value", (
    ("schema", "bad"), ("algorithm", "gillespie-direct-v1"), ("policy", {}),
    ("mode", "invalid"), ("channel_ids", ["other"]), ("signature", "bad"),
    ("start_time_s", .5), ("end_time_s", 0.), ("end_time_s", float("inf")),
    ("firings", [-1]), ("firings", [1.5]), ("firings", [True]), ("firings", [1001]),
    ("firings", [0, 1]), ("extra", 1)))
def test_invalid_pending_rejected_even_at_zero_duration(field, value):
    network = make_network()
    result = advance_stochastic(network, {"A": 1000}, initial(), .001)
    result.continuation["pending"][field] = value
    with pytest.raises((ValueError, TypeError)):
        advance_stochastic(network, result.counts, result.continuation, 0)


def test_inventory_intervention_discards_packet_but_keeps_stream():
    network = make_network()
    before = advance_stochastic(network, {"A": 1000}, initial(), .001)
    counts = {**before.counts, "A": before.counts["A"]+100}
    explicit = copy.deepcopy(before.continuation)
    explicit.update(pending=None, signature=None)
    a = advance_stochastic(network, counts, before.continuation, 1.)
    b = advance_stochastic(network, counts, explicit, 1.)
    assert a == b


@pytest.mark.parametrize("field", ("critical_threshold", "max_steps_per_advance",
                                    "max_rejections_per_leap"))
def test_pending_policy_uses_strict_integer_codec(field):
    network = make_network()
    result = advance_stochastic(network, {"A": 1000}, initial(), .001)
    policy = result.continuation["pending"]["policy"]
    policy[field] = float(policy[field])
    with pytest.raises(ValueError, match="positive integer"):
        advance_stochastic(network, result.counts, result.continuation, 0)


@pytest.mark.parametrize("limit", ("steps", "firings"))
def test_budget_failure_does_not_mutate_input(limit):
    policy = replace(POLICIES[1], max_steps_per_advance=1) if limit == "steps" else POLICIES[1]
    network = make_network(policy=policy, max_events=1 if limit == "firings" else 1000000)
    counts, state = {"A": 1000}, initial()
    before = copy.deepcopy(state)
    with pytest.raises(EventBudgetError):
        advance_stochastic(network, counts, state, 10)
    assert state == before and counts == {"A": 1000}


def test_zero_duration_and_absorbing_inventory_do_not_draw():
    for counts, duration in (({"A": 1000}, 0), ({"B": 1000}, 10)):
        result = advance_stochastic(make_network(), counts, initial(), duration)
        assert result.continuation["rng_state"] == initial()["rng_state"]
        assert result.work["random_draws"] == 0


@pytest.mark.parametrize("case", CASES)
def test_finite_network_support_and_independent_reference(case):
    result = advance_stochastic(make_network(case), case["initial"], initial(), .973)
    law = Law(case, .973)
    row = np.asarray([[result.counts[k] for k in case["species"]]])
    assert law.valid(row) and 0 <= law.cdf_distance(row) <= 1


def test_repeated_reactant_factors_and_source_coefficients():
    from chemworld.physchem.stochastic_kinetics import EventChannel

    channels = (EventChannel("mixed", (), (("A", 2), ("B", 1)), 1.),)
    factors = _species_factors(channels, {"A": 10, "B": 5})
    assert factors["A"] == pytest.approx(1.5*(2+1/9))
    assert factors["B"] == 3
    assert _species_factors(channels, {"A": 1, "B": 5})["A"] == float("inf")
    for case in CASES[-2:]:
        network = make_network(case)
        assert network.stochastic_channels[0].constant == pytest.approx(case["rows"][0][2])


def test_original_world_example():
    result = run_example()
    assert result["exact_resume"]
    assert result["first"]["solver"] == TAU_ALGORITHM
    assert result["first"]["stochastic_work"]["accepted_leaps"] > 0


@pytest.mark.parametrize("name", ("world", "interventions", "boundaries", "rollback"))
def test_original_consumer_contracts(name, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent/"reference"))
    import run_stochastic_tau51

    checks, evidence = getattr(run_stochastic_tau51, name)()
    assert all(checks.values()), (checks, evidence)


def test_independent_multinomial_fourth_moments():
    from scipy.stats import multinomial

    case = {**CASES[3], "initial": {"A": 5}}
    law = Law(case, .973)
    support = np.asarray([(a, b, 5-a-b) for a in range(6) for b in range(6-a)])
    probability = multinomial.pmf(support, 5, law.p)
    centered = support-law.mean
    products = centered[:, :, None]*centered[:, None, :]
    direct = np.sum(probability[:, None, None]*(products-law.covariance)**2, axis=0)
    np.testing.assert_allclose(law.product_variance, direct, rtol=1e-12, atol=1e-12)
