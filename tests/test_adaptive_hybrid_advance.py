"""Actual dynamic reaction repartition, persistent clocks and atomic pure candidates."""

import json
import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.runge_kutta4 import rk4_step
from chemworld.physchem.stochastic_adaptive_advance import (
    advance_adaptive_hybrid,
    initial_adaptive_continuation,
)
from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM, AdaptiveHybridSpec
from chemworld.physchem.stochastic_kinetics import (
    EventBudgetError,
    advance_stochastic,
    initial_continuation,
)
from chemworld.world.biochemical_worlds import biochemical_network


def setup(*, initial=None, seed=17, forward=0.1, reverse=0.2, **parameters):
    original = biochemical_network("reversible")
    reactions = tuple(
        replace(r, rate_law=replace(r.rate_law, parameters={"k": k}))
        for r, k in zip(original.reactions, (forward, reverse), strict=True)
    )
    network = replace(original, reactions=reactions, stochastic=None)
    spec = AdaptiveHybridSpec(**{"volume_L": 1e-15, "step_s": 0.125, **parameters})
    amounts = {
        key: value * MOLECULAR_QUANTUM for key, value in (initial or {"A": 100, "B": 20}).items()
    }
    memory = initial_adaptive_continuation(network, spec, amounts, seed=seed)
    return network, spec, amounts, memory


def test_rk4_matrix_polynomial_and_input_ownership():
    matrix = np.array([[-1.0, 2.0], [1.0, -2.0]])
    initial = np.array([13.25, 7.75])
    saved = initial.copy()
    h = 0.125
    expected = (
        sum(np.linalg.matrix_power(h * matrix, r) / math.factorial(r) for r in range(5)) @ initial
    )
    result = rk4_step(lambda _t, y: matrix @ y, 0, initial, h)
    np.testing.assert_allclose(result.values, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(initial, saved)
    assert result.evaluations == 4


def test_rk4_nonautonomous_cubic_and_zero_work():
    result = rk4_step(lambda t, y: np.array([t**3]), 2, [7], 0.5)
    assert result.values[0] == pytest.approx(7 + (2.5**4 - 2**4) / 4, abs=1e-12)
    assert rk4_step(lambda t, y: (_ for _ in ()).throw(RuntimeError()), 2, [7], 0).evaluations == 0


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_all_deterministic_limit_against_closed_solution(method):
    network, spec, amounts, memory = setup(method=method, lower_limit=0, upper_limit=0, reverse=0)
    saved = deepcopy((amounts, memory))
    result = advance_adaptive_hybrid(network, spec, amounts, memory, 1)
    z = -0.1 * spec.step_s
    expected_a = (
        100 * (1 + z + z * z / 2 + z**3 / 6 + z**4 / 24) ** 8
        if method == "RK4"
        else 100 * math.exp(-0.1)
    )
    np.testing.assert_allclose(
        [result.amounts_mol[k] / MOLECULAR_QUANTUM for k in ("A", "B")],
        [expected_a, 120 - expected_a],
        atol=1e-7,
        rtol=1e-8,
    )
    assert result.events == () and result.continuation["draws_committed"] == 0
    assert result.work["drift_evaluations"] > 0
    assert (amounts, memory) == saved


def test_all_stochastic_first_order_limit_matches_original_next_reaction_events():
    network, spec, amounts, memory = setup(lower_limit=1000, upper_limit=2000)
    actual = advance_adaptive_hybrid(network, spec, amounts, memory, 2)
    old_network = replace(
        network,
        stochastic={"volume_L": spec.volume_L, "algorithm": "gibson-bruck-next-reaction-v1"},
    )
    expected = advance_stochastic(
        old_network,
        {"A": 100, "B": 20},
        initial_continuation(17, algorithm="gibson-bruck-next-reaction-v1"),
        2,
    )
    assert actual.events == expected.events
    assert actual.amounts_mol == {k: n * MOLECULAR_QUANTUM for k, n in expected.counts.items()}
    assert actual.continuation["rng_state"] == expected.continuation["rng_state"]
    assert actual.work["drift_evaluations"] == 0


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_high_to_low_transition_uses_both_subnets_without_rounding(method):
    network, spec, amounts, memory = setup(
        initial={"A": 50.25, "B": 50.75},
        method=method,
        lower_limit=45,
        upper_limit=45,
        step_s=0.25,
        forward=1,
        reverse=0,
    )
    result = advance_adaptive_hybrid(network, spec, amounts, memory, 1)
    assert result.partition_changes[0] == {"time_s": 0.25, "before": (), "after": ("A",)}
    assert result.events and result.work["drift_evaluations"] > 0
    n = {k: v / MOLECULAR_QUANTUM for k, v in result.amounts_mol.items()}
    assert abs(sum(n.values()) - 101) < 1e-8
    assert not n["A"].is_integer()
    assert result.continuation["firings"]["forward"] == len(result.events)


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_low_to_high_transition_pauses_existing_poisson_clock(method):
    network, spec, amounts, memory = setup(
        initial={"A": 40, "B": 600},
        method=method,
        lower_limit=45,
        upper_limit=50,
        step_s=0.25,
        forward=0,
        reverse=1,
    )
    result = advance_adaptive_hybrid(network, spec, amounts, memory, 0.5)
    assert result.partition_changes[0] == {"time_s": 0.25, "before": ("A",), "after": ()}
    assert result.events and result.work["drift_evaluations"] > 0
    pending = result.continuation["pending"]
    assert pending["times_s"] == [None, None]
    assert pending["dormant_hazards"][1] > 0
    assert abs(sum(result.amounts_mol.values()) / MOLECULAR_QUANTUM - 640) < 1e-8


@pytest.mark.parametrize("method", ["RK4", "LSODA"])
def test_json_snapshot_suffix_exact_for_same_actions(method):
    network, spec, amounts, memory = setup(
        initial={"A": 50.25, "B": 50.75},
        method=method,
        lower_limit=45,
        upper_limit=45,
        step_s=0.25,
        forward=1,
        reverse=0,
    )
    first = advance_adaptive_hybrid(network, spec, amounts, memory, 0.375)
    saved = json.loads(json.dumps(first.continuation))
    a = advance_adaptive_hybrid(network, spec, first.amounts_mol, first.continuation, 0.625)
    b = advance_adaptive_hybrid(network, spec, first.amounts_mol, saved, 0.625)
    assert a == b


def test_stochastic_observation_splits_preserve_absolute_event_times_and_rng():
    network, spec, amounts, memory = setup(lower_limit=1000, upper_limit=2000)
    full = advance_adaptive_hybrid(network, spec, amounts, memory, 2)
    first = advance_adaptive_hybrid(network, spec, amounts, memory, 0.0625)
    second = advance_adaptive_hybrid(network, spec, first.amounts_mol, first.continuation, 1.9375)
    assert first.events + second.events == full.events
    assert second.amounts_mol == full.amounts_mol
    assert second.continuation["rng_state"] == full.continuation["rng_state"]


def test_reactivated_channel_resumes_its_poisson_target_without_replacement_draw():
    reactions = tuple(
        ReactionSpec.from_equation(
            reaction_id=name,
            equation=equation,
            rate_law=RateLawSpec(name, "mass_action", {"k": k}),
        )
        for name, equation, k in (("ab", "A => B", 5), ("bc", "B => C", 1), ("ca", "C => A", 0.1))
    )
    network = ReactionNetworkSpec(
        "transient-cycle", tuple(SpeciesSpec(k, "CH2O") for k in ("A", "B", "C")), reactions
    )
    spec = AdaptiveHybridSpec(1e-15, lower_limit=20, upper_limit=40, step_s=0.125)
    amounts = {k: n * MOLECULAR_QUANTUM for k, n in {"A": 100, "B": 100, "C": 0}.items()}
    memory = initial_adaptive_continuation(network, spec, amounts, seed=17)
    first = advance_adaptive_hybrid(network, spec, amounts, memory, 0.75)
    assert first.continuation["pending"]["times_s"][1] is None
    assert first.continuation["pending"]["dormant_hazards"][1] > 0
    final = advance_adaptive_hybrid(network, spec, first.amounts_mol, first.continuation, 4.25)
    assert any("B" not in c["before"] and "B" in c["after"] for c in final.partition_changes)
    assert final.continuation["pending"]["times_s"][1] is not None
    for result in (first, final):
        pending = result.continuation["pending"]
        remaining = sum(
            t is not None or h is not None
            for t, h in zip(pending["times_s"], pending["dormant_hazards"], strict=True)
        )
        # Every draw is either consumed by one firing or still held by one
        # active/paused channel. Switching must never silently replace it.
        assert (
            result.continuation["draws_committed"]
            == result.continuation["events_committed"] + remaining
        )


@pytest.mark.parametrize(
    "budget",
    [
        {"max_steps_per_advance": 1},
        {"max_events_per_advance": 1},
        {"max_evaluations_per_advance": 1},
    ],
)
def test_budgets_never_mutate_input_inventory_or_random_stream(budget):
    network, spec, amounts, memory = setup(**budget)
    saved = deepcopy((amounts, memory))
    with pytest.raises(EventBudgetError):
        advance_adaptive_hybrid(network, spec, amounts, memory, 2)
    assert (amounts, memory) == saved


def test_negative_deterministic_candidate_is_rejected_without_clipping():
    network, spec, amounts, memory = setup(
        initial={"A": 100, "B": 1},
        lower_limit=0,
        upper_limit=0,
        method="RK4",
        step_s=10,
        forward=1,
        reverse=0,
    )
    saved = deepcopy((amounts, memory))
    with pytest.raises(ArithmeticError, match="negative"):
        advance_adaptive_hybrid(network, spec, amounts, memory, 10)
    assert (amounts, memory) == saved


@pytest.mark.parametrize("change", ["erase", "rate", "time", "unknown"])
def test_pending_and_original_inventory_are_validated_before_continuing(change):
    network, spec, amounts, memory = setup()
    first = advance_adaptive_hybrid(network, spec, amounts, memory, 0.0625)
    saved = deepcopy(first.continuation)
    if change == "erase":
        saved["pending"] = None
    elif change == "rate":
        saved["pending"]["propensities"][0] += 1
    elif change == "time":
        saved["time_s"] += 0.1
    else:
        saved["unknown"] = 1
    with pytest.raises(ValueError):
        advance_adaptive_hybrid(network, spec, first.amounts_mol, saved, 0.0625)
