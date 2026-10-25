"""Saved numerical fragments, transaction failure and physical acceptance."""

import json
from copy import deepcopy
from dataclasses import replace
from math import sqrt

import numpy as np
import pytest

from chemworld.foundation.dimensions import DimensionVector
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem import (
    ReactionNoise,
    SDEParameter,
    StochasticSDESpec,
    advance_sde,
    draw_ri5,
    initial_sde_continuation,
    rebind_sde_continuation,
)
from chemworld.physchem.stochastic_kinetics import EventBudgetError
from chemworld.world.biochemical_worlds import biochemical_network


def setup(*, seed=0, **changes):
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = StochasticSDESpec(**{"volume_L": 1e-15, "step_s": 0.125, **changes})
    amounts = {"A": 100.5 / AVOGADRO, "B": 20.25 / AVOGADRO}
    continuation = initial_sde_continuation(network, spec, amounts, seed=seed)
    return network, spec, amounts, continuation


def test_exact_split_json_continuation_preserves_draws_and_full_future_state():
    network, spec, amounts, state = setup()
    original = deepcopy((amounts, state))
    full = advance_sde(network, spec, amounts, state, 0.5)
    first = advance_sde(network, spec, amounts, state, 0.0625)
    assert first.continuation["pending"] is not None
    restored = json.loads(json.dumps(first.continuation))
    second = advance_sde(network, spec, first.amounts_mol, restored, 0.4375)
    assert second.amounts_mol == full.amounts_mol
    assert second.continuation == full.continuation
    assert (amounts, state) == original
    assert full.continuation["completed_steps"] == 4
    assert sum(full.amounts_mol.values()) == pytest.approx(sum(amounts.values()), rel=1e-15, abs=0)


def test_every_dense_wait_slice_and_nonmesh_final_state_are_exact():
    network, spec, amounts, state = setup(seed=123)
    full = advance_sde(network, spec, amounts, state, 0.546875)
    current = amounts
    continuation = state
    for _ in range(35):
        result = advance_sde(network, spec, current, continuation, 0.015625)
        current, continuation = result.amounts_mol, result.continuation
    assert current == full.amounts_mol and continuation == full.continuation


def test_dense_read_uses_saved_random_packet_and_consumes_no_additional_rng():
    network, spec, amounts, state = setup()
    first = advance_sde(network, spec, amounts, state, 0.03125)
    second = advance_sde(network, spec, first.amounts_mol, first.continuation, 0.03125)
    assert first.continuation["rng_state"] == second.continuation["rng_state"]
    assert first.continuation["pending"] == second.continuation["pending"]
    assert second.diagnostics["packets_created"] == 0
    zero = advance_sde(network, spec, second.amounts_mol, second.continuation, 0)
    assert zero.amounts_mol == second.amounts_mol and zero.continuation == second.continuation


def test_zero_noise_limit_no_rng_draws_and_linear_reaction_polynomial():
    network, spec, amounts, state = setup(noises=())
    result = advance_sde(network, spec, amounts, state, 0.125)
    equilibrium = sum(amounts.values()) * 2 / 3
    z = -0.3 * 0.125
    stability = 1 + z + z * z / 2 + z * z * z / 6
    expected_A = equilibrium + (amounts["A"] - equilibrium) * stability
    assert result.amounts_mol["A"] == pytest.approx(expected_A, rel=1e-15, abs=0)
    assert result.continuation["rng_state"] == state["rng_state"]
    assert result.diagnostics["evaluations"] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"algorithm": "direct"},
        {"unknown": None},
        {"model_signature": "other"},
        {"inventory_signature": None},
        {"time_s": True},
        {"completed_steps": -1},
        {"proposals": False},
    ],
)
def test_malformed_or_wrong_model_continuation_rejected(changes):
    network, spec, amounts, state = setup()
    with pytest.raises(ValueError):
        advance_sde(network, spec, amounts, {**state, **changes}, 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "future"),
        ("end_time_s", float("nan")),
        ("left_time_s", 0.125),
        ("left_populations", [False, 2]),
        ("right_populations", [-1, 2]),
        ("right_populations", [1, 2]),
        ("proposal_index", 0),
        ("unknown", 1),
    ],
)
def test_saved_packet_is_checked_against_actual_stock_clock_and_ri5_draw(field, value):
    network, spec, amounts, state = setup()
    first = advance_sde(network, spec, amounts, state, 0.03125)
    state = deepcopy(first.continuation)
    state["pending"][field] = value
    with pytest.raises(ValueError):
        advance_sde(network, spec, first.amounts_mol, state, 0)


def test_packet_endpoint_cannot_be_changed_even_with_unchanged_current_dense_stock():
    network, spec, amounts, state = setup()
    first = advance_sde(network, spec, amounts, state, 0.0625)
    state = deepcopy(first.continuation)
    # At the midpoint offsetting the endpoints preserves the current stock.
    state["pending"]["left_populations"][0] -= 1
    state["pending"]["right_populations"][0] += 1
    with pytest.raises(ValueError):
        advance_sde(network, spec, first.amounts_mol, state, 0)


def test_material_change_requires_explicit_invalidation_and_preserves_consumed_stream():
    network, spec, amounts, state = setup()
    first = advance_sde(network, spec, amounts, state, 0.03125)
    changed = {**first.amounts_mol, "A": first.amounts_mol["A"] + 0.5 / AVOGADRO}
    with pytest.raises(ValueError, match="inventory"):
        advance_sde(network, spec, changed, first.continuation, 0)
    rebound = rebind_sde_continuation(network, spec, first.amounts_mol, changed, first.continuation)
    assert rebound["pending"] is None and rebound["invalidations"] == 1
    assert rebound["rng_state"] == first.continuation["rng_state"]
    assert rebound["time_s"] == first.continuation["time_s"]
    continued = advance_sde(network, spec, changed, rebound, 0.125)
    assert sum(continued.amounts_mol.values()) == pytest.approx(
        sum(changed.values()), rel=1e-15, abs=0
    )
    assert first.continuation["pending"] is not None
    same = rebind_sde_continuation(network, spec, changed, changed, rebound)
    assert same == rebound


@pytest.mark.parametrize(
    "changes,duration",
    [
        ({"max_steps_per_advance": 1}, 0.25),
        ({"max_evaluations_per_advance": 11}, 0.125),
    ],
)
def test_budget_failure_preserves_input_inventory_rng_and_numerical_memory(changes, duration):
    network, spec, amounts, state = setup(**changes)
    original = deepcopy((amounts, state))
    with pytest.raises(EventBudgetError):
        advance_sde(network, spec, amounts, state, duration)
    assert (amounts, state) == original


def boundary_setup(**changes):
    network, spec, _, _ = setup()
    network = replace(
        network,
        reactions=tuple(
            replace(r, rate_law=replace(r.rate_law, parameters={"k": 0.0}))
            for r in network.reactions
        ),
    )
    noise = ReactionNoise(
        "forward",
        '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>sigma</ci></math>',
        (SDEParameter("sigma", 2, DimensionVector(time=-0.5)),),
    )
    spec = replace(spec, step_s=1.0, noises=(noise,), **changes)
    # Choose the fixed lattice variable that moves A toward its boundary.
    seed = next(s for s in range(100) if draw_ri5(np.random.default_rng(s), 1).hat[0] < 0)
    amounts = {"A": 0.1 / AVOGADRO, "B": 10.0 / AVOGADRO}
    return network, spec, amounts, initial_sde_continuation(network, spec, amounts, seed=seed)


def test_physical_boundary_uses_computed_feasible_side_without_clipping():
    network, spec, amounts, state = boundary_setup()
    result = advance_sde(network, spec, amounts, state, 1e-5)
    packet = result.continuation["pending"]
    assert 0 < packet["end_time_s"] < spec.step_s
    assert result.diagnostics["backtracks"] > 0
    assert min(packet["right_populations"]) >= 0
    # The accepted A is the actual additive RI5 value for the SAME weak draw.
    expected_A = amounts["A"] * AVOGADRO + 2 * sqrt(packet["end_time_s"]) * packet["draw"]["hat"][0]
    assert packet["right_populations"][0] == pytest.approx(expected_A, abs=1e-16)
    assert result.amounts_mol["A"] > 0
    assert sum(result.amounts_mol.values()) == pytest.approx(
        sum(amounts.values()), rel=1e-15, abs=0
    )


def test_strict_negative_boundary_failure_does_not_consume_original_rng():
    network, spec, amounts, state = boundary_setup(boundary_policy="reject")
    original = deepcopy((amounts, state))
    with pytest.raises(ValueError, match="negative candidate"):
        advance_sde(network, spec, amounts, state, 1e-5)
    assert (amounts, state) == original


def test_unavailable_physical_prefix_exhausts_bounded_redraws_without_mutation():
    network, spec, amounts, state = boundary_setup(min_step_s=1.0, max_rejections_per_step=1)
    original = deepcopy((amounts, state))
    with pytest.raises(EventBudgetError, match="redraw budget"):
        advance_sde(network, spec, amounts, state, 1e-5)
    assert (amounts, state) == original


@pytest.mark.parametrize(
    "values", [{"A": True, "B": 0}, {"A": -1, "B": 0}, {"A": 1}, {"A": float("inf"), "B": 0}]
)
def test_invalid_true_inventory_cannot_enter_solver(values):
    network, spec, _, state = setup()
    with pytest.raises(ValueError):
        advance_sde(network, spec, values, state, 0)
