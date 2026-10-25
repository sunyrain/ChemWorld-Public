"""Real material changes retain spent process randomness and unspent channel targets."""

from copy import deepcopy
from dataclasses import replace

import pytest

from chemworld.physchem.stochastic_adaptive_advance import (
    advance_adaptive_hybrid,
    initial_adaptive_continuation,
    rebind_adaptive_continuation,
)
from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM, AdaptiveHybridSpec
from chemworld.world.biochemical_worlds import biochemical_network


def start(*, a=11.5, b=100, lower=10, upper=20):
    network = replace(biochemical_network("reversible"), stochastic=None)
    spec = AdaptiveHybridSpec(1e-15, lower_limit=lower, upper_limit=upper, step_s=0.125)
    amounts = {"A": a * MOLECULAR_QUANTUM, "B": b * MOLECULAR_QUANTUM}
    memory = initial_adaptive_continuation(network, spec, amounts, seed=17)
    first = advance_adaptive_hybrid(network, spec, amounts, memory, 0.000001)
    return network, spec, first


def test_dose_recalculates_rate_with_same_residual_poisson_hazard():
    network, spec, first = start(lower=1000, upper=2000)
    old = first.continuation["pending"]
    current = first.continuation["time_s"]
    new_amounts = {**first.amounts_mol, "A": first.amounts_mol["A"] + 5.25 * MOLECULAR_QUANTUM}
    saved = deepcopy((first.amounts_mol, first.continuation))
    result = rebind_adaptive_continuation(
        network, spec, first.amounts_mol, new_amounts, first.continuation
    )
    pending = result.continuation["pending"]
    residual = old["propensities"][0] * (old["times_s"][0] - current)
    assert pending["times_s"][0] == pytest.approx(
        current + residual / pending["propensities"][0], rel=1e-12
    )
    assert result.continuation["rng_state"] == first.continuation["rng_state"]
    assert result.continuation["time_s"] == current
    assert (first.amounts_mol, first.continuation) == saved
    advance_adaptive_hybrid(network, spec, new_amounts, result.continuation, 0.25)


def test_pause_and_reactivation_never_replace_existing_random_target():
    network, spec, first = start()
    high = {**first.amounts_mol, "A": 30 * MOLECULAR_QUANTUM}
    paused = rebind_adaptive_continuation(
        network, spec, first.amounts_mol, high, first.continuation
    )
    assert paused.continuation["low_species"] == []
    assert paused.continuation["pending"]["times_s"] == [None, None]
    assert all(h > 0 for h in paused.continuation["pending"]["dormant_hazards"])
    low = {**first.amounts_mol, "A": 5 * MOLECULAR_QUANTUM}
    active = rebind_adaptive_continuation(network, spec, high, low, paused.continuation)
    assert active.continuation["low_species"] == ["A"]
    assert active.continuation["rng_state"] == first.continuation["rng_state"]
    assert active.work["random_draws"] == paused.work["random_draws"] == 0
    pending = active.continuation["pending"]
    for hazard, when, rate in zip(
        paused.continuation["pending"]["dormant_hazards"],
        pending["times_s"],
        pending["propensities"],
        strict=True,
    ):
        assert rate * (when - active.continuation["time_s"]) == pytest.approx(hazard, rel=1e-12)


def test_first_stochastic_activation_uses_persisted_process_rng():
    network, spec, first = start(a=100, b=100)
    assert first.continuation["draws_committed"] == 0
    changed = {"A": 5 * MOLECULAR_QUANTUM, "B": first.amounts_mol["B"]}
    result = rebind_adaptive_continuation(
        network, spec, first.amounts_mol, changed, first.continuation
    )
    assert result.work["random_draws"] == 2
    assert result.continuation["draws_committed"] == 2
    assert result.continuation["rng_state"] != first.continuation["rng_state"]
    assert result.continuation["time_s"] == first.continuation["time_s"]
    advance_adaptive_hybrid(network, spec, changed, result.continuation, 0.125)


def test_unchanged_and_invalid_material_do_not_change_any_input_or_draw():
    network, spec, first = start()
    saved = deepcopy((first.amounts_mol, first.continuation))
    unchanged = rebind_adaptive_continuation(
        network, spec, first.amounts_mol, first.amounts_mol, first.continuation
    )
    assert unchanged.continuation == first.continuation
    with pytest.raises(ValueError):
        rebind_adaptive_continuation(
            network, spec, first.amounts_mol, {**first.amounts_mol, "A": -1e-25}, first.continuation
        )
    assert (first.amounts_mol, first.continuation) == saved
