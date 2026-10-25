"""Physical classification, independent extent balances and old SSA rate limits."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.stochastic_adaptive_hybrid import (
    MOLECULAR_QUANTUM,
    AdaptiveHybridModel,
    AdaptiveHybridSpec,
)
from chemworld.physchem.stochastic_kinetics import StochasticSpec, compile_channels
from chemworld.world.biochemical_worlds import biochemical_network


def network(kind="reversible"):
    return replace(biochemical_network(kind), stochastic=None)


def amounts(**populations):
    return {key: value * MOLECULAR_QUANTUM for key, value in populations.items()}


def test_closed_spec_roundtrip_and_stable_binding():
    spec = AdaptiveHybridSpec(1e-15, lower_limit=10, upper_limit=20, partition_steps=4)
    restored = AdaptiveHybridSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert spec == restored and spec.partition_interval_s == 0.004
    assert (
        AdaptiveHybridModel(network(), spec).signature
        == AdaptiveHybridModel(network(), restored).signature
    )
    for payload in ({**spec.to_dict(), "extra": 1}, {**spec.to_dict(), "schema": "unknown"}):
        with pytest.raises(ValueError):
            AdaptiveHybridSpec.from_dict(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"volume_L": True},
        {"volume_L": 0},
        {"volume_L": float("inf")},
        {"lower_limit": -1},
        {"lower_limit": 20, "upper_limit": 10},
        {"upper_limit": float("nan")},
        {"upper_limit": 2**41},
        {"step_s": 0},
        {"method": "RK45"},
        {"partition_steps": True},
        {"partition_steps": 0},
        {"atol_particles": 0},
        {"max_steps_per_advance": 0},
        {"max_events_per_advance": False},
        {"max_evaluations_per_advance": 0},
    ],
)
def test_invalid_model_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        AdaptiveHybridSpec(**{"volume_L": 1e-15, **change})


def test_hysteresis_strict_boundaries_preserve_real_stock():
    model = AdaptiveHybridModel(
        network(), AdaptiveHybridSpec(1e-15, lower_limit=10, upper_limit=20)
    )
    initial = amounts(A=11.75, B=88.25)
    saved = deepcopy(initial)
    partition = model.partition(initial)
    assert partition.low_species == ("A",)
    assert set(partition.stochastic_reactions) == {"forward", "reverse"}
    assert model.partition(amounts(A=20, B=80), previous_low=("A",)).low_species == ("A",)
    assert model.partition(amounts(A=20.5, B=79.5), previous_low=("A",)).low_species == ()
    assert model.partition(amounts(A=10, B=90), previous_low=()).low_species == ()
    assert model.partition(amounts(A=9.5, B=90.5), previous_low=()).low_species == ("A",)
    assert initial == saved
    assert model.populations(initial) == {"A": 11.75, "B": 88.25}


def test_low_product_switches_its_reaction_even_with_high_reactant():
    model = AdaptiveHybridModel(
        network(), AdaptiveHybridSpec(1e-15, lower_limit=10, upper_limit=20)
    )
    partition = model.partition(amounts(A=100, B=0))
    assert partition.low_species == ("B",)
    assert partition.deterministic_reactions == ()


def test_low_zero_net_catalyst_participates_in_partition():
    original = network()
    forward = replace(original.reactions[0], forward_orders={"A": 1, "Cat": 1})
    catalyzed = replace(
        original,
        species=(*original.species, SpeciesSpec("Cat", "N2")),
        reactions=(forward, original.reactions[1]),
    )
    model = AdaptiveHybridModel(
        catalyzed, AdaptiveHybridSpec(1e-15, lower_limit=10, upper_limit=20)
    )
    partition = model.partition(amounts(A=100, B=100, Cat=1.25))
    assert partition.stochastic_reactions == ("forward",)
    assert partition.deterministic_reactions == ("reverse",)
    assert model.populations(amounts(A=100, B=100, Cat=1.25))["Cat"] == 1.25


def test_exact_integer_preimage_never_changes_material_bits():
    model = AdaptiveHybridModel(network(), AdaptiveHybridSpec(1e-15))
    for n in range(1, 1025):
        original = amounts(A=n, B=0)
        saved = deepcopy(original)
        assert model.populations(original)["A"] == n
        assert original == saved


def test_sole_inventory_rejects_negative_unknown_and_unresolved_amounts():
    model = AdaptiveHybridModel(network(), AdaptiveHybridSpec(1e-15))
    for value in (
        {"A": -1e-30, "B": 0},
        {"A": 1e-10, "B": 0},
        {"A": True, "B": 0},
        {"A": 0},
        {"A": 0, "B": 0, "C": 0},
    ):
        with pytest.raises(ValueError):
            model.populations(value)
    for low in (("missing",), ("A", "A"), "A"):
        with pytest.raises(ValueError):
            model.partition(amounts(A=1, B=1), previous_low=low)


def test_deterministic_and_stochastic_extents_are_disjoint():
    model = AdaptiveHybridModel(network(), AdaptiveHybridSpec(1e-15))
    np.testing.assert_allclose(model.deterministic_drift([100, 20], ()), [-6, 6])
    np.testing.assert_allclose(model.deterministic_drift([100, 20], ("forward",)), [4, -4])
    np.testing.assert_allclose(model.deterministic_drift([100, 20], ("forward", "reverse")), [0, 0])
    assert model.propensities([100, 20], ("forward",)) == (10, 0)
    # Trial-stage extension is signed; accepted physical input remains nonnegative.
    np.testing.assert_allclose(model.deterministic_drift([-1, 0], ()), [0.1, -0.1])
    with pytest.raises(ValueError):
        model.propensities([-1, 0], ("forward",))


def third_order_network():
    return ReactionNetworkSpec(
        "adaptive-third-order",
        tuple(SpeciesSpec(k, f) for k, f in (("A", "H2"), ("B", "O2"), ("C", "H4O2"))),
        (
            ReactionSpec.from_equation(
                reaction_id="association",
                equation="2 A + B => C",
                rate_law=RateLawSpec("association", "mass_action", {"k": 1e13}),
            ),
        ),
    )


@pytest.mark.parametrize(
    "mechanism,counts",
    [
        (network(), {"A": 100, "B": 20}),
        (network("enzyme"), {"E": 8, "S": 16, "ES": 4, "P": 2}),
        (third_order_network(), {"A": 8, "B": 16, "C": 4}),
    ],
)
def test_integer_jump_propensities_match_original_ssa(mechanism, counts):
    spec = AdaptiveHybridSpec(1e-15)
    model = AdaptiveHybridModel(mechanism, spec)
    actual = model.propensities([counts[k] for k in mechanism.species_ids], mechanism.reaction_ids)
    expected = [c.propensity(counts) for c in compile_channels(mechanism, StochasticSpec(1e-15))]
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_third_order_drift_and_whole_quantum_availability():
    model = AdaptiveHybridModel(third_order_network(), AdaptiveHybridSpec(1e-15))
    flux = 1e13 * 80**2 * 90 / (AVOGADRO * 1e-15) ** 2
    drift = model.deterministic_drift([80, 90, 10], ())
    np.testing.assert_allclose(drift, [-2 * flux, -flux, flux], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.array([[1, 0, 2], [0, 1, 1]]) @ drift, 0, atol=1e-12)
    assert model.propensities([1.75, 90, 10], ("association",)) == (0,)


def test_competing_stochastic_model_and_nonelementary_law_rejected():
    spec = AdaptiveHybridSpec(1e-15)
    with pytest.raises(ValueError, match="compete"):
        AdaptiveHybridModel(biochemical_network("reversible"), spec)
    original = network()
    invalid = replace(original.reactions[0], forward_orders={"A": 0.5})
    with pytest.raises(ValueError, match="elementary"):
        AdaptiveHybridModel(replace(original, reactions=(invalid,)), spec)
