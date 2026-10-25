"""Algebraic contracts for physical units and the continuous reaction SDE view."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.foundation.dimensions import DimensionVector as D
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem import (
    ReactionFlux,
    ReactionNoise,
    ReactionSDE,
    SDEParameter,
    StochasticSDESpec,
    mathml_dimension,
)
from chemworld.physchem.mathml import MathExpression
from chemworld.physchem.reaction_rate_laws import mass_action_rate
from chemworld.world.biochemical_worlds import biochemical_network

T = D(time=1)


def math(body):
    return MathExpression('<math xmlns="http://www.w3.org/1998/Math/MathML">' + body + "</math>")


def network(kind="reversible"):
    return replace(biochemical_network(kind), stochastic=None)


def test_typed_root_and_fixed_fractional_power():
    dimensions = {"flux": T**-1, "p": D()}
    assert mathml_dimension(math("<apply><root/><ci>flux</ci></apply>"), dimensions) == T**-0.5
    expression = math("<apply><power/><ci>flux</ci><ci>p</ci></apply>")
    assert mathml_dimension(expression, dimensions, constants={"p": 0.5}) == T**-0.5
    with pytest.raises(ValueError, match="fixed exponent"):
        mathml_dimension(expression, dimensions)


@pytest.mark.parametrize(
    "body",
    [
        "<apply><plus/><ci>flux</ci><cn>1</cn></apply>",
        "<apply><exp/><ci>flux</ci></apply>",
        "<apply><power/><ci>flux</ci><ci>flux</ci></apply>",
        "<apply><root/><degree><cn>0</cn></degree><ci>flux</ci></apply>",
        "<piecewise><piece><ci>flux</ci><true/></piece><otherwise><cn>0</cn></otherwise></piecewise>",
        "<piecewise><piece><ci>flux</ci><cn>1</cn></piece></piecewise>",
        "<apply><times/><true/><ci>flux</ci></apply>",
        "<apply><lt/><ci>flux</ci><ci>flux</ci></apply>",
        "<piecewise><piece><ci>flux</ci><true/></piece><otherwise><ci>missing</ci></otherwise></piecewise>",
        "<infinity/>",
    ],
)
def test_incompatible_or_boolean_dimensions_are_rejected_before_execution(body):
    with pytest.raises(ValueError):
        mathml_dimension(math(body), {"flux": T**-1})


def test_piecewise_and_logic_check_all_branch_dimensions():
    expression = math(
        "<piecewise><piece><ci>flux</ci><apply><gt/><ci>flux</ci>"
        "<ci>threshold</ci></apply></piece><otherwise><ci>threshold</ci>"
        "</otherwise></piecewise>"
    )
    assert mathml_dimension(expression, {"flux": T**-1, "threshold": T**-1}) == T**-1


def test_source_unit_literals_and_unbound_functions_cannot_guess_dimensions():
    for expression in (
        math(
            '<cn xmlns:sbml="http://www.sbml.org/sbml/level3/version2/core" '
            'sbml:units="second">2</cn>'
        ),
        math("<apply><ci>user_function</ci><cn>2</cn></apply>"),
    ):
        with pytest.raises(ValueError, match="bound parameters"):
            mathml_dimension(expression, {})


def test_si_parameter_and_model_roundtrip_exact_closed_fields():
    parameter = SDEParameter.from_unit("tau", 2, "min")
    assert parameter.value == 120 and parameter.dimension == T
    noise = ReactionNoise(
        "forward",
        math("<apply><divide/><ci>n0</ci><apply><root/><ci>tau</ci></apply></apply>"),
        (parameter,),
    )
    flux = ReactionFlux(
        "forward", math("<apply><divide/><ci>n0</ci><ci>tau</ci></apply>"), (parameter,)
    )
    spec = StochasticSDESpec(1e-15, noises=(noise,), fluxes=(flux,))
    restored = StochasticSDESpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert ReactionSDE(network(), restored).signature == ReactionSDE(network(), spec).signature
    malformed = spec.to_dict()
    malformed["noises"][0]["parameters"][0]["dimension"]["unknown"] = 0
    with pytest.raises(ValueError, match="axes"):
        StochasticSDESpec.from_dict(malformed)
    for key, value in (
        ("schema", "future"),
        ("population_basis", "integer"),
        ("noise_unit", "mol/s"),
        ("unknown", 0),
    ):
        payload = {**spec.to_dict(), key: value}
        with pytest.raises(ValueError):
            StochasticSDESpec.from_dict(payload)


@pytest.mark.parametrize(
    "changes",
    [
        {"volume_L": True},
        {"volume_L": 0},
        {"step_s": float("nan")},
        {"step_s": 1e-15},
        {"max_steps_per_advance": True},
        {"max_evaluations_per_advance": 0},
        {"boundary_policy": "clip"},
        {"noises": (ReactionNoise("forward"), ReactionNoise("forward"))},
    ],
)
def test_invalid_physical_spec_rejected(changes):
    with pytest.raises(ValueError):
        StochasticSDESpec(**{"volume_L": 1e-15, **changes})


def test_first_order_particle_flux_and_stoichiometry():
    model = ReactionSDE(network(), StochasticSDESpec(1e-15))
    np.testing.assert_allclose(model.drift(0, [100.5, 20.25]), [-6, 6])
    diffusion = model.diffusion(0, [100.5, 20.25])
    np.testing.assert_allclose(
        diffusion, [[-np.sqrt(10.05), np.sqrt(4.05)], [np.sqrt(10.05), -np.sqrt(4.05)]]
    )
    # Real populations remain real; no integer-SSA rounding, including signed stages.
    np.testing.assert_allclose(model.drift(0, [-1.5, 0]), [0.15, -0.15])
    np.testing.assert_allclose(model.diffusion(0, [-1.5, 0])[:, 0], [-np.sqrt(0.15), np.sqrt(0.15)])


def test_bimolecular_catalytic_channel_uses_same_native_concentration_basis():
    model = ReactionSDE(network("enzyme"), StochasticSDESpec(1e-15))
    state = np.array([10.0, 100.0, 3.0, 5.0])
    binding = 1e6 * 10 * 100 / (AVOGADRO * 1e-15)
    np.testing.assert_allclose(
        model.drift(0, state), [-binding + 3.3, -binding + 3, binding - 3.3, 0.3]
    )
    # Every diffusion column preserves the enzyme and substrate pools independently.
    diffusion = model.diffusion(0, state)
    np.testing.assert_allclose(np.array([1, 0, 1, 0]) @ diffusion, 0, atol=1e-15)
    np.testing.assert_allclose(np.array([0, 1, 1, 1]) @ diffusion, 0, atol=1e-15)


def test_explicit_flux_and_noise_si_concentration_amount_time_units():
    # c0 * volume_m3 * avogadro / tau = n0/tau, all in coherent SI.
    flux = ReactionFlux(
        "forward",
        math(
            "<apply><divide/><apply><times/>"
            "<ci>c0</ci><ci>volume_m3</ci><ci>avogadro</ci></apply><ci>tau</ci></apply>"
        ),
        (SDEParameter.from_unit("tau", 2, "s"),),
    )
    noise = ReactionNoise(
        "forward",
        math(
            "<apply><times/><ci>scale</ci>"
            "<apply><root/><apply><abs/><ci>flux</ci></apply></apply></apply>"
        ),
        (SDEParameter("scale", 2),),
    )
    model = ReactionSDE(network(), StochasticSDESpec(1e-15, fluxes=(flux,), noises=(noise,)))
    np.testing.assert_allclose(model.drift(0.75, [100, 20]), [-46, 46])
    np.testing.assert_allclose(
        model.diffusion(0.75, [100, 20]), [[-2 * np.sqrt(50)], [2 * np.sqrt(50)]]
    )
    zero = ReactionSDE(network(), StochasticSDESpec(1e-15, noises=()))
    assert zero.diffusion(0, [100, 20]).shape == (2, 0)


@pytest.mark.parametrize(
    "declaration",
    [
        ReactionNoise("missing"),
        ReactionNoise("forward", math("<ci>n0</ci>")),
        ReactionNoise("forward", math("<ci>undeclared</ci>")),
        ReactionNoise("forward", math("<ci>flux</ci>"), (SDEParameter("n0", 1),)),
    ],
)
def test_noise_unknown_channel_symbol_units_or_shadowing_are_rejected(declaration):
    with pytest.raises(ValueError):
        ReactionSDE(network(), StochasticSDESpec(1e-15, noises=(declaration,)))


def test_flux_dependency_cycles_rejected_and_historical_avogadro_not_reinterpreted():
    flux = ReactionFlux("forward", math("<ci>f1</ci>"))
    with pytest.raises(ValueError, match="unknown typed"):
        ReactionSDE(network(), StochasticSDESpec(1e-15, fluxes=(flux,)))
    with pytest.raises(ValueError, match="typed native avogadro"):
        ReactionNoise(
            "forward",
            math('<csymbol definitionURL="http://www.sbml.org/sbml/symbols/avogadro">NA</csymbol>'),
        )


def test_implicit_non_elementary_flux_rejected_explicit_typed_override_supported():
    base = network()
    non_elementary = replace(
        base, reactions=(replace(base.reactions[0], forward_orders={"A": 0.5}), base.reactions[1])
    )
    with pytest.raises(ValueError, match="explicit typed fluxes"):
        ReactionSDE(non_elementary, StochasticSDESpec(1e-15))
    flux = ReactionFlux(
        "forward",
        math("<apply><divide/><ci>n0</ci><ci>tau</ci></apply>"),
        (SDEParameter("tau", 2, T),),
    )
    assert (
        ReactionSDE(non_elementary, StochasticSDESpec(1e-15, fluxes=(flux,))).noise_dimension == 2
    )
    with pytest.raises(ValueError, match="integer SSA"):
        ReactionSDE(biochemical_network(), StochasticSDESpec(1e-15))


@pytest.mark.parametrize("values", [[True, False], [1j, 0], ["1", "0"], [float("inf"), 0], [1]])
def test_callback_population_type_and_shape_are_strict(values):
    model = ReactionSDE(network(), StochasticSDESpec(1e-15))
    with pytest.raises(ValueError):
        model.drift(0, values)


def test_common_mass_action_retains_default_and_explicit_signed_polynomial_extension():
    assert mass_action_rate({"A": 1}, {"A": -2}, 3) == 0
    assert mass_action_rate({"A": 1}, {"A": -2}, 3, allow_signed_trial=True) == -6
    assert mass_action_rate({"A": 2}, {"A": -2}, 3, allow_signed_trial=True) == 12
    with pytest.raises(ValueError, match="integer powers"):
        mass_action_rate({"A": 0.5}, {"A": -2}, 3, allow_signed_trial=True)
