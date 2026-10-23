import json

import pytest

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.eos import (
    CubicEOSSpec,
    EOSComponentSpec,
    cubic_mixture_parameters,
    cubic_pure_parameters,
    evaluate_cubic_eos,
)
from chemworld.physchem.ppr78 import PPR78Spec, ppr78_interaction_response


def declaration():
    groups = {"a": {"1": 2}, "b": {"2": 1}, "c": {"1": 1, "2": 1}}
    rows = {"1|1": (0, 0), "1|2": (74.81, 165.7),
            "2|1": (74.81, 165.7), "2|2": (0, 0)}
    return PPR78Spec("original", groups, rows,
                     DataSourceProvenance("test", 0, "explicit PPR78 source rows"))


def components():
    return (EOSComponentSpec("a", 500, 4e6, .2), EOSComponentSpec("b", 650, 3e6, .6),
            EOSComponentSpec("c", 580, 3.5e6, .4))


def test_predictive_three_component_original_eos_and_temperature_response():
    spec = CubicEOSSpec("predictive", "peng_robinson", components(),
                        pure_parameterization="pr78", predictive_interaction=declaration())
    composition = {"a": .2, "b": .3, "c": .5}
    t, h = 700., .001
    mixture = cubic_mixture_parameters(spec, composition, temperature_K=t)
    high = cubic_mixture_parameters(spec, composition, temperature_K=t+h)
    low = cubic_mixture_parameters(spec, composition, temperature_K=t-h)
    assert mixture.da_mix_dT == pytest.approx((high.a_mix-low.a_mix)/(2*h), rel=1e-8)
    state = evaluate_cubic_eos(spec, composition, temperature_K=t, pressure_Pa=1e5)
    assert state.molar_residual_gibbs_J_mol == pytest.approx(
        state.molar_residual_enthalpy_J_mol-t*state.molar_residual_entropy_J_mol_K)
    restored = CubicEOSSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored.to_dict() == spec.to_dict()
    assert evaluate_cubic_eos(restored, composition, temperature_K=t, pressure_Pa=1e5) == state


def test_prediction_requires_consistent_pr78_and_preserves_default():
    original = cubic_pure_parameters(components()[1], model="peng_robinson", temperature_K=700)
    pr78 = cubic_pure_parameters(components()[1], model="peng_robinson",
                                temperature_K=700, parameterization="pr78")
    assert original.kappa != pr78.kappa
    assert original.b != pr78.b
    with pytest.raises(ValueError, match="explicit pr78"):
        CubicEOSSpec("bad", "peng_robinson", components(), predictive_interaction=declaration())
    with pytest.raises(ValueError, match="requires Peng"):
        CubicEOSSpec("bad", "srk", components(), pure_parameterization="pr78")
    with pytest.raises(ValueError, match="cannot be combined"):
        CubicEOSSpec("bad", "peng_robinson", components(), {"a|b": 0.},
                     pure_parameterization="pr78", predictive_interaction=declaration())


def test_missing_asymmetric_and_invalid_groups_reject():
    payload = declaration().to_dict()
    payload["interactions_MPa"].pop("1|2")
    with pytest.raises(ValueError):
        PPR78Spec.from_dict(payload)
    payload = declaration().to_dict()
    payload["interactions_MPa"]["1|2"] = [3, 4]
    with pytest.raises(ValueError, match="symmetric"):
        PPR78Spec.from_dict(payload)
    payload = declaration().to_dict()
    payload["groups"]["a"]["1"] = -1
    with pytest.raises(ValueError, match="positive integers"):
        PPR78Spec.from_dict(payload)
    with pytest.raises(ValueError, match="temperature"):
        ppr78_interaction_response(declaration(), *components()[:2], temperature_K=0)


def test_identical_component_predicts_zero_with_all_derivatives():
    response = ppr78_interaction_response(declaration(), components()[0], components()[0],
                                         temperature_K=298.15)
    assert set(response.values()) == {0.0}
    with pytest.raises(TypeError):
        declaration().groups["a"]["1"] = 9
