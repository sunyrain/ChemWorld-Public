import json
from dataclasses import replace

import pytest

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.corresponding_states import (
    CorrespondingStatesComponent,
    CubicShapeComponent,
    ExtendedCorrespondingStatesModel,
    HuberElyModel,
    LeeKeslerPlockerModel,
    MBWRReference,
    lee_kesler_basis,
)
from chemworld.physchem.cpa import CPAComponent, CPAModel
from chemworld.physchem.cubic_mixing import GibbsExcessMixingSpec
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.molecular_eos import (
    molecular_density_from_pressure,
    molecular_eos_state,
    molecular_scaled_derivative,
)


def shape():
    return CubicShapeComponent("ref", 0.7, 7e-5, 370.0, "soave", (0.6,), "synthetic")


def reference():
    p = shape()
    return CPAModel(
        (
            CPAComponent(
                p.component_id,
                p.attraction_Pa_m6_mol2,
                p.covolume_m3_mol,
                p.critical_temperature_K,
                p.alpha_coefficients[0],
                p.source,
            ),
        ),
        ((0.0,),),
        (10.0, 3000.0),
        "synthetic",
    )


@pytest.mark.parametrize("omega", [0.0, 0.2, 0.3978, 0.55])
def test_lkp_conformal_consistency_and_inverse(omega):
    c = CorrespondingStatesComponent("test", 370.0, 2e-4, omega, "synthetic")
    model = LeeKeslerPlockerModel((c,), ((1.0,),), (100.0, 1000.0), "test")
    state = molecular_eos_state(model, 500.0, 1000.0, (1.0,))
    inverse = molecular_density_from_pressure(
        model,
        500.0,
        state.pressure_Pa,
        (1.0,),
        initial_density_mol_m3=1100.0,
        density_bounds_mol_m3=(500.0, 1500.0),
    )
    assert inverse.molar_density_mol_m3 == pytest.approx(1000.0, rel=2e-10)
    assert model.alphar(500.0, 0.0, (1.0,)) == pytest.approx(0.0, abs=1e-14)
    for i in range(4):
        assert (
            abs(
                molecular_scaled_derivative(
                    model, 500.0, 1000.0, (1.0,), inverse_temperature_order=i, density_order=3 - i
                )
            )
            < 10
        )
    if omega in (0.0, 0.3978):
        _, dr, _ = model.reducing_parameters((1.0,))
        assert state.residual_helmholtz_J_mol / (500 * model.gas_constant_J_mol_K) == pytest.approx(
            lee_kesler_basis(500 / 370, 1000 * dr, omega != 0)[0], abs=1e-13
        )
    assert LeeKeslerPlockerModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model


def test_lkp_pressure_interpolation_is_not_fixed_density_interpolation():
    c = CorrespondingStatesComponent("test", 370.0, 2e-4, 0.2, "synthetic")
    model = LeeKeslerPlockerModel((c,), ((1.0,),), (100.0, 1000.0), "test")
    tc, ds, w = model.reducing_parameters((1.0,))
    naive = (1 - w) * lee_kesler_basis(400 / tc, 3000 * ds)[0] + w * lee_kesler_basis(
        400 / tc, 3000 * ds, True
    )[0]
    assert abs(model.alphar(400.0, 3000.0, (1.0,)) - naive) > 1e-5


@pytest.mark.parametrize("kind", ["soave", "twu91"])
def test_ecs_identity_mapping_and_temperature_derivatives(kind):
    p = (
        shape()
        if kind == "soave"
        else replace(shape(), alpha_model="twu91", alpha_coefficients=(0.8, 0.9, 1.1))
    )
    model = ExtendedCorrespondingStatesModel(
        (p,), ((0.0,),), p, reference(), (200.0, 800.0), "synthetic"
    )
    for t in (250.0, 370.0, 600.0):
        assert model.shape_factors(t, (1.0,)) == pytest.approx((1.0, 1.0), rel=2e-12)
        for i in range(4):
            actual = molecular_scaled_derivative(
                model, t, 1000.0, (1.0,), inverse_temperature_order=i
            )
            expected = molecular_scaled_derivative(
                reference(), t, 1000.0, (1.0,), inverse_temperature_order=i
            )
            assert actual == pytest.approx(expected, rel=2e-10, abs=2e-12)
    assert (
        ExtendedCorrespondingStatesModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    )


def test_huber_identity_and_nonpositive_shape_rejection():
    c = CorrespondingStatesComponent("ref", 370.0, 2e-4, 0.2, "synthetic")
    model = HuberElyModel(
        c, 0.28, c, 0.28, (0.2, -0.1), (0.3, 0.1), reference(), (200.0, 800.0), "synthetic"
    )
    assert model.alphar(500.0, 1000.0, (1.0,)) == reference().alphar(500.0, 1000.0, (1.0,))
    assert HuberElyModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    bad = replace(
        model, component=replace(c, acentric_factor=1.2), temperature_coefficients=(-2.0, 0.0)
    )
    with pytest.raises(ValueError, match="nonpositive"):
        bad.alphar(500.0, 1000.0, (1.0,))


@pytest.mark.parametrize("variant", [19, 32])
def test_mbwr_matches_pressure_definition_at_vacuum_and_dense_states(variant):
    coefficients = [0.0] * variant
    coefficients[0] = 0.1
    coefficients[-1] = 1e-4
    model = MBWRReference(
        "synthetic", variant, tuple(coefficients), 0.04, (100.0, 1000.0), "synthetic"
    )
    power, exponent = (5, -4) if variant == 19 else (13, -4)
    factor = 1000 if variant == 19 else 1
    from math import exp

    for rho in (0.0, 1e-6, 100.0, 5000.0, 15000.0):
        t = 400.0
        derivative = partial_derivative(
            lambda v, t=t: model.alphar(t, v["rho"], (1.0,)), {"rho": rho}, ("rho",)
        )
        d = rho / 1000
        pressure_residual = factor * (
            0.1 * t * d * d + 1e-4 * t**exponent * d**power * exp(-0.04 * d * d)
        )
        assert rho * rho * t * model.gas_constant_J_mol_K * derivative == pytest.approx(
            pressure_residual, rel=2e-11, abs=1e-15
        )
    assert MBWRReference.from_dict(json.loads(json.dumps(model.to_dict()))) == model


def test_domains_and_explicit_matrix_validation():
    c = CorrespondingStatesComponent("ref", 370.0, 2e-4, 0.2, "synthetic")
    with pytest.raises(ValueError, match="unit diagonal"):
        LeeKeslerPlockerModel((c,), ((0.0,),), (100.0, 1000.0), "test")
    model = LeeKeslerPlockerModel((c,), ((1.0,),), (100.0, 1000.0), "test")
    for t, rho, x in ((50.0, 10.0, (1.0,)), (500.0, -1.0, (1.0,)), (500.0, 10.0, (0.5,))):
        with pytest.raises(ValueError, match="outside declared"):
            model.alphar(t, rho, x)
    with pytest.raises(ValueError, match="positive Soave"):
        ExtendedCorrespondingStatesModel(
            (shape(),),
            ((0.0,),),
            replace(shape(), alpha_coefficients=(-1.0,)),
            reference(),
            (100.0, 800.0),
            "test",
        )
    with pytest.raises(ValueError, match="positive"):
        lee_kesler_basis(float("nan"), 1.0)


@pytest.mark.parametrize("rule", ["huron_vidal", "mhv", "mhv1", "mhv2", "wong_sandler"])
@pytest.mark.parametrize("cubic_model", ["srk", "peng_robinson"])
def test_ecs_ge_matches_direct_cubic_thermodynamics(rule, cubic_model):
    cs = (
        CPAComponent("a", 0.2, 2e-5, 500.0, 0.6, "synthetic"),
        CPAComponent("b", 0.3, 3e-5, 500.0, 0.6, "synthetic"),
    )
    activity = ActivityModelSpec(
        "synthetic",
        ("a", "b"),
        "nrtl",
        {"tau_b:a|b": 100.0, "tau_b:b|a": 50.0, "alpha:a|b": 0.3, "alpha:b|a": 0.3},
    )
    coeffs = (-0.53,) if rule == "mhv1" else (-0.4347, -0.003654) if rule == "mhv2" else ()
    ge = GibbsExcessMixingSpec(
        rule, activity, coeffs, {"a|b": 0.03} if rule == "wong_sandler" else {}
    )
    direct = CPAModel(
        cs,
        ((0.0, 0.0), (0.0, 0.0)),
        (100.0, 800.0),
        "synthetic",
        cubic_model=cubic_model,
        ge_mixing=ge,
    )
    shapes = tuple(
        CubicShapeComponent(
            c.component_id,
            c.attraction_Pa_m6_mol2,
            c.covolume_m3_mol,
            c.critical_temperature_K,
            "soave",
            (c.alpha_coefficient,),
            c.source,
        )
        for c in cs
    )
    pure = replace(direct, components=(cs[0],), interaction=((0.0,),), ge_mixing=None)
    ecs = ExtendedCorrespondingStatesModel(
        shapes,
        direct.interaction,
        shapes[0],
        pure,
        (100.0, 800.0),
        "synthetic",
        cubic_model=cubic_model,
        ge_mixing=ge,
    )
    assert ExtendedCorrespondingStatesModel.from_dict(json.loads(json.dumps(ecs.to_dict()))) == ecs
    for x in ((1.0, 0.0), (0.3, 0.7), (0.0, 1.0)):
        state = molecular_eos_state(ecs, 200.0, 100.0, x)
        expected = molecular_eos_state(direct, 200.0, 100.0, x)
        assert state.pressure_Pa == pytest.approx(expected.pressure_Pa, rel=1e-12)
        assert state.residual_cv_J_mol_K == pytest.approx(expected.residual_cv_J_mol_K, rel=1e-11)
        assert state.log_fugacity_coefficients == pytest.approx(
            expected.log_fugacity_coefficients, rel=1e-11
        )
        assert molecular_scaled_derivative(ecs, 200.0, 100.0, x, inverse_temperature_order=3) == (
            pytest.approx(
                molecular_scaled_derivative(direct, 200.0, 100.0, x, inverse_temperature_order=3),
                rel=1e-9,
                abs=1e-12,
            )
        )
