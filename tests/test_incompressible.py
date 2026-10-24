"""Equation identities, source boundaries and native caloric inversion."""

import json
import math
from collections import Counter
from dataclasses import replace

import pytest

from chemworld.data.incompressible import incompressible_catalog, incompressible_fluid
from chemworld.physchem.incompressible import (
    IncompressibleCorrelation,
    IncompressibleFluid,
    IncompressibleReference,
)
from chemworld.physchem.second_order_pair import SecondOrderPair


def test_full_catalog_and_serialization():
    cards = incompressible_catalog()
    assert len(cards) == 126
    assert Counter(f.fraction_basis for f in cards) == {"pure": 74, "mass": 39, "volume": 13}
    assert sum(f.fluid_id.startswith("Example") for f in cards) == 6
    families = set()
    for fluid in cards:
        assert IncompressibleFluid.from_dict(json.loads(json.dumps(fluid.to_dict()))) == fluid
        families.update(c.family for _, c in fluid.correlations if c is not None)
    assert families == {"polynomial", "exppolynomial", "exponential", "logexponential",
                        "polyoffset"}


@pytest.mark.parametrize("family,coefficients,y,x,expected", [
    ("polynomial", ((2, 3), (4, 5)), 3., .2, 2+3*.2+3*(4+5*.2)),
    ("exppolynomial", ((1, 2),), 3., .2, math.exp(1.4)),
    ("exponential", ((2,), (1,), (3,)), 3., .2, math.exp(-2.5)),
    ("logexponential", ((1,), (2,), (3,)), 3., .2, math.exp(3)*(5/16)**2),
    ("polyoffset", ((1., 2., 3.),), 3., .2, 2+3*(.2-1)),
    ("polyoffset", ((1.,), (2.,), (3.,)), 3., .2, 2+3*(3-1)),
])
def test_function_families(family, coefficients, y, x, expected):
    fit = IncompressibleCorrelation(family, coefficients)
    assert fit.evaluate(y, x) == pytest.approx(expected, rel=1e-13)
    jet = fit.evaluate(SecondOrderPair(y, dx=1), SecondOrderPair(x, dy=1))
    step = 1e-5
    assert jet.dx == pytest.approx((fit.evaluate(y+step, x)-fit.evaluate(y-step, x))/(2*step),
                                  rel=1e-8, abs=1e-9)
    assert jet.dy == pytest.approx((fit.evaluate(y, x+step)-fit.evaluate(y, x-step))/(2*step),
                                  rel=1e-8, abs=1e-9)


def test_caloric_integrals_hand_calculated():
    fit = IncompressibleCorrelation("polynomial", ((1000., 3.), (2., 0.)))
    h, s = fit.caloric_integrals(320., .2, 300., y_center=300., x_center=0.)
    assert h == pytest.approx(1000.6*20+20**2, abs=1e-9)
    assert s == pytest.approx(400.6*math.log(320/300)+2*20, abs=1e-10)
    h0, s0 = fit.caloric_integrals(300., .2, 300., y_center=300., x_center=0.)
    assert (h0, s0) == (0, 0)


@pytest.mark.parametrize("name,x,T", [("MEG", .3, 310), ("APG", .3, 310), ("DowQ", 1, 400)])
def test_gibbs_derivatives_and_explicit_reference(name, x, T):
    fluid = incompressible_fluid(name)
    p, step = 2e6, .001
    state = fluid.state(T, p, x)
    low, high = (fluid.state(T+d, p, x) for d in (-step, step))
    assert -(high.gibbs_energy_J_kg-low.gibbs_energy_J_kg)/(2*step) == pytest.approx(
        state.entropy_J_kg_K, rel=2e-7, abs=1e-7)
    assert T*(high.entropy_J_kg_K-low.entropy_J_kg_K)/(2*step) == pytest.approx(
        state.cp_J_kg_K, rel=2e-7)
    low, high = (fluid.state(T, p+d, x) for d in (-10, 10))
    assert (high.enthalpy_J_kg-low.enthalpy_J_kg)/20 == pytest.approx(state.dh_dp_J_kg_Pa)
    assert (high.entropy_J_kg_K-low.entropy_J_kg_K)/20 == pytest.approx(state.ds_dp_J_kg_K_Pa)
    assert (high.gibbs_energy_J_kg-low.gibbs_energy_J_kg)/20 == pytest.approx(
        state.specific_volume_m3_kg)
    ref = IncompressibleReference(T, p, 10., 20.)
    state = replace(fluid, reference=ref).state(T, p, x)
    assert state.enthalpy_J_kg == pytest.approx(10., abs=1e-9)
    assert state.entropy_J_kg_K == pytest.approx(20., abs=1e-10)


def test_pressure_capacity_correction_is_not_silently_dropped():
    fluid = incompressible_fluid("MEG")
    T, p, x = 310., 2e6, .3
    state = fluid.state(T, p, x)
    volume, _ = fluid._volume(SecondOrderPair(T, dx=1), x)
    assert state.cp_J_kg_K == pytest.approx(state.source_cp_J_kg_K-p*T*volume.dxx, abs=1e-9)
    assert abs(state.cp_J_kg_K-state.source_cp_J_kg_K) > .01


@pytest.mark.parametrize("name,x,T,bounds", [
    ("MEG", .3, 310., (280., 350.)), ("MCA", .2, 295., (280., 310.)),
    ("DowQ", 1., 400., (350., 450.)),
])
def test_ph_closure_and_unreachable_target(name, x, T, bounds):
    fluid = incompressible_fluid(name)
    h = fluid.state(T, 2e6, x).enthalpy_J_kg
    assert fluid.temperature_from_enthalpy(2e6, h, x, bounds_K=bounds) == pytest.approx(T, abs=1e-8)
    with pytest.raises(ValueError, match="outside"):
        fluid.temperature_from_enthalpy(2e6, h+1e10, x, bounds_K=bounds)


@pytest.mark.parametrize("T,p,x", [(100., 1e5, .3), (500., 1e5, .3), (300., -1., .3),
                                   (300., 1e5, -.1), (300., 1e5, .7),
                                   (float("nan"), 1e5, .3), (300., 1e5, True)])
def test_coordinate_rejections(T, p, x):
    with pytest.raises(ValueError):
        incompressible_fluid("MEG").state(T, p, x)


def test_freezing_saturation_missing_fields_and_fraction_conversion():
    meg = incompressible_fluid("MEG")
    freezing = meg.freezing_temperature(1e5, .3)
    with pytest.raises(ValueError, match="freezing"):
        meg.state(freezing-.01, 1e5, .3)
    with pytest.raises(ValueError, match="saturation pressure unavailable"):
        meg.saturation_pressure(300., .3)
    fluid = incompressible_fluid("DowQ")
    T = max(fluid.minimum_saturation_temperature_K+1., 500.)
    psat = fluid.saturation_pressure(T, 1.)
    with pytest.raises(ValueError, match="saturation"):
        fluid.state(T, psat*.5, 1.)
    with pytest.raises(ValueError, match="conversion"):
        meg.concentration(.3, basis="volume")
    with pytest.raises(ValueError, match="unavailable"):
        incompressible_fluid("FoodIce").property_value("viscosity", 260., 1e5, 1.)
    with pytest.raises(ValueError, match="nonpositive"):
        incompressible_fluid("LiBr").property_value("conductivity", 300., 2e6, .3)
    with pytest.raises(ValueError, match="fraction=1"):
        fluid.state(400., 2e6, .5)


@pytest.mark.parametrize("family,coefficients", [
    ("unknown", ((1,),)), ("polynomial", ((),)), ("polynomial", ((1.,), (2., 3.))),
    ("polynomial", ((float("inf"),),)), ("polynomial", ((True,),)),
    ("exponential", ((1., 2., 3.),)), ("polyoffset", ((1.,),)),
])
def test_invalid_parameters(family, coefficients):
    with pytest.raises(ValueError):
        IncompressibleCorrelation(family, coefficients)


def test_singular_functions_are_not_interpolated():
    with pytest.raises(ValueError, match="singular"):
        IncompressibleCorrelation("exponential", ((1,), (-2,), (3,))).evaluate(2, .2)
    with pytest.raises(ValueError, match="real domain"):
        IncompressibleCorrelation("logexponential", ((0,), (1,), (2,))).evaluate(-2, .2)


def test_public_exports():
    from chemworld import data, physchem

    assert data.incompressible_fluid is incompressible_fluid
    assert physchem.IncompressibleFluid is IncompressibleFluid


def test_reference_and_phase_coverage_is_explicit():
    state = incompressible_fluid("ExamplePure").state(373.15, 2e6, 1.)
    assert state.reference_temperature_outside_fit
    assert not state.freezing_boundary_available
    assert not state.saturation_boundary_available_at_temperature
    state = incompressible_fluid("MEG").state(300., 2e6, .3)
    assert not state.reference_temperature_outside_fit
    assert state.freezing_boundary_available
    assert not state.saturation_boundary_available_at_temperature
    assert "mixing_heat_not_measured" in state.caloric_scope


def test_build_contract_includes_only_the_intended_parameter_files():
    from scripts.community_release_build import public_path, wheel_destination

    for name in ("incompressible.json", "CoolProp-MIT.txt"):
        path = "src/chemworld/data/parameters/"+name
        assert public_path(path)
        assert wheel_destination(path) == "chemworld/data/parameters/"+name
    assert not public_path("src/chemworld/data/parameters/private-provider-response.json")


def test_source_import_keeps_column_offset_and_does_not_invent_missing_fields():
    from chemworld.data.incompressible import load_incompressible_fluid

    source = {"name": "synthetic", "description": "test", "reference": "hand calculation",
              "xid": "mass", "Tmin": 250., "Tmax": 400.,
              "density": {"type": "polynomial", "coeffs": [[1000., 1.]]},
              "specific_heat": {"type": "polynomial", "coeffs": [[1000., 2.]]},
              "T_freeze": {"type": "polyoffset", "coeffs": [1., 200., .001]}}
    fluid = load_incompressible_fluid(source, source="explicit-test-source")
    assert fluid.freezing_temperature(1e5, .2) == pytest.approx(200.+.001*(1e5-1))
    assert dict(fluid.correlations)["viscosity"] is None
