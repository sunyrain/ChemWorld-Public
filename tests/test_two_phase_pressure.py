from dataclasses import replace
from math import log10, pi, sqrt

import pytest

from chemworld.physchem.friction_correlations import (
    colebrook_friction_factor,
    pipe_friction_factor,
)
from chemworld.physchem.two_phase_flow import lockhart_martinelli_pressure_drop
from chemworld.physchem.two_phase_holdup import TwoPhaseProperties
from chemworld.physchem.two_phase_pressure import (
    TWO_PHASE_PRESSURE_REQUIREMENTS,
    available_two_phase_pressure_methods,
    two_phase_pressure_drop,
)


def properties():
    return TwoPhaseProperties(900, 10, 0.0005, 1.5e-5, 0.05, 1e7, 5e7)


def evaluate(method, **kwargs):
    arguments = {"mass_flow_kg_s": 0.01, "diameter_m": 0.005, "vapor_quality": 0.3}
    arguments.update(kwargs)
    return two_phase_pressure_drop(properties(), method=method, **arguments)


def test_colebrook_equation_and_explicit_transition():
    assert pipe_friction_factor(1000) == pytest.approx(0.064)
    factor = colebrook_friction_factor(1e5, 0.001)
    assert 1 / sqrt(factor) == pytest.approx(-2 * log10(0.001 / 3.7 + 2.51 / (1e5 * sqrt(factor))))
    assert pipe_friction_factor(2040) == colebrook_friction_factor(2040)
    assert pipe_friction_factor(2040, transition_reynolds=2300) == pytest.approx(64 / 2040)
    with pytest.raises(ValueError):
        pipe_friction_factor(100, -1)


def test_existing_lockhart_martinelli_is_used_without_changing_its_friction():
    result = evaluate("Lockhart_Martinelli")
    original = lockhart_martinelli_pressure_drop(
        mass_flow_kg_s=0.01,
        vapor_quality=0.3,
        liquid_density_kg_m3=900,
        vapor_density_kg_m3=10,
        liquid_viscosity_Pa_s=0.0005,
        vapor_viscosity_Pa_s=1.5e-5,
        diameter_m=0.005,
    )
    assert result.pressure_drop_Pa == original.pressure_drop_Pa
    assert result.pressure_kind == "friction"
    assert evaluate("Lockhart_Martinelli", length_m=7).pressure_drop_Pa == pytest.approx(
        7 * result.pressure_drop_Pa
    )


def test_availability_is_separate_from_physical_domain():
    assert len(TWO_PHASE_PRESSURE_REQUIREMENTS) == 24
    assert len(available_two_phase_pressure_methods(properties())) == 24
    assert not available_two_phase_pressure_methods(TwoPhaseProperties(900, 10))
    with pytest.raises(ValueError, match="reference_only"):
        evaluate("Bankoff")
    assert "reference_only" in " ".join(evaluate("Bankoff", reference_only=True).warnings)
    with pytest.raises(ValueError, match="microtubes"):
        evaluate("Hwang_Kim")
    assert evaluate("Hwang_Kim", check_recommended_ranges=False).warnings
    with pytest.raises(ValueError, match="roughness"):
        evaluate("Kim_Mudawar", roughness_m=1e-5)
    with pytest.raises(ValueError, match="horizontal"):
        evaluate("Chisholm", inclination_degrees=30)
    with pytest.raises(ValueError, match="unknown"):
        evaluate("unknown")


def test_beggs_reports_coupled_terms_and_signed_gravity():
    for angle in (-10, 0, 10):
        result = evaluate("Beggs-Brill", inclination_degrees=angle)
        assert result.pressure_kind == "coupled_total"
        assert result.liquid_holdup is not None and 0 < result.liquid_holdup <= 1
        assert result.friction_component_Pa is not None and result.friction_component_Pa > 0
        assert result.gravity_component_Pa is not None
        assert result.gravity_component_Pa * angle >= 0
        assert result.pressure_drop_Pa == pytest.approx(
            result.friction_component_Pa + result.gravity_component_Pa
        )
        accelerated = evaluate(
            "Beggs-Brill", inclination_degrees=angle, beggs_brill_acceleration=True
        )
        assert accelerated.pressure_drop_Pa * (
            1 - accelerated.acceleration_fraction
        ) == pytest.approx(result.pressure_drop_Pa)
    with pytest.raises(ValueError, match="singularity"):
        two_phase_pressure_drop(
            replace(properties(), pressure_Pa=100),
            method="Beggs-Brill",
            diameter_m=0.005,
            mass_flow_kg_s=0.01,
            vapor_quality=0.3,
            beggs_brill_acceleration=True,
        )


@pytest.mark.parametrize(
    "quality,flux,expected",
    [
        (0.2, 10, (False, False)),
        (0.2, 500, (True, True)),
        (0.001, 500, (True, False)),
        (0.95, 500, (False, True)),
    ],
)
def test_kim_all_four_phase_reynolds_branches(quality, flux, expected):
    result = evaluate("Kim_Mudawar", mass_flow_kg_s=flux * pi * 0.005**2 / 4, vapor_quality=quality)
    assert result.liquid_reynolds is not None and result.vapor_reynolds is not None
    assert (result.liquid_reynolds >= 2000, result.vapor_reynolds >= 2000) == expected
    assert result.pressure_drop_Pa > 0


def test_theissing_pure_phase_limits():
    for quality, density, viscosity in ((0, 900, 0.0005), (1, 10, 1.5e-5)):
        result = evaluate("Theissing", vapor_quality=quality)
        flux = 0.01 / (pi * 0.005**2 / 4)
        single = pipe_friction_factor(flux * 0.005 / viscosity) / 0.005 * flux**2 / (2 * density)
        assert result.pressure_drop_Pa == pytest.approx(single)
        nearby = evaluate("Theissing", vapor_quality=1e-10 if quality == 0 else 1 - 1e-10)
        assert nearby.pressure_drop_Pa == pytest.approx(single, rel=1e-4)
