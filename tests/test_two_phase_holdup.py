import json

import pytest

from chemworld.physchem.two_phase_holdup import (
    TWO_PHASE_VISCOSITY_METHODS,
    VOIDAGE_REQUIREMENTS,
    TwoPhaseProperties,
    available_two_phase_viscosity_methods,
    available_voidage_methods,
    two_phase_mixture_density,
    two_phase_mixture_viscosity,
    two_phase_void_fraction,
    void_fraction_from_density,
)


def props():
    return TwoPhaseProperties(900, 10, 0.0005, 1.5e-5, 0.05, 1e6, 5e6)


def test_exact_pure_phases_and_property_roundtrip():
    properties = props()
    assert TwoPhaseProperties.from_dict(json.loads(json.dumps(properties.to_dict()))) == properties
    for method in VOIDAGE_REQUIREMENTS:
        assert two_phase_void_fraction(properties, 0, method=method) == 0
        assert two_phase_void_fraction(properties, 1, method=method) == 1
    for method in TWO_PHASE_VISCOSITY_METHODS:
        assert two_phase_mixture_viscosity(properties, 0, method=method) == pytest.approx(0.0005)
        assert two_phase_mixture_viscosity(properties, 1, method=method) == pytest.approx(1.5e-5)


def test_mass_quality_is_not_voidage_density_roundtrip():
    properties = props()
    alpha = two_phase_void_fraction(properties, 0.5, method="homogeneous")
    assert alpha > 0.98
    density = two_phase_mixture_density(properties, alpha)
    assert density == pytest.approx(1 / (0.5 / 10 + 0.5 / 900))
    assert void_fraction_from_density(properties, density) == pytest.approx(alpha)
    with pytest.raises(ValueError, match="between"):
        void_fraction_from_density(properties, 901)


def test_input_availability_and_model_domain_are_distinct():
    density_only = TwoPhaseProperties(900, 10)
    candidates = available_voidage_methods(density_only)
    assert not available_two_phase_viscosity_methods(density_only)
    assert len(available_two_phase_viscosity_methods(props())) == 6
    assert "homogeneous" in candidates and "Thom" not in candidates and "Dix" not in candidates
    with pytest.raises(ValueError, match="requires"):
        two_phase_void_fraction(density_only, 0.5, method="Thom")
    with pytest.raises(ValueError, match="Frlo"):
        two_phase_void_fraction(
            props(), 0.5, method="Xu Fang voidage", mass_flux_kg_m2_s=1, diameter_m=0.02
        )
    formula = two_phase_void_fraction(
        props(),
        0.5,
        method="Xu Fang voidage",
        mass_flux_kg_m2_s=1,
        diameter_m=0.02,
        check_recommended_ranges=False,
    )
    assert 0 < formula < 1
    with pytest.raises(ValueError, match="nonphysical"):
        two_phase_void_fraction(props(), 1e-8, method="Domanski Didion")
