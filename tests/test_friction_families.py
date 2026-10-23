import json

import pytest

from chemworld.physchem.curved_friction import (
    available_helical_friction_methods,
    helical_friction_factor,
    helical_transition_reynolds,
)
from chemworld.physchem.friction_correlations import pipe_friction_factor
from chemworld.physchem.plate_friction import plate_friction_factor
from chemworld.physchem.straight_friction import (
    STRAIGHT_FRICTION_RANGES,
    available_straight_friction_methods,
    friction_transmission_factor,
    straight_friction_correlation,
)


def test_darcy_fanning_and_transmission_use_explicit_bases():
    for method in STRAIGHT_FRICTION_RANGES:
        roughness = 0 if method in {"Blasius", "Prandtl_von_Karman_Nikuradse"} else 0.001
        darcy = straight_friction_correlation(
            1e4, roughness, method=method, check_recommended_ranges=False
        )
        fanning = straight_friction_correlation(
            1e4, roughness, method=method, darcy=False, check_recommended_ranges=False
        )
        assert darcy == 4 * fanning
        transmission = friction_transmission_factor(darcy_friction_factor=darcy)
        assert friction_transmission_factor(transmission_factor=transmission) == pytest.approx(
            darcy
        )
    with pytest.raises(ValueError, match="exactly one"):
        friction_transmission_factor(darcy_friction_factor=0.02, transmission_factor=10)


def test_automatic_regime_and_recommended_domains_are_distinct():
    assert available_straight_friction_methods(100) == ("laminar",)
    assert pipe_friction_factor(100, 0.001, method="Haaland") == 0.64
    with pytest.raises(ValueError, match="recommended"):
        straight_friction_correlation(100, 0.001, method="Haaland")
    with pytest.raises(ValueError, match="unknown"):
        pipe_friction_factor(100, method="typo")
    assert pipe_friction_factor(1e5, 0.001, method="Haaland") == straight_friction_correlation(
        1e5, 0.001, method="Haaland"
    )
    rough_limit = straight_friction_correlation(1e12, 0.01, method="von_Karman")
    assert pipe_friction_factor(1e12, 0.01) == pytest.approx(rough_limit, rel=1e-7)


def test_helical_switch_and_straight_limit():
    critical = helical_transition_reynolds(0.02, 0.4)
    low = helical_friction_factor(critical * (1 - 1e-8), 0.02, 0.4)
    high = helical_friction_factor(critical * (1 + 1e-8), 0.02, 0.4)
    assert low.declared_regime == "laminar" and high.declared_regime == "turbulent"
    assert low.method == "Schmidt laminar" and high.method == "Schmidt turbulent"
    assert len(available_helical_friction_methods(1000, 0.02, 0.4)) == 3
    assert len(available_helical_friction_methods(1e5, 0.02, 0.4)) == 8
    nearly_straight = helical_friction_factor(1000, 0.01, 1e10)
    assert nearly_straight.friction_factor == pytest.approx(0.064, rel=1e-8)
    assert json.loads(json.dumps(high.to_dict()))["critical_reynolds"] == critical


def test_helical_source_extensions_and_unsupported_roughness_are_visible():
    with pytest.raises(ValueError, match="range"):
        helical_friction_factor(100, 0.01, 0.2, method="Mori Nakayama laminar")
    extended = helical_friction_factor(
        100, 0.01, 0.2, method="Mori Nakayama laminar", check_recommended_ranges=False
    )
    assert extended.friction_factor == pytest.approx(0.64 * 1.405296)
    assert any("low-De" in warning for warning in extended.warnings)
    with pytest.raises(ValueError, match="roughness"):
        helical_friction_factor(1e5, 0.02, 0.4, method="Guo", roughness_m=1e-5)
    with pytest.raises(ValueError, match="centerline"):
        helical_friction_factor(1000, 0.2, 0.1)


def test_plate_port_length_convention_limits_and_piecewise_jump():
    for method in ("Martin 1999", "Martin VDI"):
        assert plate_friction_factor(1000, 0, method=method) == pytest.approx(0.064)
        assert plate_friction_factor(1000, 0, method=method, darcy=False) == pytest.approx(0.016)
    before = plate_friction_factor(100, 30, method="Kumar")
    after = plate_friction_factor(100 * (1 + 1e-8), 30, method="Kumar")
    assert abs(after - before) / before > 1e-4
    with pytest.raises(ValueError, match="Re>1000"):
        plate_friction_factor(1000, 45, method="Muley Manglik", enlargement_factor=1.2)
    with pytest.raises(ValueError, match="enlargement"):
        plate_friction_factor(1000, 45, method="Martin VDI", enlargement_factor=1.2)
