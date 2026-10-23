import json
from dataclasses import replace
from math import pi

import pytest

from chemworld.physchem.tank_geometry import TankCalibrationTable, TankChebyshev, TankGeometry
from chemworld.physchem.tank_heads import TankHead


def test_liquid_state_consumes_current_volume_and_density():
    tank = TankGeometry(2, 3, False)
    state = tank.liquid_state(pi, density_kg_m3=998, gravity_m_s2=9.8)
    assert state["height_m"] == pytest.approx(1)
    assert state["headspace_volume_m3"] == pytest.approx(2 * pi)
    assert state["bottom_gauge_pressure_Pa"] == pytest.approx(998 * 9.8)
    assert state["wetted_area_m2"] == pytest.approx(3 * pi)
    assert state["free_surface_area_m2"] == pytest.approx(pi)
    assert TankGeometry.from_dict(json.loads(json.dumps(tank.to_dict()))) == tank


@pytest.mark.parametrize("horizontal", [True, False])
def test_concave_composition_and_inverse(horizontal):
    a = TankHead("ellipsoidal", 0.25, concave=True)
    b = TankHead("conical", 0.2, concave=True)
    tank = TankGeometry(2, 3, horizontal, a, b)
    assert tank.total_volume_m3 == pytest.approx(3 * pi - a.total_volume(2) - b.total_volume(2))
    levels = [tank.maximum_height_m * i / 20 for i in range(21)]
    volumes = [tank.volume_at_height(h) for h in levels]
    assert volumes == sorted(volumes)
    for h, v in zip(levels, volumes, strict=True):
        assert tank.height_for_volume(v) == pytest.approx(h, abs=2e-9)
    h = 0.37 * tank.maximum_height_m
    dh = 1e-5 * tank.maximum_height_m
    assert tank.cross_section_at_height(h) == pytest.approx(
        (tank.volume_at_height(h + dh) - tank.volume_at_height(h - dh)) / (2 * dh), rel=1e-7
    )
    with pytest.raises(ValueError, match="overlap"):
        TankGeometry(2, 0.5, horizontal, a, b)


def test_sphere_and_zero_length_sizing():
    hemisphere = TankHead("spherical", 0.5)
    sphere = TankGeometry(2, 0, False, hemisphere, hemisphere)
    assert sphere.total_volume_m3 == pytest.approx(4 * pi / 3)
    assert sphere.total_area_m2 == pytest.approx(4 * pi)
    assert sphere.volume_at_height(1) == pytest.approx(2 * pi / 3)
    for kw in ({"diameter_m": 2, "length_m": 0}, {"volume_m3": 4 * pi / 3, "length_m": 0}):
        solved = TankGeometry.from_dimensions(
            **kw, horizontal=False, head_a=hemisphere, head_b=hemisphere
        )
        assert solved.diameter_m == pytest.approx(2)
        assert solved.length_m == 0


def test_deep_spherical_envelope():
    head = TankHead("spherical", 0.75)
    tank = TankGeometry(1, 2, True, head)
    assert tank.maximum_height_m == pytest.approx(13 / 12)
    assert tank.total_volume_m3 == pytest.approx(pi / 2 + pi * 0.75 * (0.75 + 0.75**2) / 6)
    assert tank.volume_at_height(0) == 0
    assert tank.volume_at_height(tank.maximum_height_m / 2) == pytest.approx(
        tank.total_volume_m3 / 2
    )
    assert tank.height_for_volume(tank.total_volume_m3 * 1e-8) > 0
    with pytest.raises(ValueError, match="intersect"):
        replace(head, concave=True)


def test_guppy_total_is_continuous_with_wetted_surface():
    head = TankHead("guppy", 0.5)
    full = head.total_area(2)
    assert head.horizontal_measure(2, 2, "area") == pytest.approx(full)
    errors = [full - head.horizontal_measure(2, 2 - e, "area") for e in (1e-4, 1e-6, 1e-8)]
    assert 0 < errors[2] < errors[1] < errors[0]
    assert errors[2] / errors[1] == pytest.approx(0.1, rel=0.01)
    assert head.horizontal_measure(2, -1, "area") == 0
    assert head.horizontal_measure(2, 3, "area") == full
    tank = TankGeometry(2, 3, False, head, head)
    assert tank.volume_at_height(tank.maximum_height_m / 2) == pytest.approx(
        tank.total_volume_m3 / 2
    )


def test_calibration_roundtrip_and_no_extrapolation():
    tank = TankGeometry(1, 2, True, TankHead("ellipsoidal"), TankHead("torispherical"))
    table = TankCalibrationTable.from_dict(
        json.loads(json.dumps(tank.table(step_m=0.01).to_dict()))
    )
    poly = TankChebyshev.from_dict(json.loads(json.dumps(tank.chebyshev(40, 100).to_dict())))
    for h in (0, 0.1, 0.5, 0.9, 1):
        volume = tank.volume_at_height(h)
        assert table.height_for_volume(volume) == pytest.approx(h, abs=1e-9)
        assert poly.height_for_volume(volume) == pytest.approx(h, abs=5e-4)
        assert poly.volume_at_height(h) == pytest.approx(volume, abs=tank.total_volume_m3 * 5e-4)
    with pytest.raises(ValueError, match="outside"):
        table.height_for_volume(tank.total_volume_m3 + 1)
    with pytest.raises(ValueError, match="outside"):
        poly.volume_at_height(2)


def test_adaptive_calibration_and_budget():
    tank = TankGeometry(1, 2, False)
    poly = tank.chebyshev(None, None, tolerance=1e-10, maximum_degree=16)
    assert poly.height_for_volume(tank.total_volume_m3 / 3) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="degree budget"):
        TankGeometry(1, 2).chebyshev(None, None, tolerance=1e-12, maximum_degree=16)


def test_head_symmetry_scale_and_domain():
    for kind in ("conical", "ellipsoidal", "spherical", "torispherical"):
        head = TankHead(kind)
        assert head.horizontal_measure(2, 1) == pytest.approx(head.total_volume(2) / 2)
        assert head.horizontal_measure(2, 0.3) == pytest.approx(
            8 * head.horizontal_measure(1, 0.15)
        )
        assert head.vertical_capacity(2, head.depth * 2 + 1) == pytest.approx(
            head.total_volume(2) + pi
        )
    with pytest.raises(ValueError):
        TankHead("torispherical", crown_ratio=0.4)
    with pytest.raises(ValueError):
        TankGeometry(1, 1).height_for_volume(-1)


def test_wall_parameter_expansion_and_two_specs():
    tank = TankGeometry(1.2, 3.6, True, TankHead("conical"), TankHead("ellipsoidal"))
    outer = tank.with_wall_thickness(0.01)
    assert outer.diameter_m == 1.22
    assert outer.length_m == pytest.approx(3.62)
    assert outer.head_a.depth * outer.diameter_m == pytest.approx(0.31)
    goals = {"volume": tank.total_volume_m3, "cross_section": tank.cross_section_at_height(0.48)}
    solved = TankGeometry.from_two_specs(
        goals,
        height_m=0.48,
        head_a=tank.head_a,
        head_b=tank.head_b,
        initial_diameter_m=1.1,
        initial_length_m=3.3,
    )
    assert solved.diameter_m == pytest.approx(tank.diameter_m, rel=1e-7)
    assert solved.length_m == pytest.approx(tank.length_m, rel=1e-7)
