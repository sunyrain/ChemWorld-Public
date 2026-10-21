# ruff: noqa: C408 -- named scientific parameters in fixed formula examples
"""Published example values and elementary identities for pure pressure kernels."""

from math import inf, nextafter

import pytest

from chemworld.physchem.vapor_pressure_kernels import pressure_derivatives
from chemworld.physchem.water_saturation import (
    water_saturation_pressure,
    water_saturation_temperature,
)


@pytest.mark.parametrize(
    "equation,temperature,parameters,expected",
    (
        (
            "trc_antoine_extended",
            180,
            dict(Tc=227.51, to=-120, A=8.95894, B=510.595, C=-15.95, n=2.41377, E=-93.74, F=7425.9),
            706317.0898414153,
        ),
        ("boiling_critical_pressure", 347.2, dict(Tb=409.3, Tc=617.1, Pc=36e5), 15209.467273093938),
        ("lee_kesler_pressure", 347.2, dict(Tc=617.1, Pc=36e5, omega=0.299), 13078.694162949312),
        (
            "ambrose_walton_pressure",
            347.25,
            dict(Tc=617.15, Pc=36.09e5, omega=0.304),
            13278.878504306222,
        ),
        ("sanjari_pressure", 347.2, dict(Tc=617.1, Pc=36e5, omega=0.299), 13651.916109552523),
        ("edalat_pressure", 347.2, dict(Tc=617.1, Pc=36e5, omega=0.299), 13461.273080743307),
        ("clapeyron_sublimation", 250, dict(Tt=273.15, Pt=611, Hsub_t=51100), 76.06457150831804),
        ("iapws_saturation", 300, {}, 3536.58941301301),
    ),
)
def test_published_pressure_examples(equation, temperature, parameters, expected):
    value = pressure_derivatives(equation, temperature, parameters, order=0)[0]
    assert value == pytest.approx(expected, rel=2e-12)


def test_arrhenius_third_derivative_matches_direct_identity():
    T, P, slope = 400.0, 100000.0, -1600.0
    result = pressure_derivatives(
        "arrhenius_pressure", T, dict(T_ref=T, P_ref=P, slope=slope), order=3
    )
    l1, l2, l3 = -slope / T**2, 2 * slope / T**3, -6 * slope / T**4
    assert result == pytest.approx((P, P * l1, P * (l1 * l1 + l2), P * (l1**3 + 3 * l1 * l2 + l3)))


@pytest.mark.parametrize("temperature", (273.15, 300, 373.15, 600, 647.096))
def test_common_water_saturation_explicit_inverse_and_domain(temperature):
    pressure = water_saturation_pressure(temperature)
    assert water_saturation_temperature(pressure) == pytest.approx(temperature, abs=2e-9)
    for bad in (0, 200, 650, True):
        with pytest.raises(ValueError):
            water_saturation_pressure(bad)
    with pytest.raises(ValueError):
        water_saturation_temperature(1)


def test_water_inverse_accepts_one_ulp_endpoint_rounding_but_not_extrapolation():
    for T, direction in ((273.15, -inf), (647.096, inf)):
        pressure = water_saturation_pressure(T)
        rounded = nextafter(pressure, direction)
        assert water_saturation_temperature(rounded) == T
        with pytest.raises(ValueError):
            water_saturation_temperature(nextafter(rounded, direction))
