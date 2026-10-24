"""Endpoint resolution checks declared after the first ice block's static review."""

import math
from types import SimpleNamespace

import pytest

from chemworld.physchem import water_ice_boundaries
from chemworld.physchem.second_order_pair import SecondOrderPair


@pytest.mark.parametrize("p", [611.657+1e-5, 615, 1000, 101325, 1e7])
def test_melting_inverse_respects_float_temperature_resolution(p):
    T = water_ice_boundaries.water_melting_temperature(p)
    recovered = water_ice_boundaries.water_melting_pressure(SecondOrderPair(T, dx=1))
    resolution = 8 * math.ulp(T) * abs(recovered.dx)
    assert abs(recovered.value-p) <= max(2e-11*p, resolution)


def test_root_failure_remains_explicit(monkeypatch):
    monkeypatch.setattr(water_ice_boundaries, "solve_root", lambda *a, **k:
                        SimpleNamespace(success=False, failure_reason="forced nonconvergence"))
    with pytest.raises(ValueError, match="forced nonconvergence"):
        water_ice_boundaries.water_melting_temperature(101325)
