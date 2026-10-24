import json
import math
from types import SimpleNamespace

import pytest

from chemworld.physchem import (
    condensed_water_state,
    humid_air_condensation_point,
    humid_air_from_relative_humidity,
    humid_air_from_water_fraction,
    humid_air_relative_humidity,
    humid_air_saturation,
    humid_air_tp,
)


@pytest.mark.parametrize("T,p,phase", [(230, 101325, "ice"), (273, 101325, "ice"),
                                      (273, 5e6, "liquid"), (300, 101325, "liquid")])
def test_condensed_phase_is_selected_by_gibbs_energy(T, p, phase):
    result = humid_air_saturation(T, p)
    assert result.condensate.phase == phase
    assert not result.condensate.metastable_against_other_condensed_phase
    assert abs(result.chemical_potential_residual_J_kg) <= 1e-5
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False))["condensate"]["phase"] == phase


@pytest.mark.parametrize("phase,p,metastable", [("liquid", 101325, True), ("ice", 101325, False),
                                               ("liquid", 5e6, False), ("ice", 5e6, True)])
def test_explicit_surface_is_retained_and_metastability_is_labelled(phase, p, metastable):
    result = humid_air_saturation(273, p, condensed_phase=phase)
    assert result.condensate.phase == phase
    assert result.condensate.metastable_against_other_condensed_phase is metastable
    dew = humid_air_condensation_point(result.gas.water_mass_fraction, p, condensed_phase=phase)
    assert dew.temperature_K == pytest.approx(273, abs=2e-7)
    assert dew.condensate.phase == phase


@pytest.mark.parametrize("V", [0., 1e-20, 1e-12, .01, .1])
def test_water_coordinate_preserves_trace_species_and_shared_energy(V):
    state = humid_air_from_water_fraction(V, 300, 101325)
    assert state.water_mass_fraction == V
    assert state.humidity_ratio_kg_water_kg_dry_air == pytest.approx(V/(1-V), rel=1e-14, abs=0)
    assert state.water_mole_fraction == pytest.approx((V/.018015268)/(
        V/.018015268+(1-V)/.02896546), rel=1e-14, abs=0)
    if V:
        assert math.isfinite(state.water_chemical_potential_J_kg)
    else:
        assert state.water_chemical_potential_J_kg is None
    if V in (0., .01, .1):
        original = humid_air_tp(1-V, 300, 101325)
        assert state.enthalpy_J_kg == pytest.approx(original.enthalpy_J_kg, rel=2e-12)


def test_relative_humidity_is_wmo_mole_fraction_ratio_and_zero_is_dry():
    saturated = humid_air_saturation(350, 101325)
    half = humid_air_from_relative_humidity(350, 101325, .5)
    assert half.water_mole_fraction == pytest.approx(
        .5*saturated.gas.water_mole_fraction, rel=2e-12)
    assert half.humidity_ratio_kg_water_kg_dry_air != pytest.approx(
        .5*saturated.gas.humidity_ratio_kg_water_kg_dry_air, rel=1e-3)
    dry = humid_air_from_relative_humidity(350, 101325, 0)
    assert dry.water_mass_fraction == 0
    assert humid_air_relative_humidity(0, 350, 101325) == 0
    assert humid_air_relative_humidity(.4, 350, 101325) > 1


@pytest.mark.parametrize("T,p", [(193, 5e6), (473, 5e6)])
def test_temperature_and_pressure_endpoints_close_without_extrapolation(T, p):
    saturated = humid_air_saturation(T, p)
    dew = humid_air_condensation_point(saturated.gas.water_mass_fraction, p)
    assert dew.temperature_K == pytest.approx(T, abs=2e-7)
    assert abs(dew.gas.pressure_Pa-p) <= p*5e-11


@pytest.mark.parametrize("args", [(192.9, 1e5), (473.1, 1e5), (True, 1e5), (math.nan, 1e5),
                                  (300, 0), (300, -1), (300, 5e6+1), (300, math.inf),
                                  (300, True)])
def test_equilibrium_domain_refusals(args):
    with pytest.raises(ValueError):
        humid_air_saturation(*args)


@pytest.mark.parametrize("rh", [True, -.1, 1.1, math.nan, math.inf])
def test_invalid_rh_is_not_clipped(rh):
    with pytest.raises(ValueError):
        humid_air_from_relative_humidity(300, 101325, rh)


@pytest.mark.parametrize("V", [0, 1, 1e-20])
def test_missing_condensation_point_is_not_reported_as_an_endpoint(V):
    with pytest.raises(ValueError):
        humid_air_condensation_point(V, 101325)


def test_unreachable_saturation_and_surface_domain_are_explicit():
    with pytest.raises(ValueError, match="no converged"):
        humid_air_saturation(400, 1e4)
    with pytest.raises(ValueError, match="liquid"):
        condensed_water_state(260, 101325, phase="liquid")
    with pytest.raises(ValueError, match="ice"):
        condensed_water_state(300, 101325, phase="ice")
    with pytest.raises(ValueError, match="phase"):
        humid_air_condensation_point(.01, 101325, condensed_phase="solid")


def test_failed_shared_root_never_becomes_a_saturation_fit(monkeypatch):
    from chemworld.physchem import humid_air_equilibrium

    monkeypatch.setattr(humid_air_equilibrium, "solve_root", lambda *a, **k:
                        SimpleNamespace(success=False, failure_reason="forced failure"))
    with pytest.raises(ValueError, match="forced failure"):
        humid_air_saturation(230, 101325)
    with pytest.raises(ValueError, match="forced failure"):
        condensed_water_state(300, 101325)
