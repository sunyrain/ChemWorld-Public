import json
import math
from types import SimpleNamespace

import pytest

from chemworld.physchem import (
    humid_air_from_relative_humidity,
    humid_air_from_water_fraction,
    humid_air_ph,
    humid_air_saturation,
    humid_air_saturation_response,
    humid_air_wet_bulb,
)


@pytest.mark.parametrize("T,p,phase", [(230, 101325, "ice"), (300, 101325, "liquid"),
                                      (400, 2e6, "liquid")])
def test_saturation_response_matches_independent_centered_temperature_difference(T, p, phase):
    response = humid_air_saturation_response(T, p, condensed_phase=phase)
    low = humid_air_saturation(T-.001, p, condensed_phase=phase).gas
    high = humid_air_saturation(T+.001, p, condensed_phase=phase).gas
    for value, field in ((response.dhumidity_ratio_dT_K_inv, "humidity_ratio_kg_water_kg_dry_air"),
                         (response.dmass_density_dT_kg_m3_K, "mass_density_kg_m3"),
                         (response.denthalpy_dT_J_kg_dry_air_K, "enthalpy_J_kg_dry_air")):
        numerical = (getattr(high, field)-getattr(low, field))/.002
        assert value == pytest.approx(numerical, rel=2e-5, abs=1e-12)
    json.dumps(response.to_dict(), allow_nan=False)


@pytest.mark.parametrize("V,T,p", [(0, 193, 5e6), (0, 473, 5e6), (1, 400, 1e4),
                                  (1e-20, 210, 101325), (.02, 350, 101325)])
def test_ph_limits_preserve_the_original_composition_and_reference(V, T, p):
    inlet = humid_air_from_water_fraction(V, T, p)
    recovered = humid_air_ph(V, p, inlet.enthalpy_J_kg)
    assert recovered.water_mass_fraction == V
    assert recovered.temperature_K == pytest.approx(T, abs=2e-7)
    assert recovered.enthalpy_J_kg == pytest.approx(inlet.enthalpy_J_kg, abs=1e-6)
    json.dumps(recovered.to_dict(), allow_nan=False)


@pytest.mark.parametrize("T,p,phase", [(193, 101325, "ice"), (270, 101325, "ice"),
                                      (300, 101325, "liquid"), (473, 5e6, "liquid")])
def test_saturated_limit_has_zero_water_addition_and_no_temperature_change(T, p, phase):
    inlet = humid_air_saturation(T, p, condensed_phase=phase).gas
    result = humid_air_wet_bulb(T, p, inlet.water_mass_fraction, condensed_phase=phase)
    assert result.temperature_K == pytest.approx(T, abs=2e-7)
    assert result.water_added_kg_per_kg_dry_air == pytest.approx(0, abs=1e-12)
    assert abs(result.energy_residual_J_kg_dry_air) <= 1e-6
    json.dumps(result.to_dict(), allow_nan=False)


def test_supply_phase_changes_energy_input_without_switching_during_the_root():
    inlet = humid_air_from_relative_humidity(273.15, 101325, .9, condensed_phase="ice")
    ice, liquid = [humid_air_wet_bulb(273.15, 101325, inlet.water_mass_fraction,
                                     condensed_phase=phase) for phase in ("ice", "liquid")]
    assert ice.outlet.condensate.phase == "ice"
    assert liquid.outlet.condensate.phase == "liquid"
    assert liquid.outlet.condensate.metastable_against_other_condensed_phase
    assert abs(ice.temperature_K-liquid.temperature_K) > 1e-5
    assert abs(ice.energy_residual_J_kg_dry_air) <= 1e-6
    assert abs(liquid.energy_residual_J_kg_dry_air) <= 1e-6


@pytest.mark.parametrize("p", [1e4, 101325, 5e6])
@pytest.mark.parametrize("V", [0, .001, .1])
def test_hot_low_pressure_and_high_pressure_inlets_have_conservative_wet_bulbs(p, V):
    result = humid_air_wet_bulb(473, p, V, condensed_phase="liquid")
    assert 270 <= result.temperature_K <= 473
    assert result.water_added_kg_per_kg_dry_air >= -1e-12
    assert abs(result.energy_residual_J_kg_dry_air) <= 1e-6
    assert result.outlet.gas.pressure_Pa == pytest.approx(p, rel=5e-11)


@pytest.mark.parametrize("phase", ["auto", "solid", "gas", None])
def test_unknown_supply_phase_is_not_guessed(phase):
    with pytest.raises(ValueError, match="explicit"):
        humid_air_wet_bulb(300, 101325, .01, condensed_phase=phase)
    with pytest.raises(ValueError, match="explicit"):
        humid_air_saturation_response(300, 101325, condensed_phase=phase)


@pytest.mark.parametrize("T,p,V,phase", [(193, 101325, 0, "ice"), (260, 101325, .001, "liquid"),
                                        (300, 101325, .03, "liquid"), (300, 101325, 1, "liquid"),
                                        (True, 101325, .01, "liquid"), (300, 0, .01, "liquid"),
                                        (300, 5e6+1, .01, "liquid"),
                                        (300, 101325, math.nan, "liquid")])
def test_unsupported_or_supersaturated_wet_bulb_inputs_are_rejected(T, p, V, phase):
    with pytest.raises(ValueError):
        humid_air_wet_bulb(T, p, V, condensed_phase=phase)


@pytest.mark.parametrize("h", [True, math.nan, math.inf, 1e9, -1e9])
def test_invalid_or_unreachable_enthalpy_is_not_clipped(h):
    with pytest.raises(ValueError):
        humid_air_ph(.01, 101325, h)


def test_failed_shared_roots_do_not_return_seeds_or_saturation_fits(monkeypatch):
    from chemworld.physchem import humid_air_psychrometrics

    monkeypatch.setattr(humid_air_psychrometrics, "solve_root", lambda *a, **k:
                        SimpleNamespace(success=False, failure_reason="forced failure"))
    with pytest.raises(ValueError, match="forced failure"):
        humid_air_ph(.01, 101325, 1e5)
    with pytest.raises(ValueError, match="forced failure"):
        humid_air_wet_bulb(300, 101325, .01, condensed_phase="liquid")
