import json

import pytest
from tests.reference.run_if97_basic import FIELDS, OFFICIAL, official_state

from chemworld.physchem.water_if97 import (
    water_if97_b23_pressure,
    water_if97_gibbs_state,
    water_if97_region,
    water_if97_region3_state,
)
from chemworld.physchem.water_saturation import water_saturation_pressure


@pytest.mark.parametrize("region,temperature,coordinate,expected", OFFICIAL)
def test_official_if97_tables_and_thermodynamic_identities(
    region, temperature, coordinate, expected,
):
    state = official_state(region, temperature, coordinate)
    fields = ("pressure_Pa", *FIELDS[1:]) if region == 3 else FIELDS
    for name, value in zip(fields, expected, strict=True):
        assert getattr(state, name) == pytest.approx(value, rel=2e-8, abs=1e-10)
    volume = state.specific_volume_m3_kg
    assert state.enthalpy_J_kg == pytest.approx(
        state.internal_energy_J_kg + state.pressure_Pa * volume, rel=2e-13)
    assert state.gibbs_energy_J_kg == pytest.approx(
        state.enthalpy_J_kg - temperature * state.entropy_J_kg_K, rel=2e-13)
    assert state.cp_J_kg_K - state.cv_J_kg_K == pytest.approx(
        temperature * volume * state.thermal_expansion_K_inv**2
        / state.isothermal_compressibility_Pa_inv, rel=2e-11)
    assert json.loads(json.dumps(state.to_dict(), allow_nan=False)) == state.to_dict()


@pytest.mark.parametrize("temperature,pressure,region", [
    (273.15, 1000, 1), (623.15, 100e6, 1), (650, 25e6, 3),
    (800, .1e6, 2), (863.15, 100e6, 2), (1073.15, 100e6, 2),
    (1073.150001, 50e6, 5), (2273.15, 50e6, 5),
])
def test_region_limits_are_explicit(temperature, pressure, region):
    assert water_if97_region(temperature, pressure) == region


@pytest.mark.parametrize("temperature,pressure", [
    (True, 1e6), (300, True), (float("nan"), 1e6), (300, float("inf")),
    (273.14, 1e6), (2273.16, 1e6), (1100, 50e6 + 1), (300, 100e6 + 1),
    (300, 0), (300, -1),
])
def test_unsupported_states_are_rejected(temperature, pressure):
    with pytest.raises(ValueError):
        water_if97_gibbs_state(temperature, pressure)


def test_saturation_requires_phase_and_does_not_silently_choose_one():
    pressure = water_saturation_pressure(500)
    assert water_if97_region(500, pressure) == 4
    with pytest.raises(ValueError, match="explicit phase"):
        water_if97_gibbs_state(500, pressure)
    liquid = water_if97_gibbs_state(500, pressure, region=1)
    vapor = water_if97_gibbs_state(500, pressure, region=2)
    assert liquid.mass_density_kg_m3 > vapor.mass_density_kg_m3
    assert vapor.enthalpy_J_kg > liquid.enthalpy_J_kg
    with pytest.raises(ValueError, match="density"):
        water_if97_gibbs_state(650, 25e6)
    with pytest.raises(ValueError, match="outside"):
        water_if97_gibbs_state(300, 1e6, region=2)


def test_metastable_vapor_is_explicit_and_limited_by_the_published_moisture_boundary():
    stable = water_if97_gibbs_state(450, 1e6)
    metastable = water_if97_gibbs_state(450, 1e6, metastable=True)
    assert stable.region == 1 and metastable.region == 2
    assert metastable.stability_scope == "explicit_metastable_vapor"
    for temperature, pressure in [(300, 1e6), (500, 1e6), (273.15, 611.3), (600, 11e6)]:
        with pytest.raises(ValueError):
            water_if97_gibbs_state(temperature, pressure, metastable=True)
    with pytest.raises(ValueError, match="boolean"):
        water_if97_gibbs_state(450, 1e6, metastable=1)


def test_dense_region_rejects_critical_singular_and_unstable_states():
    with pytest.raises(ValueError, match="critical"):
        water_if97_region3_state(647.096, 322)
    with pytest.raises(ValueError, match="unstable"):
        water_if97_region3_state(640, 322)
    for temperature, density in [(600, 500), (900, 500), (650, -1), (650, 1), (650, True)]:
        with pytest.raises(ValueError):
            water_if97_region3_state(temperature, density)
    assert water_if97_b23_pressure(623.15) == pytest.approx(16.5291642526e6, rel=1e-11)


@pytest.mark.parametrize("temperature,pressure", [(300, 3e6), (700, 30e6), (1500, 30e6)])
def test_analytic_responses_match_independent_centered_perturbations(temperature, pressure):
    state = water_if97_gibbs_state(temperature, pressure)
    left = water_if97_gibbs_state(temperature - .001, pressure)
    right = water_if97_gibbs_state(temperature + .001, pressure)
    assert state.cp_J_kg_K == pytest.approx(
        (right.enthalpy_J_kg - left.enthalpy_J_kg) / .002, rel=2e-7)
    assert state.thermal_expansion_K_inv == pytest.approx(
        (right.specific_volume_m3_kg - left.specific_volume_m3_kg)
        / (.002 * state.specific_volume_m3_kg), rel=2e-7)
    low = water_if97_gibbs_state(temperature, pressure - 10)
    high = water_if97_gibbs_state(temperature, pressure + 10)
    assert state.isothermal_compressibility_Pa_inv == pytest.approx(
        -(high.specific_volume_m3_kg - low.specific_volume_m3_kg)
        / (20 * state.specific_volume_m3_kg), rel=2e-6)
