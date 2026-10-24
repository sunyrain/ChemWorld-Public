"""Physical branches, caloric joins and explicit rejection in the IF97 solver."""

import json

import pytest

from chemworld.physchem.residual_system import RootResult
from chemworld.physchem.water_if97 import _gibbs_state, water_if97_region3_state
from chemworld.physchem.water_if97_solvers import (
    IF97Equilibrium,
    water_if97_at_pressure,
    water_if97_saturation,
    water_if97_tp,
)
from chemworld.physchem.water_saturation import water_saturation_pressure


def test_public_facade_resolves_the_original_water_implementations():
    from chemworld import physchem

    for function in (water_if97_tp, water_if97_saturation, water_if97_at_pressure,
                     water_if97_region3_state, IF97Equilibrium):
        assert getattr(physchem, function.__name__) is function


@pytest.mark.parametrize("temperature,density", [(650, 500), (650, 200), (750, 500)])
def test_pressure_solver_recovers_official_dense_states(temperature, density):
    original = water_if97_region3_state(temperature, density)
    recovered = water_if97_tp(temperature, original.pressure_Pa)
    assert recovered.mass_density_kg_m3 == pytest.approx(density, rel=2e-10)
    assert recovered.enthalpy_J_kg == pytest.approx(original.enthalpy_J_kg, rel=2e-10)
    assert recovered.stability_scope == "IF97_stable_region"


@pytest.mark.parametrize("temperature,pressure", [
    (300, 3e6), (700, 3500), (625, 17e6), (640, 19e6),
    (650, 25.6e6), (647.2, 22.1e6), (2000, 30e6),
])
@pytest.mark.parametrize("target", ["enthalpy_J_kg", "entropy_J_kg_K"])
def test_caloric_inversion_recovers_stable_state_without_a_hint(temperature, pressure, target):
    original = water_if97_tp(temperature, pressure)
    recovered = water_if97_at_pressure(pressure, **{target: getattr(original, target)})
    assert recovered.homogeneous is not None
    assert recovered.temperature_K == pytest.approx(temperature, abs=1e-6)
    assert recovered.mass_density_kg_m3 == pytest.approx(original.mass_density_kg_m3, rel=2e-9)


@pytest.mark.parametrize("temperature", [300, 623.15, 640, 646.5])
@pytest.mark.parametrize("quality", [0, .2, .8, 1])
def test_coexistence_uses_mass_lever_rule_and_recovers_quality(temperature, quality):
    pressure = water_saturation_pressure(temperature)
    with pytest.raises(ValueError, match="explicit phase"):
        water_if97_tp(temperature, pressure)
    state = water_if97_saturation(pressure, quality)
    assert state.liquid.mass_density_kg_m3 > state.vapor.mass_density_kg_m3
    assert state.specific_volume_m3_kg == pytest.approx(
        (1 - quality) / state.liquid.mass_density_kg_m3 + quality / state.vapor.mass_density_kg_m3)
    assert state.enthalpy_J_kg == pytest.approx(
        state.internal_energy_J_kg + pressure * state.specific_volume_m3_kg, rel=2e-10)
    for target in ("enthalpy_J_kg", "entropy_J_kg_K"):
        recovered = water_if97_at_pressure(pressure, **{target: getattr(state, target)})
        assert recovered.vapor_mass_fraction == pytest.approx(quality, abs=1e-9)
        assert recovered.temperature_K == pytest.approx(temperature, abs=1e-8)
    payload = state.to_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert "cp_J_kg_K" not in payload and "speed_of_sound_m_s" not in payload
    if temperature == 623.15:
        assert (state.liquid.region, state.vapor.region) == (1, 2)


@pytest.mark.parametrize("temperature,pressure,phase", [
    (300, 1e6, "vapor"), (700, 1e6, "liquid"),
    (640, 22e6, "vapor"), (640, 19e6, "liquid"),
    (650, 25e6, "liquid"), (500, 1e6, "mist"),
])
def test_incompatible_phase_is_rejected(temperature, pressure, phase):
    with pytest.raises(ValueError, match="phase"):
        water_if97_tp(temperature, pressure, phase=phase)


@pytest.mark.parametrize("quality", [-.1, 1.1, True, float("nan"), float("inf"), None])
def test_invalid_quality_is_not_clamped(quality):
    with pytest.raises(ValueError, match="quality"):
        water_if97_saturation(1e6, quality)


def test_critical_response_and_no_convergence_are_explicit(monkeypatch):
    with pytest.raises(ValueError, match="critical"):
        water_if97_tp(647.096, 22.064e6)
    with pytest.raises(ValueError, match="subcritical"):
        water_if97_saturation(22.064e6, .5)
    monkeypatch.setattr("chemworld.physchem.water_if97_solvers.solve_root",
                        lambda *args, **kwargs: RootResult(None, False, "injected failure", {}))
    with pytest.raises(ValueError, match="injected failure"):
        water_if97_tp(650, 25e6)


def test_published_caloric_gap_is_not_bridged_by_extrapolation():
    pressure = 50e6
    left, right = (_gibbs_state(1073.15, pressure, region) for region in (2, 5))
    assert left.enthalpy_J_kg < right.enthalpy_J_kg
    target = (left.enthalpy_J_kg + right.enthalpy_J_kg) / 2
    with pytest.raises(ValueError, match="no accepted state"):
        water_if97_at_pressure(pressure, enthalpy_J_kg=target)


def test_published_caloric_overlap_requires_explicit_region():
    pressure = 10e6
    left, right = (_gibbs_state(1073.15, pressure, region) for region in (2, 5))
    assert left.enthalpy_J_kg > right.enthalpy_J_kg
    target = (left.enthalpy_J_kg + right.enthalpy_J_kg) / 2
    with pytest.raises(ValueError, match="ambiguous"):
        water_if97_at_pressure(pressure, enthalpy_J_kg=target)
    low = water_if97_at_pressure(pressure, enthalpy_J_kg=target, region=2)
    high = water_if97_at_pressure(pressure, enthalpy_J_kg=target, region=5)
    assert low.temperature_K < 1073.15 < high.temperature_K
    for state in (low, high):
        assert state.enthalpy_J_kg == pytest.approx(target, rel=2e-11)


@pytest.mark.parametrize("arguments", [
    {}, {"enthalpy_J_kg": 1, "entropy_J_kg_K": 1}, {"enthalpy_J_kg": True},
    {"entropy_J_kg_K": float("nan")}, {"enthalpy_J_kg": -1e8},
    {"entropy_J_kg_K": 1e8}, {"enthalpy_J_kg": 1e6, "region": 4},
    {"enthalpy_J_kg": 1e6, "region": True},
])
def test_invalid_or_unattainable_caloric_state_is_rejected(arguments):
    with pytest.raises(ValueError):
        water_if97_at_pressure(1e6, **arguments)


def test_coexistence_and_region_hint_cannot_conflict():
    state = water_if97_saturation(1e6, .5)
    with pytest.raises(ValueError, match="conflicts"):
        water_if97_at_pressure(1e6, enthalpy_J_kg=state.enthalpy_J_kg, region=1)
    endpoint = water_if97_at_pressure(
        1e6, enthalpy_J_kg=state.liquid.enthalpy_J_kg, region=1)
    assert endpoint.homogeneous == state.liquid
    with pytest.raises(ValueError, match="cannot also own"):
        IF97Equilibrium(homogeneous=state.liquid, liquid=state.liquid)
    with pytest.raises(ValueError, match="share T and P"):
        IF97Equilibrium(liquid=state.liquid, vapor=water_if97_tp(700, 1e6),
                        vapor_mass_fraction=.5)
