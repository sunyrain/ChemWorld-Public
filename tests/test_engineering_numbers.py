import pytest

from chemworld.foundation.units import convert_value
from chemworld.physchem.engineering_numbers import (
    convert_viscosity,
    current_fluid_similarity,
    engineering_group,
    engineering_resistance_conversion,
    ideal_gas_sound_speed,
    local_gravity,
)
from chemworld.physchem.transport import FluidState, PipeSpec, pipe_pressure_drop


def test_current_fluid_groups_agree_with_actual_transport_and_thermal_mass_limits():
    fluid = FluidState(1000, 0.001, 4000, 0.5)
    groups = current_fluid_similarity(
        fluid, velocity_m_s=0.02, length_m=0.01, mass_diffusivity_m2_s=1e-9
    )
    pipe = PipeSpec(1, 0.01)
    actual = pipe_pressure_drop(pipe, fluid, volumetric_flow_m3_s=pipe.area_m2 * 0.02)
    assert groups["Reynolds"] == actual.reynolds
    assert groups["Prandtl"] == 8
    assert groups["Peclet_mass"] == pytest.approx(groups["Reynolds"] * groups["Schmidt"])
    assert groups["Lewis"] == pytest.approx(groups["Schmidt"] / groups["Prandtl"])
    assert current_fluid_similarity(fluid, velocity_m_s=0, length_m=0.01)["Reynolds"] == 0


def test_signs_optional_property_paths_and_invalid_denominators():
    assert engineering_group("Archimedes", L=0.01, rhof=1000, rhop=800, mu=0.001) < 0
    assert engineering_group("Cavitation", P=1e5, Psat=2e5, rho=1000, V=1) < 0
    assert engineering_group("Prandtl", Cp=4000, k=0.5, mu=0.001) == pytest.approx(
        engineering_group("Prandtl", nu=1e-6, alpha=1.25e-7)
    )
    with pytest.raises(ValueError, match="inconsistent"):
        engineering_group("Reynolds", V=1, D=0.1, rho=1000, mu=0.001, nu=0.1)
    with pytest.raises(ValueError, match="unknown parameter"):
        engineering_group("Mach", V=1, c=300, density=1)
    with pytest.raises(ValueError, match="contrast"):
        engineering_group("Confinement", D=0.1, rhol=1000, rhog=1000, sigma=0.07)


def test_original_temperature_units_and_darcy_head_conversions():
    assert convert_value(32, "degF", "degC") == pytest.approx(0, abs=1e-12)
    assert convert_value(491.67, "degR", "K") == pytest.approx(273.15)
    resistance = engineering_resistance_conversion("K_from_f", fd=0.02, L=10, D=0.1)
    pressure = engineering_resistance_conversion("dP_from_K", K=resistance, rho=1000, V=2)
    assert pressure == 4000
    head = engineering_resistance_conversion("head_from_K", K=resistance, V=2)
    assert engineering_resistance_conversion("P_from_head", head=head, rho=1000) == pytest.approx(
        pressure
    )
    assert convert_viscosity(density_kg_m3=1000, dynamic_Pa_s=0.001) == 1e-6
    assert 9.7 < local_gravity(45, 2000) < 9.9
    assert 300 < ideal_gas_sound_speed(300, 1.4, 0.029) < 400
