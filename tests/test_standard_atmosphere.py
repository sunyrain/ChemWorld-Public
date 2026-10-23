from math import exp

import pytest

from chemworld.physchem.standard_atmosphere import (
    air_column_mass,
    standard_atmosphere,
    standard_pressure_column,
)
from chemworld.physchem.transport import PipeSpec, pipe_pressure_drop


def test_standard_layers_feed_current_fluid_and_temperature_convention():
    base = standard_atmosphere(5000)
    warm = standard_atmosphere(5000, 20)
    assert warm.pressure_Pa == base.pressure_Pa
    assert warm.density_kg_m3 < base.density_kg_m3
    pipe = PipeSpec(1, 0.05, fittings_loss_coefficient=2)
    report = pipe_pressure_drop(pipe, base.fluid_state(), volumetric_flow_m3_s=0.001)
    speed = report.velocity_m_s
    assert report.pressure_drop_fittings_Pa == pytest.approx(base.density_kg_m3 * speed**2)
    assert standard_atmosphere(0).pressure_Pa == 101325
    with pytest.raises(ValueError, match="domain"):
        standard_atmosphere(90000)
    assert standard_atmosphere(90000, extrapolate=True).extrapolated


def test_signed_pressure_column_and_vertical_horizon():
    state = standard_atmosphere(1000)
    up = standard_pressure_column(state.temperature_K, state.pressure_Pa, 100)
    down = standard_pressure_column(state.temperature_K, state.pressure_Pa, -100)
    assert down < 0 < up
    assert up == pytest.approx(state.pressure_Pa - standard_atmosphere(1100).pressure_Pa, rel=1e-9)

    def density(height):
        return 1.225 * exp(-height / 8000)

    vertical = air_column_mass(density, 90)
    horizon = air_column_mass(density, 0)
    assert vertical.mass_kg_m2 == pytest.approx(9800 * (1 - exp(-10.8)))
    assert horizon.mass_kg_m2 > 20 * vertical.mass_kg_m2
    with pytest.raises(ValueError, match="nonnegative"):
        air_column_mass(lambda h: 1 if h == 0 else -1, 20)
