from math import exp

import pytest

from chemworld.physchem.msis_atmosphere import MSISAtmosphereState, MSISParameters
from chemworld.physchem.msis_profiles import InverseTemperatureProfile
from chemworld.physchem.transport import PipeSpec, pipe_pressure_drop


def test_inverse_temperature_profile_matches_spherical_isothermal_hydrostatics():
    radius, gravity, temperature, mass = 6370.0, 980.0, 250.0, 28.0
    profile = InverseTemperatureProfile(
        (75, 55, 30, 0), (temperature,) * 4, (0, 0), radius, gravity
    )
    initial = 1e15
    for height in (75, 72.5, 32.5, 10, 0):
        potential = (height - 75) * (radius + 75) / (radius + height)
        exponent = -mass * gravity / (1 + 75 / radius) ** 2 * potential / (831.4 * temperature)
        assert profile.density(height, initial, mass) == pytest.approx(
            initial * exp(exponent), rel=1e-13
        )
        assert profile.temperature(height) == pytest.approx(temperature, rel=1e-14)


def test_msis_state_consumes_caller_transport_in_original_fluid():
    state = MSISAtmosphereState(
        0, 280, 1000, (0, 0, 2e25, 0, 0, 0, 0, 0), 1.2, 1.2, 100000, (0, 0, 1, 0, 0, 0, 0)
    )
    fluid = state.fluid_state(1.8e-5, 1005, 0.026)
    result = pipe_pressure_drop(
        PipeSpec(1, 0.05, fittings_loss_coefficient=2), fluid, volumetric_flow_m3_s=0.01
    )
    assert result.pressure_drop_fittings_Pa == pytest.approx(1.2 * result.velocity_m_s**2)
    assert state.to_dict()["species"][-1] == "O_anomalous"


def test_msis_parameters_reject_incomplete_empirical_data():
    from chemworld.data.scientific_inputs import (
        InputSource,
        InputValue,
        QuantityColumn,
        ScientificRow,
        ScientificTable,
    )
    from chemworld.physchem.data_conflicts import DataSourceProvenance

    columns = tuple(
        QuantityColumn(n, "dimensionless", "parameter", (-100, 100), "synthetic")
        for n in ("index", "value")
    )
    source = InputSource(DataSourceProvenance("test", 0, "synthetic", "1", "test"), "synthetic")
    row = ScientificRow(
        "r",
        "test",
        "pt",
        "r",
        "parameters",
        {n: InputValue(0, "dimensionless", None, "synthetic") for n in ("index", "value")},
    )
    with pytest.raises(ValueError, match="all nine"):
        MSISParameters(ScientificTable("test", "parameters", columns, (source,), (row,)))
