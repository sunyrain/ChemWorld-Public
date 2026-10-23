from math import log10

import pytest

from chemworld.physchem.valve_noise import (
    A_WEIGHT_DB,
    ValveAcousticGeometry,
    gas_control_valve_noise,
    liquid_control_valve_noise,
)


def test_liquid_published_example_spectral_sum_and_flashing_rejection():
    geometry = ValveAcousticGeometry(
        77.848, 0.1, 0.1071, 0.0036, 0.92, 0.42, "IEC published example", air_density_kg_m3=1.293
    )
    args = {
        "mass_flow_kg_s": 40,
        "upstream_pressure_Pa": 1e6,
        "downstream_pressure_Pa": 6.5e5,
        "saturation_pressure_Pa": 2320,
        "density_kg_m3": 997,
        "sound_speed_m_s": 1400,
    }
    result = liquid_control_valve_noise(geometry, **args)
    assert result.a_weighted_level_dB == pytest.approx(81.58200097996, abs=2e-9)
    assert len(result.band_levels_dB) == 33
    total = 10 * log10(
        sum(
            10 ** ((level + weight) / 10)
            for level, weight in zip(result.band_levels_dB, A_WEIGHT_DB, strict=True)
        )
    )
    assert total == pytest.approx(result.a_weighted_level_dB)
    assert result.regime == "cavitation"
    quiet = liquid_control_valve_noise(geometry, **{**args, "downstream_pressure_Pa": 9.99e5})
    assert quiet.regime == "turbulence"
    with pytest.raises(ValueError, match="flashing"):
        liquid_control_valve_noise(geometry, **{**args, "downstream_pressure_Pa": 1000})


def test_gas_published_example_outlet_density_and_expander():
    geometry = ValveAcousticGeometry(
        77.85,
        0.1,
        0.2031,
        0.008,
        0.792 / 0.98,
        0.296,
        "IEC published installed recovery",
        pipe_density_kg_m3=8000,
        air_density_kg_m3=1.293,
    )
    args = {
        "mass_flow_kg_s": 2.22,
        "upstream_pressure_Pa": 1e6,
        "downstream_pressure_Pa": 7.2e5,
        "inlet_temperature_K": 450,
        "inlet_density_kg_m3": 5.3,
        "heat_capacity_ratio": 1.22,
        "molar_mass_g_mol": 19.8,
    }
    result = gas_control_valve_noise(geometry, **args)
    assert result.a_weighted_level_dB == pytest.approx(91.67702674629604, abs=2e-9)
    cool = gas_control_valve_noise(geometry, **args, outlet_temperature_K=400)
    assert cool.outlet_density_kg_m3 == pytest.approx(5.3 * 0.72 * 450 / 400)
    source_density = gas_control_valve_noise(
        geometry, **args, outlet_temperature_K=400, outlet_density_kg_m3=5.3 * 0.72
    )
    assert cool.a_weighted_level_dB != source_density.a_weighted_level_dB
    expanded = gas_control_valve_noise(geometry, **{**args, "downstream_pressure_Pa": 1e4})
    assert expanded.regime == "aerodynamic_5"
    assert expanded.outlet_mach > 0.3 and expanded.expander_acoustic_power_W > 0
