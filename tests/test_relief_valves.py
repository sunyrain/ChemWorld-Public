import json
from math import exp, log10

import pytest
from tests.test_tabulated_functions import function

from chemworld.physchem.relief_valves import (
    API526_ORIFICES_M2,
    api520_critical_coefficient,
    api520_gas_backpressure_factor,
    api520_liquid_backpressure_factor,
    api520_steam_superheat_factor,
    api520_subcritical_factor,
    api520_viscosity_factor,
    api521_flare_noise,
    api526_select_orifice,
    size_gas_relief_valve,
    size_liquid_relief_valve,
    size_steam_relief_valve,
    vdi3732_flare_noise,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar


def steam_table():
    return function(
        {"temperature_K": (500, 900), "pressure_Pa": (1e5, 1e6)},
        lambda t, p: 1 - (t - 400) / 2000 - p / 1e9,
        output_unit="dimensionless",
        input_units={"temperature_K": "K", "pressure_Pa": "Pa"},
    )


def test_critical_coefficient_and_same_pressure_limits():
    expected = 0.03948 * exp(-0.5)
    assert api520_critical_coefficient(1) == pytest.approx(expected)
    assert api520_critical_coefficient(1 + 1e-12) == pytest.approx(expected)
    for gamma in (1, 1.000000001, 1.4):
        assert (
            api520_subcritical_factor(
                heat_capacity_ratio=gamma, upstream_pressure_Pa=1e5, downstream_pressure_Pa=1e5
            )
            == 1
        )
        assert api520_subcritical_factor(
            heat_capacity_ratio=gamma, upstream_pressure_Pa=1e5, downstream_pressure_Pa=1e5 - 1e-5
        ) == pytest.approx(1)


@pytest.mark.parametrize("pressure_ratio", [0.1, 0.8])
def test_gas_area_mass_inverse_and_branch(pressure_ratio):
    args = {
        "temperature_K": 400,
        "compressibility_factor": 0.95,
        "molar_mass_g_mol": 28.97,
        "heat_capacity_ratio": 1.4,
        "upstream_pressure_Pa": 1e6,
        "downstream_pressure_Pa": 1e6 * pressure_ratio,
    }
    result = size_gas_relief_valve(**args, mass_flow_kg_s=10)
    assert result.choked == (pressure_ratio == 0.1)
    assert size_gas_relief_valve(**args, area_m2=result.area_m2).mass_flow_kg_s == pytest.approx(10)
    revised = size_gas_relief_valve(**args, mass_flow_kg_s=10, backpressure_factor=0.8)
    assert revised.area_m2 == pytest.approx(
        result.area_m2 / 0.8 if result.choked else result.area_m2
    )


@pytest.mark.parametrize("edition", ["7E", "10E"])
def test_liquid_actual_two_step_and_mass_inverse(edition):
    args = {
        "density_kg_m3": 900,
        "upstream_pressure_Pa": 1.8e6,
        "downstream_pressure_Pa": 2e5,
        "overpressure": 0.1,
        "backpressure_factor": 0.97,
        "viscosity_Pa_s": 0.2,
        "edition": edition,
    }
    result = size_liquid_relief_valve(**args, mass_flow_kg_s=30)
    assert result.viscosity_basis_reynolds > 80
    assert result.viscosity_factor < 1
    assert size_liquid_relief_valve(**args, area_m2=result.area_m2).mass_flow_kg_s == pytest.approx(
        30
    )
    fixed = size_liquid_relief_valve(
        **args, mass_flow_kg_s=30, viscosity_factor=result.viscosity_factor
    )
    assert fixed.area_m2 == result.area_m2


def test_steam_sourced_table_roundtrip_domain_and_mass_inverse():
    table = steam_table()
    restored = TabulatedScalar.from_dict(json.loads(json.dumps(table.to_dict())))
    args = {
        "temperature_K": 700,
        "upstream_pressure_Pa": 5e5,
        "edition": "10E",
        "steam_phase": "superheated",
        "superheat_table": restored,
    }
    result = size_steam_relief_valve(**args, mass_flow_kg_s=10)
    assert result.steam_superheat_factor == pytest.approx(0.8495)
    assert size_steam_relief_valve(**args, area_m2=result.area_m2).mass_flow_kg_s == pytest.approx(
        10
    )
    with pytest.raises(ValueError, match="explicit sourced"):
        api520_steam_superheat_factor(temperature_K=700, upstream_pressure_Pa=5e5, edition="10E")
    with pytest.raises(ValueError, match="data domain"):
        api520_steam_superheat_factor(
            temperature_K=910, upstream_pressure_Pa=5e5, edition="10E", table=table
        )
    assert (
        api520_steam_superheat_factor(temperature_K=450, upstream_pressure_Pa=5e5, edition="10E")
        == 1
    )
    with pytest.raises(ValueError, match="dry saturated"):
        size_steam_relief_valve(**{**args, "steam_phase": "wet"}, mass_flow_kg_s=10)


def test_backpressure_domain_and_explicit_table_requirement():
    assert api520_gas_backpressure_factor(set_pressure_Pa=1101325, backpressure_Pa=301325) == 1
    assert api520_liquid_backpressure_factor(set_pressure_Pa=1101325, backpressure_Pa=201325) == 1
    with pytest.raises(ValueError, match="50 percent"):
        api520_gas_backpressure_factor(set_pressure_Pa=1101325, backpressure_Pa=701325)
    with pytest.raises(ValueError, match="explicit sourced"):
        api520_liquid_backpressure_factor(set_pressure_Pa=1101325, backpressure_Pa=501325)
    table = function(
        {"gauge_backpressure_fraction": (0.15, 0.5)},
        lambda b: 1 - 0.5 * b,
        output_unit="dimensionless",
        input_units={"gauge_backpressure_fraction": "dimensionless"},
    )
    assert api520_liquid_backpressure_factor(
        set_pressure_Pa=1101325, backpressure_Pa=501325, table=table
    ) == pytest.approx(0.8)
    with pytest.raises(ValueError, match="> 80"):
        api520_viscosity_factor(80)


def test_standard_size_and_noise_units_distance():
    for letter, area in API526_ORIFICES_M2:
        assert api526_select_orifice(area) == (letter, area)
        assert api526_select_orifice(0.999 * area) == (letter, area)
    with pytest.raises(ValueError, match="largest"):
        api526_select_orifice(1)
    args = {
        "mass_flow_kg_s": 10,
        "upstream_pressure_Pa": 1e6,
        "ambient_pressure_Pa": 101325,
        "sound_speed_m_s": 350,
    }
    assert api521_flare_noise(**args, distance_m=60) == pytest.approx(
        api521_flare_noise(**args, distance_m=30) - 20 * log10(2)
    )
    assert vdi3732_flare_noise(1 / 3.6, configuration="elevated") == 112
    assert vdi3732_flare_noise(1 / 3.6, configuration="ground") == 100


def test_declared_saturated_steam_and_explicit_table_clip():
    saturated = size_steam_relief_valve(
        temperature_K=600,
        upstream_pressure_Pa=1e7,
        edition="10E",
        steam_phase="dry_saturated",
        mass_flow_kg_s=10,
    )
    assert saturated.steam_superheat_factor == 1
    table = steam_table()
    assert api520_steam_superheat_factor(
        temperature_K=910,
        upstream_pressure_Pa=5e5,
        edition="10E",
        table=table,
        table_boundary_policy="clip",
    ) == pytest.approx(0.7495)
    with pytest.raises(ValueError, match=r"922\.15"):
        api520_steam_superheat_factor(
            temperature_K=923,
            upstream_pressure_Pa=5e5,
            edition="10E",
            table=table,
            table_boundary_policy="clip",
        )
