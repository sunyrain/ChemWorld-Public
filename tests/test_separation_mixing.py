import json
from math import exp, sqrt

import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.mixing_engineering import (
    agitator_mixing_time,
    ribbon_power_reynolds,
    static_mixer_performance,
    tee_mixing_cov,
)
from chemworld.physchem.separator_capacity import (
    PSI_PA,
    SeparatorCapacityCurve,
    separator_demister_capacity,
    separator_gas_capacity,
    separator_watkins_capacity,
)
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.physchem.transport import FluidState


def test_separator_source_chart_roundtrip_domain_and_current_flow_area():
    source = InputSource(
        DataSourceProvenance(
            "synthetic", 0, "Self-authored cubic capacity chart", "1", "Synthetic, not measured"
        ),
        "synthetic",
    )
    rows = tuple(
        ScientificRow(
            str(i),
            "synthetic",
            "curve",
            str(i),
            "parameters",
            {
                "ln_factor": InputValue(x, "dimensionless", None, "Synthetic"),
                "ln_K": InputValue(-2 + 0.1 * x**3, "dimensionless", None, "Synthetic"),
            },
        )
        for i, x in enumerate((-5, -3, -1, 1))
    )
    columns = tuple(
        QuantityColumn(k, "dimensionless", "parameter", (-100, 100), "Logged")
        for k in ("ln_factor", "ln_K")
    )
    curve = SeparatorCapacityCurve(
        TabulatedScalar(
            ScientificTable("chart", "parameters", columns, (source,), rows),
            (("ln_factor", "ln_factor", "dimensionless"),),
            "ln_K",
            "dimensionless",
        )
    )
    copy = SeparatorCapacityCurve.from_dict(json.loads(json.dumps(curve.to_dict())))
    assert copy.value(exp(-2)) == pytest.approx(0.3048 * exp(-2.8))
    with pytest.raises(ValueError, match="domain"):
        copy.value(exp(2))
    liquid, gas = FluidState(1000, 0.001), FluidState(2, 1.8e-5)
    quality = sqrt(0.002) / (exp(-2) + sqrt(0.002))
    k = separator_watkins_capacity(liquid, gas, vapor_mass_fraction=quality, chart=copy)
    result = separator_gas_capacity(
        liquid, gas, capacity_factor_m_s=k, gas_mass_flow_kg_s=4, design_fraction=0.8
    )
    assert result.gas_volume_flow_m3_s == 2
    assert result.required_gas_area_m2 * 0.8 * result.maximum_gas_velocity_m_s == pytest.approx(2)
    with pytest.raises(ValueError, match="explicit source"):
        separator_watkins_capacity(liquid, gas, vapor_mass_fraction=quality)
    with pytest.raises(ValueError, match="denser"):
        separator_gas_capacity(gas, liquid, capacity_factor_m_s=k)
    with pytest.raises(ValueError, match="domain"):
        separator_demister_capacity(0.5 * PSI_PA)
    assert separator_demister_capacity(
        0.5 * PSI_PA, endpoint_policy="clip"
    ) == separator_demister_capacity(PSI_PA)


def test_mixing_si_power_and_current_hydraulic_power():
    kwargs = {
        "rotation_frequency_Hz": 2,
        "fluid_power_W": 400,
        "tank_diameter_m": 2,
        "liquid_height_m": 1.5,
        "impeller_diameter_m": 1,
    }
    result = agitator_mixing_time(FluidState(1000, 0.001), **kwargs)
    assert result.power_number == 0.05
    compatible = agitator_mixing_time(
        FluidState(1000, 0.001), **kwargs, power_convention="reference_g"
    )
    assert compatible.power_number / result.power_number == pytest.approx(9.80665)
    assert "dimensionally inconsistent" in compatible.warnings[0]
    flow_kwargs = {
        "main_flow_m3_s": 0.001,
        "injected_flow_m3_s": 0.0002,
        "diameter_m": 0.1,
        "length_m": 1,
        "mixing_coefficient": 0.8,
        "manufacturer_loss_factor": 100,
    }
    first = static_mixer_performance(FluidState(1000, 0.001), **flow_kwargs)
    second = static_mixer_performance(FluidState(500, 0.0005), **flow_kwargs)
    assert second.reynolds == first.reynolds
    assert second.pressure_drop_Pa == first.pressure_drop_Pa / 2
    assert first.hydraulic_power_W == pytest.approx(first.pressure_drop_Pa * 0.0012)
    assert second.concentration_cov == first.concentration_cov
    with pytest.raises(ValueError, match="homogeneity"):
        agitator_mixing_time(FluidState(1000, 0.001), **kwargs, homogeneity=1)


def test_tee_source_ambiguity_and_ribbon_geometry_are_explicit():
    common = {"main_flow_m3_s": 1, "injected_flow_m3_s": 0.1, "main_diameter_m": 0.5}
    lower = tee_mixing_cov(**common, jet_diameter_m=0.05 / sqrt(0.7 * 0.9999))
    upper = tee_mixing_cov(**common, jet_diameter_m=0.05 / sqrt(0.7 * 1.0001))
    assert lower.branch == "lower" and upper.branch == "upper"
    assert upper.concentration_cov > lower.concentration_cov * 2
    assert "discontinuous" in upper.warnings[-1]
    with pytest.raises(ValueError, match="distance"):
        tee_mixing_cov(**common, length_diameters=0)
    with pytest.raises(ValueError, match="wall clearance"):
        ribbon_power_reynolds(
            impeller_diameter_m=1,
            ribbon_height_m=1,
            blades=2,
            pitch_m=1,
            blade_width_m=0.1,
            tank_diameter_m=1,
        )
