import json

import pytest

from chemworld.physchem.jet_pumps import (
    JET_VARIABLE_UNITS,
    LiquidJetPumpModel,
    jet_pump_report,
    solve_jet_pump_inlet,
    solve_liquid_jet_pump,
    vacuum_air_leakage,
)
from chemworld.physchem.transport import FluidState


def state(model):
    values = dict(zip(tuple(JET_VARIABLE_UNITS)[:4], (0.02, 0.045, 0.002, 0.001), strict=True))
    values["suction_pressure_Pa"] = 1e5
    values["motive_pressure_Pa"] = 1e5 + model.inlet_pressure_difference(0.02, 0.045, 0.002, 0.001)
    num, den = model.pressure_ratio_terms(0.02, 0.045, 0.002, 0.001)
    values["discharge_pressure_Pa"] = (
        values["motive_pressure_Pa"] * num + values["suction_pressure_Pa"] * den
    ) / (num + den)
    return values


def test_explicit_source_disagreement_and_current_fluid_mass_report():
    model = LiquidJetPumpModel(FluidState(1000, 0.001), FluidState(800, 0.001))
    model = LiquidJetPumpModel.from_dict(json.loads(json.dumps(model.to_dict())))
    values = state(model)
    result = jet_pump_report(model, values)
    assert result["outlet_mass_flow_kg_s"] == 2.8
    assert result["outlet_density_kg_m3"] == pytest.approx(2.8 / 0.003)
    assert result["inlet_residual_Pa"] == pytest.approx(0, abs=1e-10)
    assert result["pressure_ratio_residual"] == pytest.approx(0, abs=1e-12)
    assert result["warnings"]
    ref = LiquidJetPumpModel(
        model.motive_fluid, model.suction_fluid, retracted_loss_convention="reference_code"
    )
    assert state(ref)["discharge_pressure_Pa"] != values["discharge_pressure_Pa"]
    assert values == state(model)


def test_all_inlet_inverses_and_explicit_two_variable_branch():
    model = LiquidJetPumpModel(FluidState(1000, 0.001), FluidState(800, 0.001))
    values = state(model)
    inlet = {k: v for k, v in values.items() if k != "discharge_pressure_Pa"}
    for key in inlet:
        assert solve_jet_pump_inlet(model, **{**inlet, key: None}) == pytest.approx(inlet[key])
    missing = ("nozzle_diameter_m", "motive_pressure_Pa")
    report = solve_liquid_jet_pump(
        model,
        {k: v for k, v in values.items() if k not in missing},
        initial={k: values[k] * 1.02 for k in missing},
        bounds={k: (values[k] * 0.8, values[k] * 1.2) for k in missing},
    )
    for key in missing:
        assert report[key] == pytest.approx(values[key], rel=1e-8)
    assert report["solver"]["iterations"] > 0
    with pytest.raises(ValueError, match="nozzle"):
        model.inlet_pressure_difference(0.05, 0.045, 0.002, 0.001)
    with pytest.raises(ValueError, match="pumping branch"):
        jet_pump_report(model, {**values, "discharge_pressure_Pa": 1})


def test_vacuum_limits_and_source_table_requirement():
    report = vacuum_air_leakage("Ryans-Croll", pressure_Pa=10000, volume_m3=10)
    assert report["mass_flow_kg_s"] > 0 and report["volume_m3"] == 10
    assert vacuum_air_leakage("Seider", pressure_Pa=10000, volume_m3=10)["warnings"]
    with pytest.raises(ValueError, match="below"):
        vacuum_air_leakage("Ryans-Croll", pressure_Pa=101326, volume_m3=10)
    with pytest.raises(ValueError, match="explicit"):
        vacuum_air_leakage("HEI2633", pressure_Pa=10000, volume_m3=10)


def test_mass_flow_units_keep_rate_separate_from_mass():
    from chemworld.foundation.units import convert_value

    assert convert_value(3600, "kg/h", "kg/s") == 1
    assert convert_value(1000, "g/s", "kg/s") == 1
    with pytest.raises(ValueError):
        convert_value(1, "kg", "kg/s")
