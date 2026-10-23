import json
from dataclasses import replace

import pytest

from chemworld.physchem.gas_control_valves import (
    GasControlValveSpec,
    gas_valve_flow,
    gas_valve_outlet_pressure,
    gas_valve_piping_factors,
    size_gas_control_valve,
)
from chemworld.physchem.gas_flow import IsothermalGasState


def spec(reducers=False):
    return GasControlValveSpec(
        20,
        0.05,
        0.08 if reducers else 0.05,
        0.1 if reducers else 0.05,
        0.85,
        0.42,
        0.6,
        "Declared IEC engineering fixture",
    )


@pytest.mark.parametrize("reducers", [False, True])
def test_installed_flow_mass_pressure_inverse_and_sizing(reducers):
    valve, gas = spec(reducers), IsothermalGasState(300, 287.05, 0.98)
    args = {"viscosity_Pa_s": 1e-5, "heat_capacity_ratio": 1.4, "upstream_pressure_Pa": 680000}
    result = gas_valve_flow(valve, gas, **args, downstream_pressure_Pa=600000)
    assert not result.choked and result.regime == "turbulent"
    assert result.inlet_flow_m3_s * gas.density(680000) == pytest.approx(result.mass_flow_kg_s)
    assert result.reference_flow_m3_s * 101325 / (287.05 * 273.15) == pytest.approx(
        result.mass_flow_kg_s
    )
    inverse = gas_valve_outlet_pressure(
        valve, gas, **args, reference_flow_m3_s=result.reference_flow_m3_s
    )
    assert inverse.downstream_pressure_Pa == pytest.approx(600000)
    selected = size_gas_control_valve(
        valve,
        gas,
        **args,
        downstream_pressure_Pa=600000,
        reference_flow_m3_s=result.reference_flow_m3_s,
    )
    assert selected.kv_m3_h == pytest.approx(valve.kv_m3_h)
    assert GasControlValveSpec.from_dict(json.loads(json.dumps(valve.to_dict()))) == valve


def test_installed_choked_plateau_and_capacity_inverse():
    valve, gas = spec(True), IsothermalGasState(300, 287.05, 0.98)
    args = {"viscosity_Pa_s": 1e-5, "heat_capacity_ratio": 1.4, "upstream_pressure_Pa": 680000}
    _, xtp = gas_valve_piping_factors(valve)
    pcrit = 680000 * (1 - xtp)
    at = gas_valve_flow(valve, gas, **args, downstream_pressure_Pa=pcrit)
    below = gas_valve_flow(valve, gas, **args, downstream_pressure_Pa=0)
    assert below.choked and at.choked and at.expansion_factor == pytest.approx(2 / 3)
    assert below.reference_flow_m3_s == pytest.approx(at.reference_flow_m3_s)
    assert gas_valve_outlet_pressure(
        valve, gas, **args, reference_flow_m3_s=at.reference_flow_m3_s
    ).downstream_pressure_Pa == pytest.approx(pcrit)
    with pytest.raises(ValueError, match="capacity"):
        gas_valve_outlet_pressure(
            valve, gas, **args, reference_flow_m3_s=2 * at.reference_flow_m3_s
        )


def test_low_re_closed_and_legacy_basis():
    gas = IsothermalGasState(320, 8.31446261815324 / 0.03995)
    valve = GasControlValveSpec(
        0.02, 0.015, 0.015, 0.015, 0.98, 0.07, 0.8, "IEC small needle example"
    )
    args = {
        "viscosity_Pa_s": 5.625e-5,
        "heat_capacity_ratio": 1.67,
        "upstream_pressure_Pa": 280000,
        "downstream_pressure_Pa": 130000,
    }
    result = size_gas_control_valve(valve, gas, **args, reference_flow_m3_s=0.46 / 3600)
    assert result.laminar and result.reynolds_factor < 1
    installed = gas_valve_flow(replace(valve, kv_m3_h=result.kv_m3_h), gas, **args)
    assert installed.reference_flow_m3_s > 0.46 / 3600
    assert installed.regime == "laminar/transitional"
    assert gas_valve_flow(replace(valve, is_open=False), gas, **args).mass_flow_kg_s == 0
    with pytest.raises(ValueError, match="attached reducers"):
        size_gas_control_valve(
            replace(valve, inlet_diameter_m=0.02), gas, **args, reference_flow_m3_s=0.46 / 3600
        )
    with pytest.raises(ValueError, match="installed expansion"):
        gas_valve_outlet_pressure(
            replace(valve, expansion_basis="bare_valve_reference"),
            gas,
            viscosity_Pa_s=args["viscosity_Pa_s"],
            heat_capacity_ratio=args["heat_capacity_ratio"],
            upstream_pressure_Pa=280000,
            reference_flow_m3_s=1e-4,
        )
