import json

import pytest

from chemworld.physchem.gas_flow_thermo import (
    critical_gas_pressure,
    gas_efficiency_conversion,
    gas_polytropic_exponent,
    gas_stagnation_pressure,
    gas_stagnation_temperature,
    ideal_gas_nozzle,
    isentropic_gas_outlet_temperature,
    isentropic_gas_work_molar,
    isothermal_gas_work_molar,
    solve_isentropic_gas_work,
    stagnation_temperature_from_velocity,
)


@pytest.mark.parametrize("process,p2", [("compression", 1e6), ("expansion", 1e4)])
def test_work_efficiency_and_inverse(process, p2):
    args = {
        "inlet_pressure_Pa": 1e5,
        "outlet_pressure_Pa": p2,
        "inlet_temperature_K": 300,
        "heat_capacity_ratio": 1.4,
        "process": process,
    }
    ideal = isentropic_gas_work_molar(**args)
    actual = isentropic_gas_work_molar(**args, efficiency=0.8)
    assert actual > ideal
    for unknown in ("work_J_mol", "inlet_pressure_Pa", "outlet_pressure_Pa", "efficiency"):
        data = {**args, "work_J_mol": actual, "efficiency": 0.8}
        expected = data.pop(unknown)
        assert solve_isentropic_gas_work(**data) == pytest.approx(expected, rel=1e-12)
    t2 = isentropic_gas_outlet_temperature(**args, efficiency=0.8)
    assert t2 > 300 if process == "compression" else 0 < t2 < 300


@pytest.mark.parametrize("process,p2", [("compression", 1e6), ("expansion", 1e4)])
def test_efficiencies_roundtrip(process, p2):
    args = {
        "inlet_pressure_Pa": 1e5,
        "outlet_pressure_Pa": p2,
        "heat_capacity_ratio": 1.4,
        "process": process,
    }
    eta_s = gas_efficiency_conversion(**args, polytropic_efficiency=0.8)
    assert 0 < eta_s <= 1
    assert gas_efficiency_conversion(**args, isentropic_efficiency=eta_s) == pytest.approx(0.8)
    exponent = gas_polytropic_exponent(heat_capacity_ratio=1.4, efficiency=0.8, process=process)
    assert gas_polytropic_exponent(
        heat_capacity_ratio=1.4, exponent=exponent, process=process
    ) == pytest.approx(0.8)
    assert exponent > 1.4 if process == "compression" else 1 < exponent < 1.4


def test_zero_pressure_ratio_and_invalid_efficiency():
    args = {
        "inlet_pressure_Pa": 1e5,
        "outlet_pressure_Pa": 1e5,
        "inlet_temperature_K": 300,
        "heat_capacity_ratio": 1.4,
    }
    assert isentropic_gas_work_molar(**args) == 0
    assert (
        isothermal_gas_work_molar(inlet_pressure_Pa=1e5, outlet_pressure_Pa=1e5, temperature_K=300)
        == 0
    )
    assert (
        gas_efficiency_conversion(
            inlet_pressure_Pa=1e5,
            outlet_pressure_Pa=1e5,
            heat_capacity_ratio=1.4,
            polytropic_efficiency=0.8,
        )
        == 0.8
    )
    with pytest.raises(ValueError, match="indeterminate"):
        solve_isentropic_gas_work(**args, work_J_mol=0)
    with pytest.raises(ValueError, match="efficiency"):
        isentropic_gas_work_molar(**args, efficiency=1.1)
    with pytest.raises(ValueError, match="greater than one"):
        critical_gas_pressure(stagnation_pressure_Pa=1e5, heat_capacity_ratio=1)
    with pytest.raises(ValueError, match="must exceed"):
        gas_polytropic_exponent(heat_capacity_ratio=1.4, efficiency=0.1)


def test_stagnation_roundtrip():
    t0 = stagnation_temperature_from_velocity(
        static_temperature_K=280, velocity_m_s=350, heat_capacity_J_kg_K=1005
    )
    p0 = gas_stagnation_pressure(
        static_pressure_Pa=1e5,
        static_temperature_K=280,
        stagnation_temperature_K=t0,
        heat_capacity_ratio=1.4,
    )
    assert gas_stagnation_temperature(
        static_pressure_Pa=1e5,
        static_temperature_K=280,
        stagnation_pressure_Pa=p0,
        heat_capacity_ratio=1.4,
    ) == pytest.approx(t0)
    assert p0 > 1e5 and t0 > 280


def test_nozzle_branch_capacity_area_and_units():
    args = {
        "stagnation_pressure_Pa": 1e6,
        "stagnation_temperature_K": 300,
        "heat_capacity_ratio": 1.4,
        "specific_gas_constant_J_kg_K": 287.05,
        "discharge_coefficient": 0.9,
    }
    high = ideal_gas_nozzle(**args, downstream_pressure_Pa=9e5, area_m2=0.001)
    sonic = ideal_gas_nozzle(
        **args, downstream_pressure_Pa=high.critical_pressure_Pa, area_m2=0.001
    )
    low = ideal_gas_nozzle(**args, downstream_pressure_Pa=0, area_m2=0.001)
    assert not high.choked and sonic.choked and low.choked
    assert high.mass_flow_kg_s < sonic.mass_flow_kg_s == low.mass_flow_kg_s
    assert low.exit_pressure_Pa == low.critical_pressure_Pa
    assert low.exit_temperature_K == pytest.approx(250)
    inverse = ideal_gas_nozzle(**args, downstream_pressure_Pa=0, mass_flow_kg_s=low.mass_flow_kg_s)
    assert inverse.area_m2 == pytest.approx(0.001)
    assert json.loads(json.dumps(low.to_dict()))["choked"] is True
    doubled = ideal_gas_nozzle(
        **{**args, "stagnation_pressure_Pa": 2e6}, downstream_pressure_Pa=0, area_m2=0.001
    )
    assert doubled.mass_flow_kg_s == pytest.approx(2 * low.mass_flow_kg_s)


def test_nozzle_zero_flow_invalid_and_compressor_direction():
    args = {
        "stagnation_pressure_Pa": 1e5,
        "stagnation_temperature_K": 300,
        "heat_capacity_ratio": 1.4,
        "specific_gas_constant_J_kg_K": 287.05,
    }
    assert ideal_gas_nozzle(**args, downstream_pressure_Pa=1e5, area_m2=0.01).mass_flow_kg_s == 0
    with pytest.raises(ValueError, match="zero pressure"):
        ideal_gas_nozzle(**args, downstream_pressure_Pa=1e5, mass_flow_kg_s=1)
    with pytest.raises(ValueError, match="between zero"):
        ideal_gas_nozzle(**args, downstream_pressure_Pa=2e5, area_m2=0.01)
    with pytest.raises(ValueError, match="direction"):
        isentropic_gas_work_molar(
            inlet_pressure_Pa=1e5,
            outlet_pressure_Pa=1e4,
            inlet_temperature_K=300,
            heat_capacity_ratio=1.4,
        )
