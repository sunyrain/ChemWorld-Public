"""Electrical algebraic constraints consume the existing SPM and coupled ledger."""

from dataclasses import replace

import pytest

from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    ParticleSpec,
    advance_battery,
    battery_control_current,
    battery_readout,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram


def small_cell(**kwargs):
    particle = ParticleSpec(cells=6)
    return BatterySPMSpec(
        negative=particle, positive=particle, thermal=BatteryThermalSpec(), **kwargs
    )


@pytest.mark.parametrize(
    "unit,value,limit,mode",
    [
        ("A", 0.1, 2, "constant_current"),
        ("V", 3.7, 2, "constant_voltage"),
        ("V", 3.7, 0.01, "positive_current_limit"),
        ("V", 4.1, 0.01, "negative_current_limit"),
    ],
)
def test_residual_current_voltage_and_source_limits_agree_with_same_equation_ode(
    unit,
    value,
    limit,
    mode,
):
    spec = small_cell(maximum_current_A=limit)
    drive = ScalarControlProgram(((0, value),), unit)
    before = spec.initial_state()
    result = advance_battery(spec, before, drive, 5, evaluation_times_s=[0, 5])
    expected = advance_battery(
        replace(spec, numerical_form="ode"), before, drive, 5, evaluation_times_s=[0, 5]
    )
    assert result.final_state.amounts_mol == pytest.approx(
        expected.final_state.amounts_mol, rel=1e-7, abs=1e-12
    )
    assert result.final_state.coordinates == pytest.approx(
        expected.final_state.coordinates, rel=1e-7, abs=1e-8
    )
    assert result.final_state.temperature_K == pytest.approx(
        expected.final_state.temperature_K, abs=1e-7
    )
    assert result.solver_diagnostic["battery_control_mode"] == mode
    current = battery_control_current(spec, result.final_state, drive)
    reading = battery_readout(spec, result.final_state, current)
    algebraic = result.solver_diagnostic["algebraic_values"]
    assert algebraic["battery:current_A"] == pytest.approx(current, abs=1e-10)
    assert algebraic["battery:voltage_V"] == pytest.approx(reading.voltage_V, abs=1e-10)
    if mode == "constant_voltage":
        assert reading.voltage_V == pytest.approx(value, abs=1e-10)
    assert abs(current) <= limit
    assert not result.final_state.algebraic_values
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-6)


def test_voltage_ramp_records_actual_current_limit_transitions_without_extra_time_loop():
    spec = small_cell(maximum_current_A=0.05)
    drive = ScalarControlProgram(((0, 3.5), (10, 4.0)), "V")
    result = advance_battery(spec, spec.initial_state(), drive, 10, evaluation_times_s=[0, 5, 10])
    assert result.actual_duration_s == 10 and result.stopping_event is None
    transitions = result.solver_diagnostic["events"]
    assert [event["name"] for event in transitions] == [
        "battery:cv:positive-limit",
        "battery:cv:negative-limit",
    ]
    assert 0 < transitions[0]["time_s"] < transitions[1]["time_s"] < 10
    assert result.solver_diagnostic["battery_control_mode"] == "negative_current_limit"
    assert result.solver_diagnostic["solver_calls"] == 1
    assert sum(result.final_state.amounts_mol.values()) == pytest.approx(0.03, abs=1e-12)


def test_zero_time_residual_check_has_consistent_voltage_and_no_integration():
    spec = small_cell()
    before = spec.initial_state()
    result = advance_battery(spec, before, ScalarControlProgram(((0, 0.1),), "A"), 0)
    assert result.final_state == before
    assert result.solver_diagnostic["solver_calls"] == 0
    values = result.solver_diagnostic["algebraic_values"]
    assert values["battery:current_A"] == pytest.approx(0.1, abs=1e-12)
    assert values["battery:voltage_V"] == pytest.approx(
        battery_readout(spec, before, 0.1).voltage_V,
        abs=1e-12,
    )
