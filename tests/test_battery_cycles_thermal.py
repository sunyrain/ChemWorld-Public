"""Current/voltage control and independently checked battery free-energy balance."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    advance_battery,
    battery_control_current,
    battery_dissipation_w,
    battery_readout,
    battery_system,
)
from chemworld.physchem.coupled_system import assemble_coupled_rates
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram


def energy(spec, state):
    """Independent direct cell free-energy integral for the fixed synthetic OCV."""
    value = 0.0
    for name, p, a, b in (
        ("negative", spec.negative, 0.2, -0.1),
        ("positive", spec.positive, 4.2, -0.8),
    ):
        c = np.array([state.amounts_mol[k] for k in p.keys(name)]) / p.mesh.cell_volumes_m3
        value += float(
            np.dot(
                p.mesh.cell_volumes_m3,
                -F * (a * c + b * c * c / (2 * p.maximum_concentration_mol_m3)),
            )
        )
    return value + 100 * (state.temperature_K - spec.temperature_K)


@pytest.mark.parametrize("rate", [0.1, 0.5, 1])
@pytest.mark.parametrize("ua", [0, 0.1, 1])
def test_nine_current_cooling_conditions_close_actual_energy(rate, ua):
    spec = BatterySPMSpec(thermal=BatteryThermalSpec(loss_conductance_W_K=ua))
    initial = spec.initial_state()
    drive = ScalarControlProgram(((0, spec.current_at_c_rate(rate)),), "A")
    result = advance_battery(spec, initial, drive, 600)
    final = result.final_state
    assert result.outcome == "success"
    assert 298.15 < final.temperature_K < 300
    work = final.coordinates["delivered_work_J"]
    expected = -work - final.heat_loss_J + final.boundary_heat_J
    tolerance = 1e-6 + 1e-6 * abs(expected)
    assert energy(spec, final) - energy(spec, initial) == pytest.approx(expected, abs=tolerance)
    assert abs(result.energy_balance_residual_J) < tolerance
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-12
    )


def test_derived_dissipation_equals_instantaneous_heat_and_each_term_is_nonnegative():
    spec = BatterySPMSpec(thermal=BatteryThermalSpec(), series_resistance_ohm=0.05)
    for current in (0.3, -0.3, 0):
        drive = ScalarControlProgram(((0, current),), "A")
        final = advance_battery(spec, spec.initial_state(), drive, 300).final_state
        system = battery_system(spec, drive)
        rates = assemble_coupled_rates(
            final,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            heat_flux=system.heat_flux,
            energy_ports=system.energy_ports,
        )
        heat = battery_dissipation_w(spec, final, current)
        assert all(value >= -1e-14 for value in heat.values())
        assert sum(heat.values()) == pytest.approx(100 * rates.temperature_rate_K_s, abs=1e-12)
        assert heat["ohmic_W"] == pytest.approx(current * current * 0.05)


def test_isothermal_shared_bath_recovers_original_inventory_and_records_heat():
    old = BatterySPMSpec()
    thermal = replace(old, thermal=BatteryThermalSpec())
    drive = ScalarControlProgram(((0, old.current_at_c_rate(0.5)),), "A")
    bath = ThermalDeviceSpec(mode="isothermal", setpoint_K=298.15, minimum_setpoint_K=270)
    plain = advance_battery(old, old.initial_state(), drive, 600)
    result = advance_battery(thermal, thermal.initial_state(), drive, 600, thermal_device=bath)
    assert result.final_state.amounts_mol == pytest.approx(
        plain.final_state.amounts_mol, rel=1e-7, abs=1e-13
    )
    assert result.final_state.temperature_K == 298.15
    assert result.final_state.boundary_heat_J < 0
    assert abs(result.energy_balance_residual_J) < 1e-6


def test_constant_voltage_solves_actual_current_and_enforces_source_limit():
    spec = BatterySPMSpec(thermal=BatteryThermalSpec())
    drive = ScalarControlProgram(((0, 3.5),), "V")
    result = advance_battery(
        spec, spec.initial_state(), drive, 600, evaluation_times_s=[0, 100, 300, 600]
    )
    currents = []
    for state in result.trajectory:
        current = battery_control_current(spec, state, drive)
        voltage = battery_readout(spec, state, current).voltage_V
        if current == spec.maximum_current_A:
            assert voltage >= 3.5
        else:
            assert voltage == pytest.approx(3.5, abs=1e-9)
        currents.append(current)
    assert currents[-1] < currents[0]
    assert currents[-1] < spec.maximum_current_A
    limited = replace(spec, maximum_current_A=0.01)
    assert battery_control_current(limited, limited.initial_state(), drive) == 0.01
    assert battery_readout(limited, limited.initial_state(), 0.01).voltage_V > 3.5


def test_existing_heater_has_same_power_limit_and_preserves_energy():
    spec = BatterySPMSpec(thermal=BatteryThermalSpec())
    device = ThermalDeviceSpec(setpoint_K=320, maximum_power_W=2, minimum_setpoint_K=270)
    result = advance_battery(
        spec, spec.initial_state(), ScalarControlProgram(((0, 0),), "A"), 100, thermal_device=device
    )
    assert result.final_state.temperature_K == pytest.approx(300.15, abs=1e-8)
    assert result.final_state.boundary_heat_J == pytest.approx(200, abs=1e-7)
    assert result.final_state.coordinates["delivered_work_J"] == 0


def test_two_cycles_use_same_inventory_and_account_for_every_interval():
    from chemworld.examples.native_battery_cycles import cycle_session
    from chemworld.runtime.battery import unpack_battery

    session = cycle_session()
    _, _, spec, initial, _ = unpack_battery(session._state)
    transitions = []
    for _ in range(2):
        before = session._state.ledger.time_s
        result = session.execute({"operation": "wait", "duration_s": 14400})
        assert result.info["transaction_status"] == "committed", result.info
        assert session._state.ledger.time_s - before == pytest.approx(
            result.info["actual_duration_s"]
        )
        transitions.extend(result.info["outcome_diagnostics"]["control_transitions"])
    control = session._state.equipment.equipment["battery"].settings["electrical_control"]
    assert control["stage_index"] == 12 and control["status"] == "completed"
    assert [t["stage_index"] for t in transitions] == list(range(12))
    for event in transitions:
        stage = event["stage_index"] % 6
        if stage in (0, 3):
            assert event["terminal"]["voltage_V"] == pytest.approx(3.5 if stage == 0 else 3.9)
        if stage in (1, 4):
            assert abs(event["terminal"]["current_A"]) == pytest.approx(
                spec.current_at_c_rate(0.05)
            )
        if stage in (2, 5):
            assert event["actual_duration_s"] == 300
    final = unpack_battery(session._state)[3]
    ledger = session._state.ledger
    exchanged = ledger.flow_electrical_work_J + ledger.energy_jacket_J - ledger.heat_loss_J
    assert energy(spec, final) - energy(spec, initial) == pytest.approx(exchanged, abs=1e-3)
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-12
    )
    assert ledger.time_s < 28800
    assert control["program"]["setpoints"][-1][1] == 0


def test_mid_cv_snapshot_replays_exactly_and_material_reset_keeps_soc():
    from chemworld.examples.native_battery_cycles import cycle_session
    from chemworld.runtime import ChemWorldRuntime

    session = cycle_session()
    session.execute({"operation": "wait", "duration_s": 3000})
    source = session._state.equipment.equipment["battery"].settings["electrical_control"]
    assert source["stage_index"] == 1 and source["program"]["unit"] == "V"
    restored = ChemWorldRuntime.from_snapshot(session.snapshot())
    for candidate in (session, restored):
        result = candidate.execute({"operation": "wait", "duration_s": 2000})
        assert result.info["transaction_status"] == "committed"
    assert session._state.state_slots == restored._state.state_slots
    assert session._state.temperature_K == restored._state.temperature_K
    assert session._state.equipment == restored._state.equipment
    assert session._state.ledger == restored._state.ledger
    slots = session._state.state_slots
    session.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert session._state.state_slots == slots


def test_later_stage_numerical_failure_rolls_back_entire_candidate(monkeypatch):
    from chemworld.examples.native_battery_cycles import cycle_session
    from chemworld.runtime import battery

    session = cycle_session()
    before = session._state
    original = battery.advance_battery
    count = 0

    def fail_second(spec, state, program, duration_s, **kwargs):
        nonlocal count
        if duration_s > 0:
            count += 1
            if count == 2:
                raise RuntimeError("injected numerical failure after one candidate stage")
        return original(spec, state, program, duration_s, **kwargs)

    monkeypatch.setattr(battery, "advance_battery", fail_second)
    result = session.execute({"operation": "wait", "duration_s": 14400})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert count == 2
    assert session._state.state_slots == before.state_slots
    assert session._state.equipment == before.equipment
    assert session._state.temperature_K == before.temperature_K
    assert session._state.ledger.time_s == before.ledger.time_s
    assert len(result.info["outcome_diagnostics"]["control_intervals"]) == 2


def test_shared_heater_and_native_cv_measurement_use_actual_physics():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.battery_world import battery_world_spec

    spec = BatterySPMSpec(thermal=BatteryThermalSpec())
    device = ThermalDeviceSpec(setpoint_K=320, maximum_power_W=2, minimum_setpoint_K=270)
    session = ChemWorldRuntime(composition=battery_world_spec(spec, thermal_device=device))
    session.reset(seed=30)
    result = session.execute({"operation": "heat", "target_temperature_K": 320, "duration_s": 100})
    assert result.info["transaction_status"] == "committed", result.info
    assert session._state.temperature_K == pytest.approx(300.15)
    assert session._state.ledger.energy_jacket_J == pytest.approx(200)
    session.execute({"operation": "pause_control"})
    session.execute({"operation": "wait", "duration_s": 100})
    assert session._state.ledger.energy_jacket_J == pytest.approx(200)
    session.execute({"operation": "configure_electrical_control", "potential_V": 3.7})
    result = session.execute({"operation": "measure", "instrument": "battery_meter"})
    assert result.info["transaction_status"] == "committed", result.info
    readings = {c.channel_id: c.value for c in result.observation.channels}
    assert readings["terminal_potential"] == pytest.approx(3.7, abs=1e-8)
    assert 0 < readings["electrical_current"] < spec.maximum_current_A
