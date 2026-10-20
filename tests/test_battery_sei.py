"""SEI uses actual finite species, interfacial current and common thermodynamics."""

from dataclasses import replace

import pytest

from chemworld.physchem.battery_degradation import SEISpec
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    BatteryThermalSpec,
    F,
    advance_battery,
    battery_dissipation_w,
    battery_readout,
    battery_system,
)
from chemworld.physchem.coupled_system import assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram


def model():
    return BatterySPMSpec(thermal=BatteryThermalSpec(), sei=SEISpec())


@pytest.mark.parametrize("current", [0, 0.3, -0.3])
def test_finite_sei_balances_species_main_side_currents_and_heat(current):
    spec = model()
    initial = spec.initial_state()
    programme = ScalarControlProgram(((0, current),), "A")
    result = advance_battery(spec, initial, programme, 1000)
    assert result.outcome == "success"
    assert max(abs(v) for v in result.conservation_residuals.values()) < 1e-8
    final = result.final_state
    extent = final.amounts_mol["sei:carbonate"] - initial.amounts_mol["sei:carbonate"]
    assert extent > 0
    assert initial.amounts_mol["sei:EC"] - final.amounts_mol["sei:EC"] == pytest.approx(
        extent, abs=1e-12
    )
    assert final.amounts_mol["sei:ethylene"] == pytest.approx(extent, abs=1e-12)
    active = (*spec.negative.keys("negative"), *spec.positive.keys("positive"))
    assert sum(initial.amounts_mol[k] - final.amounts_mol[k] for k in active) == pytest.approx(
        2 * extent, abs=1e-12
    )
    reading = battery_readout(spec, final, current)
    assert reading.negative_main_current_A + reading.sei_current_A == pytest.approx(
        current, abs=1e-14
    )
    assert reading.sei_current_A < 0
    assert (
        reading.film_resistance_ohm
        > spec.sei.film_resistivity_ohm_m * spec.sei.initial_thickness_m / spec.negative.area_m2
    )
    system = battery_system(spec, programme)
    rates = assemble_coupled_rates(
        final,
        processes=system.processes,
        thermodynamics=system.thermodynamics,
        heat_flux=system.heat_flux,
        energy_ports=system.energy_ports,
    )
    heat = battery_dissipation_w(spec, final, current)
    assert min(heat.values()) >= -1e-12
    assert sum(heat.values()) == pytest.approx(100 * rates.temperature_rate_K_s, abs=1e-11)
    assert abs(result.energy_balance_residual_J) < 1e-6 + 1e-6 * abs(
        final.coordinates["delivered_work_J"]
    )
    assert abs(result.conservation_residuals["negative_faradaic_C"]) < 1e-8
    assert 2 * F * extent > 0


def test_missing_solvent_stops_growth_without_inventing_feed():
    spec = replace(model(), sei=SEISpec(initial_solvent_concentration_mol_m3=0))
    initial = spec.initial_state()
    result = advance_battery(spec, initial, ScalarControlProgram(((0, 0),), "A"), 1000)
    assert result.final_state.amounts_mol == pytest.approx(initial.amounts_mol, abs=1e-15)
    assert result.actual_exchanges == ()


def test_native_sei_wait_and_meter_commit_balanced_compound_inventory():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.battery_world import battery_world_spec

    spec = model()
    session = ChemWorldRuntime(composition=battery_world_spec(spec))
    session.reset(seed=31)
    for action in (
        {"operation": "configure_electrical_control", "current_A": 0.3},
        {"operation": "wait", "duration_s": 1000},
        {"operation": "measure", "instrument": "battery_meter"},
    ):
        result = session.execute(action)
        assert result.info["transaction_status"] == "committed", result.info
    assert session._state.ledger.time_s == 1001
    assert {c.unit for c in result.observation.channels} == {"V", "A"}
    assert session._state.state_slots.slots[0].values["sei:ethylene"] > 0


def test_disabling_entire_sei_recovers_base_model_exactly():
    plain = BatterySPMSpec(thermal=BatteryThermalSpec())
    disabled = replace(model(), sei=None)
    drive = ScalarControlProgram(((0, 0.3),), "A")
    left = advance_battery(plain, plain.initial_state(), drive, 1000)
    right = advance_battery(disabled, disabled.initial_state(), drive, 1000)
    assert left.final_state == right.final_state


def test_finite_solvent_exhaustion_limits_products_and_active_lithium_loss():
    spec = replace(model(), sei=SEISpec(solvent_diffusivity_m2_s=1e-15))
    initial = spec.initial_state()
    result = advance_battery(spec, initial, ScalarControlProgram(((0, 0),), "A"), 10000)
    assert result.outcome == "success"
    final = result.final_state
    assert 0 <= final.amounts_mol["sei:EC"] < 1e-10
    assert final.amounts_mol["sei:ethylene"] == pytest.approx(
        initial.amounts_mol["sei:EC"], abs=1e-10
    )
    assert (
        final.amounts_mol["sei:carbonate"]
        <= initial.amounts_mol["sei:carbonate"] + initial.amounts_mol["sei:EC"] + 1e-12
    )
    assert result.actual_exchanges == ()
    assert max(abs(v) for v in result.conservation_residuals.values()) < 1e-8


def test_irreversible_sei_stops_outside_its_electrochemical_driving_domain():
    spec = replace(model(), sei=SEISpec(reference_driving_energy_J_mol=0))
    initial = spec.initial_state()
    result = advance_battery(spec, initial, ScalarControlProgram(((0, 0),), "A"), 1000)
    assert result.outcome == "stopped"
    assert result.actual_duration_s == 0
    assert result.stopping_event == "battery:sei:affinity-limit"
    assert result.final_state == initial


def test_source_queue_bounds_pending_work_and_keeps_absolute_stage_identity():
    from chemworld.runtime.control_program import queue_electrical_stage
    from chemworld.world.battery_world import battery_world_spec

    state = battery_world_spec(model()).physics.initial_states["reactor"]
    action = {"current_A": 0, "stage_duration_s": 1}
    for _ in range(32):
        state = queue_electrical_stage(state, action)
    with pytest.raises(ValueError, match="pending"):
        queue_electrical_stage(state, action)
