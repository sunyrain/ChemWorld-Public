"""Shared PH closure with explicit nonvolatile liquid data."""

import json

import pytest

from chemworld.examples.native_concentration_research import synthetic_concentration_properties
from chemworld.physchem.equilibrium import rachford_rice_diagnostic_report
from chemworld.physchem.flash_units import ph_flash, tp_flash


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("pressure", [30000, 60000, 100000])
@pytest.mark.parametrize("temperature", [295, 330, 400])
def test_nonvolatile_target_uses_liquid_caloric_reference_without_gas_loss(
    binary, pressure, temperature
):
    properties = synthetic_concentration_properties(pressure, binary_volatile=binary)
    amounts = {"light": 0.012, "target": 0.002}
    if binary:
        amounts["heavy"] = 0.008
    tp = tp_flash(
        properties,
        amounts,
        temperature_K=temperature,
        pressure_Pa=pressure,
        support_heat_capacity_J_K=1,
    )
    assert tp.vapor_amounts_mol["target"] == 0
    assert tp.liquid_amounts_mol["target"] == 0.002
    assert "target" not in tp.vapor_fugacity_coefficients
    ph = ph_flash(
        properties,
        amounts,
        pressure_Pa=pressure,
        enthalpy_J=tp.total_enthalpy_j,
        support_heat_capacity_J_K=1,
    )
    assert ph.temperature_K == pytest.approx(temperature, abs=1e-7)
    assert ph.vapor_amounts_mol["target"] == 0
    assert ph.material_balance_error_mol < 1e-10
    assert ph.energy_balance_error_J < 1e-6


def test_declared_nonvolatile_rr_has_no_fabricated_beta_one_objective():
    report = rachford_rice_diagnostic_report(
        {"light": 0.8, "target": 0.2},
        {"light": 2.0, "target": 0.0},
        nonvolatile_component_ids=("target",),
    )
    assert report.vapor_fraction == pytest.approx(0.6)
    assert report.liquid_composition == pytest.approx({"light": 0.5, "target": 0.5})
    assert report.vapor_composition == {"light": 1.0, "target": 0.0}
    assert report.objective_at_one is None
    json.dumps(report.to_dict(), allow_nan=False)
    with pytest.raises(ValueError, match="positive"):
        rachford_rice_diagnostic_report(
            {"light": 0.8, "target": 0.2}, {"light": 2.0, "target": 0.0}
        )


def test_solvent_free_liquid_has_no_invented_molar_gas_enthalpy():
    properties = synthetic_concentration_properties(30000)
    composition = {"light": 0.0, "target": 1.0}
    liquid_h, vapor_h = properties.phase_enthalpies(330, 30000, composition, composition)
    assert liquid_h == pytest.approx(120 * (330 - 298.15))
    assert vapor_h is None
    result = tp_flash(properties, composition, temperature_K=330, pressure_Pa=30000)
    assert result.phase_status == "all_liquid"
    assert result.vapor_enthalpy_J == 0
    assert result.vapor_composition == {"light": 0.0, "target": 0.0}
    ph = ph_flash(properties, composition, pressure_Pa=30000, enthalpy_J=result.total_enthalpy_j)
    assert ph.temperature_K == pytest.approx(330, abs=1e-8)
    restored = type(properties).from_dict(properties.to_dict())
    assert restored.to_dict() == properties.to_dict()
    target = properties.packages[-1]
    assert not target.by_property("ideal_gas_heat_capacity")
    assert not target.by_property("vapor_pressure")


def test_trace_nonvolatile_liquid_survives_vapor_fraction_rounding_to_one():
    report = rachford_rice_diagnostic_report(
        {"light": 1.0, "target": 1e-20},
        {"light": 1000.0, "target": 0.0},
        nonvolatile_component_ids=("target",),
    )
    assert report.phase_status == "two_phase"
    assert report.vapor_fraction == 1.0
    assert 0 < report.liquid_fraction < 1e-18
    assert report.liquid_composition["target"] > 0.99
    assert report.vapor_composition["target"] == 0
    assert abs(report.residual) < 1e-12


def test_standalone_coupled_entry_retains_target_and_balances_actual_outlet_and_heat():
    from chemworld.examples.native_concentration_research import (
        initial_concentration_state,
        synthetic_concentration_spec,
    )
    from chemworld.physchem.concentration_units import simulate_vacuum_concentration
    from chemworld.physchem.coupled_concentration import (
        CoupledConcentrationRequest,
        CoupledConcentrationState,
    )
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    spec = synthetic_concentration_spec(30000)
    initial = initial_concentration_state(spec)
    request = CoupledConcentrationRequest(spec, initial, 5, ThermalDeviceSpec(setpoint_K=430))
    result = simulate_vacuum_concentration(request)
    final = result.final_state
    assert result.actual_duration_s == 5
    assert final.amounts_mol[0]["target"] == 0.002
    assert final.amounts_mol[1]["target"] == final.outlet_amounts_mol["target"] == 0
    assert final.amounts_mol[1]["light"] > 0
    assert final.outlet_amounts_mol["light"] > 0
    for key in spec.properties.component_ids:
        assert sum(n[key] for n in final.amounts_mol) + final.outlet_amounts_mol[
            key
        ] == pytest.approx(sum(n[key] for n in initial.amounts_mol), abs=1e-12)
    assert sum(final.enthalpies_J) - sum(initial.enthalpies_J) == pytest.approx(
        final.heater_energy_J - final.condenser_energy_J - final.outlet_enthalpy_J, abs=1e-7
    )
    restored = CoupledConcentrationState(**json.loads(json.dumps(final.to_dict())))
    assert restored == final
    assert type(spec).from_dict(spec.to_dict()).to_dict() == spec.to_dict()
    json.dumps(result.to_dict(), allow_nan=False)


def test_pure_volatile_depletion_keeps_absent_identity_zero_and_retains_remaining_gas():
    from dataclasses import replace

    from chemworld.examples.native_concentration_research import synthetic_concentration_spec
    from chemworld.examples.native_flash_research import synthetic_vle_properties
    from chemworld.physchem.coupled_concentration import (
        CoupledConcentrationState,
        advance_coupled_concentration,
    )
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec

    base = synthetic_concentration_spec()
    spec = replace(base, properties=synthetic_vle_properties(),
        solvent_component_ids=("light", "heavy"), target_component_id=None,
        target_solvent_remaining_fraction=0,
        equipment=replace(base.equipment, minimum_residual_volume_L=0))
    liquid, empty = {"light": .0002, "heavy": 0.}, {"light": 0., "heavy": 0.}
    initial = CoupledConcentrationState((liquid, empty),
        (spec.at_tp(0, liquid, 329).total_enthalpy_j,
         spec.at_tp(1, empty, 270).total_enthalpy_j))
    result = advance_coupled_concentration(spec, initial, duration_s=120,
        heater=ThermalDeviceSpec(maximum_power_W=10, setpoint_K=430))
    assert result.stopping_event == "liquid_solvent_exhausted"
    assert result.phases[0].liquid_amounts_mol["light"] == pytest.approx(1e-7, abs=1e-10)
    assert result.phases[0].vapor_amounts_mol["light"] > 0
    assert result.solver_diagnostic["invariant_zero_components"] == ["heavy"]
    assert all(n["heavy"] == 0 for n in result.final_state.amounts_mol)
