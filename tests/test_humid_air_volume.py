"""Stock closure, endpoint response and fixed air composition boundaries."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem import (
    HumidAirIsochoricThermodynamics,
    condensed_water_state,
    humid_air_saturation,
    humid_air_volume,
)
from chemworld.physchem.coupled_system import CoupledState, ProcessContribution
from chemworld.physchem.heat_transfer_units import ThermalSystemSpec
from chemworld.physchem.humid_air_thermal import DRY_AIR_MOLE_FRACTIONS
from chemworld.physchem.specs import ComponentSpec


def components():
    return tuple(ComponentSpec(key, formula, cas_number=cas, default_phase="gas")
                 for key, formula, cas in (("nitrogen", "N2", "7727-37-9"),
                                          ("oxygen", "O2", "7782-44-7"),
                                          ("argon", "Ar", "7440-37-1"),
                                          ("water", "H2O", "7732-18-5")))


def thermal_inputs(T=300., p=101325., phase="liquid", dry=.002, extra=.01):
    species = components()
    sat = humid_air_saturation(T, p, condensed_phase=phase)
    water = dry*(sat.gas.humidity_ratio_kg_water_kg_dry_air+extra)
    volume = (dry/sat.gas.dry_air_mass_fraction/sat.gas.mass_density_kg_m3
              + dry*extra/sat.condensate.mass_density_kg_m3)*1000
    dry_molar_mass = sum(x*c.molecular_weight_g_mol/1000
                        for x, c in zip(DRY_AIR_MOLE_FRACTIONS, species[:3], strict=True))
    amounts = {c.identifier: dry/dry_molar_mass*x
               for x, c in zip(DRY_AIR_MOLE_FRACTIONS, species[:3], strict=True)}
    amounts["water"] = water/(species[3].molecular_weight_g_mol/1000)
    thermal = HumidAirIsochoricThermodynamics(*species, volume, phase)
    return thermal, CoupledState(amounts, T, volume)


@pytest.mark.parametrize("T,phase", [(230., "ice"), (300., "liquid"), (400., "liquid")])
@pytest.mark.parametrize("p", [1e6, 2e6])
def test_saturated_boundary_has_no_duplicate_stock_and_one_sided_response(T, phase, p):
    thermal, initial = thermal_inputs(T, p, phase, extra=0)
    at = thermal.equilibrium_state(initial)
    cold = thermal.equilibrium_state(replace(initial, temperature_K=T-.001))
    hot = thermal.equilibrium_state(replace(initial, temperature_K=T+.001))
    assert at.condensed_water_mass_kg == 0
    assert cold.condensed_water_mass_kg > 0
    assert hot.condensed_water_mass_kg == 0
    for state in (at, cold, hot):
        assert state.gas_mass_kg+state.condensed_water_mass_kg == pytest.approx(
            state.dry_air_mass_kg+state.total_water_mass_kg, abs=1e-14)
        assert state.gas_volume_m3+state.condensed_volume_m3 == pytest.approx(
            initial.volume_L*1e-3, rel=2e-11)
        assert state.heat_capacity_J_K > 0
        json.dumps(state.to_dict(), allow_nan=False)
    assert abs(cold.internal_energy_J-at.internal_energy_J) <= cold.heat_capacity_J_K*.00101
    assert hot.internal_energy_J > at.internal_energy_J > cold.internal_energy_J


@pytest.mark.parametrize("index,value", [(0, True), (1, True), (2, True), (3, True),
    (0, 269.), (0, float("nan")), (1, 0.), (1, -.1), (2, -.1), (2, float("inf")), (3, 0.)])
def test_invalid_volume_inputs_are_rejected_before_cache(index, value):
    args = [300., .02, .001, .002]
    args[index] = value
    with pytest.raises(ValueError):
        humid_air_volume(*args, condensed_phase="liquid")


def test_dry_endpoint_cache_does_not_accept_boolean_zero():
    valid = humid_air_volume(300., .02, 0., .002, condensed_phase="liquid")
    assert valid.chemical_potential_residual_J_kg is None
    assert valid.condensate is None
    with pytest.raises(ValueError):
        humid_air_volume(300., .02, False, .002, condensed_phase="liquid")
    with pytest.raises(ValueError):
        humid_air_volume(300., .02, 0., .002, condensed_phase="auto")
    with pytest.raises(ValueError):
        humid_air_volume(300., .02, 0., .002, condensed_phase="ice")
    with pytest.raises(ValueError):
        humid_air_volume(300., .02, 0., 1e-8, condensed_phase="liquid")


def test_adapter_preserves_real_inventory_mass_and_tangent_energy():
    thermal, state = thermal_inputs()
    point = thermal.equilibrium_state(state)
    assert point.dry_air_mass_kg == pytest.approx(.002, rel=1e-15)
    assert point.condensed_water_mass_kg == pytest.approx(.002*.01, rel=2e-9)
    assert thermal.pressure_pa(state) == pytest.approx(101325, rel=2e-9)
    rates = {key: value*.01 for key, value in state.amounts_mol.items()}
    analytic = thermal.chemical_energy_rate_w(state, rates, ())
    step = 1e-3
    before, after = (replace(state, amounts_mol={k: n+sign*step*rates[k]
                                               for k, n in state.amounts_mol.items()})
                     for sign in (-1, 1))
    difference = (thermal.internal_energy_j(after)-thermal.internal_energy_j(before))/(2*step)
    assert analytic == pytest.approx(difference, rel=2e-4, abs=.01)
    assert thermal.chemical_energy_rate_w(state, {}, ()) == 0
    with pytest.raises(ValueError, match="ratio"):
        thermal.chemical_energy_rate_w(state, {"nitrogen": .1}, ())
    with pytest.raises(ValueError, match="duplicate reaction heat"):
        thermal.chemical_energy_rate_w(
            state, {}, (ProcessContribution(legacy_reaction_energy_W=0),))
    broken = replace(state, amounts_mol={**state.amounts_mol, "oxygen": .1})
    with pytest.raises(ValueError, match="ratio"):
        thermal.validate_state(broken)
    with pytest.raises(ValueError, match="volume"):
        thermal.validate_state(replace(state, volume_L=state.volume_L*2))
    with pytest.raises(ValueError, match="four declared"):
        thermal.validate_state(replace(state, amounts_mol={**state.amounts_mol, "CO2": 0.}))


def test_component_identity_json_and_thermal_pressure_contract():
    thermal, _ = thermal_inputs()
    restored = HumidAirIsochoricThermodynamics.from_dict(json.loads(json.dumps(thermal.to_dict())))
    assert restored == thermal
    for changes in ({"nitrogen": ComponentSpec("nitrogen", "NO")},
                    {"water": ComponentSpec("water", "H2O", cas_number="7727-37-9")},
                    {"argon": ComponentSpec("oxygen", "Ar")}, {"condensed_phase": "auto"}):
        with pytest.raises(ValueError):
            replace(thermal, **changes)
    model = ThermalSystemSpec(thermal, pressure_law="humid_air", minimum_temperature_K=270,
                              maximum_temperature_K=473, stop_temperature_K=460)
    assert ThermalSystemSpec.from_dict(json.loads(json.dumps(model.to_dict()))) == model
    for law in ("fixed_pressure", "ideal_gas", "helmholtz", "if97"):
        with pytest.raises(ValueError, match="same EOS"):
            replace(model, pressure_law=law)
    with pytest.raises(ValueError, match="domain"):
        replace(model, minimum_temperature_K=260)


def test_fixed_composition_is_not_waived_for_tiny_inventory_or_rates():
    thermal, state = thermal_inputs()
    broken = replace(state, amounts_mol={"nitrogen": 1e-20, "oxygen": 0.,
                                        "argon": 0., "water": 0.})
    with pytest.raises(ValueError, match="ratio"):
        thermal.validate_state(broken)
    with pytest.raises(ValueError, match="ratio"):
        thermal.chemical_energy_rate_w(state, {"nitrogen": 1e-20}, ())


@pytest.mark.parametrize("T,p", [(299.99273420591027, 101321.11594698219),
                                (296.27840891406623, 99434.60579040217)])
def test_recorded_dense_liquid_cancellation_points_meet_original_pressure_budget(T, p):
    liquid = condensed_water_state(T, p, phase="liquid")
    assert abs(liquid.density_pressure_residual_Pa) <= 1e-5
    assert liquid.cp_J_kg_K > 0
    json.dumps(liquid.to_dict(), allow_nan=False)
