"""Finite metal, true working-particle dynamics and original residual contracts."""

import json
from dataclasses import asdict, replace
from math import asinh

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_half_cell import (
    METAL_KEY,
    METAL_UNITS,
    BatteryHalfCellSPMSpec,
    F,
    LithiumMetalCounterElectrodeSpec,
    R,
    advance_half_cell,
    half_cell_control_current,
    half_cell_readout,
    half_cell_system,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def source():
    return InputSource(DataSourceProvenance(
        "half-cell-functional", 0, "Authored synthetic half-cell regression inputs", "1", "CC0-1.0"
    ), "synthetic")


def metal(**changes):
    values = {"area_m2": 0.01, "thickness_m": 1e-5, "conductivity_S_m": 1e7,
              "partial_molar_volume_m3_mol": 1.3e-5,
              "exchange_current_density_A_m2": 10.0, "reference_temperature_K": 298.15,
              "electrolyte_concentration_mol_m3": 1000.0}
    values.update(changes)
    table = ScientificTable("metal-functional", "parameters", tuple(
        QuantityColumn(k, u, "parameter", (0, 1e10), "Fixed synthetic input")
        for k, u in METAL_UNITS.items()), (source(),), (ScientificRow(
            "metal", source().provenance.source_id, "synthetic", "Li", "parameters",
            {k: InputValue(v, METAL_UNITS[k], None, "Fixed synthetic input")
             for k, v in values.items()}),))
    return LithiumMetalCounterElectrodeSpec(table, "metal")


def model(**changes):
    args = {"metal": metal(), "positive": ParticleSpec(cells=8, mesh_exponent=1),
            "positive_ocv": ElectrodeOCV("working", source(), (4.2, -0.8)),
            "maximum_current_A": 0.5}
    args.update(changes)
    return BatteryHalfCellSPMSpec(**args)


def programme(value, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def balances(spec, initial, final):
    charge = final.coordinates["discharged_charge_C"]-initial.coordinates["discharged_charge_C"]
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=2e-12)
    assert F*(final.amounts_mol[METAL_KEY]-initial.amounts_mol[METAL_KEY]) == pytest.approx(
        -charge, abs=2e-8)
    assert F*sum(final.amounts_mol[k]-initial.amounts_mol[k]
                 for k in spec.positive.keys("positive")) == pytest.approx(charge, abs=2e-8)


def test_metal_geometry_source_codec_and_single_material_ownership():
    spec = model()
    assert spec.metal.initial_amount_mol == pytest.approx(.01*1e-5/1.3e-5)
    assert spec.metal.resistance_ohm == pytest.approx(1e-5/(1e7*.01))
    restored = BatteryHalfCellSPMSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.initial_state() == spec.initial_state()
    assert len(spec.inventory_keys) == spec.positive.cells+1
    assert all(not k.startswith("negative:") for k in spec.inventory_keys)
    assert spec.hysteresis_keys == ()
    with pytest.raises(TypeError):
        spec.metal.native_parameters["area_m2"] = 99
    read = asdict(half_cell_readout(spec, spec.initial_state(), 0))
    assert "negative_surface_fraction" not in read


@pytest.mark.parametrize("current", [-0.2, 0.0, 0.2])
def test_actual_signed_bv_port_and_reservoir_faradaic_transfer(current):
    spec = model()
    initial = spec.initial_state()
    reading = half_cell_readout(spec, initial, current)
    theta = reading.working_surface_fraction
    p = spec.positive
    j0 = p.exchange_current_scale_A_m2*np.sqrt(theta*(1-theta))
    ew = 2*R*spec.temperature_K/F*asinh(-current/p.area_m2/(2*j0))
    em = 2*R*spec.temperature_K/F*asinh(current/.01/(2*10))
    expected = 4.2-.8*theta+ew-em-current*spec.metal.resistance_ohm
    assert reading.voltage_V == pytest.approx(expected, abs=1e-14)
    result = advance_half_cell(spec, initial, programme(current), 40)
    assert result.outcome == "success"
    balances(spec, initial, result.final_state)
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(current*40)
    assert result.energy_balance_residual_J is None
    assert all(x == 0 for x in (result.final_state.boundary_heat_J,
                               result.final_state.chemical_energy_change_J,
                               result.final_state.heat_loss_J))


def test_nonuniform_profile_relaxes_without_resetting_metal_or_charge():
    spec = model()
    first = advance_half_cell(spec, spec.initial_state(), programme(.3), 50).final_state
    rested = advance_half_cell(spec, first, programme(0), 10000).final_state
    p = spec.positive
    before = np.array([first.amounts_mol[k] for k in p.keys("positive")])/p.mesh.cell_volumes_m3
    after = np.array([rested.amounts_mol[k] for k in p.keys("positive")])/p.mesh.cell_volumes_m3
    assert np.ptp(before) > 50
    assert np.ptp(after) < 1e-3
    assert rested.coordinates == first.coordinates
    assert rested.amounts_mol[METAL_KEY] == first.amounts_mol[METAL_KEY]
    balances(spec, first, rested)


def test_true_depletion_event_snapshot_rest_and_reverse_plating():
    spec = model(metal=metal(thickness_m=1.3e-7))
    current = .2
    expected_time = spec.metal.initial_amount_mol*F/current
    result = advance_half_cell(spec, spec.initial_state(), programme(current), expected_time*2)
    assert result.stopping_event == "half-cell:metal:depleted"
    assert result.actual_duration_s == pytest.approx(expected_time, abs=2e-8)
    final = result.final_state
    assert final.amounts_mol[METAL_KEY] == pytest.approx(0, abs=2e-13)
    balances(spec, spec.initial_state(), final)
    # Exact boundary is an admissible *state*, even though forward current stops.
    exact = replace(final, amounts_mol={**final.amounts_mol, METAL_KEY: 0.0})
    forward = advance_half_cell(spec, exact, programme(current), 10)
    assert forward.actual_duration_s == 0
    assert forward.stopping_event == "half-cell:metal:depleted"
    rested = advance_half_cell(spec, exact, programme(0), 5).final_state
    assert rested.amounts_mol[METAL_KEY] == 0
    plated = advance_half_cell(spec, rested, programme(-current), 10).final_state
    assert plated.amounts_mol[METAL_KEY] == pytest.approx(current*10/F, abs=1e-13)
    assert plated.time_s == pytest.approx(expected_time+15)
    balances(spec, exact, plated)


@pytest.mark.parametrize("current,initial,limit", [(.2, .97, "saturated"), (-.2, .03, "depleted")])
def test_real_working_surface_stops_before_bulk_capacity(current, initial, limit):
    spec = model(initial_positive_fraction=initial,
                 positive=ParticleSpec(cells=8, mesh_exponent=1, diffusivity_m2_s=1e-12))
    result = advance_half_cell(spec, spec.initial_state(), programme(current), 1000)
    assert result.stopping_event == "half-cell:working:"+limit
    assert 0 < result.actual_duration_s < 1000
    reading = half_cell_readout(spec, result.final_state, current)
    boundary = 1-spec.surface_margin if current > 0 else spec.surface_margin
    assert reading.working_surface_fraction == pytest.approx(boundary, abs=1e-8)
    assert reading.working_average_fraction != pytest.approx(boundary, abs=1e-6)
    balances(spec, spec.initial_state(), result.final_state)


def test_actual_voltage_cutoff_stops_and_retains_charge():
    spec = model(minimum_voltage_V=3.82)
    result = advance_half_cell(spec, spec.initial_state(), programme(.2), 1500)
    assert result.stopping_event == "half-cell:minimum-voltage"
    assert 0 < result.actual_duration_s < 1500
    assert half_cell_readout(spec, result.final_state, .2).voltage_V == pytest.approx(
        3.82, abs=1e-8)
    balances(spec, spec.initial_state(), result.final_state)


@pytest.mark.parametrize("target,expected_mode", [(3.92, "constant_voltage"),
    (3.5, "positive_current_limit"), (4.3, "negative_current_limit")])
def test_finite_cv_solves_port_or_reports_the_real_source_limit(target, expected_mode):
    spec = model()
    drive = programme(target, "V")
    result = advance_half_cell(spec, spec.initial_state(), drive, 20)
    assert result.outcome == "success"
    assert result.solver_diagnostic["battery_control_mode"] == expected_mode
    final = result.final_state
    current = half_cell_control_current(spec, final, drive)
    read = half_cell_readout(spec, final, current)
    if expected_mode == "constant_voltage":
        assert read.voltage_V == pytest.approx(target, abs=2e-10)
        assert 0 < current < spec.maximum_current_A
    else:
        sign = 1 if expected_mode == "positive_current_limit" else -1
        assert current == sign*spec.maximum_current_A
        assert read.voltage_V != pytest.approx(target, abs=1e-3)
    balances(spec, spec.initial_state(), final)


def test_same_state_segmented_ramp_and_reversal():
    spec = model()
    drive = ScalarControlProgram(((0, .1), (20, -.1), (60, .2)), "A")
    whole = advance_half_cell(spec, spec.initial_state(), drive, 60).final_state
    first = advance_half_cell(spec, spec.initial_state(), drive, 20).final_state
    final = advance_half_cell(spec, first, drive, 40).final_state
    assert final.amounts_mol == pytest.approx(whole.amounts_mol, abs=1e-10)
    assert final.coordinates == pytest.approx(whole.coordinates, abs=1e-5)
    balances(spec, spec.initial_state(), final)


@pytest.mark.parametrize("unit,value", [("A", .17), ("V", 3.92)])
def test_complete_residual_jacobian_contains_real_surface_and_metal_bv(unit, value):
    spec = model()
    state = spec.initial_state()
    system = half_cell_system(spec, programme(value, unit))
    closure = system.algebraic
    names = tuple(v.name for v in closure.variables)
    material, coords = spec.inventory_keys, spec.coordinate_keys
    identities = (*material, "temperature", *coords, *names)
    initial = {**state.amounts_mol, "temperature": state.temperature_K, **state.coordinates,
               **closure.initial_values(state)}
    y = np.array([initial[k] for k in identities])
    scales = np.array([max(abs(v), 1e-5) for v in y])
    scales[-2:] = [v.scale for v in closure.variables]

    def view(z):
        values = dict(zip(identities, z, strict=True))
        return CoupledStateView({k: values[k] for k in material}, values["temperature"],
            state.volume_L, state.time_s, coordinates={k: values[k] for k in coords},
            algebraic_values={k: values[k] for k in names})

    def rhs(z):
        candidate = view(z)
        rates = assemble_coupled_rates(candidate, processes=system.processes,
            thermodynamics=system.thermodynamics, heat_flux=system.heat_flux, isothermal=True)
        return np.r_[[rates.amount_rates_mol_s[k] for k in material], 0.,
                     [rates.coordinate_rates[k] for k in coords],
                     -np.array(closure.residual(candidate))]

    analytic = system.residual_rhs_jacobian(view(y), identities).toarray()
    numeric = finite_jacobian(rhs, y, scales, relative_step=1e-7)
    assert np.max(abs(analytic-numeric)/np.maximum(1, abs(analytic))) < 3e-6


@pytest.mark.parametrize("field,value", [("area_m2", 0), ("conductivity_S_m", 0),
                                        ("partial_molar_volume_m3_mol", 0)])
def test_nonphysical_metal_declarations_reject(field, value):
    with pytest.raises(ValueError, match="positive"):
        metal(**{field: value})


@pytest.mark.parametrize("changes", [{"temperature_K": 300},
    {"electrolyte_concentration_mol_m3": 1200}, {"initial_positive_fraction": 1}])
def test_unsupported_kinetic_condition_or_initial_domain_rejects(changes):
    with pytest.raises(ValueError):
        model(**changes)


def test_false_inventory_or_undeclared_thermal_and_cv_direction_refuse():
    spec = model()
    state = spec.initial_state()
    wrong = replace(state, amounts_mol={**state.amounts_mol,
                                      METAL_KEY: 2*state.amounts_mol[METAL_KEY]})
    with pytest.raises(ValueError, match="Faraday"):
        advance_half_cell(spec, wrong, programme(0), 0)
    with pytest.raises(ValueError, match="thermal"):
        half_cell_system(spec, programme(0), thermal_device=ThermalDeviceSpec())
    rising = model(positive_ocv=ElectrodeOCV("working", source(), (3.6, .4)))
    assert advance_half_cell(rising, rising.initial_state(), programme(.1), 2).outcome == "success"
    with pytest.raises(ValueError, match="monotone"):
        half_cell_system(rising, programme(3.7, "V"))


def test_source_row_and_dimension_errors_cannot_silently_replace_parameters():
    declaration = metal()
    with pytest.raises(ValueError, match="existing row"):
        replace(declaration, parameter_row_id="absent")
    payload = declaration.to_dict()
    payload["parameters"]["rows"][0]["values"]["area_m2"]["unit"] = "m"
    with pytest.raises(ValueError):
        LithiumMetalCounterElectrodeSpec.from_dict(payload)
