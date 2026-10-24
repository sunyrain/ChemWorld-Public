"""Actual material current sharing, inventories and single common residual."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem import battery_particle_electrodes as shared
from chemworld.physchem.battery_composite import (
    ActiveMaterialPhaseSpec,
    BatteryCompositeSPMSpec,
    advance_composite,
    composite_algebraic,
    composite_control_mode,
    composite_fields,
    composite_readout,
    composite_system,
    validate_composite_state,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec, F, ParticleSpec, advance_battery
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def source():
    return InputSource(DataSourceProvenance("synthetic-composite", 0,
        "Authored single-valued material test input", "1", "CC0-1.0"), "synthetic")


def phase(name, *, volume=.5e-6, theta=.65, intercept=.3, slope=-.2, **kwargs):
    return ActiveMaterialPhaseSpec(name, ParticleSpec(cells=5, active_volume_m3=volume,
        diffusivity_m2_s=1e-13, **kwargs),
        ElectrodeOCV(name, source(), (intercept, slope)), theta, source())


def model(*, identical=False, **kwargs):
    return BatteryCompositeSPMSpec((phase("graphite"),
        phase("second", intercept=.3 if identical else .34)),
        phase("positive", volume=1e-6, theta=.3, intercept=4.2, slope=-.8), **kwargs)


def program(value=.2, unit="A", start=0):
    return ScalarControlProgram(((start, value),), unit)


def test_exact_immutable_material_declaration_and_phase_identity_roundtrip():
    s = model()
    payload = json.loads(json.dumps(s.to_dict(), allow_nan=False))
    decoded = BatteryCompositeSPMSpec.from_dict(payload)
    assert decoded == s and decoded.initial_state() == s.initial_state()
    assert len(s.inventory_keys) == 15
    assert len(s.coordinate_keys) == 2
    assert all(":phase:" in k for k in s.inventory_keys)
    assert s.phases[0].source.provenance.source_id == "synthetic-composite"
    payload["negative_phases"][0]["particle"]["active_volume_m3"] = 20
    assert decoded.phases[0].particle.active_volume_m3 == .5e-6
    with pytest.raises(ValueError, match="fields"):
        BatteryCompositeSPMSpec.from_dict({**s.to_dict(), "hidden_state": 1})


@pytest.mark.parametrize("change", ["count", "duplicate", "positive-slope", "domain", "mode"])
def test_unsupported_or_uncertified_materials_refuse(change):
    s = model()
    with pytest.raises((ValueError, TypeError)):
        if change == "count":
            replace(s, negative_phases=s.negative_phases[:1])
        elif change == "duplicate":
            replace(s, negative_phases=(s.phases[0], s.phases[0]))
        elif change == "positive-slope":
            replace(s, negative_phases=(phase("bad", slope=.1), s.phases[1]))
        elif change == "domain":
            replace(s, surface_margin=.001)
        else:
            BatteryCompositeSPMSpec(s.negative_phases, s.positive_phase, thermal="not supported")


def test_exact_constant_ocv_is_certified_without_numerical_slope_fudge():
    s = replace(model(), negative_phases=(phase("a", slope=0), phase("b", slope=0)))
    assert composite_readout(s, s.initial_state(), .1).voltage_V > 0


@pytest.mark.parametrize("volume,concentration,theta", [
    (1e150, 1e155, .65),  # Each capacity is finite; the Faraday charge is not.
    (1e-150, 5e-174, .02),  # Nonzero capacities; initial inventories underflow.
])
def test_finite_phase_parameters_cannot_create_invalid_composite_charge_scales(
        volume, concentration, theta):
    s = model()
    phases = tuple(replace(p, initial_fraction=theta, particle=replace(
        p.particle, active_volume_m3=volume,
        maximum_concentration_mol_m3=concentration)) for p in s.phases)
    assert all(np.isfinite(p.particle.capacity_mol) and p.particle.capacity_mol > 0
               for p in phases)
    assert all(np.isfinite(p.particle.area_m2) and p.particle.area_m2 > 0 for p in phases)
    with pytest.raises(ValueError, match="charge scale"):
        replace(s, negative_phases=phases[:2], positive_phase=phases[2])


@pytest.mark.parametrize("current", [-.2, .2])
def test_identical_materials_reduce_to_same_single_material_spm(current):
    s = model(identical=True)
    single = BatterySPMSpec(negative=replace(s.phases[0].particle, active_volume_m3=1e-6),
        positive=s.positive_phase.particle, initial_negative_fraction=.65,
        initial_positive_fraction=.3, negative_ocv=s.phases[0].ocv,
        positive_ocv=s.positive_phase.ocv)
    c = advance_composite(s, s.initial_state(), program(current), 8)
    b = advance_battery(single, single.initial_state(), program(current), 8)
    cr = composite_readout(s, c.final_state, current)
    from chemworld.physchem.battery_models import battery_readout
    br = battery_readout(single, b.final_state, current)
    assert cr.voltage_V == pytest.approx(br.voltage_V, abs=2e-8)
    assert cr.phase_readouts[0].surface_fraction == pytest.approx(
        br.negative_surface_fraction, abs=2e-8)
    assert cr.phase_readouts[0].reaction_current_A == pytest.approx(current/2, abs=1e-12)
    assert c.energy_balance_residual_J is None


def test_rest_has_real_internal_exchange_not_clipped_to_source_limit():
    s = model(maximum_current_A=.002)
    initial = s.initial_state()
    before = composite_fields(s, initial, current_A=0)
    assert before["reaction_current_A"][0] > s.maximum_current_A
    assert before["reaction_current_A"][1] < -s.maximum_current_A
    assert sum(before["reaction_current_A"][:2]) == pytest.approx(0, abs=1e-14)
    result = advance_composite(s, initial, program(0), 15)
    after = composite_readout(s, result.final_state, 0)
    assert after.phase_readouts[0].inventory_mol < sum(
        initial.amounts_mol[k] for k in s.particle_bins[0][3])
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(0, abs=1e-13)
    assert result.final_state.coordinates["delivered_work_J"] == pytest.approx(0, abs=1e-13)
    assert sum(result.final_state.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=1e-13)


def test_material_order_is_not_a_current_partition_or_gauge_choice():
    s = model()
    swapped = replace(s, negative_phases=tuple(reversed(s.negative_phases)))
    first = composite_readout(
        s, advance_composite(s, s.initial_state(), program(.2), 5).final_state, .2)
    second = composite_readout(swapped, advance_composite(
        swapped, swapped.initial_state(), program(.2), 5).final_state, .2)
    assert first.voltage_V == pytest.approx(second.voltage_V, abs=1e-10)
    a = {p.phase_id: p for p in first.phase_readouts}
    b = {p.phase_id: p for p in second.phase_readouts}
    for key in a:
        assert a[key].inventory_mol == pytest.approx(b[key].inventory_mol, abs=1e-12)
        assert a[key].reaction_current_A == pytest.approx(b[key].reaction_current_A, abs=1e-10)


@pytest.mark.parametrize("value,unit", [(0, "A"), (.2, "A"), (3.79, "V"), (2.7, "V"), (4.4, "V")])
def test_full_shared_residual_jacobian_all_material_and_electrical_columns(value, unit):
    s = model(maximum_current_A=.3)
    p, initial = program(value, unit), s.initial_state()
    f = composite_fields(s, initial, program=p)
    names = (*s.inventory_keys, "temperature", *s.coordinate_keys, *s.algebraic_names)
    y = np.r_[list(initial.amounts_mol.values()), initial.temperature_K,
              list(initial.coordinates.values()),
              [f["algebraic_values"][k] for k in s.algebraic_names]]
    n = len(s.inventory_keys)
    scales = np.r_[[1e-3]*n, 300, 1000, 4000, [.3]*len(s.reaction_keys), 1, .3, 1]
    def view(z):
        return CoupledStateView(dict(zip(s.inventory_keys, z[:n], strict=True)), z[n], s.volume_l,
            coordinates=dict(zip(s.coordinate_keys, z[n+1:n+3], strict=True)),
            algebraic_values=dict(zip(s.algebraic_names, z[n+3:], strict=True)))
    system = composite_system(s, p)
    def rhs(z):
        v = view(z)
        rates = assemble_coupled_rates(v, processes=system.processes,
            thermodynamics=system.thermodynamics, isothermal=True, heat_flux=system.heat_flux)
        return np.r_[[rates.amount_rates_mol_s[k] for k in s.inventory_keys],
                     0, [rates.coordinate_rates[k] for k in s.coordinate_keys],
                     -np.asarray(system.algebraic.residual(v))]
    actual = system.residual_rhs_jacobian(view(y), names).toarray()
    expected = finite_jacobian(rhs, y, scales)
    row_scale = np.r_[[1e-3]*n, 300, 1, 4, [1]*len(s.algebraic_names)]
    np.testing.assert_allclose(actual*scales/row_scale[:, None],
                              expected*scales/row_scale[:, None], atol=3e-5, rtol=5e-6)


def test_cv_and_reversed_stage_recompute_same_state_not_stale_algebraic():
    s = model(maximum_current_A=.3)
    one = advance_composite(s, s.initial_state(), program(.2), 4).final_state
    forward = composite_readout(s, one, .2)
    reverse = composite_readout(s, one, -.2)
    assert reverse.voltage_V > forward.voltage_V
    assert reverse.current_A == pytest.approx(-.2)
    cv = program((forward.voltage_V+reverse.voltage_V)/2, "V", start=4)
    resumed = advance_composite(s, one, cv, 4)
    reading = composite_readout(s, resumed.final_state,
                               composite_fields(s, resumed.final_state, program=cv)["current_A"])
    assert reading.voltage_V == pytest.approx(cv.value_at(8), abs=1e-8)
    assert composite_control_mode(s, resumed.final_state, cv)[0] == "constant_voltage"


def test_one_phase_surface_stop_then_reverse_keeps_actual_charge_and_inventory():
    s = model(identical=True)
    n = s.negative_phases
    s = replace(s, negative_phases=(replace(n[0], initial_fraction=.055),
                                  replace(n[1], initial_fraction=.055)))
    result = advance_composite(s, s.initial_state(), program(.3), 1000)
    assert result.stopping_event.startswith("battery:negative:phase:")
    assert 0 < result.actual_duration_s < 1000
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(
        .3*result.actual_duration_s, abs=1e-8)
    validate_composite_state(s, result.final_state)
    reversed_ = advance_composite(s, result.final_state, program(-.2), 2)
    assert reversed_.actual_duration_s == 2


@pytest.mark.parametrize("fault", ["missing", "negative", "capacity", "charge", "temperature"])
def test_invalid_persistent_phase_states_are_rejected_not_projected(fault):
    s = model()
    state = s.initial_state()
    key = s.inventory_keys[0]
    with pytest.raises(ValueError):
        if fault == "missing":
            state = replace(state, amounts_mol={k: v for k, v in state.amounts_mol.items()
                                               if k != key})
        elif fault in {"negative", "capacity"}:
            value = -1e-6 if fault == "negative" else 1
            state = replace(state, amounts_mol={**state.amounts_mol, key: value})
        elif fault == "charge":
            state = replace(state, coordinates={**state.coordinates, "discharged_charge_C": 1})
        else:
            state = replace(state, temperature_K=300)
        composite_system(s, program()).advance(state, 1)


def test_monotone_port_and_invalid_surface_root_do_not_fabricate_voltage():
    s = model(identical=True)
    values = [composite_readout(s, s.initial_state(), i).voltage_V for i in [-1, 0, 1]]
    assert values[0] > values[1] > values[2]
    slow = replace(s, negative_phases=tuple(replace(p, particle=replace(
        p.particle, diffusivity_m2_s=1e-17)) for p in s.negative_phases))
    with pytest.raises(ValueError, match="surface domain"):
        composite_readout(slow, slow.initial_state(), 1)


def test_shared_kernel_is_the_same_process_and_solver_not_a_parallel_implementation():
    s = model()
    assert composite_algebraic(s, program()).residual.__module__ == shared.__name__
    system = composite_system(s, program())
    assert system.residual_rhs_jacobian.__module__ == shared.__name__
    result = advance_composite(s, s.initial_state(), program(.2), 4)
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(.8, abs=1e-10)
    assert max(abs(x) for x in result.conservation_residuals.values()) < 1e-8
    assert F > 90000
