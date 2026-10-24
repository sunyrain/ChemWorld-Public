"""Functional rest spectra of the original three-phase particle residual."""

import json
from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_composite import (
    ActiveMaterialPhaseSpec,
    BatteryCompositeSPMSpec,
    composite_fields,
)
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec, F, ParticleSpec, R
from chemworld.physchem.coupled_system import CoupledStateView
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    finite_jacobian,
)

FREQUENCIES = (0.001, 0.01, 0.1)


def phase(identity, volume, radius, diffusion, cmax, exchange, theta, polynomial):
    source = InputSource(DataSourceProvenance(
        "synthetic-composite-impedance", 0,
        "Authored independent phase parameters for functional rest analysis",
        "1", "CC0-1.0",
    ), "synthetic")
    return ActiveMaterialPhaseSpec(
        identity,
        ParticleSpec(cells=6, active_volume_m3=volume, radius_m=radius,
                     diffusivity_m2_s=diffusion, maximum_concentration_mol_m3=cmax,
                     exchange_current_scale_A_m2=exchange),
        ElectrodeOCV(identity, source, polynomial), theta, source,
    )


def model(*, identical=False, flat=False):
    first = phase("alpha", .6e-6, 5e-6, 2e-13, 30000., 2., .4, (.3, -.25))
    second = phase("beta", .4e-6, 8e-6, 7e-14, 42000., 1.2, .6, (.5, -.5))
    positive = phase("positive", 1.6e-6, 6e-6, 1e-13, 35000., 1.6, .3, (4.2, -.7))
    if identical:
        second = replace(first, phase_id="beta",
                         particle=replace(first.particle, active_volume_m3=.4e-6))
    if flat:
        first = replace(first, ocv=replace(first.ocv, polynomial_V=(.2,)))
        second = replace(second, ocv=replace(second.ocv, polynomial_V=(.2,)))
        positive = replace(positive, ocv=replace(positive.ocv, polynomial_V=(4.,)))
    return BatteryCompositeSPMSpec((first, second), positive)


def snapshot(state):
    return {field.name: dict(value) if isinstance(value, Mapping) else value
            for field in fields(state) for value in (getattr(state, field.name),)}


def phase_responses(spec, response, point):
    indices = {v.name: i for i, v in enumerate(response.variables)}
    perturbation = point.response[:, 0]
    currents = spec.reaction_map @ perturbation[[indices[k] for k in spec.algebraic_names]]
    return {
        material.phase_id: (currents[i], sum(perturbation[indices[k]] for k in keys))
        for i, (material, (_, _, _, keys, _)) in enumerate(zip(
            spec.phases, spec.particle_bins, strict=True))
    }


def test_identical_materials_reduce_to_actual_single_material_spm_spectrum():
    spec = model(identical=True)
    first, second, positive = spec.phases
    single = BatterySPMSpec(
        negative=replace(first.particle, active_volume_m3=(
            first.particle.active_volume_m3 + second.particle.active_volume_m3)),
        positive=positive.particle, initial_negative_fraction=first.initial_fraction,
        initial_positive_fraction=positive.initial_fraction,
        negative_ocv=first.ocv, positive_ocv=positive.ocv,
    )
    result = battery_impedance(spec, FREQUENCIES)
    expected = battery_impedance(single, FREQUENCIES)
    assert result.success and expected.success
    np.testing.assert_allclose(result.impedance_ohm, expected.impedance_ohm,
                               rtol=2e-10, atol=2e-11)
    for point in result.response.points:
        values = phase_responses(spec, result.response, point)
        assert values["alpha"][0] == pytest.approx(.6, abs=1e-11)
        assert values["beta"][0] == pytest.approx(.4, abs=1e-11)


def test_heterogeneous_equilibrium_retains_three_phase_faradaic_responses():
    spec = model()
    assert spec.negative_phases[0].initial_fraction != spec.negative_phases[1].initial_fraction
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    indices = {v.name: i for i, v in enumerate(result.response.variables)}
    fractions = []
    voltage = composite_fields(spec, spec.initial_state(), current_A=0)["voltage_V"]
    for point, impedance in zip(result.response.points, result.impedance_ohm, strict=True):
        perturbation = point.response[:, 0]
        values = phase_responses(spec, result.response, point)
        s = 2j*np.pi*point.frequency_Hz
        assert impedance.real > 0 and impedance.imag < 0
        assert perturbation[indices["battery:current_A"]] == pytest.approx(1., abs=1e-12)
        assert values["alpha"][0] + values["beta"][0] == pytest.approx(1., abs=1e-12)
        assert values["positive"][0] == pytest.approx(-1., abs=1e-12)
        for current, amount in values.values():
            assert s*F*amount == pytest.approx(-current, abs=2e-11)
        assert sum(amount for _, amount in values.values()) == pytest.approx(0., abs=1e-12)
        assert perturbation[indices["discharged_charge_C"]] == pytest.approx(1/s, rel=1e-11)
        assert perturbation[indices["delivered_work_J"]] == pytest.approx(voltage/s, rel=1e-11)
        fractions.append(values["alpha"][0])
    # Sharing is solved at each frequency, not assigned in proportion to area.
    assert abs(fractions[0]-fractions[-1]) > .01
    assert result.response.diagnostics["derivative_modes"]["state"] == "analytic"
    assert result.response.diagnostics["derivative_modes"]["parameter"] == "analytic"
    json.dumps(result.response.to_dict(), allow_nan=False)


def test_full_frequency_residual_derivative_matches_independent_differences():
    spec = model()
    system, initial = battery_impedance_system(spec, spec.initial_state())
    evaluator = ResidualEvaluator(system)
    analytic = evaluator.derivative("state", 0, initial, np.zeros(len(initial))).toarray()
    numerical = finite_jacobian(
        lambda y: evaluator.value(0, y, np.zeros(len(y))), initial, system.scales,
    )
    np.testing.assert_allclose(
        analytic*system.scales/system.residual_scales[:, None],
        numerical*system.scales/system.residual_scales[:, None], rtol=2e-5, atol=2e-7,
    )
    assert not initial.flags.writeable


@pytest.mark.parametrize("equal_area", [False, True])
def test_phase_identity_and_pivot_reordering_preserve_all_physical_responses(equal_area):
    spec = model()
    if equal_area:
        first, second = spec.negative_phases
        # Exactly tied areas change the physical balance pivot upon permutation.
        second = replace(second, particle=replace(
            second.particle, radius_m=first.particle.radius_m,
            active_volume_m3=first.particle.active_volume_m3))
        spec = replace(spec, negative_phases=(first, second))
    swapped = replace(spec, negative_phases=tuple(reversed(spec.negative_phases)))
    if equal_area:
        assert spec.current_coordinates[0][0] == swapped.current_coordinates[0][0] == 0
    first = battery_impedance(spec, FREQUENCIES)
    second = battery_impedance(swapped, FREQUENCIES)
    assert first.success and second.success
    np.testing.assert_allclose(first.impedance_ohm, second.impedance_ohm, atol=1e-11, rtol=1e-10)
    for one, two in zip(first.response.points, second.response.points, strict=True):
        a = phase_responses(spec, first.response, one)
        b = phase_responses(swapped, second.response, two)
        for identity in a:
            np.testing.assert_allclose(a[identity], b[identity], atol=1e-11, rtol=1e-10)


def test_series_resistance_is_exact_real_port_shift():
    spec = model()
    first = battery_impedance(spec, FREQUENCIES)
    second = battery_impedance(replace(spec, series_resistance_ohm=.17), FREQUENCIES)
    assert first.success and second.success
    np.testing.assert_allclose(np.array(second.impedance_ohm)-first.impedance_ohm,
                               .17, rtol=1e-10, atol=1e-11)
    for one, two in zip(first.response.points, second.response.points, strict=True):
        np.testing.assert_allclose(
            list(phase_responses(spec, first.response, one).values()),
            list(phase_responses(spec, second.response, two).values()), atol=1e-11, rtol=1e-10,
        )


def test_flat_ocvs_have_independent_parallel_bv_resistance_at_every_frequency():
    spec = replace(model(flat=True), series_resistance_ohm=.07)
    resistance = []
    for material in spec.phases:
        p, theta = material.particle, material.initial_fraction
        area = 3*p.active_volume_m3/p.radius_m
        j0 = p.exchange_current_scale_A_m2*np.sqrt(theta*(1-theta))
        resistance.append(R*spec.temperature_K/(F*area*j0))
    expected = .07 + 1/(1/resistance[0]+1/resistance[1]) + resistance[2]
    result = battery_impedance(spec, (1e-5, .01, 10.))
    assert result.success
    np.testing.assert_allclose(result.impedance_ohm, expected, rtol=1e-10, atol=1e-11)


def test_zero_terminal_current_does_not_hide_internal_transfer():
    spec = model()
    first, second = spec.negative_phases
    spec = replace(spec, negative_phases=(first, replace(second, initial_fraction=.5)))
    state = spec.initial_state()
    before = snapshot(state)
    currents = composite_fields(spec, state, current_A=0)["reaction_current_A"]
    assert abs(currents[0]) > 1e-4
    assert sum(currents[:2]) == pytest.approx(0., abs=1e-12)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        battery_impedance(spec, FREQUENCIES, state=state)
    assert snapshot(state) == before


def test_zero_phase_currents_do_not_hide_internal_radial_relaxation():
    spec = model()
    state = spec.initial_state()
    amounts = dict(state.amounts_mol)
    keys = spec.particle_bins[0][3]
    delta = .1*amounts[keys[0]]
    amounts[keys[0]] -= delta
    amounts[keys[1]] += delta
    state = replace(state, amounts_mol=amounts)
    before = snapshot(state)
    np.testing.assert_allclose(
        composite_fields(spec, state, current_A=0)["reaction_current_A"], 0., atol=1e-12)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        battery_impedance(spec, FREQUENCIES, state=state)
    assert snapshot(state) == before


def uniform_amounts(spec, fractions):
    return {key: float(volume*p.maximum_concentration_mol_m3*theta)
            for (_, _, p, keys, _), theta in zip(spec.particle_bins, fractions, strict=True)
            for key, volume in zip(keys, p.mesh.cell_volumes_m3, strict=True)}


@pytest.mark.parametrize("phase_index", [0, 1, 2])
@pytest.mark.parametrize("upper", [False, True])
def test_each_material_stopping_surface_is_not_a_two_sided_operating_point(phase_index, upper):
    spec = model(flat=True)
    initial = spec.initial_state()
    fractions = [p.initial_fraction for p in spec.phases]
    target = 1-spec.surface_margin if upper else spec.surface_margin
    changed = spec.phases[phase_index].particle.capacity_mol*(target-fractions[phase_index])
    fractions[phase_index] = target
    if phase_index < 2:
        fractions[2] -= changed/spec.positive_phase.particle.capacity_mol
        charge = -F*changed
    else:
        capacity = sum(p.particle.capacity_mol for p in spec.negative_phases)
        for i in (0, 1):
            fractions[i] -= changed/capacity
        charge = F*changed
    # Test a feasible endpoint; for a large positive excursion declare a larger
    # negative capacity before constructing the same conservation-preserving state.
    if not all(0 < value < 1 for value in fractions):
        spec = replace(spec, negative_phases=tuple(replace(p, particle=replace(
            p.particle, active_volume_m3=4*p.particle.active_volume_m3))
            for p in spec.negative_phases))
        initial = spec.initial_state()
        fractions = [p.initial_fraction for p in spec.phases]
        changed = spec.phases[phase_index].particle.capacity_mol*(target-fractions[phase_index])
        fractions[phase_index] = target
        for i in (0, 1):
            fractions[i] -= changed/sum(p.particle.capacity_mol for p in spec.negative_phases)
        charge = F*changed
    state = replace(initial, amounts_mol=uniform_amounts(spec, fractions),
                    coordinates={"discharged_charge_C": charge, "delivered_work_J": 0.})
    before = snapshot(state)
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(spec, (.01,), state=state)
    assert snapshot(state) == before


@pytest.mark.parametrize("boundary", ["minimum_voltage_V", "maximum_voltage_V"])
def test_voltage_stopping_boundary_is_rejected(boundary):
    spec = model(flat=True)
    voltage = composite_fields(spec, spec.initial_state(), current_A=0)["voltage_V"]
    spec = replace(spec, **{boundary: voltage})
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(spec, (.01,))


def test_existing_charge_work_and_receipts_are_preserved_at_valid_saved_equilibrium():
    spec = model()
    initial = spec.initial_state()
    # A different authored equilibrium on the same Faradaic trajectory.
    fractions = [.36, .58, spec.positive_phase.initial_fraction]
    removed = sum(p.particle.capacity_mol*(p.initial_fraction-theta)
                  for p, theta in zip(spec.negative_phases, fractions[:2], strict=True))
    fractions[2] += removed/spec.positive_phase.particle.capacity_mol
    state = replace(initial, amounts_mol=uniform_amounts(spec, fractions), time_s=17.,
                    coordinates={"discharged_charge_C": F*removed, "delivered_work_J": 123.},
                    heat_loss_J=.7, work_J=.2, boundary_heat_J=.3,
                    exchange_energy_J={"terminal": -123.})
    before = snapshot(state)
    result = battery_impedance(spec, FREQUENCIES, state=state)
    assert result.success
    assert result.response.diagnostics["time_s"] == 17.
    assert result.response.diagnostics["physical_time_advanced_s"] == 0.
    assert snapshot(state) == before


def test_stale_algebraic_view_is_reclosed_without_mutating_inventory_or_view():
    spec = model()
    state = spec.initial_state()
    stale = dict.fromkeys(spec.algebraic_names, 19.)
    stale["battery:current_A"] = 0.  # Must not trigger the old current-only cache shortcut.
    view = CoupledStateView(state.amounts_mol, state.temperature_K, state.volume_L,
                            state.time_s, coordinates=state.coordinates, algebraic_values=stale)
    before = snapshot(view)
    result = battery_impedance(spec, FREQUENCIES, state=view)
    fresh = battery_impedance(spec, FREQUENCIES, state=state)
    assert result.success and fresh.success
    for a, b in zip(result.response.points, fresh.response.points, strict=True):
        np.testing.assert_array_equal(a.response, b.response)
    assert snapshot(view) == before
