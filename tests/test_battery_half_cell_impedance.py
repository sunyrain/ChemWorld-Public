"""Functional EIS of the original finite-metal/working-particle residual."""

from dataclasses import replace

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
    half_cell_readout,
    half_cell_system,
    validate_half_cell_state,
)
from chemworld.physchem.battery_impedance import (
    battery_impedance,
    battery_impedance_system,
    half_cell_impedance,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    finite_jacobian,
)

FREQUENCIES = (.0001, .001, .01)


def source():
    return InputSource(DataSourceProvenance(
        "synthetic-half-cell-impedance", 0,
        "Authored functional finite-metal rest analysis inputs", "1", "CC0-1.0",
    ), "synthetic")


def metal(**changes):
    values = {"area_m2": .01, "thickness_m": 1e-4, "conductivity_S_m": 1.,
              "partial_molar_volume_m3_mol": 1.3e-5,
              "exchange_current_density_A_m2": 10., "reference_temperature_K": 298.15,
              "electrolyte_concentration_mol_m3": 1000.}
    values.update(changes)
    table = ScientificTable("metal-eis-functional", "parameters", tuple(
        QuantityColumn(k, u, "parameter", (0, 1e10), "Fixed synthetic input")
        for k, u in METAL_UNITS.items()), (source(),), (ScientificRow(
            "metal", source().provenance.source_id, "synthetic", "Li", "parameters",
            {k: InputValue(v, METAL_UNITS[k], None, "Fixed synthetic input")
             for k, v in values.items()}),))
    return LithiumMetalCounterElectrodeSpec(table, "metal")


def model(*, flat=False, cells=12, **changes):
    args = {"metal": metal(), "positive": ParticleSpec(
        cells=cells, mesh_exponent=1., radius_m=5e-6, diffusivity_m2_s=1e-12,
        exchange_current_scale_A_m2=2.),
        "positive_ocv": ElectrodeOCV("working", source(), (4.,) if flat else (4.2, -.8)),
        "maximum_current_A": .5}
    args.update(changes)
    return BatteryHalfCellSPMSpec(**args)


def equilibrium(spec, theta):
    initial = spec.initial_state()
    p = spec.positive
    moved = p.capacity_mol*(theta-spec.initial_positive_fraction)
    amounts = {key: float(volume*p.maximum_concentration_mol_m3*theta)
               for key, volume in zip(p.keys("positive"), p.mesh.cell_volumes_m3, strict=True)}
    amounts[METAL_KEY] = initial.amounts_mol[METAL_KEY]-moved
    return replace(initial, amounts_mol=amounts, coordinates={
        "discharged_charge_C": F*moved, "delivered_work_J": 0.})


def resistances(spec):
    p, m, theta = spec.positive, spec.metal.native_parameters, spec.initial_positive_fraction
    working_j0 = p.exchange_current_scale_A_m2*np.sqrt(theta*(1-theta))
    metal_rct = R*spec.temperature_K/(F*m["area_m2"]*m["exchange_current_density_A_m2"])
    working_rct = R*spec.temperature_K/(F*p.area_m2*working_j0)
    return spec.metal.resistance_ohm+metal_rct+working_rct


def test_actual_finite_reservoir_particle_charge_and_work_have_faradaic_phasors():
    spec = model()
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    indices = {v.name: i for i, v in enumerate(result.response.variables)}
    assert not any(key.startswith("negative:") for key in indices)
    assert set(spec.inventory_keys) <= set(indices)
    voltage = half_cell_readout(spec, spec.initial_state(), 0.).voltage_V
    for point in result.response.points:
        values, s = point.response[:, 0], 2j*np.pi*point.frequency_Hz
        working = sum(values[indices[key]] for key in spec.positive.keys("positive"))
        counter = values[indices[METAL_KEY]]
        assert values[indices["battery:current_A"]] == pytest.approx(1., abs=2e-12)
        assert s*F*counter == pytest.approx(-1., abs=2e-10)
        assert s*F*working == pytest.approx(1., abs=2e-10)
        assert counter+working == pytest.approx(0., abs=2e-12)
        assert s*values[indices["discharged_charge_C"]] == pytest.approx(1., abs=2e-10)
        assert s*values[indices["delivered_work_J"]] == pytest.approx(voltage, abs=2e-10)
    assert result.response.diagnostics["derivative_modes"]["state"] == "analytic"
    assert result.response.diagnostics["derivative_modes"]["parameter"] == "analytic"


def test_explicit_half_cell_impedance_entrypoint_reuses_shared_spectrum():
    spec = model()
    direct = battery_impedance(spec, FREQUENCIES)
    explicit = half_cell_impedance(spec, FREQUENCIES)
    assert explicit.to_dict() == direct.to_dict()
    with pytest.raises(ValueError, match="finite-metal half-cell"):
        half_cell_impedance(object(), FREQUENCIES)


def test_resolved_sphere_has_positive_working_surface_response_and_two_bv_terms():
    spec = model(cells=64)
    p = spec.positive
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    indices = {v.name: i for i, v in enumerate(result.response.variables)}
    for point, impedance in zip(result.response.points, result.impedance_ohm, strict=True):
        s = np.sqrt(2j*np.pi*point.frequency_Hz*p.radius_m**2/p.diffusivity_m2_s)
        gain = p.radius_m/(
            F*p.area_m2*p.diffusivity_m2_s*p.maximum_concentration_mol_m3
            * (s/np.tanh(s)-1)
        )
        expected = resistances(spec)+.8*gain
        assert impedance == pytest.approx(expected, rel=3e-4, abs=1e-6)
        shell_response = point.response[indices[p.keys("positive")[-1]], 0]
        surface = shell_response/(p.mesh.cell_volumes_m3[-1]*p.maximum_concentration_mol_m3)
        surface += p.mesh.spherical_surface_distance_m/(
            F*p.area_m2*p.diffusivity_m2_s*p.maximum_concentration_mol_m3)
        assert surface == pytest.approx(gain, rel=4e-4, abs=1e-6)
        assert impedance.real > 0 and impedance.imag < 0


def test_flat_working_ocv_has_only_true_metal_ohmic_and_two_kinetic_resistances():
    spec = model(flat=True)
    result = battery_impedance(spec, (1e-6, .01, 10.))
    assert result.success
    np.testing.assert_allclose(result.impedance_ohm, resistances(spec), rtol=2e-10, atol=1e-11)


@pytest.mark.parametrize("parameter,new_value", [
    ("conductivity_S_m", .1), ("exchange_current_density_A_m2", 2.),
])
def test_metal_parameter_changes_only_the_declared_real_interface_resistance(parameter, new_value):
    spec = model()
    changed = replace(spec, metal=metal(**{parameter: new_value}))
    first, second = battery_impedance(spec, FREQUENCIES), battery_impedance(changed, FREQUENCIES)
    assert first.success and second.success
    np.testing.assert_allclose(np.array(second.impedance_ohm)-first.impedance_ohm,
                               resistances(changed)-resistances(spec), rtol=1e-10, atol=1e-11)
    indices = {v.name: i for i, v in enumerate(first.response.variables)}
    rows = [indices[key] for key in spec.inventory_keys]
    for a, b in zip(first.response.points, second.response.points, strict=True):
        np.testing.assert_allclose(a.response[rows], b.response[rows], rtol=1e-10, atol=1e-12)


def test_original_analytic_half_cell_residual_jacobian_and_control_parameter_are_used():
    spec = model()
    system, initial = battery_impedance_system(spec, spec.initial_state())
    evaluator = ResidualEvaluator(system)
    rates = np.zeros(len(initial))
    analytic = evaluator.derivative("state", 0., initial, rates).toarray()
    numerical = finite_jacobian(lambda y: evaluator.value(0., y, rates), initial, system.scales)
    np.testing.assert_allclose(
        analytic*system.scales/system.residual_scales[:, None],
        numerical*system.scales/system.residual_scales[:, None], rtol=2e-5, atol=2e-7,
    )
    names = [equation.name for equation in system.equations]
    assert "half-cell:control" in names and "battery:control" not in names
    assert not initial.flags.writeable


@pytest.mark.parametrize("limit", [.002, 2.])
def test_current_limit_is_a_scale_not_a_rescaling_of_the_physical_spectrum(limit):
    spec = model()
    first = battery_impedance(spec, FREQUENCIES)
    second = battery_impedance(replace(spec, maximum_current_A=limit), FREQUENCIES)
    assert first.success and second.success
    np.testing.assert_allclose(first.impedance_ohm, second.impedance_ohm, rtol=1e-10, atol=1e-11)


def test_depleted_metal_is_valid_for_rest_and_replating_but_not_two_sided_eis():
    spec = model(metal=metal(thickness_m=1e-6))
    theta = spec.initial_positive_fraction+spec.metal.initial_amount_mol/spec.positive.capacity_mol
    # Author the exact depleted inventory directly, not by subtracting two
    # nearly equal floating-point amounts after a fraction round trip.
    amounts = {key: float(volume*spec.positive.maximum_concentration_mol_m3*theta)
               for key, volume in zip(spec.positive.keys("positive"),
                                      spec.positive.mesh.cell_volumes_m3, strict=True)}
    state = replace(spec.initial_state(), amounts_mol={**amounts, METAL_KEY: 0.}, coordinates={
        "discharged_charge_C": F*spec.metal.initial_amount_mol, "delivered_work_J": 7.})
    validate_half_cell_state(spec, state)
    for current in (0., -.1):
        system = half_cell_system(spec, ScalarControlProgram(((0., current),), "A"))
        depleted = next(event for event in system.all_events
                        if event.event_id == "half-cell:metal:depleted")
        assert depleted.value(state) > 0
    with pytest.raises(ResidualDomainError, match="strictly positive metal"):
        battery_impedance(spec, FREQUENCIES, state=state)
    assert state.amounts_mol[METAL_KEY] == 0.


def test_strictly_positive_small_metal_has_no_implicit_analysis_inventory_floor():
    spec = model(metal=metal(thickness_m=1e-14))
    assert 0 < spec.metal.initial_amount_mol < 1e-10
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    # This derivative per ampere is not a finite-amplitude depletion guarantee.
    indices = {v.name: i for i, v in enumerate(result.response.variables)}
    for point in result.response.points:
        assert point.response[indices[METAL_KEY], 0] == pytest.approx(
            -1/(2j*np.pi*point.frequency_Hz*F), rel=1e-10)


@pytest.mark.parametrize("upper", [False, True])
def test_exact_working_surface_mol_boundary_is_not_a_two_sided_eis_point(upper):
    spec = model(flat=True)
    state = equilibrium(spec, 1-spec.surface_margin if upper else spec.surface_margin)
    validate_half_cell_state(spec, state)
    with pytest.raises(ResidualDomainError, match="working surface"):
        battery_impedance(spec, FREQUENCIES, state=state)


@pytest.mark.parametrize("boundary", ["minimum_voltage_V", "maximum_voltage_V"])
def test_voltage_stopping_face_is_not_a_two_sided_eis_point(boundary):
    spec = model(flat=True)
    voltage = half_cell_readout(spec, spec.initial_state(), 0.).voltage_V
    spec = replace(spec, **{boundary: voltage})
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(spec, FREQUENCIES)


def test_zero_current_and_uniform_surface_do_not_hide_nonstationary_radial_inventory():
    spec = model()
    state = spec.initial_state()
    keys = spec.positive.keys("positive")
    amounts = dict(state.amounts_mol)
    delta = amounts[keys[0]]*.1
    amounts[keys[0]] -= delta
    amounts[keys[1]] += delta
    state = replace(state, amounts_mol=amounts)
    validate_half_cell_state(spec, state)
    assert half_cell_readout(spec, state, 0.).working_surface_fraction == pytest.approx(
        spec.initial_positive_fraction)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        battery_impedance(spec, FREQUENCIES, state=state)
