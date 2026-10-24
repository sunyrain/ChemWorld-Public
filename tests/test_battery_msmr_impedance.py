"""Functional spectra of the actual total-Li MSMR residual, without new dynamics."""

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
from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.battery_msmr import (
    BatteryMSMRSPMSpec,
    F,
    R,
    msmr_fields,
    msmr_particle_transport,
    validate_msmr_state,
)
from chemworld.physchem.battery_msmr_materials import MSMR_UNITS, MSMRMaterialSpec
from chemworld.physchem.coupled_system import CoupledStateView
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    finite_jacobian,
)

FREQUENCIES = (.0001, .001, .01)


def material(name, *, single=False, width=1.):
    source = InputSource(DataSourceProvenance(
        "synthetic-msmr-impedance", 0, "Authored functional MSMR rest inputs", "1", "CC0-1.0",
    ), "synthetic")
    base = .2 if name == "negative" else 4.
    rows = ([(1., base, width, 3., .5)] if single else [
        (.6, base-.035, .8, 4., .35), (.4, base+.05, 1.6, 2., .65),
    ])
    identities = tuple(f"reaction-{i}" for i in range(len(rows)))
    table = ScientificTable("msmr-eis-"+name, "parameters", tuple(
        QuantityColumn(k, u, "parameter", (-100., 1e6), "Fixed synthetic input")
        for k, u in MSMR_UNITS.items()), (source,), tuple(
            ScientificRow(identity, source.provenance.source_id, "synthetic", name+identity,
                          "parameters", {k: InputValue(v, MSMR_UNITS[k], None,
                                                      "Fixed synthetic input")
                                         for k, v in zip(MSMR_UNITS, row, strict=True)})
            for identity, row in zip(identities, rows, strict=True)))
    return MSMRMaterialSpec(table, identities, name)


def model(*, single=False, width=1., cells=12, **changes):
    particle = ParticleSpec(cells=cells, mesh_exponent=1., radius_m=5e-6,
                            diffusivity_m2_s=1e-12)
    args = {"negative": particle, "positive": particle,
            "negative_material": material("negative", single=single, width=width),
            "positive_material": material("positive", single=single, width=width),
            "maximum_current_A": .5, "resistance_ohm": .02}
    args.update(changes)
    return BatteryMSMRSPMSpec(**args)


def constitutive(spec, electrode):
    """Independent scalar derivatives at the declared uniform operating point."""
    mat = getattr(spec, electrode+"_material")
    theta = getattr(spec, "initial_"+electrode+"_fraction")
    u = mat.potential(theta, spec.temperature_K)
    rows = np.asarray(mat.reaction_parameters)
    weight, u0, width, jref, alpha = rows.T
    f = F/(R*spec.temperature_K)
    xj = weight/(1+np.exp(f*(u-u0)/width))
    xju = -f/width*xj*(1-xj/weight)
    j0 = (jref*xj**(width*alpha)*(weight-xj)**(width*(1-alpha))
          * (spec.electrolyte_concentration_mol_m3
             / mat.reference_electrolyte_concentration_mol_m3)**(1-alpha))
    assert sum(xj) == pytest.approx(theta, rel=2e-12, abs=1e-14)
    return u, xju, f*j0


def chemical_diffusivity(spec, electrode):
    p = getattr(spec, electrode)
    theta = getattr(spec, "initial_"+electrode+"_fraction")
    return -p.diffusivity_m2_s*F/(R*spec.temperature_K)*theta*(1-theta)/sum(
        constitutive(spec, electrode)[1])


@pytest.mark.parametrize("width", [1., 2.])
def test_single_reaction_uniform_jacobian_is_exact_width_times_linear_fv(width):
    spec = model(single=True, width=width)
    for electrode in ("negative", "positive"):
        p = getattr(spec, electrode)
        d = chemical_diffusivity(spec, electrode)
        assert d == pytest.approx(width*p.diffusivity_m2_s, rel=2e-13)
        _, jacobian, _ = msmr_particle_transport(spec, spec.initial_state(), electrode)
        expected = np.zeros_like(jacobian)
        for i, (area, distance) in enumerate(zip(
            p.mesh.face_areas_m2[1:-1], np.diff(p.mesh.centers_m), strict=True
        )):
            conductance = width*p.diffusivity_m2_s*area/distance
            left, right = conductance/p.mesh.cell_volumes_m3[i:i+2]
            expected[i, i] -= left
            expected[i, i+1] += right
            expected[i+1, i] += left
            expected[i+1, i+1] -= right
        np.testing.assert_allclose(jacobian, expected, rtol=2e-10, atol=1e-12)


@pytest.mark.parametrize("single,width", [(True, 1.), (True, 2.), (False, 1.)])
def test_spectrum_converges_to_continuous_spheres_using_actual_chemical_diffusion(single, width):
    spec = model(single=single, width=width, cells=64)
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    for point, impedance in zip(result.response.points, result.impedance_ohm, strict=True):
        expected = spec.resistance_ohm
        for electrode in ("negative", "positive"):
            p = getattr(spec, electrode)
            _, xu, kappa = constitutive(spec, electrode)
            d = chemical_diffusivity(spec, electrode)
            s = np.sqrt(2j*np.pi*point.frequency_Hz*p.radius_m**2/d)
            gain = p.radius_m/(F*p.area_m2*d*p.maximum_concentration_mol_m3
                               * (s/np.tanh(s)-1))
            expected += 1/(p.area_m2*sum(kappa))-gain/sum(xu)
        # Original MSMR surface potential is linearly extrapolated. This finite
        # mesh comparison does not substitute the ordinary SPM half-cell value.
        assert impedance == pytest.approx(expected, rel=1e-3, abs=1e-6)


def test_actual_multi_reaction_spectrum_keeps_total_li_charge_and_work_phasors():
    spec = model()
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    indices = {v.name: i for i, v in enumerate(result.response.variables)}
    assert set(indices) == set(spec.inventory_keys+spec.coordinate_keys+(
        "battery:current_A", "battery:voltage_V"))
    v0 = msmr_fields(spec, spec.initial_state(), current_A=0.)["voltage_V"]
    for point in result.response.points:
        raw, s = point.response[:, 0], 2j*np.pi*point.frequency_Hz
        total = 0.
        for electrode, polarity in (("negative", 1.), ("positive", -1.)):
            amount = sum(raw[indices[k]] for k in getattr(spec, electrode).keys(electrode))
            assert s*F*amount == pytest.approx(-polarity, abs=2e-10)
            total += amount
        assert total == pytest.approx(0., abs=2e-12)
        assert raw[indices["battery:current_A"]] == pytest.approx(1., abs=2e-12)
        assert s*raw[indices["discharged_charge_C"]] == pytest.approx(1., abs=2e-10)
        assert s*raw[indices["delivered_work_J"]] == pytest.approx(v0, abs=2e-10)
    assert result.response.diagnostics["derivative_modes"]["state"] == "analytic"


def test_real_branch_observations_share_eta_and_occupancy_but_not_separate_inventories():
    spec = model()
    state = spec.initial_state()
    result = battery_impedance(spec, (.001,))
    assert result.success
    indices = {v.name: i for i, v in enumerate(result.response.variables)}
    raw = result.response.points[0].response[:, 0]
    for electrode, polarity in (("negative", 1.), ("positive", -1.)):
        p = getattr(spec, electrode)
        _, xu, kappa = constitutive(spec, electrode)
        delta_u = np.array([raw[indices[k]] for k in p.keys(electrode)]) / (
            p.maximum_concentration_mol_m3*p.mesh.cell_volumes_m3*sum(xu))
        surface_u = 1.5*delta_u[-1]-.5*delta_u[-2]
        branch_current = polarity*kappa/sum(kappa)
        branch_occupancy = xu*surface_u
        assert sum(branch_current) == pytest.approx(polarity)
        assert sum(branch_occupancy) == pytest.approx(sum(xu)*surface_u)
        # Quasi-equilibrium occupancy partition and interfacial reaction shares
        # differ. Only the total inventory obeys each electrode's Faraday law.
        assert np.max(abs(kappa/sum(kappa)-xu/sum(xu))) > .01
        for component in (np.real, np.imag):
            step = 1e-6
            observed = []
            for sign in (-1., 1.):
                view = CoupledStateView(
                    {k: state.amounts_mol[k]+sign*step*component(raw[indices[k]])
                     for k in spec.inventory_keys}, state.temperature_K, state.volume_L,
                    state.time_s, coordinates=state.coordinates,
                )
                observed.append(msmr_fields(
                    spec, view, current_A=sign*step*component(raw[indices["battery:current_A"]])
                ))
            for field, expected in (
                ("reaction_current_densities_A_m2", branch_current/p.area_m2),
                ("surface_occupancies", branch_occupancy),
                ("source_surface_occupancies", branch_occupancy),
            ):
                derivative = (np.array(observed[1][electrode+"_"+field])
                              - observed[0][electrode+"_"+field])/(2*step)
                np.testing.assert_allclose(derivative, component(expected),
                                           rtol=3e-6, atol=2e-8)


def test_full_original_jacobian_matches_scaled_independent_differences():
    spec = model()
    system, initial = battery_impedance_system(spec, spec.initial_state())
    evaluator = ResidualEvaluator(system)
    rate = np.zeros(len(initial))
    analytic = evaluator.derivative("state", 0., initial, rate).toarray()
    numerical = finite_jacobian(lambda y: evaluator.value(0., y, rate), initial, system.scales)
    np.testing.assert_allclose(
        analytic*system.scales/system.residual_scales[:, None],
        numerical*system.scales/system.residual_scales[:, None], rtol=3e-5, atol=3e-7,
    )
    assert "msmr:control" in [e.name for e in system.equations]
    assert not initial.flags.writeable


def test_series_resistance_changes_only_the_real_port_impedance():
    spec = model()
    first = battery_impedance(spec, FREQUENCIES)
    second = battery_impedance(replace(spec, resistance_ohm=.19), FREQUENCIES)
    assert first.success and second.success
    np.testing.assert_allclose(np.array(second.impedance_ohm)-first.impedance_ohm,
                               .17, rtol=1e-10, atol=1e-11)
    rows = [i for i, v in enumerate(first.response.variables) if v.name in spec.inventory_keys]
    for a, b in zip(first.response.points, second.response.points, strict=True):
        np.testing.assert_allclose(a.response[rows], b.response[rows], rtol=1e-10, atol=1e-12)


def test_nonuniform_li_is_rejected_even_with_zero_port_current_and_same_total_inventory():
    spec = model()
    state = spec.initial_state()
    keys = spec.negative.keys("negative")
    amounts = dict(state.amounts_mol)
    delta = amounts[keys[0]]*.1
    amounts[keys[0]] -= delta
    amounts[keys[1]] += delta
    state = replace(state, amounts_mol=amounts)
    validate_msmr_state(spec, state)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        battery_impedance(spec, FREQUENCIES, state=state)


@pytest.mark.parametrize("electrode", ["negative", "positive"])
@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("shell", [0, -1])
def test_every_radial_stopping_face_is_strictly_excluded_in_mol(electrode, upper, shell):
    spec = model(initial_negative_fraction=.5, initial_positive_fraction=.5)
    state = spec.initial_state()
    p = getattr(spec, electrode)
    key = p.keys(electrode)[shell]
    amounts = dict(state.amounts_mol)
    capacity = p.mesh.cell_volumes_m3[shell]*p.maximum_concentration_mol_m3
    amounts[key] = capacity*(1-spec.surface_margin if upper else spec.surface_margin)
    change = amounts[key]-state.amounts_mol[key]
    other = "positive" if electrode == "negative" else "negative"
    other_p = getattr(spec, other)
    other_theta = .5-change/other_p.capacity_mol
    amounts.update({k: float(v*other_p.maximum_concentration_mol_m3*other_theta)
                    for k, v in zip(other_p.keys(other), other_p.mesh.cell_volumes_m3,
                                    strict=True)})
    charge = -F*change if electrode == "negative" else F*change
    state = replace(state, amounts_mol=amounts, coordinates={
        "discharged_charge_C": charge, "delivered_work_J": 0.})
    validate_msmr_state(spec, state)
    with pytest.raises(ResidualDomainError, match="stopping boundary: "+key):
        battery_impedance(spec, FREQUENCIES, state=state)


@pytest.mark.parametrize("upper", [False, True])
def test_actual_extrapolated_surface_can_fail_before_any_radial_cell_boundary(upper):
    spec = model(initial_negative_fraction=.5, initial_positive_fraction=.5)
    state = spec.initial_state()
    p = spec.negative
    theta = np.full(p.cells, .5)
    theta[-1] = .989 if upper else .011
    volume = p.mesh.cell_volumes_m3
    theta[:-2] += (p.active_volume_m3*.5-np.dot(volume, theta))/sum(volume[:-2])
    assert np.all((theta > spec.surface_margin) & (theta < 1-spec.surface_margin))
    amounts = {**state.amounts_mol, **dict(zip(p.keys("negative"),
               volume*p.maximum_concentration_mol_m3*theta, strict=True))}
    state = replace(state, amounts_mol=amounts)
    validate_msmr_state(spec, state)
    value = msmr_fields(spec, state, current_A=0.)["negative_surface_fraction"]
    assert value > 1-spec.surface_margin if upper else value < spec.surface_margin
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(spec, FREQUENCIES, state=state)


@pytest.mark.parametrize("boundary", ["minimum_voltage_V", "maximum_voltage_V"])
def test_actual_voltage_face_is_excluded(boundary):
    spec = model()
    voltage = msmr_fields(spec, spec.initial_state(), current_A=0.)["voltage_V"]
    spec = replace(spec, **{boundary: voltage})
    with pytest.raises(ResidualDomainError, match="stopping boundary"):
        battery_impedance(spec, FREQUENCIES)
