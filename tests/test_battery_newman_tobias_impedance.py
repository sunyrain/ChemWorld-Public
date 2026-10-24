"""Functional rest EIS of two mean inventories and the original porous circuit."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
from chemworld.physchem.battery_newman_tobias import (
    BatteryNewmanTobiasSpec,
    F,
    R,
    newman_tobias_fields,
    validate_newman_tobias_state,
)
from chemworld.physchem.battery_porous_electrical import (
    PorousElectrodeGeometrySpec,
    linear_electrical,
)
from chemworld.physchem.coupled_system import CoupledStateView
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEvaluator,
    finite_jacobian,
)

FREQUENCIES = (1e-4, .01, 1.)


def model(*, cells=(12, 5, 10), **changes):
    arguments = {"geometry": PorousElectrodeGeometrySpec(cells=cells)}
    arguments.update(changes)
    return BatteryNewmanTobiasSpec(**arguments)


def network(spec):
    """Independent continuous constant-coefficient BVP, no production maps.

    Unit port current; eta, ionic current density, liquid potential and J are
    propagated by a matrix exponential. OCV perturbations are set to zero for
    the real network. They are added separately from the two mean inventories.
    """
    geometry = spec.geometry
    area = geometry.area_m2
    current_density = 1 / area
    kappas = geometry.conductivity_S_m * np.power(geometry.porosities, geometry.bruggeman)
    electrode_solutions = []
    liquid_left = None
    for name, region, i_left, i_right in (
        ("negative", 0, 0., current_density),
        ("positive", 2, current_density, 0.),
    ):
        electrode = getattr(spec, name)
        length = geometry.lengths_m[region]
        sigma = geometry.solid_conductivity_S_m[region // 2]
        kappa = kappas[region]
        theta = getattr(spec, "initial_" + name + "_fraction")
        j0 = electrode.exchange_current_scale_A_m2 * np.sqrt(
            spec.electrolyte_concentration_mol_m3
            / spec.reference_electrolyte_concentration_mol_m3 * theta * (1-theta)
        )
        specific_area = 3 * electrode.active_volume_m3 / (electrode.radius_m * area * length)
        g = specific_area * F * j0 / (R * spec.temperature_K)
        matrix = np.array([
            [0., 1/sigma+1/kappa, 0., -1/sigma],
            [g, 0., 0., 0.],
            [0., -1/kappa, 0., 0.],
            [0., 0., 0., 0.],
        ])
        end_map = expm(length * matrix)
        eta_left = (i_right-end_map[1, 1]*i_left-end_map[1, 3]*current_density) / end_map[1, 0]
        if liquid_left is None:
            liquid_left = -eta_left  # negative collector solid potential is zero
        initial = np.array([eta_left, i_left, liquid_left, current_density])
        end = end_map @ initial
        assert end[1] == pytest.approx(i_right, abs=2e-11)
        electrode_solutions.append((matrix, initial, length, sigma, kappa, g))
        liquid_left = end[2] - current_density*geometry.lengths_m[1]/kappas[1]
    voltage = end[2]+end[0]
    loss = current_density**2 * geometry.lengths_m[1]/kappas[1]
    for matrix, initial, length, sigma, kappa, g in electrode_solutions:
        def density(x, matrix=matrix, initial=initial, sigma=sigma, kappa=kappa, g=g):
            eta, ionic, _, total = expm(x*matrix) @ initial
            return (total-ionic)**2/sigma + ionic**2/kappa + g*eta**2

        loss += quad(density, 0., length, epsabs=1e-10, epsrel=1e-11)[0]
    resistance = -voltage + spec.resistance_ohm
    assert resistance == pytest.approx(area*loss+spec.resistance_ohm, rel=2e-12, abs=2e-13)
    return resistance, electrode_solutions


def pole(spec):
    return -sum(
        spec.ocv(name, getattr(spec, "initial_"+name+"_fraction"), derivative=True)
        / getattr(spec, name).capacity_mol for name in ("negative", "positive")
    ) / F


def fields_of_response(spec, result, point):
    index = {v.name: i for i, v in enumerate(result.response.variables)}
    algebraic = {key: point.response[index[key], 0] for key in spec.algebraic_names}
    # The physical field function accepts real states. Apply its real linear
    # map to both phasor components, preserving the complete complex response.
    parts = [linear_electrical(
        spec.geometry, spec.reaction_keys, spec.resistance_ohm,
        {key: component(value) for key, value in algebraic.items()},
    ) for component in (np.real, np.imag)]
    return (*[a+1j*b for a, b in zip(parts[0][:4], parts[1][:4], strict=True)],
            [a+1j*b for a, b in zip(parts[0][4], parts[1][4], strict=True)])


@pytest.mark.parametrize("kind", ["ordinary", "low-liquid", "low-solid"])
def test_actual_spectrum_matches_independent_continuous_network_and_inventory_pole(kind):
    geometry = PorousElectrodeGeometrySpec(
        cells=(64, 32, 64),
        conductivity_S_m=.15 if kind == "low-liquid" else 1.,
        solid_conductivity_S_m=(.2, .1) if kind == "low-solid" else (100., 100.),
    )
    spec = model(geometry=geometry)
    resistance, continuous = network(spec)
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    expected = resistance + pole(spec)/(2j*np.pi*np.array(FREQUENCIES))
    # Finite-volume centre BV and continuous BVP differ at finite thickness mesh.
    np.testing.assert_allclose(result.impedance_ohm, expected, rtol=2e-4, atol=2e-6)
    assert resistance > 0
    q, _, phi_e, phi_s, _ = fields_of_response(spec, result, result.response.points[0])
    integrated, liquid, solid = [], [], []
    for region, (matrix, initial, length, *_) in zip((0, 2), continuous, strict=True):
        faces = np.linspace(0., length, geometry.cells[region]+1)
        centres = (faces[:-1]+faces[1:])/2
        face_values = np.array([expm(x*matrix) @ initial for x in faces])
        centre_values = np.array([expm(x*matrix) @ initial for x in centres])
        integrated.extend(geometry.area_m2*np.diff(face_values[:, 1]))
        liquid.extend(centre_values[:, 2])
        solid.extend(centre_values[:, 0]+centre_values[:, 2])
        if region == 0:
            sep_centres = (np.arange(geometry.cells[1])+.5)*geometry.lengths_m[1]/geometry.cells[1]
            kappa_sep = geometry.conductivity_S_m*geometry.porosities[1]**geometry.bruggeman[1]
            liquid.extend(face_values[-1, 2]-sep_centres/(geometry.area_m2*kappa_sep))
    s = 2j*np.pi*FREQUENCIES[0]
    dun = -spec.negative_ocv_slope_V/(s*F*spec.negative.capacity_mol)
    dup = spec.positive_ocv_slope_V/(s*F*spec.positive.capacity_mol)
    full_solid = np.array(solid, dtype=complex)
    full_solid[geometry.cells[0]:] += dup-dun
    np.testing.assert_allclose(q, integrated, rtol=1e-3, atol=1e-6)
    np.testing.assert_allclose(phi_e, np.array(liquid)-dun, rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(phi_s, full_solid, rtol=2e-4, atol=2e-6)


def test_thickness_currents_are_frequency_independent_and_obey_every_charge_boundary():
    spec = model(geometry=PorousElectrodeGeometrySpec(
        cells=(7, 4, 9), conductivity_S_m=.15, solid_conductivity_S_m=(.3, .2),
    ))
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    nn, ns, np_ = spec.geometry.cells
    area = spec.geometry.area_m2
    first = fields_of_response(spec, result, result.response.points[0])
    for point in result.response.points:
        q, ionic, _, _, solid = fields_of_response(spec, result, point)
        np.testing.assert_allclose(q, first[0], rtol=3e-12, atol=3e-13)
        np.testing.assert_allclose(ionic, first[1], rtol=3e-12, atol=3e-11)
        np.testing.assert_allclose(q.imag, 0., atol=3e-12)
        assert sum(q[:nn]) == pytest.approx(1., abs=2e-12)
        assert sum(q[nn:]) == pytest.approx(-1., abs=2e-12)
        np.testing.assert_allclose(
            area*np.diff(ionic), np.r_[q[:nn], np.zeros(ns), q[nn:]], atol=2e-13,
        )
        assert ionic[0] == 0
        assert ionic[-1] == pytest.approx(0., abs=2e-12)
        np.testing.assert_allclose(ionic[nn:nn+ns+1]*area, 1., atol=2e-12)
        np.testing.assert_allclose(solid[0]+ionic[:nn+1], 1/area, atol=2e-12)
        np.testing.assert_allclose(solid[1]+ionic[-np_-1:], 1/area, atol=2e-12)
    assert np.ptp(first[0][:nn].real) > .005


def test_voltage_potentials_and_local_reactions_match_actual_residual_perturbations():
    spec = model(cells=(4, 3, 5))
    state = spec.initial_state()
    result = battery_impedance(spec, (.01,))
    assert result.success
    index = {v.name: i for i, v in enumerate(result.response.variables)}
    raw = result.response.points[0].response[:, 0]
    projected = fields_of_response(spec, result, result.response.points[0])
    for component in (np.real, np.imag):
        step = 1e-6
        samples = []
        for sign in (-1., 1.):
            view = CoupledStateView(
                {k: state.amounts_mol[k]+sign*step*component(raw[index[k]])
                 for k in spec.inventory_keys}, state.temperature_K, state.volume_L, 0.,
                coordinates={k: state.coordinates[k]+sign*step*component(raw[index[k]])
                             for k in spec.coordinate_keys},
            )
            samples.append(newman_tobias_fields(
                spec, view, current_A=sign*step*component(raw[index["battery:current_A"]]),
            ))
        for field, expected in (
            ("reaction_current_A", projected[0]),
            ("electrolyte_face_current_A_m2", projected[1]),
            ("electrolyte_potential_V", projected[2]),
            ("solid_potential_V", projected[3]),
        ):
            observed = (samples[1][field]-samples[0][field])/(2*step)
            np.testing.assert_allclose(observed, component(expected), rtol=2e-6, atol=2e-7)


def test_original_two_stocks_and_signed_charge_work_obey_faradaic_response():
    spec = model()
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    index = {v.name: i for i, v in enumerate(result.response.variables)}
    assert set(index) == set(spec.inventory_keys+spec.coordinate_keys+spec.algebraic_names)
    voltage = spec.ocv("positive", .3)-spec.ocv("negative", .7)
    for point in result.response.points:
        raw, s = point.response[:, 0], 2j*np.pi*point.frequency_Hz
        assert s*F*raw[index["negative:Li"]] == pytest.approx(-1., abs=2e-12)
        assert s*F*raw[index["positive:Li"]] == pytest.approx(1., abs=2e-12)
        assert raw[index["negative:Li"]]+raw[index["positive:Li"]] == pytest.approx(0., abs=1e-13)
        assert s*raw[index["discharged_charge_C"]] == pytest.approx(1., abs=2e-12)
        assert s*raw[index["delivered_work_J"]] == pytest.approx(voltage, abs=2e-12)
    assert result.response.diagnostics["derivative_modes"]["state"] == "analytic"


def test_flat_ocv_removes_only_voltage_inventory_pole_not_actual_li_response():
    spec = model(negative_ocv_slope_V=0., positive_ocv_slope_V=0.)
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    z = np.array(result.impedance_ohm)
    np.testing.assert_allclose(z, z[0].real, rtol=3e-12, atol=3e-13)
    assert z[0].real > 0
    index = {v.name: i for i, v in enumerate(result.response.variables)}
    assert abs(result.response.points[0].response[index["negative:Li"], 0]) > .001


def test_high_conductivity_approaches_total_bv_resistances_without_diffusion_arc():
    geometry = PorousElectrodeGeometrySpec(
        cells=(6, 3, 8), conductivity_S_m=1e6, solid_conductivity_S_m=(1e6, 2e6),
    )
    spec = model(geometry=geometry)
    result = battery_impedance(spec, FREQUENCIES)
    assert result.success
    rct = 0.
    for name in ("negative", "positive"):
        electrode = getattr(spec, name)
        theta = getattr(spec, "initial_"+name+"_fraction")
        j0 = electrode.exchange_current_scale_A_m2*np.sqrt(theta*(1-theta))
        rct += R*spec.temperature_K/(F*electrode.area_m2*j0)
    z = np.array(result.impedance_ohm)
    np.testing.assert_allclose(z.real, rct, rtol=1e-6, atol=2e-8)
    np.testing.assert_allclose(2j*np.pi*np.array(FREQUENCIES)*(z-z.real), pole(spec), atol=2e-14)


def test_series_resistance_changes_port_only_and_preserves_reaction_distribution():
    spec = model()
    base = battery_impedance(spec, FREQUENCIES)
    changed = battery_impedance(replace(spec, resistance_ohm=.17), FREQUENCIES)
    assert base.success and changed.success
    np.testing.assert_allclose(np.array(changed.impedance_ohm)-base.impedance_ohm,
                               .17, rtol=2e-12, atol=2e-13)
    index = {v.name: i for i, v in enumerate(base.response.variables)}
    retained = [index[k] for k in (*spec.inventory_keys, *spec.reaction_keys)]
    for first, second in zip(base.response.points, changed.response.points, strict=True):
        np.testing.assert_allclose(first.response[retained], second.response[retained],
                                   rtol=2e-12, atol=2e-13)


def test_full_original_mean_inventory_and_spatial_closure_jacobian():
    spec = model(cells=(3, 2, 4), resistance_ohm=.1)
    system, initial = battery_impedance_system(spec, spec.initial_state())
    evaluator = ResidualEvaluator(system)
    rate = np.zeros(len(initial))
    analytic = evaluator.derivative("state", 0., initial, rate).toarray()
    numerical = finite_jacobian(lambda y: evaluator.value(0., y, rate), initial, system.scales)
    np.testing.assert_allclose(
        analytic*system.scales/system.residual_scales[:, None],
        numerical*system.scales/system.residual_scales[:, None], rtol=3e-6, atol=3e-8,
    )
    assert "battery:control" in [e.name for e in system.equations]
    assert not initial.flags.writeable


@pytest.mark.parametrize("electrode", ["negative", "positive"])
@pytest.mark.parametrize("upper", [False, True])
def test_exact_mean_mol_faces_reject_but_one_ulp_inside_is_valid(electrode, upper):
    spec = model(initial_negative_fraction=.5, initial_positive_fraction=.5)
    spec = replace(spec, **{electrode: replace(getattr(spec, electrode), active_volume_m3=.5e-6)})
    initial = spec.initial_state()
    capacity = getattr(spec, electrode).capacity_mol
    face = capacity*(1-spec.surface_margin if upper else spec.surface_margin)

    def at(amount):
        amounts = dict(initial.amounts_mol)
        delta = amount-amounts[electrode+":Li"]
        amounts[electrode+":Li"] = amount
        other = "positive" if electrode == "negative" else "negative"
        amounts[other+":Li"] -= delta
        q = F*delta*(-1 if electrode == "negative" else 1)
        return replace(initial, amounts_mol=amounts, coordinates={
            "discharged_charge_C": q, "delivered_work_J": 0.,
        })

    boundary = at(face)
    validate_newman_tobias_state(spec, boundary)  # legal for rest/inward time recovery
    with pytest.raises(ResidualDomainError, match="stopping boundary: "+electrode+":Li"):
        battery_impedance(spec, FREQUENCIES, state=boundary)
    interior = at(np.nextafter(face, 0. if upper else capacity))
    assert battery_impedance(spec, FREQUENCIES, state=interior).success


@pytest.mark.parametrize("upper", [False, True])
def test_exact_terminal_voltage_face_is_not_a_two_sided_analysis_point(upper):
    spec = model()
    voltage = newman_tobias_fields(spec, spec.initial_state(), current_A=0.)["voltage_V"]
    spec = replace(spec, **{"maximum_voltage_V" if upper else "minimum_voltage_V": voltage})
    with pytest.raises(ResidualDomainError, match="stopping boundary: battery:"):
        battery_impedance(spec, FREQUENCIES)


def test_noninitial_mean_inventory_readout_discards_stale_control_without_mutation():
    spec = model()
    payload = json.loads(json.dumps(spec.to_dict()))
    spec = BatteryNewmanTobiasSpec.from_dict(payload)
    initial = spec.initial_state()
    q = 10.
    state = replace(initial, amounts_mol={
        "negative:Li": initial.amounts_mol["negative:Li"]-q/F,
        "positive:Li": initial.amounts_mol["positive:Li"]+q/F,
    }, coordinates={"discharged_charge_C": q, "delivered_work_J": 38.}, time_s=17.)
    stale = CoupledStateView(
        state.amounts_mol, state.temperature_K, state.volume_L, state.time_s,
        coordinates=state.coordinates, algebraic_values=dict.fromkeys(spec.algebraic_names, .3),
    )
    before = dict(stale.algebraic_values)
    clean = battery_impedance(spec, FREQUENCIES, state=state)
    result = battery_impedance(spec, FREQUENCIES, state=stale)
    assert clean.success and result.success
    np.testing.assert_array_equal(clean.impedance_ohm, result.impedance_ohm)
    assert dict(stale.algebraic_values) == before
    assert stale.time_s == 17. and stale.coordinates == state.coordinates
    assert stale.amounts_mol == state.amounts_mol
    assert spec.to_dict() == payload
    assert result.response.diagnostics["physical_time_advanced_s"] == 0.
