"""MSMR conservative nonlinear diffusion, shared kinetics and actual coupled residual."""

import json
from collections.abc import Mapping
from dataclasses import replace
from math import log, log1p

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC, ParticleSpec
from chemworld.physchem.battery_msmr import (
    BatteryMSMRSPMSpec,
    F,
    advance_msmr,
    msmr_control_mode,
    msmr_fields,
    msmr_particle_transport,
    msmr_readout,
    msmr_system,
    validate_msmr_state,
)
from chemworld.physchem.battery_msmr_materials import MSMR_UNITS, MSMRMaterialSpec
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def material(name, *, width=1.0, single=False):
    source = InputSource(
        DataSourceProvenance(
            "msmr-functional", 0, "Authored synthetic MSMR regression inputs", "1", "CC0-1.0"
        ),
        "synthetic",
    )
    base = 0.2 if name == "negative" else 4.0
    numbers = (
        [(1.0, base, width, 3.0, 0.5)]
        if single
        else [(0.6, base - 0.035, 0.8, 4.0, 0.35), (0.4, base + 0.05, 1.6, 2.0, 0.65)]
    )
    ids = tuple(f"reaction-{i}" for i in range(len(numbers)))
    table = ScientificTable(
        "msmr-" + name,
        "parameters",
        tuple(
            QuantityColumn(k, u, "parameter", (-100, 1e6), "Fixed synthetic input")
            for k, u in MSMR_UNITS.items()
        ),
        (source,),
        tuple(
            ScientificRow(
                identity,
                source.provenance.source_id,
                "synthetic",
                name + "-" + identity,
                "parameters",
                {
                    k: InputValue(v, MSMR_UNITS[k], None, "Fixed synthetic input")
                    for k, v in zip(MSMR_UNITS, row, strict=True)
                },
            )
            for identity, row in zip(ids, numbers, strict=True)
        ),
    )
    return MSMRMaterialSpec(table, ids, name)


def model(*, single=False, width=1.0, **changes):
    p = ParticleSpec(cells=6, mesh_exponent=1.0, diffusivity_m2_s=1e-13)
    args = {
        "negative": p,
        "positive": p,
        "negative_material": material("negative", width=width, single=single),
        "positive_material": material("positive", width=width, single=single),
        "maximum_current_A": 0.5,
    }
    args.update(changes)
    return BatteryMSMRSPMSpec(**args)


def program(value=0.2, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


def nonuniform(spec, amplitude=0.03):
    state = spec.initial_state()
    amounts = dict(state.amounts_mol)
    for name in ("negative", "positive"):
        p = getattr(spec, name)
        shape = np.linspace(-amplitude, amplitude, p.cells)
        shape -= np.average(shape, weights=p.mesh.cell_volumes_m3)
        fraction = getattr(spec, "initial_" + name + "_fraction") + shape
        amounts.update(
            zip(
                p.keys(name),
                p.maximum_concentration_mol_m3 * p.mesh.cell_volumes_m3 * fraction,
                strict=True,
            )
        )
    return replace(state, amounts_mol=amounts)


def balances(spec, initial, final):
    charge = final.coordinates["discharged_charge_C"] - initial.coordinates["discharged_charge_C"]
    assert sum(final.amounts_mol.values()) == pytest.approx(
        sum(initial.amounts_mol.values()), abs=2e-12
    )
    for name, sign in (("negative", -1), ("positive", 1)):
        actual = F * sum(
            final.amounts_mol[k] - initial.amounts_mol[k] for k in getattr(spec, name).keys(name)
        )
        assert actual == pytest.approx(sign * charge, abs=2e-8)
    assert final.temperature_K == spec.temperature_K


def test_codec_uses_only_original_li_and_qw_no_potential_inventory():
    spec = model()
    restored = BatteryMSMRSPMSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.initial_state() == spec.initial_state()
    assert len(spec.inventory_keys) == 12
    assert spec.coordinate_keys == ("discharged_charge_C", "delivered_work_J")


@pytest.mark.parametrize("current", [0.2, -0.2, 0.0])
def test_actual_cc_rest_charge_and_work_share_single_inventory(current):
    spec = model()
    initial = spec.initial_state()
    final = advance_msmr(spec, initial, program(current), 3).final_state
    balances(spec, initial, final)
    assert final.time_s == 3
    assert final.coordinates["discharged_charge_C"] == pytest.approx(3 * current, abs=1e-10)
    if current:
        assert final.coordinates["delivered_work_J"] * current > 0
    else:
        np.testing.assert_allclose(
            list(initial.amounts_mol.values()), list(final.amounts_mol.values()), rtol=0, atol=2e-18
        )


@pytest.mark.parametrize("width", [1.0, 2.0])
def test_single_reaction_small_gradient_limit_is_width_times_fick(width):
    spec = model(single=True, width=width)
    state = nonuniform(spec, amplitude=2e-6)
    p = spec.negative
    _, _, actual = msmr_particle_transport(spec, state, "negative")
    c = np.array([state.amounts_mol[k] for k in p.keys("negative")]) / p.mesh.cell_volumes_m3
    expected = (
        -width
        * p.diffusivity_m2_s
        * p.mesh.face_areas_m2[1:-1]
        * np.diff(c)
        / np.diff(p.mesh.centers_m)
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-20)


def test_finite_gradient_nonlinear_transport_is_not_an_old_fick_matrix():
    spec = model()
    state = nonuniform(spec, amplitude=0.1)
    rates, jac, flux = msmr_particle_transport(spec, state, "negative")
    assert sum(rates) == pytest.approx(0, abs=1e-18)
    np.testing.assert_allclose(jac.sum(axis=0), 0, atol=1e-15)
    assert np.all(flux < 0)
    p = spec.negative
    c = np.array([state.amounts_mol[k] for k in p.keys("negative")]) / p.mesh.cell_volumes_m3
    fick = -p.diffusivity_m2_s * p.mesh.face_areas_m2[1:-1] * np.diff(c) / np.diff(p.mesh.centers_m)
    assert not np.allclose(flux, fick, rtol=0.05, atol=0)


@pytest.mark.parametrize("value,unit", [(0.2, "A"), (0.0, "A"), (3.8, "V"), (2.7, "V"), (4.4, "V")])
def test_full_analytic_jacobian_matches_same_shared_rhs(value, unit):
    spec = model(resistance_ohm=0.1)
    initial, control = nonuniform(spec), program(value, unit)
    fields = msmr_fields(spec, initial, program=control)
    names = (*spec.inventory_keys, "temperature", *spec.coordinate_keys, *BATTERY_ALGEBRAIC)
    n = len(spec.inventory_keys)
    y = np.r_[
        list(initial.amounts_mol.values()),
        initial.temperature_K,
        0.0,
        0.0,
        fields["current_A"],
        fields["voltage_V"],
    ]
    scales = np.r_[[1e-3] * n, 300.0, 1000.0, 4000.0, 0.5, 1.0]

    def view(z):
        return CoupledStateView(
            dict(zip(spec.inventory_keys, z[:n], strict=True)),
            z[n],
            spec.volume_l,
            coordinates=dict(zip(spec.coordinate_keys, z[n + 1 : n + 3], strict=True)),
            algebraic_values=dict(zip(BATTERY_ALGEBRAIC, z[n + 3 :], strict=True)),
        )

    system = msmr_system(spec, control)

    def rhs(z):
        v = view(z)
        rates = assemble_coupled_rates(
            v,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            isothermal=True,
            heat_flux=system.heat_flux,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in spec.inventory_keys],
            0.0,
            [rates.coordinate_rates[k] for k in spec.coordinate_keys],
            -np.asarray(system.algebraic.residual(v)),
        ]

    actual = system.residual_rhs_jacobian(view(y), names).toarray()
    lower, upper = np.full(len(y), -np.inf), np.full(len(y), np.inf)
    lower[-2], upper[-2] = -spec.maximum_current_A, spec.maximum_current_A
    expected = finite_jacobian(rhs, y, scales, lower=lower, upper=upper)
    rows = np.r_[[1e-3] * n, 300.0, 1.0, 4.0, 1.0, 1.0]
    np.testing.assert_allclose(
        actual * scales / rows[:, None], expected * scales / rows[:, None], atol=3e-6, rtol=3e-6
    )


@pytest.mark.parametrize(
    "target,mode",
    [(3.84, "constant_voltage"), (2.7, "positive_current_limit"), (4.4, "negative_current_limit")],
)
def test_cv_real_dynamic_state_and_source_limit(target, mode):
    spec = model()
    control = program(target, "V")
    final = advance_msmr(spec, spec.initial_state(), control, 2).final_state
    fields = msmr_fields(spec, final, program=control)
    assert msmr_control_mode(spec, final, control)[0] == mode
    if mode == "constant_voltage":
        assert fields["voltage_V"] == pytest.approx(target, abs=2e-9)
    else:
        assert abs(fields["current_A"]) == pytest.approx(spec.maximum_current_A)
    balances(spec, spec.initial_state(), final)


def test_nonuniform_rest_relaxes_and_json_restart_matches():
    spec = model()
    initial = nonuniform(spec)
    one = advance_msmr(spec, initial, program(0), 2).final_state
    payload = {k: dict(v) if isinstance(v, Mapping) else v for k, v in vars(one).items()}
    restored = type(one)(**json.loads(json.dumps(payload)))
    a = advance_msmr(spec, one, program(0), 2).final_state
    b = advance_msmr(spec, restored, program(0), 2).final_state
    assert a == b
    balances(spec, initial, a)

    def spread(s):
        return np.ptp(msmr_fields(spec, s, current_A=0)["negative_particle_potentials_V"])

    assert spread(a) < spread(initial)


def test_series_resistance_is_explicit_and_passive():
    spec = model()
    initial = spec.initial_state()
    before = msmr_readout(spec, initial, 0.2)
    after = msmr_readout(replace(spec, resistance_ohm=0.3), initial, 0.2)
    assert before.voltage_V - after.voltage_V == pytest.approx(0.06, abs=1e-13)


def test_surface_and_voltage_events_return_actual_stopped_time():
    spec = model(initial_negative_fraction=0.025)
    result = advance_msmr(spec, spec.initial_state(), program(0.5), 100)
    assert result.final_state.time_s < 100
    assert "depleted" in result.stopping_event
    balances(spec, spec.initial_state(), result.final_state)
    base = model()
    v = msmr_readout(base, base.initial_state(), 0.2).voltage_V
    spec = replace(base, minimum_voltage_V=v - 1e-4)
    result = advance_msmr(spec, spec.initial_state(), program(0.2), 100)
    assert result.final_state.time_s < 100
    assert result.stopping_event == "msmr:minimum-voltage"


def test_invalid_accepted_state_never_clips_li_or_resets_charge():
    spec = model()
    state = spec.initial_state()
    values = dict(state.amounts_mol)
    values[spec.inventory_keys[0]] *= 1.1
    with pytest.raises(ValueError, match="Faraday"):
        validate_msmr_state(spec, replace(state, amounts_mol=values))
    with pytest.raises(ValueError, match="isothermal"):
        validate_msmr_state(spec, replace(state, temperature_K=300))
    with pytest.raises(ValueError, match="thermal"):
        msmr_system(spec, program(), thermal_device=object())


@pytest.mark.parametrize("fraction", [1e-14, 1 - 1e-14])
def test_accepted_tail_inventory_uses_true_single_reaction_potential(fraction):
    spec = model(
        single=True,
        surface_margin=1e-16,
        initial_negative_fraction=fraction,
        minimum_voltage_V=-100,
        maximum_voltage_V=100,
    )
    state = spec.initial_state()
    validate_msmr_state(spec, state)
    fields = msmr_fields(spec, state, current_A=0.0)
    actual_mean = fields["negative_average_fraction"]
    expected = 0.2 + 8.31446261815324 * spec.temperature_K / F * (
        log1p(-actual_mean) - log(actual_mean)
    )
    assert fields["negative_potential_V"] == pytest.approx(expected, abs=2e-12)
    # Relative errors of the *small* occupancy or vacancy, never merely abs(x).
    if fraction < 0.5:
        assert fields["negative_surface_fraction"] == pytest.approx(actual_mean, rel=2e-12, abs=0)
    else:
        assert 1 - fields["negative_surface_fraction"] == pytest.approx(
            1 - actual_mean, rel=0.02, abs=0
        )


@pytest.mark.parametrize("fraction", [1e-14, 1 - 1e-14])
def test_two_reaction_inverse_resolves_small_occupancy_and_vacancy(fraction):
    material_ = material("negative")
    potential = material_.potential(fraction, 298.15)
    factor = F / (8.31446261815324 * 298.15)
    # Evaluate the independent complementary logistic without subtraction.
    small = sum(
        weight / (1 + np.exp((1 if fraction < 0.5 else -1) * factor * (potential - u0) / width))
        for weight, u0, width, _, _ in material_.reaction_parameters
    )
    assert small == pytest.approx(fraction if fraction < 0.5 else 1 - fraction, rel=5e-12, abs=0)


def test_legal_low_occupancy_face_flux_and_jacobian_are_not_frozen_by_extension():
    spec = model(
        single=True,
        initial_negative_fraction=1e-14,
        surface_margin=1e-16,
        minimum_voltage_V=-100,
        maximum_voltage_V=100,
    )
    p, initial = spec.negative, spec.initial_state()
    capacity = p.mesh.cell_volumes_m3 * p.maximum_concentration_mol_m3
    shape = np.linspace(0.7, 1.3, p.cells)
    shape /= np.average(shape, weights=capacity)
    amounts = dict(initial.amounts_mol)
    amounts.update(zip(p.keys("negative"), capacity * 1e-14 * shape, strict=True))
    state = replace(initial, amounts_mol=amounts)
    validate_msmr_state(spec, state)
    rates, jac, face = msmr_particle_transport(spec, state, "negative")
    assert np.all(face < 0)
    assert np.any(rates != 0)
    # Independent single-reaction potential and harmonic-M flux.
    x = 1e-14 * shape
    f = F / (8.31446261815324 * spec.temperature_K)
    u = 0.2 + (np.log1p(-x) - np.log(x)) / f
    mobility = p.maximum_concentration_mol_m3 * x * (1 - x) * f * p.diffusivity_m2_s
    harmonic = 2 / (1 / mobility[:-1] + 1 / mobility[1:])
    expected = p.mesh.face_areas_m2[1:-1] * harmonic * np.diff(u) / np.diff(p.mesh.centers_m)
    np.testing.assert_allclose(face, expected, rtol=2e-12, atol=0)
    n = np.array([amounts[k] for k in p.keys("negative")])
    numerical = np.zeros_like(jac)
    for j in range(p.cells):
        step = 1e-5 * n[j]

        def evaluated(sign, j=j, step=step):
            changed = dict(amounts)
            changed[p.keys("negative")[j]] += sign * step
            view = CoupledStateView(
                changed, spec.temperature_K, spec.volume_l, coordinates=state.coordinates
            )
            return msmr_particle_transport(spec, view, "negative")[0]

        numerical[:, j] = (evaluated(1) - evaluated(-1)) / (2 * step)
    np.testing.assert_allclose(jac, numerical, rtol=3e-7, atol=1e-10)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -1e-4, 1.0001])
def test_boundary_extension_is_only_for_unaccepted_solver_trials(fraction):
    spec, state = model(), model().initial_state()
    p = spec.negative
    values = dict(state.amounts_mol)
    values[p.keys("negative")[0]] = (
        fraction * p.mesh.cell_volumes_m3[0] * p.maximum_concentration_mol_m3
    )
    trial = CoupledStateView(
        values, state.temperature_K, state.volume_L, coordinates=state.coordinates
    )
    with pytest.raises(ValueError, match="strictly inside"):
        validate_msmr_state(spec, trial)
    rates, jac, _ = msmr_particle_transport(spec, trial, "negative")
    assert np.all(np.isfinite(rates)) and np.all(np.isfinite(jac))


def test_unrepresentable_inverse_kinetics_is_explicitly_rejected():
    original = material("negative", single=True)
    row = original.parameters.rows[0]
    modified = replace(
        row, values={**row.values, "jref_A_m2": replace(row.values["jref_A_m2"], value=1e-320)}
    )
    authored = replace(original, parameters=replace(original.parameters, rows=(modified,)))
    with pytest.raises(ValueError, match="not representable"):
        authored.kinetic_response(0.2, 0.0, 298.15, 1000.0)
