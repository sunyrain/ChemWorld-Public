"""Functional physics of spatial 2+1 materials, separate from source qualification."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.battery_composite import ActiveMaterialPhaseSpec
from chemworld.physchem.battery_dfn import BatteryDFNSpec, dfn_fields
from chemworld.physchem.battery_dfn_composite import (
    BatteryCompositeDFNSpec,
    _fields,
    advance_composite_dfn,
    composite_dfn_algebraic,
    composite_dfn_control_mode,
    composite_dfn_fields,
    composite_dfn_readout,
    composite_dfn_system,
)
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import F, ParticleSpec
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def model(*, same=False, cells=(3, 2, 3), radial=5, **kwargs):
    source = InputSource(
        DataSourceProvenance("test:composite-dfn", 0, "Authored physics fixture", "1", "MIT"),
        "synthetic",
    )

    def phase(identity, volume, ocv, fraction):
        return ActiveMaterialPhaseSpec(
            identity,
            ParticleSpec(cells=radial, active_volume_m3=volume, diffusivity_m2_s=1e-12),
            ElectrodeOCV(identity, source, ocv, supported_models=("DFN", "SPM")),
            fraction,
            source,
        )

    return BatteryCompositeDFNSpec(
        (
            phase("alpha", 0.6e-6, (0.2, -0.1), 0.7),
            phase("beta", 0.4e-6, (0.2, -0.1) if same else (0.25, -0.15), 0.7 if same else 0.55),
        ),
        phase("positive", 1e-6, (4.2, -0.8), 0.3),
        electrolyte=BatteryElectrolyteSpec(cells=cells),
        **kwargs,
    )


def control(value, unit="A"):
    return ScalarControlProgram(((0.0, value),), unit)


@pytest.mark.parametrize("current", [-0.25, 0.0, 0.25])
def test_local_phase_currents_share_fields_and_close_charge_exactly(current):
    spec = model()
    f = composite_dfn_fields(spec, spec.initial_state(), current_A=current)
    nn, ns, _ = spec.electrolyte.cells
    q = f["reaction_current_A"]
    np.testing.assert_allclose(
        f["cell_reaction_current_A"][:nn], q[:nn] + q[nn : 2 * nn], atol=1e-15
    )
    assert sum(q[: 2 * nn]) == pytest.approx(current, abs=1e-14)
    assert sum(q[2 * nn :]) == pytest.approx(-current, abs=1e-14)
    np.testing.assert_allclose(
        np.diff(f["electrolyte_face_current_A_m2"]) * spec.electrolyte.area_m2,
        np.r_[q[:nn] + q[nn : 2 * nn], np.zeros(ns), q[2 * nn :]],
        atol=1e-14,
    )
    np.testing.assert_allclose(f["bv_residual_V"], 0, atol=1e-11)
    assert abs(f["negative_solid_face_current_A_m2"][-1]) < 1e-12
    assert abs(f["positive_solid_face_current_A_m2"][0]) < 1e-12
    if current == 0:
        assert np.max(np.abs(q[: 2 * nn])) > 0.01
        assert sum(q[:nn]) * sum(q[nn : 2 * nn]) < 0


@pytest.mark.parametrize("current", [-0.3, 0.0, 0.3])
def test_identical_phases_merge_to_the_original_dfn(current):
    spec = model(same=True)
    merged = BatteryDFNSpec(
        negative=replace(spec.negative_phases[0].particle, active_volume_m3=1e-6),
        positive=spec.positive_phase.particle,
        electrolyte=spec.electrolyte,
    )
    actual = composite_dfn_fields(spec, spec.initial_state(), current_A=current)
    expected = dfn_fields(merged, merged.initial_state(), current_A=current)
    for name in (
        "voltage_V",
        "electrolyte_potential_V",
        "electrolyte_face_current_A_m2",
        "solid_potential_V",
        "negative_solid_face_current_A_m2",
        "positive_solid_face_current_A_m2",
    ):
        np.testing.assert_allclose(actual[name], expected[name], atol=2e-11, rtol=1e-10)
    np.testing.assert_allclose(
        actual["cell_reaction_current_A"], expected["reaction_current_A"], atol=2e-12
    )
    nn = spec.electrolyte.cells[0]
    np.testing.assert_allclose(
        actual["reaction_current_A"][:nn], 0.6 * expected["reaction_current_A"][:nn], atol=2e-12
    )


@pytest.mark.parametrize(
    "target,expected_mode",
    [(3.8, "constant_voltage"), (2.6, "positive_current_limit"), (4.4, "negative_current_limit")],
)
def test_finite_cv_source_is_solved_with_all_phase_currents(target, expected_mode):
    spec = model(maximum_current_A=0.4)
    f = composite_dfn_fields(spec, spec.initial_state(), program=control(target, "V"))
    assert composite_dfn_control_mode(spec, spec.initial_state(), control(target, "V"))[0] == (
        expected_mode
    )
    if expected_mode == "constant_voltage":
        assert f["voltage_V"] == pytest.approx(target, abs=1e-11)
    else:
        assert abs(f["current_A"]) == pytest.approx(0.4, abs=1e-11)
    np.testing.assert_allclose(f["bv_residual_V"], 0.0, atol=1e-11)


@pytest.mark.parametrize("current", [-0.2, 0.0, 0.2])
def test_actual_advance_couples_each_phase_salt_and_single_charge_work(current):
    spec = model()
    initial = spec.initial_state()
    result = advance_composite_dfn(spec, initial, control(current), 1.0)
    assert result.outcome == "success"
    final = result.final_state
    assert final.coordinates["discharged_charge_C"] == pytest.approx(current, abs=1e-10)
    for name, residual in result.conservation_residuals.items():
        assert abs(residual) < (1e-8 if name.endswith("_C") else 1e-12)
    reading = composite_dfn_readout(spec, final, current)
    assert len(reading.phase_readouts) == 3
    if current == 0:
        q = final.coordinates
        assert q["discharged_charge_C"] == pytest.approx(0, abs=1e-12)
        assert q["delivered_work_J"] == pytest.approx(0, abs=1e-12)
        first = spec.particle_bins[0][3]
        assert abs(sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in first)) > 1e-8
    else:
        assert final.coordinates["delivered_work_J"] * current > 0
        c = spec.electrolyte.concentrations(final)
        assert (c[0] - c[-1]) * current > 0


@pytest.mark.parametrize("value,unit", [(0.2, "A"), (0.0, "A"), (3.8, "V"), (2.6, "V"), (4.4, "V")])
def test_full_sparse_rhs_jacobian_includes_phase_salt_and_current_chain(value, unit):
    spec = model(maximum_current_A=0.4)
    system = composite_dfn_system(spec, control(value, unit))
    initial = spec.initial_state()
    keys, anames = spec.inventory_keys, spec.algebraic_names
    names = (
        *keys,
        "temperature",
        "boundary_heat",
        "chemical_energy",
        "heat_loss",
        "work",
        "discharged_charge_C",
        "delivered_work_J",
        *anames,
    )
    count = len(names) - len(anames)
    guess = system.algebraic.initial_values(initial)
    values = np.r_[
        [initial.amounts_mol[k] for k in keys],
        initial.temperature_K,
        np.zeros(6),
        [guess[k] for k in anames],
    ]
    scales = np.r_[
        [system.inventory_scales_mol[k] for k in keys],
        300,
        np.ones(6),
        [v.scale for v in system.algebraic.variables],
    ]

    def view(vector):
        return CoupledStateView(
            dict(zip(keys, vector[: len(keys)], strict=True)),
            vector[len(keys)],
            initial.volume_L,
            coordinates={
                "discharged_charge_C": vector[count - 2],
                "delivered_work_J": vector[count - 1],
            },
            algebraic_values=dict(zip(anames, vector[count:], strict=True)),
        )

    def physical_rhs(vector):
        state = view(vector)
        rates = assemble_coupled_rates(
            state,
            processes=system.processes,
            thermodynamics=system.thermodynamics,
            heat_flux=system.heat_flux,
            isothermal=True,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in keys],
            np.zeros(5),
            [rates.coordinate_rates[k] for k in spec.coordinate_keys],
            -np.asarray(system.algebraic.residual(state)),
        ]

    actual = system.residual_rhs_jacobian(view(values), names).toarray() * scales
    expected = finite_jacobian(physical_rhs, values, scales, relative_step=1e-6) * scales
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-6)
    permutation = np.random.default_rng(3).permutation(len(names))
    reordered = tuple(names[i] for i in permutation)
    permuted = system.residual_rhs_jacobian(view(values), reordered).toarray()
    np.testing.assert_array_equal(
        permuted,
        system.residual_rhs_jacobian(view(values), names).toarray()[
            np.ix_(permutation, permutation)
        ],
    )


def test_phase_order_changes_only_labels_and_row_order():
    spec = model()
    swapped = replace(spec, negative_phases=spec.negative_phases[::-1])
    first = composite_dfn_fields(spec, spec.initial_state(), current_A=0.2)
    second = composite_dfn_fields(swapped, swapped.initial_state(), current_A=0.2)
    assert first["voltage_V"] == pytest.approx(second["voltage_V"], abs=1e-12)
    nn = spec.electrolyte.cells[0]
    for field in ("reaction_current_A", "surface_fraction", "average_fraction"):
        np.testing.assert_allclose(
            first[field],
            np.r_[second[field][nn : 2 * nn], second[field][:nn], second[field][2 * nn :]],
            atol=1e-11,
        )
    np.testing.assert_allclose(
        first["electrolyte_potential_V"], second["electrolyte_potential_V"], atol=1e-12
    )


def test_resolved_salt_changes_both_local_phase_kinetics_without_changing_inventory():
    spec = model()
    state = spec.initial_state()
    amounts = dict(state.amounts_mol)
    e = spec.electrolyte
    delta = 0.1 * amounts[e.keys[0]]
    amounts[e.keys[0]] -= delta
    amounts[e.keys[-1]] += delta
    changed = replace(state, amounts_mol=amounts)
    f = composite_dfn_fields(spec, changed, current_A=0.2)
    baseline = composite_dfn_fields(spec, state, current_A=0.2)
    assert np.max(np.abs(f["reaction_current_A"] - baseline["reaction_current_A"])) > 1e-4
    assert dict(changed.amounts_mol) == amounts


def test_phase_source_rates_use_local_q_and_cell_sum_for_salt():
    spec = model()
    program = control(0.2)
    system = composite_dfn_system(spec, program)
    state = spec.initial_state()
    values = composite_dfn_algebraic(spec, program).initial_values(state)
    # Nonconverged algebraic trial is intentional: exact conservation is structural.
    values[spec.reaction_keys[0]] = -0.1
    trial = CoupledStateView(
        state.amounts_mol,
        state.temperature_K,
        state.volume_L,
        coordinates=state.coordinates,
        algebraic_values=values,
    )
    f = _fields(spec, trial, values)
    contribution = system.processes[-1].evaluate(trial)
    for q, (_, _, _, keys, _) in zip(f["reaction_current_A"], spec.particle_bins, strict=True):
        assert contribution.amount_rates_mol_s[keys[-1]] == -q / F
    nn, ns, _ = spec.electrolyte.cells
    expected = (
        np.r_[f["cell_reaction_current_A"][:nn], np.zeros(ns), f["cell_reaction_current_A"][nn:]]
        * (1 - spec.electrolyte.cation_transference)
        / F
    )
    np.testing.assert_allclose(
        [contribution.amount_rates_mol_s[k] for k in spec.electrolyte.keys], expected, atol=1e-20
    )


def test_surface_event_identifies_the_actual_material_and_partial_duration():
    spec = model(surface_margin=0.2, maximum_current_A=0.05)
    positive = replace(
        spec.positive_phase,
        initial_fraction=0.78,
        particle=replace(
            spec.positive_phase.particle,
            active_volume_m3=1e-8,
            radius_m=1e-6,
            diffusivity_m2_s=1e-10,
        ),
    )
    spec = replace(spec, positive_phase=positive, minimum_voltage_V=1.0)
    result = advance_composite_dfn(spec, spec.initial_state(), control(0.02), 100.0)
    assert result.outcome == "stopped"
    assert 0 < result.final_state.time_s < 100
    assert "positive:phase:positive:x:" in result.stopping_event
    assert result.stopping_event.endswith(":saturated")
    f = composite_dfn_fields(spec, result.final_state, current_A=0.02)
    assert max(f["surface_fraction"][-3:]) == pytest.approx(0.8, abs=1e-7)


@pytest.mark.parametrize("bad", ["source-model", "positive-slope", "oversize", "duplicate"])
def test_material_domain_and_geometry_reject_invalid_declarations(bad):
    spec = model()
    alpha, beta = spec.negative_phases
    if bad == "source-model":
        alpha = replace(alpha, ocv=replace(alpha.ocv, supported_models=("SPM",)))
    elif bad == "positive-slope":
        alpha = replace(alpha, ocv=replace(alpha.ocv, polynomial_V=(0.2, 0.1)))
    elif bad == "oversize":
        alpha = replace(alpha, particle=replace(alpha.particle, active_volume_m3=1e-4))
    else:
        beta = replace(beta, phase_id=alpha.phase_id)
    with pytest.raises(ValueError):
        replace(spec, negative_phases=(alpha, beta))


def test_series_resistance_only_changes_terminal_voltage_for_fixed_current():
    spec = model()
    other = replace(spec, series_resistance_ohm=0.7)
    original = composite_dfn_fields(spec, spec.initial_state(), current_A=0.2)
    shifted = composite_dfn_fields(other, other.initial_state(), current_A=0.2)
    assert shifted["voltage_V"] - original["voltage_V"] == pytest.approx(-0.14, abs=1e-12)
    for key in ("reaction_current_A", "solid_potential_V", "electrolyte_potential_V"):
        np.testing.assert_allclose(original[key], shifted[key], atol=1e-12)


@pytest.mark.parametrize("extra", [0.0, 4 * np.finfo(float).eps, 1e-9])
def test_zero_binder_electrode_allows_only_roundoff_above_geometric_product(extra):
    spec = model()
    e = replace(spec.electrolyte, area_m2=0.01)
    alpha, beta = spec.negative_phases
    alpha = replace(alpha, particle=replace(alpha.particle, active_volume_m3=4e-7))
    beta = replace(beta, particle=replace(beta.particle, active_volume_m3=2e-7 * (1 + extra)))
    positive = replace(
        spec.positive_phase, particle=replace(spec.positive_phase.particle, active_volume_m3=6e-7)
    )
    if extra == 1e-9:
        with pytest.raises(ValueError, match="non-pore"):
            replace(spec, negative_phases=(alpha, beta), positive_phase=positive, electrolyte=e)
    else:
        selected = replace(
            spec, negative_phases=(alpha, beta), positive_phase=positive, electrolyte=e
        )
        assert selected.volume_l == pytest.approx(0.00225)
