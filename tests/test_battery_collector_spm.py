import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_collector_spm import (
    BatteryCollectorSPMSpec,
    CollectorSPMReadout,
    advance_collector_spm,
    collector_spm_fields,
    collector_spm_readout,
    collector_spm_system,
    default_collector_spm_spec,
)
from chemworld.physchem.battery_current_collector import CollectorGeometrySpec
from chemworld.physchem.battery_models import F, ParticleSpec
from chemworld.physchem.coupled_system import CoupledStateView, assemble_coupled_rates
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import finite_jacobian


def test_green_mapping_and_tab_layouts_are_distinct_and_conservative():
    left = CollectorGeometrySpec(cells=4, negative_tab="left")
    right = CollectorGeometrySpec(cells=4, negative_tab="right")
    qn = np.array([0.1, 0.2, 0.3, 0.4])
    qp = -qn
    ln, lp = left.potentials(qn, qp, 3.8)
    rn, _rp = right.potentials(qn, qp, 3.8)
    assert np.isfinite(ln[0])
    assert np.isfinite(rn[-1])
    assert not np.allclose(ln, rn)
    assert np.isfinite(left.boundary_voltage_v(lp))


def test_nonuniform_rest_has_internal_collector_currents():
    particle = ParticleSpec(cells=5, active_volume_m3=5e-7)
    spec = BatteryCollectorSPMSpec(
        negative=particle,
        positive=particle,
        collector_cells=3,
        initial_negative_fractions=(0.6, 0.7, 0.8),
        initial_positive_fractions=(0.2, 0.3, 0.4),
    )
    fields = collector_spm_fields(spec, spec.initial_state(), current_A=0.0)
    assert abs(np.sum(fields["negative_reaction_current_A"])) < 1e-10
    assert abs(np.sum(fields["positive_reaction_current_A"])) < 1e-10
    assert np.ptp(fields["negative_reaction_current_A"]) > 1e-4
    assert np.ptp(fields["positive_reaction_current_A"]) > 1e-4


def test_cc_advance_conserves_faradaic_charge_and_keeps_q_out_of_state():
    spec = default_collector_spm_spec(cells=4, collector_cells=3)
    state = spec.initial_state()
    program = ScalarControlProgram(((0.0, 0.05),), "A")
    result = advance_collector_spm(spec, state, program, 0.02)
    assert result.outcome == "success"
    assert np.isclose(result.final_state.coordinates["discharged_charge_C"], 0.001, rtol=1e-8)
    assert all(":reaction_A" not in key for key in result.final_state.coordinates)
    assert all(":reaction_A" not in key for key in result.final_state.amounts_mol)
    assert result.final_state.time_s == pytest.approx(0.02)


def test_source_boundary_voltage_is_diagnostic_separate_from_tab_voltage():
    particle = ParticleSpec(cells=4, active_volume_m3=5e-7)
    spec = BatteryCollectorSPMSpec(negative=particle, positive=particle, collector_cells=4)
    fields = collector_spm_fields(spec, spec.initial_state(), current_A=0.1)
    assert np.isfinite(fields["source_boundary_voltage_V"])
    assert np.isfinite(fields["voltage_V"])
    assert "source_boundary_voltage_V" in fields
    assert np.allclose(
        fields["local_voltage_V"],
        fields["collector_potential_positive_V"] - fields["collector_potential_negative_V"],
    )


def test_codec_and_rejections():
    spec = default_collector_spm_spec()
    restored = BatteryCollectorSPMSpec.from_dict(spec.to_dict())
    assert restored.to_dict() == spec.to_dict()
    with pytest.raises(ValueError, match=r"fixed-T|collector SPM"):
        BatteryCollectorSPMSpec(thermal={"heat_capacity_J_K": 1.0})
    with pytest.raises(ValueError, match="negative_tab"):
        BatteryCollectorSPMSpec(negative_tab="middle")


def test_readout_and_system_identity():
    spec = default_collector_spm_spec(cells=3, collector_cells=2)
    state = spec.initial_state()
    readout = collector_spm_readout(spec, state, 0.0)
    assert isinstance(readout, CollectorSPMReadout)
    system = collector_spm_system(spec, ScalarControlProgram(((0.0, 0.0),), "A"))
    assert system.numerical_form == "residual"
    assert len(system.coordinates) == 2


def test_readout_codec_roundtrips_derived_collector_diagnostics():
    spec = default_collector_spm_spec(cells=3, collector_cells=2)
    readout = collector_spm_readout(spec, spec.initial_state(), 0.0)
    payload = json.loads(json.dumps(readout.to_dict(), allow_nan=False))
    restored = CollectorSPMReadout.from_dict(payload)
    assert restored == readout
    assert "reaction_current_A" not in payload
    with pytest.raises(ValueError, match="family"):
        CollectorSPMReadout.from_dict({**payload, "model_family": "other"})


@pytest.mark.parametrize("tab", ["left", "right"])
def test_physical_collector_passivity_and_signed_face_boundary_currents(tab):
    geometry = CollectorGeometrySpec(
        height_m=0.1,
        width_m=0.1,
        negative_thickness_m=25e-6,
        positive_thickness_m=30e-6,
        negative_conductivity_S_m=1e4,
        positive_conductivity_S_m=2e4,
        cells=4,
        negative_tab=tab,
    )
    q = np.array([0.03, -0.01, 0.07, 0.11])
    current, voltage = float(q.sum()), 3.7
    negative, positive = geometry.potentials(q, -q, voltage)
    fn, fp = geometry.face_currents(q, -q)
    np.testing.assert_allclose(np.diff(fn), -q, atol=1e-16)
    np.testing.assert_allclose(np.diff(fp), q, atol=1e-16)
    assert (fn[0], fn[-1]) == pytest.approx(
        (current, 0.0) if tab == "left" else (0.0, -current), abs=1e-16
    )
    assert (fp[0], fp[-1]) == pytest.approx((0.0, current), abs=1e-16)
    distances = np.full(geometry.cells + 1, geometry.dz_m)
    distances[[0, -1]] *= 0.5
    loss = np.sum(
        distances
        * (
            fn**2 / geometry.negative_sheet_conductance_s
            + fp**2 / geometry.positive_sheet_conductance_s
        )
    )
    assert loss > 0
    assert q @ (positive - negative - voltage) == pytest.approx(loss, rel=2e-14)
    np.testing.assert_allclose(positive, voltage + geometry.positive_green @ q, atol=1e-14)
    assert geometry.boundary_voltage_v(positive) - voltage == pytest.approx(
        geometry.dz_m * q[-1] / (2 * geometry.positive_sheet_conductance_s), abs=1e-14
    )


@pytest.mark.parametrize("tab", ["left", "right"])
def test_every_local_particle_pair_transfers_equal_opposite_current(tab):
    particle = ParticleSpec(cells=4, active_volume_m3=5e-7, diffusivity_m2_s=1e-12)
    spec = BatteryCollectorSPMSpec(
        negative=particle,
        positive=particle,
        collector_cells=4,
        negative_tab=tab,
        height_m=0.1,
        width_m=0.1,
        negative_collector_conductivity_S_m=1e5,
        positive_collector_conductivity_S_m=2e5,
        initial_negative_fractions=(0.60, 0.63, 0.66, 0.69),
        initial_positive_fractions=(0.30, 0.31, 0.32, 0.33),
    )
    for current in (0.0, 0.1, -0.1):
        f = collector_spm_fields(spec, spec.initial_state(), current_A=current)
        np.testing.assert_array_equal(
            f["negative_reaction_current_A"], -f["positive_reaction_current_A"]
        )
        assert sum(f["negative_reaction_current_A"]) == pytest.approx(current, abs=1e-15)
    assert len(spec.reaction_keys) == spec.collector_cells - 1
    assert len(spec.algebraic_names) == spec.collector_cells + 1


@pytest.mark.parametrize("electrode,delta", [("negative", 0.07), ("positive", 0.09)])
def test_open_circuit_observes_each_electrodes_own_stoichiometry(electrode, delta):
    spec = default_collector_spm_spec(cells=3, collector_cells=3)
    base = collector_spm_readout(spec, spec.initial_state(), 0.0)
    initial = getattr(spec, f"initial_{electrode}_fractions")[0]
    altered = replace(spec, **{f"initial_{electrode}_fractions": initial + delta})
    after = collector_spm_readout(altered, altered.initial_state(), 0.0)
    slope = getattr(spec, f"{electrode}_ocv_slope_V")
    sign = 1 if electrode == "positive" else -1
    assert after.voltage_V - base.voltage_V == pytest.approx(sign * slope * delta, abs=1e-12)
    assert base.voltage_V == pytest.approx(
        spec.ocv("positive", spec.initial_positive_fractions[0])
        - spec.ocv("negative", spec.initial_negative_fractions[0]),
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "electrode,lower,current",
    [
        ("negative", True, 0.05),
        ("negative", False, -0.05),
        ("positive", True, -0.05),
        ("positive", False, 0.05),
    ],
)
def test_local_surface_cutoff_stops_before_negative_inventory_and_allows_reversal(
    electrode, lower, current
):
    particle = ParticleSpec(
        cells=4,
        active_volume_m3=1e-8,
        diffusivity_m2_s=1e-10,
        exchange_current_scale_A_m2=20.0,
    )
    boundary_profile = (0.0119, 0.012, 0.0121)
    if not lower:
        boundary_profile = tuple(1 - value for value in boundary_profile)
    spec = BatteryCollectorSPMSpec(
        negative=particle,
        positive=particle,
        collector_cells=3,
        initial_negative_fractions=boundary_profile if electrode == "negative" else 0.6,
        initial_positive_fractions=boundary_profile if electrode == "positive" else 0.4,
        maximum_current_A=0.1,
        minimum_voltage_V=0.0,
        maximum_voltage_V=5.0,
    )
    control = ScalarControlProgram(((0.0, current),), "A")
    system = collector_spm_system(spec, control)
    names = {event.event_id for event in system.all_events}
    assert "battery:collector:negative:z:0:minimum-surface" in names
    result = advance_collector_spm(spec, spec.initial_state(), control, 5.0)
    assert result.outcome == "stopped"
    assert result.stopping_event.startswith(f"battery:collector:{electrode}:z:")
    assert result.stopping_event.endswith(":minimum-surface" if lower else ":maximum-surface")
    assert 0 < result.final_state.time_s < 5
    f = collector_spm_fields(spec, result.final_state, current_A=current)
    values = f["surface_fraction"][:3] if electrode == "negative" else f["surface_fraction"][3:]
    extreme = min(values) if lower else max(values)
    assert extreme == pytest.approx(
        spec.surface_margin if lower else 1 - spec.surface_margin, abs=2e-8
    )
    assert min(result.final_state.amounts_mol.values()) > 0
    reverse = advance_collector_spm(
        spec, result.final_state, ScalarControlProgram(((0.0, -current / 2),), "A"), 0.1
    )
    assert reverse.outcome == "success"
    assert reverse.final_state.time_s == pytest.approx(result.final_state.time_s + 0.1)


@pytest.mark.parametrize("value,unit", [(0.08, "A"), (0.0, "A"), (3.72, "V"), (3.0, "V")])
def test_complete_collector_rhs_jacobian_matches_independent_finite_differences(value, unit):
    particle = ParticleSpec(cells=3, active_volume_m3=5e-7, diffusivity_m2_s=1e-12)
    spec = BatteryCollectorSPMSpec(
        negative=particle,
        positive=replace(particle, cells=4),
        collector_cells=3,
        initial_negative_fractions=(0.61, 0.63, 0.65),
        initial_positive_fractions=(0.33, 0.34, 0.35),
        maximum_current_A=0.2,
        series_resistance_ohm=0.1,
    )
    state = spec.initial_state()
    program = ScalarControlProgram(((0.0, value),), unit)
    system = collector_spm_system(spec, program)
    assert system.residual_rhs_jacobian is not None
    fields = collector_spm_fields(spec, state, program=program)
    keys = spec.inventory_keys
    count = len(keys)
    names = (*keys, "temperature", *spec.coordinate_keys, *spec.algebraic_names)
    values = np.r_[
        [state.amounts_mol[k] for k in keys],
        state.temperature_K,
        0.0,
        0.0,
        [fields["algebraic_values"][k] for k in spec.algebraic_names],
    ]
    scales = np.r_[
        [p.capacity_mol / p.cells for _, _, p, bins, _ in spec.particle_bins for _ in bins],
        300.0,
        100.0,
        400.0,
        [spec.maximum_current_A] * len(spec.reaction_keys),
        0.2,
        4.0,
    ]

    def view(y):
        return CoupledStateView(
            dict(zip(keys, y[:count], strict=True)),
            y[count],
            spec.volume_l,
            coordinates=dict(zip(spec.coordinate_keys, y[count + 1 : count + 3], strict=True)),
            algebraic_values=dict(zip(spec.algebraic_names, y[count + 3 :], strict=True)),
        )

    def rhs(y):
        candidate = view(y)
        rates = assemble_coupled_rates(
            candidate,
            thermodynamics=system.thermodynamics,
            processes=system.processes,
            heat_flux=system.heat_flux,
            isothermal=True,
        )
        return np.r_[
            [rates.amount_rates_mol_s[k] for k in keys],
            0.0,
            [rates.coordinate_rates[k] for k in spec.coordinate_keys],
            -system.algebraic.residual(candidate),
        ]

    analytic = system.residual_rhs_jacobian(view(values), names).toarray() * scales
    numeric = finite_jacobian(rhs, values, scales, relative_step=2e-7) * scales
    np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=3e-8)
    permutation = np.arange(len(names))[::-1]
    reordered = (
        system.residual_rhs_jacobian(view(values), tuple(names[i] for i in permutation)).toarray()
        * scales[permutation]
    )
    np.testing.assert_array_equal(reordered, analytic[np.ix_(permutation, permutation)])
    # Local Li-pair conservation must hold rowwise, not only after whole-cell summation.
    for z in range(spec.collector_cells):
        rows = [i for i, key in enumerate(keys) if f":z:{z}:" in key]
        np.testing.assert_allclose(analytic[rows].sum(axis=0), 0.0, atol=2e-17)
    assert analytic[names.index("discharged_charge_C"), names.index("battery:current_A")] == 0.2
    for electrode, sign in (("negative", -1), ("positive", 1)):
        rows = [i for i, key in enumerate(keys) if key.startswith(electrode + ":")]
        assert analytic[rows, names.index("battery:current_A")].sum() == pytest.approx(
            sign * 0.2 / F, abs=1e-17
        )


@pytest.mark.parametrize(
    "target,mode",
    [
        (3.8, "constant_voltage"),
        (3.0, "positive_current_limit"),
        (4.4, "negative_current_limit"),
    ],
)
def test_finite_cv_and_actual_short_step_use_the_same_tab_port(target, mode):
    from chemworld.physchem.battery_collector_spm import collector_spm_control_mode

    p = ParticleSpec(
        cells=4, active_volume_m3=5e-7, diffusivity_m2_s=1e-12, exchange_current_scale_A_m2=4.0
    )
    spec = BatteryCollectorSPMSpec(
        negative=p, positive=p, collector_cells=3, maximum_current_A=0.5, series_resistance_ohm=0.1
    )
    program = ScalarControlProgram(((0.0, target),), "V")
    state = spec.initial_state()
    original = dict(state.amounts_mol)
    f = collector_spm_fields(spec, state, program=program)
    assert collector_spm_control_mode(spec, state, program)[0] == mode
    if mode == "constant_voltage":
        assert f["voltage_V"] == pytest.approx(target, abs=1e-11)
        assert abs(f["current_A"]) < spec.maximum_current_A
    else:
        expected = spec.maximum_current_A * (1 if mode.startswith("positive") else -1)
        assert f["current_A"] == pytest.approx(expected, abs=1e-11)
    result = advance_collector_spm(spec, state, program, 0.1)
    assert result.outcome == "success"
    assert result.final_state.time_s == 0.1
    assert state.amounts_mol == original
    assert state.algebraic_values == result.final_state.algebraic_values == {}
    final = collector_spm_fields(spec, result.final_state, program=program)
    assert max(abs(final["bv_residual_V"])) < 2e-10
    for z in range(spec.collector_cells):
        keys = (*spec.particle_bins[z][3], *spec.particle_bins[3 + z][3])
        assert sum(result.final_state.amounts_mol[k] for k in keys) == pytest.approx(
            sum(state.amounts_mol[k] for k in keys), abs=2e-14
        )


def test_initial_physical_voltage_cutoff_is_not_hidden_by_collector_loss_cancellation():
    p = ParticleSpec(cells=4, active_volume_m3=5e-7, diffusivity_m2_s=1e-12)
    spec = BatteryCollectorSPMSpec(
        negative=p,
        positive=p,
        collector_cells=3,
        height_m=0.1,
        width_m=0.1,
        negative_collector_conductivity_S_m=100.0,
        positive_collector_conductivity_S_m=100.0,
    )
    control = ScalarControlProgram(((0.0, 0.3),), "A")
    result = advance_collector_spm(spec, spec.initial_state(), control, 1.0)
    assert result.outcome == "stopped"
    assert result.stopping_event == "battery:collector:minimum-voltage"
    assert result.final_state.time_s == 0.0
    assert result.final_state.coordinates == spec.initial_state().coordinates


@pytest.mark.parametrize("current", [-0.2, 0.0, 0.2])
def test_high_conductivity_limit_matches_original_uniform_spm(current):
    from chemworld.physchem.battery_models import BatterySPMSpec, battery_readout

    negative = ParticleSpec(cells=4, active_volume_m3=5e-7, diffusivity_m2_s=1e-12)
    positive = replace(negative, active_volume_m3=7e-7, exchange_current_scale_A_m2=2.0)
    spec = BatteryCollectorSPMSpec(
        negative=negative,
        positive=positive,
        collector_cells=5,
        initial_negative_fractions=0.63,
        initial_positive_fractions=0.37,
        height_m=0.1,
        width_m=0.1,
        negative_collector_conductivity_S_m=1e16,
        positive_collector_conductivity_S_m=1e16,
        series_resistance_ohm=0.1,
    )
    single = BatterySPMSpec(
        negative=negative,
        positive=positive,
        initial_negative_fraction=0.63,
        initial_positive_fraction=0.37,
        series_resistance_ohm=0.1,
        negative_ocv_intercept_V=spec.negative_ocv_intercept_V,
        negative_ocv_slope_V=spec.negative_ocv_slope_V,
        positive_ocv_intercept_V=spec.positive_ocv_intercept_V,
        positive_ocv_slope_V=spec.positive_ocv_slope_V,
    )
    f = collector_spm_fields(spec, spec.initial_state(), current_A=current)
    reference = battery_readout(single, single.initial_state(), current)
    assert f["voltage_V"] == pytest.approx(reference.voltage_V, abs=1e-10)
    np.testing.assert_allclose(f["through_current_A"], current / 5, atol=1e-10, rtol=0)
    np.testing.assert_allclose(
        f["surface_fraction"][:5], reference.negative_surface_fraction, atol=1e-11, rtol=0
    )
    np.testing.assert_allclose(
        f["surface_fraction"][5:], reference.positive_surface_fraction, atol=1e-11, rtol=0
    )
