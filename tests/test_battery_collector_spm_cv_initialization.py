"""Fixed limited-CV repair probes; not substitute qualification histories."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_collector_spm import (
    BatteryCollectorSPMSpec,
    advance_collector_spm,
    collector_spm_algebraic,
    collector_spm_fields,
)
from chemworld.physchem.battery_models import F, ParticleSpec
from chemworld.physchem.coupled_system import CoupledStateView
from chemworld.physchem.reactor_shared import ScalarControlProgram


def limited_spec(tab, grid):
    """Exact physical S05/S06 inputs, with the fixed two joint meshes."""
    nz, nr = grid
    negative = ParticleSpec(
        radius_m=5e-6,
        diffusivity_m2_s=1e-13,
        maximum_concentration_mol_m3=30000.0,
        active_volume_m3=3e-7,
        cells=nr,
        mesh_exponent=1.0,
        exchange_current_scale_A_m2=4.0,
    )
    return BatteryCollectorSPMSpec(
        negative=negative,
        positive=replace(negative, active_volume_m3=4e-7),
        initial_negative_fractions=0.6,
        initial_positive_fractions=0.4,
        height_m=0.1,
        width_m=0.1,
        negative_collector_thickness_m=25e-6,
        positive_collector_thickness_m=25e-6,
        negative_collector_conductivity_S_m=1e4,
        positive_collector_conductivity_S_m=1e4,
        collector_cells=nz,
        negative_tab=tab,
        negative_ocv_intercept_V=0.25,
        negative_ocv_slope_V=-0.1,
        positive_ocv_intercept_V=4.1,
        positive_ocv_slope_V=-0.4,
        maximum_current_A=0.001,
    )


def view(state, values):
    return CoupledStateView(
        state.amounts_mol,
        state.temperature_K,
        state.volume_L,
        state.time_s,
        coordinates=state.coordinates,
        algebraic_values=values,
    )


def assert_faradaic(spec, initial, final):
    charge = final.coordinates["discharged_charge_C"] - initial.coordinates["discharged_charge_C"]
    for name, sign in (("negative", -1), ("positive", 1)):
        delta = sum(
            final.amounts_mol[k] - initial.amounts_mol[k]
            for k in spec.inventory_keys
            if k.startswith(name + ":")
        )
        assert F * delta == pytest.approx(sign * charge, rel=0, abs=1e-10)
    for z in range(spec.collector_cells):
        keys = (*spec.particle_bins[z][3], *spec.particle_bins[spec.collector_cells + z][3])
        assert sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in keys) == pytest.approx(
            0, rel=0, abs=2e-17
        )


@pytest.mark.parametrize("target,sign", [(3.4, 1), (4.1, -1)], ids=["S05", "S06"])
@pytest.mark.parametrize("tab", ["left", "right"], ids=["opposite", "same_end"])
@pytest.mark.parametrize("grid", [(8, 12), (16, 24)], ids=["coarse", "fine"])
def test_fixed_limited_cv_short_history(target, sign, tab, grid):
    spec = limited_spec(tab, grid)
    initial = spec.initial_state()
    program = ScalarControlProgram(((0.0, target),), "V")
    result = advance_collector_spm(
        spec, initial, program, 1e-4, evaluation_times_s=(0.0, 5e-5, 1e-4)
    )
    assert result.outcome == "success"
    assert result.final_state.time_s == 1e-4
    assert [s.time_s for s in result.trajectory] == [0.0, 5e-5, 1e-4]
    closure = collector_spm_algebraic(spec, program)
    for state in result.trajectory:
        fields = collector_spm_fields(spec, state, program=program)
        assert fields["current_A"] == pytest.approx(sign * 0.001, rel=0, abs=1e-11)
        assert sign * (fields["voltage_V"] - target) > 0
        # Every returned solution must satisfy the original CV complementarity,
        # not just an auxiliary constant-current candidate.
        np.testing.assert_allclose(
            closure.residual(view(state, fields["algebraic_values"])), 0, rtol=0, atol=1e-10
        )
        np.testing.assert_array_equal(
            fields["negative_reaction_current_A"], -fields["positive_reaction_current_A"]
        )
        assert state.algebraic_values == {}
        assert_faradaic(spec, initial, state)
    assert result.final_state.coordinates["discharged_charge_C"] == pytest.approx(
        sign * 1e-7, rel=0, abs=1e-13
    )


@pytest.mark.parametrize("target,sign", [(3.4, 1), (4.1, -1)], ids=["S05", "S06"])
@pytest.mark.parametrize("tab", ["left", "right"], ids=["opposite", "same_end"])
def test_limit_cc_seed_satisfies_original_complete_cv_residual(target, sign, tab):
    spec = limited_spec(tab, (8, 12))
    state = spec.initial_state()
    program = ScalarControlProgram(((0.0, target),), "V")
    closure = collector_spm_algebraic(spec, program)
    seed = closure.initial_values(state)
    cc = collector_spm_fields(spec, state, current_A=sign * spec.maximum_current_A)
    assert seed == cc["algebraic_values"]
    np.testing.assert_allclose(closure.residual(view(state, seed)), 0, rtol=0, atol=1e-11)
    # The other endpoint is an actual CC solution but is not a CV solution.
    opposite = collector_spm_fields(spec, state, current_A=-sign * spec.maximum_current_A)
    assert abs(closure.residual(view(state, opposite["algebraic_values"]))[-1]) > 0.3


@pytest.mark.parametrize("target,sign", [(3.4, 1), (4.1, -1)], ids=["S05", "S06"])
@pytest.mark.parametrize("tab", ["left", "right"], ids=["opposite", "same_end"])
@pytest.mark.parametrize("grid", [(8, 12), (16, 24)], ids=["coarse", "fine"])
def test_fixed_limited_cv_continues_from_actual_final_inventory(target, sign, tab, grid):
    """Separate prospectively fixed continuation block, using real final states."""
    spec = limited_spec(tab, grid)
    original = spec.initial_state()
    program = ScalarControlProgram(((0.0, target),), "V")
    first = advance_collector_spm(
        spec, original, program, 1e-4, evaluation_times_s=(0.0, 5e-5, 1e-4)
    )
    prior = first.final_state
    prior_amounts = dict(prior.amounts_mol)
    prior_coordinates = dict(prior.coordinates)
    resumed = advance_collector_spm(
        spec, prior, program, 1e-4, evaluation_times_s=(0.0, 5e-5, 1e-4)
    )
    assert resumed.outcome == "success"
    assert resumed.initial_state == first.final_state
    assert resumed.initial_state.coordinates == prior_coordinates
    assert resumed.initial_state.amounts_mol == prior_amounts
    assert prior.amounts_mol == prior_amounts
    assert prior.coordinates == prior_coordinates
    assert resumed.final_state.time_s == 2e-4
    assert [s.time_s for s in resumed.trajectory] == [
        prior.time_s + offset for offset in (0.0, 5e-5, 1e-4)
    ]
    assert resumed.final_state.coordinates["discharged_charge_C"] == pytest.approx(
        sign * 2e-7, rel=0, abs=2e-13
    )
    closure = collector_spm_algebraic(spec, program)
    voltages = []
    for state in resumed.trajectory:
        fields = collector_spm_fields(spec, state, program=program)
        assert fields["current_A"] == pytest.approx(sign * 0.001, rel=0, abs=1e-11)
        np.testing.assert_allclose(
            closure.residual(view(state, fields["algebraic_values"])), 0, rtol=0, atol=1e-10
        )
        voltages.append(fields["voltage_V"])
        assert_faradaic(spec, original, state)
        assert_faradaic(spec, prior, state)
    # Q/W retain their existing cumulative coordinates rather than resetting
    # on the second solve. The independently sampled IV quadrature agrees.
    delta_work = resumed.final_state.coordinates["delivered_work_J"] - prior_coordinates[
        "delivered_work_J"
    ]
    sampled_work = sign * 0.001 * 1e-4 / 6 * (voltages[0] + 4 * voltages[1] + voltages[2])
    assert delta_work == pytest.approx(sampled_work, rel=0, abs=1e-12)

