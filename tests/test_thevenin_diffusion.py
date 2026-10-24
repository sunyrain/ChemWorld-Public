"""Functional invariants of the shared-residual optional SOC diffusion module."""

import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.scientific_inputs import synthetic_source
from chemworld.physchem.battery_thevenin import (
    CELL_T,
    SOC,
    TheveninSpec,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.battery_thevenin_diffusion import TheveninDiffusionSpec
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualDomainError
from chemworld.physchem.thevenin_frequency import thevenin_impedance
from test_battery_thevenin import fixture
from test_tabulated_functions import function


def diffusion_fixture(tau=30., cells=8, *, faces=None):
    source = synthetic_source()
    name = "diffusion_time_constant_s"
    table = ScientificTable("diffusion-fixture", "parameters",
        (QuantityColumn(name, "s", "parameter", (1e-9, 1e9), "Functional-test time scale"),),
        (source,), (ScientificRow("cell", source.provenance.source_id, "model", "diffusion",
            "parameters", {name: InputValue(tau, "s", None, "Uncertainty not supplied")}),))
    return TheveninDiffusionSpec(table, "cell", tuple(np.linspace(0., 1., cells+1))
                                if faces is None else faces)


def spec_fixture(**kwargs):
    return replace(fixture(0), diffusion=diffusion_fixture(**kwargs))


def cc(value):
    return ScalarControlProgram(((0., value),), "A")


def uniform_matrix(n, tau):
    # Independent constant-coefficient cell balance, not a native FV call.
    matrix = np.diag(np.r_[-1., np.full(n-2, -2.), -1.])
    matrix += np.diag(np.ones(n-1), 1)+np.diag(np.ones(n-1), -1)
    return matrix*n*n/tau


def test_codec_keeps_old_payload_exact_and_new_grid_and_source_roundtrip():
    old = fixture()
    assert "diffusion" not in old.to_dict()
    assert TheveninSpec.from_dict(old.to_dict()).to_dict() == old.to_dict()
    spec = spec_fixture(faces=(0., .1, .3, .6, 1.))
    payload = json.loads(json.dumps(spec.to_dict(), allow_nan=False))
    restored = TheveninSpec.from_dict(payload)
    assert restored.to_dict() == payload
    assert len(restored.diffusion.coordinate_keys) == 3
    assert restored.initial_state().coordinates == spec.initial_state().coordinates


@pytest.mark.parametrize("defect", ("grid", "cells", "convention", "tau", "ocv"))
def test_invalid_physical_declarations_fail_explicitly(defect):
    diffusion = diffusion_fixture()
    with pytest.raises(ValueError):
        if defect == "grid":
            replace(diffusion, faces=(0., .3, .2, 1.))
        elif defect == "cells":
            replace(diffusion, faces=(0., 1.))
        elif defect == "convention":
            replace(diffusion, convention="unlabelled-upstream-25.4.2")
        elif defect == "tau":
            diffusion_fixture(tau=-1.)
        else:
            replace(fixture(), diffusion=diffusion,
                    ocv=function({"soc": (0., 1.)}, lambda z: 4-z, output_unit="V"))


def test_nonuniform_eliminated_state_conserves_weighted_mean_and_boundary_flux():
    diffusion = diffusion_fixture(faces=(0., .1, .35, .7, 1.))
    coordinates = dict(zip(diffusion.coordinate_keys, (.03, -.02, .01), strict=True))
    profile = diffusion.profile(.6, coordinates)
    assert diffusion.weights @ profile == pytest.approx(.6, abs=1e-15)
    rates = diffusion.deviation_rates(coordinates, .2, 100.)
    derivative = diffusion.deviations(rates)-.002
    assert diffusion.weights @ derivative == pytest.approx(-.002, abs=1e-16)
    boundary = diffusion._operator @ profile
    boundary[-1] -= .002/diffusion.weights[-1]
    np.testing.assert_allclose(derivative, boundary, atol=1e-15)
    centers = diffusion.centers
    expected = profile[-1]+(1-centers[-1])*(profile[-1]-profile[-2])/(centers[-1]-centers[-2])
    assert diffusion.surface_soc(.6, coordinates) == pytest.approx(expected)


@pytest.mark.parametrize("current", (-.2, .2))
def test_actual_common_kernel_matches_independent_cell_solution_and_heat_sign(current):
    spec = spec_fixture()
    initial = spec.initial_state()
    result = thevenin_system(spec, cc(current)).advance(initial, 3.)
    assert result.outcome == "success"
    fields = thevenin_fields(spec, result.final_state, cc(current))
    augmented = np.zeros((9, 9))
    augmented[:8, :8] = uniform_matrix(8, 30.)
    augmented[7, 8] = -current/100.*8
    expected = (expm(3.*augmented) @ np.r_[np.full(8, .7), 1.])[:8]
    np.testing.assert_allclose(fields["distributed_soc"], expected, atol=4e-8)
    assert fields["surface_soc"] == pytest.approx(1.5*expected[-1]-.5*expected[-2], abs=5e-8)
    assert fields["voltage_V"] == pytest.approx(3+fields["surface_soc"]-.1*current, abs=1e-10)
    assert current*fields["diffusion_overpotential_V"] < 0
    assert fields["diffusion_heat_W"] > 0
    assert result.final_state.coordinates[CELL_T] > initial.coordinates[CELL_T]
    assert result.conservation_residuals["circuit_charge_C"] == pytest.approx(0., abs=1e-8)
    assert result.conservation_residuals["empirical_node_heat_J"] == pytest.approx(0., abs=1e-7)
    assert result.final_state.amounts_mol == {}


def test_reverse_current_heat_is_not_clipped_and_rest_reuses_saved_profile():
    spec = spec_fixture()
    discharged = thevenin_system(spec, cc(.2)).advance(spec.initial_state(), 10.).final_state
    before = thevenin_fields(spec, discharged, cc(.2))
    reverse = thevenin_fields(spec, discharged, cc(-.2))
    assert reverse["diffusion_heat_W"] == pytest.approx(-before["diffusion_heat_W"])
    assert reverse["diffusion_heat_W"] < 0
    relaxed = thevenin_system(spec, cc(0.)).advance(discharged, 30.).final_state
    after = thevenin_fields(spec, relaxed, cc(0.))
    assert relaxed.coordinates[SOC] == discharged.coordinates[SOC]
    assert abs(after["diffusion_overpotential_V"]) < abs(before["diffusion_overpotential_V"])/100
    assert after["diffusion_heat_W"] == 0.


@pytest.mark.parametrize("defect", ("internal", "surface"))
def test_actual_profile_not_just_mean_soc_is_validated(defect):
    spec = spec_fixture(cells=4)
    deviations = (.6, -.2, -.2) if defect == "internal" else (.0, -.3, -.1)
    state = spec.initial_state()
    coordinates = {**state.coordinates, SOC: .5,
                   **dict(zip(spec.diffusion.coordinate_keys, deviations, strict=True))}
    with pytest.raises(ValueError, match="internal/surface SOC"):
        thevenin_system(spec, cc(0.)).advance(replace(state, coordinates=coordinates), 0.)


def test_surface_stopping_event_precedes_bulk_exhaustion_without_clipping():
    spec = replace(fixture(0, initial_soc=.08), diffusion=diffusion_fixture(tau=300.))
    result = thevenin_system(spec, cc(.5)).advance(spec.initial_state(), 15.)
    assert result.actual_duration_s < 15.
    fields = thevenin_fields(spec, result.final_state, cc(.5))
    assert fields["surface_soc"] == pytest.approx(0., abs=2e-14)
    assert fields["soc"] > .03
    assert min(fields["distributed_soc"]) > 0.


def test_finite_source_cv_uses_the_same_surface_voltage_and_profile():
    spec = replace(fixture(0, maximum_current_A=.2), diffusion=diffusion_fixture())
    program = ScalarControlProgram(((0., 3.66),), "V")
    result = thevenin_system(spec, program).advance(spec.initial_state(), 15.)
    fields = thevenin_fields(spec, result.final_state, program)
    assert fields["voltage_V"] == pytest.approx(3.66, abs=1e-9)
    assert 0 < fields["current_A"] < .2
    assert fields["surface_soc"] < fields["soc"]


@pytest.mark.parametrize("soc,slope", ((.5-1e-7, .4), (.5+1e-7, 2.)))
def test_frequency_reuses_spatial_residual_and_stays_in_the_local_ocv_segment(soc, slope):
    spec = replace(fixture(0, initial_soc=soc), diffusion=diffusion_fixture(),
        ocv=function({"soc": (0., .5, 1.)}, lambda z: 3+.4*z if z <= .5 else 2.2+2*z,
                     output_unit="V"))
    state = spec.initial_state()
    snapshot = dict(state.coordinates)
    frequencies = (.0001, .01, 1.)
    result = thevenin_impedance(spec, frequencies)
    assert result.success
    matrix = uniform_matrix(8, 30.)
    forcing = np.r_[np.zeros(7), -.08]
    surface = np.r_[np.zeros(6), -.5, 1.5]
    expected = [.1-slope*surface @ np.linalg.solve(2j*np.pi*f*np.eye(8)-matrix, forcing)
                for f in frequencies]
    np.testing.assert_allclose(result.impedance_ohm, expected, rtol=5e-7, atol=1e-8)
    assert dict(state.coordinates) == snapshot
    changed = replace(state, coordinates={**state.coordinates,
                                           spec.diffusion.coordinate_keys[0]: .001})
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        thevenin_impedance(spec, frequencies, state=changed)


@pytest.mark.parametrize("cells", (64, 128))
def test_fine_grid_cv_crosses_current_limit_without_projection_stencil_stalling(cells):
    spec = replace(fixture(0, initial_soc=.6, maximum_current_A=.2),
                   diffusion=diffusion_fixture(cells=cells))
    program = ScalarControlProgram(((0., 3.57),), "V")
    system = thevenin_system(spec, program)
    system = replace(system, policy=replace(system.policy, rtol=1e-9, atol=1e-12))
    result = system.advance(spec.initial_state(), 1., evaluation_times_s=(0., .5, 1.))
    initial = thevenin_fields(spec, spec.initial_state(), program)
    final = thevenin_fields(spec, result.final_state, program)
    assert initial["current_A"] == pytest.approx(.2, abs=1e-12)
    assert initial["voltage_V"] > 3.57
    assert result.actual_duration_s == 1. and result.outcome == "success"
    assert 0 < final["current_A"] < .2
    assert final["voltage_V"] == pytest.approx(3.57, abs=1e-9)


def test_monotone_current_dependent_resistance_cv_switch_is_continuous_on_both_limits():
    spec = replace(fixture(0, maximum_current_A=.2), diffusion=diffusion_fixture(),
        series_resistance=function({"current_A": (-.2, 0., .2)}, lambda i: .1+.05*i,
                                   input_units={"current_A": "A"}))
    state = spec.initial_state()
    for sign in (-1, 1):
        limit = sign*.2
        voltage_at_limit = 3.7-limit*(.1+.05*limit)
        fields = []
        for offset in (-1e-7, 0., 1e-7):
            target = voltage_at_limit+offset
            program = ScalarControlProgram(((0., target),), "V")
            f = thevenin_fields(spec, state, program)
            if sign*offset <= 0:
                assert f["current_A"] == pytest.approx(limit, abs=1e-10)
                assert f["voltage_V"] == pytest.approx(voltage_at_limit, abs=1e-10)
            else:
                assert abs(f["current_A"]) < .2
                assert f["voltage_V"] == pytest.approx(target, abs=1e-10)
            fields.append(f)
        assert max(f["current_A"] for f in fields)-min(f["current_A"] for f in fields) < 2e-6


@pytest.mark.parametrize("zero", (True, False))
def test_diffusion_cv_still_rejects_zero_or_nonmonotone_instantaneous_resistance(zero):
    resistance = function({}, lambda: 0.) if zero else function(
        {"current_A": (-.2, 0., .2)}, lambda i: .1-.4*i, input_units={"current_A": "A"})
    spec = replace(fixture(0, maximum_current_A=.2), diffusion=diffusion_fixture(),
                   series_resistance=resistance)
    # CC remains defined, including its zero-series-resistance limit.
    thevenin_system(spec, cc(.1)).advance(spec.initial_state(), 0.)
    with pytest.raises(ValueError, match="certified nonsingular monotone"):
        thevenin_system(spec, ScalarControlProgram(((0., 3.65),), "V"))
