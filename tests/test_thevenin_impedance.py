"""Same-residual rest spectrum, known RC response and actual body heat response."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.battery_thevenin import (
    AMBIENT_LOSS,
    CELL_T,
    CHARGE,
    CURRENT,
    JIG_T,
    MODEL_HEAT,
    SOC,
    THROUGHPUT_E,
    THROUGHPUT_Q,
    WORK,
    rc_key,
)
from chemworld.physchem.residual_system import ResidualDomainError
from chemworld.physchem.thevenin_frequency import thevenin_impedance, thevenin_impedance_system
from test_battery_thevenin import fixture
from test_tabulated_functions import function


@pytest.mark.parametrize("count", (0, 1, 2, 4, 64))
@pytest.mark.parametrize("limit", (.1, 10.))
def test_real_port_spectrum_matches_rc_and_soc_capacitance(count, limit):
    spec = fixture(count, maximum_current_A=limit)
    frequencies = np.geomspace(1e-4, 1e2, 13)
    omega = 2*np.pi*frequencies
    expected = .1+sum(.2*(k+1)/(1+1j*omega) for k in range(count))+1/(1j*omega*100.)
    result = thevenin_impedance(spec, frequencies)
    assert result.success
    np.testing.assert_allclose(result.impedance_ohm, expected, rtol=2e-7, atol=1e-8)
    assert result.response.diagnostics["derivative_modes"]["state"] == "finite_difference"
    assert result.response.diagnostics["derivative_modes"]["parameter"] == "analytic"
    assert result.response.diagnostics["physical_time_advanced_s"] == 0
    index = {v.name: i for i, v in enumerate(result.response.variables)}
    assert {CELL_T, JIG_T, SOC, CURRENT} <= index.keys()
    assert not {CHARGE, WORK, THROUGHPUT_Q, THROUGHPUT_E, MODEL_HEAT, AMBIENT_LOSS} & index.keys()
    for point in result.response.points:
        response = point.response[:, 0]
        angular = 2*np.pi*point.frequency_Hz
        assert response[index[CURRENT]] == pytest.approx(1., abs=1e-9)
        assert response[index[SOC]] == pytest.approx(-1/(1j*angular*100), rel=2e-7)
        for k in range(count):
            assert response[index[rc_key(k)]] == pytest.approx(.2*(k+1)/(1+1j*angular), rel=2e-7)
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("entropy", (-1e-4, 0., 1e-4))
def test_two_real_body_responses_match_an_independent_thermal_linear_system(entropy):
    spec = fixture(2, entropy=entropy)
    result = thevenin_impedance(spec, [1e-4, .01, 1.])
    index = {v.name: i for i, v in enumerate(result.response.variables)}
    matrix = np.array([[-.5/100., .5/100.], [.5/500., -(.5+.2)/500.]])
    forcing = np.array([-298.15*entropy/100., 0.])
    for point in result.response.points:
        expected = np.linalg.solve(2j*np.pi*point.frequency_Hz*np.eye(2)-matrix, forcing)
        np.testing.assert_allclose(point.response[[index[CELL_T], index[JIG_T]], 0], expected,
                                   rtol=2e-7, atol=1e-10)


def test_receipt_history_is_not_a_second_state_or_a_source_of_spurious_derivatives():
    spec = fixture(2, entropy=1e-4)
    state = replace(spec.initial_state(), time_s=100., coordinates={
        **spec.initial_state().coordinates, CHARGE: 123., WORK: -12., THROUGHPUT_Q: 456.,
        THROUGHPUT_E: 789., MODEL_HEAT: 98., AMBIENT_LOSS: 42.})
    before = (dict(state.coordinates), state.time_s)
    result = thevenin_impedance(spec, [1., .001, .1, .001], state=state)
    expected = thevenin_impedance(spec, [1., .001, .1, .001])
    np.testing.assert_array_equal(result.impedance_ohm, expected.impedance_ohm)
    assert [p.frequency_Hz for p in result.response.points] == [1., .001, .1, .001]
    assert before == (dict(state.coordinates), state.time_s)
    _, initial = thevenin_impedance_system(spec, state)
    assert not initial.flags.writeable


@pytest.mark.parametrize("coordinate,value", ((CELL_T, 310.), (JIG_T, 310.),
                                              (rc_key(0), .01)))
def test_nonstationary_point_is_not_quietly_cooled_or_relaxed(coordinate, value):
    spec = fixture()
    state = replace(spec.initial_state(), coordinates={
        **spec.initial_state().coordinates, coordinate: value})
    before = dict(state.coordinates)
    with pytest.raises(ResidualDomainError, match="nonstationary"):
        thevenin_impedance(spec, [.01], state=state)
    assert before == dict(state.coordinates)


@pytest.mark.parametrize("defect", ("missing", "bulk", "frame", "cutoff"))
def test_wrong_ownership_frame_and_actual_stopping_faces_reject(defect):
    spec = fixture()
    state = spec.initial_state()
    if defect == "missing":
        state = replace(state, coordinates={})
    elif defect == "bulk":
        state = replace(state, amounts_mol={"fake-ion": 1.})
    elif defect == "frame":
        state = replace(state, temperature_K=310.)
    else:
        state = replace(state, coordinates={**state.coordinates, SOC: 0.})
    with pytest.raises((ValueError, ResidualDomainError)):
        thevenin_impedance(spec, [.01], state=state)


def test_true_ocv_kink_has_no_unique_spectrum_but_continuous_rest_rc_kink_is_valid():
    spec = fixture()
    kink = function({"soc": (0., .7, 1.)}, lambda z: 3+abs(z-.7), output_unit="V")
    with pytest.raises(ValueError, match="unique derivative"):
        thevenin_impedance(replace(spec, ocv=kink), [.01])
    # At zero polarization, R's parameter derivative multiplies zero. The
    # complete physical RHS is differentiable even when this continuous R is not.
    resistance = function({"current_A": (-10., 0., 10.)}, lambda i: .2+.01*abs(i),
                          input_units={"current_A": "A"})
    altered = replace(spec, rc_elements=((resistance, spec.rc_elements[0][1]),))
    a, b = thevenin_impedance(spec, [.01, 1.]), thevenin_impedance(altered, [.01, 1.])
    assert a.success and b.success
    np.testing.assert_allclose(a.impedance_ohm, b.impedance_ohm, rtol=2e-7, atol=1e-8)


@pytest.mark.parametrize("soc,slope", ((.5-1e-7, .4), (.5+1e-7, 2.)))
def test_ocv_local_slope_near_a_knot_is_not_averaged_across_segments(soc, slope):
    nodes = {0.: 3., .5: 3.2, 1.: 4.2}
    spec = replace(fixture(0, initial_soc=soc), ocv=function(
        {"soc": tuple(nodes)}, lambda z: nodes[z], output_unit="V"))
    frequencies = np.array([1e-4, .01, 1.])
    result = thevenin_impedance(spec, frequencies)
    assert result.success
    expected = .1+slope/(2j*np.pi*frequencies*100.)
    np.testing.assert_allclose(result.impedance_ohm, expected, rtol=2e-7, atol=1e-8)


def test_lazy_public_entry_is_the_real_native_analysis():
    from chemworld import physchem

    assert physchem.thevenin_impedance is thevenin_impedance
    assert physchem.thevenin_impedance_system is thevenin_impedance_system
