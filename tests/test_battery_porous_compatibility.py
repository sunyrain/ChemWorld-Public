"""Old DFN behavior across extraction of the shared porous electrical kernel.

The frozen electrical values below were captured before extraction from an
asymmetric, explicitly declared DFN. They are development regression evidence,
not independent PyBaMM qualification. Polynomial antiderivatives separately
check the physical integration convention without copying its quadrature rule.
"""

import json

import numpy as np
import pytest
from numpy.polynomial import Polynomial

from chemworld.physchem.battery_dfn import (
    PHI_E,
    BatteryDFNSpec,
    _integral_centres,
    _linear_electrical,
    _reaction_currents,
    dfn_fields,
)
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import (
    BATTERY_ALGEBRAIC,
    BATTERY_COORDINATES,
    F,
    ParticleSpec,
    advance_battery,
    battery_spec_from_dict,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram


def asymmetric_dfn():
    return BatteryDFNSpec(
        negative=ParticleSpec(cells=4, active_volume_m3=5e-7),
        positive=ParticleSpec(cells=5, active_volume_m3=7e-7),
        electrolyte=BatteryElectrolyteSpec(
            cells=(3, 2, 4), area_m2=.017, lengths_m=(8e-5, 3e-5, 1.3e-4),
            porosities=(.35, .55, .42), bruggeman=(1.4, 1.7, 1.3),
            conductivity_S_m=.8, solid_conductivity_S_m=(3., 7.),
        ),
        series_resistance_ohm=.02,
        maximum_current_A=.4,
    )


@pytest.mark.parametrize("direction", [-1, 1])
def test_legacy_integral_is_exact_antiderivative_at_physical_centres(direction):
    faces = np.array([0., .03, .14, .21])
    conductivity = np.array([.2, 3., 1.1])
    field = Polynomial([direction * 2., direction * -7.])
    primitive = field.integ()
    expected, cumulative = [], 0.
    for left, right, coefficient in zip(faces[:-1], faces[1:], conductivity, strict=True):
        centre = (left + right) / 2
        expected.append(cumulative + (primitive(centre) - primitive(left)) / coefficient)
        cumulative += (primitive(right) - primitive(left)) / coefficient
    actual, total = _integral_centres(field(faces), np.diff(faces), conductivity)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-15)
    assert total == pytest.approx(cumulative, rel=0, abs=2e-15)


def test_old_dfn_electrical_fields_and_linear_maps_keep_asymmetric_geometry():
    spec = asymmetric_dfn()
    vector = np.array([.08, .04, -.03, -.05, -.09, -.12, .23, 3.7])
    assert spec.algebraic_names == (
        "battery:negative:reaction:0_A", "battery:negative:reaction:1_A",
        "battery:positive:reaction:0_A", "battery:positive:reaction:1_A",
        "battery:positive:reaction:2_A", PHI_E, *BATTERY_ALGEBRAIC,
    )
    values = dict(zip(spec.algebraic_names, vector, strict=True))
    q, ionic, electrolyte, solid, solid_faces = _linear_electrical(spec, values)
    expected_q = [.08, .04, .11, -.03, -.05, -.09, -.06]
    expected_electrolyte = [
        -.12008525788636419, -.12072469203409572, -.12182238732103483,
        -.12303607951662476, -.12373699170892953, -.12490859382979978,
        -.12636635868266968, -.1274366164227514, -.12790716077399422,
    ]
    expected_solid = [
        -.00005490196078431373, -.00013594771241830068, -.00018888888888888894,
        3.704706853991597, 3.7046979779411764, 3.704674763655462, 3.7046293592436976,
    ]
    np.testing.assert_allclose(q, expected_q, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(q, _reaction_currents(spec, values))
    np.testing.assert_allclose(electrolyte, expected_electrolyte, rtol=0, atol=2e-15)
    np.testing.assert_allclose(solid, expected_solid, rtol=0, atol=2e-15)
    # Gauss balance uses physical area and a zero-source separator, not a
    # uniform current distribution inferred from the number of electrode bins.
    np.testing.assert_allclose(
        np.diff(ionic) * .017, [.08, .04, .11, 0., 0., -.03, -.05, -.09, -.06],
        rtol=0, atol=1e-15,
    )
    assert ionic[0] == ionic[-1] == 0
    assert solid_faces[0][-1] == solid_faces[1][0] == 0
    assert solid_faces[0][0] == pytest.approx(.23 / .017)
    assert solid_faces[1][-1] == pytest.approx(.23 / .017)
    q_map, voltage_map = spec.electrical_maps
    np.testing.assert_allclose(q_map @ vector, expected_q, rtol=0, atol=1e-15)
    np.testing.assert_allclose(
        voltage_map @ vector, np.array(expected_solid) - np.r_[
            expected_electrolyte[:3], expected_electrolyte[-4:]], rtol=0, atol=2e-15,
    )
    # The eliminated last reaction retains both signed integral constraints
    # for every Jacobian column, including voltage and electrolyte gauge.
    expected_sums = np.zeros((2, len(vector)))
    expected_sums[:, -2] = (1., -1.)
    np.testing.assert_array_equal(
        np.vstack((q_map[:3].sum(axis=0), q_map[3:].sum(axis=0))), expected_sums,
    )


def test_old_dfn_codec_retains_radial_and_salt_stocks_without_nt_reinterpretation():
    spec = asymmetric_dfn()
    encoded = json.loads(json.dumps(spec.to_dict(), sort_keys=True))
    decoded = battery_spec_from_dict(encoded)
    assert type(decoded) is BatteryDFNSpec
    assert decoded == spec and decoded.to_dict() == spec.to_dict()
    assert decoded.inventory_keys == spec.inventory_keys
    assert len(decoded.particle_bins) == 7
    assert len(decoded.inventory_keys) == 3 * 4 + 4 * 5 + 9
    assert tuple(decoded.initial_state().coordinates) == BATTERY_COORDINATES
    assert decoded.initial_state().amounts_mol == spec.initial_state().amounts_mol
    assert "negative:Li" not in decoded.inventory_keys
    assert "positive:Li" not in decoded.inventory_keys


@pytest.mark.parametrize("target,unit,current,voltage", [
    (.2, "A", .2, 3.708380268410054),
    (-.2, "A", -.2, 3.9531866863573732),
    (3.8, "V", .04227355291790422, 3.8),
    (2.6, "V", .4, 3.63267866943138),
    (4.4, "V", -.4, 4.031023604154895),
])
def test_old_dfn_cc_and_finite_cv_match_pre_extraction_response(target, unit, current, voltage):
    spec = asymmetric_dfn()
    state = spec.initial_state()
    before = dict(state.amounts_mol)
    control = ScalarControlProgram(((0., target),), unit)
    fields = dfn_fields(spec, state, program=control)
    assert fields["current_A"] == pytest.approx(current, rel=0, abs=2e-12)
    assert fields["voltage_V"] == pytest.approx(voltage, rel=0, abs=2e-12)
    assert np.max(np.abs(fields["bv_residual_V"])) < 1e-11
    assert state.amounts_mol == before
    np.testing.assert_allclose(
        [sum(fields["reaction_current_A"][:3]), sum(fields["reaction_current_A"][3:])],
        [current, -current], rtol=0, atol=2e-12,
    )


@pytest.mark.parametrize("target,unit", [(.12, "A"), (3.8, "V")])
def test_old_dfn_short_cc_cv_retains_unique_faradaic_and_salt_accounts(target, unit):
    spec = asymmetric_dfn()
    initial = spec.initial_state()
    control = ScalarControlProgram(((0., target),), unit)
    result = advance_battery(spec, initial, control, .2)
    assert result.outcome == "success"
    final = result.final_state
    assert final.time_s == pytest.approx(.2, rel=0, abs=1e-14)
    charge = final.coordinates["discharged_charge_C"]
    assert charge > 0
    for name, sign in (("negative", -1), ("positive", 1)):
        keys = [key for electrode, _, _, keys, _ in spec.particle_bins
                if electrode == name for key in keys]
        change = sum(final.amounts_mol[k] - initial.amounts_mol[k] for k in keys)
        assert change == pytest.approx(sign * charge / F, rel=0, abs=1e-13)
    assert sum(final.amounts_mol[k] for k in spec.electrolyte.keys) == pytest.approx(
        sum(initial.amounts_mol[k] for k in spec.electrolyte.keys), rel=0, abs=1e-13,
    )
    fields = dfn_fields(spec, final, program=control)
    if unit == "A":
        assert charge == pytest.approx(.2 * target, rel=0, abs=1e-10)
    else:
        assert fields["voltage_V"] == pytest.approx(target, rel=0, abs=1e-10)
    assert set(final.amounts_mol) == set(initial.amounts_mol)
    assert set(final.coordinates) == set(BATTERY_COORDINATES)
