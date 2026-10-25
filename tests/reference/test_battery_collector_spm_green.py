"""Deterministic independent-oracle tests; no production collector imports.

These checks establish the reference's mathematics and conventions. They do
not evaluate a new native/source trajectory or alter frozen scientific results.
"""

from copy import deepcopy
from itertools import pairwise

import numpy as np
import pytest
from battery_collector_spm_green import (
    F,
    R,
    StaticClosureError,
    integrate_collectors,
    solve_static_collector,
    static_inputs_from_case,
)
from scipy.integrate import quad


def inputs(cells=3):
    return {
        "z_edges_m": np.linspace(0, 0.12, cells + 1),
        "negative_conductance_S_m": 0.3,
        "positive_conductance_S_m": 0.4,
        "negative": {
            "outer_fraction": 0.6,
            "interface_area_m2": 0.025,
            "exchange_current_scale_A_m2": 4.0,
            "ocv_intercept_V": 0.25,
            "ocv_slope_V": -0.1,
        },
        "positive": {
            "outer_fraction": 0.4,
            "interface_area_m2": 0.035,
            "exchange_current_scale_A_m2": 3.0,
            "ocv_intercept_V": 4.1,
            "ocv_slope_V": -0.4,
        },
        "current_limit_A": 0.6,
    }


@pytest.mark.parametrize("tab", ["left", "right"])
@pytest.mark.parametrize("current", [-0.8, 0.7])
def test_uniform_current_density_matches_continuous_quadratic_and_average(tab, current):
    edges = np.array([0, 0.01, 0.045, 0.12])
    h, kn, kp, voltage = edges[-1], 0.13, 0.21, 3.7
    widths = np.diff(edges)
    density = current / h
    result = integrate_collectors(
        density * widths,
        z_edges_m=edges,
        negative_conductance_S_m=kn,
        positive_conductance_S_m=kp,
        negative_tab=tab,
        physical_tab_voltage_V=voltage,
    )

    def phi_n(z):
        if tab == "left":
            return -density * (h * z - z * z / 2) / kn
        return density * (z * z - h * h) / (2 * kn)

    def phi_p(z):
        return voltage + density * (h * h - z * z) / (2 * kp)

    for name, function in (("negative", phi_n), ("positive", phi_p)):
        np.testing.assert_allclose(result[name]["face_potential_V"], function(edges), atol=1e-15)
        np.testing.assert_allclose(
            result[name]["continuous_centre_V"],
            function((edges[1:] + edges[:-1]) / 2),
            rtol=0,
            atol=1e-15,
        )
        averages = [quad(function, a, b)[0] / (b - a) for a, b in pairwise(edges)]
        np.testing.assert_allclose(result[name]["continuous_mean_V"], averages, rtol=0, atol=1e-15)


@pytest.mark.parametrize("tab", ["left", "right"])
def test_signed_nonuniform_currents_obey_continuity_ohm_and_both_power_identities(tab):
    edges = np.array([0, 0.01, 0.04, 0.09, 0.12])
    widths = np.diff(edges)
    q = np.array([0.1, -0.3, 0.25, 0.2])
    kn, kp, voltage = 0.2, 0.31, 3.6
    result = integrate_collectors(
        q,
        z_edges_m=edges,
        negative_conductance_S_m=kn,
        positive_conductance_S_m=kp,
        negative_tab=tab,
        physical_tab_voltage_V=voltage,
    )
    np.testing.assert_allclose(
        np.diff(result["negative"]["face_current_A"]), -q, rtol=0, atol=1e-16
    )
    np.testing.assert_allclose(np.diff(result["positive"]["face_current_A"]), q, rtol=0, atol=1e-16)
    current = sum(q)
    assert result["negative"]["face_current_A"][0 if tab == "left" else -1] == pytest.approx(
        current if tab == "left" else -current, abs=1e-15
    )
    assert result["negative"]["face_potential_V"][0 if tab == "left" else -1] == 0
    assert result["positive"]["face_potential_V"][-1] == voltage
    quadrature_power = 0
    for name, conductance in (("negative", kn), ("positive", kp)):
        face = result[name]["face_current_A"]
        for i, (left, right) in enumerate(pairwise(edges)):

            def local_current(z, i=i, left=left, right=right, face=face):
                return face[i] + (face[i + 1] - face[i]) * (z - left) / (right - left)

            drop = -quad(local_current, left, right)[0] / conductance
            assert np.diff(result[name]["face_potential_V"])[i] == pytest.approx(drop, abs=1e-15)
            quadrature_power += quad(
                lambda z, conductance=conductance: local_current(z) ** 2 / conductance,
                left,
                right,
            )[0]
    assert result["continuous_joule_W"] == pytest.approx(quadrature_power, rel=1e-14)
    for observation, power in (
        ("continuous_mean_V", "continuous_joule_W"),
        ("fv_centre_V", "fv_joule_W"),
    ):
        assert np.dot(
            q, result["local_voltage_V"][observation]
        ) - current * voltage == pytest.approx(result[power], abs=5e-16)
        assert result[power] > 0
    assert result["fv_joule_W"] - result["continuous_joule_W"] == pytest.approx(
        np.sum(q**2 * widths) / 6 * (1 / kn + 1 / kp), rel=1e-13
    )


@pytest.mark.parametrize("tab", ["left", "right"])
def test_three_observations_and_source_tab_bias_are_separate_without_gauge_fit(tab):
    edges = np.array([0, 0.013, 0.05, 0.1])
    widths = np.diff(edges)
    q = np.array([0.2, -0.05, 0.4])
    kn, kp = 0.17, 0.26
    result = integrate_collectors(
        q,
        z_edges_m=edges,
        negative_conductance_S_m=kn,
        positive_conductance_S_m=kp,
        negative_tab=tab,
        physical_tab_voltage_V=3.8,
    )
    for name, sign, conductance in (("negative", 1, kn), ("positive", -1, kp)):
        field = result[name]
        for observation, denominator in (("continuous_mean_V", 6), ("continuous_centre_V", 8)):
            np.testing.assert_allclose(
                field[observation] - field["fv_centre_V"],
                sign * q * widths / (denominator * conductance),
                rtol=0,
                atol=8e-16,
            )
    assert result["source_boundary_voltage_V"] - 3.8 == pytest.approx(
        q[-1] * widths[-1] / (2 * kp), abs=1e-15
    )
    tab_cell = 0 if tab == "left" else -1
    assert result["negative_extrapolated_tab_voltage_V"] == pytest.approx(
        -q[tab_cell] * widths[tab_cell] / (2 * kn), abs=1e-15
    )
    shifted = integrate_collectors(
        q,
        z_edges_m=edges,
        negative_conductance_S_m=kn,
        positive_conductance_S_m=kp,
        negative_tab=tab,
        physical_tab_voltage_V=5.8,
    )
    np.testing.assert_array_equal(
        shifted["negative"]["fv_centre_V"], result["negative"]["fv_centre_V"]
    )
    np.testing.assert_allclose(
        shifted["positive"]["fv_centre_V"] - result["positive"]["fv_centre_V"],
        2,
        rtol=0,
        atol=1e-15,
    )
    assert shifted["continuous_joule_W"] == result["continuous_joule_W"]


@pytest.mark.parametrize(
    "observation,network_factor",
    [
        ("fv_centre_V", 3 / 4),
        ("continuous_centre_V", 11 / 16),
        ("continuous_mean_V", 2 / 3),
    ],
)
@pytest.mark.parametrize("current", [-0.3, 0.3])
def test_static_two_cell_symmetry_matches_independent_series_bv_and_resistance(
    observation, network_factor, current
):
    kwargs = inputs(2)
    kwargs["positive_conductance_S_m"] = kwargs["negative_conductance_S_m"]
    rs = 0.12
    answer = solve_static_collector(
        **kwargs, target=current, series_resistance_ohm=rs, collector_observation=observation
    )
    np.testing.assert_allclose(answer["through_current_A"], current / 2, rtol=0, atol=1e-12)
    ocv, losses = {}, []
    for name in ("negative", "positive"):
        material = kwargs[name]
        theta = material["outer_fraction"]
        ocv[name] = material["ocv_intercept_V"] + material["ocv_slope_V"] * theta
        j0 = material["exchange_current_scale_A_m2"] * np.sqrt(theta * (1 - theta))
        losses.append(
            2 * R * 298.15 / F * np.arcsinh(current / (4 * material["interface_area_m2"] * j0))
        )
    expected = ocv["positive"] - ocv["negative"] - sum(losses)
    expected -= current * (rs + network_factor * 0.12 / kwargs["negative_conductance_S_m"])
    assert answer["voltage_V"] == pytest.approx(expected, abs=1e-12)


def test_zero_terminal_current_permits_real_internal_currents_exceeding_source_limit():
    kwargs = inputs()
    kwargs["negative"]["outer_fraction"] = [0.2, 0.5, 0.8]
    kwargs["positive"]["outer_fraction"] = [0.8, 0.4, 0.2]
    answer = solve_static_collector(**(kwargs | {"current_limit_A": 0.01}), target=0)
    q = answer["through_current_A"]
    assert min(q) < -0.01 and max(q) > 0.01
    assert sum(q) == pytest.approx(0, abs=1e-13)
    assert answer["collectors"]["continuous_joule_W"] > 0
    for name, sign in (("negative", 1), ("positive", -1)):
        field = answer[name]
        forward = (
            2
            * kwargs[name]["interface_area_m2"]
            * field["j0_A_m2"]
            * np.sinh(F * field["kinetic_overpotential_V"] / (2 * R * 298.15))
        )
        np.testing.assert_allclose(forward, sign * q, rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("voltage_observation", ["physical-tab", "source-boundary"])
@pytest.mark.parametrize("current", [-0.25, 0.25])
def test_cv_returns_same_independent_cc_root_for_selected_voltage_observation(
    voltage_observation, current
):
    kwargs = inputs()
    cc = solve_static_collector(**kwargs, target=current, voltage_observation=voltage_observation)
    cv = solve_static_collector(
        **kwargs,
        control_mode="V",
        target=cc["controlled_voltage_V"],
        voltage_observation=voltage_observation,
    )
    assert cv["mode"] == "CV"
    np.testing.assert_allclose(cv["through_current_A"], cc["through_current_A"], rtol=0, atol=2e-12)
    assert cv["voltage_V"] == pytest.approx(cc["voltage_V"], abs=1e-12)
    if voltage_observation == "source-boundary":
        assert abs(cv["voltage_V"] - cv["controlled_voltage_V"]) > 1e-4


@pytest.mark.parametrize(
    "target,mode,sign", [(2.5, "positive-limit", 1), (5.0, "negative-limit", -1)]
)
def test_finite_cv_enumerates_active_limits_and_keeps_rejected_unlimited_candidate(
    target, mode, sign
):
    answer = solve_static_collector(**inputs(), control_mode="V", target=target)
    assert answer["mode"] == mode
    assert answer["current_A"] == pytest.approx(sign * 0.6, abs=1e-12)
    assert sign * (answer["controlled_voltage_V"] - target) > 0
    assert answer["solver_attempts"][0]["branch"] == "CV"
    assert not answer["solver_attempts"][0]["admissible"]


def test_shell_flux_bound_is_enforced_without_state_or_terminal_current_clipping():
    kwargs = inputs()
    kwargs["negative"]["surface_flux_coefficient_A_inv"] = 0.3
    kwargs["positive"]["surface_flux_coefficient_A_inv"] = 0.2
    answer = solve_static_collector(**kwargs, target=0.2)
    q = answer["through_current_A"]
    np.testing.assert_allclose(
        answer["negative"]["surface_fraction"], 0.6 - 0.3 * q, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        answer["positive"]["surface_fraction"], 0.4 + 0.2 * q, rtol=0, atol=1e-15
    )
    assert answer["current_A"] == pytest.approx(0.2, abs=1e-12)
    assert kwargs["negative"]["outer_fraction"] == 0.6
    kwargs["negative"]["surface_flux_coefficient_A_inv"] = 100
    with pytest.raises(StaticClosureError) as error:
        solve_static_collector(**kwargs, target=0.2)
    attempts = error.value.diagnostic["attempts"]
    assert len(attempts) == 1 and attempts[0]["residual_max"] > 1e-10


def case_inputs():
    # Explicit synthetic authored data; no production or frozen-matrix adapter.
    particle = {
        "radius_m": 5e-6,
        "diffusivity_m2_s": 1e-13,
        "maximum_concentration_mol_m3": 30000,
        "active_volume_m3": 3e-7,
        "exchange_current_scale_A_m2": 4.0,
        "ocv": (0.25, -0.1),
    }
    return {
        "height_m": 0.1,
        "width_m": 0.08,
        "sigma_cc_S_m": 1000,
        "collector_thickness_m": 2e-5,
        "temperature_K": 298.15,
        "maximum_current_A": 1.0,
        "series_resistance_ohm": 0.02,
        "surface_margin": 0.01,
        "negative": particle,
        "positive": deepcopy(particle) | {"ocv": (4.1, -0.4)},
        "initial_z_fractions": {"negative": [0.5, 0.6, 0.7], "positive": [0.4, 0.3, 0.2]},
    }


def test_authored_conversion_keeps_equal_grid_values_and_declares_surface_convention():
    data = case_inputs()
    original = deepcopy(data)
    layout = {"negative_tab": "right", "positive_tab": "right"}
    shell = static_inputs_from_case(data, {"z": 3, "r": 1}, layout)
    given = static_inputs_from_case(data, {"z": 3, "r": 1}, layout, surface_model="given-surface")
    for name in ("negative", "positive"):
        np.testing.assert_array_equal(
            shell[name]["outer_fraction"], data["initial_z_fractions"][name]
        )
        radius = data[name]["radius_m"]
        area = 3 * data[name]["active_volume_m3"] / (radius * 3)
        expected_b = radius / (5 * F * area * 1e-13 * 30000)
        assert shell[name]["surface_flux_coefficient_A_inv"] == pytest.approx(expected_b, rel=2e-15)
        assert given[name]["surface_flux_coefficient_A_inv"] == 0
    assert data == original
    assert shell["negative_conductance_S_m"] == pytest.approx(0.0016)
    supplied = static_inputs_from_case(
        data,
        {"z": 3, "r": 2},
        layout,
        surface_model="given-surface",
        negative_outer_fraction=[0.7, 0.6, 0.5],
    )
    np.testing.assert_array_equal(supplied["negative"]["outer_fraction"], [0.7, 0.6, 0.5])


@pytest.mark.parametrize(
    "changes",
    [
        {"z_edges_m": [0, 0.1]},
        {"z_edges_m": [0, 0.1, 0.09]},
        {"negative_conductance_S_m": 0},
        {"negative_tab": "middle"},
        {"target": np.nan},
        {"target": 1.0},
        {"control_mode": "W"},
        {"collector_observation": "undeclared"},
        {"voltage_observation": "fitted-gauge"},
        {"surface_margin": 0},
        {"temperature_K": -1},
        {"series_resistance_ohm": -1},
    ],
)
def test_static_contract_rejects_invalid_geometry_material_domain_or_control(changes):
    with pytest.raises(ValueError):
        solve_static_collector(**(inputs() | changes))


def test_static_contract_rejects_unsupported_nonmonotone_ocv_and_supplied_surface():
    kwargs = inputs()
    kwargs["positive"]["ocv_slope_V"] = 0.2
    with pytest.raises(ValueError, match="monotone"):
        solve_static_collector(**kwargs)
    kwargs = inputs()
    kwargs["negative"]["outer_fraction"] = 1e-10
    with pytest.raises(ValueError, match="margin"):
        solve_static_collector(**kwargs)
