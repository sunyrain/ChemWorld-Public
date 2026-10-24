"""Preparation-only construction contracts; not fixed-block scientific outcomes."""

from copy import deepcopy

import numpy as np
import pytest
from battery_msmr_cases import case_data, initial_profile, occupancies, potential
from battery_msmr_pybamm import build_reference, extract_segment


def test_fixed_denominators():
    cases = [case_data(j) for j in range(1, 13)]
    assert len(cases) == 12
    assert sum(x["expected"] not in {"static", "refusals"} for x in cases) == 10
    assert 10 * 4 + 4 + 1 + 8 == 53


@pytest.mark.parametrize("index", range(2, 12))
def test_released_source_build(index):
    sim = build_reference(case_data(index), 8)
    sim.build()
    assert sim.built_model is not None
    assert sim.model.options["particle"] == "MSMR"
    assert sim.model.options["intercalation kinetics"] == "MSMR"
    assert sim.model.options["open-circuit potential"] == "MSMR"


@pytest.mark.parametrize("index", (4, 7, 9))
def test_short_source_shapes_and_actual_initial_profile(index):
    data = deepcopy(case_data(index))
    data["samples_per_segment"] = 3
    sim = build_reference(data, 8)
    _, target, unit = data["waveform"][0]
    solution = sim.step(
        0.2,
        save=False,
        t_eval=np.linspace(0.0, 0.2, 3),
        inputs={"MSMR voltage target" if unit == "V" else "Current function [A]": target},
    )
    segment = extract_segment(solution, data)
    assert segment["times_s"][0] == 0.0
    for e in data["electrodes"]:
        _, u, x, weights = initial_profile(e, 8, data["temperature_K"])
        raw = segment["raw_profiles"][e["name"]]
        assert np.asarray(raw["theta"]).shape == (3, 8)
        # PyBaMM's CODATA F has extra digits beyond the native declared constant.
        assert raw["theta"][0] == pytest.approx(x, abs=2e-10)
        assert raw["potential_V"][0] == pytest.approx(u, abs=2e-12)
        assert raw["shell_weights"] == pytest.approx(weights, abs=2e-14)


def test_initial_potential_is_same_declared_logistic():
    for e in case_data(4)["electrodes"]:
        u = potential(e["rows"], e["initial_fraction"], 298.15)
        assert occupancies(e["rows"], u, 298.15).sum() == pytest.approx(
            e["initial_fraction"], abs=1e-12
        )


@pytest.mark.parametrize("index", range(2, 12))
def test_native_declared_build_and_initial_inventory(index):
    from battery_msmr_cases import F, native_initial_state, native_spec

    from chemworld.physchem.battery_msmr import msmr_readout

    data = case_data(index)
    spec = native_spec(data, 8)
    state = native_initial_state(spec, data, 8)
    msmr_readout(spec, state, 0.0)
    for e in data["electrodes"]:
        p = getattr(spec, e["name"])
        amount = sum(state.amounts_mol[k] for k in p.keys(e["name"]))
        assert amount == pytest.approx(
            p.capacity_mol * getattr(spec, "initial_" + e["name"] + "_fraction"), abs=1e-14
        )
    assert state.coordinates["discharged_charge_C"] == 0.0
    assert F > 96485.0


@pytest.mark.parametrize("index", (4, 7, 9))
def test_short_native_matches_response_shapes(index):
    from battery_msmr_traces import native_time

    data = case_data(index)
    _, target, unit = data["waveform"][0]
    data["waveform"], data["samples_per_segment"] = ((0.2, target, unit),), 3
    route = native_time(data, 8)
    segment = route["segments"][0]
    assert segment["times_s"][0] == 0.0
    for e in data["electrodes"]:
        raw = segment["raw_profiles"][e["name"]]
        assert np.asarray(raw["theta"]).shape == (3, 8)
        assert np.asarray(raw["potential_V"]).shape == (3, 8)


def test_matrix_limit_conserves_and_uses_continuum_multiplier():
    from battery_msmr_traces import linear_fick_limit

    data = case_data(3)
    data["waveform"], data["samples_per_segment"] = ((0.2, 0.02, "A"),), 3
    trace = linear_fick_limit(data, 8)
    assert trace["lithium_balance_mol"] < 2e-12
    for profile in trace["profiles"].values():
        assert profile["diffusivity_m2_s"] == 2e-13
        assert profile["faraday_error_C"] < 1e-8


def test_short_assessment_and_source_scalar_wiring():
    from battery_msmr_assess import compare, history_metrics
    from battery_msmr_pybamm import independent_bv, reference_time, static_reference_values
    from battery_msmr_traces import native_time

    data = case_data(4)
    data["waveform"], data["samples_per_segment"] = ((0.2, 0.4, "A"),), 3
    native, reference = native_time(data, 8), reference_time(data, 8)
    assert compare(native, reference)["passed"]
    assert history_metrics(native, data)["negative_faraday_error_C"] < 2e-4
    electrode = data["electrodes"][0]
    actual = static_reference_values(data, electrode, 298.15, 0.16, 0.03, 500.0)
    independent = independent_bv(electrode["rows"], 0.16, 0.03, 298.15, 500.0)
    for key in actual:
        assert actual[key] == pytest.approx(independent[key], rel=2e-9, abs=2e-10)


@pytest.mark.parametrize(
    "raw", ("msmr:minimum-voltage", "event: battery:minimum-voltage", "battery:minimum-voltage")
)
def test_event_identity_preserves_raw_and_canonicalizes_family_prefix(raw):
    from battery_msmr_assess import termination

    segment = {"termination": raw, "end_time_s": 12.345}
    assert termination(segment) == "battery:minimum-voltage"
    assert segment == {"termination": raw, "end_time_s": 12.345}
