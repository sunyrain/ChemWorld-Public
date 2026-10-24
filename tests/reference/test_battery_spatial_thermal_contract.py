"""Mapping contracts before the fixed x-full qualification block is released."""

from copy import deepcopy

import numpy as np
import pytest
from battery_spatial_thermal_cases import (
    GRIDS,
    case_data,
    initial_temperatures,
    matrix_temperature,
    thermal_geometry,
)
from battery_spatial_thermal_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 12))
def test_all_fixed_cards_build_actual_source(index):
    import pybamm

    data = case_data(index)
    simulation = build_reference(data, *GRIDS["reference"][0])
    assert simulation.model.options["thermal"] == "x-full"
    assert type(simulation.model.submodels["thermal"]).__name__ == "OneDimensionalX"
    assert simulation.model.options["heat of mixing"] == "true"
    simulation.build()
    assert (
        sum(
            value.size
            for var, value in simulation.built_model.rhs.items()
            if "temperature" in str(var).lower()
        )
        == 38
    )
    for name in ("negative", "positive"):
        electrode = next(e for e in data["electrodes"] if e["name"] == name)
        parameter = getattr(simulation.model.param, name[0]).prim
        for theta in (0.02, 0.35, 0.6, 0.98):
            actual = simulation.parameter_values.process_symbol(
                parameter.U(pybamm.Scalar(theta), pybamm.Scalar(310.0))
            ).evaluate()
            assert actual == pytest.approx(electrode["ocv_V"], abs=1e-13)


def test_source_actual_local_temperature_and_zero_undeclared_heat():
    data = deepcopy(case_data(3))
    data["waveform"] = ((0.2, 0.8, "A"),)
    data["samples_per_segment"] = 3
    result = reference_time(data, GRIDS["reference"][0])
    segment = result["segments"][0]
    values = segment["values"]
    assert segment["termination"] == "final time"
    assert np.asarray(segment["temperatures_K"]).shape == (3, 38)
    assert values["generated_W"][-1] > 0
    for key in (
        "reversible_W",
        "mixing_W",
        "ohmic_W",
        "negative_CC_ohmic_W_m3",
        "positive_CC_ohmic_W_m3",
    ):
        assert np.max(np.abs(values[key])) < 1e-10, key
    assert values["maximum_K"][-1] > values["minimum_K"][-1]


def test_independent_thermal_matrix_conserves_c_weighted_energy():
    data, cells = case_data(1), 12
    geometry = thermal_geometry(data, cells)
    capacity = geometry["capacities_J_K"]
    temperatures = matrix_temperature(data, cells, (0.0, 0.1, 1.0, 60.0))
    expected = initial_temperatures(data, cells)
    assert temperatures[0] == pytest.approx(expected, abs=1e-12)
    assert temperatures @ capacity == pytest.approx(np.full(4, expected @ capacity), abs=1e-8)
    c_mean = expected @ capacity / capacity.sum()
    v = geometry["volumes_m3"]
    assert abs(c_mean - expected @ v / v.sum()) > 0.01


@pytest.mark.parametrize("index", range(1, 12))
def test_all_native_cards_have_real_independent_thermal_state(index):
    from battery_spatial_thermal_cases import native_spec

    from chemworld.physchem.battery_spatial_thermal import spatial_thermal_fields

    data = case_data(index)
    spec = native_spec(data, GRIDS["native"][0])
    state = spec.initial_state()
    fields = spatial_thermal_fields(spec, state, current_A=0.0)
    assert fields["temperature_K"] == pytest.approx(initial_temperatures(data, 6), abs=1e-12)
    assert len(spec.thermal_coordinate_keys) == 19
    assert fields["heat_capacity_mean_temperature_K"] == pytest.approx(state.temperature_K)
    assert fields["temperature_K"] @ thermal_geometry(data, 6)["capacities_J_K"] == pytest.approx(
        state.temperature_K * sum(thermal_geometry(data, 6)["capacities_J_K"]), abs=1e-10
    )


def test_native_trace_carries_original_thermal_receipt():
    from battery_spatial_thermal_traces import history_metrics, native_time

    data = deepcopy(case_data(3))
    data["waveform"], data["samples_per_segment"] = ((0.2, 0.8, "A"),), 3
    trace = native_time(data, GRIDS["native"][0])
    assert np.asarray(trace["segments"][0]["temperatures_K"]).shape == (3, 20)
    metrics = history_metrics(trace, data)
    assert metrics["energy_closure_maximum_tolerance_ratio"] < 1
    assert metrics["C_mean_reconstruction_K"] < 1e-10
    assert metrics["volume_mean_reconstruction_K"] < 1e-10
    assert trace["segments"][0]["values"]["loss_J"][-1] > 0


def test_fixed_route_failures_are_all_preserved(monkeypatch):
    import run_battery_spatial_thermal_block as runner

    def fail(*_args):
        raise RuntimeError("deliberate contract failure, no physics executed")

    for function in ("native_time", "reference_time", "refusal"):
        monkeypatch.setattr(runner, function, fail)
    count = 0
    for index in range(1, 13):
        item = {"id": f"XT{index:02d}", "inputs": case_data(index)}
        runner.run_unit(item, lambda _: None, lambda: None)
        assert not item["passed"]
        count += len(item["routes"])
        assert all("exception" in route for route in item["routes"].values())
    assert count == 52


def test_interface_errors_are_not_valid_domain_refusals():
    from battery_spatial_thermal_assess import valid_refusal

    assert not valid_refusal(
        {
            "rejected": True,
            "input_state_unchanged": True,
            "exception_type": "AttributeError",
            "exception": "missing API",
        }
    )
