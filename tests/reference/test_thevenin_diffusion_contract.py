"""Scientific scope, independently attributed reference and failure-retention contracts."""

import json
import sys
from itertools import pairwise

import numpy as np
import pytest
import run_thevenin_diffusion_block as runner
from thevenin_diffusion_cases import (
    FREQUENCIES,
    GRIDS,
    case_data,
    frequency_names,
    native_case,
    time_names,
)
from thevenin_diffusion_traces import analytic_frequency, native_time, reference_model


def test_frozen_scope_contains_physical_edges_and_both_frequency_models():
    assert GRIDS == (32, 64, 128)
    assert FREQUENCIES.tolist() == [10.**(-4+k/3) for k in range(13)]
    assert [case_data(i)["kind"] for i in range(1, 13)] == ["time"]*10+["frequency"]*2
    assert case_data(2)["segments"][0]["value"] < 0
    assert [case_data(i)["tau_s"] for i in (1, 3, 4)] == [30., 3., 300.]
    assert case_data(7)["segments"][0]["unit"] == "V"
    assert case_data(8)["expected_surface_event"] and case_data(8)["initial_soc"] == .08
    assert len(case_data(9)["ocv_nodes"]) == 3
    cycle = case_data(10)
    assert cycle["r0_ohm"] == 0. and cycle["entropy_V_K"] == 0.
    assert sum(p["duration_s"]*p["value"] for p in cycle["segments"]) == 0.
    assert not case_data(11)["rc_resistances_ohm"]
    assert len(case_data(12)["rc_resistances_ohm"]) == 2
    assert case_data(12)["entropy_V_K"] != 0
    with pytest.raises(ValueError, match="exactly twelve"):
        case_data(13)
    with pytest.raises(ValueError, match="three fixed meshes"):
        native_case(1, 16)


@pytest.mark.parametrize("index", range(1, 13))
def test_both_reference_variants_build_without_material_defaults(index):
    pybamm = pytest.importorskip("pybamm")
    simulations = {variant: reference_model(index, 32, variant, frequency=index >= 11)
                   for variant in ("released", "corrected")}
    released = simulations["released"].model
    corrected = simulations["corrected"].model
    assert type(released.submodels["Diffusion"]) is (
        pybamm.equivalent_circuit_elements.DiffusionElement)
    assert type(released.submodels["Thermal"]) is pybamm.equivalent_circuit_elements.ThermalSubModel
    assert type(corrected.submodels["Diffusion"]).__name__ == "CorrectedDiffusion"
    assert type(corrected.submodels["Thermal"]).__name__ == "CorrectedThermal"
    # All remaining upstream submodel types are untouched by the local correction.
    for key in released.submodels:
        if key not in {"Diffusion", "Thermal"}:
            assert type(released.submodels[key]) is type(corrected.submodels[key])
    for simulation in simulations.values():
        parameters = simulation.parameter_values
        assert parameters["Cell capacity [A.h]"] == 100/3600.
        assert parameters["Diffusion time constant [s]"] == case_data(index)["tau_s"]
        assert parameters["Initial SoC"] == case_data(index)["initial_soc"]
        simulation.build()
        assert simulation.built_model.concatenated_rhs.size == (
            32+3+len(case_data(index)["rc_resistances_ohm"])+(0 if index >= 11 else 4))
        diffusion = simulation.built_model.variables["Surface SoC"]
        assert diffusion.size == 1


def test_continuous_diffusion_has_passive_low_frequency_increment():
    trace = analytic_frequency(11)
    values = runner.frequency_values(trace)
    data = case_data(11)
    diffusion = values[:, 0]-data["r0_ohm"]+values[:, 1]
    assert np.all(diffusion.real > 0)
    assert diffusion[0].real == pytest.approx(data["tau_s"]/(3*data["capacity_C"]), rel=1e-4)
    assert np.all(values[:, 2].real < 0)
    assert np.all(values[:, 2].imag > 0)
    assert runner.valid_frequency(trace, 11)
    trace["points"][3]["response_real"][0] = float("nan")
    assert not runner.valid_frequency(trace, 11)


def fixture_time(index=1, cells=32, variant="native"):
    data = case_data(index)
    rows = np.zeros((21, len(time_names(index))))
    rows[:, 2:4] = data["initial_soc"]
    rows[:, 4:6] = data["initial_temperature_K"]
    return {"variant": variant, "cells": cells, "response_order": time_names(index),
        "maximum_charge_balance_C": 0., "maximum_profile_mean_error": 0.,
        "maximum_node_heat_balance_J": 0.,
        "segments": [{"times_s": np.linspace(0., 20., 21).tolist(), "rows": rows.tolist(),
            "profiles": np.full((21, cells), data["initial_soc"]).tolist(),
            "termination": None if variant == "native" else "final time",
            "outcome": "success", "end_time_s": 20.}]}


def test_profile_and_missing_trace_cannot_be_hidden_by_terminal_agreement():
    left, right = fixture_time(), fixture_time(variant="corrected")
    assert runner.valid_time(left, 1, 32)
    assert runner.time_comparison(left, right, 1)["passed"]
    right["segments"][0]["profiles"][3][7] += 1e-3
    assert not runner.time_comparison(left, right, 1)["passed"]
    right["segments"][0]["rows"].pop()
    assert not runner.valid_time(right, 1, 32)


def test_all_nine_routes_attempted_when_native_and_released_fail(monkeypatch):
    calls = []

    def native(index, cells):
        calls.append((cells, "native"))
        raise RuntimeError("retained fixture-only native failure")

    def reference(index, cells, variant):
        calls.append((cells, variant))
        if variant == "released":
            raise RuntimeError("retained fixture-only release discrepancy")
        return fixture_time(index, cells, variant)

    monkeypatch.setattr(runner, "native_time", native)
    monkeypatch.setattr(runner, "reference_time", reference)
    record = {}
    runner.run_unit(record, 1, lambda _: None, lambda: None)
    assert calls == [(n, route) for n in GRIDS for route in ("native", "released", "corrected")]
    assert not record["passed"]
    assert all("retained fixture-only" in record["grids"][str(n)]["native"]["exception"]
               for n in GRIDS)
    json.dumps(record, allow_nan=False)


def test_cli_preserves_full_denominator_and_original_output(tmp_path, monkeypatch):
    pytest.importorskip("pybamm")
    output = tmp_path / "fixture-only-not-scientific-results.json"
    monkeypatch.setattr(sys, "argv", ["diffusion-block", str(output)])
    monkeypatch.setattr(runner, "run_unit", lambda row, index, progress, checkpoint:
                        row.update(passed=index != 8, fixture_only=True))
    assert runner.main() == 1
    original = output.read_bytes()
    data = json.loads(original)
    assert data["denominator"] == len(data["units"]) == 12 and data["passed"] == 11
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == original


def test_frequency_output_contract_includes_surface_and_both_thermal_states():
    assert frequency_names(11) == ["Z_ohm", "soc_per_A", "surface_soc_per_A",
                                    "cell_K_per_A", "jig_K_per_A"]
    assert frequency_names(12)[-2:] == ["rc0_V_per_A", "rc1_V_per_A"]


def test_segment_relative_api_preserves_reversal_history_and_absolute_clock():
    trace = native_time(6, 32)
    assert runner.valid_time(trace, 6, 32)
    parts = trace["segments"]
    assert [p["times_s"][0] for p in parts] == [0., 10., 30., 40.]
    assert [p["times_s"][-1] for p in parts] == [10., 30., 40., 70.]
    assert [p["rows"][0][1] for p in parts] == pytest.approx([.2, 0., -.2, 0.])
    # Inventory and accumulated charge remain continuous across source switches.
    for before, after in pairwise(parts):
        assert after["rows"][0][2] == pytest.approx(before["rows"][-1][2], abs=1e-12)
        assert after["rows"][0][6] == pytest.approx(before["rows"][-1][6], abs=1e-12)
    assert trace["final_net_charge_C"] == pytest.approx(0., abs=1e-6)
