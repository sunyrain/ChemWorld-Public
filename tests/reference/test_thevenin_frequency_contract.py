"""Fixed denominator, explicit released construction, strict complex-error and failure rules."""

import json
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import run_thevenin_frequency_block as runner
from thevenin_frequency_cases import FREQUENCIES, case_data, native_case, response_order
from thevenin_frequency_traces import reference_model


def test_frozen_matrix_and_scientific_edge_conditions():
    assert [case_data(i)["rc_count"] for i in range(1, 13)] == [0, 1, 2, 4, 1, 2, 1, 1, 1, 0, 4, 2]
    assert FREQUENCIES.tolist() == [10.**(-4+.5*k) for k in range(13)]
    assert case_data(5)["feedback"] == "affine"
    assert case_data(9)["feedback"] == "current_cusp"
    assert case_data(10)["r0_ohm"] == 0.
    assert case_data(11)["ocv_nodes"] == [[0., 3.6], [1., 3.6]]
    hot = case_data(12)
    assert hot["temperature_K"] == 310. and hot["maximum_current_A"] == .1
    assert hot["cell_jig_conductance_W_K"] == hot["jig_ambient_conductance_W_K"] == 0.
    fourth = native_case(4)
    constants = [r.value({})*c.value({}) for r, c in fourth.rc_elements]
    assert len(set(constants)) == 4
    with pytest.raises(ValueError, match="exactly twelve"):
        native_case(13)


@pytest.mark.parametrize("index", range(1, 13))
def test_released_physical_states_have_no_silent_receipt_or_material_defaults(index):
    pybamm = pytest.importorskip("pybamm")
    assert pybamm.__version__ == "25.4.2"
    simulation, outputs = reference_model(index)
    data = case_data(index)
    model, parameters = simulation.model, simulation.parameter_values
    assert len(model.rhs) == data["rc_count"]+3 and not model.algebraic
    assert not any(k.startswith("Qualification integral: ") for k in model.variables)
    assert len(outputs) == len(response_order(index)) == 4+data["rc_count"]
    assert parameters["Current function [A]"].name == "Current function [A]"
    assert parameters["Initial SoC"] == data["soc"]
    assert parameters["Initial temperature [K]"] == data["temperature_K"]
    assert parameters["Cell capacity [A.h]"] == 100/3600.
    parameters.process_model(model, inplace=False).check_well_posedness()


def trace(index=2):
    return {"success": True, "response_order": response_order(index),
        "input_state_unchanged": True, "maximum_stationary_rhs": 0.,
        "maximum_stationary_algebraic": 0.,
        "diagnostics": {"physical_time_advanced_s": 0., "maximum_scaled_stationary_residual": 0.},
        "points": [{"frequency_Hz": float(f), "success": True,
            "response_real": [0.]*len(response_order(index)),
            "response_imag": [0.]*len(response_order(index)),
            "relative_linear_residual": 0.} for f in FREQUENCIES]}


def record():
    return {name: trace() for name in ("native", "reference", "closed")}


def test_all_three_comparisons_and_original_absolute_bounds_are_required():
    data = record()
    runner.assess(data, 2)
    assert data["passed"]
    assert data["atol"] == [2e-6, 1e-7, 1e-6, 1e-6, 1e-7]
    assert data["rtol"] == [2e-6, 1e-6, 1e-5, 1e-5, 1e-6]
    for source in ("native", "reference", "closed"):
        for column, tolerance in enumerate(data["atol"]):
            changed = record()
            changed[source]["points"][4]["response_imag"][column] = tolerance*1.1
            runner.assess(changed, 2)
            assert not changed["passed"]
            assert not all(changed["checks"][name] for name in (
                "native_reference", "native_closed", "reference_closed"))


@pytest.mark.parametrize("defect", ("missing", "duplicate", "failed", "nonfinite", "wrong_units",
                                     "advanced", "native_stationary", "reference_stationary",
                                     "linear"))
def test_incomplete_or_invalid_scientific_trace_cannot_be_a_pass(defect):
    data = record()
    if defect == "missing":
        data["reference"]["points"].pop()
    elif defect == "duplicate":
        data["reference"]["points"][1] = deepcopy(data["reference"]["points"][0])
    elif defect == "failed":
        data["native"]["points"][4]["success"] = False
    elif defect == "nonfinite":
        data["native"]["points"][4]["response_real"][0] = np.nan
    elif defect == "wrong_units":
        data["reference"]["response_order"][2] = "Celsius_not_K_per_A"
    elif defect == "advanced":
        data["native"]["input_state_unchanged"] = False
    elif defect == "native_stationary":
        data["native"]["diagnostics"]["maximum_scaled_stationary_residual"] = 2e-9
    elif defect == "reference_stationary":
        data["reference"]["maximum_stationary_rhs"] = 2e-10
    else:
        data["native"]["points"][4]["relative_linear_residual"] = 2e-10
    runner.assess(data, 2)
    assert not data["passed"]


def test_all_routes_attempted_and_nonfinite_originals_retained_as_failed_records(monkeypatch):
    calls, checkpoints = [], []

    def fail(index):
        calls.append("native")
        raise RuntimeError("retained native fixture failure")

    def reference(index):
        calls.append("reference")
        return trace(index)

    def closed(index):
        calls.append("closed")
        return trace(index)

    monkeypatch.setattr(runner, "native_trace", fail)
    monkeypatch.setattr(runner, "reference_trace", reference)
    monkeypatch.setattr(runner, "closed_trace", closed)
    data = {}
    runner.run_unit(data, 2, lambda _: None, lambda: checkpoints.append(True))
    assert calls == ["native", "reference", "closed"] and len(checkpoints) == 3
    assert not data["passed"] and "retained native fixture failure" in data["native"]["exception"]
    invalid = runner.capture(lambda: {"original": float("nan")})
    assert "invalid numerical trace" in invalid["exception"] and "nan" in invalid["raw_repr"]
    json.dumps(invalid, allow_nan=False)


def test_cli_runs_all_twelve_units_and_never_replaces_the_original_output(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pybamm", SimpleNamespace(__version__="25.4.2"))
    output = tmp_path / "contract-fixture-not-scientific-results.json"
    monkeypatch.setattr(sys, "argv", ["thevenin-frequency", str(output)])

    def run(data, index, progress, checkpoint):
        data.update(passed=index != 4, fixture_only=True)
        checkpoint()

    monkeypatch.setattr(runner, "run_unit", run)
    assert runner.main() == 1
    saved = json.loads(output.read_text())
    assert saved["denominator"] == len(saved["units"]) == 12
    assert saved["passed"] == 11 and "active_unit" not in saved
    with pytest.raises(FileExistsError):
        runner.main()
    assert saved == json.loads(output.read_text())
