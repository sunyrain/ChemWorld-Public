"""Fixed coverage and actual released-submodel construction, without scientific runs."""

import numpy as np
import pytest
from thevenin_control_cases import (
    UNITS,
    case_data,
    dynamic,
    native_program,
    native_spec,
    response_atol,
    response_names,
)
from thevenin_control_pybamm import build_reference, output_names, reference_inputs
from thevenin_control_traces import independent_rejection_fact


def test_fixed_denominators_and_measurements():
    cases = [case_data(i, j) for i in range(1, 17) for j in range(len(UNITS[i-1]))]
    assert len(UNITS) == 16 and len(cases) == 18
    assert sum(d["expected"] == "trajectory" for d in cases) == 14
    assert sum(d["expected"] != "trajectory" for d in cases) == 4
    assert sum(4 if d["expected"] == "trajectory" else 2 for d in cases) == 64
    for data in cases:
        assert data["samples_per_segment"] == 41
        assert len(response_names(data)) == len(response_atol(data)) == len(output_names(data))
        assert ("controller_current_A" in response_names(data)) is dynamic(data)


@pytest.mark.parametrize("index", range(1, 17))
def test_native_and_released_submodel_construction(index):
    pybamm = pytest.importorskip("pybamm")
    for j in range(len(UNITS[index-1])):
        data = case_data(index, j)
        spec = native_spec(data)
        assert spec.control.native_parameters["antiwindup_time_s"] == data["capacity_C"]/3600.
        for part in data["segments"]:
            program = native_program(data, part)
            assert program.control_mode == data["mode"]
            assert program.value_at(0) == part["target"]
        simulation = build_reference(data, False)
        submodel = simulation.model.submodels["external circuit"]
        expected_class = {"power": pybamm.external_circuit.PowerFunctionControl,
            "resistance": pybamm.external_circuit.ResistanceFunctionControl,
            "differential_power": pybamm.external_circuit.PowerFunctionControl,
            "differential_resistance": pybamm.external_circuit.ResistanceFunctionControl,
            "cccv": pybamm.external_circuit.CCCVFunctionControl}[data["mode"]]
        assert type(submodel) is expected_class
        assert submodel.control == ("differential with max" if data["mode"] == "cccv"
                                    else "differential" if dynamic(data) else "algebraic")
        simulation.build()
        model = simulation.built_model
        inputs = reference_inputs(data, data["segments"][0])
        initial = np.asarray(model.concatenated_initial_conditions.evaluate(inputs=inputs)).ravel()
        q = model.variables["Current variable [A]"].evaluate(t=0., y=initial, inputs=inputs)
        assert float(np.asarray(q).item()) == pytest.approx(data["initial_current_A"], abs=1e-14)
        assert np.isfinite(initial).all()


@pytest.mark.parametrize("index,variant", [(4, 0), (5, 0), (7, 1), (12, 0)])
def test_independent_rejection_facts(index, variant):
    assert independent_rejection_fact(case_data(index, variant))["passed"]


def test_high_root_source_capability_and_real_antiwindup_state():
    data = case_data(3, 1)
    u, r, p = data["ocv_intercept_V"]+data["initial_soc"], data["r0_ohm"], 3.
    high_root = (u+np.sqrt(u*u-4*r*p))/(2*r)
    assert data["branch_A"][0] < high_root < data["branch_A"][1] < data["maximum_current_A"]
    data = case_data(14)
    assert data["initial_current_A"] != data["current_limit_A"]
    assert abs(data["initial_current_A"]-data["current_limit_A"]) >= .5


def test_failed_numeric_path_does_not_reduce_denominator(monkeypatch):
    import run_thevenin_control_block as runner

    def failed(*_args):
        raise RuntimeError("deliberate failed route")

    monkeypatch.setattr(runner, "native_trace", failed)
    monkeypatch.setattr(runner, "reference_trace", failed)
    record = {"passed": False}
    runner.run_unit(record, 3, lambda _: None, lambda: None)
    assert not record["passed"]
    assert len(record["variants"]) == 2
    assert all(len(v["routes"]) == 4 for v in record["variants"])
    assert all("deliberate failed route" in route["exception"]
               for item in record["variants"] for route in item["routes"].values())


def test_unrelated_exception_cannot_pass_native_refusal():
    from run_thevenin_control_block import assess_refusal

    item = {"routes": {"native_probe": {"rejected": True, "exception_type": "TypeError",
                "exception": "branch implementation mistake", "input_state_unchanged": True},
            "reference_probe": {"exception": "the reference also failed"}}}
    assess_refusal(item, case_data(4))
    assert not item["passed"]
    assert not item["checks"]["explicit_native_domain_refusal"]
