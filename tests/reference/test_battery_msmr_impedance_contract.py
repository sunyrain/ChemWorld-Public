"""Short preparation contracts, deliberately separate from the fixed science block."""

from copy import deepcopy

import numpy as np
import pytest
from battery_msmr_impedance_assess import (
    assess_unit,
    initial_checks,
    invariants,
    nonstationary_checks,
    structural,
)
from battery_msmr_impedance_cases import GRIDS, REFUSALS, case_data, route_plan
from battery_msmr_impedance_oracle import analytic_spectrum
from battery_msmr_impedance_traces import native_spectrum, refusal, source_spectrum

FREQUENCIES = np.array([0.001, 0.01])


@pytest.fixture(scope="module", params=("native", "reference"))
def trace(request):
    function = native_spectrum if request.param == "native" else source_spectrum
    return function(case_data(1), 8, frequencies=FREQUENCIES)


def checked(record):
    return structural(record, case_data(1), record["implementation"], 8, FREQUENCIES)


def test_route_denominator():
    assert sum(len(route_plan(i)) for i in range(1, 11)) == 50
    assert sum(len(route_plan(i)) for i in range(1, 9)) == 40
    assert len(route_plan(9)) == 2
    assert len(route_plan(10)) == 8
    assert 40 * 17 == 680


def test_actual_short_native_source_contract(trace):
    result = checked(trace) + invariants(trace, case_data(1))
    assert all(c["passed"] for c in result), result


@pytest.mark.parametrize(
    "field",
    (
        "initial_state",
        "stationary_residual",
        "binding",
        "response_order",
        "response_units",
        "points",
        "grid",
    ),
)
def test_missing_fields_fail_closed(trace, field):
    damaged = deepcopy(trace)
    del damaged[field]
    assert not all(c["passed"] for c in checked(damaged))


@pytest.mark.parametrize(
    "damage",
    (
        "wrong-unit",
        "wrong-D",
        "nan",
        "frequency",
        "raw-state",
        "raw-initial",
        "units",
        "missing-QW-or-abs",
        "false-work-or-Q",
        "diagnostics-or-feedback",
        "source-scale-or-native-state",
    ),
)
def test_tampered_trace_fails(trace, damage):
    bad = deepcopy(trace)
    if damage == "wrong-unit":
        bad["binding"]["unit"] = "M03"
    elif damage == "wrong-D":
        bad["binding"]["inputs"]["electrodes"][0]["diffusivity_m2_s"] *= 2
    elif damage == "nan":
        bad["points"][0]["response_real"][0] = float("nan")
    elif damage == "frequency":
        bad["points"][0]["frequency_Hz"] = 0.2
    elif damage == "raw-state":
        bad["points"][0]["state_response_real"] = []
    elif damage == "raw-initial":
        bad["initial_state"][0] += 0.1
    elif damage == "units":
        bad["response_units"][0] = "kg"
    elif damage == "missing-QW-or-abs":
        if bad["implementation"] == "native":
            bad["variables"] = [v for v in bad["variables"] if v["name"] != "delivered_work_J"]
        else:
            bad["source_state_layout"] = [
                v for v in bad["source_state_layout"] if v["name"] != "Throughput capacity [A.h]"
            ]
    elif damage == "false-work-or-Q":
        if bad["implementation"] == "native":
            bad["points"][0]["response_real"][3] += 1
        else:
            bad["points"][0]["response_real"][6] += 1
    elif damage == "diagnostics-or-feedback":
        if bad["implementation"] == "native":
            bad["diagnostics"] = {}
        else:
            bad["source_linearization"]["maximum_observer_feedback_rhs"] = float("inf")
    elif bad["implementation"] == "native":
        del bad["state_before"]["volume_L"]
    else:
        bad["source_state_layout"][0]["scale"] *= 2
    assert not all(c["passed"] for c in checked(bad)), damage


@pytest.mark.parametrize("kind", ("native", "reference"))
def test_gradient_refusal_raw_record(kind):
    function = native_spectrum if kind == "native" else source_spectrum
    trace = function(case_data(9), 8, frequencies=FREQUENCIES, gradient=True)
    checks = initial_checks(trace, case_data(9), kind, 8, gradient=True)
    assert trace["rejected"] and not trace["success"] and trace["points"] == []
    assert all(c["passed"] for c in checks), checks
    assert all(c["passed"] for c in nonstationary_checks(trace, case_data(9), kind, 8, FREQUENCIES))
    if kind == "native":
        assert "nonstationary" in trace["exception"]
    else:
        assert trace["failure"] == "reference_not_stationary"
    for field in ("initial_state", "stationary_residual"):
        bad = deepcopy(trace)
        del bad[field]
        assert not all(c["passed"] for c in initial_checks(bad, case_data(9), kind, 8, True))
    for field in ("frequencies_Hz", "response_order", "response_units"):
        bad = deepcopy(trace)
        del bad[field]
        assert not all(
            c["passed"] for c in nonstationary_checks(bad, case_data(9), kind, 8, FREQUENCIES)
        )
    if kind == "native":
        bad = deepcopy(trace)
        bad["residual_scales"][0] *= 2
        assert not all(
            c["passed"] for c in nonstationary_checks(bad, case_data(9), kind, 8, FREQUENCIES)
        )


def test_native_stationary_residual_scales_bind_actual_declaration():
    trace = native_spectrum(case_data(1), 8, frequencies=FREQUENCIES)
    trace["residual_scales"][0] *= 2
    assert not all(c["passed"] for c in checked(trace))


def test_analytic_conservation_short():
    trace = analytic_spectrum(case_data(1), frequencies=FREQUENCIES)
    checks = structural(trace, case_data(1), "analytic", None, FREQUENCIES) + invariants(
        trace, case_data(1)
    )
    assert all(c["passed"] for c in checks), checks


def test_all_refusals_and_missing_snapshots(monkeypatch):
    monkeypatch.setitem(GRIDS, "coarse", 32)  # Contract only; not a fixed block execution.
    routes = {name: refusal(case_data(10), name, 32) for name in REFUSALS}
    assert assess_unit(10, case_data(10), routes)["passed"], routes
    for name in REFUSALS:
        for field in ("state_before", "state_after", "binding", "exception_type"):
            bad = deepcopy(routes)
            del bad[name][field]
            assert not assess_unit(10, case_data(10), bad)["passed"]


def test_source_observer_and_original_model_retained():
    record = source_spectrum(case_data(1), 8, frequencies=np.array([0.01]))
    names = [v["name"] for v in record["source_state_layout"]]
    assert "Throughput capacity [A.h]" in names
    assert record["source_linearization"]["maximum_observer_feedback_rhs"] == 0.0
    assert record["source_linearization"]["maximum_observer_feedback_output"] == 0.0
    assert "Throughput energy [W.h]" in names
    assert "Discharge energy [W.h]" in names
    assert "work_J_A" in record["response_order"]
    bad = deepcopy(record)
    bad["source_retained_model"]["rhs"].remove("Throughput capacity [A.h]")
    assert not all(c["passed"] for c in structural(bad, case_data(1), "reference", 8, [0.01]))


def test_nonfinite_checkpoint_is_lossless_tagged_json(tmp_path):
    import json

    from run_battery_msmr_impedance_block import write_json

    path = tmp_path / "checkpoint.json"
    write_json(path, {"raw": [float("nan"), float("inf"), -float("inf")], "passed": False})
    result = json.loads(path.read_text())
    assert result["raw"] == [{"__nonfinite_float__": v} for v in ("NaN", "+Infinity", "-Infinity")]
    assert result["passed"] is False
    write_json(path, {"next": True})
    assert json.loads(path.read_text()) == {"next": True}


def test_actual_source_fixed_electrolyte_reference_mapping():
    data = case_data(8)
    record = source_spectrum(data, 8, frequencies=FREQUENCIES)
    checks = structural(record, data, "reference", 8, FREQUENCIES) + invariants(record, data)
    assert all(c["passed"] for c in checks), checks
    assert all(row["factor"] != 1 for row in record["source_exchange_reference_mapping"])
    bad = deepcopy(record)
    bad["source_exchange_reference_mapping"][0]["source_jref_A_m2"] *= 2
    assert not all(c["passed"] for c in structural(bad, data, "reference", 8, FREQUENCIES))
