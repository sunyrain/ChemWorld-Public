"""Preparation-only schema/source/API contracts, not the fixed science matrix."""

from copy import deepcopy

import numpy as np
import pytest
from battery_composite_impedance_assess import invariants, structural
from battery_composite_impedance_cases import case_data, route_plan
from battery_composite_impedance_oracle import analytic_spectrum
from battery_composite_impedance_traces import native_spectrum, source_spectrum
from battery_frequency_pybamm import frequency_from_simulation


@pytest.fixture(scope="module", params=("native", "reference"))
def trace(request):
    function = native_spectrum if request.param == "native" else source_spectrum
    return function(case_data(2), 8, frequencies=np.array([0.01]))


def test_route_denominator():
    assert len([r for i in range(1, 13) for r in route_plan(i)]) == 66
    assert sum(len(route_plan(i)) for i in range(1, 11)) == 54
    assert len(route_plan(11)) == 4
    assert len(route_plan(12)) == 8


def test_actual_native_source_shape_and_invariants(trace):
    results = structural(
        trace, case_data(2), trace["implementation"], 8, frequencies=np.array([0.01])
    )
    results += invariants(trace, case_data(2))
    assert all(row["passed"] for row in results), results


@pytest.mark.parametrize(
    "field", ("initial_state", "stationary_residual", "response_order", "points", "grid")
)
def test_missing_required_field_rejects(trace, field):
    damaged = deepcopy(trace)
    del damaged[field]
    assert not all(
        r["passed"]
        for r in structural(
            damaged, case_data(2), trace["implementation"], 8, frequencies=np.array([0.01])
        )
    )


@pytest.mark.parametrize(
    "damage", ("nan", "frequency", "identity", "raw-response", "raw-initial", "metadata")
)
def test_tampered_records_reject(trace, damage):
    damaged = deepcopy(trace)
    if damage == "nan":
        damaged["points"][0]["response_real"][0] = float("nan")
    elif damage == "frequency":
        damaged["points"][0]["frequency_Hz"] = 0.02
    elif damage == "identity":
        damaged["response_order"][1:5], damaged["response_order"][5:9] = (
            damaged["response_order"][5:9],
            damaged["response_order"][1:5],
        )
    elif damage == "raw-response":
        damaged["points"][0]["state_response_real"] = []
    elif damage == "raw-initial":
        damaged["initial_state"] = []
    elif damaged["implementation"] == "native":
        damaged["diagnostics"] = {}
    else:
        damaged["source_state_layout"] = []
    assert not all(
        r["passed"]
        for r in structural(
            damaged, case_data(2), trace["implementation"], 8, frequencies=np.array([0.01])
        )
    )


@pytest.mark.parametrize("kind", ("native", "reference"))
@pytest.mark.parametrize("reason", ("internal-current", "radial-gradient"))
def test_nonstationary_short_contract(kind, reason):
    function = native_spectrum if kind == "native" else source_spectrum
    result = function(case_data(2), 8, frequencies=np.array([0.01]), nonstationary=reason)
    assert result["rejected"] and not result["success"], result
    if kind == "native":
        assert "nonstationary" in result["exception"]
    else:
        assert result["failure"] == "reference_not_stationary"


@pytest.mark.parametrize("single", (False, True))
def test_same_material_short_source_native_build(single):
    for function in (native_spectrum, source_spectrum):
        record = function(case_data(1), 8, frequencies=np.array([0.01]), single=single)
        results = structural(
            record, case_data(1), record["implementation"], 8, single, np.array([0.01])
        )
        assert all(c["passed"] for c in results), results


def test_continuous_flat_constant():
    record = analytic_spectrum(case_data(8), frequencies=np.array([0.001, 0.01]))
    assert record["points"][0]["response_real"][0] == record["points"][1]["response_real"][0]
    assert record["points"][0]["response_imag"][0] == 0.0
    assert all(c["passed"] for c in invariants(record, case_data(8)))


@pytest.mark.parametrize("names", (None, ("Z_ohm",), ("Z_ohm", "Z_ohm"), ("Z_ohm", "")))
def test_nondefault_multi_output_requires_correct_names(names):
    with pytest.raises(ValueError):
        frequency_from_simulation(
            None,
            [0.01],
            family="contract",
            grid={},
            output_names=("Voltage [V]", "Current [A]"),
            response_names=names,
        )


def test_source_old_default_and_single_labels():
    from battery_composite_pybamm import build_reference
    from battery_frequency_pybamm import OUTPUTS

    for output_names in (OUTPUTS, ("Voltage [V]",)):
        result = frequency_from_simulation(
            build_reference(case_data(2), 8),
            [0.01],
            family="contract",
            grid={},
            output_names=output_names,
        )
        assert result["success"]
        assert len(result["response_order"]) == len(output_names)
        assert "initial_state" not in result
        assert len(result["points"][0]["response_real"]) == len(output_names)


@pytest.mark.parametrize(
    "damage",
    ("wrong-unit", "wrong-input-D", "wrong-units", "missing-QW", "false-initial", "bad-layout"),
)
def test_new_canonical_binding_checks(trace, damage):
    altered = deepcopy(trace)
    if damage == "wrong-unit":
        altered["binding"]["unit"] = "E03"
    elif damage == "wrong-input-D":
        altered["binding"]["inputs"]["phases"][0]["diffusivity_m2_s"] *= 2
    elif damage == "wrong-units":
        if altered["implementation"] == "native":
            altered["variables"][0]["unit"] = "kg"
        else:
            altered["definition"] = "wrong-sign"
    elif damage == "missing-QW" and altered["implementation"] == "native":
        indices = [
            i
            for i, v in enumerate(altered["variables"])
            if v["name"] in ("discharged_charge_C", "delivered_work_J")
        ]
        for key in ("variables", "initial_state", "stationary_residual", "residual_scales"):
            altered[key] = [v for i, v in enumerate(altered[key]) if i not in indices]
        for key in ("state_response_real", "state_response_imag"):
            altered["points"][0][key] = [
                v for i, v in enumerate(altered["points"][0][key]) if i not in indices
            ]
    elif damage == "false-initial":
        altered["initial_state"] = [v + 0.001 for v in altered["initial_state"]]
    elif altered["implementation"] == "native":
        del altered["state_before"]["volume_L"]
    else:
        del altered["source_state_layout"]
    assert not all(
        c["passed"]
        for c in structural(
            altered, case_data(2), trace["implementation"], 8, frequencies=np.array([0.01])
        )
    )


def test_negative_inventory_construction_is_recorded():
    from battery_composite_impedance_traces import refusal

    record = refusal(case_data(12), "negative-inventory")
    assert record["rejected"] and record["state_unchanged"]
    assert record["rejection_stage"] == "input-construction"


def test_nonfinite_archive_is_standard_explicit_and_atomic(tmp_path):
    import json

    from run_battery_composite_impedance_block import write_json

    path = tmp_path / "checkpoint.json"
    write_json(path, {"bad": [float("nan"), float("inf"), -float("inf")], "passed": False})
    restored = json.loads(path.read_text())
    assert restored["bad"] == [
        {"__nonfinite_float__": v} for v in ("NaN", "+Infinity", "-Infinity")
    ]
    assert restored["passed"] is False
    write_json(path, {"next": True})
    assert json.loads(path.read_text()) == {"next": True}


def test_legacy_helper_exact_full_objects():
    import json
    from pathlib import Path

    from battery_composite_impedance_legacy_frequency import frequency_from_simulation as legacy
    from battery_composite_pybamm import build_reference
    from battery_frequency_pybamm import OUTPUTS

    records = []
    for outputs in (OUTPUTS, ("Voltage [V]",)):
        args = {
            "family": "SPM composite legacy contract",
            "grid": {"radial_cells": 8},
            "output_names": outputs,
        }
        old = legacy(build_reference(case_data(2), 8), [0.001, 0.01], **args)
        new = frequency_from_simulation(build_reference(case_data(2), 8), [0.001, 0.01], **args)
        records.append(
            {
                "source_commit": "252a1b8d",
                "output_names": outputs,
                "equal_full_return_object": old == new,
                "legacy": old,
                "current": new,
            }
        )
    Path("/tmp/chemworld-composite-eis-legacy-helper-comparison.json").write_text(
        json.dumps(records, indent=2, allow_nan=False)
    )
    assert all(row["equal_full_return_object"] for row in records), records


def test_refusal_complete_and_tampered(monkeypatch):
    from battery_composite_impedance_assess import assess_unit
    from battery_composite_impedance_cases import GRIDS, REFUSALS
    from battery_composite_impedance_traces import refusal

    monkeypatch.setitem(GRIDS, "coarse", 8)  # preparation contract only; no fixed block execution
    routes = {}
    for reason in ("internal-current", "radial-gradient"):
        for kind, function in (("native", native_spectrum), ("reference", source_spectrum)):
            routes[f"{kind}_{reason}"] = function(
                case_data(11), 8, frequencies=np.array([0.01]), nonstationary=reason
            )
    assert assess_unit(11, case_data(11), routes)["passed"]
    for route in routes:
        for missing in ("initial_state", "stationary_residual", "binding"):
            damaged = deepcopy(routes)
            del damaged[route][missing]
            assert not assess_unit(11, case_data(11), damaged)["passed"]
    for damage in ("state", "infinite", "copy-reason"):
        damaged = deepcopy(routes)
        if damage == "state":
            del damaged["native_internal-current"]["state_before"]
            del damaged["native_internal-current"]["state_after"]
        elif damage == "infinite":
            damaged["reference_internal-current"]["stationary_residual"][0] = float("inf")
        else:
            damaged["reference_radial-gradient"] = damaged["reference_internal-current"]
        assert not assess_unit(11, case_data(11), damaged)["passed"]
    routes = {name: refusal(case_data(12), name) for name in REFUSALS}
    assert assess_unit(12, case_data(12), routes)["passed"], routes
    for name in REFUSALS:
        damaged = deepcopy(routes)
        del damaged[name]["state_before"]
        del damaged[name]["state_after"]
        assert not assess_unit(12, case_data(12), damaged)["passed"]
