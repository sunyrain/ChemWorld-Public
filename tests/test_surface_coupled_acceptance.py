"""Actual shared-apparatus acceptance beyond the independent numeric matrices."""

import runpy
from collections import Counter
from pathlib import Path

import pytest

from chemworld.eval.surface_acceptance import acceptance_cases, compare_thermal, summarize_cases


def test_fixed_coverage_has_all_prespecified_units_once():
    cases = acceptance_cases()
    assert Counter(c["block"] for c in cases) == {"rate": 36, "isothermal": 27, "thermal": 12}
    assert len({c["case_id"] for c in cases}) == 75


@pytest.mark.parametrize(
    "surface,bulk", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("mobile,terminal", [(False, False), (True, True)])
def test_one_example_operates_four_chemical_configurations_and_both_carrier_types(
    surface, bulk, mobile, terminal
):
    example = runpy.run_path(str(Path(__file__).parents[1] / "examples/native_surface_research.py"))
    result = example["run_example"](
        include_surface=surface, include_bulk=bulk, mobile=mobile, terminal_assay=terminal
    )
    assert result["ended"]
    assert result["physical_time_s"] == 36
    assert result["analysis_time_s"] == (5 if terminal else 0)
    assert all(step["transaction"] == "committed" for step in result["steps"])
    assert all(
        step["measurement_status"] == "valid" for step in result["steps"] if step["observation"]
    )
    assert sum(step["ended"] for step in result["steps"]) == 1


def test_thermal_reference_reports_actual_runtime_endpoint_and_convergence():
    case = next(c for c in acceptance_cases() if c["case_id"] == "thermal-12")
    report = compare_thermal(case)
    assert report["passed"], report
    assert report["actual_duration_s"] <= 600
    assert report["max_normalized_convergence_error"] <= 1e-6
    assert all(row["passed"] for row in report["balances"])


def test_unavailable_reference_is_an_uncompleted_unit_not_a_successful_skip():
    case = acceptance_cases()[0]
    result = summarize_cases([case], [{"case": case, "passed": False, "status": "unavailable"}])
    assert result["rate"]["expected"] == result["rate"]["unavailable"] == 1
    assert result["rate"]["completed"] == result["rate"]["passed"] == result["skipped"] == 0
    assert not result["all_passed"]
