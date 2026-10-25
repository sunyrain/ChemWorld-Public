"""Synthetic ingestion/comparison contracts; no qualification integration."""

import gc
import json
import weakref
from copy import deepcopy

import battery_collector_spm_refinement_assess as assessment
import numpy as np
import pytest
from battery_collector_spm_refinement_cases import (
    EDGES,
    FINE_EDGES,
    IMPLEMENTATIONS,
    LAYOUT,
    MESHES,
    UNITS,
    history_plan,
    unit_data,
)
from test_battery_collector_spm_supplement_assess import synthetic_native


def normalized_history(data, grid):
    """Arrays for testing comparison orchestration, not a physical solution."""
    history, start, charge, work = [], 0.0, 0.0, 0.0
    for duration, target, mode in data["waveform"]:
        times = np.linspace(start, start + duration, data["samples_per_segment"])
        current = target if mode == "A" else 0.03
        voltage = target if mode == "V" else 3.8
        values = np.column_stack(
            (
                np.full(len(times), voltage),
                np.full(len(times), current),
                charge + current * (times - start),
                work + voltage * current * (times - start),
            )
        )
        q = np.full((len(times), grid["z"]), current / grid["z"])
        history.append(
            {
                "times": times,
                "values": values,
                "grid": dict(grid),
                "profiles": {
                    name: {
                        "total": 0.01 + sign * values[:, 2] / 96485.33212,
                        "q": -sign * q,
                        "theta": np.zeros((len(times), grid["z"], grid["r"])),
                        "amounts": np.zeros((len(times), grid["z"], grid["r"])),
                    }
                    for name, sign in (("negative", -1), ("positive", 1))
                },
            }
        )
        charge, work = values[-1, 2:]
        start += duration
    return history


def stub_trace(data, mesh, implementation):
    return {
        "unit": data["id"],
        "inputs": deepcopy(data),
        "grid": dict(MESHES[mesh]),
        "implementation": implementation,
        "layout": dict(LAYOUT),
        "normalized": normalized_history(data, MESHES[mesh]),
    }


@pytest.fixture
def delegated_validator(monkeypatch):
    """Spy the validator boundary; separate tests below call the real validator."""
    calls = []

    def validate(book, label, trace, data, grid, layout, source=False, uniform=False):
        calls.append((deepcopy(data), dict(grid), dict(layout), source, uniform))
        for key, expected in (
            ("unit", data["id"]),
            ("inputs", data),
            ("grid", grid),
            ("layout", layout),
            ("implementation", "source" if source else "native"),
        ):
            book.equal(label + ":stub-binding:" + key, trace[key], expected)
        if trace.get("scientific_failure"):
            book.bound(label + ":retained-scientific-failure", [0.003], "response_voltage_V")
        if trace.get("raise"):
            raise ValueError("synthetic validator failure")
        return trace["normalized"]

    monkeypatch.setattr(assessment, "validate_trace", validate)
    return calls


def populated(unit="S01", *, adjust=None):
    result = assessment.RefinementAssessment(unit)
    for mesh, implementation in history_plan(unit):
        trace = stub_trace(result.data, mesh, implementation)
        if adjust:
            adjust(mesh, implementation, trace)
        result.add_history(mesh, implementation, trace)
    return result


@pytest.mark.parametrize("unit", UNITS)
def test_all_fixed_histories_and_seventeen_pairs_are_mandatory(unit, delegated_validator):
    assessment_unit = populated(unit)
    report = assessment_unit.finish()
    assert report["passed"], report["failures"]
    assert (
        len(delegated_validator) == report["history_count"] == report["history_denominator"] == 10
    )
    assert len(report["comparisons"]) == report["comparison_denominator"] == 17
    stages = len(unit_data(unit)["waveform"])
    assert sum(c["stages_evaluated"] for c in report["comparisons"]) == 17 * stages
    assert report["comparison_stage_denominator"] == 17 * stages
    for c in report["comparisons"]:
        assert c["label"] == c["name"]
        assert c["stages"] == stages
        assert len(c["norms"]) == 7 * stages
    grid_edges = [
        (tuple(c["edge"]), c["left"][1])
        for c in report["comparisons"]
        if c["kind"] == "mesh-refinement"
    ]
    assert grid_edges == [(edge, impl) for impl in IMPLEMENTATIONS for edge in EDGES]
    assert report["fine_edge_description"] == {
        "comparisons": 6,
        "passed": 6,
        "strict_acceptance_override": False,
    }
    assert len(report["refinement_ratios"]) == 14 * stages
    assert all(
        r["ratio"] is None and r["asymptotic_order"] is None for r in report["refinement_ratios"]
    )
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert assessment_unit.finish() == report  # No duplicate comparisons on repeat reads.


@pytest.mark.parametrize("missing", [("A", "native"), ("E", "source")])
def test_missing_history_keeps_all_comparisons_and_fails(delegated_validator, missing):
    result = assessment.RefinementAssessment("S01")
    for mesh, implementation in history_plan("S01"):
        if (mesh, implementation) != missing:
            result.add_history(mesh, implementation, stub_trace(result.data, mesh, implementation))
    report = result.finish()
    assert not report["passed"]
    assert len(report["comparisons"]) == 17
    assert report["history_count"] == 9
    assert any(c["name"].endswith(":comparison-schema") for c in report["failures"])


def test_empty_assessment_never_passes_vacuously():
    report = assessment.RefinementAssessment("S01").finish()
    assert not report["passed"] and report["failures"]
    assert len(report["histories"]) == 10
    assert len(report["comparisons"]) == 17
    assert all(not c["passed"] and c["stages_evaluated"] == 0 for c in report["comparisons"])


def test_duplicate_cannot_replace_original_or_hide_failed_history(delegated_validator):
    result = populated(
        adjust=lambda mesh, impl, t: (
            t.update(scientific_failure=True) if (mesh, impl) == ("A", "native") else None
        )
    )
    rejected = result.add_history("A", "native", stub_trace(result.data, "A", "native"))
    assert rejected["accepted"] is False
    report = result.finish()
    assert report["history_count"] == 10 and report["history_attempts"] == 11
    assert not report["passed"]
    assert any(c["name"].endswith(":retained-scientific-failure") for c in report["failures"])
    assert any(c["name"].endswith(":unique-history") for c in report["failures"])
    with pytest.raises(RuntimeError, match="finished"):
        result.add_history("A", "native", {})


@pytest.mark.parametrize("mesh,implementation", [("Z", "native"), ("A", "other"), ([], "native")])
def test_unknown_ingestion_is_recorded_as_failure(mesh, implementation):
    result = assessment.RefinementAssessment("S01")
    assert result.add_history(mesh, implementation, {})["accepted"] is False
    assert not result.finish()["passed"]


@pytest.mark.parametrize(
    "damage", ["unit", "inputs", "grid", "implementation", "layout", "exception", "raise"]
)
def test_wrong_binding_or_exception_cannot_be_success(delegated_validator, damage):
    def adjust(mesh, implementation, trace):
        if (mesh, implementation) != ("A", "native"):
            return
        if damage == "inputs":
            trace["inputs"]["temperature_K"] += 1
        elif damage == "grid":
            trace["grid"]["z"] *= 2
        elif damage == "layout":
            trace["layout"]["negative_tab"] = "left"
        else:
            trace[damage] = "unexpected"

    report = populated(adjust=adjust).finish()
    assert not report["passed"] and report["failures"]
    assert len(report["comparisons"]) == 17


@pytest.mark.parametrize("damage", ["missing-stages", "missing-samples", "nonfinite"])
def test_damaged_normalized_history_is_not_silently_skipped(delegated_validator, damage):
    def adjust(mesh, implementation, trace):
        if (mesh, implementation) != ("A", "native"):
            return
        if damage == "missing-stages":
            trace["normalized"] = []
        elif damage == "missing-samples":
            trace["normalized"][0]["times"] = np.array([0.0])
        else:
            trace["normalized"][0]["values"][0, 0] = np.nan

    report = populated(adjust=adjust).finish()
    assert not report["passed"] and report["failures"]
    json.dumps(report, allow_nan=False)


def test_fine_edge_success_never_overrides_failed_coarser_edge(delegated_validator):
    def adjust(mesh, implementation, trace):
        if mesh == "A":
            trace["normalized"][0]["values"][:, 0] += 0.01

    report = populated(adjust=adjust).finish()
    assert not report["passed"]
    assert report["fine_edge_description"]["passed"] == 6
    assert report["fine_edge_description"]["strict_acceptance_override"] is False
    assert len(report["failures"]) == 2
    assert all(c["threshold"] == "response_voltage_V" for c in report["failures"])


def test_raw_radial_data_and_array_views_are_not_retained(delegated_validator):
    result = assessment.RefinementAssessment("S01")
    trace = stub_trace(result.data, "A", "native")
    raw_radial = trace["normalized"][0]["profiles"]["negative"]["theta"]
    reference = weakref.ref(raw_radial)
    raw_values = trace["normalized"][0]["values"]
    result.add_history("A", "native", trace)
    compact = result._normalized[("A", "native")][0]
    assert set(compact["profiles"]["negative"]) == {"total", "q"}
    assert not np.shares_memory(raw_values, compact["values"])
    raw_values[:] = 9
    assert not np.all(compact["values"] == 9)
    del trace, raw_radial
    gc.collect()
    assert reference() is None
    exposed = result.data
    exposed["temperature_K"] = -1
    assert result.data["temperature_K"] != -1


@pytest.mark.parametrize(
    "damage", [None, "wrong-grid", "wrong-implementation", "nonfinite", "missing-profile"]
)
def test_real_complete_validator_is_used_before_compaction(monkeypatch, damage):
    # The pre-existing synthetic algebraic trace is a tiny validator fixture,
    # never a changed refinement case or replacement scientific history.
    data, grid, layout, trace = synthetic_native()
    monkeypatch.setattr(assessment, "unit_data", lambda _unit: deepcopy(data))
    monkeypatch.setattr(assessment, "MESHES", {**MESHES, "A": grid})
    monkeypatch.setattr(assessment, "LAYOUT", layout)
    if damage == "wrong-grid":
        trace["grid"] = {"z": 8, "r": 12}
    elif damage == "wrong-implementation":
        trace["implementation"] = "source"
    elif damage == "nonfinite":
        trace["segments"][0]["raw_profiles"]["negative"]["theta"][0][0][0] = np.nan
    elif damage == "missing-profile":
        del trace["segments"][0]["raw_profiles"]["positive"]
    result = assessment.RefinementAssessment("S01")
    record = result.add_history("A", "native", trace)
    assert bool(record["passed"]) is (damage is None)
    if damage is None:
        names = [c["name"] for c in result._book.items]
        assert any(n.endswith(":native-recorded-BV-residual") for n in names)
        assert any(n.endswith(":local-BV") for n in names)
        assert any(n.endswith(":static-q") for n in names)
        assert any(n.endswith(":total-Li") for n in names)
        assert record["check_range"][1] - record["check_range"][0] > 100


def test_fixed_common_partition_ratio_does_not_imply_order(delegated_validator):
    def adjust(mesh, implementation, trace):
        offset = {"A": 3e-4, "B": 1e-4, "C": 0.0, "D": 1e-4, "E": 0.0}[mesh]
        trace["normalized"][0]["values"][:, 0] += offset

    report = populated(adjust=adjust).finish()
    ratios = [r for r in report["refinement_ratios"] if r.get("quantity") == "voltage_V"]
    assert len(ratios) == 2
    assert all(r["ratio"] == pytest.approx(2) and r["asymptotic_order"] is None for r in ratios)
    assert all(
        tuple(c["edge"]) in FINE_EDGES for c in report["comparisons"] if c["descriptive_fine_edge"]
    )
