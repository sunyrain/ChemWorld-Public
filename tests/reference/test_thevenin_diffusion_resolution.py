"""Continuous transient limits and honest coarse-resolution accounting."""

import json
import sys

import numpy as np
import pytest
import run_thevenin_diffusion_resolution as runner
from thevenin_diffusion_resolution import (
    COMMON_TIMES,
    RESOLUTION_GRIDS,
    assess_resolution_block,
    continuous_solution,
    continuous_surface,
    observation_indices,
    sampled_rows,
)


def test_continuous_start_and_early_half_space_limit_are_independent_checks():
    assert continuous_surface(0.) == .08
    with pytest.raises(ValueError, match="nonnegative"):
        continuous_surface(-1.)
    # For t << tau the remote reflecting boundary has exponentially small influence.
    for time_s in (.0325, .1, .65):
        half_space = .08-2*.5/100*np.sqrt(300*time_s/np.pi)
        assert continuous_surface(time_s) == pytest.approx(half_space, abs=1e-12)
    result = continuous_solution()
    assert result["maximum_series_tail_bound"] < 1e-12
    assert abs(continuous_surface(result["event_time_s"])) < 1e-12
    assert result["event_time_s"] > COMMON_TIMES[-1]


def test_common_samples_cannot_be_replaced_by_sparse_interpolation():
    trace = {"segments": [{"times_s": [*COMMON_TIMES, .67],
                          "rows": np.zeros((22, 12)), "profiles": np.zeros((22, 128))}]}
    assert sampled_rows(trace)[0].shape == (21, 12)
    trace["segments"][0]["times_s"][7] += .0001
    with pytest.raises(ValueError, match="no interpolation"):
        sampled_rows(trace)


def test_internal_event_nodes_do_not_replace_fixed_observations():
    times = np.r_[COMMON_TIMES, .66, .67, .6703]
    indices = observation_indices(times)
    assert np.array_equal(times[indices][:21], COMMON_TIMES)
    assert times[indices][-1] == .6703
    assert len(indices) == 22
    with pytest.raises(ValueError, match="absent or duplicated"):
        observation_indices(np.delete(times, 3))


def test_finest_qualification_does_not_relabel_failed_coarse_grids():
    units = []
    for index, cells in enumerate(RESOLUTION_GRIDS):
        error = {"maximum_absolute_error": [.003/4**index]*3,
                 "event_time_error_s": .007/4**index}
        units.append({"cells": cells, "internally_valid": True, "passed": index >= 2,
                      "analytic_comparisons": {"native": error, "corrected": error}})
    data = {"units": units, "analytic": {"maximum_series_tail_bound": 0.}}
    assess_resolution_block(data)
    assert data["resolution_qualified"] and data["passed"] == 2
    assert [row["passed"] for row in data["units"]] == [False, False, True, True]


def test_resolution_failures_preserve_all_eight_routes_and_output(tmp_path, monkeypatch):
    pytest.importorskip("pybamm")
    calls = []

    def failed(name, cells):
        calls.append((cells, name))
        raise RuntimeError("fixture-only failure; not scientific data")

    monkeypatch.setattr(runner, "native_resolution", lambda cells: failed("native", cells))
    monkeypatch.setattr(runner, "reference_resolution", lambda cells: failed("corrected", cells))
    output = tmp_path / "fixture-only-resolution.json"
    monkeypatch.setattr(sys, "argv", ["resolution", str(output)])
    assert runner.main() == 1
    assert calls == [(n, r) for n in RESOLUTION_GRIDS for r in ("native", "corrected")]
    original = output.read_bytes()
    data = json.loads(original)
    assert data["path_denominator"] == data["paths_completed"] == 8
    assert data["denominator"] == len(data["units"]) == 4 and data["passed"] == 0
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_bytes() == original
