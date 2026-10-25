"""The optional CPU loader is fail-closed and leaves the base package usable."""

import pytest

from chemworld.cpu import CpuBackendUnavailable, cpu_status, select_backend


def test_reference_selection_does_not_require_a_native_library(tmp_path):
    missing = tmp_path / "missing-cpu-library.dll"
    assert select_backend("reference", missing) == "reference"
    assert select_backend("auto", missing) == "reference"
    status = cpu_status(missing)
    assert status["available"] is False
    assert status["backend"] == "reference"


def test_explicit_cpu_selection_fails_at_initialization(tmp_path):
    with pytest.raises(CpuBackendUnavailable, match="CPU backend unavailable"):
        select_backend("cpu", tmp_path / "missing-cpu-library.dll")
