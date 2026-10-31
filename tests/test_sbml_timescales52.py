"""Trajectory modes differ from steady relaxation and finite-window growth."""

import json
import subprocess
import sys

import numpy as np
import pytest
from tests.reference.sbml_analysis52_close_cases import model, reference
from tests.reference.sbml_analysis52_close_world import exercise

from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.sbml_timescales import sbml_trajectory_timescales
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.sbml_world import sbml_world


def analyze(kind, *, scaled=False, **options):
    spec = SBMLSystemSpec(model(kind, scaled=scaled))
    return sbml_trajectory_timescales(
        spec,
        spec.frame.initial_values,
        start_time_s=0,
        sample_offsets_s=(0, 0.2, 1),
        amount_scales_mol=dict.fromkeys(spec.frame._species, 1.0),
        **options,
    )


@pytest.mark.parametrize(
    "kind", ("separated", "nonlinear", "closed", "growth", "neutral", "cycle", "defective")
)
def test_trajectory_states_and_local_modes_have_independent_equations(kind):
    report = analyze(kind)
    assert report["status"] == "complete", report["failure"]
    for row in report["records"]:
        expected = reference(kind, row["time_s"])
        values = [row["coordinates"]["N:" + s] for s in report["species_ids"]]
        assert values == pytest.approx(expected["amounts"], rel=3e-6)
        spectrum = np.array(
            [complex(m["real_s_inverse"], m["imaginary_rad_s"]) for m in row["modes"]]
        )
        assert np.sort_complex(spectrum) == pytest.approx(
            np.sort_complex(expected["eigenvalues"]), rel=3e-6, abs=3e-8
        )
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_fast_slow_gap_and_nonnormality_are_both_visible():
    row = analyze("separated")["records"][0]
    assert row["has_resolved_fast_slow_separation"]
    assert row["stable_decay_gaps"][0]["ratio"] == pytest.approx(40)
    assert row["nonnormal_commutator_norm"] > 0


def test_complex_pair_is_not_split_and_defective_modes_are_unresolved():
    cycle = analyze("cycle")["records"][0]
    assert len(cycle["modes"]) == 2
    assert all(m["oscillation_period_s"] is not None for m in cycle["modes"])
    assert not cycle["has_resolved_fast_slow_separation"]
    defective = analyze("defective")["records"][0]
    assert not defective["modal_basis_resolved"]


def test_neutral_is_not_infinite_decay_and_growth_is_not_negative_relaxation():
    neutral = analyze("neutral")["records"][0]["modes"][0]
    assert neutral["decay_time_s"] is None and neutral["growth_time_s"] is None
    growth = analyze("growth")["records"][0]["modes"][0]
    assert growth["growth_time_s"] == pytest.approx(2.5)
    assert growth["decay_time_s"] is None


def test_nonstationary_local_decay_changes_along_same_trajectory():
    rows = analyze("nonlinear")["records"]
    assert rows[0]["modes"][0]["decay_time_s"] != rows[-1]["modes"][0]["decay_time_s"]


def test_scale_equivalence_and_budget_diagnostics():
    a, b = analyze("nonnormal"), analyze("nonnormal", scaled=True)
    for ra, rb in zip(a["records"], b["records"], strict=True):
        assert np.array(ra["reduced_jacobian_s_inverse"]) == pytest.approx(
            np.array(rb["reduced_jacobian_s_inverse"]), rel=3e-6
        )
    limited = analyze("nonlinear", maximum_rhs_evaluations=1)
    assert limited["status"] == "partial" and limited["completed_samples"] == 1
    assert limited["rhs_evaluations"] == 1 and limited["directional_rate_evaluations"] == 1


@pytest.mark.parametrize("kind", ("qr", "timescales"))
def test_actual_world_query_full_json_and_fresh_process_continuation(kind):
    runtime = ChemWorldRuntime(
        composition=sbml_world(model("closed"), controlled_parameters=("k1",))
    )
    try:
        runtime.reset(seed=52)
        for action in (
            {"operation": "set_model_parameters", "values": {"k1": 0.7}},
            {"operation": "wait", "duration_s": 0.4},
        ):
            assert runtime.execute(action).info["transaction_status"] == "committed"
        before = runtime.snapshot().to_dict()
        result = exercise(runtime, kind)
        assert result["full_snapshot_unchanged"] and result["followup_committed"]
        assert result["report"]["initial_coordinates"]["P:k1"] == 0.7
        assert result["report"]["start_time_s"] == 0.4
        child = subprocess.run(
            [sys.executable, "-m", "tests.reference.sbml_analysis52_close_world"],
            input=json.dumps({"snapshot": before, "kind": kind}),
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        assert json.loads(child.stdout) == json.loads(json.dumps(result, allow_nan=False))
    finally:
        runtime.close()
