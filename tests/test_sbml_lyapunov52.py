"""QR must retain decaying directions, original state, and finite cost/failures."""

import json

import numpy as np
import pytest
from tests.reference.sbml_analysis52_close_cases import model, qr_reference, reference

from chemworld.physchem.sbml_lyapunov import sbml_lyapunov_spectrum
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_lyapunov import analyze_sbml_lyapunov_world
from chemworld.world.sbml_world import sbml_world


def analyze(kind, *, scaled=False, **options):
    spec = SBMLSystemSpec(model(kind, scaled=scaled))
    return sbml_lyapunov_spectrum(
        spec,
        spec.frame.initial_values,
        start_time_s=0,
        qr_interval_s=0.5,
        measurement_duration_s=options.pop("measurement_duration_s", 4),
        amount_scales_mol=dict.fromkeys(spec.frame._species, 1.0),
        **options,
    )


@pytest.mark.parametrize("kind", ("birth", "closed", "nonnormal", "growth", "nonlinear", "neutral"))
def test_original_path_and_independent_periodic_qr(kind):
    report = analyze(kind, transient_duration_s=1)
    assert report["status"] == "complete", report["failure"]
    expected = qr_reference(
        kind,
        report["basis_scaled_amounts"],
        list(report["amount_scales_mol"].values()),
        report["records"],
    )
    assert report["exponents_s_inverse"] == pytest.approx(expected, rel=3e-6, abs=3e-8)
    final = report["records"][-1]
    assert [final["coordinates"]["N:" + s] for s in report["species_ids"]] == pytest.approx(
        reference(kind, 5)["amounts"], rel=3e-6
    )
    assert final["measurement_elapsed_s"] == 4
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_qr_does_not_lose_long_duration_contracting_directions():
    result = analyze("birth", measurement_duration_s=120)
    assert result["status"] == "complete"
    assert result["exponents_s_inverse"] == pytest.approx([-0.5], rel=3e-6)


def test_source_reparameterization_preserves_the_mol_analysis():
    a, b = analyze("nonnormal"), analyze("nonnormal", scaled=True)
    assert a["exponents_s_inverse"] == pytest.approx(b["exponents_s_inverse"], rel=3e-6)


def test_budget_keeps_completed_prefix_and_actual_attempt_cost():
    result = analyze("birth", maximum_rhs_evaluations=130)
    assert result["status"] == "partial"
    assert 0 < result["completed_intervals"] < result["requested_intervals"]
    assert result["rhs_evaluations"] == 130
    assert "budget" in result["failure"]


@pytest.mark.parametrize(
    "options",
    (
        {"qr_interval_s": 0},
        {"measurement_duration_s": 0},
        {"transient_duration_s": -1},
        {"maximum_intervals": 1},
        {"maximum_rhs_evaluations": False},
    ),
)
def test_invalid_duration_or_budget_is_rejected(options):
    spec = SBMLSystemSpec(model("birth"))
    kwargs = {
        "start_time_s": 0,
        "qr_interval_s": 0.5,
        "measurement_duration_s": 4,
        "amount_scales_mol": {"X": 1},
        **options,
    }
    with pytest.raises(ValueError):
        sbml_lyapunov_spectrum(spec, spec.frame.initial_values, **kwargs)


def test_actual_controlled_world_and_full_snapshot_stay_unchanged():
    composition = sbml_world(model("closed"), controlled_parameters=("k1",))
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=52)
        for action in (
            {"operation": "set_model_parameters", "values": {"k1": 0.7}},
            {"operation": "wait", "duration_s": 0.2},
        ):
            assert runtime.execute(action).info["transaction_status"] == "committed"
        before = runtime.snapshot().to_dict()
        options = {
            "qr_interval_s": 0.2,
            "measurement_duration_s": 1,
            "amount_scales_mol": {"A": 1, "B": 1},
        }
        report = analyze_sbml_lyapunov_world(runtime._state, composition, **options)
        assert report["exponents_s_inverse"] == pytest.approx([-0.9], rel=3e-6)
        assert runtime.snapshot().to_dict() == before
        restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(before)))
        assert analyze_sbml_lyapunov_world(restored._state, composition, **options) == report
        assert np.isfinite(report["rhs_evaluations"])
    finally:
        runtime.close()
        if restored:
            restored.close()
