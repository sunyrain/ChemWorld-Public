"""Original SBML analysis, conservation, units, noise scope and read-only worlds."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.data.sbml import SBMLModel
from chemworld.physchem.biochemical_analysis import analyze_biochemical_steady_state
from chemworld.physchem.biochemical_analysis_specs import (
    BiochemicalAnalysisPolicy,
    BiochemicalAnalysisResult,
)
from chemworld.physchem.sbml_kinetics import SBMLKinetics
from reference.biochemical_analysis52_cases import analytic_case, build_model, frame_case


def analyze(kind="birth", **kwargs):
    frame, values = frame_case(kind, **kwargs)
    return analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()


@pytest.mark.parametrize("kind", ["birth", "cascade", "closed"])
def test_same_source_rates_give_analytic_steady_control_and_noise(kind):
    report = analyze(kind)
    expected = analytic_case(kind)
    assert report["status"] == "steady_state_found"
    assert report["stability"] == "asymptotically_stable"
    np.testing.assert_allclose(
        list(report["steady_amounts_mol"].values()), expected["amounts"], rtol=2e-7, atol=2e-10
    )
    np.testing.assert_allclose(report["jacobian_s_inv"], expected["jacobian"], atol=2e-10)
    for field in ("scaled_amount_control", "scaled_flux_control"):
        np.testing.assert_allclose(report["control"][field], expected[field], atol=2e-10)
    np.testing.assert_allclose(
        report["linear_noise"]["covariance_mol2"], expected["covariance"], rtol=3e-7, atol=1e-32
    )
    assert report["linear_noise"]["lyapunov_relative_residual"] < 1e-10
    assert report["chemical_composition_scope"] == "unknown_formal_entities"
    assert report["energy_scope"] == "not_declared_formal_kinetics"


def test_closed_pool_removes_conservation_zero_mode_and_uses_current_total():
    frame, values = frame_case("closed")
    values["N:A"] += 0.2
    before = dict(values)
    report = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
    assert values == before
    assert sum(report["steady_amounts_mol"].values()) == pytest.approx(1.0)
    assert report["removed_conservation_modes"] == 1
    assert len(report["reduced_eigenvalues_s_inv"]) == 1
    assert any(abs(row["real"]) < 1e-12 for row in report["full_eigenvalues_s_inv"])
    np.testing.assert_allclose(
        report["linear_noise"]["conservation_covariance_residual_mol2"], 0, atol=1e-32
    )
    assert report["stability"] == "asymptotically_stable"


def test_source_mmol_ml_minute_units_match_native_amount_analysis():
    original = analyze("birth", volume=0.5)
    scaled = analyze("birth", volume=0.5, scaled=True)
    for key in ("steady_amounts_mol", "jacobian_s_inv", "stoichiometry"):
        assert scaled[key] == original[key]
    np.testing.assert_allclose(
        scaled["linear_noise"]["covariance_mol2"],
        original["linear_noise"]["covariance_mol2"],
        atol=1e-32,
    )


def test_unstable_and_marginal_roots_never_claim_stationary_noise():
    unstable = analyze("unstable")
    assert unstable["status"] == "steady_state_found"
    assert unstable["stability"] == "unstable"
    assert unstable["reduced_eigenvalues_s_inv"][0]["real"] == pytest.approx(0.5)
    assert unstable["linear_noise"]["status"] == "unavailable"
    marginal = analyze("birth", j=0, k1=0, zero=True)
    assert marginal["stability"] == "marginal_or_unresolved"
    assert marginal["linear_noise"]["status"] == "unavailable"
    assert marginal["control"]["status"] == "unavailable"


def test_zero_amount_and_zero_flux_normalization_is_undefined():
    report = analyze("birth", j=0, zero=True)
    assert report["steady_amounts_mol"]["X"] == pytest.approx(0, abs=1e-12)
    assert report["control"]["scaled_amount_control"] == [[None, None]]
    assert report["control"]["scaled_flux_control"] == [[None, None], [None, None]]
    assert report["linear_noise"]["covariance_mol2"] == [[0.0]]


def changed_frame(text, original):
    frame = SBMLKinetics(SBMLModel(text, original.source))
    return frame, dict(frame.initial_values)


def test_boundary_pool_is_held_and_excluded_from_conservation_dynamic_modes():
    source = build_model("closed")
    text = source.raw_text
    marker = '<species id="B"'
    start = text.index(marker)
    text = text[:start] + text[start:].replace(
        'boundaryCondition="false"', 'boundaryCondition="true"', 1
    )
    frame, values = changed_frame(text, source)
    result = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
    assert result["species_ids"] == ["A"]
    assert result["removed_conservation_modes"] == 0
    assert result["fixed_coordinates"]["N:B"] == values["N:B"]
    assert result["steady_amounts_mol"]["A"] == pytest.approx(3 * values["N:B"])


def test_smooth_assignment_parameter_is_differentiated_through_original_context():
    source = build_model()
    text = source.raw_text.replace('id="k1" value="0.5"', 'id="base" value="0.5"')
    text = text.replace(
        "</listOfParameters>",
        '<parameter id="k1" constant="false" '
        'units="inverse_time"/></listOfParameters><listOfRules>'
        '<assignmentRule variable="k1"><math xmlns="http://www.w3.org/1998/Math/MathML">'
        "<apply><times/><cn>2</cn><ci>base</ci></apply></math></assignmentRule></listOfRules>",
    )
    frame, values = changed_frame(text, source)
    report = analyze_biochemical_steady_state(frame, values).to_dict()
    assert report["steady_amounts_mol"]["X"] == pytest.approx(0.2)
    assert report["jacobian_s_inv"] == [[-1.0]]


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda text: text.replace('reversible="false"', 'reversible="true"'), "reversible_net"),
        (
            lambda text: text.replace('extentUnits="amount_unit"', 'extentUnits="mole"'),
            "noninteger",
        ),
    ],
)
def test_lna_requires_real_unit_converted_integer_independent_channels(change, reason):
    source = build_model(scaled=True)
    frame, values = changed_frame(change(source.raw_text), source)
    report = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
    assert report["status"] == "steady_state_found"
    assert report["linear_noise"]["status"] == "unavailable"
    assert reason in report["linear_noise"]["reason"]


def test_conversion_factor_can_restore_integer_jumps_after_source_unit_conversion():
    source = build_model(scaled=True)
    text = source.raw_text.replace(
        'extentUnits="amount_unit"', 'extentUnits="mole" conversionFactor="cf"'
    )
    text = text.replace(
        "</listOfParameters>",
        '<parameter id="cf" value="1000" '
        'constant="true" units="dimensionless"/></listOfParameters>',
    )
    frame, values = changed_frame(text, source)
    report = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
    assert report["stoichiometry"] == [[1.0, -1.0]]
    assert report["linear_noise"]["status"] == "available"


def test_result_json_roundtrip_and_input_snapshot_preservation():
    frame, values = frame_case()
    original = frame.model.to_dict()
    first = analyze_biochemical_steady_state(frame, values, include_lna=True)
    second = BiochemicalAnalysisResult.from_dict(
        json.loads(json.dumps(first.to_dict(), allow_nan=False))
    )
    assert first.to_dict() == second.to_dict()
    assert frame.model.to_dict() == original


def test_world_consumer_keeps_current_coordinates_clock_and_every_ledger_unchanged():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.sbml_analysis import analyze_sbml_world_state
    from chemworld.world.sbml_world import sbml_world

    composition = sbml_world(build_model("closed"))
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=52)
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.5})
        assert outcome.info["transaction_status"] == "committed"
        before = runtime.snapshot().to_dict()
        model = composition.physics.models_of("sbml")[runtime._state.vessel_id]
        result = analyze_sbml_world_state(runtime._state, model, include_lna=True)
        assert runtime.snapshot().to_dict() == before
        assert result.success
        assert result.to_dict()["time_s"] == 0.5
        assert sum(result.steady_amounts_mol.values()) == pytest.approx(0.8)
    finally:
        runtime.close()


def test_budget_exhaustion_preserves_failed_root_diagnostic():
    frame, values = frame_case("unstable")
    values["N:X"] = 2.0
    result = analyze_biochemical_steady_state(
        frame, values, policy=BiochemicalAnalysisPolicy(maximum_iterations=1)
    ).to_dict()
    assert result["status"] == "steady_state_not_found"
    assert result["failure_reason"] == "iteration_limit"
    assert "steady_amounts_mol" not in result
    assert values["N:X"] == 2.0


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("<ci>X</ci>", "<apply><abs/><ci>X</ci></apply>"),
        lambda text: text.replace(
            "<ci>j</ci>",
            '<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">time</csymbol>',
        ),
    ],
)
def test_nonsmooth_or_nonautonomous_dynamics_cannot_be_called_steady(change):
    source = build_model()
    frame, values = changed_frame(change(source.raw_text), source)
    with pytest.raises(ValueError, match=r"smooth|time-dependent"):
        analyze_biochemical_steady_state(frame, values)


def test_policy_rejects_invalid_scales_and_erasing_rank_tolerances():
    with pytest.raises(ValueError):
        BiochemicalAnalysisPolicy(rank_relative_tolerance=1)
    with pytest.raises(ValueError):
        replace(BiochemicalAnalysisPolicy(), amount_scale_mol=-1)


@pytest.mark.parametrize("items_per_mole,available", [(6.02214076e23, True), (1e20, False)])
def test_item_units_need_explicit_molecular_conversion_for_lna(items_per_mole, available):
    source = build_model()
    text = source.raw_text.replace('kind="mole"', 'kind="item"')
    frame = SBMLKinetics(SBMLModel(text, source.source, items_per_mole=items_per_mole))
    report = analyze_biochemical_steady_state(
        frame, frame.initial_values, include_lna=True
    ).to_dict()
    assert (report["linear_noise"]["status"] == "available") is available
    if available:
        assert report["steady_amounts_mol"]["X"] == pytest.approx(
            0.4 / items_per_mole, rel=1e-8, abs=0
        )
        assert report["linear_noise"]["covariance_mol2"][0][0] == pytest.approx(
            0.4 / items_per_mole**2, rel=1e-8, abs=0
        )
    else:
        assert "item_policy" in report["linear_noise"]["reason"]


def test_negative_net_propensity_can_have_deterministic_root_but_no_lna():
    source = build_model()
    text = source.raw_text.replace('id="j" value="0.2"', 'id="j" value="-0.2"')
    old = "<apply><times/><ci>V</ci><ci>k1</ci><ci>X</ci></apply>"
    text = text.replace(
        old, "<apply><minus/>" + old + "<apply><times/><ci>V</ci><cn>0.4</cn></apply></apply>"
    )
    frame, values = changed_frame(text, source)
    report = analyze_biochemical_steady_state(frame, values, include_lna=True).to_dict()
    assert report["stability"] == "asymptotically_stable"
    assert report["linear_noise"]["reason"] == "negative_reaction_propensity"


def test_rate_rules_and_events_are_rejected_without_advancing_them():
    from test_sbml import model

    frame = SBMLKinetics(model())
    with pytest.raises(ValueError, match="event-free"):
        analyze_biochemical_steady_state(frame, frame.initial_values)
