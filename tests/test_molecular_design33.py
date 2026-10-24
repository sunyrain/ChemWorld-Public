"""Bounded design consumes actual paths and preserves numerical/physical boundaries."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from chemworld.examples.molecular_process_design import design_receiver, two_stage_design
from chemworld.physchem.bounded_design import (
    BoundedDesignPolicy,
    DesignConstraint,
    DesignEvaluationError,
    DesignPoint,
    DesignVariable,
    solve_bounded_design,
)
from chemworld.physchem.molecular_processes import MolecularHeatDuty, MolecularPressureStage
from chemworld.physchem.molecular_recycle import MolecularRecyclePath
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_process_design import (
    MolecularDesignControl,
    MolecularProcessDesignSpec,
    MolecularUtilityPrices,
    evaluate_molecular_design,
    optimize_molecular_process,
    optimized_process_receiver,
)
from chemworld.world.molecular_process_world import build_molecular_process_world
from reference.molecular_design33_cases import specification
from reference.molecular_design33_reference import CASES, optimum, self_check


def test_independent_reference_checks():
    assert self_check()["derivative_checks"] == 16


@pytest.mark.parametrize("index", (0, 4, 5, 6, 7))
def test_original_path_design_matches_analytic_active_constraints(index):
    spec = specification(index)
    before = spec.to_dict()
    result = optimize_molecular_process(spec)
    expected = optimum(CASES[index])
    assert result.success, result.optimization.failure_reason
    assert result.optimization.values["interstage_pressure"] == pytest.approx(
        expected["pressure"], rel=2e-4
    )
    assert result.optimization.values["cooling_duty"] == pytest.approx(
        -expected["cooling"], abs=5e-4
    )
    assert result.optimization.point.objective == pytest.approx(expected["cost"], abs=5e-5)
    assert spec.to_dict() == before
    assert result.selected_path.evaluate().success
    assert result.optimization.to_dict()["experiment_resource_debit"] is None
    assert result.optimization.trials[-1]["fresh_verification"]


def test_generic_common_bounded_coordinates_constrained_minimum():
    variables = tuple(DesignVariable(k, "1", (0.0, 1.0), 0.1) for k in ("x", "y"))
    constraints = (
        DesignConstraint("sum", "1", 1.0, 1e-8, lower=1.2),
        DesignConstraint("difference", "1", 1.0, 1e-8, lower=-0.4, upper=-0.4),
    )

    def evaluate(v):
        return DesignPoint(
            (v["x"] - 0.2) ** 2 + (v["y"] - 0.6) ** 2,
            {"sum": v["x"] + v["y"], "difference": v["x"] - v["y"]},
        )

    result = solve_bounded_design(variables, constraints, evaluate)
    assert result.success
    assert result.values == pytest.approx({"x": 0.4, "y": 0.8}, abs=1e-7)


def test_original_multistart_selects_best_local_candidate_without_global_claim():
    spec = specification(7)
    spec = replace(spec, starts=(spec.starts[0], specification(7, 0.8).starts[0]))
    result = optimize_molecular_process(spec)
    assert result.success
    trials = result.optimization.trials
    objectives = [trials[a["verified_trial"]]["objective"] for a in result.optimization.attempts]
    assert objectives[0] < objectives[1] - 8.0
    assert result.optimization.point.objective == pytest.approx(min(objectives), abs=1e-8)
    assert trials[-1]["start"] == 0
    assert "local_candidate" in result.optimization.to_dict()["scope"]


def test_multistart_preserves_failed_physics_without_invented_objective():
    variable = DesignVariable("x", "1", (0.0, 1.0), 0.7)
    calls = []

    def evaluate(v):
        calls.append(v)
        if v["x"] < 0.2:
            raise DesignEvaluationError("outside_original_domain", {"physical_failure": True})
        return DesignPoint((v["x"] - 0.6) ** 2, {})

    result = solve_bounded_design((variable,), (), evaluate, starts=({"x": 0.1}, {"x": 0.8}))
    assert result.success
    assert len(result.trials) == len(calls)
    assert not result.attempts[0]["success"] and result.attempts[1]["success"]
    assert "objective" not in result.trials[0]
    assert result.trials[0]["diagnostics"]["physical_failure"]
    assert result.trials[-1]["start"] == 1


def test_budget_exhaustion_keeps_feasible_diagnostic_but_no_selected_path():
    spec = replace(two_stage_design(), policy=BoundedDesignPolicy(maximum_evaluations=2))
    result = optimize_molecular_process(spec)
    assert not result.success and result.selected_path is None
    assert len(result.optimization.trials) == 2
    assert result.optimization.best_feasible_trial is not None
    assert "budget_exhausted" in result.optimization.failure_reason
    with pytest.raises(ValueError, match="unverified or exhausted"):
        optimized_process_receiver(result, design_receiver(spec)[0])


def test_selected_candidate_requires_another_fresh_physical_verification(monkeypatch):
    monkeypatch.setattr(
        "chemworld.physchem.bounded_design.minimize",
        lambda *a, **k: SimpleNamespace(
            success=True, x=[0.5], status=0, nit=1, message="isolated optimizer stub"
        ),
    )
    calls = []

    def evaluate(v):
        calls.append(v)
        return DesignPoint(0.0, {"output": 1.0 if len(calls) == 1 else 0.0})

    result = solve_bounded_design(
        (DesignVariable("x", "1", (0.0, 1.0), 0.5),),
        (DesignConstraint("output", "1", 1.0, 0.0, lower=1.0),),
        evaluate,
    )
    assert not result.success and result.values is None
    assert result.failure_reason == "selected_candidate_failed_fresh_constraints"
    assert len(result.trials) == 2 and all(t["fresh_verification"] for t in result.trials)


def test_actual_out_of_domain_path_has_no_substituted_cost():
    spec = two_stage_design()
    bad = MolecularDesignControl(
        DesignVariable("cold", "W", (-500.0, -400.0), -450.0), "heat_duty_W", 1
    )
    result = optimize_molecular_process(replace(spec, controls=(bad,), starts=()))
    assert not result.success
    assert len(result.optimization.trials) == 1
    assert "objective" not in result.optimization.trials[0]
    assert "original_process" not in result.optimization.trials[0].get("diagnostics", {})
    assert "physical_evaluation_failed" in result.optimization.attempts[0]["failure_reason"]


def test_json_roundtrip_and_strict_units_binding_and_variables():
    spec = two_stage_design()
    data = json.loads(json.dumps(spec.to_dict()))
    assert MolecularProcessDesignSpec.from_dict(data).to_dict() == spec.to_dict()
    with pytest.raises(ValueError, match="unknown or incomplete"):
        MolecularProcessDesignSpec.from_dict({**data, "hidden_override": True})
    data["controls"][0]["hidden_override"] = True
    with pytest.raises(ValueError, match="control"):
        MolecularProcessDesignSpec.from_dict(data)
    with pytest.raises(ValueError, match="matching SI unit"):
        replace(spec.controls[0], variable=replace(spec.controls[0].variable, unit="bar"))
    with pytest.raises(ValueError, match="same device field"):
        replace(
            spec,
            controls=(
                spec.controls[0],
                replace(
                    spec.controls[0],
                    variable=replace(spec.controls[0].variable, parameter_id="duplicate"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="device type"):
        replace(spec, controls=(replace(spec.controls[0], device_index=1),))
    with pytest.raises(ValueError, match="observable and SI"):
        replace(spec, constraints=(DesignConstraint("missing", "K", 1.0, 0.0, lower=0.0),))
    with pytest.raises(ValueError, match="exactly"):
        spec.compile({"interstage_pressure": 3e5})
    with pytest.raises(ValueError, match="outside"):
        spec.compile({"interstage_pressure": 8e5, "cooling_duty": 0.0})


def test_utility_cost_counts_gross_signed_devices_without_net_cancellation():
    spec = two_stage_design()
    domain = spec.path.domain
    path = replace(
        spec.path,
        devices=(
            MolecularPressureStage(domain, "compressor", 2e5, 0.8),
            MolecularHeatDuty(domain, 8.0),
            MolecularHeatDuty(domain, -3.0),
            MolecularPressureStage(domain, "expander", 1e5, 0.8),
        ),
    )
    prices = MolecularUtilityPrices(2.0, 0.3, 4.0, 5.0)
    spec = replace(
        spec, path=path, controls=(spec.controls[0],), constraints=(), starts=(), prices=prices
    )
    point = evaluate_molecular_design(spec, {"interstage_pressure": 2e5})
    obs = point.observations
    assert obs["heating_W"] == 8 and obs["cooling_W"] == 3
    assert obs["shaft_import_W"] > 0 and obs["shaft_export_W"] > 0
    assert point.objective == pytest.approx(
        2 * obs["shaft_import_W"] - 0.3 * obs["shaft_export_W"] + 32 + 15
    )


def test_original_single_recycle_optimized_path_reaches_world_wait():
    original = two_stage_design(x_a=0.25)
    p, d = original.path, original.path.domain
    recycle = MolecularRecyclePath(
        d, p.source_id, p.inlet, (MolecularHeatDuty(d, 5.0),), 0.2, 1e5, (0.000625, 0.001875), 330.0
    )
    control = MolecularDesignControl(
        DesignVariable("heat", "W", (0.0, 20.0), 5.0), "heat_duty_W", 0
    )
    spec = MolecularProcessDesignSpec(
        recycle,
        (control,),
        (DesignConstraint("outlet_temperature_K", "K", 300.0, 2e-5, lower=340.0),),
        MolecularUtilityPrices(shaft_import_per_J=0.0, heating_per_J=1.0),
    )
    result = optimize_molecular_process(spec)
    assert result.success, result.optimization.failure_reason
    assert isinstance(result.selected_path, MolecularRecyclePath)
    assert result.optimization.values["heat"] == pytest.approx(
        0.01 * 8.31446261815324 * 4.25 * 20, abs=2e-5
    )
    template, initial = design_receiver(spec)
    before = initial.to_dict(include_hidden=True)
    selected = optimized_process_receiver(result, template)
    assert initial.to_dict(include_hidden=True) == before
    world = build_molecular_process_world(selected, initial, formulas={"A": "Ar", "B": "C3H8"})
    runtime = ChemWorldRuntime(composition=world)
    try:
        runtime.reset(seed=7)
        outcome = runtime.execute({"operation": "wait", "duration_s": 1.0})
        assert outcome.info["transaction_status"] == "committed", outcome.info
        assert sum(runtime._state.species_amounts.values()) == pytest.approx(0.06)
        assert runtime._state.ledger.flow_heat_J == pytest.approx(
            result.optimization.values["heat"]
        )
    finally:
        runtime.close()
    recycle_control = MolecularDesignControl(
        DesignVariable("fraction", "1", (0.1, 0.4), 0.2), "recycle_fraction"
    )
    alternate = replace(spec, controls=(control, recycle_control)).compile(
        {"heat": 7.0, "fraction": 0.3}
    )
    assert alternate.recycle_fraction == 0.3 and alternate.evaluate().success


def test_explicit_fractional_wait_declaration_preserves_default_duration():
    template, state = design_receiver(two_stage_design())
    default = build_molecular_process_world(template, state, formulas={"A": "Ar", "B": "C3H8"})
    thermal = next(c for c in default.to_dict()["components"] if c["kind"] == "thermal")
    assert thermal["parameters"]["duration_range_s"] == [1, 14400]
    default_runtime = ChemWorldRuntime(composition=default)
    try:
        default_runtime.reset(seed=7)
        rejected = default_runtime.execute({"operation": "wait", "duration_s": 0.5})
        assert rejected.info["transaction_status"] == "validation_failed"
        assert default_runtime._state.ledger.time_s == 0
        assert sum(default_runtime._state.species_amounts.values()) == pytest.approx(0.05)
    finally:
        default_runtime.close()
    world = build_molecular_process_world(
        template, state, formulas={"A": "Ar", "B": "C3H8"}, duration_range_s=(0.5, 14400.0)
    )
    runtime = ChemWorldRuntime(composition=world)
    try:
        runtime.reset(seed=7)
        rejected = runtime.execute({"operation": "wait", "duration_s": 0.25})
        assert rejected.info["transaction_status"] == "validation_failed"
        assert runtime._state.ledger.time_s == 0
        assert sum(runtime._state.species_amounts.values()) == pytest.approx(0.05)
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.5})
        assert outcome.info["transaction_status"] == "committed", outcome.info
        assert runtime._state.ledger.time_s == 0.5
        assert sum(runtime._state.species_amounts.values()) == pytest.approx(0.055)
    finally:
        runtime.close()
    for bounds in (
        (0.0, 1.0),
        (2.0, 1.0),
        (1.0, 1.0),
        (1.0, float("inf")),
        (1e-7, 1.0),
        (1.0, 20000.0),
    ):
        with pytest.raises(ValueError, match="duration bounds"):
            build_molecular_process_world(
                template, state, formulas={"A": "Ar", "B": "C3H8"}, duration_range_s=bounds
            )
