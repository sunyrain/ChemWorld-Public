"""Actual objective design, source-unit binding and original-world nonmutation."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.sbml_design import design_spec, design_world
from chemworld.physchem.bounded_design import DesignConstraint
from chemworld.physchem.sbml_design import SBMLDesignMetric, SBMLDesignSpec, optimize_sbml_design
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_design import analyze_sbml_design_world
from chemworld.runtime.sbml_models import validate_model_state
from reference.sbml_design52_cases import BOUNDARY_CASES, OPTIMIZATION_CASES, inputs
from reference.sbml_design52_reference import optimum


@pytest.mark.parametrize("name", OPTIMIZATION_CASES)
def test_declared_objective_and_constraints_use_original_sbml(name):
    model, current, spec = inputs(name)
    before = dict(current)
    result = optimize_sbml_design(model, current, spec)
    expected = optimum(name)
    assert result["optimization"]["success"], result["optimization"]["failure_reason"]
    assert result["optimization"]["values"] == pytest.approx(expected["controls"], abs=2e-5)
    assert result["objective"]["value"] == pytest.approx(expected["objective"], abs=2e-6)
    assert result["optimization"]["point"]["observations"] == pytest.approx(
        expected["metrics"], abs=2e-6, rel=2e-6
    )
    assert current == before
    assert result["optimization"]["trials"][-1]["fresh_verification"]
    assert not result["world_advanced"] and not result["live_resources_charged"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("name", BOUNDARY_CASES)
def test_real_failures_are_not_penalty_objectives_or_installed_answers(name):
    model, current, spec = inputs(name)
    result = optimize_sbml_design(model, current, spec)
    optimization = result["optimization"]
    assert not optimization["success"]
    assert result["recommended_source_parameters"] is None and result["objective"]["value"] is None
    assert optimization["attempts"] and optimization["trials"]
    assert optimization["failure_reason"]
    if name in {"physical_failure", "steady_seed_failure"}:
        failed = [t for t in optimization["trials"] if not t["success"]]
        assert failed and all("objective" not in t for t in failed)


@pytest.mark.parametrize("scaled", [False, True])
def test_readonly_current_state_design_then_real_set_wait_and_public_measure(scaled):
    world = design_world(scaled=scaled, initial_assignment=True)
    runtime = ChemWorldRuntime(composition=world)
    try:
        runtime.reset(seed=52)
        runtime.execute({"operation": "wait", "duration_s": 0.4})
        before = runtime.snapshot().to_dict()
        model = world.physics.models_of("sbml")["formal_workspace"]
        _, current, _ = validate_model_state(runtime._state, model)
        specification = design_spec("trajectory_min")
        report = analyze_sbml_design_world(runtime._state, world, specification)
        assert report["optimization"]["success"]
        assert runtime.snapshot().to_dict() == before
        expected = optimum("trajectory_min", initial_a=current["N:A"], initial_b=current["N:B"])
        assert report["optimization"]["values"] == pytest.approx(expected["controls"], abs=2e-5)
        candidate = report["optimization"]["point"]["diagnostics"][
            "initial_coordinates_after_control"
        ]
        assert candidate["N:A"] == current["N:A"]
        for action in (
            {
                "operation": "set_model_parameters",
                "values": report["recommended_source_parameters"],
            },
            {"operation": "wait", "duration_s": 2.0},
            {"operation": "measure", "instrument": "formal_quantity_meter"},
        ):
            reply = runtime.execute(action)
            assert reply.info["transaction_status"] == "committed"
        observed = {channel.channel_id: channel for channel in reply.observation.channels}
        assert observed["amount_B"].value == pytest.approx(0.6, abs=2e-6)
        assert runtime._state.ledger.time_s == pytest.approx(2.4)
    finally:
        runtime.close()


def test_json_spec_retains_source_bindings_and_detaches_authored_starts():
    original = design_spec("steady_budget")
    payload = original.to_dict()
    restored = SBMLDesignSpec.from_dict(json.loads(json.dumps(payload)))
    assert restored.to_dict() == payload
    payload["starts"][0]["forward_rate"] = 999
    assert restored.starts[0]["forward_rate"] == 0.3


@pytest.mark.parametrize("kind", ["channel", "unit", "undeclared_control", "duplicate", "offset"])
def test_invalid_bindings_do_not_turn_into_numerical_trials(kind):
    model, current, spec = inputs("steady_max")
    with pytest.raises(ValueError):
        if kind == "channel":
            spec = replace(spec, metrics=(SBMLDesignMetric("product", "missing"),), constraints=())
        elif kind == "unit":
            spec = replace(
                spec, constraints=(DesignConstraint("product", "K", 1.0, 0.0, lower=0.2),)
            )
        elif kind == "undeclared_control":
            spec = replace(spec, controls=(replace(spec.controls[0], source_id="total_rate"),))
        elif kind == "duplicate":
            spec = replace(spec, controls=(spec.controls[0], spec.controls[0]))
        elif kind == "offset":
            spec = replace(
                spec, metrics=(SBMLDesignMetric("product", "amount_B", 1.0),), constraints=()
            )
        optimize_sbml_design(model, current, spec)
