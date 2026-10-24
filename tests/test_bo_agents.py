from __future__ import annotations

import math

import pytest

from chemworld.agents.bo import (
    GaussianProcessBOAgent,
    GaussianProcessPIAgent,
    GaussianProcessUCBAgent,
    RandomForestEIAgent,
    SafetyConstrainedBOAgent,
    StructuredGaussianProcessBOAgent,
    StructuredGaussianProcessPIAgent,
    StructuredGaussianProcessUCBAgent,
    StructuredRandomForestEIAgent,
    StructuredSafetyConstrainedBOAgent,
)
from chemworld.agents.task_recipes import TASK_RECIPE_SPACE_VERSION
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.metrics import evaluate_records
from chemworld.eval.runner import run_agent


def test_safe_bo_uses_the_stricter_public_task_limit() -> None:
    agent = SafetyConstrainedBOAgent(risk_threshold=0.65)
    agent.reset({"safety_limit": 0.35}, seed=0)
    assert agent.effective_risk_threshold == 0.35
    assert agent.manifest()["risk_threshold"] == 0.35


def test_surrogate_baselines_smoke(tmp_path) -> None:
    agents = [
        GaussianProcessBOAgent(n_initial=2, n_candidates=16),
        GaussianProcessPIAgent(n_initial=2, n_candidates=16),
        GaussianProcessUCBAgent(n_initial=2, n_candidates=16),
        RandomForestEIAgent(n_initial=2, n_candidates=16, n_estimators=8),
        SafetyConstrainedBOAgent(n_initial=2, n_candidates=16),
        StructuredGaussianProcessBOAgent(n_initial=2, n_candidates=16),
        StructuredGaussianProcessPIAgent(n_initial=2, n_candidates=16),
        StructuredGaussianProcessUCBAgent(n_initial=2, n_candidates=16),
        StructuredRandomForestEIAgent(n_initial=2, n_candidates=16, n_estimators=8),
        StructuredSafetyConstrainedBOAgent(n_initial=2, n_candidates=16),
    ]
    for index, agent in enumerate(agents):
        history = run_agent(
            env_id="ChemWorld",
            agent=agent,
            world_split="public-dev",
            budget=12,
            objective="balanced",
            seed=index,
            output_path=tmp_path / f"{agent.name}.jsonl",
        )
        assert 1 <= len(history) <= 12
        assert history[-1].action == {"operation": "measure", "instrument": "final_assay"}
        assert max(record.reward for record in history) >= 0.0


def test_structured_safe_bo_declares_typed_recipe_encoding() -> None:
    agent = StructuredSafetyConstrainedBOAgent(n_initial=2, n_candidates=16)
    agent.reset(
        {
            "budget": 24,
            "task_id": "electrochemical-conversion",
            "safety_limit": 0.4,
        },
        seed=0,
    )

    manifest = agent.manifest()
    assert manifest["recipe_encoding"] == "continuous_plus_material_one_hot"
    assert manifest["surrogate_family"] == "safe_gaussian_process"


def test_all_formal_acquisition_variants_declare_typed_recipe_encoding() -> None:
    agents = (
        StructuredGaussianProcessBOAgent(),
        StructuredGaussianProcessPIAgent(),
        StructuredGaussianProcessUCBAgent(),
        StructuredRandomForestEIAgent(),
        StructuredSafetyConstrainedBOAgent(),
    )
    for agent in agents:
        manifest = agent.manifest()
        assert manifest["recipe_encoding"] == "continuous_plus_material_one_hot"
        assert manifest["search_space_version"] == TASK_RECIPE_SPACE_VERSION


def test_bo_campaign_runs_multiple_recipes(tmp_path) -> None:
    path = tmp_path / "campaign_bo.jsonl"
    history = run_agent(
        env_id="ChemWorld",
        agent=GaussianProcessBOAgent(n_initial=2, n_candidates=16),
        world_split="public-test",
        budget=72,
        objective="balanced",
        seed=3,
        task_id="reaction-optimization-standard",
        output_path=path,
    )
    records = load_jsonl(path)
    final_assays = [
        record
        for record in records
        if record.get("operation_type") == "measure"
        and record.get("instrument") == "final_assay"
    ]
    assert len(history) > 6
    assert len(final_assays) >= 2
    assert not any(
        record["constraint_flags"].get("precondition_failed", False)
        for record in records
    )
    assert records[0]["episode_mode"] == "campaign"
    assert trajectory_value(records[0], 'benchmark_task_id') == "reaction-optimization-standard"


def test_default_gp_bo_enters_acquisition_under_benchmark_budget(tmp_path) -> None:
    gp_path = tmp_path / "gp_bo_default.jsonl"
    random_path = tmp_path / "random_default.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=GaussianProcessBOAgent(),
        world_split="public-test",
        budget=72,
        objective="balanced",
        seed=0,
        task_id="reaction-optimization-standard",
        output_path=gp_path,
    )
    from chemworld.agents.random import RandomAgent

    run_agent(
        env_id="ChemWorld",
        agent=RandomAgent(),
        world_split="public-test",
        budget=72,
        objective="balanced",
        seed=0,
        task_id="reaction-optimization-standard",
        output_path=random_path,
    )

    gp_result = evaluate_records(load_jsonl(gp_path))
    random_result = evaluate_records(load_jsonl(random_path))
    assert gp_result.bo_initial_recipe_count == 4
    assert gp_result.bo_acquisition_recipe_count >= 1
    assert gp_result.bo_entered_acquisition is True
    assert gp_result.final_assay_count >= 5
    # A single seed cannot establish method superiority. This regression only
    # requires a real acquisition phase and valid scores; paired multi-seed
    # inference belongs to the benchmark validity/power workstream.
    assert math.isfinite(gp_result.total_score)
    assert math.isfinite(random_result.total_score)
    assert gp_result.final_best_score < 0.95


def test_gp_acquisition_variants_enter_model_based_phase(tmp_path) -> None:
    for agent in (
        GaussianProcessPIAgent(n_initial=2, n_candidates=24),
        GaussianProcessUCBAgent(n_initial=2, n_candidates=24),
    ):
        path = tmp_path / f"{agent.name}.jsonl"
        run_agent(
            env_id="ChemWorld",
            agent=agent,
            world_split="public-test",
            budget=24,
            objective="balanced",
            seed=1,
            task_id="reaction-optimization-standard",
            output_path=path,
        )
        records = load_jsonl(path)
        policies = [
            item["selected_policy"]
            for item in records[-1]["agent_trace"]
            if item["phase"] == "acquisition"
        ]
        assert policies
        expected = "probability_improvement" if agent.name == "gp_pi" else "upper_confidence"
        assert all(expected in policy for policy in policies)


def test_surrogate_compute_events_are_nonoverlapping_and_cumulative(tmp_path) -> None:
    path = tmp_path / "instrumented-gp.jsonl"
    agent = StructuredGaussianProcessBOAgent(n_initial=2, n_candidates=24)
    run_agent(
        env_id="ChemWorld",
        agent=agent,
        world_split="public-dev",
        budget=24,
        objective="balanced",
        seed=11_000,
        task_id="partition-discovery",
        output_path=path,
        budget_override=30,
    )
    events = agent.formal_compute_events()
    assert [item["event_kind"] for item in events] == [
        "fit",
        "acquisition_optimization",
    ]
    assert all(item["cpu_time_s"] >= 0.0 and item["wall_time_s"] >= 0.0 for item in events)
    usage = agent.method_resource_usage()
    assert usage["accounting_complete"] is True
    assert usage["cpu_time_s"] == pytest.approx(
        sum(float(item["cpu_time_s"]) for item in events)
    )
