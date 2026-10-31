from __future__ import annotations

from copy import deepcopy

import gymnasium as gym

import chemworld  # noqa: F401
from chemworld.eval.runtime_domain_affordance_audit import (
    _candidate_actions,
    _physical_domain_rejection,
    audit_runtime_domain_affordances,
)
from chemworld.tasks import SERIOUS_TASK_IDS


def test_candidate_actions_cover_public_midpoint_and_field_boundaries() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        env.unwrapped._state = env.unwrapped._state.replace(temperature_K=260.0)
        schema = env.unwrapped.action_schema("cool_crystallize")
        candidates = dict(_candidate_actions(schema))

        assert schema["schema_version"] == "chemworld-public-action-affordance-0.2"
        assert candidates["midpoint"] == {
            "operation": "cool_crystallize",
            "target_temperature_K": 255.0,
            "duration_s": 7200.5,
        }
        assert candidates["target_temperature_K:low"]["target_temperature_K"] == 250.0
        assert candidates["target_temperature_K:high"]["target_temperature_K"] == 260.0
        assert candidates["duration_s:low"]["duration_s"] == 1.0
        assert candidates["duration_s:high"]["duration_s"] == 14_400.0
    finally:
        env.close()


def test_all_serious_task_public_affordances_conform_to_runtime_domains() -> None:
    report = audit_runtime_domain_affordances(source_commit="test-source", seed=0)

    assert report["passed"] is True
    assert report["task_ids"] == list(SERIOUS_TASK_IDS)
    assert report["summary"]["candidate_count"] > 200
    assert report["summary"]["finding_count"] == 0
    assert report["findings"] == []
    assert report["benchmark_claim_allowed"] is False
    assert report["summary"]["physical_rejection_count"] == len(report["physical_rejections"])
    assert report["summary"]["physical_rejection_count"] > 0
    assert report["summary"]["validator_valid_count"] == (
        report["summary"]["runtime_committed_count"]
        + report["summary"]["physical_rejection_count"]
    )


def test_physical_rejection_requires_both_a_proven_limit_and_an_atomic_rollback() -> None:
    with gym.make("ChemWorld", task_id="reaction-to-distillation", budget_override=8) as env:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.0165},
        ):
            assert env.step(action)[-1]["transaction_status"] == "committed"
        before = env.unwrapped._state
        result = env.unwrapped.runtime.apply_transaction(before, {
            "operation": "heat", "target_temperature_K": 360.0,
            "duration_s": 900.0, "stirring_speed_rpm": 650.0,
        })
        info = result.info_payload()
        assert _physical_domain_rejection(before, result.state, info)
        assert not _physical_domain_rejection(
            before, result.state.replace(volume_L=before.volume_L * 1.1), info,
        )
        assert not _physical_domain_rejection(before, result.state, {
            **info, "operation_outcome": "numerical_failure",
        })
        missing_evidence = deepcopy(info)
        missing_evidence["outcome_diagnostics"]["validity_violations"] = []
        assert not _physical_domain_rejection(before, result.state, missing_evidence)
        false_limit = deepcopy(info)
        false_limit["outcome_diagnostics"]["reactor_diagnostic"][
            "maximum_temperature_rate_K_s"
        ] = 1.0
        assert not _physical_domain_rejection(before, result.state, false_limit)
