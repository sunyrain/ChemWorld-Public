"""Executable score provenance, replay, and tamper controls."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.layered_evaluation import TaskEvaluationContract, evaluate_layered_records
from chemworld.eval.metrics import EVALUATION_METRICS_VERSION, evaluate_records
from chemworld.eval.result_artifacts import (
    EVALUATION_RESULT_SCHEMA_VERSION,
    SCORE_REPLAY_BINDING_VERSION,
    build_verified_evaluation_result,
    validate_verified_evaluation_result,
)
from chemworld.eval.runner import run_agent
from chemworld.legacy.agent_registry import make_agent
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.task_design import SERIOUS_TASK_DESIGNS
from chemworld.tasks import SERIOUS_TASK_IDS

SCORE_REPLAY_PROTOCOL_VERSION = "chemworld-score-replay-protocol-0.1"
SCORE_REPLAY_AUDIT_VERSION = "chemworld-score-replay-audit-0.1"
DEFAULT_SCORE_REPLAY_PROTOCOL_PATH = configuration_root() / "benchmark" / "score_replay_vnext.json"


def load_score_replay_protocol(
    path: str | Path = DEFAULT_SCORE_REPLAY_PROTOCOL_PATH,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("score/replay protocol must be a JSON object")
    return payload


def audit_score_replay_protocol(
    protocol: dict[str, Any],
    *,
    workspace: str | Path,
) -> dict[str, Any]:
    """Run static contract checks and adversarial probes against a real trajectory."""

    configured_tasks = protocol.get("tasks", {})
    policies = protocol.get("policies", {})
    checks = {
        "schema": protocol.get("schema_version") == SCORE_REPLAY_PROTOCOL_VERSION,
        "result_schema": protocol.get("result_schema_version") == EVALUATION_RESULT_SCHEMA_VERSION,
        "binding_schema": protocol.get("binding_schema_version") == SCORE_REPLAY_BINDING_VERSION,
        "task_scope": tuple(configured_tasks) == tuple(SERIOUS_TASK_IDS),
        "primary_metrics": all(
            configured_tasks.get(task_id, {}).get("primary_metric")
            == SERIOUS_TASK_DESIGNS[task_id].primary_metric
            for task_id in SERIOUS_TASK_IDS
        ),
        "directions_explicit": all(
            configured_tasks.get(task_id, {}).get("direction") == "maximize"
            for task_id in SERIOUS_TASK_IDS
        ),
        "exact_source_binding": policies.get("source_binding") == "exact_trajectory_bytes_sha256",
        "replay_required": policies.get("deterministic_replay") == "required",
        "trajectory_is_metric_source": policies.get("leaderboard_metric_source")
        == "recomputed_from_trajectory",
        "nonfinite_fails_closed": policies.get("nonfinite_numeric") == "fail",
        "missing_primary_fails_closed": policies.get("missing_primary") == "fail",
        "online_reward_excluded": policies.get("online_reward_is_primary") is False,
        "metrics_versioned": EVALUATION_METRICS_VERSION == "chemworld-evaluation-metrics-0.3",
    }
    probes = _run_adversarial_probes(Path(workspace))
    required_probes = tuple(protocol.get("required_adversarial_probes", ()))
    checks["required_probes_declared"] = tuple(probes) == required_probes
    controls_ready = all(checks.values()) and all(probes.values())
    return {
        "schema_version": SCORE_REPLAY_AUDIT_VERSION,
        "protocol_id": protocol.get("protocol_id"),
        "status": "passed" if controls_ready else "failed",
        "controls_ready": controls_ready,
        "checks": checks,
        "adversarial_probes": probes,
        "task_contracts": {
            task_id: {
                "primary_metric": configured_tasks[task_id]["primary_metric"],
                "direction": configured_tasks[task_id]["direction"],
                "missing_primary_policy": "fail",
            }
            for task_id in SERIOUS_TASK_IDS
        },
        "source_digests": _source_digests(),

    }


def _run_adversarial_probes(workspace: Path) -> dict[str, bool]:
    workspace.mkdir(parents=True, exist_ok=True)
    trajectory = workspace / "score-replay-probe.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent("scripted_chemistry"),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=trajectory,
    )
    records = load_jsonl(trajectory)
    result = build_verified_evaluation_result(records, trajectory_path=trajectory)
    validate_verified_evaluation_result(result, replay=True)

    probes = {
        "record_source_mismatch": _record_source_mismatch_rejected(records, trajectory),
        "trajectory_byte_tamper": _trajectory_byte_tamper_rejected(result, trajectory),
        "reported_metric_tamper": _metric_tamper_rejected(result),
        "binding_tamper": _binding_tamper_rejected(result),
        "nonfinite_objective": _nonfinite_objective_rejected(records),
        "missing_terminal_primary": _missing_primary_rejected(),
        "online_reward_decoy": _online_reward_decoy_is_excluded(records),
    }
    return probes


def _record_source_mismatch_rejected(records: list[dict[str, Any]], trajectory: Path) -> bool:
    changed = copy.deepcopy(records)
    changed_reward = float(trajectory_value(changed[0], "reward", 0.0)) + 0.01
    if "reward" in changed[0]:
        changed[0]["reward"] = changed_reward
    if isinstance(changed[0].get("evaluation_outcome"), dict):
        # Keep the v0.2 compatibility alias internally coherent so this probe
        # reaches the exact source-byte binding rather than the earlier schema
        # consistency guard.
        changed[0]["evaluation_outcome"]["online_transition_reward"] = changed_reward
    try:
        build_verified_evaluation_result(changed, trajectory_path=trajectory)
    except ValueError as exc:
        return "bound trajectory source" in str(exc)
    return False


def _trajectory_byte_tamper_rejected(result: dict[str, Any], trajectory: Path) -> bool:
    original = trajectory.read_bytes()
    try:
        trajectory.write_bytes(original + b"\n")
        try:
            validate_verified_evaluation_result(result, replay=True)
        except ValueError as exc:
            return "digest" in str(exc)
        return False
    finally:
        trajectory.write_bytes(original)


def _metric_tamper_rejected(result: dict[str, Any]) -> bool:
    changed = copy.deepcopy(result)
    changed["total_score"] = 1.0 - float(changed["total_score"])
    try:
        validate_verified_evaluation_result(changed, replay=True)
    except ValueError as exc:
        return "does not match trajectory evaluation" in str(exc)
    return False


def _binding_tamper_rejected(result: dict[str, Any]) -> bool:
    changed = copy.deepcopy(result)
    changed["score_replay"]["trajectory_sha256"] = "0" * 64
    try:
        validate_verified_evaluation_result(changed, replay=False)
    except ValueError as exc:
        return "binding does not match" in str(exc)
    return False


def _nonfinite_objective_rejected(records: list[dict[str, Any]]) -> bool:
    changed = copy.deepcopy(records)
    terminal = next(
        record for record in reversed(changed)
        if trajectory_value(record, "leaderboard_score", None) is not None
    )
    if isinstance(terminal.get("evaluation_outcome"), dict):
        terminal["evaluation_outcome"]["leaderboard_score"] = math.nan
        # This probe tests the evaluator's finite-value guard independently of
        # duplicate-alias consistency, which has its own validation tests.
        terminal.pop("leaderboard_score", None)
    else:
        terminal["leaderboard_score"] = math.nan
    try:
        evaluate_records(changed)
    except ValueError as exc:
        return "leaderboard_score must be finite" in str(exc)
    return False


def _missing_primary_rejected() -> bool:
    contract = TaskEvaluationContract.for_task("partition-discovery")
    records = [
        {
            "leaderboard_score": 0.5,
            "observation": {
                "product_in_organic": None,
                "cost": 0.2,
                "safety_risk": 0.1,
            },
            "reward": 0.9,
            "measurement_cost": 0.03,
            "constraint_flags": {},
        }
    ]
    try:
        evaluate_layered_records(records, contract=contract)
    except ValueError as exc:
        return "product_in_organic" in str(exc)
    return False


def _online_reward_decoy_is_excluded(records: list[dict[str, Any]]) -> bool:
    reference = evaluate_records(records)
    changed = copy.deepcopy(records)
    for record in changed:
        if isinstance(record.get("evaluation_outcome"), dict):
            record["evaluation_outcome"]["online_transition_reward"] = 1.0
        if "reward" in record:
            record["reward"] = 1.0
        record["observed_reward"] = 1.0
    decoy = evaluate_records(changed)
    return (
        decoy.final_best_score == reference.final_best_score
        and decoy.total_score == reference.total_score
    )


def _source_digests() -> dict[str, str]:
    from chemworld.eval import layered_evaluation, metrics, result_artifacts, result_summary

    modules = {
        "metrics": metrics,
        "result_artifacts": result_artifacts,
        "leaderboard": result_summary,
        "layered_evaluation": layered_evaluation,
    }
    return {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for name, module in modules.items()}


__all__ = [
    "DEFAULT_SCORE_REPLAY_PROTOCOL_PATH",
    "SCORE_REPLAY_AUDIT_VERSION",
    "SCORE_REPLAY_PROTOCOL_VERSION",
    "audit_score_replay_protocol",
    "load_score_replay_protocol",
]
