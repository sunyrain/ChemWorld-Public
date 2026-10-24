"""Verified evaluation-result artifacts and trajectory integrity checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.data.validation import validate_records
from chemworld.eval.layered_evaluation import TaskEvaluationContract, evaluate_layered_records
from chemworld.eval.metrics import EVALUATION_METRICS_VERSION, evaluate_records
from chemworld.eval.risk_policy import RiskCostTaskPolicy
from chemworld.eval.verify import verify_records
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.tasks import SERIOUS_TASK_IDS

EVALUATION_RESULT_SCHEMA_VERSION = "chemworld-evaluation-result-0.3"
V02_EVALUATION_RESULT_SCHEMA_VERSION = "chemworld-evaluation-result-0.2"
SCORE_REPLAY_BINDING_VERSION = "chemworld-score-replay-binding-0.1"


def trajectory_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_verified_evaluation_result(
    records: list[dict[str, Any]],
    *,
    trajectory_path: str | Path,
    threshold: float = 0.75,
    world_interventions: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
    risk_limit_override: float | None = None,
) -> dict[str, Any]:
    """Replay a trajectory and return an integrity-bound evaluation result."""

    path = Path(trajectory_path).resolve()
    validate_records(records)
    persisted_records = load_jsonl(path)
    if persisted_records != records:
        raise ValueError("Evaluation records do not match the bound trajectory source")
    verification = verify_records(
        records,
        world_interventions=world_interventions,
    ).to_dict()
    if not bool(verification["verified"]):
        raise ValueError(
            "Trajectory replay verification failed; refusing to produce an evaluation result"
        )
    digest = trajectory_sha256(path)
    result = evaluate_records(records, threshold=threshold).to_dict()
    result.update(
        {
            "result_schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
            "verified": True,
            "verification": verification,
            "evaluation_threshold": float(threshold),
            "trajectory_path": str(path),
            "trajectory_sha256": digest,
            "score_replay": _score_replay_payload(
                records,
                threshold=threshold,
                trajectory_digest=digest,
                risk_limit_override=risk_limit_override,
            ),
        }
    )
    return result


def validate_verified_evaluation_result(
    result: dict[str, Any],
    *,
    replay: bool = False,
    world_interventions: tuple[dict[str, Any], ...] | list[dict[str, Any]] | None = None,
) -> None:
    """Validate result metadata, trajectory digest, and optionally replay it."""

    schema_version = result.get("result_schema_version")
    if schema_version not in {
        EVALUATION_RESULT_SCHEMA_VERSION,
        V02_EVALUATION_RESULT_SCHEMA_VERSION,
    }:
        raise ValueError("Leaderboard input must use a supported verified evaluation-result schema")
    if result.get("verified") is not True:
        raise ValueError("Leaderboard input is not replay verified")
    verification = result.get("verification")
    if not isinstance(verification, dict) or verification.get("verified") is not True:
        raise ValueError("Leaderboard input is missing a successful verification report")
    trajectory_value = result.get("trajectory_path")
    expected_digest = result.get("trajectory_sha256")
    if not isinstance(trajectory_value, str) or not trajectory_value:
        raise ValueError("Leaderboard input is missing trajectory_path")
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ValueError("Leaderboard input is missing a valid trajectory_sha256")
    trajectory = Path(trajectory_value)
    if not trajectory.is_file():
        raise ValueError(f"Leaderboard trajectory does not exist: {trajectory}")
    actual_digest = trajectory_sha256(trajectory)
    if actual_digest != expected_digest:
        raise ValueError("Leaderboard trajectory digest does not match evaluation result")
    if schema_version == EVALUATION_RESULT_SCHEMA_VERSION:
        binding = result.get("score_replay")
        if not isinstance(binding, dict):
            raise ValueError("Leaderboard input is missing the score/replay binding")
        if binding.get("schema_version") != SCORE_REPLAY_BINDING_VERSION:
            raise ValueError("Leaderboard input has an unsupported score/replay binding")
        if binding.get("trajectory_sha256") != actual_digest:
            raise ValueError("Score/replay binding does not match trajectory digest")
    if replay:
        records = load_jsonl(trajectory)
        replay_result = verify_records(
            records,
            world_interventions=world_interventions,
        )
        if not replay_result.verified:
            raise ValueError("Leaderboard trajectory no longer passes replay verification")
        threshold = result.get("evaluation_threshold")
        if not isinstance(threshold, int | float) or isinstance(threshold, bool):
            raise ValueError("Leaderboard input is missing evaluation_threshold")
        recomputed = evaluate_records(records, threshold=float(threshold)).to_dict()
        for key, expected_value in recomputed.items():
            actual_value = result.get(key)
            if isinstance(expected_value, float) and isinstance(actual_value, int | float):
                matches = math.isclose(
                    float(actual_value),
                    expected_value,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            else:
                matches = actual_value == expected_value
            if not matches:
                raise ValueError(f"Leaderboard metric {key!r} does not match trajectory evaluation")
        if schema_version == EVALUATION_RESULT_SCHEMA_VERSION:
            current_binding = result.get("score_replay")
            replay_risk_limit: float | None = None
            if (
                isinstance(current_binding, dict)
                and current_binding.get("risk_limit_source") == "bound_override"
            ):
                contract_payload = current_binding.get("task_evaluation_contract")
                if not isinstance(contract_payload, dict):
                    raise ValueError("Bound risk override is missing its evaluation contract")
                supplied_limit = contract_payload.get("safety_limit")
                if (
                    isinstance(supplied_limit, bool)
                    or not isinstance(supplied_limit, int | float)
                ):
                    raise ValueError("Bound risk override is invalid")
                replay_risk_limit = float(supplied_limit)
            expected_binding = _score_replay_payload(
                records,
                threshold=float(threshold),
                trajectory_digest=actual_digest,
                risk_limit_override=replay_risk_limit,
            )
            if result.get("score_replay") != expected_binding:
                raise ValueError(
                    "Leaderboard score/replay binding does not match trajectory evaluation"
                )


def _score_replay_payload(
    records: list[dict[str, Any]],
    *,
    threshold: float,
    trajectory_digest: str,
    risk_limit_override: float | None = None,
) -> dict[str, Any]:
    first = records[0]
    task_id = str(trajectory_value(first, 'benchmark_task_id', None) or "")
    payload: dict[str, Any] = {
        "schema_version": SCORE_REPLAY_BINDING_VERSION,
        "evaluation_metrics_version": EVALUATION_METRICS_VERSION,
        "metric_source": "exact_trajectory_replay",
        "trajectory_sha256": trajectory_digest,
        "source_record_count": len(records),
        "evaluation_threshold": float(threshold),
        "nonfinite_numeric_policy": "fail",
        "online_reward_is_primary": False,
    }
    if task_id not in SERIOUS_TASK_IDS:
        payload["layered_evaluation"] = None
        payload["task_evaluation_contract"] = None
        return payload

    if risk_limit_override is None:
        risk_protocol = json.loads(
            (configuration_root() / "benchmark" / "risk_cost_vnext.json").read_text(
                encoding="utf-8"
            )
        )
        risk_limit = RiskCostTaskPolicy.from_protocol(task_id, risk_protocol).risk_limit
        risk_limit_source = "risk_cost_vnext"
    else:
        risk_limit = float(risk_limit_override)
        if not math.isfinite(risk_limit) or not 0.0 < risk_limit < 1.0:
            raise ValueError("risk_limit_override must be finite and in (0, 1)")
        risk_limit_source = "bound_override"
    contract = TaskEvaluationContract.for_task(task_id, risk_limit=risk_limit)
    payload["risk_limit_source"] = risk_limit_source
    payload["task_evaluation_contract"] = contract.to_dict()
    payload["layered_evaluation"] = evaluate_layered_records(records, contract=contract)
    return payload


__all__ = [
    "EVALUATION_RESULT_SCHEMA_VERSION",
    "SCORE_REPLAY_BINDING_VERSION",
    "V02_EVALUATION_RESULT_SCHEMA_VERSION",
    "build_verified_evaluation_result",
    "trajectory_sha256",
    "validate_verified_evaluation_result",
]
