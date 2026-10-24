"""Scoring-contract audit helpers for replayed trajectory records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from chemworld.data.records import trajectory_value
from chemworld.eval.metrics import evaluate_records
from chemworld.tasks import TASK_REGISTRY
from chemworld.world.scoring import TaskScoringContract, task_score_observation


@dataclass(frozen=True)
class ScoringAuditFailure:
    step: int
    field: str
    expected: float | str | None
    actual: float | str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ScoringAuditResult:
    checked_steps: int
    final_assay_count: int
    scoring_contract_hash: str
    max_score_error: float
    max_leaderboard_error: float
    max_processed_metric_error: float
    failures: list[ScoringAuditFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked_steps": self.checked_steps,
            "final_assay_count": self.final_assay_count,
            "scoring_contract_hash": self.scoring_contract_hash,
            "max_score_error": self.max_score_error,
            "max_leaderboard_error": self.max_leaderboard_error,
            "max_processed_metric_error": self.max_processed_metric_error,
            "failures": [failure.to_dict() for failure in self.failures],
            "passed": self.passed,
        }


def _record_task_id(records: list[dict[str, Any]]) -> str:
    first = records[0]
    return str(trajectory_value(first, 'benchmark_task_id', None) or first["task_id"])


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def audit_scoring_contract(
    records: list[dict[str, Any]],
    *,
    tolerance: float = 1.0e-6,
) -> ScoringAuditResult:
    """Check that trajectory scores agree with the task scoring contract."""

    if not records:
        raise ValueError("Cannot audit an empty trajectory")
    task_id = _record_task_id(records)
    task = TASK_REGISTRY[task_id]
    first = records[0]
    contract = TaskScoringContract.from_success_metrics(
        objective=str(first["objective"]),
        success_metrics=task.success_metrics,
    )
    expected_contract_hash = contract.contract_hash
    failures: list[ScoringAuditFailure] = []
    max_score_error = 0.0
    max_leaderboard_error = 0.0
    max_processed_metric_error = 0.0
    final_assay_count = 0
    previous_measurement_score = 0.0

    for record in records:
        step = int(record["step"])
        actual_contract_hash = record.get("scoring_contract_hash")
        if actual_contract_hash != expected_contract_hash:
            failures.append(
                ScoringAuditFailure(
                    step=step,
                    field="scoring_contract_hash",
                    expected=expected_contract_hash,
                    actual=str(actual_contract_hash),
                    message="record scoring contract hash does not match task contract",
                )
            )
        observation = trajectory_value(record, 'observation', {})
        recomputed_score = task_score_observation(contract=contract, values=observation)
        observed_score = _float_or_none(observation.get("score"))
        if observed_score is not None:
            score_error = abs(recomputed_score - observed_score)
            max_score_error = max(max_score_error, score_error)
            if score_error > tolerance:
                failures.append(
                    ScoringAuditFailure(
                        step=step,
                        field="observation.score",
                        expected=recomputed_score,
                        actual=observed_score,
                        message="observation score does not match recomputed contract score",
                    )
                )
        precondition_failed = bool(
            record.get("constraint_flags", {}).get("precondition_failed", False)
        )
        successful_measurement = (
            record.get("operation_type") == "measure" and not precondition_failed
        )
        expected_reward = (
            observed_score - previous_measurement_score
            if successful_measurement and observed_score is not None
            else 0.0
        )
        reward = _float_or_none(trajectory_value(record, 'reward', None))
        observed_reward = _float_or_none(record.get("observed_reward"))
        for reward_field, value in (("reward", reward), ("observed_reward", observed_reward)):
            if value is not None:
                reward_error = abs(value - expected_reward)
                if reward_error > tolerance:
                    failures.append(
                        ScoringAuditFailure(
                            step=step,
                            field=reward_field,
                            expected=expected_reward,
                            actual=value,
                            message=(
                                f"{reward_field} does not match the fresh-measurement "
                                "score delta"
                            ),
                        )
                    )
        if successful_measurement and observed_score is not None:
            previous_measurement_score = observed_score

        is_final_assay = (
            record.get("operation_type") == "measure"
            and record.get("instrument") == "final_assay"
            and not precondition_failed
        )
        leaderboard_score = _float_or_none(trajectory_value(record, 'leaderboard_score', None))
        if is_final_assay:
            final_assay_count += 1
            leaderboard_error = abs((leaderboard_score or 0.0) - recomputed_score)
            max_leaderboard_error = max(max_leaderboard_error, leaderboard_error)
            if leaderboard_score is None or leaderboard_error > tolerance:
                failures.append(
                    ScoringAuditFailure(
                        step=step,
                        field="leaderboard_score",
                        expected=recomputed_score,
                        actual=leaderboard_score,
                        message="final assay leaderboard score does not match recomputed score",
                    )
                )
        elif leaderboard_score is not None:
            failures.append(
                ScoringAuditFailure(
                    step=step,
                    field="leaderboard_score",
                    expected=None,
                    actual=leaderboard_score,
                    message="non-final-assay step must not expose leaderboard score",
                )
            )
        if is_final_assay:
            # Campaign-mode final assays replace the process ledger with a
            # fresh experiment state after the terminal observation is scored.
            previous_measurement_score = 0.0

        processed = record.get("processed_estimate", {})
        if isinstance(processed, dict):
            for key, processed_value in processed.items():
                if key not in observation or observation.get(key) is None:
                    continue
                processed_float = _float_or_none(processed_value)
                observation_float = _float_or_none(observation.get(key))
                if processed_float is None or observation_float is None:
                    continue
                processed_error = abs(processed_float - observation_float)
                max_processed_metric_error = max(
                    max_processed_metric_error,
                    processed_error,
                )
                if processed_error > tolerance:
                    failures.append(
                        ScoringAuditFailure(
                            step=step,
                            field=f"processed_estimate.{key}",
                            expected=observation_float,
                            actual=processed_float,
                            message=(
                                "processed final/instrument metric diverges from "
                                "public observation"
                            ),
                        )
                    )

    evaluation = evaluate_records(records, threshold=task.threshold)
    official_best = max(
        (
            float(trajectory_value(record, 'leaderboard_score'))
            for record in records
            if trajectory_value(record, 'leaderboard_score', None) is not None
        ),
        default=0.0,
    )
    if abs(evaluation.final_best_score - official_best) > tolerance:
        failures.append(
            ScoringAuditFailure(
                step=len(records),
                field="evaluation.final_best_score",
                expected=official_best,
                actual=evaluation.final_best_score,
                message="evaluation final_best_score does not match trajectory leaderboard scores",
            )
        )

    return ScoringAuditResult(
        checked_steps=len(records),
        final_assay_count=final_assay_count,
        scoring_contract_hash=expected_contract_hash,
        max_score_error=max_score_error,
        max_leaderboard_error=max_leaderboard_error,
        max_processed_metric_error=max_processed_metric_error,
        failures=failures,
    )
