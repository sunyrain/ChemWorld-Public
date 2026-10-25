"""Outcome-blind evaluator plans for Work II final recommendations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.tasks import get_task

BLIND_EVALUATOR_VERSION = "chemworld-work-ii-blind-evaluator-plan-0.1"
BLIND_EVALUATION_REPORT_VERSION = "chemworld-work-ii-blind-evaluation-report-0.1"

_BLIND_ISOLATION_CONTRACT = {
    "paired_noise_within_replicate": True,
    "participant_feedback_from_blind_evaluator": False,
    "evaluator_provider_calls": 0,
    "evaluator_trajectory_separate_from_participant": True,
    "evaluator_resources_excluded_from_participant_ledger": True,
}


def effective_blind_evaluator_contract(
    cell: Mapping[str, Any],
    global_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Derive one cell's blind denominator while preserving global isolation.

    C2 blocks have different participant experiment denominators (8/10/12), so
    those values cannot be inherited from one A_E-shaped global contract.  The
    cell is authoritative for all counts; the global contract remains
    authoritative for the fixed target identities and evaluator isolation.
    """

    for field, expected in _BLIND_ISOLATION_CONTRACT.items():
        if field not in global_contract or global_contract.get(field) != expected:
            raise ValueError(f"blind evaluator global isolation invariant failed: {field}")
    targets = global_contract.get("blind_targets_per_cell")
    if targets != ["observed_incumbent", "participant_final_recommendation"]:
        raise ValueError("blind evaluator global target identities drifted")

    cell_fields = (
        "complete_experiment_count",
        "participant_final_recommendation_count",
        "blind_validation_target_count",
        "blind_replicates_per_target",
        "blind_validation_execution_count",
    )
    counts: dict[str, int] = {}
    for field in cell_fields:
        value = cell.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"formal cell has an invalid or missing {field}")
        counts[field] = value
    if counts["participant_final_recommendation_count"] != 1:
        raise ValueError("blind evaluator requires exactly one final recommendation")
    if counts["blind_validation_target_count"] != len(targets):
        raise ValueError("blind evaluator cell target count differs from global targets")
    if counts["blind_replicates_per_target"] != 3:
        raise ValueError("blind evaluator requires three paired replicates per target")
    if (
        counts["blind_validation_target_count"]
        * counts["blind_replicates_per_target"]
        != counts["blind_validation_execution_count"]
    ):
        raise ValueError("blind evaluator cell target/replicate/execution counts disagree")

    effective = dict(global_contract)
    experiment_count = counts["complete_experiment_count"]
    effective.update(
        {
            "participant_complete_experiments_per_cell": experiment_count,
            "candidate_experiment_indices": list(range(1, experiment_count + 1)),
            "participant_final_recommendations_per_cell": counts[
                "participant_final_recommendation_count"
            ],
            "blind_targets_per_cell": list(targets),
            "blind_replicates_per_target": counts["blind_replicates_per_target"],
            "blind_validation_execution_count": counts[
                "blind_validation_execution_count"
            ],
        }
    )
    return effective


def blind_execution_directory_name(execution: Mapping[str, Any]) -> str:
    """Return a short deterministic directory name while receipts keep the full ID."""

    index = execution.get("execution_index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("blind evaluator execution index is invalid")
    execution_id = str(execution.get("execution_id", ""))
    if not execution_id:
        raise ValueError("blind evaluator execution ID is missing")
    digest = hashlib.sha256(execution_id.encode()).hexdigest()[:16]
    return f"{index:02d}-{digest}"


class _FrozenBlindReplayAgent(BaseAgent):
    name = "work_ii_frozen_blind_replay"

    def __init__(self, actions: list[Mapping[str, Any]]) -> None:
        self._frozen_actions = [deepcopy(dict(action)) for action in actions]

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._index = 0
        self._pending: dict[str, Any] | None = None

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if self._pending is not None:
            raise RuntimeError("blind replay requested a new action before receiving its outcome")
        if self._index >= len(self._frozen_actions):
            raise RuntimeError("blind replay exhausted its frozen action plan before termination")
        self._pending = deepcopy(self._frozen_actions[self._index])
        self._index += 1
        return deepcopy(self._pending)

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        del observation, reward, info
        if self._pending is None or action != self._pending:
            raise RuntimeError("blind replay outcome does not match its frozen action")
        self._pending = None

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "requires_online_model": False,
                "execution_role": "evaluator_blind_replay",
                "participant_feedback": False,
                "frozen_action_count": len(self._frozen_actions),
                "frozen_action_plan_sha256": canonical_json_sha256(self._frozen_actions),
            }
        )
        return manifest


def _candidate_experiment_indices(contract: Mapping[str, Any]) -> list[int]:
    raw = contract.get("candidate_experiment_indices", [1, 2, 3, 4])
    if not isinstance(raw, list) or not raw:
        raise ValueError("blind evaluator candidate experiment indices are invalid")
    indices: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("blind evaluator candidate experiment indices are invalid")
        indices.append(value)
    if indices != list(range(1, len(indices) + 1)):
        raise ValueError("blind evaluator candidate experiments must be consecutive and 1-based")
    declared = contract.get("participant_complete_experiments_per_cell")
    if declared is not None and (
        isinstance(declared, bool)
        or not isinstance(declared, int)
        or declared != len(indices)
    ):
        raise ValueError("blind evaluator participant experiment denominator drifted")
    return indices


def _experiment_rows(
    summary: Mapping[str, Any],
    expected_indices: list[int],
) -> dict[int, dict[str, Any]]:
    analysis = summary.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("formal cell summary lacks analysis")
    rows = analysis.get("experiments")
    if not isinstance(rows, list):
        raise ValueError("formal cell summary lacks experiment rows")
    experiments: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("formal cell experiment row is malformed")
        index = row.get("experiment_index")
        score = row.get("leaderboard_score")
        actions = row.get("operations")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not isinstance(actions, list)
            or not actions
            or not all(isinstance(action, Mapping) for action in actions)
        ):
            raise ValueError("formal cell experiment is not blind-replayable")
        if index in experiments:
            raise ValueError("formal cell experiment indices are not unique")
        final_action = actions[-1]
        if final_action.get("operation") != "measure" or final_action.get("instrument") != (
            "final_assay"
        ):
            raise ValueError("blind-replayable experiment must end in final_assay")
        experiments[index] = {
            "experiment_index": index,
            "leaderboard_score": float(score),
            "operations": [deepcopy(dict(action)) for action in actions],
        }
    if set(experiments) != set(expected_indices):
        raise ValueError("blind evaluator requires the configured completed experiments")
    return experiments


def _paired_noise(cell_key_sha256: str, replicate_index: int) -> dict[str, Any]:
    payload = f"work-ii-blind-v0.1:{cell_key_sha256}:{replicate_index}".encode()
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "paired_noise_id_sha256": digest,
        "observation_seed": int(digest[:8], 16) % 2_147_483_647,
        "observation_noise_namespace": (
            f"work-ii-blind-v0.1-{cell_key_sha256[:16]}-rep-{replicate_index:02d}"
        ),
    }


def build_blind_evaluation_plan(
    cell: Mapping[str, Any],
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    allow_unqualified_terminal_trajectory: bool = False,
) -> dict[str, Any]:
    """Bind a qualified cell's committed choice to six evaluator-owned replays."""

    expected_indices = _candidate_experiment_indices(contract)
    qualification_passed = summary.get("completed") is True
    development_override = not qualification_passed
    if development_override:
        analysis = summary.get("analysis")
        replay = summary.get("exact_replay")
        terminal_scientific_trajectory = (
            isinstance(analysis, Mapping)
            and int(analysis.get("complete_experiment_count", 0)) == len(expected_indices)
            and analysis.get("right_censored_open_experiment") is False
            and isinstance(replay, Mapping)
            and replay.get("verified") is True
        )
        if not allow_unqualified_terminal_trajectory:
            raise ValueError("blind evaluator plan requires a qualified completed cell")
        if summary.get("formal_result") is True:
            raise ValueError("formal blind evaluation forbids an unqualified trajectory override")
        if not terminal_scientific_trajectory:
            raise ValueError("development blind override requires a terminal scientific trajectory")
    if int(contract.get("participant_final_recommendations_per_cell", -1)) != 1:
        raise ValueError("blind evaluator final-recommendation denominator drifted")
    targets = contract.get("blind_targets_per_cell")
    if targets != ["observed_incumbent", "participant_final_recommendation"]:
        raise ValueError("blind evaluator target contract drifted")
    replicates = int(contract.get("blind_replicates_per_target", -1))
    if replicates != 3:
        raise ValueError("blind evaluator replicate contract drifted")
    if (
        contract.get("paired_noise_within_replicate") is not True
        or contract.get("participant_feedback_from_blind_evaluator") is not False
        or int(contract.get("evaluator_provider_calls", -1)) != 0
        or contract.get("evaluator_trajectory_separate_from_participant") is not True
        or contract.get("evaluator_resources_excluded_from_participant_ledger") is not True
    ):
        raise ValueError("blind evaluator isolation contract drifted")

    experiments = _experiment_rows(summary, expected_indices)
    analysis = summary["analysis"]
    recommendation = analysis.get("final_recommendation")
    if not isinstance(recommendation, Mapping):
        raise ValueError("formal cell lacks a committed final recommendation")
    recommendation_digest = canonical_json_sha256(recommendation)
    if recommendation_digest != analysis.get("final_recommendation_sha256"):
        raise ValueError("final recommendation digest differs from participant receipt")
    selected_index = recommendation.get("selected_experiment_index")
    if isinstance(selected_index, bool) or selected_index not in experiments:
        raise ValueError("final recommendation does not select a completed experiment")
    incumbent_index = min(
        experiments,
        key=lambda index: (-experiments[index]["leaderboard_score"], index),
    )
    if incumbent_index != analysis.get("observed_incumbent_experiment_index"):
        raise ValueError("observed incumbent differs from the frozen tie rule")

    execution_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    target_indices = {
        "observed_incumbent": incumbent_index,
        "participant_final_recommendation": int(selected_index),
    }
    for target in targets:
        experiment = experiments[target_indices[target]]
        action_plan = experiment["operations"]
        action_plan_sha256 = canonical_json_sha256(action_plan)
        target_rows.append(
            {
                "target": target,
                "source_experiment_index": experiment["experiment_index"],
                "participant_observed_leaderboard_score": experiment["leaderboard_score"],
                "action_plan": action_plan,
                "action_plan_sha256": action_plan_sha256,
            }
        )
        for replicate_index in range(1, replicates + 1):
            execution_rows.append(
                {
                    "execution_index": len(execution_rows) + 1,
                    "execution_id": (
                        f"{cell['cell_id']}--blind-{target}--rep-{replicate_index:02d}"
                    ),
                    "target": target,
                    "replicate_index": replicate_index,
                    "source_experiment_index": experiment["experiment_index"],
                    "action_plan_sha256": action_plan_sha256,
                    **_paired_noise(str(cell["cell_key_sha256"]), replicate_index),
                }
            )
    plan: dict[str, Any] = {
        "schema_version": BLIND_EVALUATOR_VERSION,
        "formal_result": summary.get("formal_result") is True,
        "formal_preflight_sha256": (
            summary.get("formal_preflight_sha256") if summary.get("formal_result") is True else None
        ),
        "cell_id": cell["cell_id"],
        "cell_key_sha256": cell["cell_key_sha256"],
        "task_id": cell["task_id"],
        "world_seed": cell["world_seed"],
        "recommendation_sha256": recommendation_digest,
        "participant_final_recommendation_count": 1,
        "participant_complete_experiment_count": len(expected_indices),
        "candidate_experiment_indices": expected_indices,
        "participant_operational_qualification_passed": qualification_passed,
        "development_terminal_trajectory_override": development_override,
        "blind_target_count": len(target_rows),
        "blind_execution_count": len(execution_rows),
        "evaluator_provider_call_count": 0,
        "participant_operation_denominator_impact": 0,
        "participant_feedback_allowed": False,
        "targets": target_rows,
        "executions": execution_rows,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return plan


def validate_blind_evaluation_plan(plan: Mapping[str, Any]) -> list[str]:
    """Check plan self-binding, paired noise and frozen denominators."""

    errors: list[str] = []
    if plan.get("schema_version") != BLIND_EVALUATOR_VERSION:
        errors.append("unexpected blind evaluator plan schema")
    qualification_passed = plan.get("participant_operational_qualification_passed")
    development_override = plan.get("development_terminal_trajectory_override")
    if not isinstance(qualification_passed, bool) or not isinstance(development_override, bool):
        errors.append("blind evaluator participant qualification binding is malformed")
    elif qualification_passed == development_override:
        errors.append("blind evaluator participant qualification binding is inconsistent")
    if plan.get("formal_result") is True:
        digest = plan.get("formal_preflight_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("formal blind evaluator plan lacks its preflight binding")
        if development_override is True:
            errors.append("formal blind evaluator plan uses a development override")
    elif plan.get("formal_preflight_sha256") is not None:
        errors.append("development blind evaluator plan carries a formal preflight binding")
    expected_hash = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected_hash:
        errors.append("blind evaluator plan self-hash mismatch")
    try:
        candidate_indices = _candidate_experiment_indices(plan)
    except ValueError as error:
        errors.append(str(error))
        candidate_indices = []
    if plan.get("participant_complete_experiment_count") not in (
        None,
        len(candidate_indices),
    ):
        errors.append("blind evaluator participant experiment denominator mismatch")
    targets = plan.get("targets")
    executions = plan.get("executions")
    if not isinstance(targets, list) or len(targets) != 2:
        errors.append("blind evaluator plan does not contain two targets")
        targets = []
    if not isinstance(executions, list) or len(executions) != 6:
        errors.append("blind evaluator plan does not contain six executions")
        executions = []
    target_digests = {
        str(target.get("target")): target.get("action_plan_sha256")
        for target in targets
        if isinstance(target, Mapping)
    }
    for execution in executions:
        if not isinstance(execution, Mapping):
            errors.append("blind evaluator execution row is malformed")
            continue
        if target_digests.get(str(execution.get("target"))) != execution.get("action_plan_sha256"):
            errors.append("blind evaluator action plan binding mismatch")
    for replicate_index in range(1, 4):
        rows = [
            row
            for row in executions
            if isinstance(row, Mapping) and row.get("replicate_index") == replicate_index
        ]
        paired_ids = {row.get("paired_noise_id_sha256") for row in rows}
        seeds = {row.get("observation_seed") for row in rows}
        namespaces = {row.get("observation_noise_namespace") for row in rows}
        if len(rows) != 2 or len(paired_ids) != 1 or len(seeds) != 1 or len(namespaces) != 1:
            errors.append(f"blind evaluator replicate {replicate_index} is not paired")
    if (
        plan.get("evaluator_provider_call_count") != 0
        or plan.get("participant_operation_denominator_impact") != 0
        or plan.get("participant_feedback_allowed") is not False
    ):
        errors.append("blind evaluator isolation invariant failed")
    return errors


def execute_blind_evaluation_plan(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Execute all six evaluator-only replays once, retaining every failure."""

    plan_errors = validate_blind_evaluation_plan(plan)
    if plan_errors:
        raise ValueError("invalid blind evaluator plan: " + "; ".join(plan_errors))
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite blind evaluator output: {output_root}")
    output_root.mkdir(parents=True)
    write_json_atomic(output_root / "plan.json", dict(plan))
    targets = {
        str(target["target"]): dict(target)
        for target in plan["targets"]
        if isinstance(target, Mapping)
    }
    receipts: list[dict[str, Any]] = []
    for execution in plan["executions"]:
        if not isinstance(execution, Mapping):
            raise ValueError("blind evaluator execution row is malformed")
        execution_id = str(execution["execution_id"])
        execution_root = output_root / "executions" / blind_execution_directory_name(execution)
        execution_root.mkdir(parents=True, exist_ok=False)
        target = targets[str(execution["target"])]
        actions = target["action_plan"]
        if canonical_json_sha256(actions) != execution["action_plan_sha256"]:
            raise ValueError("blind evaluator execution action binding drifted")
        trajectory_path = execution_root / "trajectory.jsonl"
        receipt: dict[str, Any] = {
            "schema_version": BLIND_EVALUATION_REPORT_VERSION,
            "execution_id": execution_id,
            "target": execution["target"],
            "replicate_index": execution["replicate_index"],
            "paired_noise_id_sha256": execution["paired_noise_id_sha256"],
            "action_plan_sha256": execution["action_plan_sha256"],
            "evaluator_provider_call_count": 0,
            "participant_operation_denominator_impact": 0,
            "participant_feedback_emitted": False,
        }
        try:
            run_agent(
                env_id=get_task(str(config["task_id"])).env_id,
                agent=_FrozenBlindReplayAgent(actions),
                world_split=str(config["world_split"]),
                budget=len(actions),
                objective=str(config["objective"]),
                seed=int(plan["world_seed"]),
                agent_seed=0,
                observation_seed=int(execution["observation_seed"]),
                task_id=str(config["task_id"]),
                output_path=trajectory_path,
                budget_override=len(actions),
                episode_mode_override="single_experiment",
                world_interventions=config.get("world_interventions"),
                electrochemical_material_family_id=config.get("electrochemical_material_family_id"),
                crystallization_material_family_id=config.get("crystallization_material_family_id"),
                electrochemical_workflow_mode=config.get("electrochemical_workflow_mode"),
                scoring_contract_id=config.get("scoring_contract_id"),
                observation_noise_mode=str(config["observation_noise_mode"]),
                observation_noise_namespace=str(execution["observation_noise_namespace"]),
            )
            records = load_jsonl(trajectory_path)
            observed_actions = [record.get("action") for record in records]
            if observed_actions != actions:
                raise ValueError("blind evaluator trajectory differs from its frozen action plan")
            final_rows = [
                record
                for record in records
                if record.get("transaction_status") == "committed"
                and record.get("operation_type") == "measure"
                and record.get("instrument") == "final_assay"
            ]
            if len(final_rows) != 1:
                raise ValueError("blind evaluator trajectory lacks exactly one final assay")
            score = trajectory_value(final_rows[0], "leaderboard_score", None)
            if (
                isinstance(score, bool)
                or not isinstance(score, int | float)
                or not math.isfinite(float(score))
            ):
                raise ValueError("blind evaluator final score is not finite")
            replay = verify_records(
                records,
                tolerance=0.0,
                world_interventions=config.get("world_interventions"),
            ).to_dict()
            if replay.get("verified") is not True:
                raise ValueError("blind evaluator trajectory does not replay exactly")
            receipt.update(
                {
                    "status": "completed",
                    "leaderboard_score": float(score),
                    "operation_attempt_count": len(records),
                    "trajectory": {
                        "path": trajectory_path.relative_to(output_root).as_posix(),
                        "sha256": file_sha256(trajectory_path),
                    },
                    "exact_replay": replay,
                }
            )
        except Exception as error:  # retain failed evaluator denominator without replacement
            receipt.update(
                {
                    "status": "failed",
                    "failure_type": type(error).__name__,
                    "failure_message": str(error),
                    "leaderboard_score": None,
                    "trajectory": (
                        {
                            "path": trajectory_path.relative_to(output_root).as_posix(),
                            "sha256": file_sha256(trajectory_path),
                        }
                        if trajectory_path.is_file()
                        else None
                    ),
                }
            )
        receipt["receipt_sha256"] = canonical_json_sha256(receipt)
        write_json_atomic(execution_root / "receipt.json", receipt)
        receipts.append(receipt)
    completed = [receipt for receipt in receipts if receipt["status"] == "completed"]
    target_scores = {
        target: [
            float(receipt["leaderboard_score"])
            for receipt in completed
            if receipt["target"] == target
        ]
        for target in targets
    }
    target_means = {
        target: (sum(scores) / len(scores) if len(scores) == 3 else None)
        for target, scores in target_scores.items()
    }
    recommendation_mean = target_means["participant_final_recommendation"]
    incumbent_mean = target_means["observed_incumbent"]
    report: dict[str, Any] = {
        "schema_version": BLIND_EVALUATION_REPORT_VERSION,
        "formal_result": plan.get("formal_result") is True,
        "cell_id": plan["cell_id"],
        "cell_key_sha256": plan["cell_key_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "status": "completed" if len(completed) == 6 else "failed_retained_no_replacement",
        "scheduled_execution_count": 6,
        "completed_execution_count": len(completed),
        "failed_execution_count": 6 - len(completed),
        "evaluator_provider_call_count": 0,
        "participant_operation_denominator_impact": 0,
        "participant_feedback_emitted": False,
        "target_score_means": target_means,
        "recommendation_gain_over_incumbent": (
            recommendation_mean - incumbent_mean
            if recommendation_mean is not None and incumbent_mean is not None
            else None
        ),
        "receipt_sha256": [receipt["receipt_sha256"] for receipt in receipts],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    write_json_atomic(output_root / "report.json", report)
    return report


def validate_blind_evaluation_report(
    report: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipts: list[Mapping[str, Any]],
) -> list[str]:
    """Validate a blind report and its six retained execution receipts."""

    errors = validate_blind_evaluation_plan(plan)
    expected_hash = canonical_json_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    if report.get("schema_version") != BLIND_EVALUATION_REPORT_VERSION:
        errors.append("unexpected blind evaluator report schema")
    if report.get("report_sha256") != expected_hash:
        errors.append("blind evaluator report self-hash mismatch")
    if report.get("plan_sha256") != plan.get("plan_sha256") or report.get(
        "cell_key_sha256"
    ) != plan.get("cell_key_sha256"):
        errors.append("blind evaluator report plan binding mismatch")
    if len(receipts) != 6:
        errors.append("blind evaluator report must retain six receipts")
    valid_receipts: list[Mapping[str, Any]] = []
    for receipt in receipts:
        expected_receipt_hash = canonical_json_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        if receipt.get("receipt_sha256") != expected_receipt_hash:
            errors.append("blind evaluator execution receipt self-hash mismatch")
            continue
        valid_receipts.append(receipt)
    observed_hashes = [receipt.get("receipt_sha256") for receipt in receipts]
    if report.get("receipt_sha256") != observed_hashes:
        errors.append("blind evaluator report receipt binding mismatch")
    completed = [receipt for receipt in valid_receipts if receipt.get("status") == "completed"]
    if (
        report.get("scheduled_execution_count") != 6
        or report.get("completed_execution_count") != len(completed)
        or report.get("failed_execution_count") != 6 - len(completed)
    ):
        errors.append("blind evaluator report execution denominator mismatch")
    if (
        report.get("evaluator_provider_call_count") != 0
        or report.get("participant_operation_denominator_impact") != 0
        or report.get("participant_feedback_emitted") is not False
    ):
        errors.append("blind evaluator report isolation invariant failed")
    target_means: dict[str, float | None] = {}
    for target in ("observed_incumbent", "participant_final_recommendation"):
        scores = [
            float(receipt["leaderboard_score"])
            for receipt in completed
            if receipt.get("target") == target
            and isinstance(receipt.get("leaderboard_score"), int | float)
            and not isinstance(receipt.get("leaderboard_score"), bool)
        ]
        target_means[target] = sum(scores) / 3.0 if len(scores) == 3 else None
    if report.get("target_score_means") != target_means:
        errors.append("blind evaluator target means do not reconcile")
    recommendation = target_means["participant_final_recommendation"]
    incumbent = target_means["observed_incumbent"]
    expected_gain = (
        recommendation - incumbent if recommendation is not None and incumbent is not None else None
    )
    if report.get("recommendation_gain_over_incumbent") != expected_gain:
        errors.append("blind evaluator recommendation gap does not reconcile")
    return errors


__all__ = [
    "BLIND_EVALUATION_REPORT_VERSION",
    "BLIND_EVALUATOR_VERSION",
    "blind_execution_directory_name",
    "build_blind_evaluation_plan",
    "effective_blind_evaluator_contract",
    "execute_blind_evaluation_plan",
    "validate_blind_evaluation_plan",
    "validate_blind_evaluation_report",
]
