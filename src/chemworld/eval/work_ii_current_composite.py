"""Provider-free evaluation of the terminal DeepSeek public C2 composite.

The current participant cohort was produced by a prospective runner rather than
the historical ``WorkIIFormalCellStore``.  This module binds the two preserved
run roots directly, replaces the complete A-S crystallization block as declared,
and reuses the production truth, prediction, law, blind-replay and C2 estimand
implementations without pretending that a historical release manifest owns the
new trajectories.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)
from chemworld.eval.work_ii_analysis import score_cell_checkpoint_errors
from chemworld.eval.work_ii_blind import (
    blind_execution_directory_name,
    build_blind_evaluation_plan,
    effective_blind_evaluator_contract,
    execute_blind_evaluation_plan,
    validate_blind_evaluation_plan,
    validate_blind_evaluation_report,
)
from chemworld.eval.work_ii_formal import (
    EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT,
    FORMAL_ARMS,
)
from chemworld.eval.work_ii_law_summary import evaluate_final_law_summary
from chemworld.eval.work_ii_public_c2 import (
    LOCUS_IDS,
    PUBLIC_C2_PLAN_CONTRACT,
    build_public_c2_cluster_rows,
    build_public_c2_locus_gate,
)
from chemworld.eval.work_ii_truth import (
    build_evaluator_truth_plan,
    execute_evaluator_truth_plan,
    validate_evaluator_truth_plan,
    validate_evaluator_truth_report,
)

CURRENT_COMPOSITE_INPUT_VERSION = "chemworld-work-ii-current-composite-input-0.1"
CURRENT_COMPOSITE_DATASET_VERSION = "chemworld-work-ii-current-composite-dataset-0.1"
CURRENT_COMPOSITE_REPORT_VERSION = "chemworld-work-ii-current-composite-evaluation-0.1"
CURRENT_COMPOSITE_COHORT_ID = "work-ii-deepseek-c2-current-composite-v0.1"
EXPECTED_QUERY_COUNTS = {"A_E": 4, "A_P": 16, "A_S": 16}
EXPECTED_COUNTS = {
    "A_E": {"tasks": 5, "clusters": 25, "cells": 75},
    "A_P": {"tasks": 2, "clusters": 10, "cells": 30},
    "A_S": {"tasks": 2, "clusters": 10, "cells": 30},
}


@dataclass(frozen=True)
class CompositeCell:
    cell_id: str
    cell_key_sha256: str
    cluster_id: str
    block: str
    locus: str
    task_id: str
    world_seed: int
    prior_arm: str
    scheduled_experiment_count: int
    complete_experiment_count: int
    terminal_state: str
    terminal_reason_code: str
    qualification_passed: bool
    discard_count: int
    resource_rejection_count: int
    source_root: Path
    cell_root: Path
    config_path: Path
    summary_sha256: str
    trajectory_sha256: str
    terminal_artifact_sha256: str
    summary: dict[str, Any]
    config: dict[str, Any]


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError:
        return str(resolved)


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _finite_mean(values: Sequence[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return fmean(finite) if finite else None


def _discard_count(trajectory_path: Path) -> int:
    count = 0
    with trajectory_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                isinstance(row, Mapping)
                and row.get("transaction_status") == "committed"
                and row.get("operation_type") == "discard_batch"
            ):
                count += 1
    return count


def _terminal_state(
    *,
    qualification_passed: bool,
    completed_experiments: int,
    scheduled_experiments: int,
    failed_checks: Sequence[Any],
) -> tuple[str, str]:
    if qualification_passed:
        return "completed", "qualification_passed"
    if completed_experiments < scheduled_experiments:
        return "right_censored", "participant_stopped_before_planned_experiment_count"
    rendered_checks = [str(value) for value in failed_checks if str(value)]
    suffix = ":" + ",".join(rendered_checks) if rendered_checks else ""
    return "failed", "qualification_contract_failed_after_planned_experiments" + suffix


def _validate_checkpoint_schedule(
    *,
    cell_id: str,
    observed_stages: Sequence[str],
    snapshot_stages: Sequence[str],
    terminal_state: str,
) -> None:
    """Allow only the frozen full schedule or a right-censored ordered prefix."""

    observed = list(observed_stages)
    scheduled = list(snapshot_stages)
    if observed == scheduled:
        return
    if (
        terminal_state == "right_censored"
        and len(observed) < len(scheduled)
        and observed == scheduled[: len(observed)]
    ):
        return
    raise ValueError(f"{cell_id}: checkpoint schedule differs from config")


def _task_specs(plan: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    blocks = plan.get("public_blocks")
    if not isinstance(blocks, list):
        raise ValueError("execution plan lacks public blocks")
    specs: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_block in blocks:
        if not isinstance(raw_block, Mapping):
            raise ValueError("execution plan contains a malformed block")
        block = str(raw_block.get("block", ""))
        locus = str(raw_block.get("locus", ""))
        rounds = raw_block.get("rounds_per_session")
        tasks = raw_block.get("tasks")
        if locus not in LOCUS_IDS or not block or not isinstance(rounds, int):
            raise ValueError("execution plan block identity is invalid")
        if not isinstance(tasks, list):
            raise ValueError(f"{block}: tasks are missing")
        for raw_task in tasks:
            if not isinstance(raw_task, Mapping):
                raise ValueError(f"{block}: malformed task")
            task_id = str(raw_task.get("task_id", ""))
            key = (block, task_id)
            if not task_id or key in specs:
                raise ValueError(f"duplicate or missing task identity: {key}")
            specs[key] = {
                "block": block,
                "locus": locus,
                "rounds": rounds,
                "task_id": task_id,
                "config": str(raw_task.get("config", "")),
                "world_seeds": list(raw_task.get("world_seeds", [])),
            }
    return specs


def build_current_composite_inputs(
    root: Path,
    *,
    base_run: Path,
    replacement_run: Path,
    cohort_id: str = CURRENT_COMPOSITE_COHORT_ID,
    replacement_block: str = "A_S",
    replacement_task: str = "reaction-to-crystallization",
    analysis_plan_path: Path,
    formal_design_path: Path,
) -> tuple[dict[str, Any], list[CompositeCell], dict[str, Any], dict[str, Any]]:
    """Bind the exact 120+15 current participant composite."""

    root = root.resolve()
    base_run = base_run.resolve()
    replacement_run = replacement_run.resolve()
    single_source_run = base_run == replacement_run
    if not cohort_id:
        raise ValueError("current-composite cohort id must be non-empty")
    analysis_plan_path = analysis_plan_path.resolve()
    formal_design_path = formal_design_path.resolve()
    base_plan_path = base_run / "execution_plan.json"
    replacement_plan_path = replacement_run / "execution_plan.json"
    base_plan = _load_object(base_plan_path)
    replacement_plan = _load_object(replacement_plan_path)
    if base_plan.get("prior_arms") != list(FORMAL_ARMS):
        raise ValueError("base execution plan prior arms drifted")
    if replacement_plan.get("prior_arms") != list(FORMAL_ARMS):
        raise ValueError("replacement execution plan prior arms drifted")
    base_specs = _task_specs(base_plan)
    replacement_specs = _task_specs(replacement_plan)
    selector = (replacement_block, replacement_task)
    if selector not in base_specs or selector not in replacement_specs:
        raise ValueError("replacement task is absent from an execution plan")
    if replacement_specs[selector]["world_seeds"] != base_specs[selector]["world_seeds"]:
        raise ValueError("replacement task changes the frozen world roster")

    analysis_plan = _load_object(analysis_plan_path)
    if analysis_plan.get("public_C2_confirmatory_extension") != PUBLIC_C2_PLAN_CONTRACT:
        raise ValueError("analysis plan C2 extension differs from the frozen contract")
    formal_design = _load_object(formal_design_path)
    blind_contract = formal_design.get("blind_evaluator_contract")
    if not isinstance(blind_contract, Mapping):
        raise ValueError("formal design lacks the blind evaluator isolation contract")

    cells: list[CompositeCell] = []
    config_bindings: dict[str, dict[str, Any]] = {}
    for key, base_spec in base_specs.items():
        use_replacement = not single_source_run and key == selector
        spec = replacement_specs[key] if use_replacement else base_spec
        source_run = replacement_run if use_replacement else base_run
        config_path = (root / spec["config"]).resolve()
        config = _load_object(config_path)
        if config.get("task_id") != spec["task_id"]:
            raise ValueError(f"{spec['task_id']}: campaign config task binding drifted")
        snapshot_stages = config.get("snapshot_stages")
        if (
            not isinstance(snapshot_stages, list)
            or len(snapshot_stages) != 5
            or snapshot_stages[0] != "pre_evidence"
            or snapshot_stages[-1] != "final"
        ):
            raise ValueError(f"{spec['task_id']}: frozen checkpoint schedule is invalid")
        config_key = _relative(root, config_path)
        config_bindings[config_key] = {
            "path": config_key,
            "file_sha256": file_sha256(config_path),
            "canonical_json_sha256": canonical_json_sha256(config),
            "task_id": spec["task_id"],
            "locus": spec["locus"],
            "scheduled_experiment_count": spec["rounds"],
            "snapshot_stages": list(snapshot_stages),
        }
        for world_seed in spec["world_seeds"]:
            if isinstance(world_seed, bool) or not isinstance(world_seed, int):
                raise ValueError(f"{spec['task_id']}: invalid world seed")
            cluster_id = f"{spec['locus']}--{spec['task_id']}--seed{world_seed}"
            for prior_arm in FORMAL_ARMS:
                cell_id = f"{spec['block']}--{spec['task_id']}--seed{world_seed}--{prior_arm}"
                cell_root = source_run / "cells" / cell_id
                summary_path = cell_root / "summary.json"
                trajectory_path = cell_root / "trajectory.jsonl"
                if not summary_path.is_file() or not trajectory_path.is_file():
                    raise FileNotFoundError(f"current composite cell is incomplete: {cell_id}")
                summary = _load_object(summary_path)
                identity = summary.get("prospective_cohort_cell")
                analysis = summary.get("analysis")
                qualification = summary.get("qualification")
                replay = summary.get("exact_replay")
                if not isinstance(identity, Mapping) or not isinstance(analysis, Mapping):
                    raise ValueError(f"{cell_id}: participant identity or analysis is missing")
                expected_identity = {
                    "block": spec["block"],
                    "locus": spec["locus"],
                    "task_id": spec["task_id"],
                    "world_seed": world_seed,
                    "prior_arm": prior_arm,
                }
                if any(identity.get(field) != value for field, value in expected_identity.items()):
                    raise ValueError(f"{cell_id}: participant identity drifted")
                if summary.get("arm") != prior_arm or summary.get(
                    "prospective_formal_result"
                ) is not True:
                    raise ValueError(f"{cell_id}: participant result classification drifted")
                if not isinstance(replay, Mapping) or replay.get("verified") is not True:
                    raise ValueError(f"{cell_id}: exact replay is not verified")
                complete_count = analysis.get("complete_experiment_count")
                if (
                    isinstance(complete_count, bool)
                    or not isinstance(complete_count, int)
                    or not 0 <= complete_count <= int(spec["rounds"])
                ):
                    raise ValueError(f"{cell_id}: complete-experiment denominator is invalid")
                qualification = qualification if isinstance(qualification, Mapping) else {}
                qualification_passed = qualification.get("passed") is True
                failed_checks = qualification.get("failed_checks")
                failed_checks = failed_checks if isinstance(failed_checks, list) else []
                terminal_state, reason = _terminal_state(
                    qualification_passed=qualification_passed,
                    completed_experiments=complete_count,
                    scheduled_experiments=int(spec["rounds"]),
                    failed_checks=failed_checks,
                )
                snapshots = analysis.get("belief_snapshots")
                observed_stages = [
                    str(row.get("stage", ""))
                    for row in snapshots
                    if isinstance(row, Mapping)
                ] if isinstance(snapshots, list) else []
                _validate_checkpoint_schedule(
                    cell_id=cell_id,
                    observed_stages=observed_stages,
                    snapshot_stages=snapshot_stages,
                    terminal_state=terminal_state,
                )
                summary_digest = file_sha256(summary_path)
                trajectory_digest = file_sha256(trajectory_path)
                terminal_digest = canonical_json_sha256(
                    {
                        "cell_id": cell_id,
                        "summary_sha256": summary_digest,
                        "trajectory_sha256": trajectory_digest,
                    }
                )
                cell_key = canonical_json_sha256(
                    {
                        "cohort_id": cohort_id,
                        "cell_id": cell_id,
                        "terminal_artifact_sha256": terminal_digest,
                    }
                )
                resource_rejections = analysis.get("resource_rejection_count", 0)
                resource_rejections = (
                    int(resource_rejections)
                    if isinstance(resource_rejections, int)
                    and not isinstance(resource_rejections, bool)
                    else 0
                )
                cells.append(
                    CompositeCell(
                        cell_id=cell_id,
                        cell_key_sha256=cell_key,
                        cluster_id=cluster_id,
                        block=str(spec["block"]),
                        locus=str(spec["locus"]),
                        task_id=str(spec["task_id"]),
                        world_seed=world_seed,
                        prior_arm=prior_arm,
                        scheduled_experiment_count=int(spec["rounds"]),
                        complete_experiment_count=complete_count,
                        terminal_state=terminal_state,
                        terminal_reason_code=reason,
                        qualification_passed=qualification_passed,
                        discard_count=_discard_count(trajectory_path),
                        resource_rejection_count=resource_rejections,
                        source_root=source_run,
                        cell_root=cell_root,
                        config_path=config_path,
                        summary_sha256=summary_digest,
                        trajectory_sha256=trajectory_digest,
                        terminal_artifact_sha256=terminal_digest,
                        summary=summary,
                        config=config,
                    )
                )

    if len(cells) != 135 or len({cell.cell_id for cell in cells}) != 135:
        raise ValueError("current composite must contain 135 unique cells")
    unique_clusters = {cell.cluster_id: cell for cell in cells}
    cluster_counts = Counter(cell.locus for cell in unique_clusters.values())
    cell_counts = Counter(cell.locus for cell in cells)
    task_counts = {
        locus: len({cell.task_id for cell in cells if cell.locus == locus})
        for locus in LOCUS_IDS
    }
    for locus, expected in EXPECTED_COUNTS.items():
        if (
            task_counts[locus] != expected["tasks"]
            or cluster_counts[locus] != expected["clusters"]
            or cell_counts[locus] != expected["cells"]
        ):
            raise ValueError(f"{locus}: current composite roster mismatch")

    cell_bindings = [
        {
            "cell_id": cell.cell_id,
            "cell_key_sha256": cell.cell_key_sha256,
            "world_cluster_id": cell.cluster_id,
            "block": cell.block,
            "locus": cell.locus,
            "task_id": cell.task_id,
            "world_seed": cell.world_seed,
            "prior_arm": cell.prior_arm,
            "scheduled_experiment_count": cell.scheduled_experiment_count,
            "complete_experiment_count": cell.complete_experiment_count,
            "terminal_state": cell.terminal_state,
            "terminal_reason_code": cell.terminal_reason_code,
            "qualification_passed": cell.qualification_passed,
            "discard_count": cell.discard_count,
            "resource_rejection_count": cell.resource_rejection_count,
            "source_root": _relative(root, cell.source_root),
            "summary": {
                "path": _relative(root, cell.cell_root / "summary.json"),
                "sha256": cell.summary_sha256,
            },
            "trajectory": {
                "path": _relative(root, cell.cell_root / "trajectory.jsonl"),
                "sha256": cell.trajectory_sha256,
            },
            "terminal_artifact_sha256": cell.terminal_artifact_sha256,
            "config_path": _relative(root, cell.config_path),
        }
        for cell in cells
    ]
    state_counts = Counter(cell.terminal_state for cell in cells)
    input_manifest: dict[str, Any] = {
        "schema_version": CURRENT_COMPOSITE_INPUT_VERSION,
        "status": "passed",
        "formal_result": False,
        "prospective_formal_result": True,
        "provider_call_count": 0,
        "participant_feedback_allowed": False,
        "cohort_id": cohort_id,
        "composition": {
            "base_retained_cell_count": 135 if single_source_run else 120,
            "replacement_cell_count": 0 if single_source_run else 15,
            "replacement_block": replacement_block,
            "replacement_task": replacement_task,
            **({"single_source_run": True} if single_source_run else {}),
        },
        "source_runs": {
            "base": {
                "path": _relative(root, base_run),
                "cohort_id": base_plan.get("cohort_id"),
            },
            "replacement": {
                "path": _relative(root, replacement_run),
                "cohort_id": replacement_plan.get("cohort_id"),
            },
        },
        "source_execution_plans": {
            "base": {
                "path": _relative(root, base_plan_path),
                "sha256": file_sha256(base_plan_path),
            },
            "replacement": {
                "path": _relative(root, replacement_plan_path),
                "sha256": file_sha256(replacement_plan_path),
            },
        },
        "analysis_contract": {
            "path": _relative(root, analysis_plan_path),
            "sha256": file_sha256(analysis_plan_path),
            "public_C2_confirmatory_extension_sha256": canonical_json_sha256(
                analysis_plan["public_C2_confirmatory_extension"]
            ),
        },
        "blind_contract": {
            "path": _relative(root, formal_design_path),
            "sha256": file_sha256(formal_design_path),
            "contract_sha256": canonical_json_sha256(blind_contract),
        },
        "roster": {
            "task_count": 9,
            "cluster_count": 45,
            "cell_count": 135,
            "terminal_state_counts": dict(sorted(state_counts.items())),
            "by_locus": {
                locus: {
                    "task_count": task_counts[locus],
                    "cluster_count": cluster_counts[locus],
                    "cell_count": cell_counts[locus],
                }
                for locus in LOCUS_IDS
            },
        },
        "config_bindings": [config_bindings[key] for key in sorted(config_bindings)],
        "cell_bindings": cell_bindings,
    }
    input_manifest["input_manifest_sha256"] = _self_hash(
        input_manifest, "input_manifest_sha256"
    )
    return input_manifest, cells, analysis_plan, dict(blind_contract)


def _final_snapshot(analysis: Mapping[str, Any]) -> Mapping[str, Any] | None:
    snapshots = analysis.get("belief_snapshots")
    if not isinstance(snapshots, list):
        return None
    return next(
        (
            snapshot
            for snapshot in reversed(snapshots)
            if isinstance(snapshot, Mapping) and snapshot.get("stage") == "final"
        ),
        None,
    )


def _validate_bound_trajectory(unit_root: Path, receipt: Mapping[str, Any]) -> None:
    binding = receipt.get("trajectory")
    if receipt.get("status") != "completed":
        return
    if not isinstance(binding, Mapping):
        raise ValueError(f"{receipt.get('execution_id')}: completed receipt lacks trajectory")
    relative = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(relative, str) or not isinstance(digest, str):
        raise ValueError(f"{receipt.get('execution_id')}: trajectory binding is malformed")
    path = (unit_root / relative).resolve()
    if not path.is_relative_to(unit_root.resolve()) or not path.is_file():
        raise ValueError(f"{receipt.get('execution_id')}: trajectory is missing")
    if file_sha256(path) != digest:
        raise ValueError(f"{receipt.get('execution_id')}: trajectory digest drifted")
    replay = receipt.get("exact_replay")
    if not isinstance(replay, Mapping) or replay.get("verified") is not True:
        raise ValueError(f"{receipt.get('execution_id')}: exact replay is not verified")


def _load_truth_unit(unit_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    stored_plan = _load_object(unit_root / "plan.json")
    if stored_plan != dict(plan):
        raise ValueError(f"{unit_root.name}: truth plan binding drifted")
    report = _load_object(unit_root / "report.json")
    errors = validate_evaluator_truth_report(report, plan)
    if errors:
        raise ValueError(f"{unit_root.name}: invalid truth report: {'; '.join(errors)}")
    receipts = report.get("receipts")
    if not isinstance(receipts, list):
        raise ValueError(f"{unit_root.name}: truth receipts are missing")
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError(f"{unit_root.name}: malformed truth receipt")
        _validate_bound_trajectory(unit_root, receipt)
    return report


def _load_blind_unit(unit_root: Path, plan: Mapping[str, Any]) -> dict[str, Any]:
    stored_plan = _load_object(unit_root / "plan.json")
    if stored_plan != dict(plan):
        raise ValueError(f"{unit_root.name}: blind plan binding drifted")
    report = _load_object(unit_root / "report.json")
    receipts: list[Mapping[str, Any]] = []
    for execution in plan["executions"]:
        receipt_path = (
            unit_root
            / "executions"
            / blind_execution_directory_name(execution)
            / "receipt.json"
        )
        receipt = _load_object(receipt_path)
        _validate_bound_trajectory(unit_root, receipt)
        receipts.append(receipt)
    errors = validate_blind_evaluation_report(report, plan, receipts)
    if errors:
        raise ValueError(f"{unit_root.name}: invalid blind report: {'; '.join(errors)}")
    return report


def _compact_law(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "status",
        "present",
        "schema_version_matches",
        "feature_count",
        "metric_law_count",
        "term_count",
        "evidence_reference_count",
        "confidence",
        "evaluator_executability_status",
        "continuous_prediction_validity_status",
        "registered_query_count",
        "registered_query_metric_count",
        "normalized_mae",
        "pre_to_law_summary_improvement",
        "summary_minus_effective_final_error",
        "prediction_consistency_normalized_mae",
        "evaluation_error",
    )
    return {field: record.get(field) for field in fields}


def _compact_checkpoint(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep estimands and denominators while dropping per-query error terms."""

    scores = record.get("checkpoint_scores")
    scores = scores if isinstance(scores, Mapping) else {}
    return {
        "terminal_state": record.get("terminal_state"),
        "scheduled_snapshot_count": record.get("scheduled_snapshot_count"),
        "observed_snapshot_count": record.get("observed_snapshot_count"),
        "scored_snapshot_count": record.get("scored_snapshot_count"),
        "checkpoint_scores": {
            str(stage): {
                "error": score.get("error"),
                "term_count": score.get("term_count"),
            }
            for stage, score in scores.items()
            if isinstance(score, Mapping)
        },
        "unscorable_snapshots": record.get("unscorable_snapshots", []),
        "effective_pre_error": record.get("effective_pre_error"),
        "effective_final_error": record.get("effective_final_error"),
        "effective_final_stage": record.get("effective_final_stage"),
        "primary_improvement": record.get("primary_improvement"),
        "confirmatory_improvement_bounds": record.get(
            "confirmatory_improvement_bounds"
        ),
        "missing_failure_rule": record.get("missing_failure_rule"),
    }


def _observed_point_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    observed = []
    for row in rows:
        clone = dict(row)
        checkpoint = dict(clone["checkpoint_error"])
        improvement = float(checkpoint["primary_improvement"])
        checkpoint["confirmatory_improvement_bounds"] = [improvement, improvement]
        clone["checkpoint_error"] = checkpoint
        observed.append(clone)
    return observed


def _descriptive_complete_case(
    cluster_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    complete = [row for row in cluster_rows if row.get("complete_case") is True]
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in complete:
        by_task[str(row["task_id"])].append(float(row["H3_primary_contrast"]))
    return {
        "cluster_count": len(complete),
        "mean_H3_primary_contrast": _finite_mean(
            [row["H3_primary_contrast"] for row in complete]
        ),
        "task_means": {
            task: _finite_mean(values) for task, values in sorted(by_task.items())
        },
    }


def _prediction_summary(
    compact_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_locus_rows = {
        locus: [row for row in compact_rows if row["locus_id"] == locus]
        for locus in LOCUS_IDS
    }
    locus_results: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for locus, rows in by_locus_rows.items():
        cluster_rows = build_public_c2_cluster_rows(locus, rows)
        gate = build_public_c2_locus_gate(locus, cluster_rows)
        observed_clusters = build_public_c2_cluster_rows(locus, _observed_point_rows(rows))
        observed_gate = build_public_c2_locus_gate(locus, observed_clusters)
        p_value = (
            gate["intersection_union_p_value"]
            if locus == "A_E"
            else gate["effective_intersection_union_p_value"]
        )
        p_values[locus] = float(p_value)
        by_arm: dict[str, Any] = {}
        for arm in FORMAL_ARMS:
            arm_rows = [row for row in rows if row["prior_arm"] == arm]
            by_arm[arm] = {
                "cell_count": len(arm_rows),
                "terminal_state_counts": dict(
                    sorted(Counter(str(row["terminal_state"]) for row in arm_rows).items())
                ),
                "scored_pre_count": sum(
                    row["checkpoint_error"].get("effective_pre_error") is not None
                    for row in arm_rows
                ),
                "scored_final_count": sum(
                    "final" in row["checkpoint_error"].get("checkpoint_scores", {})
                    for row in arm_rows
                ),
                "mean_effective_pre_error": _finite_mean(
                    [row["checkpoint_error"].get("effective_pre_error") for row in arm_rows]
                ),
                "mean_effective_final_error": _finite_mean(
                    [row["checkpoint_error"].get("effective_final_error") for row in arm_rows]
                ),
                "mean_primary_improvement": _finite_mean(
                    [row["checkpoint_error"].get("primary_improvement") for row in arm_rows]
                ),
            }
        stage_rows: dict[str, dict[str, Any]] = {}
        stages = sorted(
            {
                stage
                for row in rows
                for stage in row["checkpoint_error"].get("checkpoint_scores", {})
            },
            key=lambda stage: (
                0 if stage == "pre_evidence" else 2 if stage == "final" else 1,
                stage,
            ),
        )
        for stage in stages:
            stage_rows[stage] = {}
            for arm in FORMAL_ARMS:
                values = [
                    row["checkpoint_error"]["checkpoint_scores"][stage]["error"]
                    for row in rows
                    if row["prior_arm"] == arm
                    and stage in row["checkpoint_error"].get("checkpoint_scores", {})
                ]
                stage_rows[stage][arm] = {
                    "scored_cell_count": len(values),
                    "mean_normalized_mae": _finite_mean(values),
                }
        locus_results[locus] = {
            "gate": gate,
            "observed_point_sensitivity_gate": observed_gate,
            "cluster_rows": cluster_rows,
            "complete_case_descriptive": _descriptive_complete_case(cluster_rows),
            "by_arm": by_arm,
            "checkpoint_trajectory": stage_rows,
        }
    passed = all(locus_results[locus]["gate"]["passed"] is True for locus in LOCUS_IDS)
    return {
        "locus_results": locus_results,
        "C2_intersection_union": {
            "required_loci": list(LOCUS_IDS),
            "locus_gate_passed": {
                locus: locus_results[locus]["gate"]["passed"] for locus in LOCUS_IDS
            },
            "locus_p_values": p_values,
            "intersection_union_p_value": max(p_values.values()),
            "all_three_locus_gates_required": True,
            "naive_nine_task_pooling_performed": False,
            "passed": passed,
        },
    }


def _law_summary(compact_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for locus in (*LOCUS_IDS, "overall"):
        locus_rows = (
            list(compact_rows)
            if locus == "overall"
            else [row for row in compact_rows if row["locus_id"] == locus]
        )
        result[locus] = {}
        for arm in (*FORMAL_ARMS, "all"):
            rows = (
                locus_rows
                if arm == "all"
                else [row for row in locus_rows if row["prior_arm"] == arm]
            )
            evaluated = [row for row in rows if row["law_summary"]["status"] == "evaluated"]
            deltas = [
                row["law_summary"]["summary_minus_effective_final_error"]
                for row in evaluated
                if row["law_summary"]["summary_minus_effective_final_error"] is not None
            ]
            result[locus][arm] = {
                "cell_count": len(rows),
                "evaluated_count": len(evaluated),
                "failed_or_missing_count": len(rows) - len(evaluated),
                "mean_normalized_mae": _finite_mean(
                    [row["law_summary"]["normalized_mae"] for row in evaluated]
                ),
                "mean_pre_to_law_improvement": _finite_mean(
                    [
                        row["law_summary"]["pre_to_law_summary_improvement"]
                        for row in evaluated
                    ]
                ),
                "mean_summary_minus_final_error": _finite_mean(deltas),
                "law_better_than_final_prediction_count": sum(
                    isinstance(value, int | float) and float(value) < 0.0 for value in deltas
                ),
                "law_worse_than_final_prediction_count": sum(
                    isinstance(value, int | float) and float(value) > 0.0 for value in deltas
                ),
                "mean_prediction_consistency_error": _finite_mean(
                    [
                        row["law_summary"]["prediction_consistency_normalized_mae"]
                        for row in evaluated
                    ]
                ),
            }
    return result


def _blind_summary(compact_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for locus in (*LOCUS_IDS, "overall"):
        locus_rows = (
            list(compact_rows)
            if locus == "overall"
            else [row for row in compact_rows if row["locus_id"] == locus]
        )
        completed = [row for row in locus_rows if row["blind"]["status"] == "completed"]
        gains = [row["blind"]["recommendation_gain_over_incumbent"] for row in completed]
        result[locus] = {
            "assigned_cell_count": len(locus_rows),
            "evaluable_cell_count": sum(row["terminal_state"] == "completed" for row in locus_rows),
            "completed_blind_cell_count": len(completed),
            "scheduled_execution_count": sum(
                int(row["blind"]["scheduled_execution_count"]) for row in locus_rows
            ),
            "launched_execution_count": sum(
                int(row["blind"]["launched_execution_count"]) for row in locus_rows
            ),
            "completed_execution_count": sum(
                int(row["blind"]["completed_execution_count"]) for row in locus_rows
            ),
            "unstarted_execution_count": sum(
                int(row["blind"]["scheduled_execution_count"])
                - int(row["blind"]["launched_execution_count"])
                for row in locus_rows
            ),
            "mean_recommendation_gain_over_incumbent": _finite_mean(gains),
            "recommendation_better_count": sum(
                isinstance(value, int | float) and float(value) > 1.0e-12 for value in gains
            ),
            "recommendation_equivalent_count": sum(
                isinstance(value, int | float) and abs(float(value)) <= 1.0e-12
                for value in gains
            ),
            "recommendation_worse_count": sum(
                isinstance(value, int | float) and float(value) < -1.0e-12 for value in gains
            ),
        }
    return result


def execute_current_composite_evaluator(
    root: Path,
    *,
    base_run: Path,
    replacement_run: Path,
    cohort_id: str = CURRENT_COMPOSITE_COHORT_ID,
    analysis_plan_path: Path,
    formal_design_path: Path,
    output_root: Path,
    replacement_block: str = "A_S",
    replacement_task: str = "reaction-to-crystallization",
    resume: bool = False,
    preflight_only: bool = False,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute or resume all current-composite provider-free evaluator units."""

    root = root.resolve()
    output_root = output_root.resolve()
    input_manifest, cells, analysis_plan, blind_contract = build_current_composite_inputs(
        root,
        base_run=base_run,
        replacement_run=replacement_run,
        cohort_id=cohort_id,
        replacement_block=replacement_block,
        replacement_task=replacement_task,
        analysis_plan_path=analysis_plan_path,
        formal_design_path=formal_design_path,
    )
    if preflight_only:
        return input_manifest
    if output_root.exists() and not resume:
        raise FileExistsError(f"refusing to overwrite evaluator output root: {output_root}")
    if not output_root.exists() and resume:
        raise FileNotFoundError("missing-only resume requires an existing evaluator output root")
    if output_root.exists():
        if _load_object(output_root / "input_manifest.json") != input_manifest:
            raise ValueError("stored current-composite input binding drifted")
    else:
        output_root.mkdir(parents=True)
        write_json_atomic(output_root / "input_manifest.json", input_manifest)

    def emit(payload: Mapping[str, Any]) -> None:
        if progress is not None:
            progress(payload)

    clusters: dict[str, list[CompositeCell]] = defaultdict(list)
    for cell in cells:
        clusters[cell.cluster_id].append(cell)
    truth_plans: dict[str, dict[str, Any]] = {}
    for cluster_id, members in clusters.items():
        first = members[0]
        if len(members) != 3 or {cell.prior_arm for cell in members} != set(FORMAL_ARMS):
            raise ValueError(f"{cluster_id}: current composite arm triplet is incomplete")
        plan = build_evaluator_truth_plan(
            {
                "world_cluster_id": cluster_id,
                "task_id": first.task_id,
                "world_seed": first.world_seed,
            },
            first.config,
            formal_result=False,
            formal_preflight_sha256=None,
        )
        errors = validate_evaluator_truth_plan(plan)
        if errors:
            raise ValueError(f"{cluster_id}: invalid truth plan: {'; '.join(errors)}")
        if plan["truth_query_count"] != EXPECTED_QUERY_COUNTS[first.locus]:
            raise ValueError(f"{cluster_id}: registered truth-query denominator drifted")
        truth_plans[cluster_id] = plan
    if sum(int(plan["truth_query_count"]) for plan in truth_plans.values()) != 420:
        raise ValueError("current composite truth denominator must be 420")

    emit(
        {
            "event": "current_composite_evaluator_started",
            "truth_cluster_total": len(truth_plans),
            "participant_cell_total": len(cells),
            "eligible_blind_cell_total": sum(cell.terminal_state == "completed" for cell in cells),
        }
    )
    truth_reports: dict[str, dict[str, Any]] = {}
    for index, cluster_id in enumerate(sorted(truth_plans), start=1):
        plan = truth_plans[cluster_id]
        member = clusters[cluster_id][0]
        unit_root = output_root / "truth" / cluster_id
        emit(
            {
                "event": "truth_cluster_started",
                "completed": index - 1,
                "total": len(truth_plans),
                "world_cluster_id": cluster_id,
            }
        )
        if unit_root.exists():
            report = _load_truth_unit(unit_root, plan)
        else:
            report = execute_evaluator_truth_plan(plan, member.config, unit_root)
            report = _load_truth_unit(unit_root, plan)
        truth_reports[cluster_id] = report
        emit(
            {
                "event": "truth_cluster_finished",
                "completed": index,
                "total": len(truth_plans),
                "world_cluster_id": cluster_id,
                "status": report["status"],
            }
        )

    primary_error = analysis_plan["primary_prediction_error"]
    metric_scales = primary_error.get("registered_metric_scale_overrides", {})
    default_metric_scale = float(primary_error.get("default_metric_scale", 1.0))
    detailed_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    blind_plans: dict[str, tuple[CompositeCell, dict[str, Any]]] = {}
    for cell in cells:
        truth = truth_reports[cell.cluster_id].get("truth")
        truth = truth if isinstance(truth, Mapping) else {}
        analysis = cell.summary["analysis"]
        checkpoint = score_cell_checkpoint_errors(
            analysis,
            truth,
            terminal_state=cell.terminal_state,
            snapshot_stages=cell.config["snapshot_stages"],
            metric_scales=metric_scales,
            default_metric_scale=default_metric_scale,
        )
        final = _final_snapshot(analysis)
        law = evaluate_final_law_summary(
            final.get("law_summary") if isinstance(final, Mapping) else None,
            truth_plan=truth_plans[cell.cluster_id],
            evaluator_truth=truth,
            final_checkpoint_predictions=(
                final.get("predictions") if isinstance(final, Mapping) else None
            ),
            effective_pre_error=checkpoint.get("effective_pre_error"),
            effective_final_error=checkpoint.get("effective_final_error"),
            evaluation_contract=EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT,
        )
        base_row = {
            "cell_id": cell.cell_id,
            "cell_key_sha256": cell.cell_key_sha256,
            "terminal_artifact_sha256": cell.terminal_artifact_sha256,
            "world_cluster_id": cell.cluster_id,
            "block": cell.block,
            "locus_id": cell.locus,
            "task_id": cell.task_id,
            "world_seed": cell.world_seed,
            "prior_arm": cell.prior_arm,
            "terminal_state": cell.terminal_state,
            "terminal_reason_code": cell.terminal_reason_code,
            "qualification_passed": cell.qualification_passed,
            "scheduled_experiment_count": cell.scheduled_experiment_count,
            "complete_experiment_count": cell.complete_experiment_count,
            "discard_count": cell.discard_count,
            "resource_rejection_count": cell.resource_rejection_count,
        }
        blind_summary: dict[str, Any] = {
            "status": f"not_launched_participant_{cell.terminal_state}",
            "scheduled_execution_count": 6,
            "launched_execution_count": 0,
            "completed_execution_count": 0,
            "recommendation_gain_over_incumbent": None,
        }
        compact: dict[str, Any] = {
            **base_row,
            "checkpoint_error": _compact_checkpoint(checkpoint),
            "law_summary": _compact_law(law),
            "blind": blind_summary,
        }
        detailed_rows.append(
            {
                **base_row,
                "checkpoint_error": checkpoint,
                "law_summary": law,
                "blind": dict(blind_summary),
            }
        )
        compact_rows.append(compact)
        if cell.terminal_state != "completed":
            failures.append(
                {
                    "stage": "participant",
                    "cell_id": cell.cell_id,
                    "terminal_state": cell.terminal_state,
                    "reason_code": cell.terminal_reason_code,
                    "affected_blind_execution_count": 6,
                }
            )
        for unscorable in checkpoint["unscorable_snapshots"]:
            failures.append(
                {
                    "stage": "checkpoint_scoring",
                    "cell_id": cell.cell_id,
                    **dict(unscorable),
                }
            )
        if law.get("status") != "evaluated":
            failures.append(
                {
                    "stage": "law_summary",
                    "cell_id": cell.cell_id,
                    "status": law.get("status"),
                    "error": law.get("evaluation_error"),
                }
            )
        if cell.terminal_state == "completed":
            effective_contract = effective_blind_evaluator_contract(
                {
                    "complete_experiment_count": cell.scheduled_experiment_count,
                    "participant_final_recommendation_count": 1,
                    "blind_validation_target_count": 2,
                    "blind_replicates_per_target": 3,
                    "blind_validation_execution_count": 6,
                },
                blind_contract,
            )
            plan = build_blind_evaluation_plan(
                {
                    "cell_id": cell.cell_id,
                    "cell_key_sha256": cell.cell_key_sha256,
                    "task_id": cell.task_id,
                    "world_seed": cell.world_seed,
                },
                cell.summary,
                effective_contract,
            )
            plan_errors = validate_blind_evaluation_plan(plan)
            if plan_errors:
                raise ValueError(f"{cell.cell_id}: invalid blind plan: {'; '.join(plan_errors)}")
            blind_plans[cell.cell_key_sha256] = (cell, plan)

    compact_by_key = {row["cell_key_sha256"]: row for row in compact_rows}
    detailed_by_key = {row["cell_key_sha256"]: row for row in detailed_rows}
    for index, key in enumerate(sorted(blind_plans), start=1):
        cell, plan = blind_plans[key]
        unit_root = output_root / "blind" / key
        emit(
            {
                "event": "blind_cell_started",
                "completed": index - 1,
                "total": len(blind_plans),
                "cell_id": cell.cell_id,
            }
        )
        if unit_root.exists():
            report = _load_blind_unit(unit_root, plan)
        else:
            report = execute_blind_evaluation_plan(plan, cell.config, unit_root)
            report = _load_blind_unit(unit_root, plan)
        blind = {
            "status": report["status"],
            "scheduled_execution_count": 6,
            "launched_execution_count": 6,
            "completed_execution_count": int(report["completed_execution_count"]),
            "failed_execution_count": int(report["failed_execution_count"]),
            "target_score_means": report["target_score_means"],
            "recommendation_gain_over_incumbent": report[
                "recommendation_gain_over_incumbent"
            ],
            "report_sha256": report["report_sha256"],
        }
        compact_by_key[key]["blind"] = blind
        detailed_by_key[key]["blind"] = blind
        if report["status"] != "completed":
            failures.append(
                {
                    "stage": "blind",
                    "cell_id": cell.cell_id,
                    "status": report["status"],
                    "failed_execution_count": report["failed_execution_count"],
                }
            )
        emit(
            {
                "event": "blind_cell_finished",
                "completed": index,
                "total": len(blind_plans),
                "cell_id": cell.cell_id,
                "status": report["status"],
            }
        )

    for cluster_id, report in truth_reports.items():
        for receipt in report["receipts"]:
            if receipt["status"] != "completed":
                failures.append(
                    {
                        "stage": "truth",
                        "world_cluster_id": cluster_id,
                        "query_id": receipt["query_id"],
                        "failure_type": receipt.get("failure_type"),
                        "failure_message": receipt.get("failure_message"),
                    }
                )

    detailed_rows = [detailed_by_key[cell.cell_key_sha256] for cell in cells]
    compact_rows = [compact_by_key[cell.cell_key_sha256] for cell in cells]
    dataset: dict[str, Any] = {
        "schema_version": CURRENT_COMPOSITE_DATASET_VERSION,
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "provider_call_count": 0,
        "cell_rows": detailed_rows,
    }
    dataset["dataset_sha256"] = _self_hash(dataset, "dataset_sha256")
    write_json_atomic(output_root / "analysis_dataset.json", dataset)

    prediction = _prediction_summary(compact_rows)
    law = _law_summary(compact_rows)
    blind = _blind_summary(compact_rows)
    truth_completed = sum(
        int(report["completed_truth_query_count"]) for report in truth_reports.values()
    )
    truth_metrics = sum(
        int(report["completed_truth_query_metric_count"])
        for report in truth_reports.values()
    )
    scored_snapshots = sum(
        int(row["checkpoint_error"]["scored_snapshot_count"]) for row in compact_rows
    )
    composite_report: dict[str, Any] = {
        "schema_version": CURRENT_COMPOSITE_REPORT_VERSION,
        "status": "completed",
        "formal_result": False,
        "prospective_formal_result": True,
        "provider_call_count": 0,
        "participant_feedback_emitted": False,
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "analysis_dataset_sha256": dataset["dataset_sha256"],
        "denominators": {
            "task_count": 9,
            "cluster_count": 45,
            "cell_count": 135,
            "terminal_state_counts": dict(
                sorted(Counter(row["terminal_state"] for row in compact_rows).items())
            ),
            "truth_scheduled_execution_count": 420,
            "truth_completed_execution_count": truth_completed,
            "truth_failed_execution_count": 420 - truth_completed,
            "truth_completed_query_metric_count": truth_metrics,
            "checkpoint_scheduled_count": 675,
            "checkpoint_scored_count": scored_snapshots,
            "checkpoint_unscorable_count": 675 - scored_snapshots,
            "law_summary_scheduled_count": 135,
            "law_summary_evaluated_count": law["overall"]["all"]["evaluated_count"],
            "blind_scheduled_execution_count": 810,
            "blind_launched_execution_count": blind["overall"]["launched_execution_count"],
            "blind_completed_execution_count": blind["overall"]["completed_execution_count"],
            "blind_unstarted_execution_count": blind["overall"]["unstarted_execution_count"],
            "discard_affected_cell_count": sum(row["discard_count"] > 0 for row in compact_rows),
            "resource_rejection_affected_cell_count": sum(
                row["resource_rejection_count"] > 0 for row in compact_rows
            ),
            "retained_failure_record_count": len(failures),
        },
        "prediction_correction": prediction,
        "executable_law": law,
        "blind_action": blind,
        "cell_rows": compact_rows,
        "all_retained_failures": failures,
        "task_completion": {
            "prediction_truth_and_scoring": (
                "completed"
                if truth_completed == 420 and scored_snapshots == 675
                else "terminal_with_failures"
            ),
            "final_law_summary_evaluation": (
                "completed"
                if law["overall"]["all"]["evaluated_count"] == 135
                else "terminal_with_failures"
            ),
            "blind_action_evaluation": (
                "completed_with_participant_failure_denominator"
                if blind["overall"]["launched_execution_count"]
                == blind["overall"]["completed_execution_count"]
                else "terminal_with_evaluator_failures"
            ),
        },
        "claim_decisions": {
            "selective_scientific_correction_supported_across_C2": prediction[
                "C2_intersection_union"
            ]["passed"],
            "executable_law_recovery_evaluated": law["overall"]["all"][
                "evaluated_count"
            ]
            == 135,
            "blind_recommendation_mean_gain_over_incumbent": blind["overall"][
                "mean_recommendation_gain_over_incumbent"
            ],
            "private_transfer_evaluated": False,
        },
    }
    composite_report["report_sha256"] = _self_hash(composite_report, "report_sha256")
    write_json_atomic(output_root / "summary.json", composite_report)
    emit(
        {
            "event": "current_composite_evaluator_finished",
            "status": composite_report["status"],
            "truth_completed": truth_completed,
            "checkpoints_scored": scored_snapshots,
            "laws_evaluated": law["overall"]["all"]["evaluated_count"],
            "blind_completed": blind["overall"]["completed_execution_count"],
            "C2_passed": prediction["C2_intersection_union"]["passed"],
        }
    )
    return composite_report


__all__ = [
    "CURRENT_COMPOSITE_COHORT_ID",
    "CURRENT_COMPOSITE_DATASET_VERSION",
    "CURRENT_COMPOSITE_INPUT_VERSION",
    "CURRENT_COMPOSITE_REPORT_VERSION",
    "CompositeCell",
    "build_current_composite_inputs",
    "execute_current_composite_evaluator",
]
