"""Post-hoc evaluator confirmation for retained Work II development cells."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_analysis import (
    WORK_II_ANALYSIS_ARMS,
    build_cluster_correction_record,
)

WORK_II_DEVELOPMENT_CONFIRMATION_VERSION = (
    "chemworld-work-ii-development-evaluator-confirmation-0.1"
)

NONPHYSICAL_CAMPAIGN_FIELDS = frozenset(
    {
        "schema_version",
        "pilot_id",
        "method_resources",
        "method_resource_semantics",
        "execution",
        "provider",
        "qualification",
        "provider_qualification",
    }
)


def physical_campaign_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the participant-visible and executable physical campaign contract."""

    return {
        str(key): value
        for key, value in config.items()
        if str(key) not in NONPHYSICAL_CAMPAIGN_FIELDS
    }


def collect_development_cells(
    source_manifest: Mapping[str, Any],
    loaded_sources: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Collect the exact retained cell denominator without provider-group mixing."""

    provider_group = str(source_manifest.get("provider_group", ""))
    cells: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for source, matrix in loaded_sources:
        if str(source.get("provider_group")) != provider_group:
            raise ValueError("development confirmation cannot mix provider groups")
        task_id = str(source["task_id"])
        source_id = str(source["source_id"])
        seed_reports = matrix.get("seed_reports")
        if not isinstance(seed_reports, list):
            raise ValueError(f"{source_id}: matrix lacks seed reports")
        for seed_report in seed_reports:
            if not isinstance(seed_report, Mapping):
                raise ValueError(f"{source_id}: malformed seed report")
            world_seed = int(seed_report["world_seed"])
            results = seed_report.get("results")
            if not isinstance(results, list):
                raise ValueError(f"{source_id}: seed report lacks result rows")
            for result in results:
                if not isinstance(result, Mapping):
                    raise ValueError(f"{source_id}: malformed result row")
                arm = str(result.get("arm", ""))
                if arm not in WORK_II_ANALYSIS_ARMS:
                    raise ValueError(f"{source_id}: invalid prior arm {arm}")
                identity = (task_id, world_seed, arm)
                if identity in seen:
                    raise ValueError(f"duplicate development cell: {identity}")
                seen.add(identity)
                qualification = result.get("qualification")
                qualification = (
                    qualification if isinstance(qualification, Mapping) else {}
                )
                completed_and_qualified = (
                    result.get("completed") is True
                    and qualification.get("passed") is True
                )
                cell_key_payload = {
                    "provider_group": provider_group,
                    "source_id": source_id,
                    "task_id": task_id,
                    "world_seed": world_seed,
                    "prior_arm": arm,
                }
                cells.append(
                    {
                        "cell_id": (
                            f"deepseek-development--{task_id}--seed-{world_seed}--{arm}"
                        ),
                        "cell_key_sha256": canonical_json_sha256(cell_key_payload),
                        "source_id": source_id,
                        "task_id": task_id,
                        "world_seed": world_seed,
                        "prior_arm": arm,
                        "participant_state": (
                            "completed" if completed_and_qualified else "failed"
                        ),
                        "completed_and_qualified": completed_and_qualified,
                        "participant_failure": result.get("failure"),
                        "result": dict(result),
                    }
                )
    return sorted(
        cells,
        key=lambda row: (
            str(row["task_id"]),
            int(row["world_seed"]),
            WORK_II_ANALYSIS_ARMS.index(str(row["prior_arm"])),
        ),
    )


def build_development_confirmation_preflight(
    *,
    source_manifest: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    task_configs: Mapping[str, Mapping[str, Any]],
    participant_configs: Mapping[str, Mapping[str, Any]],
    source_bindings: Sequence[Mapping[str, Any]],
    source_commit: str,
) -> dict[str, Any]:
    """Freeze exact post-hoc evaluator denominators before local execution."""

    identities = {
        (str(cell["task_id"]), int(cell["world_seed"]), str(cell["prior_arm"]))
        for cell in cells
    }
    clusters = {(task_id, seed) for task_id, seed, _arm in identities}
    task_ids = {task_id for task_id, _seed, _arm in identities}
    errors: list[str] = []
    if len(cells) != 75 or len(identities) != 75:
        errors.append("retained participant denominator is not exactly 75 unique cells")
    if len(clusters) != 25:
        errors.append("retained world-cluster denominator is not exactly 25")
    if len(task_ids) != 5:
        errors.append("retained task denominator is not exactly five")
    for task_id, seed in sorted(clusters):
        arms = {
            arm
            for candidate_task, candidate_seed, arm in identities
            if candidate_task == task_id and candidate_seed == seed
        }
        if arms != set(WORK_II_ANALYSIS_ARMS):
            errors.append(f"{task_id}/seed-{seed}: cluster lacks its exact arm triplet")
    config_rows: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        evaluator_config = task_configs.get(task_id)
        participant_config = participant_configs.get(task_id)
        if not isinstance(evaluator_config, Mapping) or not isinstance(
            participant_config, Mapping
        ):
            errors.append(f"{task_id}: missing evaluator or participant config")
            continue
        evaluator_contract = physical_campaign_contract(evaluator_config)
        participant_contract = physical_campaign_contract(participant_config)
        matched = evaluator_contract == participant_contract
        if not matched:
            errors.append(f"{task_id}: evaluator physical contract differs from participant")
        config_rows.append(
            {
                "task_id": task_id,
                "evaluator_config_sha256": canonical_json_sha256(evaluator_config),
                "participant_config_sha256": canonical_json_sha256(participant_config),
                "physical_contract_sha256": canonical_json_sha256(
                    evaluator_contract
                ),
                "physical_contract_matched": matched,
            }
        )
    qualified = sum(
        cell.get("completed_and_qualified") is True for cell in cells
    )
    report: dict[str, Any] = {
        "schema_version": WORK_II_DEVELOPMENT_CONFIRMATION_VERSION,
        "status": "passed" if not errors else "failed",
        "formal_result": False,
        "provider_group": source_manifest.get("provider_group"),
        "source_analysis_id": source_manifest.get("analysis_id"),
        "source_commit": source_commit,
        "source_bindings": [dict(item) for item in source_bindings],
        "task_configs": config_rows,
        "expected_task_count": 5,
        "expected_world_cluster_count": 25,
        "expected_participant_cell_count": 75,
        "retained_participant_cell_count": len(cells),
        "qualified_blind_cell_count": qualified,
        "participant_failed_or_unqualified_cell_count": len(cells) - qualified,
        "scheduled_truth_query_count": len(clusters) * 4,
        "scheduled_blind_execution_count": qualified * 6,
        "evaluator_provider_call_count": 0,
        "participant_operation_denominator_impact": 0,
        "errors": errors,
    }
    report["preflight_sha256"] = canonical_json_sha256(report)
    return report


def numeric_summary(values: Sequence[object]) -> dict[str, float | int | None]:
    numeric = [
        float(value)
        for value in values
        if not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    ]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric) if numeric else None,
        "median": statistics.median(numeric) if numeric else None,
        "sample_sd": statistics.stdev(numeric) if len(numeric) > 1 else None,
        "minimum": min(numeric) if numeric else None,
        "maximum": max(numeric) if numeric else None,
    }


def build_confirmation_summary(
    *,
    preflight: Mapping[str, Any],
    cell_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    prior_infrastructure_attempt: Mapping[str, Any] | None = None,
    accepted_raw_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact, exact-denominator development confirmation report."""

    task_arm_rows: list[dict[str, Any]] = []
    for task_id in sorted({str(row["task_id"]) for row in cell_rows}):
        for arm in WORK_II_ANALYSIS_ARMS:
            rows = [
                row
                for row in cell_rows
                if row["task_id"] == task_id and row["prior_arm"] == arm
            ]
            task_arm_rows.append(
                {
                    "task_id": task_id,
                    "prior_arm": arm,
                    "cell_count": len(rows),
                    "completed_cell_count": sum(
                        row["participant_state"] == "completed" for row in rows
                    ),
                    "pre_error": numeric_summary(
                        [row.get("effective_pre_error") for row in rows]
                    ),
                    "final_error": numeric_summary(
                        [row.get("effective_final_error") for row in rows]
                    ),
                    "checkpoint_improvement": numeric_summary(
                        [row.get("checkpoint_improvement") for row in rows]
                    ),
                    "law_summary_error": numeric_summary(
                        [row.get("law_summary_error") for row in rows]
                    ),
                    "law_summary_improvement": numeric_summary(
                        [row.get("law_summary_improvement") for row in rows]
                    ),
                    "blind_recommendation_gain": numeric_summary(
                        [row.get("blind_recommendation_gain") for row in rows]
                    ),
                }
            )
    law_status = Counter(str(row.get("law_summary_status")) for row in cell_rows)
    report: dict[str, Any] = {
        "schema_version": WORK_II_DEVELOPMENT_CONFIRMATION_VERSION,
        "analysis_date": "2026-08-11",
        "formal_result": False,
        "status": "passed" if not failures else "failed",
        "preflight_sha256": preflight["preflight_sha256"],
        "provider_group": preflight["provider_group"],
        "source_analysis_id": preflight["source_analysis_id"],
        "denominators": {
            "task_count": len({str(row["task_id"]) for row in cell_rows}),
            "world_cluster_count": len(cluster_rows),
            "participant_cell_count": len(cell_rows),
            "participant_completed_and_qualified_cell_count": sum(
                row["participant_state"] == "completed" for row in cell_rows
            ),
            "participant_failed_or_unqualified_cell_count": sum(
                row["participant_state"] != "completed" for row in cell_rows
            ),
            "truth_query_count": sum(
                int(row.get("query_count", 0)) for row in truth_rows
            ),
            "truth_completed_query_count": sum(
                int(row.get("completed_query_count", 0)) for row in truth_rows
            ),
            "truth_exact_replay_count": sum(
                int(row.get("exact_replay_count", 0)) for row in truth_rows
            ),
            "blind_scheduled_execution_count": sum(
                int(row.get("blind_scheduled_execution_count", 0))
                for row in cell_rows
            ),
            "blind_completed_execution_count": sum(
                int(row.get("blind_completed_execution_count", 0))
                for row in cell_rows
            ),
            "checkpoint_final_scored_cell_count": sum(
                row.get("effective_final_error") is not None for row in cell_rows
            ),
            "law_summary_evaluated_cell_count": law_status["evaluated"],
        },
        "cluster_contrasts": {
            "H1_prior_utility": numeric_summary(
                [row.get("H1_prior_utility") for row in cluster_rows]
            ),
            "H2_prior_vulnerability": numeric_summary(
                [row.get("H2_prior_vulnerability") for row in cluster_rows]
            ),
            "H3_primary_contrast": numeric_summary(
                [row.get("H3_primary_contrast") for row in cluster_rows]
            ),
            "complete_case_cluster_count": sum(
                row.get("complete_case") is True for row in cluster_rows
            ),
        },
        "law_summary_status_counts": dict(sorted(law_status.items())),
        "task_arm_summaries": task_arm_rows,
        "truth_rows": [dict(row) for row in truth_rows],
        "cluster_rows": [dict(row) for row in cluster_rows],
        "cell_rows": [dict(row) for row in cell_rows],
        "failures": [dict(item) for item in failures],
        "audit": {
            "participant_sessions_rerun": False,
            "failed_participant_cells_replaced": False,
            "evaluator_provider_call_count": 0,
            "participant_operation_denominator_impact": 0,
            "formal_hypothesis_tests_run": False,
            "private_transfer_evaluated": False,
            "interpretation": (
                "provider-separated development evidence only; no formal inference, "
                "private transfer claim or cross-provider ranking"
            ),
            "prior_infrastructure_attempt": (
                dict(prior_infrastructure_attempt)
                if isinstance(prior_infrastructure_attempt, Mapping)
                else None
            ),
            "accepted_raw_receipt": (
                dict(accepted_raw_receipt)
                if isinstance(accepted_raw_receipt, Mapping)
                else None
            ),
        },
    }
    report["analysis_sha256"] = canonical_json_sha256(report)
    return report


def build_cluster_rows(
    cell_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in cell_rows:
        grouped.setdefault(
            (str(row["task_id"]), int(row["world_seed"])), {}
        )[str(row["prior_arm"])] = row
    output: list[dict[str, Any]] = []
    for (task_id, world_seed), arms in sorted(grouped.items()):
        checkpoint_records = {
            arm: {
                "primary_improvement": row["checkpoint_improvement"],
                "effective_pre_error": row["effective_pre_error"],
            }
            for arm, row in arms.items()
        }
        contrast = build_cluster_correction_record(checkpoint_records)
        output.append(
            {
                "task_id": task_id,
                "world_seed": world_seed,
                "complete_case": all(
                    row["participant_state"] == "completed"
                    and row["checkpoint_missing_rule"] == "observed_final"
                    for row in arms.values()
                ),
                **contrast,
            }
        )
    return output


__all__ = [
    "WORK_II_DEVELOPMENT_CONFIRMATION_VERSION",
    "build_cluster_rows",
    "build_confirmation_summary",
    "build_development_confirmation_preflight",
    "collect_development_cells",
    "numeric_summary",
    "physical_campaign_contract",
]
