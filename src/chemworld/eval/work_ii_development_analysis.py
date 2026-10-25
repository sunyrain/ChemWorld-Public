"""Descriptive analysis for retained Work II development campaign matrices."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256

WORK_II_DEVELOPMENT_ANALYSIS_VERSION = "chemworld-work-ii-development-analysis-0.1"
WORK_II_SINGLE_PROVIDER_DEVELOPMENT_ANALYSIS_VERSION = (
    "chemworld-work-ii-single-provider-development-analysis-0.1"
)
ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _summary(values: Sequence[Any]) -> dict[str, Any]:
    numeric = [number for value in values if (number := _number(value)) is not None]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "sample_sd": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "sample_sd": statistics.stdev(numeric) if len(numeric) > 1 else None,
        "minimum": min(numeric),
        "maximum": max(numeric),
    }


def _snapshot(analysis: Mapping[str, Any], stage: str) -> Mapping[str, Any] | None:
    raw = analysis.get("belief_snapshots")
    for item in (raw if isinstance(raw, list) else ()):
        if isinstance(item, Mapping) and item.get("stage") == stage:
            return item
    return None


def _prior_reliability(snapshot: Mapping[str, Any] | None) -> float | None:
    prior = snapshot.get("prior_assessment") if isinstance(snapshot, Mapping) else None
    return _number(prior.get("reliability_probability")) if isinstance(prior, Mapping) else None


def _misindexed_fields(snapshot: Mapping[str, Any] | None) -> list[str]:
    prior = snapshot.get("prior_assessment") if isinstance(snapshot, Mapping) else None
    fields = prior.get("suspected_misindexed_fields") if isinstance(prior, Mapping) else None
    return [str(value) for value in fields] if isinstance(fields, list) else []


def _law_summary(snapshot: Mapping[str, Any] | None) -> Mapping[str, Any]:
    raw = snapshot.get("law_summary") if isinstance(snapshot, Mapping) else None
    return raw if isinstance(raw, Mapping) else {}


def _experiment_scores(analysis: Mapping[str, Any]) -> list[float]:
    experiments = analysis.get("experiments")
    scores: list[float] = []
    for item in (experiments if isinstance(experiments, list) else ()):
        if (
            isinstance(item, Mapping)
            and (score := _number(item.get("leaderboard_score"))) is not None
        ):
            scores.append(score)
    return scores


def _committed_operations(analysis: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    experiments = analysis.get("experiments")
    for experiment in (experiments if isinstance(experiments, list) else ()):
        operations = experiment.get("operations") if isinstance(experiment, Mapping) else None
        if isinstance(operations, list):
            rows.extend(item for item in operations if isinstance(item, Mapping))
    return rows


def _material_recipe_count(task_id: str, analysis: Mapping[str, Any]) -> int | None:
    experiments = analysis.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return None
    identities: set[tuple[Any, ...]] = set()
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            continue
        operations = experiment.get("operations")
        if not isinstance(operations, list):
            continue
        solvent = None
        catalyst = None
        electrolyte = None
        for operation in operations:
            if not isinstance(operation, Mapping):
                continue
            if operation.get("operation") == "add_solvent":
                solvent = operation.get("solvent")
            elif operation.get("operation") == "add_catalyst":
                catalyst = operation.get("catalyst")
            elif operation.get("operation") == "set_potential":
                electrolyte = operation.get("electrolyte_profile")
        identity = (
            (solvent, electrolyte)
            if task_id == "electrochemical-conversion"
            else (catalyst, solvent)
        )
        identities.add(identity)
    return len(identities)


def _receipt_mcp_failures(result: Mapping[str, Any]) -> int:
    method = result.get("method_resources")
    value = method.get("recovered_mcp_tool_failure_count") if isinstance(method, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    count = 0
    receipts = result.get("provider_receipts")
    for receipt in (receipts if isinstance(receipts, list) else ()):
        calls = receipt.get("mcp_tool_calls") if isinstance(receipt, Mapping) else None
        count += sum(
            1
            for call in (calls if isinstance(calls, list) else ())
            if isinstance(call, Mapping) and call.get("status") != "completed"
        )
    return count


def _receipt_provider_errors(result: Mapping[str, Any]) -> tuple[int, int]:
    events = 0
    entries = 0
    receipts = result.get("provider_receipts")
    for receipt in (receipts if isinstance(receipts, list) else ()):
        if not isinstance(receipt, Mapping):
            continue
        raw_events = receipt.get("provider_error_event_count")
        if isinstance(raw_events, int) and not isinstance(raw_events, bool) and raw_events >= 0:
            events += raw_events
        else:
            counts = receipt.get("event_counts")
            if isinstance(counts, Mapping):
                events += int(counts.get("error", 0)) + int(counts.get("turn.failed", 0))
        raw_entries = receipt.get("provider_errors")
        entries += len(raw_entries) if isinstance(raw_entries, list) else 0
    return events, entries


def _cell_record(
    *,
    source_id: str,
    provider_group: str,
    provider_id: str,
    task_id: str,
    world_seed: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    analysis = result.get("analysis")
    analysis = analysis if isinstance(analysis, Mapping) else {}
    scores = _experiment_scores(analysis)
    operations = _committed_operations(analysis)
    pre = _snapshot(analysis, "pre_evidence")
    final = _snapshot(analysis, "final")
    pre_reliability = _prior_reliability(pre)
    final_reliability = _prior_reliability(final)
    pre_confidence = _number(pre.get("overall_confidence")) if isinstance(pre, Mapping) else None
    final_confidence = (
        _number(final.get("overall_confidence")) if isinstance(final, Mapping) else None
    )
    law = _law_summary(final)
    metric_laws = law.get("metric_laws")
    term_count = sum(
        len(terms)
        for item in (metric_laws if isinstance(metric_laws, list) else ())
        if isinstance(item, Mapping) and isinstance((terms := item.get("terms")), list)
    )
    final_resources = analysis.get("final_campaign_resources")
    state = final_resources.get("state") if isinstance(final_resources, Mapping) else None
    report_only = state.get("report_only") if isinstance(state, Mapping) else None
    report_only = report_only if isinstance(report_only, Mapping) else {}
    attempts = int(analysis.get("operation_attempt_count", 0) or 0)
    rejections = int(analysis.get("resource_rejection_count", 0) or 0)
    process_profile = analysis.get("process_profile")
    process_counts = (
        process_profile.get("counts") if isinstance(process_profile, Mapping) else None
    )
    profile_committed = (
        process_counts.get("committed_operation_count")
        if isinstance(process_counts, Mapping)
        else None
    )
    ledger_committed_counts = (
        state.get("operation_committed_counts") if isinstance(state, Mapping) else None
    )
    if isinstance(profile_committed, int) and not isinstance(profile_committed, bool):
        committed = profile_committed
    elif isinstance(ledger_committed_counts, Mapping):
        committed = sum(
            int(value)
            for value in ledger_committed_counts.values()
            if isinstance(value, int) and not isinstance(value, bool)
        )
    else:
        committed = len(operations)
    method = result.get("method_resources")
    method = method if isinstance(method, Mapping) else {}
    provider_error_events, provider_error_entries = _receipt_provider_errors(result)
    qualification = result.get("qualification")
    exact_replay = result.get("exact_replay")
    failure = result.get("failure")
    final_recommendation = analysis.get("final_recommendation")
    if not isinstance(final_recommendation, Mapping):
        receipts = result.get("provider_receipts")
        final_recommendation = next(
            (
                receipt.get("final_recommendation")
                for receipt in (receipts if isinstance(receipts, list) else ())
                if isinstance(receipt, Mapping)
                and isinstance(receipt.get("final_recommendation"), Mapping)
            ),
            None,
        )
    nonfinal_measurements = sum(
        operation.get("operation") == "measure"
        and operation.get("instrument") != "final_assay"
        for operation in operations
    )
    return {
        "source_id": source_id,
        "provider_group": provider_group,
        "provider_id": provider_id,
        "task_id": task_id,
        "world_seed": int(world_seed),
        "arm": result.get("arm"),
        "completed": result.get("completed") is True,
        "qualification_passed": (
            qualification.get("passed") is True if isinstance(qualification, Mapping) else False
        ),
        "exact_replay_verified": (
            exact_replay.get("verified") is True if isinstance(exact_replay, Mapping) else False
        ),
        "complete_experiment_count": int(analysis.get("complete_experiment_count", 0) or 0),
        "experiment_scores": scores,
        "first_score": scores[0] if scores else None,
        "final_score": scores[-1] if scores else None,
        "best_score": max(scores) if scores else None,
        "best_minus_first": max(scores) - scores[0] if scores else None,
        "pre_prior_reliability": pre_reliability,
        "final_prior_reliability": final_reliability,
        "prior_reliability_delta": (
            final_reliability - pre_reliability
            if final_reliability is not None and pre_reliability is not None
            else None
        ),
        "pre_overall_confidence": pre_confidence,
        "final_overall_confidence": final_confidence,
        "overall_confidence_delta": (
            final_confidence - pre_confidence
            if final_confidence is not None and pre_confidence is not None
            else None
        ),
        "final_suspected_misindexed_fields": _misindexed_fields(final),
        "final_law_confidence": _number(law.get("confidence")),
        "final_law_feature_count": len(law.get("feature_ids", []))
        if isinstance(law.get("feature_ids"), list)
        else 0,
        "final_law_term_count": term_count,
        "final_evidence_count": len(final.get("evidence_ids", []))
        if isinstance(final, Mapping) and isinstance(final.get("evidence_ids"), list)
        else 0,
        "final_recommendation_committed": isinstance(final_recommendation, Mapping),
        "operation_attempt_count": attempts,
        "committed_operation_count": committed,
        "validation_failure_count": max(0, attempts - committed - rejections),
        "resource_rejection_count": rejections,
        "recovered_mcp_tool_failure_count": _receipt_mcp_failures(result),
        "provider_error_event_count": provider_error_events,
        "provider_error_entry_count": provider_error_entries,
        "nonfinal_measurement_count": nonfinal_measurements,
        "unique_material_recipe_count": _material_recipe_count(task_id, analysis),
        "process_time_s": _number(report_only.get("process_time_s")),
        "physical_cost": _number(report_only.get("physical_cost")),
        "accumulated_risk": _number(report_only.get("accumulated_risk")),
        "input_token_count": _number(method.get("input_token_count")),
        "cached_input_token_count": _number(method.get("cached_input_token_count")),
        "uncached_input_token_count": _number(method.get("uncached_input_token_count")),
        "output_token_count": _number(method.get("output_token_count")),
        "provider_usage_accounting_complete": (
            method.get("provider_usage_accounting_complete") is True
        ),
        "elapsed_s": _number(result.get("elapsed_s")),
        "failure": dict(failure) if isinstance(failure, Mapping) else None,
        "qualification_failed_checks": list(qualification.get("failed_checks", []))
        if isinstance(qualification, Mapping)
        and isinstance(qualification.get("failed_checks"), list)
        else [],
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    return _summary([row.get(key) for row in rows])


def _arm_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("completed") is True]
    experiment_positions: dict[str, dict[str, Any]] = {}
    for index in range(4):
        values = [
            scores[index]
            for row in completed
            if isinstance((scores := row.get("experiment_scores")), list) and len(scores) > index
        ]
        experiment_positions[str(index + 1)] = _summary(values)
    final_snapshot_rows = [row for row in completed if row.get("final_law_confidence") is not None]
    return {
        "terminal_cell_count": len(rows),
        "completed_cell_count": len(completed),
        "complete_experiment_count": sum(int(row["complete_experiment_count"]) for row in rows),
        "endpoint": {
            "first_score": _metric_summary(completed, "first_score"),
            "final_score": _metric_summary(completed, "final_score"),
            "best_score": _metric_summary(completed, "best_score"),
            "best_minus_first": _metric_summary(completed, "best_minus_first"),
            "score_by_experiment_index": experiment_positions,
        },
        "belief": {
            "pre_prior_reliability": _metric_summary(completed, "pre_prior_reliability"),
            "final_prior_reliability": _metric_summary(completed, "final_prior_reliability"),
            "prior_reliability_delta": _metric_summary(completed, "prior_reliability_delta"),
            "pre_overall_confidence": _metric_summary(completed, "pre_overall_confidence"),
            "final_overall_confidence": _metric_summary(completed, "final_overall_confidence"),
            "overall_confidence_delta": _metric_summary(completed, "overall_confidence_delta"),
            "final_misindex_flag_count": sum(
                bool(row.get("final_suspected_misindexed_fields")) for row in completed
            ),
            "final_misindex_flag_denominator": len(final_snapshot_rows),
            "final_law_confidence": _metric_summary(completed, "final_law_confidence"),
            "final_law_feature_count": _metric_summary(completed, "final_law_feature_count"),
            "final_law_term_count": _metric_summary(completed, "final_law_term_count"),
            "final_evidence_count": _metric_summary(completed, "final_evidence_count"),
            "final_recommendation_committed_count": sum(
                row.get("final_recommendation_committed") is True for row in completed
            ),
        },
        "execution": {
            key: _metric_summary(completed, key)
            for key in (
                "operation_attempt_count",
                "committed_operation_count",
                "validation_failure_count",
                "resource_rejection_count",
                "recovered_mcp_tool_failure_count",
                "provider_error_event_count",
                "nonfinal_measurement_count",
                "unique_material_recipe_count",
                "process_time_s",
                "physical_cost",
                "accumulated_risk",
                "input_token_count",
                "cached_input_token_count",
                "uncached_input_token_count",
                "output_token_count",
                "elapsed_s",
            )
        },
    }


def _paired_contrast(
    rows: Sequence[Mapping[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
    metric: str,
) -> dict[str, Any]:
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_seed[int(row["world_seed"])][str(row["arm"])] = row
    values: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        left = by_seed[seed].get(left_arm)
        right = by_seed[seed].get(right_arm)
        left_value = _number(left.get(metric)) if isinstance(left, Mapping) else None
        right_value = _number(right.get(metric)) if isinstance(right, Mapping) else None
        if left_value is None or right_value is None:
            continue
        values.append(
            {
                "world_seed": seed,
                "left": left_value,
                "right": right_value,
                "difference": left_value - right_value,
            }
        )
    differences = [item["difference"] for item in values]
    return {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "metric": metric,
        "paired_seed_count": len(values),
        "pairs": values,
        "difference_summary": _summary(differences),
        "positive_count": sum(value > 0 for value in differences),
        "negative_count": sum(value < 0 for value in differences),
        "zero_count": sum(value == 0 for value in differences),
    }


def _provider_summary(
    rows: Sequence[Mapping[str, Any]], expected_cell_count: int
) -> dict[str, Any]:
    failed = [row for row in rows if row.get("qualification_passed") is not True]
    usage_fields = (
        "input_token_count",
        "cached_input_token_count",
        "uncached_input_token_count",
        "output_token_count",
    )
    usage_totals = {
        key: sum(value for row in rows if (value := _number(row.get(key))) is not None)
        for key in usage_fields
    }
    input_total = usage_totals["input_token_count"]
    return {
        "expected_cell_count": expected_cell_count,
        "terminal_record_count": len(rows),
        "completed_cell_count": sum(row.get("completed") is True for row in rows),
        "qualified_cell_count": sum(row.get("qualification_passed") is True for row in rows),
        "complete_experiment_count": sum(int(row["complete_experiment_count"]) for row in rows),
        "operation_attempt_count": sum(int(row["operation_attempt_count"]) for row in rows),
        "committed_operation_count": sum(int(row["committed_operation_count"]) for row in rows),
        "validation_failure_count": sum(int(row["validation_failure_count"]) for row in rows),
        "resource_rejection_count": sum(int(row["resource_rejection_count"]) for row in rows),
        "exact_replay_verified_count": sum(
            row.get("exact_replay_verified") is True for row in rows
        ),
        "recovered_mcp_tool_failure_count": sum(
            int(row["recovered_mcp_tool_failure_count"]) for row in rows
        ),
        "provider_error_event_count": sum(int(row["provider_error_event_count"]) for row in rows),
        "provider_error_entry_count": sum(int(row["provider_error_entry_count"]) for row in rows),
        "provider_usage_accounting_complete_cell_count": sum(
            row.get("provider_usage_accounting_complete") is True for row in rows
        ),
        "provider_usage_totals": {
            **usage_totals,
            "input_cache_hit_ratio": (
                usage_totals["cached_input_token_count"] / input_total
                if input_total > 0
                else None
            ),
        },
        "failed_cells": [
            {
                "task_id": row["task_id"],
                "world_seed": row["world_seed"],
                "arm": row["arm"],
                "complete_experiment_count": row["complete_experiment_count"],
                "failure": row["failure"],
                "qualification_failed_checks": row["qualification_failed_checks"],
            }
            for row in failed
        ],
    }


def _task_reports(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        task_rows = [row for row in rows if row["task_id"] == task_id]
        reports[task_id] = {
            "arm_summaries": {
                arm: _arm_summary([row for row in task_rows if row["arm"] == arm])
                for arm in ARMS
            },
            "paired_endpoint_contrasts": {
                "aligned_minus_opaque_best_score": _paired_contrast(
                    task_rows,
                    left_arm="aligned_nominal",
                    right_arm="opaque",
                    metric="best_score",
                ),
                "misindexed_minus_opaque_best_score": _paired_contrast(
                    task_rows,
                    left_arm="misindexed_nominal",
                    right_arm="opaque",
                    metric="best_score",
                ),
                "aligned_minus_misindexed_best_score": _paired_contrast(
                    task_rows,
                    left_arm="aligned_nominal",
                    right_arm="misindexed_nominal",
                    metric="best_score",
                ),
            },
            "paired_belief_contrasts": {
                "aligned_minus_misindexed_reliability_delta": _paired_contrast(
                    task_rows,
                    left_arm="aligned_nominal",
                    right_arm="misindexed_nominal",
                    metric="prior_reliability_delta",
                )
            },
        }
    return reports


def _three_arm_cluster_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    terminal_clusters: dict[tuple[str, int], set[str]] = defaultdict(set)
    completed_clusters: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row["task_id"]), int(row["world_seed"]))
        terminal_clusters[key].add(str(row["arm"]))
        if row["completed"] is True:
            completed_clusters[key].add(str(row["arm"]))
    return (
        sum(arms == set(ARMS) for arms in completed_clusters.values()),
        sum(arms == set(ARMS) for arms in terminal_clusters.values()),
    )


def build_development_analysis(
    source_manifest: Mapping[str, Any],
    loaded_sources: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build one deterministic, provider-separated descriptive development report."""

    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    expected_by_group: dict[str, int] = defaultdict(int)
    for source, source_report in loaded_sources:
        source_id = str(source["source_id"])
        provider_group = str(source["provider_group"])
        provider_id = str(source["provider_id"])
        task_id = str(source["task_id"])
        expected = int(source_report.get("expected_cell_count", 0))
        expected_by_group[provider_group] += expected
        source_records.append(
            {
                "source_id": source_id,
                "provider_group": provider_group,
                "provider_id": provider_id,
                "task_id": task_id,
                "path": source["path"],
                "sha256": source["sha256"],
                "expected_cell_count": expected,
                "matrix_elapsed_s": source_report.get("elapsed_s"),
            }
        )
        seed_reports = source_report.get("seed_reports")
        for seed_report in (seed_reports if isinstance(seed_reports, list) else ()):
            if not isinstance(seed_report, Mapping):
                continue
            world_seed = int(seed_report["world_seed"])
            results = seed_report.get("results")
            for result in (results if isinstance(results, list) else ()):
                if isinstance(result, Mapping):
                    rows.append(
                        _cell_record(
                            source_id=source_id,
                            provider_group=provider_group,
                            provider_id=provider_id,
                            task_id=task_id,
                            world_seed=world_seed,
                            result=result,
                        )
                    )

    wellau_rows = [row for row in rows if row["provider_group"] == "wellau_fallback"]
    deepseek_rows = [row for row in rows if row["provider_group"] == "deepseek_attempt"]
    task_reports = _task_reports(wellau_rows)
    complete_cluster_count, terminal_cluster_count = _three_arm_cluster_counts(wellau_rows)

    report: dict[str, Any] = {
        "schema_version": WORK_II_DEVELOPMENT_ANALYSIS_VERSION,
        "analysis_id": source_manifest["analysis_id"],
        "analysis_date": source_manifest["analysis_date"],
        "formal_result": False,
        "interpretation_contract": dict(source_manifest["interpretation_contract"]),
        "sources": source_records,
        "wellau_fallback": {
            "denominators": _provider_summary(
                wellau_rows, expected_by_group["wellau_fallback"]
            ),
            "complete_three_arm_cluster_count": complete_cluster_count,
            "terminal_three_arm_cluster_count": terminal_cluster_count,
            "task_reports": task_reports,
        },
        "deepseek_attempt": {
            "denominators": _provider_summary(
                deepseek_rows, expected_by_group["deepseek_attempt"]
            ),
            "task_attempts": {
                task_id: _provider_summary(
                    [row for row in deepseek_rows if row["task_id"] == task_id],
                    sum(
                        int(source["expected_cell_count"])
                        for source in source_records
                        if source["provider_group"] == "deepseek_attempt"
                        and source["task_id"] == task_id
                    ),
                )
                for task_id in sorted({str(row["task_id"]) for row in deepseek_rows})
            },
        },
        "audit": {
            "wellau_predictions_scored_against_evaluator_truth": False,
            "wellau_final_law_summaries_schema_present_count": sum(
                row["final_law_confidence"] is not None for row in wellau_rows
            ),
            "wellau_final_recommendation_committed_count": sum(
                row["final_recommendation_committed"] is True for row in wellau_rows
            ),
            "provider_groups_mixed_in_prior_contrasts": False,
            "formal_hypothesis_tests_run": False,
            "limitations": [
                "development data only",
                "five world seeds per task",
                "one retained WellAU failed cell",
                "no evaluator-truth prediction-error scoring",
                "no blind recommendation replay",
                "DeepSeek partial attempts are harness evidence and are not mixed into "
                "prior contrasts",
            ],
        },
    }
    report["analysis_sha256"] = canonical_json_sha256(report)
    return report


def build_single_provider_development_analysis(
    source_manifest: Mapping[str, Any],
    loaded_sources: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build one deterministic prior-arm analysis without mixing provider groups."""

    provider_group = str(source_manifest["provider_group"])
    rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    expected_cell_count = 0
    provider_ids: set[str] = set()
    for source, matrix in loaded_sources:
        if str(source["provider_group"]) != provider_group:
            raise ValueError("single-provider analysis cannot mix provider groups")
        source_id = str(source["source_id"])
        provider_id = str(source["provider_id"])
        provider_ids.add(provider_id)
        task_id = str(source["task_id"])
        expected = int(matrix.get("expected_cell_count", 0))
        expected_cell_count += expected
        source_records.append(
            {
                "source_id": source_id,
                "provider_group": provider_group,
                "provider_id": provider_id,
                "task_id": task_id,
                "path": source["path"],
                "sha256": source["sha256"],
                "expected_cell_count": expected,
                "matrix_elapsed_s": matrix.get("elapsed_s"),
            }
        )
        seed_reports = matrix.get("seed_reports")
        for seed_report in (seed_reports if isinstance(seed_reports, list) else ()):
            if not isinstance(seed_report, Mapping):
                continue
            world_seed = int(seed_report["world_seed"])
            results = seed_report.get("results")
            for result in (results if isinstance(results, list) else ()):
                if isinstance(result, Mapping):
                    rows.append(
                        _cell_record(
                            source_id=source_id,
                            provider_group=provider_group,
                            provider_id=provider_id,
                            task_id=task_id,
                            world_seed=world_seed,
                            result=result,
                        )
                    )
    if len(provider_ids) != 1:
        raise ValueError("single-provider analysis requires exactly one provider_id")
    complete_clusters, terminal_clusters = _three_arm_cluster_counts(rows)
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            str(row["task_id"]),
            int(row["world_seed"]),
            ARMS.index(str(row["arm"])),
        ),
    )
    report: dict[str, Any] = {
        "schema_version": WORK_II_SINGLE_PROVIDER_DEVELOPMENT_ANALYSIS_VERSION,
        "analysis_id": source_manifest["analysis_id"],
        "analysis_date": source_manifest["analysis_date"],
        "formal_result": False,
        "provider_group": provider_group,
        "provider_id": next(iter(provider_ids)),
        "interpretation_contract": dict(source_manifest["interpretation_contract"]),
        "sources": source_records,
        "denominators": _provider_summary(rows, expected_cell_count),
        "complete_three_arm_cluster_count": complete_clusters,
        "terminal_three_arm_cluster_count": terminal_clusters,
        "task_reports": _task_reports(rows),
        "cell_records": ordered_rows,
        "audit": {
            "predictions_scored_against_evaluator_truth": False,
            "final_law_summaries_schema_present_count": sum(
                row["final_law_confidence"] is not None for row in rows
            ),
            "final_recommendation_committed_count": sum(
                row["final_recommendation_committed"] is True for row in rows
            ),
            "provider_groups_mixed_in_prior_contrasts": False,
            "formal_hypothesis_tests_run": False,
            "limitations": list(
                source_manifest["interpretation_contract"].get("limitations", [])
            ),
        },
    }
    report["analysis_sha256"] = canonical_json_sha256(report)
    return report


__all__ = [
    "ARMS",
    "WORK_II_DEVELOPMENT_ANALYSIS_VERSION",
    "WORK_II_SINGLE_PROVIDER_DEVELOPMENT_ANALYSIS_VERSION",
    "build_development_analysis",
    "build_single_provider_development_analysis",
]
