"""Outcome-blind confirmatory inference for the Work II public formal matrix."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import stats

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.eval.work_ii_analysis import (
    WORK_II_ANALYSIS_ARMS,
    WorkIIAnalysisError,
    build_cluster_correction_record,
)
from chemworld.eval.work_ii_report import WORK_II_FORMAL_ANALYSIS_DATASET_VERSION

WORK_II_CONFIRMATORY_ANALYSIS_VERSION = "chemworld-work-ii-confirmatory-analysis-0.1"
EXPECTED_ANALYSIS_PLAN_VERSION = "chemworld-work-ii-analysis-plan-0.3"
EXPECTED_TASK_COUNT = 5
EXPECTED_CLUSTER_COUNT = 25
EXPECTED_CELL_COUNT = 75
BOOTSTRAP_SEED = 20260810
BOOTSTRAP_REPLICATES = 10_000
ONE_SIDED_ALPHA = 0.05
ALIGNED_NONINFERIORITY_MARGIN = -0.05


class WorkIIConfirmatoryAnalysisError(ValueError):
    """Raised when a formal dataset cannot support the frozen analysis."""


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkIIConfirmatoryAnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkIIConfirmatoryAnalysisError(f"{label} must be finite")
    return result


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: value for key, value in payload.items() if key != field})


def _tail_statistics(
    *, estimate: float, standard_error: float, df: int, null: float
) -> dict[str, Any]:
    if df <= 0:
        raise WorkIIConfirmatoryAnalysisError("inference requires positive residual df")
    if standard_error < 0.0 or not math.isfinite(standard_error):
        raise WorkIIConfirmatoryAnalysisError("standard error must be finite and non-negative")
    if standard_error <= 1.0e-15:
        if estimate > null:
            statistic, p_value = math.inf, 0.0
        elif estimate < null:
            statistic, p_value = -math.inf, 1.0
        else:
            statistic, p_value = 0.0, 0.5
        lower = estimate
        ci_low = estimate
        ci_high = estimate
    else:
        statistic = (estimate - null) / standard_error
        p_value = float(stats.t.sf(statistic, df))
        lower = estimate - float(stats.t.ppf(0.95, df)) * standard_error
        half_width = float(stats.t.ppf(0.975, df)) * standard_error
        ci_low = estimate - half_width
        ci_high = estimate + half_width
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "residual_degrees_of_freedom": df,
        "null_value": null,
        "t_statistic": (
            "Infinity"
            if statistic == math.inf
            else "-Infinity"
            if statistic == -math.inf
            else statistic
        ),
        "one_sided_p_value": p_value,
        "one_sided_95pct_lower_bound": lower,
        "two_sided_95pct_interval": [ci_low, ci_high],
        "passed": lower > null,
    }


def _task_fixed_effect_fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    null: float,
    covariance: str,
) -> dict[str, Any]:
    tasks = sorted({str(row.get("task_id", "")) for row in rows})
    if len(tasks) != EXPECTED_TASK_COUNT or "" in tasks:
        return {
            "status": "not_estimable_missing_task_level",
            "method": covariance,
            "row_count": len(rows),
            "one_sided_p_value": 1.0,
            "passed": False,
        }
    values = np.asarray(
        [
            _finite(row.get(value_field), f"{value_field}[{index}]")
            for index, row in enumerate(rows)
        ],
        dtype=float,
    )
    x = np.zeros((len(rows), len(tasks)), dtype=float)
    task_index = {task: index for index, task in enumerate(tasks)}
    for row_index, row in enumerate(rows):
        x[row_index, task_index[str(row["task_id"])]] = 1.0
    rank = int(np.linalg.matrix_rank(x))
    df = len(rows) - rank
    if rank != len(tasks) or df <= 0:
        return {
            "status": "not_estimable_rank_or_df",
            "method": covariance,
            "row_count": len(rows),
            "rank": rank,
            "residual_degrees_of_freedom": df,
            "one_sided_p_value": 1.0,
            "passed": False,
        }
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ values
    residuals = values - x @ beta
    contrast = np.full(len(tasks), 1.0 / len(tasks), dtype=float)
    estimate = float(contrast @ beta)
    if covariance == "classical_OLS":
        sigma2 = float(residuals @ residuals) / df
        covariance_matrix = sigma2 * xtx_inv
    elif covariance == "HC3":
        leverage = np.sum((x @ xtx_inv) * x, axis=1)
        adjusted = residuals / np.maximum(1.0 - leverage, 1.0e-12)
        meat = x.T @ np.diag(adjusted**2) @ x
        covariance_matrix = xtx_inv @ meat @ xtx_inv
    else:
        raise WorkIIConfirmatoryAnalysisError(f"unsupported covariance: {covariance}")
    variance = max(float(contrast @ covariance_matrix @ contrast), 0.0)
    result = _tail_statistics(
        estimate=estimate,
        standard_error=math.sqrt(variance),
        df=df,
        null=null,
    )
    task_values: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        task_values[str(row["task_id"])].append(float(value))
    result.update(
        {
            "status": "estimated",
            "method": covariance,
            "row_count": len(rows),
            "task_count": len(tasks),
            "task_means": {task: float(np.mean(task_values[task])) for task in tasks},
            "task_sample_standard_deviations": {
                task: (
                    float(np.std(task_values[task], ddof=1)) if len(task_values[task]) > 1 else None
                )
                for task in tasks
            },
        }
    )
    return result


def _bootstrap_task_stratified(
    rows: Sequence[Mapping[str, Any]], *, value_field: str, null: float
) -> dict[str, Any]:
    task_values: dict[str, np.ndarray] = {}
    for task in sorted({str(row.get("task_id", "")) for row in rows}):
        values = [
            _finite(row.get(value_field), value_field)
            for row in rows
            if str(row.get("task_id")) == task
        ]
        task_values[task] = np.asarray(values, dtype=float)
    if len(task_values) != EXPECTED_TASK_COUNT or any(
        len(values) == 0 for values in task_values.values()
    ):
        return {
            "status": "not_estimable_missing_task_level",
            "one_sided_p_value": 1.0,
            "passed": False,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    arrays = list(task_values.values())
    for index in range(BOOTSTRAP_REPLICATES):
        task_means = [
            float(np.mean(rng.choice(values, size=len(values), replace=True))) for values in arrays
        ]
        replicates[index] = float(np.mean(task_means))
    estimate = float(np.mean([float(np.mean(values)) for values in arrays]))
    lower = float(np.quantile(replicates, 0.05))
    p_value = float((1 + int(np.sum(replicates <= null))) / (BOOTSTRAP_REPLICATES + 1))
    return {
        "status": "estimated",
        "method": "task_stratified_cluster_bootstrap",
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "row_count": len(rows),
        "estimate": estimate,
        "null_value": null,
        "one_sided_95pct_lower_bound": lower,
        "one_sided_p_value": p_value,
        "passed": lower > null,
    }


def _primary_family(
    rows: Sequence[Mapping[str, Any]], *, covariance: str, failure_aware: bool = True
) -> dict[str, Any]:
    fields = (
        {
            "H3_primary_contrast": "H3_primary_contrast_lower_bound",
            "H3_misindexed_improvement": "H3_misindexed_improvement_lower_bound",
            "H3_aligned_noninferiority": "H3_aligned_improvement_lower_bound",
        }
        if failure_aware
        else {
            "H3_primary_contrast": "H3_primary_contrast",
            "H3_misindexed_improvement": "H3_misindexed_improvement",
            "H3_aligned_noninferiority": "H3_aligned_improvement",
        }
    )
    components = {
        "H3_primary_contrast": _task_fixed_effect_fit(
            rows,
            value_field=fields["H3_primary_contrast"],
            null=0.0,
            covariance=covariance,
        ),
        "H3_misindexed_improvement": _task_fixed_effect_fit(
            rows,
            value_field=fields["H3_misindexed_improvement"],
            null=0.0,
            covariance=covariance,
        ),
        "H3_aligned_noninferiority": _task_fixed_effect_fit(
            rows,
            value_field=fields["H3_aligned_noninferiority"],
            null=ALIGNED_NONINFERIORITY_MARGIN,
            covariance=covariance,
        ),
    }
    p_values = [float(result.get("one_sided_p_value", 1.0)) for result in components.values()]
    passed = all(result.get("passed") is True for result in components.values())
    return {
        "status": "estimated"
        if all(result.get("status") == "estimated" for result in components.values())
        else "not_fully_estimable",
        "method": covariance,
        "estimand": (
            "symmetric_failure_aware_adverse_bounds"
            if failure_aware
            else "observed_point_summary_not_confirmatory"
        ),
        "success_is_intersection_union": True,
        "intersection_union_p_value": max(p_values),
        "passed": passed,
        "components": components,
    }


def _bootstrap_primary_family(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    components = {
        "H3_primary_contrast": _bootstrap_task_stratified(
            rows, value_field="H3_primary_contrast_lower_bound", null=0.0
        ),
        "H3_misindexed_improvement": _bootstrap_task_stratified(
            rows, value_field="H3_misindexed_improvement_lower_bound", null=0.0
        ),
        "H3_aligned_noninferiority": _bootstrap_task_stratified(
            rows,
            value_field="H3_aligned_improvement_lower_bound",
            null=ALIGNED_NONINFERIORITY_MARGIN,
        ),
    }
    return {
        "status": "estimated"
        if all(result.get("status") == "estimated" for result in components.values())
        else "not_fully_estimable",
        "method": "task_stratified_cluster_bootstrap",
        "estimand": "symmetric_failure_aware_adverse_bounds",
        "intersection_union_p_value": max(
            float(result.get("one_sided_p_value", 1.0)) for result in components.values()
        ),
        "passed": all(result.get("passed") is True for result in components.values()),
        "components": components,
    }


def _h4_fit(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    for row in cell_rows:
        checkpoint = row.get("checkpoint_error")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        blind = row.get("blind_outcome")
        blind = blind if isinstance(blind, Mapping) else {}
        improvement = checkpoint.get("primary_improvement")
        gain = blind.get("recommendation_gain_over_incumbent")
        if (
            row.get("terminal_state") == "completed"
            and blind.get("completed_execution_count") == 6
            and isinstance(improvement, int | float)
            and not isinstance(improvement, bool)
            and math.isfinite(float(improvement))
            and isinstance(gain, int | float)
            and not isinstance(gain, bool)
            and math.isfinite(float(gain))
        ):
            eligible.append(dict(row))
    tasks = sorted({str(row.get("task_id", "")) for row in eligible})
    arms = list(WORK_II_ANALYSIS_ARMS)
    groups = sorted({str(row.get("world_cluster_id", "")) for row in eligible})
    parameter_count = 2 + max(len(tasks) - 1, 0) + len(arms) - 1
    if len(tasks) != EXPECTED_TASK_COUNT or len(groups) < 2 or len(eligible) <= parameter_count:
        return {
            "status": "not_estimable_insufficient_complete_blind_cells",
            "eligible_cell_count": len(eligible),
            "eligible_cluster_count": len(groups),
            "one_sided_p_value": 1.0,
            "passed": False,
        }
    task_reference = tasks[0]
    arm_reference = arms[0]
    columns = ["intercept", "epistemic_error_reduction"]
    columns.extend(f"task[{task}]" for task in tasks[1:])
    columns.extend(f"arm[{arm}]" for arm in arms[1:])
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    cluster_ids: list[str] = []
    for row in eligible:
        checkpoint = row["checkpoint_error"]
        blind = row["blind_outcome"]
        task = str(row["task_id"])
        arm = str(row["prior_arm"])
        x_rows.append(
            [1.0, float(checkpoint["primary_improvement"])]
            + [1.0 if task == item else 0.0 for item in tasks[1:]]
            + [1.0 if arm == item else 0.0 for item in arms[1:]]
        )
        y_values.append(float(blind["recommendation_gain_over_incumbent"]))
        cluster_ids.append(str(row["world_cluster_id"]))
    x = np.asarray(x_rows, dtype=float)
    y = np.asarray(y_values, dtype=float)
    rank = int(np.linalg.matrix_rank(x))
    n, k = x.shape
    if rank != k or n <= k:
        return {
            "status": "not_estimable_rank_or_df",
            "eligible_cell_count": n,
            "eligible_cluster_count": len(groups),
            "rank": rank,
            "parameter_count": k,
            "one_sided_p_value": 1.0,
            "passed": False,
        }
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residuals = y - x @ beta
    meat = np.zeros((k, k), dtype=float)
    for cluster_id in groups:
        indexes = [index for index, value in enumerate(cluster_ids) if value == cluster_id]
        x_g = x[indexes, :]
        u_g = residuals[indexes]
        score = x_g.T @ u_g
        meat += np.outer(score, score)
    g = len(groups)
    correction = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    covariance = correction * (xtx_inv @ meat @ xtx_inv)
    estimate = float(beta[1])
    variance = max(float(covariance[1, 1]), 0.0)
    result = _tail_statistics(
        estimate=estimate,
        standard_error=math.sqrt(variance),
        df=g - 1,
        null=0.0,
    )
    result.update(
        {
            "status": "estimated",
            "method": "cell_level_OLS_task_and_prior_fixed_effects_CR1_cluster_robust",
            "eligible_cell_count": n,
            "eligible_cluster_count": g,
            "task_reference": task_reference,
            "prior_arm_reference": arm_reference,
            "coefficient_names": columns,
            "coefficients": [float(value) for value in beta],
        }
    )
    return result


def _holm_family(raw: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        (
            (hypothesis, float(result.get("one_sided_p_value", 1.0)))
            for hypothesis, result in raw.items()
        ),
        key=lambda item: (item[1], item[0]),
    )
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for index, (hypothesis, p_value) in enumerate(ordered):
        candidate = min(1.0, (family_size - index) * p_value)
        running = max(running, candidate)
        adjusted[hypothesis] = running
    return {
        "method": "Holm_familywise_alpha_0.05",
        "family_size": family_size,
        "ordered_raw_p_values": [
            {"hypothesis": hypothesis, "raw_p_value": p_value} for hypothesis, p_value in ordered
        ],
        "results": {
            hypothesis: {
                "raw_p_value": float(result.get("one_sided_p_value", 1.0)),
                "adjusted_p_value": adjusted[hypothesis],
                "rejected": adjusted[hypothesis] <= 0.05,
                "estimability_status": result.get("status"),
            }
            for hypothesis, result in raw.items()
        },
    }


def _phenotypes(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    by_arm: dict[str, Counter[str]] = defaultdict(Counter)
    for row in cell_rows:
        checkpoint = row.get("checkpoint_error")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        blind = row.get("blind_outcome")
        blind = blind if isinstance(blind, Mapping) else {}
        improvement = checkpoint.get("primary_improvement")
        gain = blind.get("recommendation_gain_over_incumbent")
        if (
            row.get("terminal_state") != "completed"
            or blind.get("completed_execution_count") != 6
            or not isinstance(improvement, int | float)
            or isinstance(improvement, bool)
            or not isinstance(gain, int | float)
            or isinstance(gain, bool)
        ):
            label = "unclassified_failed_or_missing"
        elif float(improvement) > 0.0 and float(gain) > 0.0:
            label = "understands_and_acts"
        elif float(improvement) > 0.0:
            label = "understands_but_cannot_act"
        elif float(gain) > 0.0:
            label = "acts_without_understanding"
        else:
            label = "neither"
        counts[label] += 1
        by_arm[str(row.get("prior_arm"))][label] += 1
    return {
        "status": "descriptive_not_confirmatory",
        "thresholds": {
            "epistemic_positive": "primary_improvement_greater_than_zero",
            "action_positive": "blind_recommendation_gain_greater_than_zero",
        },
        "counts": dict(sorted(counts.items())),
        "counts_by_prior_arm": {
            arm: dict(sorted(counter.items())) for arm, counter in sorted(by_arm.items())
        },
    }


def _law_summary_denominators(cell_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: list[Mapping[str, Any]] = []
    for row in cell_rows:
        final_law_summary = row.get("final_law_summary")
        summaries.append(final_law_summary if isinstance(final_law_summary, Mapping) else {})
    evaluated_rows = [
        (row, summary)
        for row, summary in zip(cell_rows, summaries, strict=True)
        if isinstance(summary.get("normalized_mae"), int | float)
        and not isinstance(summary.get("normalized_mae"), bool)
    ]
    errors_by_arm: dict[str, list[float]] = defaultdict(list)
    for row, summary in evaluated_rows:
        errors_by_arm[str(row.get("prior_arm"))].append(float(summary["normalized_mae"]))
    errors = [float(summary["normalized_mae"]) for _, summary in evaluated_rows]
    return {
        "scheduled_cell_count": len(cell_rows),
        "typed_final_summary_present_count": sum(
            summary.get("present") is True for summary in summaries
        ),
        "schema_version_matching_count": sum(
            summary.get("schema_version_matches") is True for summary in summaries
        ),
        "evaluator_executability_evaluated_count": sum(
            not str(summary.get("evaluator_executability_status", "")).startswith(
                "not_evaluated"
            )
            for summary in summaries
        ),
        "evaluator_executability_passed_count": sum(
            summary.get("evaluator_executability_status")
            == "passed_registered_query_execution"
            for summary in summaries
        ),
        "continuous_prediction_validity_evaluated_count": sum(
            str(summary.get("continuous_prediction_validity_status", "")).startswith(
                "evaluated_"
            )
            for summary in summaries
        ),
        "descriptive_normalized_mae": {
            "evaluated_cell_count": len(errors),
            "mean": None if not errors else sum(errors) / len(errors),
            "mean_by_prior_arm": {
                arm: (None if not values else sum(values) / len(values))
                for arm, values in sorted(errors_by_arm.items())
            },
            "formal_test_performed": False,
        },
        "law_discovery_joint_rule_status": (
            "not_established_without_evaluator_executability_and_private_transfer"
        ),
    }


def validate_confirmatory_inputs(
    dataset: Mapping[str, Any], analysis_plan: Mapping[str, Any]
) -> None:
    errors: list[str] = []
    if dataset.get("schema_version") != WORK_II_FORMAL_ANALYSIS_DATASET_VERSION:
        errors.append("unexpected formal analysis dataset schema")
    if dataset.get("dataset_sha256") != _self_hash(dataset, "dataset_sha256"):
        errors.append("formal analysis dataset self-hash mismatch")
    if dataset.get("formal_result") is not True or dataset.get("status") != "passed":
        errors.append("confirmatory analysis requires a passed formal dataset")
    if dataset.get("errors") != []:
        errors.append("formal analysis dataset contains retained construction errors")
    if (
        dataset.get("retained_cell_count") != EXPECTED_CELL_COUNT
        or dataset.get("cluster_contrast_count") != EXPECTED_CLUSTER_COUNT
    ):
        errors.append("formal analysis dataset denominator differs from 75 cells / 25 clusters")
    if analysis_plan.get("schema_version") != EXPECTED_ANALYSIS_PLAN_VERSION:
        errors.append("unexpected Work II analysis-plan version")
    contract = analysis_plan.get("analysis_implementation_contract")
    if not isinstance(contract, Mapping):
        errors.append("analysis plan lacks its implementation contract")
    else:
        if (
            contract.get("expected_cluster_count") != EXPECTED_CLUSTER_COUNT
            or contract.get("expected_cell_count") != EXPECTED_CELL_COUNT
            or contract.get("input_dataset_schema") != WORK_II_FORMAL_ANALYSIS_DATASET_VERSION
        ):
            errors.append("analysis implementation contract denominator or schema mismatch")
    cell_rows = dataset.get("cell_rows")
    cluster_rows = dataset.get("cluster_rows")
    if not isinstance(cell_rows, list) or len(cell_rows) != EXPECTED_CELL_COUNT:
        errors.append("formal analysis dataset does not contain exactly 75 cell rows")
        cell_rows = []
    if not isinstance(cluster_rows, list) or len(cluster_rows) != EXPECTED_CLUSTER_COUNT:
        errors.append("formal analysis dataset does not contain exactly 25 cluster rows")
        cluster_rows = []
    cells_by_cluster: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for index, row in enumerate(cell_rows):
        if not isinstance(row, Mapping):
            errors.append(f"cell_rows[{index}] is not an object")
            continue
        cluster_id = str(row.get("world_cluster_id", ""))
        arm = str(row.get("prior_arm", ""))
        if not cluster_id or arm not in WORK_II_ANALYSIS_ARMS:
            errors.append(f"cell_rows[{index}] has an invalid cluster or arm identity")
            continue
        if arm in cells_by_cluster[cluster_id]:
            errors.append(f"{cluster_id} contains a duplicate {arm} cell")
        cells_by_cluster[cluster_id][arm] = row
        checkpoint = row.get("checkpoint_error")
        if not isinstance(checkpoint, Mapping):
            errors.append(f"{cluster_id}/{arm} lacks checkpoint_error")
            continue
        try:
            improvement = _finite(
                checkpoint.get("primary_improvement"),
                f"{cluster_id}/{arm}.primary_improvement",
            )
            if not -1.0 <= improvement <= 1.0:
                errors.append(f"{cluster_id}/{arm} primary improvement is outside [-1,1]")
            pre_error = checkpoint.get("effective_pre_error")
            if pre_error is not None and not 0.0 <= _finite(
                pre_error, f"{cluster_id}/{arm}.effective_pre_error"
            ) <= 1.0:
                errors.append(f"{cluster_id}/{arm} pre error is outside [0,1]")
        except WorkIIConfirmatoryAnalysisError as error:
            errors.append(str(error))

    cluster_by_id: dict[str, Mapping[str, Any]] = {}
    rebuilt_fields = (
        "H1_prior_utility",
        "H2_prior_vulnerability",
        "H3_misindexed_improvement",
        "H3_aligned_improvement",
        "H3_primary_contrast",
        "H3_primary_contrast_lower_bound",
        "H3_misindexed_improvement_lower_bound",
        "H3_aligned_improvement_lower_bound",
    )
    for index, row in enumerate(cluster_rows):
        if not isinstance(row, Mapping):
            errors.append(f"cluster_rows[{index}] is not an object")
            continue
        cluster_id = str(row.get("world_cluster_id", ""))
        if not cluster_id or cluster_id in cluster_by_id:
            errors.append(f"cluster_rows[{index}] has a missing or duplicate identity")
            continue
        cluster_by_id[cluster_id] = row
        arm_cells = cells_by_cluster.get(cluster_id, {})
        if set(arm_cells) != set(WORK_II_ANALYSIS_ARMS):
            errors.append(f"{cluster_id} does not contain its exact three-arm cell triplet")
            continue
        task_ids = {str(cell.get("task_id", "")) for cell in arm_cells.values()}
        if task_ids != {str(row.get("task_id", ""))}:
            errors.append(f"{cluster_id} task identity differs between cell and cluster rows")
        arm_records = {
            arm: arm_cells[arm].get("checkpoint_error", {})
            for arm in WORK_II_ANALYSIS_ARMS
        }
        try:
            rebuilt = build_cluster_correction_record(arm_records)
        except (WorkIIAnalysisError, TypeError, ValueError) as error:
            errors.append(f"{cluster_id} cannot rebuild frozen contrasts: {error}")
            continue
        for field in rebuilt_fields:
            observed = row.get(field)
            expected = rebuilt[field]
            if observed is None or expected is None:
                matches = observed is expected
            else:
                try:
                    matches = math.isclose(
                        _finite(observed, f"{cluster_id}.{field}"),
                        _finite(expected, f"{cluster_id}.rebuilt.{field}"),
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                except WorkIIConfirmatoryAnalysisError:
                    matches = False
            if not matches:
                errors.append(f"{cluster_id}.{field} differs from its three cell rows")
        expected_complete_case = all(
            arm_cells[arm].get("terminal_state") == "completed"
            and arm_records[arm].get("missing_failure_rule") == "observed_final"
            for arm in WORK_II_ANALYSIS_ARMS
        )
        if row.get("complete_case") is not expected_complete_case:
            errors.append(f"{cluster_id}.complete_case differs from its three cell rows")
    if set(cluster_by_id) != set(cells_by_cluster):
        errors.append("formal analysis cell/cluster identity roster mismatch")
    if errors:
        raise WorkIIConfirmatoryAnalysisError("; ".join(errors))


def build_confirmatory_analysis(
    dataset: Mapping[str, Any], analysis_plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute the frozen public confirmatory analysis without changing its rules."""

    validate_confirmatory_inputs(dataset, analysis_plan)
    cluster_rows = [dict(row) for row in dataset["cluster_rows"]]
    cell_rows = [dict(row) for row in dataset["cell_rows"]]
    primary = _primary_family(cluster_rows, covariance="classical_OLS", failure_aware=True)
    observed_point_summary = _primary_family(
        cluster_rows, covariance="classical_OLS", failure_aware=False
    )
    complete_case_rows = [row for row in cluster_rows if row.get("complete_case") is True]
    complete_case = _primary_family(complete_case_rows, covariance="classical_OLS")
    hc3 = _primary_family(cluster_rows, covariance="HC3", failure_aware=True)
    bootstrap = _bootstrap_primary_family(cluster_rows)

    h1_rows = [
        {**row, "H1_value": 0.0 if row.get("H1_prior_utility") is None else row["H1_prior_utility"]}
        for row in cluster_rows
    ]
    h2_rows = [
        {
            **row,
            "H2_value": 0.0
            if row.get("H2_prior_vulnerability") is None
            else row["H2_prior_vulnerability"],
        }
        for row in cluster_rows
    ]
    h1 = _task_fixed_effect_fit(
        h1_rows, value_field="H1_value", null=0.0, covariance="classical_OLS"
    )
    h1["missing_pair_contrast_zero_imputation_count"] = sum(
        row.get("H1_prior_utility") is None for row in cluster_rows
    )
    h2 = _task_fixed_effect_fit(
        h2_rows, value_field="H2_value", null=0.0, covariance="classical_OLS"
    )
    h2["missing_pair_contrast_zero_imputation_count"] = sum(
        row.get("H2_prior_vulnerability") is None for row in cluster_rows
    )
    h4 = _h4_fit(cell_rows)
    confirmatory_secondary_raw = {
        "H1_prior_utility": h1,
        "H2_prior_vulnerability": h2,
    }
    report: dict[str, Any] = {
        "schema_version": WORK_II_CONFIRMATORY_ANALYSIS_VERSION,
        "status": "completed",
        "formal_result": True,
        "analysis_plan_sha256": canonical_json_sha256(analysis_plan),
        "formal_analysis_dataset_sha256": dataset["dataset_sha256"],
        "formal_preflight_sha256": dataset["formal_preflight_sha256"],
        "denominators": {
            "scheduled_cell_count": EXPECTED_CELL_COUNT,
            "retained_cell_count": len(cell_rows),
            "independent_cluster_count": len(cluster_rows),
            "complete_case_cluster_count": len(complete_case_rows),
            "non_complete_case_cluster_count": len(cluster_rows) - len(complete_case_rows),
            "terminal_state_counts": dict(dataset.get("state_counts", {})),
            "missing_failure_rule_counts": dict(
                sorted(
                    Counter(
                        str(row.get("checkpoint_error", {}).get("missing_failure_rule"))
                        for row in cell_rows
                    ).items()
                )
            ),
        },
        "primary_H3": primary,
        "sensitivity_analyses": {
            "observed_point_summary": observed_point_summary,
            "complete_case": complete_case,
            "HC3": hc3,
            "task_stratified_cluster_bootstrap": bootstrap,
            "primary_decision_is_not_replaced_by_sensitivity_results": True,
        },
        "confirmatory_secondary": {
            "unadjusted": confirmatory_secondary_raw,
            "Holm": _holm_family(confirmatory_secondary_raw),
        },
        "exploratory_H4_knowledge_to_action_translation": {
            **h4,
            "confirmatory": False,
            "interpretation": (
                "descriptive_only_due_to_post_assignment_eligibility_"
                "and_no_separate_power"
            ),
        },
        "descriptive_task_heterogeneity": {
            "H3_primary_contrast_task_means": primary.get("components", {})
            .get("H3_primary_contrast", {})
            .get("task_means", {}),
            "H3_primary_contrast_task_sample_standard_deviations": primary.get("components", {})
            .get("H3_primary_contrast", {})
            .get("task_sample_standard_deviations", {}),
            "agent_model_variance_status": "not_estimable_single_frozen_agent_method",
            "provider_session_variance_status": "not_separately_estimable_one_session_per_cell",
            "provider_repeat_variance_status": "not_estimable_one_nested_repeat_per_cell",
        },
        "descriptive_joint_phenotypes": _phenotypes(cell_rows),
        "law_summary_and_transfer_boundary": _law_summary_denominators(cell_rows),
        "claim_decisions": {
            "selective_evidence_driven_wrong_prior_correction": primary.get("passed") is True,
            "knowledge_to_action_translation": "not_confirmatory_exploratory_only",
            "reusable_law_discovery": False,
            "private_transfer": "not_collected_by_public_analysis",
            "single_leaderboard_score_used": False,
        },
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def validate_confirmatory_analysis(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != WORK_II_CONFIRMATORY_ANALYSIS_VERSION:
        errors.append("unexpected confirmatory analysis schema")
    if report.get("report_sha256") != _self_hash(report, "report_sha256"):
        errors.append("confirmatory analysis self-hash mismatch")
    if report.get("status") != "completed" or report.get("formal_result") is not True:
        errors.append("confirmatory analysis did not reach a completed formal state")
    denominators = report.get("denominators")
    denominators = denominators if isinstance(denominators, Mapping) else {}
    if (
        denominators.get("scheduled_cell_count") != EXPECTED_CELL_COUNT
        or denominators.get("retained_cell_count") != EXPECTED_CELL_COUNT
        or denominators.get("independent_cluster_count") != EXPECTED_CLUSTER_COUNT
    ):
        errors.append("confirmatory analysis denominator mismatch")
    if report.get("claim_decisions", {}).get("single_leaderboard_score_used") is not False:
        errors.append("confirmatory analysis collapsed outcomes into a leaderboard score")
    return errors


__all__ = [
    "WORK_II_CONFIRMATORY_ANALYSIS_VERSION",
    "WorkIIConfirmatoryAnalysisError",
    "build_confirmatory_analysis",
    "validate_confirmatory_analysis",
    "validate_confirmatory_inputs",
]
