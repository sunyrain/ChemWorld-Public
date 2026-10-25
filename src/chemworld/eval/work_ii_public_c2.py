"""Manifest-bound cross-locus confirmatory analysis for the public Work II C2 scope.

This module is deliberately separate from the frozen A-E formal analysis.  It defines
the missing cell-report contract for terminal A-P and A-S blocks and combines the
three locus decisions with an intersection-union rule.  It never pools the nine task
families as exchangeable observations.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_analysis import (
    WORK_II_ANALYSIS_ARMS,
    WorkIIAnalysisError,
    build_cluster_correction_record,
)
from chemworld.eval.work_ii_confirmatory import validate_confirmatory_analysis
from chemworld.eval.work_ii_formal import FORMAL_TERMINAL_STATES

PUBLIC_C2_MANIFEST_VERSION = "chemworld-work-ii-public-c2-analysis-manifest-0.1"
PUBLIC_C2_LOCUS_REPORT_VERSION = "chemworld-work-ii-public-c2-locus-cell-report-0.1"
PUBLIC_C2_ANALYSIS_VERSION = "chemworld-work-ii-public-c2-confirmatory-analysis-0.1"
LOCUS_IDS = ("A_E", "A_P", "A_S")
EXPECTED_TASK_COUNTS = {"A_E": 5, "A_P": 2, "A_S": 2}
EXPECTED_CLUSTER_COUNTS = {"A_E": 25, "A_P": 10, "A_S": 10}
EXPECTED_CELL_COUNTS = {"A_E": 75, "A_P": 30, "A_S": 30}
EXPECTED_WORLDS_PER_TASK = 5
ONE_SIDED_ALPHA = 0.05
ALIGNED_NONINFERIORITY_MARGIN = -0.05
FAILURE_AWARE_IMPROVEMENT_BOUNDS = (-1.0, 1.0)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40,64}$")

ANALYSIS_CONTRACT = {
    "analysis_unit": "matched_task_x_world_cluster",
    "prior_arms": list(WORK_II_ANALYSIS_ARMS),
    "one_sided_alpha": ONE_SIDED_ALPHA,
    "failure_aware_arm_bounds": list(FAILURE_AWARE_IMPROVEMENT_BOUNDS),
    "A_E_gate": "existing_H3_three_component_intersection_union_recomputed_from_bound_rows",
    "A_P_gate": "two_task_fixed_effect_C_prior_adverse_lower_bound_and_both_task_means_positive",
    "A_S_gate": "two_task_fixed_effect_C_prior_adverse_lower_bound_and_both_task_means_positive",
    "C2_success_rule": "intersection_union_all_A_E_A_P_A_S_locus_gates",
    "naive_nine_task_pooling": False,
    "provider_calls_for_analysis": 0,
}
PUBLIC_C2_PLAN_CONTRACT = {
    "status": "outcome_blind_contract_frozen_no_results",
    "input_locus_cell_report_schema": PUBLIC_C2_LOCUS_REPORT_VERSION,
    "analysis_provider_call_count": 0,
    "prior_arms": list(WORK_II_ANALYSIS_ARMS),
    "failure_aware_arm_bounds": list(FAILURE_AWARE_IMPROVEMENT_BOUNDS),
    "locus_denominators": {
        locus: {
            "tasks": EXPECTED_TASK_COUNTS[locus],
            "worlds_per_task": EXPECTED_WORLDS_PER_TASK,
            "matched_task_world_clusters": EXPECTED_CLUSTER_COUNTS[locus],
            "scheduled_and_retained_cells": EXPECTED_CELL_COUNTS[locus],
        }
        for locus in LOCUS_IDS
    },
    "locus_gates": {
        "A_E": {
            "rule": (
                "existing_H3_three_component_intersection_union_"
                "recomputed_from_bound_cell_rows"
            ),
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "aligned_noninferiority_margin": ALIGNED_NONINFERIORITY_MARGIN,
        },
        "A_P": {
            "rule": "task_fixed_effect_C_prior_adverse_lower_bound_greater_than_zero",
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "all_two_task_means_must_be_positive": True,
        },
        "A_S": {
            "rule": "task_fixed_effect_C_prior_adverse_lower_bound_greater_than_zero",
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "all_two_task_means_must_be_positive": True,
        },
    },
    "task_fixed_effect_inference": {
        "design_matrix": "task_indicator_columns_without_intercept",
        "estimand": "equal_weighted_mean_of_task_specific_cluster_means",
        "covariance": "classical_OLS",
        "reference_distribution": "Student_t_residual_df",
        "A_E_residual_degrees_of_freedom": 20,
        "A_P_residual_degrees_of_freedom": 8,
        "A_S_residual_degrees_of_freedom": 8,
        "exact_zero_standard_error_limit": (
            "se_le_1e-15_treated_as_numerically_zero_"
            "p_0_if_estimate_above_null_p_0.5_if_equal_p_1_if_below"
        ),
        "nonfinite_or_materially_negative_variance": "fail_closed",
    },
    "global_intersection_union": {
        "required_loci": list(LOCUS_IDS),
        "all_three_locus_gates_must_pass": True,
        "overall_p_value": "maximum_of_A_E_A_P_A_S_effective_gate_p_values",
        "direction_inconsistent_A_P_or_A_S_effective_p_value": 1.0,
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "naive_nine_task_pooling_forbidden": True,
    },
}
ANALYSIS_SOURCE_PATHS = (
    "scripts/analyze_work_ii_public_c2.py",
    "src/chemworld/eval/work_ii_public_c2.py",
)


class WorkIIPublicC2AnalysisError(ValueError):
    """Raised when bound public C2 inputs cannot support frozen inference."""


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != field}
    )


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkIIPublicC2AnalysisError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise WorkIIPublicC2AnalysisError(f"{label} must be finite")
    return result


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_public_c2_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate the outcome-blind cross-locus analysis manifest."""

    errors: list[str] = []
    if manifest.get("schema_version") != PUBLIC_C2_MANIFEST_VERSION:
        errors.append("unexpected public C2 analysis-manifest schema")
    if manifest.get("manifest_sha256") != _self_hash(manifest, "manifest_sha256"):
        errors.append("public C2 analysis manifest self-hash mismatch")
    if manifest.get("program_scope") != "C2" or manifest.get("status") != "frozen":
        errors.append("public C2 analysis manifest is not a frozen C2 contract")
    if manifest.get("formal_result") is not True:
        errors.append("public C2 analysis manifest is not marked formal")
    if manifest.get("analysis_contract") != ANALYSIS_CONTRACT:
        errors.append("public C2 analysis contract drifted")
    if not _is_sha256(manifest.get("analysis_plan_sha256")):
        errors.append("public C2 analysis-plan binding is invalid")
    runtime_commit = manifest.get("runtime_commit")
    if not isinstance(runtime_commit, str) or _COMMIT.fullmatch(runtime_commit) is None:
        errors.append("public C2 runtime commit binding is invalid")
    sources = manifest.get("analysis_source_sha256")
    if not isinstance(sources, Mapping) or set(sources) != set(ANALYSIS_SOURCE_PATHS):
        errors.append("public C2 analysis source roster is invalid")
    elif not all(_is_sha256(sources[path]) for path in ANALYSIS_SOURCE_PATHS):
        errors.append("public C2 analysis source hash is invalid")
    blocks = manifest.get("blocks")
    if not isinstance(blocks, Mapping) or set(blocks) != set(LOCUS_IDS):
        errors.append("public C2 manifest must contain exactly A_E, A_P, and A_S")
        return errors
    report_ids: set[str] = set()
    for locus in LOCUS_IDS:
        block = blocks.get(locus)
        if not isinstance(block, Mapping):
            errors.append(f"{locus} manifest block is not an object")
            continue
        expected = {
            "locus_id": locus,
            "expected_task_count": EXPECTED_TASK_COUNTS[locus],
            "worlds_per_task": EXPECTED_WORLDS_PER_TASK,
            "expected_cluster_count": EXPECTED_CLUSTER_COUNTS[locus],
            "expected_cell_count": EXPECTED_CELL_COUNTS[locus],
        }
        for field, value in expected.items():
            if block.get(field) != value:
                errors.append(f"{locus} manifest {field} mismatch")
        task_ids = block.get("task_ids")
        if (
            not isinstance(task_ids, list)
            or len(task_ids) != EXPECTED_TASK_COUNTS[locus]
            or len(set(task_ids)) != EXPECTED_TASK_COUNTS[locus]
            or not all(isinstance(task, str) and task for task in task_ids)
        ):
            errors.append(f"{locus} manifest task roster is invalid")
        report_id = block.get("analysis_report_id")
        if not isinstance(report_id, str) or not report_id or report_id in report_ids:
            errors.append(f"{locus} analysis report identity is invalid or duplicated")
        else:
            report_ids.add(report_id)
        if not _is_sha256(block.get("execution_manifest_sha256")):
            errors.append(f"{locus} execution-manifest binding is invalid")
        legacy_report_hash = block.get("legacy_confirmatory_report_sha256")
        if locus == "A_E":
            if not _is_sha256(legacy_report_hash):
                errors.append("A_E existing confirmatory-report binding is invalid")
        elif legacy_report_hash is not None:
            errors.append(f"{locus} must not carry an A_E confirmatory-report binding")
    return errors


def validate_public_c2_source_files(
    manifest: Mapping[str, Any], *, root: Path
) -> list[str]:
    """Verify that the frozen analysis-source hashes match materialized files."""

    errors = validate_public_c2_manifest(manifest)
    if errors:
        return errors
    resolved_root = root.resolve()
    sources = manifest["analysis_source_sha256"]
    for relative in ANALYSIS_SOURCE_PATHS:
        try:
            path = (resolved_root / relative).resolve()
            if not path.is_relative_to(resolved_root) or not path.is_file():
                raise ValueError("source is outside the repository or missing")
            if sources[relative] != file_sha256(path):
                errors.append(f"public C2 analysis source hash mismatch: {relative}")
        except (OSError, TypeError, ValueError) as error:
            errors.append(f"public C2 analysis source cannot be read: {relative}: {error}")
    return errors


def validate_public_c2_analysis_plan(
    manifest: Mapping[str, Any], analysis_plan: Mapping[str, Any]
) -> list[str]:
    """Verify the manifest hash and frozen C2 extension in the analysis plan."""

    errors = validate_public_c2_manifest(manifest)
    if errors:
        return errors
    if manifest.get("analysis_plan_sha256") != canonical_json_sha256(analysis_plan):
        errors.append("public C2 analysis-plan hash binding mismatch")
    if analysis_plan.get("schema_version") != "chemworld-work-ii-analysis-plan-0.3":
        errors.append("public C2 analysis plan has an unexpected schema")
    if analysis_plan.get("public_C2_confirmatory_extension") != PUBLIC_C2_PLAN_CONTRACT:
        errors.append("public C2 analysis-plan extension drifted")
    return errors


def _row_validation_errors(
    row: Mapping[str, Any], *, locus: str, index: int
) -> list[str]:
    label = f"{locus}.cell_rows[{index}]"
    errors: list[str] = []
    required_text = ("cell_id", "world_cluster_id", "task_id", "prior_arm")
    for field in required_text:
        if not isinstance(row.get(field), str) or not str(row[field]):
            errors.append(f"{label}.{field} is missing")
    if row.get("prior_arm") not in WORK_II_ANALYSIS_ARMS:
        errors.append(f"{label}.prior_arm is invalid")
    if row.get("terminal_state") not in FORMAL_TERMINAL_STATES:
        errors.append(f"{label}.terminal_state is invalid")
    world_seed = row.get("world_seed")
    if isinstance(world_seed, bool) or not isinstance(world_seed, int) or world_seed < 0:
        errors.append(f"{label}.world_seed is invalid")
    if not _is_sha256(row.get("terminal_receipt_sha256")):
        errors.append(f"{label}.terminal_receipt_sha256 is invalid")
    checkpoint = row.get("checkpoint_error")
    if not isinstance(checkpoint, Mapping):
        errors.append(f"{label}.checkpoint_error is missing")
        return errors
    try:
        improvement = _finite(
            checkpoint.get("primary_improvement"), f"{label}.primary_improvement"
        )
        if not -1.0 <= improvement <= 1.0:
            errors.append(f"{label}.primary_improvement is outside [-1,1]")
        raw_bounds = checkpoint.get("confirmatory_improvement_bounds")
        if (
            isinstance(raw_bounds, (str, bytes))
            or not isinstance(raw_bounds, Sequence)
            or len(raw_bounds) != 2
        ):
            errors.append(f"{label}.confirmatory_improvement_bounds is invalid")
        else:
            bounds = (
                _finite(raw_bounds[0], f"{label}.bound_lower"),
                _finite(raw_bounds[1], f"{label}.bound_upper"),
            )
            if bounds[0] > bounds[1] or bounds[0] < -1.0 or bounds[1] > 1.0:
                errors.append(f"{label}.confirmatory bounds are outside [-1,1]")
            if row.get("terminal_state") == "completed":
                if bounds != (improvement, improvement):
                    errors.append(f"{label} completed outcome does not use a point bound")
                if checkpoint.get("missing_failure_rule") != "observed_final":
                    errors.append(f"{label} completed outcome lacks observed_final binding")
            elif bounds != FAILURE_AWARE_IMPROVEMENT_BOUNDS:
                errors.append(f"{label} failed outcome lacks symmetric adverse bounds")
    except WorkIIPublicC2AnalysisError as error:
        errors.append(str(error))
    return errors


def validate_locus_cell_report(
    report: Mapping[str, Any],
    *,
    locus: str,
    manifest: Mapping[str, Any],
) -> list[str]:
    """Validate one manifest-bound matched-three-arm locus report."""

    errors: list[str] = []
    block = manifest.get("blocks", {}).get(locus, {})
    if report.get("schema_version") != PUBLIC_C2_LOCUS_REPORT_VERSION:
        errors.append(f"{locus} has an unexpected locus cell-report schema")
    if report.get("report_sha256") != _self_hash(report, "report_sha256"):
        errors.append(f"{locus} locus report self-hash mismatch")
    if report.get("formal_result") is not True or report.get("status") != "passed":
        errors.append(f"{locus} locus report is not a passed formal report")
    if report.get("errors") != []:
        errors.append(f"{locus} locus report contains retained construction errors")
    if report.get("analysis_provider_call_count") != 0:
        errors.append(f"{locus} locus analysis must be provider-free")
    if report.get("locus_id") != locus:
        errors.append(f"{locus} locus identity mismatch")
    if report.get("report_id") != block.get("analysis_report_id"):
        errors.append(f"{locus} report identity differs from its C2 manifest")
    if report.get("source_c2_manifest_sha256") != manifest.get("manifest_sha256"):
        errors.append(f"{locus} report does not bind the C2 analysis manifest")
    if report.get("execution_manifest_sha256") != block.get(
        "execution_manifest_sha256"
    ):
        errors.append(f"{locus} report execution-manifest binding mismatch")
    if report.get("runtime_commit") != manifest.get("runtime_commit"):
        errors.append(f"{locus} report runtime commit mismatch")
    if report.get("analysis_source_manifest_sha256") != canonical_json_sha256(
        manifest.get("analysis_source_sha256")
    ):
        errors.append(f"{locus} analysis-source manifest binding mismatch")
    if not _is_sha256(report.get("source_analysis_dataset_sha256")):
        errors.append(f"{locus} source analysis-dataset binding is invalid")

    rows = report.get("cell_rows")
    if not isinstance(rows, list):
        errors.append(f"{locus} locus report cell_rows is not a list")
        return errors
    if len(rows) != EXPECTED_CELL_COUNTS[locus]:
        errors.append(f"{locus} cell denominator mismatch")
    ids: set[str] = set()
    receipt_ids: set[str] = set()
    clusters: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            errors.append(f"{locus}.cell_rows[{index}] is not an object")
            continue
        errors.extend(_row_validation_errors(row, locus=locus, index=index))
        cell_id = str(row.get("cell_id", ""))
        if cell_id in ids:
            errors.append(f"{locus} contains duplicate cell identity {cell_id}")
        ids.add(cell_id)
        receipt_id = str(row.get("terminal_receipt_sha256", ""))
        if receipt_id in receipt_ids:
            errors.append(f"{locus} contains duplicate terminal receipt binding")
        receipt_ids.add(receipt_id)
        clusters[str(row.get("world_cluster_id", ""))].append(row)
    if len(clusters) != EXPECTED_CLUSTER_COUNTS[locus] or "" in clusters:
        errors.append(f"{locus} cluster denominator or identity is invalid")
    task_clusters: dict[str, set[str]] = defaultdict(set)
    task_worlds: dict[str, set[int]] = defaultdict(set)
    for cluster_id, members in clusters.items():
        arms = [str(row.get("prior_arm")) for row in members]
        tasks = {str(row.get("task_id", "")) for row in members}
        worlds = {row.get("world_seed") for row in members}
        if Counter(arms) != Counter(WORK_II_ANALYSIS_ARMS):
            errors.append(f"{locus}/{cluster_id} lacks one exact matched arm triplet")
        if len(tasks) != 1 or "" in tasks or len(worlds) != 1:
            errors.append(f"{locus}/{cluster_id} task/world identity differs across arms")
            continue
        task = next(iter(tasks))
        world = next(iter(worlds))
        if isinstance(world, int):
            task_clusters[task].add(cluster_id)
            task_worlds[task].add(world)
    expected_tasks = set(block.get("task_ids", []))
    if set(task_clusters) != expected_tasks:
        errors.append(f"{locus} report task roster differs from its C2 manifest")
    for task in sorted(expected_tasks):
        if (
            len(task_clusters.get(task, set())) != EXPECTED_WORLDS_PER_TASK
            or len(task_worlds.get(task, set())) != EXPECTED_WORLDS_PER_TASK
        ):
            errors.append(f"{locus}/{task} does not contain five distinct world clusters")
    if report.get("scheduled_cell_count") != EXPECTED_CELL_COUNTS[locus]:
        errors.append(f"{locus} scheduled cell denominator mismatch")
    if report.get("retained_cell_count") != len(rows):
        errors.append(f"{locus} retained cell denominator mismatch")
    if report.get("independent_cluster_count") != len(clusters):
        errors.append(f"{locus} independent cluster denominator mismatch")

    legacy = report.get("legacy_A_E_confirmatory_analysis")
    if locus == "A_E":
        if not isinstance(legacy, Mapping):
            errors.append("A_E locus report lacks its existing confirmatory analysis")
        else:
            errors.extend(
                f"A_E legacy: {error}"
                for error in validate_confirmatory_analysis(legacy)
            )
            if legacy.get("formal_analysis_dataset_sha256") != report.get(
                "source_analysis_dataset_sha256"
            ):
                errors.append("A_E legacy analysis-dataset binding mismatch")
            if legacy.get("formal_preflight_sha256") != report.get(
                "execution_manifest_sha256"
            ):
                errors.append("A_E legacy execution-manifest binding mismatch")
            if legacy.get("report_sha256") != block.get(
                "legacy_confirmatory_report_sha256"
            ):
                errors.append("A_E legacy report differs from its C2 manifest binding")
    elif legacy is not None:
        errors.append(f"{locus} must not substitute an A_E legacy decision")
    return errors


def build_locus_cell_report(
    *,
    manifest: Mapping[str, Any],
    locus: str,
    source_analysis_dataset_sha256: str,
    cell_rows: Sequence[Mapping[str, Any]],
    legacy_A_E_confirmatory_analysis: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one immutable locus report from manifest-bound rows."""

    if locus not in LOCUS_IDS:
        raise WorkIIPublicC2AnalysisError(f"unknown public C2 locus: {locus}")
    manifest_errors = validate_public_c2_manifest(manifest)
    if manifest_errors:
        raise WorkIIPublicC2AnalysisError("; ".join(manifest_errors))
    block = manifest["blocks"][locus]
    rows = [dict(row) for row in cell_rows]
    report: dict[str, Any] = {
        "schema_version": PUBLIC_C2_LOCUS_REPORT_VERSION,
        "report_id": block["analysis_report_id"],
        "locus_id": locus,
        "status": "passed",
        "formal_result": True,
        "errors": [],
        "analysis_provider_call_count": 0,
        "source_c2_manifest_sha256": manifest["manifest_sha256"],
        "execution_manifest_sha256": block["execution_manifest_sha256"],
        "source_analysis_dataset_sha256": source_analysis_dataset_sha256,
        "runtime_commit": manifest["runtime_commit"],
        "analysis_source_manifest_sha256": canonical_json_sha256(
            manifest["analysis_source_sha256"]
        ),
        "scheduled_cell_count": EXPECTED_CELL_COUNTS[locus],
        "retained_cell_count": len(rows),
        "independent_cluster_count": len(
            {str(row.get("world_cluster_id", "")) for row in rows}
        ),
        "cell_rows": rows,
        "legacy_A_E_confirmatory_analysis": (
            dict(legacy_A_E_confirmatory_analysis)
            if legacy_A_E_confirmatory_analysis is not None
            else None
        ),
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    errors = validate_locus_cell_report(report, locus=locus, manifest=manifest)
    if errors:
        raise WorkIIPublicC2AnalysisError("; ".join(errors))
    return report


def build_public_c2_cluster_rows(
    locus: str,
    cell_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rebuild one locus's matched-cluster estimands without release metadata.

    The scientific estimand belongs to the retained cell rows.  Formal manifests
    remain release/provenance authorities, but are not required to recompute the
    registered C2 contrasts from an independently bound current composite.
    """

    if locus not in LOCUS_IDS:
        raise WorkIIPublicC2AnalysisError(f"unknown public C2 locus: {locus}")
    if len(cell_rows) != EXPECTED_CELL_COUNTS[locus]:
        raise WorkIIPublicC2AnalysisError(f"{locus} cell denominator mismatch")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cell_rows:
        grouped[str(row["world_cluster_id"])].append(row)
    if len(grouped) != EXPECTED_CLUSTER_COUNTS[locus] or "" in grouped:
        raise WorkIIPublicC2AnalysisError(f"{locus} cluster denominator mismatch")
    records = []
    for cluster_id in sorted(grouped):
        cells = grouped[cluster_id]
        if len(cells) != len(WORK_II_ANALYSIS_ARMS):
            raise WorkIIPublicC2AnalysisError(
                f"{locus}/{cluster_id} does not contain one three-arm triplet"
            )
        by_arm = {str(row["prior_arm"]): row for row in cells}
        if set(by_arm) != set(WORK_II_ANALYSIS_ARMS):
            raise WorkIIPublicC2AnalysisError(
                f"{locus}/{cluster_id} lacks the registered prior arms"
            )
        task_ids = {str(row["task_id"]) for row in cells}
        world_seeds = {row["world_seed"] for row in cells}
        if len(task_ids) != 1 or len(world_seeds) != 1:
            raise WorkIIPublicC2AnalysisError(
                f"{locus}/{cluster_id} task/world identity drifts across arms"
            )
        arm_records = {
            arm: by_arm[arm]["checkpoint_error"] for arm in WORK_II_ANALYSIS_ARMS
        }
        try:
            contrast = build_cluster_correction_record(arm_records)
        except (WorkIIAnalysisError, KeyError, TypeError, ValueError) as error:
            raise WorkIIPublicC2AnalysisError(
                f"{locus}/{cluster_id} cannot rebuild C_prior: {error}"
            ) from error
        first = cells[0]
        records.append(
            {
                "locus_id": locus,
                "world_cluster_id": cluster_id,
                "task_id": first["task_id"],
                "world_seed": first["world_seed"],
                "complete_case": all(
                    row["terminal_state"] == "completed" for row in cells
                ),
                **contrast,
            }
        )
    task_worlds: dict[str, set[int]] = defaultdict(set)
    for row in records:
        task_worlds[str(row["task_id"])].add(int(row["world_seed"]))
    if len(task_worlds) != EXPECTED_TASK_COUNTS[locus] or any(
        len(worlds) != EXPECTED_WORLDS_PER_TASK for worlds in task_worlds.values()
    ):
        raise WorkIIPublicC2AnalysisError(f"{locus} task/world roster mismatch")
    return records


def _cluster_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    return build_public_c2_cluster_rows(
        str(report["locus_id"]),
        report["cell_rows"],
    )


def _task_fixed_effect(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    expected_task_count: int,
    null: float,
) -> dict[str, Any]:
    tasks = sorted({str(row["task_id"]) for row in rows})
    if len(tasks) != expected_task_count:
        raise WorkIIPublicC2AnalysisError(
            f"{value_field} requires exactly {expected_task_count} task levels"
        )
    values = np.asarray(
        [_finite(row.get(value_field), value_field) for row in rows], dtype=float
    )
    x = np.zeros((len(rows), len(tasks)), dtype=float)
    task_index = {task: index for index, task in enumerate(tasks)}
    for index, row in enumerate(rows):
        x[index, task_index[str(row["task_id"])]] = 1.0
    rank = int(np.linalg.matrix_rank(x))
    df = len(rows) - rank
    if rank != len(tasks) or df <= 0:
        raise WorkIIPublicC2AnalysisError(f"{value_field} inference is rank deficient")
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ values
    residuals = values - x @ beta
    contrast = np.full(len(tasks), 1.0 / len(tasks), dtype=float)
    estimate = float(contrast @ beta)
    sigma2 = float(residuals @ residuals) / df
    variance = float(contrast @ (sigma2 * xtx_inv) @ contrast)
    if not math.isfinite(variance) or variance < -1.0e-15:
        raise WorkIIPublicC2AnalysisError(
            f"{value_field} inference variance is invalid"
        )
    variance = max(variance, 0.0)
    standard_error = math.sqrt(variance)
    if standard_error <= 1.0e-15:
        statistic = math.inf if estimate > null else -math.inf if estimate < null else 0.0
        p_value = 0.0 if estimate > null else 1.0 if estimate < null else 0.5
        lower = estimate
    else:
        statistic = (estimate - null) / standard_error
        p_value = float(stats.t.sf(statistic, df))
        lower = estimate - float(stats.t.ppf(1.0 - ONE_SIDED_ALPHA, df)) * standard_error
    task_values: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        task_values[str(row["task_id"])].append(float(value))
    return {
        "status": "estimated",
        "method": "task_fixed_effect_OLS_equal_weighted_task_mean",
        "value_field": value_field,
        "row_count": len(rows),
        "task_count": len(tasks),
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
        "one_sided_alpha": ONE_SIDED_ALPHA,
        "one_sided_p_value": p_value,
        "one_sided_95pct_lower_bound": lower,
        "task_means": {
            task: float(np.mean(task_values[task])) for task in tasks
        },
        "passed": lower > null and p_value <= ONE_SIDED_ALPHA,
    }


def _same_inference(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if left.get("passed") is not right.get("passed"):
        return False
    if (
        left.get("status") != right.get("status")
        or left.get("row_count") != right.get("row_count")
        or left.get("task_count") != right.get("task_count")
        or left.get("residual_degrees_of_freedom")
        != right.get("residual_degrees_of_freedom")
        or left.get("t_statistic") != right.get("t_statistic")
    ):
        return False
    left_task_means = left.get("task_means")
    right_task_means = right.get("task_means")
    if (
        not isinstance(left_task_means, Mapping)
        or not isinstance(right_task_means, Mapping)
        or set(left_task_means) != set(right_task_means)
    ):
        return False
    try:
        if any(
            not math.isclose(
                _finite(left_task_means[task], f"{task} task mean"),
                _finite(right_task_means[task], f"{task} task mean"),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            for task in left_task_means
        ):
            return False
    except WorkIIPublicC2AnalysisError:
        return False
    for field in (
        "estimate",
        "standard_error",
        "one_sided_p_value",
        "one_sided_95pct_lower_bound",
        "null_value",
    ):
        try:
            if not math.isclose(
                _finite(left.get(field), field),
                _finite(right.get(field), field),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                return False
        except WorkIIPublicC2AnalysisError:
            return False
    return True


def build_public_c2_locus_gate(
    locus: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen failure-aware gate to one locus's cluster rows."""

    if locus not in LOCUS_IDS:
        raise WorkIIPublicC2AnalysisError(f"unknown public C2 locus: {locus}")
    if len(rows) != EXPECTED_CLUSTER_COUNTS[locus]:
        raise WorkIIPublicC2AnalysisError(f"{locus} cluster denominator mismatch")
    if locus != "A_E":
        inference = _task_fixed_effect(
            rows,
            value_field="H3_primary_contrast_lower_bound",
            expected_task_count=2,
            null=0.0,
        )
        directions = {
            task: value > 0.0 for task, value in inference["task_means"].items()
        }
        direction_consistent = all(directions.values())
        passed = inference["passed"] is True and direction_consistent
        return {
            "gate_id": f"{locus}_C_prior_adverse_lower_bound",
            "method": "two_task_fixed_effect_C_prior_lower_bound",
            "failure_aware": True,
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "inference": inference,
            "task_direction_positive": directions,
            "both_tasks_direction_consistent": direction_consistent,
            "effective_intersection_union_p_value": (
                float(inference["one_sided_p_value"]) if direction_consistent else 1.0
            ),
            "passed": passed,
        }

    components = {
        "H3_primary_contrast": _task_fixed_effect(
            rows,
            value_field="H3_primary_contrast_lower_bound",
            expected_task_count=5,
            null=0.0,
        ),
        "H3_misindexed_improvement": _task_fixed_effect(
            rows,
            value_field="H3_misindexed_improvement_lower_bound",
            expected_task_count=5,
            null=0.0,
        ),
        "H3_aligned_noninferiority": _task_fixed_effect(
            rows,
            value_field="H3_aligned_improvement_lower_bound",
            expected_task_count=5,
            null=ALIGNED_NONINFERIORITY_MARGIN,
        ),
    }
    p_value = max(float(result["one_sided_p_value"]) for result in components.values())
    return {
        "gate_id": "A_E_current_H3",
        "method": "three_component_H3_intersection_union",
        "failure_aware": True,
        "components": components,
        "intersection_union_p_value": p_value,
        "passed": all(result["passed"] for result in components.values()),
    }


def _ae_gate(
    rows: Sequence[Mapping[str, Any]], legacy: Mapping[str, Any]
) -> dict[str, Any]:
    current = build_public_c2_locus_gate("A_E", rows)
    components = current["components"]
    legacy_primary = legacy.get("primary_H3")
    legacy_primary = legacy_primary if isinstance(legacy_primary, Mapping) else {}
    legacy_components = legacy_primary.get("components")
    legacy_components = legacy_components if isinstance(legacy_components, Mapping) else {}
    binding_matches = all(
        isinstance(legacy_components.get(component), Mapping)
        and _same_inference(result, legacy_components[component])
        for component, result in components.items()
    )
    recomputed_pass = current["passed"] is True
    legacy_pass = legacy_primary.get("passed") is True
    if not binding_matches or recomputed_pass is not legacy_pass:
        raise WorkIIPublicC2AnalysisError(
            "A_E existing H3 decision differs from its manifest-bound cell rows"
        )
    p_value = float(current["intersection_union_p_value"])
    try:
        legacy_p_value = _finite(
            legacy_primary.get("intersection_union_p_value"),
            "A_E legacy intersection-union p value",
        )
    except WorkIIPublicC2AnalysisError as error:
        raise WorkIIPublicC2AnalysisError(
            "A_E existing H3 intersection-union p value is invalid"
        ) from error
    legacy_claim = legacy.get("claim_decisions", {}).get(
        "selective_evidence_driven_wrong_prior_correction"
    )
    if (
        legacy_primary.get("status") != "estimated"
        or legacy_primary.get("method") != "classical_OLS"
        or legacy_primary.get("estimand") != "symmetric_failure_aware_adverse_bounds"
        or legacy_primary.get("success_is_intersection_union") is not True
        or not math.isclose(legacy_p_value, p_value, rel_tol=0.0, abs_tol=1.0e-12)
        or legacy_claim is not legacy_pass
    ):
        raise WorkIIPublicC2AnalysisError(
            "A_E existing H3 report structure or claim decision is inconsistent"
        )
    return {
        "gate_id": "A_E_existing_H3",
        "method": "existing_three_component_H3_intersection_union",
        "failure_aware": True,
        "legacy_report_sha256": legacy["report_sha256"],
        "legacy_decision_reproduced_from_bound_rows": True,
        "components": components,
        "intersection_union_p_value": p_value,
        "passed": legacy_pass and recomputed_pass,
    }


def _terminal_locus_gate(
    locus: str, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return build_public_c2_locus_gate(locus, rows)


def _failure_records(
    reports: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    failures = []
    for locus in LOCUS_IDS:
        for row in reports[locus]["cell_rows"]:
            if row["terminal_state"] == "completed":
                continue
            failures.append(
                {
                    "locus_id": locus,
                    "task_id": row["task_id"],
                    "world_cluster_id": row["world_cluster_id"],
                    "world_seed": row["world_seed"],
                    "cell_id": row["cell_id"],
                    "prior_arm": row["prior_arm"],
                    "terminal_state": row["terminal_state"],
                    "terminal_reason_code": row.get("terminal_reason_code"),
                    "missing_failure_rule": row["checkpoint_error"][
                        "missing_failure_rule"
                    ],
                    "terminal_receipt_sha256": row["terminal_receipt_sha256"],
                    "confirmatory_improvement_bounds": row["checkpoint_error"][
                        "confirmatory_improvement_bounds"
                    ],
                }
            )
    return sorted(
        failures,
        key=lambda row: (row["locus_id"], row["task_id"], row["cell_id"]),
    )


def build_public_c2_analysis(
    manifest: Mapping[str, Any],
    locus_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the provider-free public C2 intersection-union analysis."""

    errors = validate_public_c2_manifest(manifest)
    if set(locus_reports) != set(LOCUS_IDS):
        errors.append("public C2 inputs must contain exactly A_E, A_P, and A_S reports")
    else:
        for locus in LOCUS_IDS:
            errors.extend(
                validate_locus_cell_report(
                    locus_reports[locus], locus=locus, manifest=manifest
                )
            )
        rows = [
            row
            for locus in LOCUS_IDS
            for row in locus_reports[locus].get("cell_rows", [])
            if isinstance(row, Mapping)
        ]
        if len({row.get("cell_id") for row in rows}) != sum(
            EXPECTED_CELL_COUNTS.values()
        ):
            errors.append("public C2 cell identities are not globally unique")
        if len({row.get("terminal_receipt_sha256") for row in rows}) != sum(
            EXPECTED_CELL_COUNTS.values()
        ):
            errors.append("public C2 terminal receipt bindings are not globally unique")
    if errors:
        raise WorkIIPublicC2AnalysisError("; ".join(errors))

    cluster_rows = {locus: _cluster_rows(locus_reports[locus]) for locus in LOCUS_IDS}
    legacy = locus_reports["A_E"]["legacy_A_E_confirmatory_analysis"]
    locus_gates = {
        "A_E": _ae_gate(cluster_rows["A_E"], legacy),
        "A_P": _terminal_locus_gate("A_P", cluster_rows["A_P"]),
        "A_S": _terminal_locus_gate("A_S", cluster_rows["A_S"]),
    }
    failures = _failure_records(locus_reports)
    locus_denominators = {}
    terminal_state_counts: Counter[str] = Counter()
    for locus in LOCUS_IDS:
        rows = locus_reports[locus]["cell_rows"]
        states = Counter(str(row["terminal_state"]) for row in rows)
        terminal_state_counts.update(states)
        locus_denominators[locus] = {
            "task_count": EXPECTED_TASK_COUNTS[locus],
            "independent_cluster_count": len(cluster_rows[locus]),
            "scheduled_cell_count": EXPECTED_CELL_COUNTS[locus],
            "retained_cell_count": len(rows),
            "complete_case_cluster_count": sum(
                row["complete_case"] is True for row in cluster_rows[locus]
            ),
            "non_complete_case_cluster_count": sum(
                row["complete_case"] is not True for row in cluster_rows[locus]
            ),
            "terminal_state_counts": {
                state: states.get(state, 0) for state in sorted(FORMAL_TERMINAL_STATES)
            },
        }
    p_values = {
        "A_E": float(locus_gates["A_E"]["intersection_union_p_value"]),
        "A_P": float(locus_gates["A_P"]["effective_intersection_union_p_value"]),
        "A_S": float(locus_gates["A_S"]["effective_intersection_union_p_value"]),
    }
    c2_passed = all(locus_gates[locus]["passed"] is True for locus in LOCUS_IDS)
    report: dict[str, Any] = {
        "schema_version": PUBLIC_C2_ANALYSIS_VERSION,
        "status": "completed",
        "formal_result": True,
        "analysis_provider_call_count": 0,
        "program_scope": "public_C2",
        "analysis_contract": ANALYSIS_CONTRACT,
        "source_bindings": {
            "c2_manifest_sha256": manifest["manifest_sha256"],
            "analysis_plan_sha256": manifest["analysis_plan_sha256"],
            "runtime_commit": manifest["runtime_commit"],
            "analysis_source_sha256": dict(manifest["analysis_source_sha256"]),
            "analysis_source_manifest_sha256": canonical_json_sha256(
                manifest["analysis_source_sha256"]
            ),
            "execution_manifest_sha256_by_locus": {
                locus: locus_reports[locus]["execution_manifest_sha256"]
                for locus in LOCUS_IDS
            },
            "analysis_dataset_sha256_by_locus": {
                locus: locus_reports[locus]["source_analysis_dataset_sha256"]
                for locus in LOCUS_IDS
            },
            "locus_report_sha256": {
                locus: locus_reports[locus]["report_sha256"] for locus in LOCUS_IDS
            },
            "A_E_legacy_confirmatory_report_sha256": legacy["report_sha256"],
        },
        "denominators": {
            "task_count": sum(EXPECTED_TASK_COUNTS.values()),
            "independent_cluster_count": sum(EXPECTED_CLUSTER_COUNTS.values()),
            "scheduled_cell_count": sum(EXPECTED_CELL_COUNTS.values()),
            "retained_cell_count": sum(
                len(locus_reports[locus]["cell_rows"]) for locus in LOCUS_IDS
            ),
            "failure_count": len(failures),
            "terminal_state_counts": {
                state: terminal_state_counts.get(state, 0)
                for state in sorted(FORMAL_TERMINAL_STATES)
            },
            "by_locus": locus_denominators,
        },
        "all_retained_failures": failures,
        "locus_results": {
            locus: {
                "cluster_rows": cluster_rows[locus],
                "gate": locus_gates[locus],
            }
            for locus in LOCUS_IDS
        },
        "C2_intersection_union": {
            "required_loci": list(LOCUS_IDS),
            "locus_gate_passed": {
                locus: locus_gates[locus]["passed"] for locus in LOCUS_IDS
            },
            "locus_p_values": p_values,
            "intersection_union_p_value": max(p_values.values()),
            "one_sided_alpha": ONE_SIDED_ALPHA,
            "all_three_locus_gates_required": True,
            "naive_nine_task_pooling_performed": False,
            "passed": c2_passed,
        },
        "claim_decision": {
            "cross_locus_initial_world_model_effects_supported": c2_passed,
            "highest_public_claim_scope": "C2" if c2_passed else "below_C2",
        },
    }
    report["report_sha256"] = _self_hash(report, "report_sha256")
    return report


def validate_public_c2_analysis(
    report: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any] | None = None,
    locus_reports: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Validate a C2 output, optionally rebuilding it from all bound raw inputs."""

    errors: list[str] = []
    if report.get("schema_version") != PUBLIC_C2_ANALYSIS_VERSION:
        errors.append("unexpected public C2 confirmatory-analysis schema")
    if report.get("report_sha256") != _self_hash(report, "report_sha256"):
        errors.append("public C2 confirmatory analysis self-hash mismatch")
    if report.get("status") != "completed" or report.get("formal_result") is not True:
        errors.append("public C2 confirmatory analysis did not complete formally")
    if report.get("analysis_provider_call_count") != 0:
        errors.append("public C2 confirmatory analysis must be provider-free")
    if report.get("program_scope") != "public_C2":
        errors.append("public C2 program scope mismatch")
    if report.get("analysis_contract") != ANALYSIS_CONTRACT:
        errors.append("public C2 output analysis contract drifted")
    bindings = report.get("source_bindings")
    bindings = bindings if isinstance(bindings, Mapping) else {}
    if not _is_sha256(bindings.get("c2_manifest_sha256")):
        errors.append("public C2 source manifest binding is invalid")
    if not _is_sha256(bindings.get("analysis_plan_sha256")):
        errors.append("public C2 source analysis-plan binding is invalid")
    runtime_commit = bindings.get("runtime_commit")
    if not isinstance(runtime_commit, str) or _COMMIT.fullmatch(runtime_commit) is None:
        errors.append("public C2 source runtime commit binding is invalid")
    bound_sources = bindings.get("analysis_source_sha256")
    if (
        not isinstance(bound_sources, Mapping)
        or set(bound_sources) != set(ANALYSIS_SOURCE_PATHS)
        or not all(_is_sha256(bound_sources.get(path)) for path in ANALYSIS_SOURCE_PATHS)
    ):
        errors.append("public C2 source-code binding roster is invalid")
    elif bindings.get("analysis_source_manifest_sha256") != canonical_json_sha256(
        bound_sources
    ):
        errors.append("public C2 source-code manifest binding mismatch")
    for field in (
        "execution_manifest_sha256_by_locus",
        "analysis_dataset_sha256_by_locus",
        "locus_report_sha256",
    ):
        values = bindings.get(field)
        if (
            not isinstance(values, Mapping)
            or set(values) != set(LOCUS_IDS)
            or not all(_is_sha256(values.get(locus)) for locus in LOCUS_IDS)
        ):
            errors.append(f"public C2 {field} binding roster is invalid")
    if not _is_sha256(bindings.get("A_E_legacy_confirmatory_report_sha256")):
        errors.append("public C2 A_E legacy report binding is invalid")
    denominators = report.get("denominators")
    denominators = denominators if isinstance(denominators, Mapping) else {}
    if (
        denominators.get("task_count") != 9
        or denominators.get("independent_cluster_count") != 45
        or denominators.get("scheduled_cell_count") != 135
        or denominators.get("retained_cell_count") != 135
    ):
        errors.append("public C2 exact denominator mismatch")
    failures = report.get("all_retained_failures")
    if not isinstance(failures, list) or denominators.get("failure_count") != len(failures):
        errors.append("public C2 retained failure denominator mismatch")
    terminal_counts = denominators.get("terminal_state_counts")
    if (
        not isinstance(terminal_counts, Mapping)
        or set(terminal_counts) != set(FORMAL_TERMINAL_STATES)
        or any(
            isinstance(terminal_counts.get(state), bool)
            or not isinstance(terminal_counts.get(state), int)
            or terminal_counts[state] < 0
            for state in FORMAL_TERMINAL_STATES
        )
        or sum(terminal_counts.values()) != 135
    ):
        errors.append("public C2 terminal-state denominator mismatch")
    elif denominators.get("failure_count") != sum(
        terminal_counts[state]
        for state in FORMAL_TERMINAL_STATES
        if state != "completed"
    ):
        errors.append("public C2 noncompleted-state failure denominator mismatch")
    by_locus = denominators.get("by_locus")
    if not isinstance(by_locus, Mapping) or set(by_locus) != set(LOCUS_IDS):
        errors.append("public C2 by-locus denominator roster mismatch")
    else:
        for locus in LOCUS_IDS:
            values = by_locus.get(locus)
            values = values if isinstance(values, Mapping) else {}
            if (
                values.get("task_count") != EXPECTED_TASK_COUNTS[locus]
                or values.get("independent_cluster_count")
                != EXPECTED_CLUSTER_COUNTS[locus]
                or values.get("scheduled_cell_count") != EXPECTED_CELL_COUNTS[locus]
                or values.get("retained_cell_count") != EXPECTED_CELL_COUNTS[locus]
            ):
                errors.append(f"public C2 {locus} exact denominator mismatch")
    c2 = report.get("C2_intersection_union")
    c2 = c2 if isinstance(c2, Mapping) else {}
    if c2.get("required_loci") != list(LOCUS_IDS):
        errors.append("public C2 required-locus order or roster mismatch")
    if c2.get("one_sided_alpha") != ONE_SIDED_ALPHA:
        errors.append("public C2 one-sided alpha mismatch")
    if c2.get("all_three_locus_gates_required") is not True:
        errors.append("public C2 no longer requires all three locus gates")
    gates = c2.get("locus_gate_passed")
    p_values = c2.get("locus_p_values")
    if not isinstance(gates, Mapping) or set(gates) != set(LOCUS_IDS):
        errors.append("public C2 locus-gate roster mismatch")
    if not isinstance(p_values, Mapping) or set(p_values) != set(LOCUS_IDS):
        errors.append("public C2 locus p-value roster mismatch")
    else:
        try:
            numeric_p_values = {
                locus: _finite(p_values[locus], locus) for locus in LOCUS_IDS
            }
            if any(not 0.0 <= value <= 1.0 for value in numeric_p_values.values()):
                errors.append("public C2 locus p value is outside [0,1]")
            expected_p = max(numeric_p_values.values())
            if not math.isclose(
                _finite(c2.get("intersection_union_p_value"), "C2 p value"),
                expected_p,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                errors.append("public C2 intersection-union p-value mismatch")
        except WorkIIPublicC2AnalysisError as error:
            errors.append(str(error))
    expected_pass = (
        isinstance(gates, Mapping)
        and all(gates.get(locus) is True for locus in LOCUS_IDS)
    )
    if c2.get("passed") is not expected_pass:
        errors.append("public C2 intersection-union decision mismatch")
    if c2.get("naive_nine_task_pooling_performed") is not False:
        errors.append("public C2 analysis naively pooled nine task families")
    locus_results = report.get("locus_results")
    if not isinstance(locus_results, Mapping) or set(locus_results) != set(LOCUS_IDS):
        errors.append("public C2 locus-result roster mismatch")
    elif isinstance(gates, Mapping) and isinstance(p_values, Mapping):
        for locus in LOCUS_IDS:
            result = locus_results.get(locus)
            result = result if isinstance(result, Mapping) else {}
            gate = result.get("gate")
            gate = gate if isinstance(gate, Mapping) else {}
            p_field = (
                "intersection_union_p_value"
                if locus == "A_E"
                else "effective_intersection_union_p_value"
            )
            if gate.get("passed") is not gates.get(locus):
                errors.append(f"public C2 {locus} embedded gate decision mismatch")
            try:
                if not math.isclose(
                    _finite(gate.get(p_field), f"{locus} embedded gate p value"),
                    _finite(p_values.get(locus), f"{locus} locus p value"),
                    rel_tol=0.0,
                    abs_tol=1.0e-15,
                ):
                    errors.append(f"public C2 {locus} embedded gate p-value mismatch")
            except WorkIIPublicC2AnalysisError as error:
                errors.append(str(error))
    claim = report.get("claim_decision")
    claim = claim if isinstance(claim, Mapping) else {}
    if claim.get("cross_locus_initial_world_model_effects_supported") is not expected_pass:
        errors.append("public C2 claim decision mismatch")
    expected_scope = "C2" if expected_pass else "below_C2"
    if claim.get("highest_public_claim_scope") != expected_scope:
        errors.append("public C2 highest public claim scope mismatch")
    if manifest is not None or locus_reports is not None:
        if manifest is None or locus_reports is None:
            errors.append("public C2 raw validation requires manifest and all locus reports")
        else:
            try:
                rebuilt = build_public_c2_analysis(manifest, locus_reports)
            except WorkIIPublicC2AnalysisError as error:
                errors.append(f"public C2 raw inputs cannot rebuild: {error}")
            else:
                if report != rebuilt:
                    errors.append("public C2 report differs from its bound raw inputs")
    return errors


__all__ = [
    "ANALYSIS_CONTRACT",
    "ANALYSIS_SOURCE_PATHS",
    "LOCUS_IDS",
    "PUBLIC_C2_ANALYSIS_VERSION",
    "PUBLIC_C2_LOCUS_REPORT_VERSION",
    "PUBLIC_C2_MANIFEST_VERSION",
    "PUBLIC_C2_PLAN_CONTRACT",
    "WorkIIPublicC2AnalysisError",
    "build_locus_cell_report",
    "build_public_c2_analysis",
    "build_public_c2_cluster_rows",
    "build_public_c2_locus_gate",
    "validate_locus_cell_report",
    "validate_public_c2_analysis",
    "validate_public_c2_analysis_plan",
    "validate_public_c2_manifest",
    "validate_public_c2_source_files",
]
