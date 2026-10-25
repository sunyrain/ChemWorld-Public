"""Frozen design and analysis for the Work II catalyst-deactivation Q0."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CATALYST_DEACTIVATION_Q0_VERSION = "chemworld-work-ii-catalyst-deactivation-q0-0.1"
WORLD_SEED = 0
TASK_ID = "reaction-safety-constrained"
LAW_IDS = ("deactivating_baseline", "stable_catalyst")
DIRECT_METRICS = ("yield", "conversion", "selectivity")
PRODUCT_METRICS = ("yield", "conversion")
DECLARED_SIGMA = {"yield": 0.012, "conversion": 0.012, "selectivity": 0.018}
FORBIDDEN_PUBLIC_TOKENS = (
    "mechanism_family",
    "world_intervention",
    "private_seed",
    "hidden_state",
    "evaluator_truth",
)


def stable_catalyst_intervention() -> dict[str, Any]:
    return {
        "kind": "mechanism_family",
        "mode": "topology_family",
        "severity": 1.0,
        "topology_change": {
            "reaction_role": "catalyst_deactivation_pathway",
            "transform_id": "stable_catalyst_topology_v1",
        },
    }


def registered_cells() -> list[dict[str, Any]]:
    temperatures = (350.0, 410.0, 465.0)
    durations = (1_800.0, 7_200.0, 14_400.0)
    catalyst_doses = (0.000120, 0.000315, 0.000520)
    return [
        {
            "cell_id": (
                f"temperature-{temperature_index}-duration-{duration_index}"
                f"-dose-{dose_index}"
            ),
            "temperature_index": temperature_index,
            "duration_index": duration_index,
            "dose_index": dose_index,
            "temperature_K": temperature,
            "duration_s": duration,
            "catalyst_amount_mol": dose,
        }
        for temperature_index, temperature in enumerate(temperatures)
        for duration_index, duration in enumerate(durations)
        for dose_index, dose in enumerate(catalyst_doses)
    ]


def analyze(
    rows: Sequence[Mapping[str, Any]],
    mechanism_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "fixed_execution_denominator": len(rows) == 54,
        "all_outcomes_classified": all(
            row.get("status") in {"completed", "physical_failure", "platform_failure"}
            for row in rows
        ),
        "all_exact_replay": all(row.get("exact_replay") is True for row in rows),
        "zero_platform_failures": not any(
            row.get("status") == "platform_failure" for row in rows
        ),
        "paired_action_plans": _paired_equal(rows, "action_plan_sha256"),
        "paired_observation_noise": _paired_equal(rows, "direct_noise_key_sha256"),
        "mechanism_removes_one_deactivation_reaction": (
            mechanism_audit.get("removed_reaction_count") == 1
            and mechanism_audit.get("removed_reaction_id") == "catalyst_deactivation"
        ),
        "mechanism_hash_changes": mechanism_audit.get("mechanism_hash_changed") is True,
        "mechanism_binding_deterministic": (
            mechanism_audit.get("stable_hash_deterministic") is True
        ),
        "execution_mechanism_binding_matches": (
            mechanism_audit.get("execution_mechanism_binding_matches") is True
        ),
    }
    if not all(checks.values()):
        return _early_result(rows, mechanism_audit, checks)

    pairs = _paired_rows(rows)
    completed_pairs = [
        pair
        for pair in pairs
        if pair["deactivating"].get("status") == "completed"
        and pair["stable"].get("status") == "completed"
    ]
    safe_pairs = [
        pair
        for pair in completed_pairs
        if pair["deactivating"].get("safe") is True and pair["stable"].get("safe") is True
    ]
    checks.update(
        {
            "at_least_24_completed_pairs": len(completed_pairs) >= 24,
            "at_least_18_safe_pairs": len(safe_pairs) >= 18,
            "safe_pairs_cover_all_temperature_levels": (
                {int(pair["temperature_index"]) for pair in safe_pairs} == {0, 1, 2}
            ),
            "safe_pairs_cover_all_duration_levels": (
                {int(pair["duration_index"]) for pair in safe_pairs} == {0, 1, 2}
            ),
            "safe_pairs_cover_all_dose_levels": (
                {int(pair["dose_index"]) for pair in safe_pairs} == {0, 1, 2}
            ),
        }
    )
    if not all(checks.values()):
        return _early_result(rows, mechanism_audit, checks)

    completed_rows = [row for row in rows if row.get("status") == "completed"]
    metrics_finite = all(
        all(math.isfinite(float(row["direct_metrics"][metric])) for metric in DIRECT_METRICS)
        for row in completed_rows
    )
    metrics_observed = all(
        all(row["direct_observed_mask"].get(metric) is True for metric in DIRECT_METRICS)
        for row in completed_rows
    )
    leakage_matches = sorted(
        {
            str(token)
            for row in rows
            for token in row.get("participant_visible_leakage_matches", [])
        }
        | {
            token
            for row in rows
            for token in FORBIDDEN_PUBLIC_TOKENS
            if token in json.dumps(row.get("participant_visible_payload", {}), sort_keys=True)
        }
    )
    metric_reports: dict[str, dict[str, Any]] = {}
    for metric in DIRECT_METRICS:
        sigma = DECLARED_SIGMA[metric]
        gate = max(0.05, 3.0 * sigma)
        cell_gaps = [
            {
                "cell_id": pair["cell_id"],
                "temperature_index": pair["temperature_index"],
                "duration_index": pair["duration_index"],
                "dose_index": pair["dose_index"],
                "both_safe": (
                    pair["deactivating"]["safe"] is True
                    and pair["stable"]["safe"] is True
                ),
                "deactivating": float(pair["deactivating"]["direct_metrics"][metric]),
                "stable": float(pair["stable"]["direct_metrics"][metric]),
                "signed_gap": (
                    float(pair["stable"]["direct_metrics"][metric])
                    - float(pair["deactivating"]["direct_metrics"][metric])
                ),
            }
            for pair in completed_pairs
        ]
        maximum = max(abs(cell["signed_gap"]) for cell in cell_gaps)
        metric_reports[metric] = {
            "declared_sigma": sigma,
            "effect_gate": gate,
            "max_absolute_paired_gap": maximum,
            "effect_passed": maximum >= gate,
            "cell_gaps": cell_gaps,
        }

    passing_metric_count = sum(report["effect_passed"] for report in metric_reports.values())
    supporting_cells: set[tuple[int, int, int, str]] = {
        (
            int(cell["temperature_index"]),
            int(cell["duration_index"]),
            int(cell["dose_index"]),
            str(cell["cell_id"]),
        )
        for report in metric_reports.values()
        for cell in report["cell_gaps"]
        if cell["both_safe"] and abs(float(cell["signed_gap"])) >= float(report["effect_gate"])
    }
    separated_support = any(
        abs(left[0] - right[0]) + abs(left[1] - right[1]) + abs(left[2] - right[2]) >= 2
        for item_index, left in enumerate(sorted(supporting_cells))
        for right in sorted(supporting_cells)[item_index + 1 :]
    )
    dose_coverage = len({cell[2] for cell in supporting_cells}) >= 2
    accumulation_reports = {}
    for metric in PRODUCT_METRICS:
        report = metric_reports[metric]
        shortest = [
            float(cell["signed_gap"])
            for cell in report["cell_gaps"]
            if int(cell["duration_index"]) == 0
        ]
        longest = [
            float(cell["signed_gap"])
            for cell in report["cell_gaps"]
            if int(cell["duration_index"]) == 2
        ]
        threshold = max(0.03, 2.0 * float(report["declared_sigma"]))
        increase = float(np.mean(longest) - np.mean(shortest))
        accumulation_reports[metric] = {
            "shortest_duration_mean_signed_gap": float(np.mean(shortest)),
            "longest_duration_mean_signed_gap": float(np.mean(longest)),
            "increase": increase,
            "threshold": threshold,
            "passed": increase >= threshold,
        }

    checks.update(
        {
            "all_direct_metrics_finite": metrics_finite,
            "all_direct_metrics_publicly_observed": metrics_observed,
            "at_least_two_direct_metrics_resolve_topology": passing_metric_count >= 2,
            "two_separated_safe_supporting_cells": separated_support,
            "support_spans_two_catalyst_doses": dose_coverage,
            "duration_accumulation_signature": any(
                report["passed"] for report in accumulation_reports.values()
            ),
            "participant_visible_leakage_free": not leakage_matches,
        }
    )
    return {
        "task_id": TASK_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": _denominators(rows),
        "mechanism_audit": dict(mechanism_audit),
        "metric_reports": metric_reports,
        "passing_metric_count": passing_metric_count,
        "supporting_cells": [cell[3] for cell in sorted(supporting_cells)],
        "separated_support": separated_support,
        "dose_coverage": dose_coverage,
        "accumulation_reports": accumulation_reports,
        "leakage_matches": leakage_matches,
    }


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for cell in registered_cells():
        selected = [row for row in rows if row.get("cell_id") == cell["cell_id"]]
        if len(selected) != len(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} has the wrong paired denominator")
        laws = {str(row["law_id"]): row for row in selected}
        if set(laws) != set(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} lacks its paired laws")
        pairs.append(
            {
                **cell,
                "deactivating": laws["deactivating_baseline"],
                "stable": laws["stable_catalyst"],
            }
        )
    return pairs


def _paired_equal(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    try:
        return all(
            pair["deactivating"].get(field) == pair["stable"].get(field)
            and pair["deactivating"].get(field) is not None
            for pair in _paired_rows(rows)
        )
    except (IndexError, KeyError, ValueError):
        return False


def _denominators(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "attempted": len(rows),
        "completed": sum(row.get("status") == "completed" for row in rows),
        "exact_replay": sum(row.get("exact_replay") is True for row in rows),
        "physical_failures": sum(row.get("status") == "physical_failure" for row in rows),
        "platform_failures": sum(row.get("status") == "platform_failure" for row in rows),
        "unsafe_completed": sum(
            row.get("status") == "completed" and row.get("safe") is False for row in rows
        ),
    }


def _early_result(
    rows: Sequence[Mapping[str, Any]],
    mechanism_audit: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": _denominators(rows),
        "mechanism_audit": dict(mechanism_audit),
        "metric_reports": None,
        "passing_metric_count": 0,
        "supporting_cells": [],
        "separated_support": False,
        "dose_coverage": False,
        "accumulation_reports": None,
        "leakage_matches": [],
    }


__all__ = [
    "CATALYST_DEACTIVATION_Q0_VERSION",
    "DIRECT_METRICS",
    "LAW_IDS",
    "TASK_ID",
    "WORLD_SEED",
    "analyze",
    "registered_cells",
    "stable_catalyst_intervention",
]
