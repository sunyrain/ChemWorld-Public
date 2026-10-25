"""Frozen design and analysis for the Work II static-topology Q0 screen."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

STATIC_TOPOLOGY_Q0_VERSION = "chemworld-work-ii-static-topology-q0-0.1"
WORLD_SEED = 0
LAW_IDS = ("baseline", "reversible_target_pathway")
FORBIDDEN_PUBLIC_TOKENS = (
    "mechanism_family",
    "world_intervention",
    "private_seed",
    "hidden_state",
    "evaluator_truth",
)


def topology_intervention() -> dict[str, Any]:
    return {
        "kind": "mechanism_family",
        "mode": "topology_family",
        "severity": 0.8,
        "topology_change": {
            "reaction_role": "primary_target_pathway",
            "transform_id": "reversible_target_pathway_stress_v1",
            "reverse_rate_constant_s_inv_at_full_severity": 0.000625,
        },
    }


def task_specs() -> dict[str, dict[str, Any]]:
    return {
        "reaction-to-crystallization": {
            "temperature_levels": (350.0, 385.0, 420.0),
            "time_levels": (1200.0, 3600.0, 7200.0),
            "time_field": "reaction_duration_s",
            "direct_instrument": "hplc",
            "direct_metrics": ("yield", "conversion", "selectivity"),
            "product_metrics": ("yield", "conversion"),
            "terminal_metrics": ("crystal_yield", "score"),
            "declared_sigma": {
                "yield": 0.012,
                "conversion": 0.012,
                "selectivity": 0.018,
            },
            "objective": "balanced",
            "crystallization_material_family_id": (
                "reaction-crystallization-latent-materials-v1"
            ),
        },
        "flow-reaction-optimization": {
            "temperature_levels": (350.0, 390.0, 425.0),
            "time_levels": (300.0, 900.0, 1800.0),
            "time_field": "residence_time_s",
            "direct_instrument": "uvvis",
            "direct_metrics": ("yield", "selectivity", "flow_conversion"),
            "product_metrics": ("yield", "flow_conversion"),
            "terminal_metrics": ("yield", "flow_conversion", "score"),
            "declared_sigma": {
                "yield": 0.045,
                "selectivity": 0.040,
                "flow_conversion": 0.040,
            },
            "objective": "balanced",
            "crystallization_material_family_id": None,
        },
    }


def registered_cells(task_id: str) -> list[dict[str, Any]]:
    spec = task_specs()[task_id]
    return [
        {
            "cell_id": f"temperature-{temperature_index}-time-{time_index}",
            "temperature_index": temperature_index,
            "time_index": time_index,
            "temperature_K": float(temperature),
            "time_s": float(time_s),
        }
        for temperature_index, temperature in enumerate(spec["temperature_levels"])
        for time_index, time_s in enumerate(spec["time_levels"])
    ]


def analyze_task(
    task_id: str,
    rows: Sequence[Mapping[str, Any]],
    mechanism_audit: Mapping[str, Any],
) -> dict[str, Any]:
    spec = task_specs()[task_id]
    checks: dict[str, bool] = {
        "fixed_execution_denominator": len(rows) == 18,
        "all_completed": all(row.get("status") == "completed" for row in rows),
        "all_exact_replay": all(row.get("exact_replay") is True for row in rows),
        "zero_physical_failures": not any(
            row.get("status") == "physical_failure" for row in rows
        ),
        "zero_platform_failures": not any(
            row.get("status") == "platform_failure" for row in rows
        ),
        "paired_action_plans": _paired_equal(rows, "action_plan_sha256"),
        "paired_observation_noise": _paired_equal(rows, "direct_noise_key_sha256"),
        "mechanism_adds_one_reverse_reaction": (
            mechanism_audit.get("added_reaction_count") == 1
        ),
        "mechanism_hash_changes": mechanism_audit.get("mechanism_hash_changed") is True,
        "mechanism_binding_deterministic": (
            mechanism_audit.get("reversible_hash_deterministic") is True
        ),
        "execution_mechanism_binding_matches": (
            mechanism_audit.get("execution_mechanism_binding_matches") is True
        ),
    }
    if not all(checks.values()):
        return _early_result(task_id, rows, mechanism_audit, checks)

    metrics_finite = all(
        all(
            math.isfinite(float(row["direct_metrics"][metric]))
            for metric in spec["direct_metrics"]
        )
        for row in rows
    )
    metrics_observed = all(
        all(row["direct_observed_mask"].get(metric) is True for metric in spec["direct_metrics"])
        for row in rows
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
    pairs = _paired_rows(rows)
    metric_reports: dict[str, dict[str, Any]] = {}
    for metric in spec["direct_metrics"]:
        sigma = float(spec["declared_sigma"][metric])
        gate = max(0.05, 3.0 * sigma)
        cell_gaps = [
            {
                "cell_id": pair["cell_id"],
                "temperature_index": pair["temperature_index"],
                "time_index": pair["time_index"],
                "baseline": float(pair["baseline"]["direct_metrics"][metric]),
                "reversible": float(pair["reversible"]["direct_metrics"][metric]),
                "signed_gap": (
                    float(pair["baseline"]["direct_metrics"][metric])
                    - float(pair["reversible"]["direct_metrics"][metric])
                ),
            }
            for pair in pairs
        ]
        max_absolute_gap = max(abs(row["signed_gap"]) for row in cell_gaps)
        metric_reports[metric] = {
            "declared_sigma": sigma,
            "effect_gate": gate,
            "max_absolute_paired_gap": max_absolute_gap,
            "effect_passed": max_absolute_gap >= gate,
            "cell_gaps": cell_gaps,
        }

    passing_metric_count = sum(
        report["effect_passed"] for report in metric_reports.values()
    )
    supporting_cells = sorted(
        {
            (
                int(cell["temperature_index"]),
                int(cell["time_index"]),
                str(cell["cell_id"]),
            )
            for report in metric_reports.values()
            for cell in report["cell_gaps"]
            if abs(float(cell["signed_gap"])) >= float(report["effect_gate"])
        }
    )
    separated_support = any(
        abs(left[0] - right[0]) + abs(left[1] - right[1]) >= 2
        for index, left in enumerate(supporting_cells)
        for right in supporting_cells[index + 1 :]
    )
    accumulation_reports = {}
    for metric in spec["product_metrics"]:
        report = metric_reports[metric]
        shortest = [
            float(cell["signed_gap"])
            for cell in report["cell_gaps"]
            if int(cell["time_index"]) == 0
        ]
        longest = [
            float(cell["signed_gap"])
            for cell in report["cell_gaps"]
            if int(cell["time_index"]) == 2
        ]
        threshold = max(0.03, 2.0 * float(report["declared_sigma"]))
        increase = float(np.mean(longest) - np.mean(shortest))
        accumulation_reports[metric] = {
            "shortest_time_mean_signed_gap": float(np.mean(shortest)),
            "longest_time_mean_signed_gap": float(np.mean(longest)),
            "increase": increase,
            "threshold": threshold,
            "passed": increase >= threshold,
        }

    checks.update(
        {
            "all_direct_metrics_finite": metrics_finite,
            "all_direct_metrics_publicly_observed": metrics_observed,
            "at_least_two_direct_metrics_resolve_topology": passing_metric_count >= 2,
            "two_separated_supporting_cells": separated_support,
            "duration_accumulation_signature": any(
                report["passed"] for report in accumulation_reports.values()
            ),
            "participant_visible_leakage_free": not leakage_matches,
        }
    )
    return {
        "task_id": task_id,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": _denominators(rows),
        "mechanism_audit": dict(mechanism_audit),
        "metric_reports": metric_reports,
        "passing_metric_count": passing_metric_count,
        "supporting_cells": [item[2] for item in supporting_cells],
        "separated_support": separated_support,
        "accumulation_reports": accumulation_reports,
        "leakage_matches": leakage_matches,
    }


def _paired_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("static-topology Q0 rows are empty")
    pairs = []
    for cell in registered_cells(str(rows[0]["task_id"])):
        selected = [row for row in rows if row.get("cell_id") == cell["cell_id"]]
        if len(selected) != len(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} has the wrong paired denominator")
        laws = {str(row["law_id"]): row for row in selected}
        if set(laws) != set(LAW_IDS):
            raise ValueError(f"cell {cell['cell_id']} lacks its paired laws")
        pairs.append(
            {
                **cell,
                "baseline": laws["baseline"],
                "reversible": laws["reversible_target_pathway"],
            }
        )
    return pairs


def _paired_equal(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    try:
        return all(
            pair["baseline"].get(field) == pair["reversible"].get(field)
            and pair["baseline"].get(field) is not None
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
    task_id: str,
    rows: Sequence[Mapping[str, Any]],
    mechanism_audit: Mapping[str, Any],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": _denominators(rows),
        "mechanism_audit": dict(mechanism_audit),
        "metric_reports": None,
        "passing_metric_count": 0,
        "supporting_cells": [],
        "separated_support": False,
        "accumulation_reports": None,
        "leakage_matches": [],
    }


__all__ = [
    "LAW_IDS",
    "STATIC_TOPOLOGY_Q0_VERSION",
    "WORLD_SEED",
    "analyze_task",
    "registered_cells",
    "task_specs",
    "topology_intervention",
]
