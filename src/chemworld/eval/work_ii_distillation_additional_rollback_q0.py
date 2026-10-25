"""Frozen design and analysis for distillation additional-rollback A-S Q0."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

QUALIFICATION_VERSION = "chemworld-work-ii-distillation-additional-rollback-q0-0.1"
TASK_ID = "reaction-to-distillation"
WORLD_SEED = 0
LAW_IDS = ("native_reversible_network", "native_plus_additional_rollback")
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


def registered_cells() -> list[dict[str, Any]]:
    return [
        {
            "cell_id": f"temperature-{temperature_index}-time-{time_index}",
            "temperature_index": temperature_index,
            "time_index": time_index,
            "temperature_K": float(temperature),
            "time_s": float(time_s),
        }
        for temperature_index, temperature in enumerate((350.0, 385.0, 420.0))
        for time_index, time_s in enumerate((1200.0, 3600.0, 7200.0))
    ]


def analyze(
    rows: Sequence[Mapping[str, Any]], mechanism_audit: Mapping[str, Any]
) -> dict[str, Any]:
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
        "native_target_reaction_is_reversible": (
            mechanism_audit.get("native_target_reaction_is_reversible") is True
        ),
        "native_target_reaction_preserved": (
            mechanism_audit.get("native_target_reaction_preserved") is True
        ),
        "adds_exactly_one_rollback_reaction": (
            mechanism_audit.get("added_reaction_count") == 1
            and mechanism_audit.get("added_reaction_id") == "family_reverse_channel"
            and mechanism_audit.get("added_reaction_reactants")
            == {"Ester": 1.0, "Water": 1.0}
            and mechanism_audit.get("added_reaction_products")
            == {"Acid": 1.0, "Alcohol": 1.0}
        ),
        "additional_rollback_rate_is_frozen": math.isclose(
            float(mechanism_audit.get("effective_reverse_rate_constant_s_inv", math.nan)),
            0.0005,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "mechanism_hash_changes": mechanism_audit.get("mechanism_hash_changed") is True,
        "mechanism_binding_deterministic": (
            mechanism_audit.get("intervention_hash_deterministic") is True
        ),
        "execution_mechanism_binding_matches": (
            mechanism_audit.get("execution_mechanism_binding_matches") is True
        ),
    }
    if not all(checks.values()):
        return _early_result(rows, mechanism_audit, checks)

    metrics_finite = all(
        all(math.isfinite(float(row["direct_metrics"][metric])) for metric in DIRECT_METRICS)
        for row in rows
    )
    metrics_observed = all(
        all(row["direct_observed_mask"].get(metric) is True for metric in DIRECT_METRICS)
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
    for metric in DIRECT_METRICS:
        sigma = DECLARED_SIGMA[metric]
        gate = max(0.05, 3.0 * sigma)
        cell_gaps = [
            {
                "cell_id": pair["cell_id"],
                "temperature_index": pair["temperature_index"],
                "time_index": pair["time_index"],
                "native_reversible_network": float(
                    pair["native_reversible_network"]["direct_metrics"][metric]
                ),
                "native_plus_additional_rollback": float(
                    pair["native_plus_additional_rollback"]["direct_metrics"][metric]
                ),
                "signed_gap": float(
                    pair["native_reversible_network"]["direct_metrics"][metric]
                )
                - float(pair["native_plus_additional_rollback"]["direct_metrics"][metric]),
            }
            for pair in pairs
        ]
        maximum = max(abs(row["signed_gap"]) for row in cell_gaps)
        metric_reports[metric] = {
            "declared_sigma": sigma,
            "effect_gate": gate,
            "max_absolute_paired_gap": maximum,
            "effect_passed": maximum >= gate,
            "cell_gaps": cell_gaps,
        }

    passing_metric_count = sum(
        report["effect_passed"] for report in metric_reports.values()
    )
    supporting_cells = sorted(
        {
            (int(cell["temperature_index"]), int(cell["time_index"]), str(cell["cell_id"]))
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
    for metric in PRODUCT_METRICS:
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
        "task_id": TASK_ID,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, value in checks.items() if not value),
        "denominators": denominators(rows),
        "mechanism_audit": dict(mechanism_audit),
        "metric_reports": metric_reports,
        "passing_metric_count": passing_metric_count,
        "supporting_cells": [item[2] for item in supporting_cells],
        "separated_support": separated_support,
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
        pairs.append({**cell, **laws})
    return pairs


def _paired_equal(rows: Sequence[Mapping[str, Any]], field: str) -> bool:
    try:
        return all(
            pair[LAW_IDS[0]].get(field) == pair[LAW_IDS[1]].get(field)
            and pair[LAW_IDS[0]].get(field) is not None
            for pair in _paired_rows(rows)
        )
    except (KeyError, ValueError):
        return False


def denominators(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "planned": 18,
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
        "denominators": denominators(rows),
        "mechanism_audit": dict(mechanism_audit),
        "metric_reports": None,
        "passing_metric_count": 0,
        "supporting_cells": [],
        "separated_support": False,
        "accumulation_reports": None,
        "leakage_matches": [],
    }


__all__ = [
    "DECLARED_SIGMA",
    "DIRECT_METRICS",
    "LAW_IDS",
    "QUALIFICATION_VERSION",
    "TASK_ID",
    "WORLD_SEED",
    "analyze",
    "denominators",
    "registered_cells",
    "topology_intervention",
]
