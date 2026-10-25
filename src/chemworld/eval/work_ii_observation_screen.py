"""Design and analysis for the Work II observation-layer development screen."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chemworld.eval.work_ii_structural_candidate_qualification import candidate_specs

OBSERVATION_SCREEN_VERSION = "chemworld-work-ii-observation-screen-0.1"
REPLICATES = 3
EFFECT_FLOOR = 0.03
SIGMA_MULTIPLIER = 3.0
FORBIDDEN_PUBLIC_TOKENS = (
    "private_seed",
    "hidden_state",
    "evaluator_truth",
    "provider_path",
)


def screen_specs() -> dict[str, dict[str, Any]]:
    structural = candidate_specs()
    electro = structural["electrochemical_transport"]
    crystal = structural["crystallization_nucleation_growth"]
    return {
        "electrochemical_observation": {
            "task_id": electro["task_id"],
            "config": electro["config"],
            "fixed_context": {
                **electro["fixed_context"],
                "controlled_potential_V": 1.05,
            },
            "level_field": "controlled_current_mA",
            "levels": (15.0, 91.0, 190.0),
            "metrics": tuple(electro["metrics"]),
            "effect_metrics": (
                "selective_product_yield",
                "faradaic_efficiency",
                "transport_efficiency",
            ),
        },
        "crystallization_observation": {
            "task_id": crystal["task_id"],
            "config": crystal["config"],
            "fixed_context": {
                **crystal["fixed_context"],
                "crystallization_temperature_K": 290.0,
            },
            "level_field": "seed_mass_g",
            "levels": (0.001, 0.008, 0.015),
            "metrics": tuple(crystal["metrics"]),
            "effect_metrics": (
                "crystal_yield",
                "crystal_csd_quality",
                "crystal_fines_fraction",
            ),
        },
    }


def observation_queries(screen_id: str) -> list[dict[str, Any]]:
    spec = screen_specs()[screen_id]
    rows = []
    for level_index, level in enumerate(spec["levels"]):
        for replicate in range(1, REPLICATES + 1):
            rows.append(
                {
                    "query_id": f"level-{level_index}-replicate-{replicate}",
                    "phase": "observation_replicate",
                    "level_index": level_index,
                    "replicate": replicate,
                    "axis_a_index": level_index,
                    "axis_b_index": 0,
                    "feature_values": {
                        **spec["fixed_context"],
                        spec["level_field"]: level,
                    },
                    "metric_ids": list(spec["metrics"]),
                }
            )
    return rows


def truth_queries(screen_id: str) -> list[dict[str, Any]]:
    return [
        {
            **query,
            "query_id": f"truth-level-{query['level_index']}",
            "phase": "evaluator_truth",
            "replicate": None,
        }
        for query in observation_queries(screen_id)[::REPLICATES]
    ]


def analyze_observation_world(
    screen_id: str,
    noisy_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    spec = screen_specs()[screen_id]
    checks: dict[str, bool] = {
        "fixed_noisy_denominator": len(noisy_rows) == 9,
        "fixed_truth_denominator": len(truth_rows) == 3,
        "all_noisy_completed": all(row.get("status") == "completed" for row in noisy_rows),
        "all_truth_completed": all(row.get("status") == "completed" for row in truth_rows),
        "all_noisy_exact_replay": all(row.get("exact_replay") is True for row in noisy_rows),
        "all_truth_exact_replay": all(row.get("exact_replay") is True for row in truth_rows),
        "zero_platform_failures": not any(
            row.get("status") == "platform_failure" for row in (*noisy_rows, *truth_rows)
        ),
        "zero_physical_failures": not any(
            row.get("status") == "physical_failure" for row in (*noisy_rows, *truth_rows)
        ),
    }
    if not all(checks.values()):
        return _early_result(screen_id, noisy_rows, truth_rows, checks)

    metric_reports: dict[str, dict[str, Any]] = {}
    for metric in spec["metrics"]:
        by_level: list[dict[str, Any]] = []
        for level_index in range(3):
            observations = [
                float(row["metrics"][metric])
                for row in noisy_rows
                if int(row["level_index"]) == level_index
            ]
            truth = next(
                float(row["metrics"][metric])
                for row in truth_rows
                if int(row["level_index"]) == level_index
            )
            mean = float(np.mean(observations))
            sigma = float(np.std(observations, ddof=1))
            by_level.append(
                {
                    "level_index": level_index,
                    "observations": observations,
                    "mean": mean,
                    "sigma": sigma,
                    "truth": truth,
                    "bias": mean - truth,
                }
            )
        max_sigma = max(float(row["sigma"]) for row in by_level)
        gate = max(EFFECT_FLOOR, SIGMA_MULTIPLIER * max_sigma)
        max_bias = max(abs(float(row["bias"])) for row in by_level)
        effect = max(float(row["mean"]) for row in by_level) - min(
            float(row["mean"]) for row in by_level
        )
        metric_reports[metric] = {
            "levels": by_level,
            "max_replicate_sigma": max_sigma,
            "gate": gate,
            "max_absolute_bias": max_bias,
            "bias_passed": max_bias <= gate,
            "three_level_effect": effect,
            "effect_passed": effect >= gate,
        }
    finite = all(
        all(
            math.isfinite(float(row["metrics"][metric]))
            for metric in spec["metrics"]
        )
        for row in (*noisy_rows, *truth_rows)
    )
    observed = all(
        row.get("all_registered_metrics_observed", True) is True for row in noisy_rows
    )
    public_payload = json.dumps(list(noisy_rows), sort_keys=True)
    leakage_matches = sorted(
        token for token in FORBIDDEN_PUBLIC_TOKENS if token in public_payload
    )
    effect_candidates = {
        metric: metric_reports[metric] for metric in spec["effect_metrics"]
    }
    best_effect_metric = max(
        effect_candidates,
        key=lambda metric: (
            float(effect_candidates[metric]["three_level_effect"])
            - float(effect_candidates[metric]["gate"])
        ),
    )
    checks.update(
        {
            "all_registered_metrics_finite": finite,
            "all_registered_metrics_observed": observed,
            "all_metric_biases_within_gate": all(
                report["bias_passed"] for report in metric_reports.values()
            ),
            "task_owned_effect_above_noise": any(
                report["effect_passed"] for report in effect_candidates.values()
            ),
            "public_payload_leakage_free": not leakage_matches,
        }
    )
    return {
        "screen_id": screen_id,
        "passed": all(checks.values()),
        "checks": checks,
        "failures": sorted(key for key, passed in checks.items() if not passed),
        "denominators": _denominators(noisy_rows, truth_rows),
        "metric_reports": metric_reports,
        "best_effect_metric": best_effect_metric,
        "leakage_matches": leakage_matches,
    }


def _denominators(
    noisy_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    combined = [*noisy_rows, *truth_rows]
    return {
        "noisy_attempted": len(noisy_rows),
        "truth_attempted": len(truth_rows),
        "completed": sum(row.get("status") == "completed" for row in combined),
        "physical_failures": sum(row.get("status") == "physical_failure" for row in combined),
        "platform_failures": sum(row.get("status") == "platform_failure" for row in combined),
        "unsafe_completed": sum(
            row.get("status") == "completed" and row.get("safe") is False for row in combined
        ),
        "exact_replay": sum(row.get("exact_replay") is True for row in combined),
    }


def _early_result(
    screen_id: str,
    noisy_rows: Sequence[Mapping[str, Any]],
    truth_rows: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
) -> dict[str, Any]:
    return {
        "screen_id": screen_id,
        "passed": False,
        "checks": dict(checks),
        "failures": sorted(key for key, passed in checks.items() if not passed),
        "denominators": _denominators(noisy_rows, truth_rows),
        "metric_reports": None,
        "best_effect_metric": None,
        "leakage_matches": [],
    }


__all__ = [
    "OBSERVATION_SCREEN_VERSION",
    "analyze_observation_world",
    "observation_queries",
    "screen_specs",
    "truth_queries",
]
