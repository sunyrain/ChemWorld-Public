"""Aggregate local evaluation result files."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

from chemworld.eval.result_artifacts import validate_verified_evaluation_result


def _finite_score(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if value is None or isinstance(value, bool):
        raise ValueError(f"Leaderboard metric {key!r} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Leaderboard metric {key!r} must be finite")
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"Leaderboard metric {key!r} must be in [0, 1]")
    return number


def load_results(
    paths: Sequence[str | Path],
    *,
    replay_verify: bool = True,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            results.extend(payload)
        else:
            results.append(payload)
    for result in results:
        validate_verified_evaluation_result(result, replay=replay_verify)
    return results


def aggregate_leaderboard(
    results: list[dict[str, Any]],
    *,
    replay_verify: bool = True,
) -> list[dict[str, Any]]:
    for result in results:
        validate_verified_evaluation_result(result, replay=replay_verify)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(str(result["agent_name"]), str(result["world_split"]))].append(result)

    rows: list[dict[str, Any]] = []
    split_means: dict[tuple[str, str], float] = {}
    for (agent_name, split), items in grouped.items():
        scores = [_finite_score(item, "total_score") for item in items]
        performance = [_finite_score(item, "final_best_score") for item in items]
        safety = [_finite_score(item, "safety_aware_score") for item in items]
        mean_total = fmean(scores)
        std_total = pstdev(scores) if len(scores) > 1 else 0.0
        sem_total = std_total / (len(scores) ** 0.5) if len(scores) > 1 else 0.0
        ci95 = 1.96 * sem_total
        split_means[(agent_name, split)] = mean_total
        rows.append(
            {
                "agent_name": agent_name,
                "world_split": split,
                "runs": len(items),
                "mean_total_score": mean_total,
                "std_total_score": std_total,
                "sem_total_score": sem_total,
                "ci95_total_score_low": max(0.0, mean_total - ci95),
                "ci95_total_score_high": min(1.0, mean_total + ci95),
                "mean_performance": fmean(performance),
                "mean_safety_aware_score": fmean(safety),
            }
        )
    for row in rows:
        agent_name = row["agent_name"]
        private_score = split_means.get((agent_name, "private-eval"))
        public_score = split_means.get((agent_name, "public-test"))
        row["public_private_gap"] = (
            None if private_score is None or public_score is None else public_score - private_score
        )
    rows.sort(key=lambda row: row["mean_total_score"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows
