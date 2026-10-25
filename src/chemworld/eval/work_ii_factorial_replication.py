"""Outcome-blind replication roster and cluster-level factorial analysis."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np

from chemworld.eval.work_ii_factorial import CONDITIONS, MODELS, TASKS, development_protocol

CONTRASTS = {
    "F-X_minus_L-X": {"F-X": 1, "L-X": -1},
    "L-X_minus_L-A": {"L-X": 1, "L-A": -1},
    "F-A_minus_L-A": {"F-A": 1, "L-A": -1},
    "F-X_minus_F-A": {"F-X": 1, "F-A": -1},
    "interaction": {"F-X": 1, "L-X": -1, "F-A": -1, "L-A": 1},
}


def replication_protocol() -> dict[str, Any]:
    protocol = development_protocol()
    protocol.update(
        {
            "version": "work-ii-m1-replication-1",
            "formal_result": True,
            "mode": "release-freeze",
            "noise_namespace": "work-ii-m1-replication-20260905",
            "provider_call_opportunities": 120,
            "physical_execution_count": 200,
            "condition_slots": 160,
            "model_repeats": 2,
            "intervention": None,
            "primary_contrast": "F-X_minus_L-X",
            "minimum_practical_improvement": 0.01,
            "bootstrap_replicates": 20000,
            "bootstrap_seed": 20260906,
            "primary_interval_level": 0.95,
            "secondary_interval_level": 0.9875,
            "source_schedule_seed": 20260905,
        }
    )
    protocol.pop("world_seed")
    worlds = []
    for task in TASKS:
        for index in range(5):
            label = f"chemworld-m1-replication-v1/{task}/{index + 1}"
            seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big") % (2**31)
            worlds.append(
                {
                    "cluster_id": f"{task}--w{index + 1:02d}",
                    "task": task,
                    "world_index": index,
                    "world_seed": seed,
                }
            )
    protocol["worlds"] = worlds
    protocol["providers"] = {
        "deepseek": {
            "id": "deepseek",
            "name": "DeepSeek",
            "model": "deepseek-v4-flash",
            "reasoning_effort": "high",
            "base_url": "https://api.deepseek.com/",
            "wire_api": "responses",
            "auth_mode": "experimental_bearer_token",
            "api_key_file": "api.md",
            "model_catalog_json": "configs/providers/deepseek_v4_flash_models.json",
        },
        "gpt": {
            "id": "chemworld_openai_https",
            "name": "OpenAI",
            "model": "gpt-5.6-sol",
            "reasoning_effort": "medium",
            "auth_mode": "none",
            "wire_api": "responses",
        },
    }
    return protocol


def source_schedule(protocol: dict) -> list[dict]:
    states = []
    for world in protocol["worlds"]:
        for model_index, model in enumerate(MODELS):
            for repeat in range(protocol["model_repeats"]):
                reverse = (
                    world["world_index"] + TASKS.index(world["task"]) + model_index + repeat
                ) % 2
                states.append(
                    {
                        **world,
                        "model": model,
                        "repeat": repeat + 1,
                        "state_id": f"{world['cluster_id']}--{model}--r{repeat + 1}",
                        "decision_order": ["F", "L"] if reverse else ["L", "F"],
                    }
                )
    rng = np.random.default_rng(protocol["source_schedule_seed"])
    return [states[index] for index in rng.permutation(len(states))]


def bootstrap_interval(
    values_by_task: list[list[float]], *, count: int, seed: int, level: float
) -> list[float]:
    """Resample worlds within task, preserving all nested arms/models/repeats."""
    rng = np.random.default_rng(seed)
    means = []
    for values in values_by_task:
        array = np.asarray(values, dtype=float)
        indices = rng.integers(0, len(array), size=(count, len(array)))
        means.append(array[indices].mean(axis=1))
    samples = np.mean(means, axis=0)
    tail = (1 - level) / 2
    return np.quantile(samples, [tail, 1 - tail]).tolist()


def summarize_factorial(
    rows: list[dict], protocol: dict, *, conditions=CONDITIONS, contrasts=CONTRASTS
) -> dict:
    expected = {
        (state["state_id"], condition)
        for state in source_schedule(protocol)
        for condition in conditions
    }
    actual = [(row["state_id"], row["condition"]) for row in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("factorial denominator is incomplete or duplicated")
    roster = {state["state_id"]: state for state in source_schedule(protocol)}
    states: dict[str, dict] = defaultdict(dict)
    for row in rows:
        if any(
            row[key] != roster[row["state_id"]][key]
            for key in ("cluster_id", "task", "model", "repeat")
        ):
            raise ValueError("factorial cell metadata differs from the scheduled unit")
        if (
            not np.isfinite(row["failure_aware_regret"])
            or not 0 <= row["failure_aware_regret"] <= 1
        ):
            raise ValueError("regret outside fixed utility scale")
        states[row["state_id"]][row["condition"]] = row
    world_rows, condition_rows, contrast_rows = [], [], []
    for world in protocol["worlds"]:
        nested = [
            state
            for state in states.values()
            if state[conditions[0]]["cluster_id"] == world["cluster_id"]
        ]
        for contrast, weights in contrasts.items():
            values = [
                sum(state[key]["failure_aware_regret"] * weight for key, weight in weights.items())
                for state in nested
            ]
            completed_values = [
                sum(state[key]["raw_regret"] * weight for key, weight in weights.items())
                for state in nested
                if all(state[key]["status"] == "completed" for key in weights)
            ]
            world_rows.append(
                {
                    "cluster_id": world["cluster_id"],
                    "task": world["task"],
                    "contrast": contrast,
                    "nested_state_count": len(values),
                    "mean_difference": float(np.mean(values)),
                    "completed_pair_count": len(completed_values),
                    "completed_only_difference": float(np.mean(completed_values))
                    if completed_values
                    else None,
                }
            )
    for model in MODELS:
        for condition in conditions:
            selected = [
                row for row in rows if row["model"] == model and row["condition"] == condition
            ]
            completed = [row for row in selected if row["status"] == "completed"]
            condition_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    "scheduled": len(selected),
                    "completed": len(completed),
                    "mean_failure_aware_regret": float(
                        np.mean([row["failure_aware_regret"] for row in selected])
                    ),
                    "completed_only_regret": float(
                        np.mean([row["raw_regret"] for row in completed])
                    )
                    if completed
                    else None,
                    "near_optimal_count": sum(row["near_optimal"] for row in selected),
                    "top1_count": sum(row["top1"] for row in selected),
                }
            )
    for contrast in contrasts:
        selected = [row for row in world_rows if row["contrast"] == contrast]
        values = [
            [row["mean_difference"] for row in selected if row["task"] == task] for task in TASKS
        ]
        primary = contrast == protocol["primary_contrast"]
        level = (
            protocol["primary_interval_level"] if primary else protocol["secondary_interval_level"]
        )
        interval = bootstrap_interval(
            values,
            count=protocol["bootstrap_replicates"],
            seed=protocol["bootstrap_seed"],
            level=level,
        )
        contrast_rows.append(
            {
                "contrast": contrast,
                "primary": primary,
                "world_clusters": len(selected),
                "mean_difference": float(np.mean([np.mean(value) for value in values])),
                "interval_level": level,
                "interval": interval,
                "task_means": {
                    task: float(np.mean(value)) for task, value in zip(TASKS, values, strict=True)
                },
            }
        )
    primary = next(row for row in contrast_rows if row["primary"])
    return {
        "condition_summaries": condition_rows,
        "contrasts": contrast_rows,
        "world_contrasts": world_rows,
        "primary_material_benefit_supported": primary["interval"][1]
        < -protocol["minimum_practical_improvement"],
        "inference_limit": "Ten sampled task-world clusters, five per task. Percentile bootstrap "
        "intervals are approximate with this small sample. Models and repeated sessions are "
        "nested observations, not additional independent worlds. "
        f"One primary {protocol['primary_interval_level'] * 100:g}% interval; "
        f"{len(contrasts) - 1} secondary intervals use "
        f"{protocol['secondary_interval_level'] * 100:g}% marginal coverage "
        "(Bonferroni family adjustment).",
    }
