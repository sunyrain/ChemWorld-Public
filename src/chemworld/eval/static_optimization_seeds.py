"""Deterministic observation-seed namespaces for static optimization."""

from __future__ import annotations

import hashlib


def exploration_observation_seed(task_id: str, world_seed: int) -> int:
    return _seed(f"static-optimization-s0|{task_id}|{world_seed}")


def validation_observation_seed(
    task_id: str,
    world_seed: int,
    target: str,
    replicate_index: int,
) -> int:
    return _seed(
        "static-optimization-s0-validation|"
        f"{task_id}|{world_seed}|{target}|{replicate_index}"
    )


def predictive_observation_seed(
    task_id: str,
    world_seed: int,
    query_id: str,
    replicate_index: int,
) -> int:
    return _seed(
        "static-optimization-s0-predictive|"
        f"{task_id}|{world_seed}|{query_id}|{replicate_index}"
    )


def _seed(namespace: str) -> int:
    digest = hashlib.sha256(namespace.encode()).digest()
    return int.from_bytes(digest[:4], "big")


__all__ = [
    "exploration_observation_seed",
    "predictive_observation_seed",
    "validation_observation_seed",
]
