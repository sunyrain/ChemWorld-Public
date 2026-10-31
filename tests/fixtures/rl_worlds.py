"""Small authored RL inputs, independent of historical study allocations."""

from chemworld.rl.environment import AllocationName, RLWorldAllocation


def example_allocation(
    task_id: str, name: AllocationName = "train",
) -> RLWorldAllocation:
    seed_start = {"train": 20, "dev": 30, "bench": 40}[name]
    return RLWorldAllocation.from_protocol(
        {"world_family_allocation": {name: {
            "base_seeds": {"start": seed_start, "stop_inclusive": seed_start + 2},
            "cells": {"interpolation": [-0.25, 0.25]},
            "namespace_id": "authored-rl-test-inputs",
        }}},
        task_id=task_id,
        name=name,
    )
