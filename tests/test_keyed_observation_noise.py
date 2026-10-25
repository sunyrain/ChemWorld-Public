from __future__ import annotations

import numpy as np

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.envs.observation_noise import (
    ObservationNoiseCoordinate,
    keyed_observation_rng,
)


def test_keyed_noise_is_coordinate_deterministic_and_isolated() -> None:
    coordinate = ObservationNoiseCoordinate(
        namespace="paired-test",
        base_observation_seed=99,
        experiment_index=2,
        operation_type="measure",
        instrument="hplc",
        replicate_index=0,
    )
    first = keyed_observation_rng(coordinate).normal(size=8)
    second = keyed_observation_rng(coordinate).normal(size=8)
    assert np.array_equal(first, second)

    changed = ObservationNoiseCoordinate(
        **{
            **coordinate.to_private_dict(),
            "replicate_index": 1,
        }
    )
    assert not np.array_equal(first, keyed_observation_rng(changed).normal(size=8))


def test_invalid_operation_does_not_advance_keyed_semantic_coordinate() -> None:
    kwargs = {
        "task_id": "flow-reaction-optimization",
        "seed": 1200,
        "budget_override": 4,
        "episode_mode_override": "campaign",
        "observation_seed_override": 501,
        "observation_noise_mode": "keyed",
        "observation_noise_namespace": "paired-test",
    }
    direct = ChemWorldEnv(**kwargs)
    interrupted = ChemWorldEnv(**kwargs)
    direct.reset(seed=1200)
    interrupted.reset(seed=1200)
    action = {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}

    direct.step(action)
    interrupted.step({"operation": "unknown"})
    interrupted.step(action)

    direct_noise = direct.observation_noise_provenance()
    interrupted_noise = interrupted.observation_noise_provenance()
    assert direct_noise["noise_key_sha256"] == interrupted_noise["noise_key_sha256"]
    assert direct_noise["coordinate"] == interrupted_noise["coordinate"]
    assert direct_noise["sequential_rng_position_used"] is False
    direct.close()
    interrupted.close()

