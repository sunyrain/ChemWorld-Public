from __future__ import annotations

from uuid import UUID

import gymnasium as gym

import chemworld  # noqa: F401


def test_paired_world_identity_is_private_but_evaluator_provenance_is_retained() -> None:
    baseline = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    changed = gym.make(
        "ChemWorld",
        task_id="electrochemical-conversion",
        seed=0,
        world_interventions=(
            {
                "kind": "mechanism_family",
                "mode": "constitutive_law_family",
                "severity": 0.8,
            },
        ),
    )
    try:
        _, baseline_info = baseline.reset(seed=0)
        _, changed_info = changed.reset(seed=0)
        assert baseline_info == changed_info
        assert "seed" not in baseline_info
        assert baseline.unwrapped.evaluator_provenance()["world_seed"] == 0
        forbidden_public_keys = {
            "world_id",
            "world_provider",
            "mechanism_id",
            "mechanism_hash",
            "mechanism_version",
            "world_family_intervention_hash",
            "mechanism_family_intervention_hash",
            "material_law_counterfactual_hash",
        }

        def public_keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | {
                    nested for item in value.values() for nested in public_keys(item)
                }
            if isinstance(value, list):
                return {nested for item in value for nested in public_keys(item)}
            return set()

        assert public_keys(baseline_info).isdisjoint(forbidden_public_keys)
        baseline_provenance = baseline.unwrapped.evaluator_provenance()
        changed_provenance = changed.unwrapped.evaluator_provenance()
        assert baseline_provenance["world_id"] != changed_provenance["world_id"]
        assert changed_provenance["mechanism_family_intervention_hash"]

        _, _, _, _, step_info = baseline.step(
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0}
        )
        assert step_info["campaign_id"].startswith("episode-")
        UUID(step_info["campaign_id"].removeprefix("episode-"))
    finally:
        baseline.close()
        changed.close()
