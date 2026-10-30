from __future__ import annotations

import pytest

from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.world.crystallization_material_family import (
    REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
)
from chemworld.world.scoring import CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1


def test_material_instance_is_fixed_and_private_for_one_world() -> None:
    first = ChemWorldEnv(
        task_id="reaction-to-crystallization",
        seed=3,
        crystallization_material_family_id=(REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY),
        scoring_contract_id=CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
    )
    second = ChemWorldEnv(
        task_id="reaction-to-crystallization",
        seed=3,
        crystallization_material_family_id=(REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY),
        scoring_contract_id=CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
    )
    try:
        first_hash = first.scenario_instance.initial_state.metadata[
            "crystallization_material_instance_sha256"
        ]
        second_hash = second.scenario_instance.initial_state.metadata[
            "crystallization_material_instance_sha256"
        ]
        assert first_hash == second_hash
        assert first.world.world_id == second.world.world_id
        assert "crystallization_material_family_id" not in first.reset(seed=3)[1]
    finally:
        first.close()
        second.close()


def test_nonlegacy_crystallization_family_rejects_other_tasks() -> None:
    with pytest.raises(ValueError, match="reaction-to-crystallization"):
        ChemWorldEnv(
            task_id="reaction-to-assay",
            seed=0,
            crystallization_material_family_id=(REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY),
        )


def test_material_choice_changes_complete_crystallization_outcome() -> None:
    outcomes: list[tuple[dict[str, object], dict[str, object]]] = []
    for solvent in (0, 2):
        env = ChemWorldEnv(
            task_id="reaction-to-crystallization",
            seed=0,
            budget_override=30,
            crystallization_material_family_id=REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
            scoring_contract_id=CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
        )
        try:
            env.reset(seed=0)
            for action in [
                {"operation": "add_solvent", "volume_L": 0.03, "solvent": solvent},
                {"operation": "add_reagent", "amount_mol": 0.01},
                {"operation": "add_catalyst", "catalyst": 1, "catalyst_amount_mol": 0.00025},
                {
                    "operation": "heat",
                    "target_temperature_K": 350.0,
                    "duration_s": 1200.0,
                    "stirring_speed_rpm": 600.0,
                },
                {"operation": "quench"},
                {
                    "operation": "cool_crystallize",
                    "target_temperature_K": 278.0,
                    "duration_s": 1800.0,
                },
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "filter_crystals"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]:
                obs, _, _, _, info = env.step(action)
                assert info["transaction_status"] == "committed"
            outcomes.append((info, obs))
        finally:
            env.close()

    assert outcomes[0][0]["leaderboard_score"] != outcomes[1][0]["leaderboard_score"]
    assert outcomes[0][1]["crystal_yield"] != outcomes[1][1]["crystal_yield"]
