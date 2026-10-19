"""Public runtime subset of the material-family tests; study protocols stay private."""

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
