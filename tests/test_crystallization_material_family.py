from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chemworld.agents.crystallization_single_stage import (
    crystallization_single_stage_parameters_from_unit_vector,
)
from chemworld.agents.static_optimization import StaticOptimizationPlan
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.eval.static_optimization_execution import (
    StaticOptimizationExperimentSession,
)
from chemworld.eval.static_optimization_protocol import (
    static_optimization_crystallization_material_family_id,
    static_optimization_scoring_contract_id,
    validate_static_optimization_protocol,
)
from chemworld.world.crystallization_material_family import (
    REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
)
from chemworld.world.scoring import CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "configs/benchmark/"
    "scientific_optimization_s0_v0.8_crystallization_material_opaque_20x5_dev.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def _plan(*, solvent: int) -> StaticOptimizationPlan:
    vector = np.full(10, 0.5)
    vector[6] = (solvent + 0.1) / 4.0
    parameters = crystallization_single_stage_parameters_from_unit_vector(vector)
    parameters["solvent"] = solvent
    return StaticOptimizationPlan(
        experiment_intent="audit the versioned catalyst-solvent material family",
        search_vector=tuple(float(value) for value in vector),
        requested_measurement_slots=(
            "diagnostic-01-hplc",
            "diagnostic-02-hplc",
        ),
        measurement_objective="measure reaction and crystallization outcomes",
        expected_effect="produce one complete auditable product score",
        uncertainty=0.5,
        recipe_parameters=parameters,
    )


def test_replacement_crystallization_protocol_binds_explicit_contracts() -> None:
    protocol = _protocol()

    validate_static_optimization_protocol(protocol)

    assert static_optimization_crystallization_material_family_id(protocol) == (
        REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
    )
    assert static_optimization_scoring_contract_id(protocol) == (
        CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1
    )
    assert protocol["formal_result"] is False


def test_material_instance_is_fixed_and_private_for_one_world() -> None:
    first = ChemWorldEnv(
        task_id="reaction-to-crystallization",
        seed=3,
        crystallization_material_family_id=(
            REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
        ),
        scoring_contract_id=CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
    )
    second = ChemWorldEnv(
        task_id="reaction-to-crystallization",
        seed=3,
        crystallization_material_family_id=(
            REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
        ),
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
            crystallization_material_family_id=(
                REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
            ),
        )


def test_material_choice_changes_complete_crystallization_outcome() -> None:
    outcomes: list[tuple[dict[str, object], dict[str, object]]] = []
    for solvent in (0, 2):
        with StaticOptimizationExperimentSession(
            task_id="reaction-to-crystallization",
            seed=0,
            experiment_horizon=1,
            observation_seed=101,
            crystallization_material_family_id=(
                REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
            ),
            scoring_contract_id=CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
        ) as session:
            result = session.execute(_plan(solvent=solvent))
            outcomes.append(
                (
                    result.terminal_summary,
                    result.measurement_evidence[-1]["processed_estimate"],
                )
            )

    assert outcomes[0][0]["leaderboard_score"] != outcomes[1][0]["leaderboard_score"]
    assert outcomes[0][1]["crystal_yield"] != outcomes[1][1]["crystal_yield"]
