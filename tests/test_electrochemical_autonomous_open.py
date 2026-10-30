from __future__ import annotations

import json
import re
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from chemworld.eval import runner
from chemworld.legacy.agents.random import RandomAgent
from chemworld.materials import (
    STATIC_MATERIAL_INFORMATION_NOMINAL,
    STATIC_MATERIAL_INFORMATION_OPAQUE,
)
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE,
    ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1,
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
)
from chemworld.tasks import get_task
from chemworld.world.actions import ELECTROLYTE_PROFILES, SOLVENTS
from chemworld.world.crystallization_material_family import (
    REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY,
)
from chemworld.world.electrochemical_material_family import (
    NOMINAL_PRIOR_MATERIAL_FAMILY,
)
from chemworld.world.scoring import ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2


def _setup_cell(env: Any) -> None:
    actions = (
        {"operation": "add_solvent", "volume_L": 0.026, "solvent": 2},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {
            "operation": "set_potential",
            "potential_V": 1.15,
            "current_mA": 75.0,
            "electrolyte_profile": 2,
        },
    )
    for action in actions:
        _, _, _, _, info = env.step(action)
        assert info["transaction_status"] == "committed"


def _valid_operations(env: Any) -> set[str]:
    return {
        item["operation"]
        for item in env.unwrapped.available_actions()
    }


def test_autonomous_open_workflow_exposes_physical_choice_without_forced_stages() -> None:
    env = gym.make(
        "ChemWorld",
        task_id="electrochemical-conversion",
        seed=0,
        electrochemical_workflow_mode=(
            ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
        ),
    )
    try:
        env.reset(seed=0)
        env.step(
            {"operation": "add_solvent", "volume_L": 0.026, "solvent": 2}
        )
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        assert _valid_operations(env) == {
            "add_solvent",
            "add_reagent",
            "set_potential",
            "measure",
        }
        _, _, _, _, baseline = env.step(
            {"operation": "measure", "instrument": "uvvis"}
        )
        assert baseline["transaction_status"] == "committed"
        _, _, _, _, configured = env.step(
            {
                "operation": "set_potential",
                "potential_V": 1.15,
                "current_mA": 75.0,
                "electrolyte_profile": 2,
            }
        )
        assert configured["transaction_status"] == "committed"

        assert _valid_operations(env) == {
            "add_solvent",
            "add_reagent",
            "set_potential",
            "electrolyze",
            "measure",
        }
        _, _, _, _, semibatch_solvent = env.step(
            {"operation": "add_solvent", "volume_L": 0.001, "solvent": 2}
        )
        assert semibatch_solvent["transaction_status"] == "committed"
        _, _, _, _, semibatch_reagent = env.step(
            {"operation": "add_reagent", "amount_mol": 0.001}
        )
        assert semibatch_reagent["transaction_status"] == "committed"
        assert env.unwrapped.validate_action({"operation": "terminate"})[
            "valid"
        ] is False
        repeated = {
            "operation": "set_potential",
            "potential_V": 1.15,
            "current_mA": 75.0,
            "electrolyte_profile": 2,
        }
        assert env.unwrapped.validate_action(repeated)["valid"] is True
        setpoint_constraints = env.unwrapped.action_schema("set_potential")[
            "constraints"
        ]
        assert not any(
            item["id"] == "payload_adapts:electrochemical_setpoint"
            for item in setpoint_constraints
        )

        _, _, _, _, diagnostic = env.step(
            {"operation": "measure", "instrument": "uvvis"}
        )
        assert diagnostic["transaction_status"] == "committed"
        _, _, _, _, repeated_info = env.step(repeated)
        assert repeated_info["transaction_status"] == "committed"
        _, _, _, _, electrolysis = env.step(
            {"operation": "electrolyze", "duration_s": 120.0}
        )
        assert electrolysis["transaction_status"] == "committed"

        assert _valid_operations(env) == {
            "add_solvent",
            "add_reagent",
            "set_potential",
            "electrolyze",
            "measure",
            "terminate",
        }
        assert env.unwrapped.validate_action({"operation": "terminate"})[
            "valid"
        ] is True
        instrument_field = env.unwrapped.action_schema("measure")["fields"][0]
        assert set(instrument_field["choices"]) == {"ph_meter", "uvvis"}
    finally:
        env.close()


@pytest.mark.parametrize(
    ("workflow_mode", "operation_after_diagnostics"),
    (
        (ELECTROCHEMICAL_WORKFLOW_ADAPTIVE_TWO_STAGE, "set_potential"),
        (ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE, "terminate"),
    ),
)
def test_legacy_electrochemical_workflow_modes_keep_their_gates(
    workflow_mode: str,
    operation_after_diagnostics: str,
) -> None:
    env = gym.make(
        "ChemWorld",
        task_id="electrochemical-conversion",
        seed=0,
        electrochemical_workflow_mode=workflow_mode,
    )
    try:
        env.reset(seed=0)
        _setup_cell(env)
        assert _valid_operations(env) == {"electrolyze"}
        env.step({"operation": "electrolyze", "duration_s": 120.0})
        assert _valid_operations(env) == {"measure"}
        assert set(
            env.unwrapped.action_schema("measure")["fields"][0]["choices"]
        ) == {"ph_meter", "uvvis"}
        env.step({"operation": "measure", "instrument": "ph_meter"})
        assert env.unwrapped.action_schema("measure")["fields"][0][
            "choices"
        ] == ["uvvis"]
        env.step({"operation": "measure", "instrument": "uvvis"})
        assert _valid_operations(env) == {operation_after_diagnostics}
    finally:
        env.close()


@pytest.mark.parametrize(
    "mode",
    (STATIC_MATERIAL_INFORMATION_OPAQUE, STATIC_MATERIAL_INFORMATION_NOMINAL),
)
def test_electrochemical_material_interfaces_use_only_anonymous_ids(
    mode: str,
) -> None:
    env = gym.make(
        "ChemWorld",
        task_id="electrochemical-conversion",
        seed=1,
        electrochemical_material_family_id=NOMINAL_PRIOR_MATERIAL_FAMILY,
        material_information={"mode": mode},
    )
    try:
        env.reset(seed=1)
        task_info = env.unwrapped.task_info()
        catalog = task_info["material_catalog"]
        assert [
            item["display_name"] for item in catalog["solvents"]
        ] == [f"solvent-S{index}" for index in range(4)]
        assert [
            item["display_name"] for item in catalog["electrolyte_profiles"]
        ] == [f"electrolyte-E{index}" for index in range(4)]
        solvent_schema = env.unwrapped.action_schema("add_solvent")
        solvent_field = next(
            item
            for item in solvent_schema["fields"]
            if item["field"] == "solvent"
        )
        assert solvent_field["choice_labels"] == {
            str(index): f"solvent-S{index}" for index in range(4)
        }
        electrolyte_schema = env.unwrapped.action_schema("set_potential")
        electrolyte_field = next(
            item
            for item in electrolyte_schema["fields"]
            if item["field"] == "electrolyte_profile"
        )
        assert electrolyte_field["choice_labels"] == {
            str(index): f"electrolyte-E{index}" for index in range(4)
        }

        serialized = json.dumps(task_info, sort_keys=True).lower()
        for hidden_name in (*SOLVENTS, *ELECTROLYTE_PROFILES):
            # Physical field names such as water_flux are not material labels.
            # A free-standing name in any public text still reveals identity.
            assert re.search(rf"\b{re.escape(hidden_name.lower())}\b", serialized) is None
        public_information = task_info["material_information"]
        assert public_information["mode"] == mode
        if mode == STATIC_MATERIAL_INFORMATION_OPAQUE:
            assert set(public_information) == {"mode"}
        else:
            assert public_information["dossier_sha256"]
            dossier = public_information["dossier"]
            assert [
                item["anonymous_material_id"]
                for item in dossier["choices"]["solvent"]
            ] == [f"solvent-S{index}" for index in range(4)]
            assert [
                item["anonymous_material_id"]
                for item in dossier["choices"]["electrolyte_profile"]
            ] == [f"electrolyte-E{index}" for index in range(4)]
            assert "world_id" not in json.dumps(dossier, sort_keys=True)
    finally:
        env.close()


def test_material_information_arms_share_world_physics_and_keyed_noise() -> None:
    common = {
        "task_id": "electrochemical-conversion",
        "seed": 3,
        "electrochemical_workflow_mode": (
            ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
        ),
        "electrochemical_material_family_id": NOMINAL_PRIOR_MATERIAL_FAMILY,
        "observation_seed_override": 2718,
        "observation_noise_mode": "keyed",
        "observation_noise_namespace": "electrochemical-material-pair",
    }
    opaque = gym.make(
        "ChemWorld",
        **common,
        material_information={"mode": STATIC_MATERIAL_INFORMATION_OPAQUE},
    )
    nominal = gym.make(
        "ChemWorld",
        **common,
        material_information={"mode": STATIC_MATERIAL_INFORMATION_NOMINAL},
    )
    actions = (
        {"operation": "add_solvent", "volume_L": 0.026, "solvent": 1},
        {"operation": "add_reagent", "amount_mol": 0.010},
        {
            "operation": "set_potential",
            "potential_V": 1.05,
            "current_mA": 80.0,
            "electrolyte_profile": 1,
        },
        {"operation": "electrolyze", "duration_s": 180.0},
        {"operation": "measure", "instrument": "uvvis"},
    )
    try:
        opaque.reset(seed=3)
        nominal.reset(seed=3)
        opaque_provenance = opaque.unwrapped.evaluator_provenance()
        nominal_provenance = nominal.unwrapped.evaluator_provenance()
        for key in (
            "world_id",
            "mechanism_hash",
            "electrochemical_material_family_sha256",
            "electrochemical_material_instance_sha256",
            "observation_noise_mode",
            "observation_noise_namespace",
        ):
            assert opaque_provenance[key] == nominal_provenance[key]

        for action in actions:
            left_observation, left_reward, *_ = opaque.step(action)
            right_observation, right_reward, *_ = nominal.step(action)
            assert left_reward == pytest.approx(right_reward)
            assert left_observation.keys() == right_observation.keys()
            for key in left_observation:
                np.testing.assert_array_equal(
                    left_observation[key],
                    right_observation[key],
                )
        assert (
            opaque.unwrapped.observation_noise_provenance()
            == nominal.unwrapped.observation_noise_provenance()
        )
    finally:
        opaque.close()
        nominal.close()


@pytest.mark.parametrize("native_session", (True, False))
def test_runner_forwards_explicit_autonomous_material_and_pairing_contract(
    monkeypatch: pytest.MonkeyPatch,
    native_session: bool,
) -> None:
    captured: dict[str, Any] = {}
    real_make = runner.ChemWorldRuntime if native_session else runner.gym.make

    def recording_make(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_make(*args, **kwargs)

    if native_session:
        monkeypatch.setattr(runner, "ChemWorldRuntime", recording_make)
    else:
        monkeypatch.setattr(runner.gym, "make", recording_make)
    wrapper = None if native_session else lambda env: env
    task = get_task("electrochemical-conversion")
    runner.run_agent(
        env_id=task.env_id,
        agent=RandomAgent(),
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=0,
        task_id=task.task_id,
        budget_override=1,
        material_information={"mode": STATIC_MATERIAL_INFORMATION_NOMINAL},
        electrochemical_material_family_id=NOMINAL_PRIOR_MATERIAL_FAMILY,
        crystallization_material_family_id=None,
        electrochemical_workflow_mode=(
            ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
        ),
        scoring_contract_id=ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2,
        observation_noise_mode="keyed",
        observation_noise_namespace="runner-forwarding-test",
        env_wrapper=wrapper,
    )

    assert captured["material_information"] == {
        "mode": STATIC_MATERIAL_INFORMATION_NOMINAL
    }
    assert (
        captured["electrochemical_material_family_id"]
        == NOMINAL_PRIOR_MATERIAL_FAMILY
    )
    assert (
        captured["electrochemical_workflow_mode"]
        == ELECTROCHEMICAL_WORKFLOW_AUTONOMOUS_OPEN_V1
    )
    assert (
        captured["scoring_contract_id"]
        == ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2
    )
    assert captured["observation_noise_mode"] == "keyed"
    assert (
        captured["observation_noise_namespace"]
        == "runner-forwarding-test"
    )

    captured.clear()
    crystallization_task = get_task("reaction-to-crystallization")
    runner.run_agent(
        env_id=crystallization_task.env_id,
        agent=RandomAgent(),
        world_split=crystallization_task.world_split,
        budget=crystallization_task.budget,
        objective=crystallization_task.objective,
        seed=0,
        task_id=crystallization_task.task_id,
        budget_override=1,
        material_information={"mode": STATIC_MATERIAL_INFORMATION_NOMINAL},
        crystallization_material_family_id=(
            REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
        ),
        env_wrapper=wrapper,
    )
    assert (
        captured["crystallization_material_family_id"]
        == REACTION_CRYSTALLIZATION_LATENT_MATERIAL_FAMILY
    )
