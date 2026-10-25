from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from examples.demo_dataset_agent_trace_export import build_demo

import chemworld  # noqa: F401
from chemworld.agent_interface import rl_observation_spec
from chemworld.agents.interactive_codex_experiment import _material_information_payload
from chemworld.data.datasets import flatten_record
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.envs.spaces import OBSERVATION_KEYS
from chemworld.eval.runner import run_agent
from chemworld.foundation.public_leakage import audit_public_payload
from chemworld.legacy.agents.llm import LLMReplayAgent, ToolUsingLLMStubAgent
from chemworld.wrappers import (
    ActionSuggestionWrapper,
    AgentInfoWrapper,
    LLMObservationWrapper,
    RLObservationWrapper,
)


@pytest.mark.parametrize(
    "mode",
    [
        "opaque_codes",
        "anonymous_nominal_properties",
        "anonymous_misindexed_properties",
    ],
)
def test_partition_public_surfaces_hide_identity_and_deliver_selected_prior(mode) -> None:
    material = {"mode": mode}
    if mode == "anonymous_misindexed_properties":
        material.update(target_field="extractant", descriptor_permutation=[3, 1, 2, 0])
    env = gym.make(
        "ChemWorld", task_id="partition-discovery", seed=0, material_information=material
    )
    try:
        _, info = env.reset(seed=0)
        base = env.unwrapped
        task = base.task_info()
        public = _material_information_payload(task)
        schemas = [base.action_schema(op) for op in task["allowed_operations"]]
        surfaces = json.dumps([base.task_prompt(), public, schemas, info]).lower()
        for forbidden in ("ethanol", "acetonitrile", "toluene", "cas_number", '"formula"'):
            assert forbidden not in surfaces
        # Water as the public aqueous phase is legitimate; a material named Water is not.
        assert '"display_name": "water"' not in surfaces
        assert public["material_catalog"]["solvents"][0]["display_name"] == "solvent-S0"
        assert public["material_catalog"]["extractants"][0]["display_name"] == "extractant-X0"
        for operation, field, prefix in (
            ("add_solvent", "solvent", "solvent-S"),
            ("add_extractant", "extractant", "extractant-X"),
        ):
            fields = {f["field"]: f for f in base.action_schema(operation)["fields"]}
            assert fields[field]["choice_labels"] == {str(i): f"{prefix}{i}" for i in range(4)}
        dossier = public["material_information"]["dossier"]
        if mode == "opaque_codes":
            assert dossier is None
        else:
            from chemworld.materials import static_material_information_dossier

            assert dossier == static_material_information_dossier(material, task_id=task["task_id"])
            assert "product_distribution_coefficients" not in json.dumps(dossier)
        assert "misindexed_properties" not in json.dumps(public)
    finally:
        env.close()


def test_env_exposes_agent_facing_methods() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        env.reset(seed=0)
        prompt = env.unwrapped.task_prompt()
        assert prompt["task_id"] == "reaction-to-purification"
        assert "budget" in prompt["text"].lower()
        assert "final_assay" in prompt["text"]
        assert "hplc" in prompt["allowed_instruments"]
        assert "purity" in prompt["success_metrics"]
        assert prompt["material_catalog"]["solvents"][0]["display_name"] == "Water"
        assert (
            "categorical benchmark effects" in prompt["material_catalog"]["interpretation_policy"]
        )

        actions = env.unwrapped.available_actions()
        operations = {entry["operation"] for entry in actions}
        assert "add_solvent" in operations
        assert "heat" not in operations

        heat_schema = env.unwrapped.action_schema("heat")
        assert heat_schema["fields"]
        assert "target_temperature_K" in heat_schema["required_fields"]
        temperature_field = {field["field"]: field for field in heat_schema["fields"]}[
            "target_temperature_K"
        ]
        assert temperature_field["unit"] == "K"
        assert temperature_field["bounds"]["low"] == 250.0

        solvent_schema = env.unwrapped.action_schema("add_solvent")
        solvent_field = {field["field"]: field for field in solvent_schema["fields"]}["solvent"]
        assert solvent_field["choice_labels"]["0"].startswith("Water · H2O")

        before = env.unwrapped.campaign_state()
        validation = env.unwrapped.validate_action({"operation": "heat", "duration_s": 1.0})
        after = env.unwrapped.campaign_state()
        assert not validation["valid"]
        assert before == after
    finally:
        env.close()


def test_core_action_contract_matches_effective_runtime_semantics() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        schemas = {
            operation: env.unwrapped.action_schema(operation)
            for operation in (
                "add_phase",
                "add_extractant",
                "heat",
                "separate_phase",
                "seed_crystals",
                "cool_crystallize",
                "evaporate",
                "transfer",
            )
        }

        def field(operation: str, field_name: str) -> dict[str, object]:
            return next(
                item for item in schemas[operation]["fields"] if item["field"] == field_name
            )

        assert field("add_phase", "phase")["choices"] == ["aqueous", "organic"]
        assert field("add_phase", "volume_L")["bounds"] == {"low": 0.0, "high": 0.06}
        assert field("add_extractant", "extractant")["choices"] == [0, 1, 2, 3]
        # C materials are anonymous even when a generic extraction schema is queried.
        assert field("add_extractant", "extractant")["choice_labels"]["3"] == "solvent-S3"
        assert field("separate_phase", "target_phase")["choices"] == [
            "aqueous",
            "organic",
        ]
        assert field("seed_crystals", "seed_mass_g")["bounds"] == {
            "low": 1.0e-6,
            "high": 0.05,
        }
        assert field("heat", "duration_s")["bounds"] == {
            "low": 1.0,
            "high": 14_400.0,
        }
        assert field("transfer", "transfer_fraction")["bounds"] == {
            "low": 1.0e-4,
            "high": 1.0,
        }
        cooling_temperature = field("cool_crystallize", "target_temperature_K")
        assert cooling_temperature["bounds"] == {
            "low": 250.0,
            "high": pytest.approx(min(330.0, env.unwrapped._state.temperature_K)),
        }
        assert cooling_temperature["state_dependent_bounds"] is True
        assert field("evaporate", "target_temperature_K")["bounds"] == {
            "low": 298.15,
            "high": 390.0,
        }

        invalid_phase = env.unwrapped.validate_action(
            {"operation": "add_phase", "phase": "solid", "volume_L": 0.02}
        )
        invalid_extractant = env.unwrapped.validate_action(
            {"operation": "add_extractant", "extractant": 4, "volume_L": 0.02}
        )
        invalid_seed = env.unwrapped.validate_action(
            {"operation": "seed_crystals", "seed_mass_g": 0.5}
        )
        assert not invalid_phase["valid"]
        assert not invalid_phase["dispatchable_to_runtime"]
        assert not invalid_extractant["valid"]
        assert not invalid_extractant["dispatchable_to_runtime"]
        assert "outside [0, 3]" in invalid_extractant["invalid_reasons"][0]
        assert not invalid_seed["valid"]
        assert "payload_bounds:seed_mass_g" in invalid_seed["invalid_reasons"]
    finally:
        env.close()


def test_public_volume_bounds_match_capacity_and_terminal_assay_stays_reachable() -> None:
    capacity_env = gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0)
    try:
        capacity_env.reset(seed=0)
        capacity_env.step({"operation": "add_solvent", "volume_L": 0.08, "solvent": 0})
        schema = capacity_env.unwrapped.action_schema("add_solvent")
        volume = next(field for field in schema["fields"] if field["field"] == "volume_L")
        assert volume["state_dependent_bounds"] is True
        assert 0.0 < volume["bounds"]["high"] < 0.08
        accepted = capacity_env.unwrapped.validate_action(
            {
                "operation": "add_solvent",
                "volume_L": volume["bounds"]["high"],
                "solvent": 0,
            }
        )
        rejected = capacity_env.unwrapped.validate_action(
            {
                "operation": "add_solvent",
                "volume_L": volume["bounds"]["high"] + 1.0e-4,
                "solvent": 0,
            }
        )
        assert accepted["valid"] is True
        assert rejected["valid"] is False
        assert "payload_bounds:total_volume_L" in rejected["invalid_reasons"]
    finally:
        capacity_env.close()

    assay_env = gym.make("ChemWorld", task_id="reaction-to-distillation", seed=0)
    try:
        assay_env.reset(seed=0)
        assay_env.step({"operation": "add_solvent", "volume_L": 0.0005, "solvent": 0})
        assay_env.step({"operation": "add_reagent", "amount_mol": 0.01})
        assay_env.step({"operation": "sample", "sample_volume_L": 0.00025})

        blocked = assay_env.unwrapped.validate_action({"operation": "terminate"})
        operations = {item["operation"] for item in assay_env.unwrapped.available_actions()}
        assert blocked["valid"] is False
        assert "final_assay_sample_available" in blocked["invalid_reasons"]
        assert "terminate" not in operations
        assert "add_solvent" in operations

        assay_env.step({"operation": "add_solvent", "volume_L": 0.001, "solvent": 0})
        ready = assay_env.unwrapped.validate_action({"operation": "terminate"})
        assert ready["valid"] is True
        assay_env.step({"operation": "terminate"})
        _, _, _, _, info = assay_env.step({"operation": "measure", "instrument": "final_assay"})
        assert info["experiment_ended"] is True
        assert info["constraint_flags"]["precondition_failed"] is False
    finally:
        assay_env.close()


def test_public_reagent_bounds_prevent_guaranteed_pressure_rollback() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        env.reset(seed=0)
        base = env.unwrapped
        env.step({"operation": "add_solvent", "volume_L": 0.01, "solvent": 0})
        validator = base.operation_validator
        original_safe_amount = validator._maximum_safe_reagent_amount_mol(base._state)
        assert original_safe_amount > 0.01

        desired_safe_amount = 0.004
        pressure_relevant_species = next(
            species_id
            for species_id in base._state.species_amounts
            if not species_id.startswith("Cat")
        )
        species_amounts = dict(base._state.species_amounts)
        species_amounts[pressure_relevant_species] += (
            original_safe_amount - desired_safe_amount
        ) * validator.reagent_charge_molar_multiplier
        base._state = base._state.replace(species_amounts=species_amounts)

        schema = base.action_schema("add_reagent")
        amount = next(field for field in schema["fields"] if field["field"] == "amount_mol")
        safe_high = amount["bounds"]["high"]
        assert amount["state_dependent_bounds"] is True
        assert safe_high == pytest.approx(desired_safe_amount)
        assert base.validate_action({"operation": "add_reagent", "amount_mol": safe_high})["valid"]
        rejected = base.validate_action(
            {"operation": "add_reagent", "amount_mol": safe_high + 1.0e-4}
        )
        assert rejected["valid"] is False
        assert "payload_bounds:amount_mol" in rejected["invalid_reasons"]

        _, _, _, _, info = env.step({"operation": "add_reagent", "amount_mol": safe_high})
        assert info["transaction_status"] == "committed"
        assert info["constraint_flags"]["constitution_failed"] is False
        vessel = base._state.vessels.vessels[base._state.vessel_id]
        assert base._state.pressure_Pa <= vessel.max_pressure_Pa + base.constitution.tolerance
    finally:
        env.close()


def test_public_liquid_bounds_prevent_material_first_pressure_rollback() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        _observation, _reward, _terminated, _truncated, reagent_info = env.step(
            {"operation": "add_reagent", "amount_mol": 0.040}
        )
        assert reagent_info["transaction_status"] == "committed"

        schema = env.unwrapped.action_schema("add_solvent")
        volume = next(field for field in schema["fields"] if field["field"] == "volume_L")
        safe_low = volume["bounds"]["low"]
        assert volume["state_dependent_bounds"] is True
        assert safe_low > 0.0
        rejected = env.unwrapped.validate_action(
            {"operation": "add_solvent", "volume_L": safe_low * 0.5, "solvent": 0}
        )
        assert rejected["valid"] is False
        assert "payload_bounds:volume_L" in rejected["invalid_reasons"]

        _observation, _reward, _terminated, _truncated, solvent_info = env.step(
            {
                "operation": "add_solvent",
                "volume_L": safe_low + 1.0e-6,
                "solvent": 0,
            }
        )
        assert solvent_info["transaction_status"] == "committed"
        assert solvent_info["constraint_flags"]["constitution_failed"] is False
    finally:
        env.close()


def test_electrolyte_profile_is_locked_and_solvent_blending_is_public() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        env.reset(seed=0)
        solvent_schema = env.unwrapped.action_schema("add_solvent")
        solvent = next(field for field in solvent_schema["fields"] if field["field"] == "solvent")
        assert solvent["choices"] == [0, 1, 2, 3]
        assert "state_dependent_choices" not in solvent

        env.step({"operation": "add_solvent", "volume_L": 0.026, "solvent": 2})
        locked_solvent_schema = env.unwrapped.action_schema("add_solvent")
        locked_solvent = next(
            field for field in locked_solvent_schema["fields"] if field["field"] == "solvent"
        )
        assert locked_solvent["choices"] == [0, 1, 2, 3]
        assert "locked_for_current_experiment" not in locked_solvent
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        profile_schema = env.unwrapped.action_schema("set_potential")
        profile = next(
            field for field in profile_schema["fields"] if field["field"] == "electrolyte_profile"
        )
        assert profile["choices"] == [0, 1, 2, 3]
        env.step(
            {
                "operation": "set_potential",
                "potential_V": 1.15,
                "current_mA": 75.0,
                "electrolyte_profile": 2,
            }
        )

        locked_schema = env.unwrapped.action_schema("set_potential")
        locked = next(
            field for field in locked_schema["fields"] if field["field"] == "electrolyte_profile"
        )
        assert locked["choices"] == [2]
        assert locked["state_dependent_choices"] is True
        switched = env.unwrapped.validate_action(
            {
                "operation": "set_potential",
                "potential_V": 1.10,
                "current_mA": 70.0,
                "electrolyte_profile": 1,
            }
        )
        assert switched["valid"] is False
        assert "payload_locked:electrolyte_profile" in switched["invalid_reasons"]

        injected = env.unwrapped.validate_action(
            {
                "operation": "set_potential",
                "potential_V": 1.10,
                "current_mA": 70.0,
                "electrolyte_profile": 2,
                "electrolyte_conductivity_S_m": 100.0,
            }
        )
        assert injected["valid"] is False
        assert "payload_fields_declared" in injected["invalid_reasons"]
    finally:
        env.close()


def test_core_allows_mixed_solvents_in_current_experiment() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.02, "solvent": 1})
        schema = env.unwrapped.action_schema("add_solvent")
        solvent_field = next(field for field in schema["fields"] if field["field"] == "solvent")
        assert solvent_field["choices"] == [0, 1, 2, 3]
        assert "locked_for_current_experiment" not in solvent_field

        same = env.unwrapped.validate_action(
            {"operation": "add_solvent", "volume_L": 0.005, "solvent": 1}
        )
        switched = env.unwrapped.validate_action(
            {"operation": "add_solvent", "volume_L": 0.005, "solvent": 2}
        )
        assert same["valid"]
        assert switched["valid"]
        assert switched["dispatchable_to_runtime"]
    finally:
        env.close()


def test_core_task_prompts_are_structured_and_public() -> None:
    expectations = {
        "reaction-to-assay": {
            "must_include": [
                "single-experiment",
                "final_assay_score",
                "trajectory_validity",
                "Terminate",
            ],
            "metrics": {"final_assay_score", "trajectory_validity"},
        },
        "reaction-to-purification": {
            "must_include": [
                "purity",
                "recovery",
                "process_mass_balance_error",
                "phase separation",
                "final_assay",
            ],
            "metrics": {"score", "purity", "recovery", "process_mass_balance_error"},
        },
        "partition-discovery": {
            "must_include": [
                "Campaign task",
                "partition",
                "phase_ratio",
                "product_in_organic",
                "product_in_aqueous",
            ],
            "metrics": {"phase_ratio", "product_in_organic", "product_in_aqueous"},
        },
    }
    for task_id, expected in expectations.items():
        env = gym.make("ChemWorld", task_id=task_id, seed=0)
        try:
            env.reset(seed=0)
            prompt = env.unwrapped.task_prompt()
            text = prompt["text"]
            assert prompt["prompt_version"] == "chemworld-agent-task-prompt-0.3"
            lifecycle = prompt["experiment_lifecycle"]
            assert lifecycle["terminate_action_template"] == '{"operation":"terminate"}'
            assert lifecycle["final_assay_action_template"] == (
                '{"operation":"measure","instrument":"final_assay"}'
            )
            assert "does not by itself complete" in lifecycle["terminate_effect"]
            assert "instrument=final_assay" in lifecycle["final_assay_precondition"]
            if prompt["episode_mode"] == "campaign":
                assert "fresh experiment" in lifecycle["final_assay_effect"]
            else:
                assert "ends the episode" in lifecycle["final_assay_effect"]
            assert "do not complete" in lifecycle["intermediate_measurement_effect"]
            assert prompt["task_id"] == task_id
            assert str(prompt["budget"]) in text
            assert "budget" in text
            assert "Hidden information policy" in text
            assert "Allowed tools" in text
            assert "Success criteria" in text
            assert "Submission requirements" in text
            assert prompt["task_goal"]
            assert prompt["constraints"]
            assert prompt["success_criteria"]
            assert prompt["measurement_policy"]
            assert prompt["recommended_strategy"]
            assert prompt["failure_modes"]
            assert prompt["hidden_information_policy"]
            assert prompt["allowed_tools"]["operations"] == prompt["allowed_operations"]
            assert prompt["allowed_tools"]["instruments"] == prompt["allowed_instruments"]
            assert set(prompt["success_metrics"]) == expected["metrics"]
            lower_text = text.lower()
            for phrase in expected["must_include"]:
                assert phrase.lower() in lower_text
            hidden_species = set(env.unwrapped.scenario_instance.compiled_mechanism.species_index)
            assert not audit_public_payload(prompt, hidden_species_ids=hidden_species)
        finally:
            env.close()


def test_observation_views_are_json_safe_and_public() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 382.0,
                "duration_s": 1350.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "measure", "instrument": "hplc"},
        ):
            env.step(action)

        rl_view = env.unwrapped.observation_view("rl")
        assert rl_view["schema_version"] == "chemworld-rl-view-0.2"
        assert rl_view["nan_safe"]
        assert rl_view["missing_value"] == -1.0
        assert len(rl_view["vector"]) == len(rl_view["mask"])
        assert len(rl_view["bounds"]["low"]) == len(rl_view["vector"])
        assert len(rl_view["bounds"]["high"]) == len(rl_view["vector"])
        assert all(np.isfinite(float(value)) for value in rl_view["vector"])
        for value, observed in zip(rl_view["vector"], rl_view["mask"], strict=True):
            if observed == 0.0:
                assert value == rl_view["missing_value"]
        assert rl_view["keys"] == [*OBSERVATION_KEYS, "cost_signal"]
        assert rl_view["cost"] >= 0.0

        tool_view = env.unwrapped.observation_view("tool_json")
        json.dumps(tool_view, sort_keys=True)
        hidden_species = set(env.unwrapped.scenario_instance.compiled_mechanism.species_index)
        species_ids = _nested_values_for_key(tool_view["raw_signal"], "species_id")
        assert species_ids.isdisjoint(hidden_species)

        lab_report = env.unwrapped.observation_view("lab_report")
        assert "Visible score" in lab_report["text"]
        assert "Key public metrics" in lab_report["text"]
        assert "Campaign progress" in lab_report["text"]
        assert "Spectra packet" in lab_report["text"]
        assert lab_report["report_version"] == "chemworld-lab-report-0.2"
        assert "spectra_summary" in lab_report
        assert lab_report["spectra_summary"]["has_spectral_packet"]
        assert lab_report["spectra_summary"]["peak_table_count"] >= 1
        assert lab_report["instrument_summary"]["instrument"] == "hplc"
        assert lab_report["instrument_summary"]["is_measurement"]
        assert "score" in lab_report["visible_metrics"]
        assert lab_report["campaign_progress"]["remaining_budget"] >= 0
    finally:
        env.close()


def test_lab_report_includes_failed_action_recovery() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "heat", "duration_s": 100.0})
        report = env.unwrapped.observation_view("lab_report")
        assert report["status"] == "failed_precondition"
        assert report["recovery_suggestion"]
        assert report["failure_summary"]["precondition_failed"]
        assert report["failure_summary"]["failed_preconditions"]
        assert report["next_action_hints"]
        assert "Recovery suggestion" in report["text"]
    finally:
        env.close()


def test_agent_wrappers_preserve_gym_shape_and_add_views() -> None:
    info_env = AgentInfoWrapper(gym.make("ChemWorld", task_id="reaction-to-assay", seed=0))
    try:
        _, info = info_env.reset(seed=0)
        assert "task_prompt" in info
        assert "campaign_state" in info
        assert "available_actions" in info
        result = info_env.step({"operation": "add_solvent", "volume_L": 0.02, "solvent": 1})
        assert len(result) == 5
    finally:
        info_env.close()

    llm_env = LLMObservationWrapper(gym.make("ChemWorld", task_id="reaction-to-assay", seed=0))
    try:
        _, info = llm_env.reset(seed=0)
        assert "lab_report" in info
        _, _, _, _, info = llm_env.step({"operation": "heat", "duration_s": 10.0})
        assert "recovery suggestion" in info["lab_report"]["text"].lower()
    finally:
        llm_env.close()

    rl_env = RLObservationWrapper(gym.make("ChemWorld", task_id="reaction-to-assay", seed=0))
    try:
        observation, info = rl_env.reset(seed=0)
        assert observation.shape == rl_env.observation_space.shape
        assert np.all(np.isfinite(observation))
        assert rl_env.observation_space.contains(observation)
        assert np.all(np.isfinite(rl_env.observation_space.low))
        assert np.all(np.isfinite(rl_env.observation_space.high))
        assert "rl_view" in info
        spec = rl_observation_spec()
        assert info["rl_view"]["keys"] == spec["keys"]
        assert info["rl_view"]["bounds"] == spec["value_bounds"]
        assert info["observation_mask"].shape == (len(spec["keys"]),)
        assert "cost_signal" in info
    finally:
        rl_env.close()

    suggestion_env = ActionSuggestionWrapper(
        gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    )
    try:
        _, info = suggestion_env.reset(seed=0)
        assert "action_suggestions" in info
        _, _, _, _, info = suggestion_env.step({"operation": "heat", "duration_s": 10.0})
        assert "recovery_suggestion" in info
    finally:
        suggestion_env.close()


def test_campaign_state_updates_after_campaign_final_assay() -> None:
    env = gym.make("ChemWorld", task_id="reaction-optimization-standard", seed=0)
    try:
        env.reset(seed=0)
        initial_state = env.unwrapped._state
        sequence = [
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "quench"},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
        info = {}
        for action in sequence:
            _, _, terminated, truncated, info = env.step(action)
        state = env.unwrapped.campaign_state()
        report = env.unwrapped.observation_view("lab_report")
        assert not terminated
        assert not truncated
        assert info["experiment_ended"]
        assert state["final_assay_count"] == 1
        assert state["experiment_index"] == 1
        assert state["best_score"] is not None
        assert report["final_assay_summary"]["is_final_assay"]
        assert report["final_assay_summary"]["leaderboard_eligible"]
        assert report["campaign_progress"]["final_assay_count"] == 1
        assert "Final assay" in report["text"]
        assert "Best score so far" in report["text"]
        fresh_state = env.unwrapped._state
        assert fresh_state.volume_L == initial_state.volume_L
        assert fresh_state.species_amounts == initial_state.species_amounts
    finally:
        env.close()


def test_reaction_heat_and_material_actions_accumulate_in_one_vessel() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "add_catalyst",
                "catalyst_amount_mol": 0.00025,
                "catalyst": 1,
            },
        ):
            env.step(action)

        env.step(
            {
                "operation": "heat",
                "target_temperature_K": 360.0,
                "duration_s": 600.0,
                "stirring_speed_rpm": 720.0,
            }
        )
        after_first_heat = env.unwrapped._state
        env.step(
            {
                "operation": "heat",
                "target_temperature_K": 360.0,
                "duration_s": 600.0,
                "stirring_speed_rpm": 720.0,
            }
        )
        after_second_heat = env.unwrapped._state

        assert after_first_heat.ledger.time_s == pytest.approx(600.0)
        assert after_second_heat.ledger.time_s == pytest.approx(1200.0)
        assert any(
            after_second_heat.species_amounts[key]
            != pytest.approx(after_first_heat.species_amounts[key])
            for key in after_second_heat.species_amounts
        )

        before_solvent = after_second_heat
        env.step({"operation": "add_solvent", "volume_L": 0.010, "solvent": 2})
        after_solvent = env.unwrapped._state
        assert after_solvent.volume_L == pytest.approx(before_solvent.volume_L + 0.010)
        assert after_solvent.species_amounts == before_solvent.species_amounts
        assert after_solvent.ledger.time_s == before_solvent.ledger.time_s
        assert env.unwrapped.campaign_state()["experiment_index"] == 0
    finally:
        env.close()


def test_tool_using_llm_stub_runs_agent_facing_tasks(tmp_path) -> None:
    output = tmp_path / "tool_stub.jsonl"
    history = run_agent(
        env_id="ChemWorld",
        agent=ToolUsingLLMStubAgent(),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=output,
    )
    records = load_jsonl(output)
    assert history[-1].info["leaderboard_score"] is not None
    assert trajectory_value(records[-1], 'agent_view')["lab_report"]["mode"] == "lab_report"
    assert records[-1]["agent_trace"]
    latest_trace = records[-1]["agent_trace"][-1]
    assert latest_trace["prompt_input"]
    assert latest_trace["selected_action"]
    assert latest_trace["validator_result"]
    assert latest_trace["observation_summary"]
    assert latest_trace["memory_note"]
    flattened = flatten_record(records[-1])
    assert "environment_outcome" in flattened
    assert "agent_visible_observation" in flattened
    assert "evaluation_outcome" in flattened
    assert "agent_view" in flattened
    assert "agent_trace" in flattened
    assert flattened["agent_trace_step_count"] == len(records[-1]["agent_trace"])
    assert json.loads(flattened["agent_trace_prompt_summary"])
    assert json.loads(flattened["agent_trace_selected_action"])["operation"] == "measure"
    assert json.loads(flattened["agent_trace_validation_result"])
    assert json.loads(flattened["agent_trace_observation_summary"])
    assert flattened["agent_trace_memory_note"]


def test_dataset_agent_trace_export_demo(tmp_path) -> None:
    summary = build_demo(tmp_path / "agent_trace_dataset_demo")
    assert Path(summary["trajectory_path"]).exists()
    assert summary["jsonl_export"]["record_count"] == summary["record_count"]
    assert summary["agent_trace_step_count"] == summary["record_count"]
    assert json.loads(summary["agent_trace_selected_action"])["operation"] == "measure"
    assert json.loads(summary["agent_trace_validation_result"])
    assert json.loads(summary["agent_trace_observation_summary"])
    assert summary["agent_trace_memory_note"]
    assert "Visible score" in summary["lab_report_text"]


def test_llm_replay_agent_replays_action_trace(tmp_path) -> None:
    replay = tmp_path / "trace.jsonl"
    actions = [
        {"action": {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}},
        {"action": {"operation": "add_reagent", "amount_mol": 0.010}},
        {"action": {"operation": "terminate"}},
        {"action": {"operation": "measure", "instrument": "final_assay"}},
    ]
    replay.write_text(
        "\n".join(json.dumps(action, sort_keys=True) for action in actions),
        encoding="utf-8",
    )
    history = run_agent(
        env_id="ChemWorld",
        agent=LLMReplayAgent(replay),
        world_split="public-dev",
        budget=8,
        objective="balanced",
        seed=3,
        task_id="reaction-to-assay",
    )
    assert [record.action["operation"] for record in history[:4]] == [
        "add_solvent",
        "add_reagent",
        "terminate",
        "measure",
    ]


def _nested_values_for_key(payload: object, key: str) -> set[str]:
    if isinstance(payload, dict):
        values = {str(payload[key])} if key in payload else set()
        for value in payload.values():
            values.update(_nested_values_for_key(value, key))
        return values
    if isinstance(payload, list | tuple):
        values: set[str] = set()
        for item in payload:
            values.update(_nested_values_for_key(item, key))
        return values
    return set()
