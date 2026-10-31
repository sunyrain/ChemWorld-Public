from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import chemworld  # noqa: F401
from chemworld.action_codec import ActionCodec
from chemworld.agents.base import HistoryRecord
from chemworld.envs.spaces import NullableScalarBox, to_observation
from chemworld.foundation import upsert_equipment_record
from chemworld.legacy.agents.event import ScriptedChemistryAgent
from chemworld.operation_validator import OperationValidator
from chemworld.runtime import TaskRuntimeProfile, make_chemworld_constitution
from chemworld.tasks import (
    CORE_TASK_IDS,
    get_task,
    get_task_card,
    list_core_task_cards,
    list_core_tasks,
    list_tasks,
)
from chemworld.world.actions import canonicalize_action, vector_to_action
from chemworld.world.operations import (
    CRYSTALLIZATION_OPERATIONS,
    DISTILLATION_OPERATIONS,
    ELECTROCHEMISTRY_OPERATIONS,
    FLOW_OPERATIONS,
    OPERATION_TYPES,
    REACTION_OPERATIONS,
    SEPARATION_OPERATIONS,
    operation_contracts,
)
from chemworld.world.parameters import WORLD_FAMILY_VERSION
from chemworld.world.scoring import (
    DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2,
    FLOW_S0_BALANCED_PROCESS_V1,
    PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
    TaskScoringContract,
    score_observation,
    task_score_observation,
)
from chemworld.world.state_factory import initial_chemworld_state
from chemworld.wrappers import (
    ActionMaskWrapper,
    ContinuousEventActionWrapper,
    NaNObservationWrapper,
    SafetyCostWrapper,
    validate_event_action,
    validate_operation_affordance,
)


def test_builtin_tasks_are_instantiable() -> None:
    task_ids = {task.task_id for task in list_tasks()}
    assert {
        "reaction-to-assay",
        "reaction-optimization-standard",
        "reaction-safety-constrained",
        "public-private-generalization",
        "reaction-mechanism-explanation",
        "reaction-to-crystallization",
        "reaction-to-distillation",
        "flow-reaction-optimization",
        "electrochemical-conversion",
    } <= task_ids
    task = get_task("reaction-optimization-standard")
    assert task.env_id == "ChemWorld"
    assert task.world_law_id == WORLD_FAMILY_VERSION
    assert task.episode_mode == "campaign"
    assert task.termination_policy == "budget"
    assert task.env_kwargs(seed=7)["seed"] == 7
    assert "hplc" in task.allowed_instruments
    card = get_task_card("reaction-optimization-standard")
    assert card["task_id"] == "reaction-optimization-standard"
    assert card["world_law_id"] == WORLD_FAMILY_VERSION
    assert "baseline_reference_scores" in card
    assert "failure_modes" in card
    assay_task = get_task("reaction-to-assay")
    assert assay_task.episode_mode == "single_experiment"
    assert assay_task.termination_policy == "final-assay-or-budget"


def test_core_task_contracts_describe_current_presets() -> None:
    tasks = {task.task_id: task for task in list_core_tasks()}
    assert tuple(tasks) == CORE_TASK_IDS

    assay = tasks["reaction-to-assay"]
    assert assay.world_split == "public-dev"
    assert assay.budget == 18
    assert assay.seeds == (0,)
    assert assay.episode_mode == "single_experiment"
    assert assay.allowed_operations == REACTION_OPERATIONS
    assert assay.allowed_instruments == ("hplc", "gc", "uvvis", "final_assay", "nmr", "ir", "ms")
    assert assay.success_metrics == ("final_assay_score", "trajectory_validity")
    assert assay.safety_limit == 0.65

    purification = tasks["reaction-to-purification"]
    assert purification.world_split == "public-test"
    assert purification.budget == 90
    assert purification.seeds == (0, 1, 2, 3, 4)
    assert purification.episode_mode == "single_experiment"
    assert purification.allowed_operations == (
        *REACTION_OPERATIONS,
        *SEPARATION_OPERATIONS,
    )
    assert not set(purification.allowed_operations).intersection(
        {
            *CRYSTALLIZATION_OPERATIONS,
            *DISTILLATION_OPERATIONS,
            *FLOW_OPERATIONS,
            *ELECTROCHEMISTRY_OPERATIONS,
        }
    )
    assert purification.success_metrics == (
        "score",
        "purity",
        "recovery",
        "process_mass_balance_error",
    )
    assert purification.safety_limit == 0.65

    partition = tasks["partition-discovery"]
    assert partition.world_split == "public-test"
    assert partition.budget == 48
    assert partition.seeds == (0, 1, 2, 3, 4)
    assert partition.episode_mode == "campaign"
    assert partition.success_metrics == (
        "phase_ratio",
        "product_in_organic",
        "product_in_aqueous",
    )
    assert partition.safety_limit == 0.65


def test_core_task_cards_are_complete_release_contracts() -> None:
    cards = {card["task_id"]: card for card in list_core_task_cards()}
    assert tuple(cards) == CORE_TASK_IDS

    for task_id, card in cards.items():
        task = get_task(task_id)
        contract = card["benchmark_contract"]
        assert "core" in card["suite_memberships"]
        assert card["task_contract_version"] == "chemworld-task-contract-1.1"
        assert card["task_contract_hash"] == task.contract_hash
        assert len(card["task_contract_hash"]) == 64
        assert contract["objective"] == task.objective
        assert contract["budget"] == task.budget
        assert contract["episode_mode"] == task.episode_mode
        assert contract["world_split"] == task.world_split
        assert contract["success_metrics"] == list(task.success_metrics)
        assert contract["allowed_operations"] == list(task.allowed_operations)
        assert contract["allowed_instruments"] == list(task.allowed_instruments)
        assert contract["safety_limit"] == task.safety_limit
        assert card["reward_leaderboard_metric"]["objective_score_threshold"] == task.threshold
        assert card["reward_leaderboard_metric"]["threshold_semantics"] == (
            "leaderboard_objective_score_not_primary_metric"
        )
        assert card["kernel_maturity"]["modules"]
        assert card["physics_maturity"] in {"proxy", "lite", "reference_validated"}
        assert card["expected_qualitative_behavior"]


def test_all_tasks_share_one_world_law() -> None:
    tasks = list_tasks()
    assert {task.env_id for task in tasks} == {"ChemWorld"}
    assert {task.world_law_id for task in tasks} == {WORLD_FAMILY_VERSION}
    assert "reaction-to-purification" in {task.task_id for task in tasks}
    assert "partition-discovery" in {task.task_id for task in tasks}
    assert "purity-yield-tradeoff" in {task.task_id for task in tasks}
    assert "reaction-to-crystallization" in {task.task_id for task in tasks}
    assert "flow-reaction-optimization" in {task.task_id for task in tasks}


def test_task_specific_scoring_contracts_are_explicit() -> None:
    values = {
        "yield": 0.40,
        "selectivity": 0.75,
        "conversion": 0.60,
        "cost": 0.10,
        "safety_risk": 0.20,
        "purity": 0.90,
        "recovery": 0.70,
        "process_mass_balance_error": 0.02,
    }
    reaction_contract = TaskScoringContract.from_success_metrics(
        objective="balanced",
        success_metrics=("score", "yield", "selectivity"),
    )
    purification_contract = TaskScoringContract.from_success_metrics(
        objective="balanced",
        success_metrics=("score", "purity", "recovery", "process_mass_balance_error"),
    )

    reaction_score = score_observation(
        objective="balanced",
        product_yield=0.40,
        selectivity=0.75,
        conversion=0.60,
        cost=0.10,
        safety_risk=0.20,
    )
    expected_purification = 0.35 * reaction_score + 0.35 * 0.90 + 0.25 * 0.70 - 0.10 * 0.02

    assert reaction_contract.score_family == "reaction"
    assert purification_contract.score_family == "purification"
    assert task_score_observation(contract=reaction_contract, values=values) == (reaction_score)
    assert task_score_observation(contract=purification_contract, values=values) == (
        expected_purification
    )


def test_distillation_s0_v2_records_but_does_not_score_safety_risk() -> None:
    task = get_task("reaction-to-distillation")
    contract = TaskScoringContract.from_success_metrics(
        objective=task.objective,
        success_metrics=task.success_metrics,
        contract_id=DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2,
    )
    values = {
        "yield": 0.55,
        "selectivity": 0.85,
        "conversion": 0.65,
        "cost": 0.40,
        "distillate_purity": 0.60,
        "distillate_recovery": 0.75,
        "solvent_loss": 0.20,
    }

    low_risk = task_score_observation(
        contract=contract,
        values={**values, "safety_risk": 0.05},
    )
    high_risk = task_score_observation(
        contract=contract,
        values={**values, "safety_risk": 0.95},
    )

    assert contract.score_family == "distillation"
    assert low_risk == high_risk


def test_partition_score_contains_only_active_partition_components() -> None:
    task = get_task("partition-discovery")
    contract = TaskScoringContract.from_success_metrics(
        objective=task.objective,
        success_metrics=task.success_metrics,
        contract_id=PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
    )
    partition_values = {
        "product_in_organic": 0.70,
        "phase_ratio": 0.60,
        "product_in_aqueous": 0.20,
    }

    low_reaction_metrics = task_score_observation(
        contract=contract,
        values={
            **partition_values,
            "yield": 0.0,
            "conversion": 0.0,
            "selectivity": 0.0,
            "cost": 1.0,
            "safety_risk": 1.0,
        },
    )
    high_reaction_metrics = task_score_observation(
        contract=contract,
        values={
            **partition_values,
            "yield": 1.0,
            "conversion": 1.0,
            "selectivity": 1.0,
            "cost": 0.0,
            "safety_risk": 0.0,
        },
    )

    assert contract.score_family == "partition"
    assert contract.component_weights == {
        "product_in_organic": 0.85,
        "product_in_aqueous": -0.10,
        "phase_ratio": -0.10,
    }
    assert low_reaction_metrics == pytest.approx(0.515)
    assert high_reaction_metrics == low_reaction_metrics


def test_flow_score_has_no_nested_reaction_composite() -> None:
    task = get_task("flow-reaction-optimization")
    contract = TaskScoringContract.from_success_metrics(
        objective=task.objective,
        success_metrics=task.success_metrics,
        contract_id=FLOW_S0_BALANCED_PROCESS_V1,
    )
    values = {
        "flow_conversion": 0.60,
        "yield": 0.50,
        "selectivity": 0.70,
        "cost": 0.20,
        "safety_risk": 0.10,
    }

    score = task_score_observation(
        contract=contract,
        values={
            **values,
            "conversion": 0.0,
        },
    )
    score_with_unrelated_reaction_fields = task_score_observation(
        contract=contract,
        values={
            **values,
            "conversion": 1.0,
        },
    )

    assert contract.component_weights == {
        "flow_conversion": 0.40,
        "yield": 0.30,
        "selectivity": 0.20,
        "cost": -0.04,
        "safety_risk": -0.06,
    }
    assert "reaction_score" not in contract.component_weights
    assert score == pytest.approx(0.516)
    assert score_with_unrelated_reaction_fields == score


def test_env_task_info_exposes_scoring_contract() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        _, info = env.reset(seed=0)
        contract = info["scoring_contract"]
        assert contract["objective"] == "balanced"
        assert contract["score_family"] == "purification"
        assert "purity" in contract["component_weights"]
        assert "process_mass_balance_error" in contract["success_metrics"]
        assert contract["contract_hash"] == info["scoring_contract_hash"]
        assert len(info["scoring_contract_hash"]) == 64
    finally:
        env.close()


def test_env_task_info_exposes_task_and_runtime_profile_hashes() -> None:
    task = get_task("reaction-to-purification")
    env = gym.make("ChemWorld", task_id=task.task_id, seed=0)
    try:
        _, info = env.reset(seed=0)
        profile = TaskRuntimeProfile.from_task(task)

        assert task.to_dict()["contract_hash"] == task.contract_hash
        assert profile.to_dict()["profile_hash"] == profile.profile_hash
        assert info["task_contract_hash"] == task.contract_hash
        assert (
            env.unwrapped.runtime.to_dict()["profile"]["profile_hash"]
            == info["runtime_profile_hash"]
        )
        assert info["runtime_profile_hash"] == profile.profile_hash
        assert len(info["task_contract_hash"]) == 64
        assert len(info["runtime_profile_hash"]) == 64
    finally:
        env.close()


def test_env_task_info_exposes_task_observation_contract() -> None:
    reaction_env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    purification_env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        _, reaction_info = reaction_env.reset(seed=0)
        _, purification_info = purification_env.reset(seed=0)

        reaction_contract = reaction_info["observation_contract"]
        purification_contract = purification_info["observation_contract"]

        assert reaction_contract["score_family"] == "reaction"
        assert "yield" in reaction_contract["instrument_observable_keys"]["hplc"]
        assert "purity" not in reaction_contract["instrument_observable_keys"]["hplc"]
        assert "recovery" not in reaction_contract["instrument_observable_keys"]["final_assay"]

        assert purification_contract["score_family"] == "purification"
        assert "purity" in purification_contract["instrument_observable_keys"]["hplc"]
        assert "recovery" in purification_contract["instrument_observable_keys"]["final_assay"]
        assert (
            "process_mass_balance_error"
            in purification_contract["instrument_observable_keys"]["final_assay"]
        )
        assert "mechanism_observable_mapping" not in purification_contract
        assert (
            purification_contract["mapping_visibility_policy"]
            == "hidden mechanism-to-species mapping is not public"
        )
        assert reaction_contract["contract_hash"] == reaction_info["observation_contract_hash"]
        assert (
            purification_contract["contract_hash"] == purification_info["observation_contract_hash"]
        )
        assert len(purification_info["observation_contract_hash"]) == 64
    finally:
        reaction_env.close()
        purification_env.close()


def test_observation_kernel_filters_measurements_by_task_contract() -> None:
    reaction_env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    purification_env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        reaction_env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "heat",
                "target_temperature_K": 360.0,
                "duration_s": 600.0,
                "stirring_speed_rpm": 650.0,
            },
        ):
            reaction_env.step(action)
        _, _, _, _, reaction_info = reaction_env.step(
            {"operation": "measure", "instrument": "hplc"}
        )
        assert "yield" in reaction_info["observed_keys"]
        assert "purity" not in reaction_info["observed_keys"]
        assert "purity" not in reaction_info["processed_estimate"]

        purification_env.reset(seed=0)
        for action in (
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
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
            {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
            {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
            {"operation": "settle", "duration_s": 420.0},
            {"operation": "separate_phase", "target_phase": "organic"},
            {"operation": "terminate"},
        ):
            purification_env.step(action)
        _, _, terminated, _, purification_info = purification_env.step(
            {"operation": "measure", "instrument": "final_assay"}
        )
        assert terminated
        assert "purity" in purification_info["observed_keys"]
        assert "recovery" in purification_info["observed_keys"]
        assert "process_mass_balance_error" in purification_info["processed_estimate"]
    finally:
        reaction_env.close()
        purification_env.close()


def test_env_raw_signal_uses_public_task_species_contract() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        env.reset(seed=0)
        for action in (
            {"operation": "add_solvent", "volume_L": 0.026, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "set_potential",
                "potential_V": 1.15,
                "current_mA": 75.0,
                "electrolyte_profile": 1,
            },
            {"operation": "electrolyze", "duration_s": 1800.0},
            {"operation": "measure", "instrument": "ph_meter"},
            {"operation": "measure", "instrument": "uvvis"},
            {
                "operation": "set_potential",
                "potential_V": 1.25,
                "current_mA": 75.0,
                "electrolyte_profile": 1,
            },
            {"operation": "electrolyze", "duration_s": 1800.0},
            {"operation": "measure", "instrument": "uvvis"},
            {"operation": "terminate"},
        ):
            env.step(action)
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "final_assay"})

        raw_signal = info["raw_signal"]
        species_ids = _nested_values_for_key(raw_signal, "species_id")
        detection_limit_species = _nested_detection_limit_species(raw_signal)
        public_species = {
            "reactant_public",
            "target_public",
            "impurity_public",
            "degradation_public",
        }
        hidden_species = set(env.unwrapped.scenario_instance.compiled_mechanism.species_index)

        assert raw_signal["visibility"] == "task_observation_contract"
        assert species_ids
        assert species_ids <= public_species
        assert detection_limit_species <= public_species
        assert species_ids.isdisjoint(hidden_species)
        assert detection_limit_species.isdisjoint(hidden_species)
    finally:
        env.close()


def test_action_mask_wrapper_reports_valid_operations() -> None:
    env = ActionMaskWrapper(gym.make("ChemWorld", task_id="reaction-to-assay", seed=0))
    try:
        _, info = env.reset(seed=0)
        assert "action_mask" in info
        assert "add_solvent" in info["valid_operations"]
        assert "heat" not in info["valid_operations"]

        validation = validate_event_action({"operation": "heat"}, env)
        assert not validation["valid"]
        assert not validation["preconditions"]["has_volume"]

        _, _, _, _, step_info = env.step(
            {"operation": "add_solvent", "volume_L": 0.02, "solvent": 1}
        )
        assert "heat" not in step_info["valid_operations"]
        _, _, _, _, step_info = env.step({"operation": "add_reagent", "amount_mol": 0.01})
        assert "heat" in step_info["valid_operations"]
    finally:
        env.close()


def test_action_codec_roundtrip_vector() -> None:
    codec = ActionCodec()
    action = {
        "operation": "heat",
        "target_temperature_K": 386.0,
        "duration_s": 900.0,
        "instrument": "hplc",
    }
    vector = codec.encode_vector(action)
    decoded = codec.decode_vector(vector)
    assert decoded["operation"] == "heat"
    assert set(decoded) == {
        "operation",
        "target_temperature_K",
        "duration_s",
    }
    assert decoded["target_temperature_K"] == 386.0
    assert decoded["duration_s"] == 900.0
    process_vector = codec.encode_vector(
        {
            "operation": "distill",
            "target_temperature_K": 360.0,
            "duration_s": 900.0,
            "reflux_ratio": 2.5,
        }
    )
    process_decoded = codec.decode_vector(process_vector)
    assert process_decoded["operation"] == "distill"
    assert process_decoded["reflux_ratio"] == 2.5


def test_raw_and_flat_action_spaces_emit_only_operation_conditional_fields() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        codec = env.unwrapped.operation_validator.action_codec
        contracts = codec.contracts()
        for _ in range(32):
            action = env.action_space.sample()
            operation = codec.operation_types[int(action["operation"])]
            assert set(action) == {"operation", *contracts[operation].required_fields}
            assert (
                "payload_fields_declared"
                not in env.unwrapped.validate_action(action)["invalid_reasons"]
            )

        wrapped = ContinuousEventActionWrapper(env)
        contracts = operation_contracts()
        for _ in range(32):
            action = wrapped.action(wrapped.action_space.sample())
            operation = OPERATION_TYPES[int(action["operation"])]
            assert set(action) == {"operation", *contracts[operation].required_fields}
    finally:
        env.close()


@pytest.mark.parametrize(
    "vector",
    [
        np.zeros(22, dtype=float),
        np.zeros(24, dtype=float),
        np.full(23, np.nan, dtype=float),
        np.full(23, np.inf, dtype=float),
    ],
    ids=("short", "long", "nan", "infinite"),
)
def test_action_codec_rejects_non_contract_vectors(vector: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite with shape"):
        ActionCodec().decode_vector(vector)


def test_action_codec_rejects_nonfinite_encoded_payloads_and_choices() -> None:
    codec = ActionCodec()
    with pytest.raises(ValueError, match="amount_mol must be finite"):
        codec.encode_vector({"operation": "add_reagent", "amount_mol": float("inf")})
    with pytest.raises(ValueError, match="finite integer categorical index"):
        codec.canonicalize({"operation": "add_solvent", "volume_L": 0.02, "solvent": float("inf")})


def test_terminal_recipe_action_boundaries_reject_nonfinite_coordinates() -> None:
    action = {
        "temperature": 80.0,
        "time": 1.0,
        "initial_concentration": 0.5,
        "stirring_speed": 600.0,
        "catalyst": 1,
        "solvent": 2,
    }
    with pytest.raises(ValueError, match="finite"):
        canonicalize_action({**action, "temperature": float("nan")})
    with pytest.raises(ValueError, match="finite"):
        canonicalize_action({**action, "catalyst": float("inf")})
    with pytest.raises(ValueError, match="finite"):
        vector_to_action(np.asarray([0.0, 0.5, np.nan, 1.0, 0.0, 0.0]))

    clipped = canonicalize_action({**action, "catalyst": 99, "solvent": -2})
    assert clipped["catalyst"] == 3
    assert clipped["solvent"] == 0


def test_nullable_observation_space_accepts_nan_but_rejects_infinity() -> None:
    box = NullableScalarBox(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    assert box.contains(np.asarray([np.nan], dtype=np.float32))
    assert not box.contains(np.asarray([np.inf], dtype=np.float32))
    assert not box.contains(np.asarray([-np.inf], dtype=np.float32))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01, True])
def test_observation_encoding_rejects_noncontract_values(value: object) -> None:
    with pytest.raises(ValueError, match="observation value"):
        to_observation({"score": value})  # type: ignore[dict-item]


def test_action_codec_accepts_common_recipe_aliases() -> None:
    codec = ActionCodec()
    catalyst_action = codec.canonicalize(
        {"operation": "add_catalyst", "catalyst": 2, "amount_mol": 0.0004}
    )
    assert catalyst_action["catalyst_amount_mol"] == 0.0004

    heat_action = codec.canonicalize(
        {
            "operation": "heat",
            "temperature_K": 350.0,
            "duration_s": 1200.0,
            "stirring_rpm": 800.0,
        }
    )
    assert heat_action["target_temperature_K"] == 350.0
    assert heat_action["stirring_speed_rpm"] == 800.0

    phase_action = codec.canonicalize({"operation": "separate_phase", "phase": "organic"})
    assert phase_action["target_phase"] == "organic"

    solvent_action = codec.canonicalize(
        {"operation": "add_solvent", "volume_L": 0.02, "solvent": "ethanol"}
    )
    assert solvent_action["solvent"] == 1
    named_catalyst = codec.canonicalize(
        {"operation": "add_catalyst", "catalyst_amount_mol": 0.0002, "catalyst": "cat_c"}
    )
    assert named_catalyst["catalyst"] == 2

    with pytest.raises(ValueError, match="Unknown material choice"):
        codec.canonicalize({"operation": "add_solvent", "volume_L": 0.02, "solvent": "unobtainium"})


def test_action_mask_is_task_aware() -> None:
    reaction_env = ActionMaskWrapper(
        gym.make("ChemWorld", task_id="reaction-optimization-standard", seed=0)
    )
    purification_env = ActionMaskWrapper(
        gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    )
    try:
        _, reaction_info = reaction_env.reset(seed=0)
        _, purification_info = purification_env.reset(seed=0)
        assert "add_extractant" not in reaction_info["valid_operations"]
        assert "add_extractant" not in purification_info["valid_operations"]

        validation = validate_event_action({"operation": "add_extractant"}, reaction_env)
        assert not validation["valid"]
        assert not validation["preconditions"]["operation_allowed_by_task"]
        assert "operation_allowed_by_task" in validation["invalid_reasons"]

        purification_env.step({"operation": "add_solvent", "volume_L": 0.02, "solvent": 1})
        purification_env.step({"operation": "add_reagent", "amount_mol": 0.01})
        _, _, _, _, info = purification_env.step(
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.010}
        )
        assert "add_extractant" in info["valid_operations"]
    finally:
        reaction_env.close()
        purification_env.close()


def test_agent_affordances_match_masks_validator_and_task_policy() -> None:
    for task in list_tasks():
        env = ActionMaskWrapper(gym.make("ChemWorld", task_id=task.task_id, seed=0))
        try:
            _, info = env.reset(seed=0)
            base = env.unwrapped
            validator_valid = set(base.operation_validator.valid_operations(base._state))
            wrapper_valid = set(info["valid_operations"])
            agent_entries = base.available_actions()
            agent_valid = {entry["operation"] for entry in agent_entries}
            mask = list(base.operation_validator.action_mask(base._state))
            masked_valid = {
                operation
                for operation, is_valid in zip(base.operation_validator.operation_types,
                                              mask, strict=True)
                if is_valid
            }

            assert validator_valid == agent_valid == masked_valid
            assert wrapper_valid == validator_valid.intersection(OPERATION_TYPES)
            assert list(info["action_mask"]) == [op in validator_valid for op in OPERATION_TYPES]
            assert len(mask) == len(base.operation_validator.operation_types)
            assert all(entry["valid"] for entry in agent_entries)
            assert all(not entry["invalid_reasons"] for entry in agent_entries)

            all_entries = {
                entry["operation"]: entry for entry in base.available_actions(include_invalid=True)
            }
            assert set(all_entries) == set(base.operation_validator.operation_types)
            for operation, entry in all_entries.items():
                affordance = validate_operation_affordance(operation, env)
                assert entry["schema"]["task_allowed"] == (
                    operation in set(task.allowed_operations)
                )
                assert entry["valid"] == (operation in validator_valid)
                if operation == "measure":
                    # The agent-facing schema explains every installed
                    # instrument; the operation validator checks one witness.
                    assert set(affordance["invalid_reasons"]) <= set(entry["invalid_reasons"])
                    field = entry["schema"]["fields"][0]
                    assert bool(field["choices"]) == entry["valid"]
                    if not entry["valid"]:
                        assert "measurement_choice_available" in entry["invalid_reasons"]
                        for row in field["choice_status"]:
                            if row["installed"] and row["task_allowed"]:
                                assert set(row["invalid_reasons"]) <= set(entry["invalid_reasons"])
                else:
                    assert entry["invalid_reasons"] == affordance["invalid_reasons"]
                assert not any(
                    reason.startswith("payload_has:") or reason.startswith("payload_bounds:")
                    for reason in entry["invalid_reasons"]
                )
                if operation not in task.allowed_operations:
                    assert "operation_allowed_by_task" in entry["invalid_reasons"]
        finally:
            env.close()


def test_safety_cost_wrapper_preserves_gym_return_shape() -> None:
    env = SafetyCostWrapper(gym.make("ChemWorld", budget=2, seed=0))
    try:
        observation, info = env.reset(seed=0)
        assert isinstance(observation, dict)
        assert info["cost_signal"] == 0.0
        result = env.step({"operation": "heat", "duration_s": 100.0})
        assert len(result) == 5
        _, reward, _, _, step_info = result
        assert isinstance(reward, float)
        assert step_info["cost_signal"] > 0.0
        assert "precondition_failure" in step_info["cost_components"]
        assert step_info["constraint_budget_remaining"] <= 1.0
    finally:
        env.close()


def test_task_safety_limit_reaches_constraint_flags() -> None:
    env = gym.make("ChemWorld", task_id="reaction-safety-constrained", seed=0)
    try:
        _, info = env.reset(seed=0)
        assert info["safety_limit"] == 0.35
        actions = [
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.030},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.0003, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 500.0,
                "duration_s": 2400.0,
                "stirring_speed_rpm": 1100.0,
            },
        ]
        step_info = {}
        for action in actions:
            _, _, _, _, step_info = env.step(action)
        assert step_info["safety_limit"] == 0.35
        assert (
            step_info["constraint_flags"]["unsafe"]
            == step_info["constraint_flags"]["unsafe_by_task_limit"]
        )
    finally:
        env.close()


def test_safety_limit_override_is_explicit_and_validated() -> None:
    env = gym.make(
        "ChemWorld",
        task_id="partition-discovery",
        safety_limit_override=0.2,
        seed=0,
    )
    try:
        _, info = env.reset(seed=0)
        assert info["safety_limit"] == pytest.approx(0.2)
        assert env.unwrapped.task_info()["safety_limit"] == pytest.approx(0.2)
    finally:
        env.close()
    with pytest.raises(ValueError, match="safety_limit_override"):
        gym.make(
            "ChemWorld",
            task_id="partition-discovery",
            safety_limit_override=float("nan"),
        )


def test_operation_validator_enforces_instrument_policy() -> None:
    validator = OperationValidator(
        constitution=make_chemworld_constitution(),
        allowed_operations={"measure"},
        allowed_instruments={"uvvis"},
    )
    state = initial_chemworld_state().replace(volume_L=0.02)
    allowed = validator.validate({"operation": "measure", "instrument": "uvvis"}, state)
    blocked = validator.validate({"operation": "measure", "instrument": "hplc"}, state)
    assert allowed.preconditions["instrument_allowed_by_task"]
    assert not blocked.preconditions["instrument_allowed_by_task"]
    assert "instrument_allowed_by_task" in blocked.invalid_reasons


def test_measurement_affordance_requires_a_feasible_public_instrument() -> None:
    validator = OperationValidator(
        constitution=make_chemworld_constitution(),
        allowed_operations={"measure"},
        allowed_instruments={"hplc", "uvvis", "final_assay"},
    )
    active = initial_chemworld_state().replace(volume_L=0.00010)

    uvvis = validator.validate({"operation": "measure", "instrument": "uvvis"}, active)
    hplc = validator.validate({"operation": "measure", "instrument": "hplc"}, active)

    assert uvvis.is_valid
    assert not hplc.preconditions["measurement_sample_available"]
    assert validator.operation_affordance("measure", active).is_valid

    terminated = active.replace(volume_L=0.00030, terminated=True)
    final_assay = validator.validate(
        {"operation": "measure", "instrument": "final_assay"},
        terminated,
    )
    post_termination_hplc = validator.validate(
        {"operation": "measure", "instrument": "hplc"},
        terminated,
    )

    assert final_assay.is_valid
    assert not post_termination_hplc.preconditions["measure_after_termination_requires_final_assay"]
    assert validator.operation_affordance("measure", terminated).is_valid

    no_sample = active.replace(volume_L=0.00001)
    assert not validator.operation_affordance("measure", no_sample).is_valid
    assert "measure" not in validator.valid_operations(no_sample)


def test_crystallization_affordance_requires_dissolved_target_feed() -> None:
    seed_target_mol = 0.001
    initial = initial_chemworld_state()
    equipment = upsert_equipment_record(
        initial.equipment,
        equipment_id="crystallizer",
        equipment_type="crystallizer",
        attached_vessel_id=initial.vessel_id,
        status="seeded",
        settings={"seed_target_mol": seed_target_mol},
    )
    state = initial.replace(
        volume_L=0.020,
        species_amounts={"P": seed_target_mol},
        equipment=equipment,
    )
    validator = OperationValidator(
        constitution=make_chemworld_constitution(),
        allowed_operations={"cool_crystallize"},
        target_species=("P",),
    )
    action = {
        "operation": "cool_crystallize",
        "target_temperature_K": 278.15,
        "duration_s": 1200.0,
    }

    blocked = validator.validate(action, state)
    assert blocked.preconditions["cool_crystallize_requires_reaction_or_seed"]
    assert not blocked.preconditions["cool_crystallize_target_feed_available"]
    assert "cool_crystallize" not in validator.valid_operations(state)

    feed_available = state.replace(species_amounts={"P": seed_target_mol + 0.001})
    allowed = validator.validate(action, feed_available)
    assert allowed.preconditions["cool_crystallize_target_feed_available"]
    assert allowed.is_valid


def test_process_preconditions_are_stateful() -> None:
    env = ActionMaskWrapper(gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0))
    try:
        _, info = env.reset(seed=0)
        assert "filter_crystals" not in info["valid_operations"]
        assert "seed_crystals" not in info["valid_operations"]
        for action in (
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 385.0,
                "duration_s": 1500.0,
                "stirring_speed_rpm": 720.0,
            },
        ):
            _, _, _, _, info = env.step(action)
        assert "seed_crystals" not in info["valid_operations"]
        blocked_without_assay = env.unwrapped.operation_validator.validate(
            {"operation": "seed_crystals", "seed_mass_g": 0.006},
            env.unwrapped._state,
        )
        assert not blocked_without_assay.preconditions[
            "seed_crystals_requires_current_reaction_assay"
        ]
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "hplc"})
        assert "seed_crystals" in info["valid_operations"]
        _, _, _, _, info = env.step({"operation": "seed_crystals", "seed_mass_g": 0.006})
        assert {"heat", "wait", "seed_crystals", "cool_crystallize", "measure"} <= set(
            info["valid_operations"]
        )
        allowed_after_seed = env.unwrapped.operation_validator.validate(
            {
                "operation": "wait",
                "duration_s": 60.0,
                "stirring_speed_rpm": 600.0,
            },
            env.unwrapped._state,
        )
        assert allowed_after_seed.is_valid
        _, _, _, _, info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1200.0,
            }
        )
        assert {"heat", "wait", "seed_crystals", "cool_crystallize", "measure"} <= set(
            info["valid_operations"]
        )
        assert "filter_crystals" not in info["valid_operations"]
        blocked_filter_without_assay = env.unwrapped.operation_validator.validate(
            {"operation": "filter_crystals"},
            env.unwrapped._state,
        )
        assert not blocked_filter_without_assay.preconditions[
            "filter_crystals_requires_current_slurry_assay"
        ]
        allowed_after_cooling = env.unwrapped.operation_validator.validate(
            {
                "operation": "wait",
                "duration_s": 60.0,
                "stirring_speed_rpm": 600.0,
            },
            env.unwrapped._state,
        )
        assert allowed_after_cooling.is_valid
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "hplc"})
        assert "filter_crystals" in info["valid_operations"]
        assert "heat" in info["valid_operations"]
        _, _, _, _, info = env.step({"operation": "filter_crystals"})
        assert set(info["valid_operations"]) == {"measure", "terminate", "create_container"}
        blocked = env.unwrapped.operation_validator.validate(
            {"operation": "wait", "duration_s": 60.0},
            env.unwrapped._state,
        )
        assert not blocked.preconditions["isolated_crystals_require_assay_or_termination"]
    finally:
        env.close()


def test_nan_observation_wrapper_returns_vector_with_mask() -> None:
    env = NaNObservationWrapper(gym.make("ChemWorld", task_id="reaction-to-assay", seed=0))
    try:
        observation, _ = env.reset(seed=0)
        width = len(env.observation_keys)
        assert observation.shape == (width * 2,)
        assert np.any(observation[:width] == -1.0)
        assert set(np.unique(observation[width:])).issubset({0.0, 1.0})
    finally:
        env.close()


def test_purification_task_reaches_downstream_assay() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=2)
    try:
        obs, info = env.reset(seed=2)
        del obs
        assert "separate_phase" in info["allowed_operations"]
        actions = [
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
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
            {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
            {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
            {"operation": "settle", "duration_s": 420.0},
            {"operation": "separate_phase", "target_phase": "organic"},
            {"operation": "wash", "wash_volume_L": 0.008},
            {"operation": "dry"},
            {"operation": "concentrate", "duration_s": 600.0},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
        final_info = {}
        observation = {}
        for action in actions:
            observation, _, terminated, _, final_info = env.step(action)
        assert terminated
        assert final_info["leaderboard_score"] is not None
        assert float(observation["purity"][0]) >= 0.0
        assert float(observation["recovery"][0]) >= 0.0
        assert final_info["processed_estimate"]["process_mass_balance_error"] >= 0.0
    finally:
        env.close()


def test_purification_recipe_accepts_user_facing_aliases() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=1)
    try:
        env.reset(seed=1)
        actions = [
            {"operation": "add_solvent", "volume_L": 0.03, "solvent": 1},
            {"operation": "add_reagent", "amount_mol": 0.012},
            {"operation": "add_catalyst", "catalyst": 2, "amount_mol": 0.0004},
            {
                "operation": "heat",
                "temperature_K": 350.0,
                "duration_s": 1200.0,
                "stirring_rpm": 800.0,
            },
            {"operation": "quench"},
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
            {"operation": "add_extractant", "volume_L": 0.02, "extractant": "organic"},
            {"operation": "mix", "duration_s": 120.0, "stirring_rpm": 600.0},
            {"operation": "settle", "duration_s": 300.0},
            {"operation": "separate_phase", "phase": "organic"},
            {"operation": "wash", "volume_L": 0.01},
            {"operation": "dry"},
            {
                "operation": "concentrate",
                "duration_s": 300.0,
            },
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
        final_info = {}
        for action in actions:
            _, _, terminated, _, final_info = env.step(action)
            assert not final_info["constraint_flags"]["precondition_failed"]
        assert terminated
        assert final_info["leaderboard_score"] is not None
    finally:
        env.close()


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


def _nested_detection_limit_species(payload: object) -> set[str]:
    if isinstance(payload, dict):
        values = set()
        limits = payload.get("detection_limits_mol_L")
        if isinstance(limits, dict):
            values.update(str(key) for key in limits)
        for value in payload.values():
            values.update(_nested_detection_limit_species(value))
        return values
    if isinstance(payload, list | tuple):
        values: set[str] = set()
        for item in payload:
            values.update(_nested_detection_limit_species(item))
        return values
    return set()


def _run_scripted_task(task_id: str) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    task = get_task(task_id)
    env = gym.make("ChemWorld", task_id=task_id, seed=0)
    try:
        observation, info = env.reset(seed=0)
        agent = ScriptedChemistryAgent()
        agent.reset(info, seed=0)
        history: list[HistoryRecord] = []
        step_info: dict[str, object] = {}
        assay_observation: dict[str, np.ndarray] | None = None
        assay_info: dict[str, object] | None = None
        for _ in range(task.budget):
            action = agent.act(history)
            observation, reward, terminated, truncated, step_info = env.step(action)
            if action.get("operation") == "measure" and action.get("instrument") == "final_assay":
                assay_observation = observation
                assay_info = step_info
            history.append(
                HistoryRecord(
                    step=len(history) + 1,
                    action=action,
                    observation=observation,
                    reward=reward,
                    info=step_info,
                )
            )
            if terminated or truncated:
                break
        invalid_records = [
            record
            for record in history
            if record.info.get("constraint_flags", {}).get("precondition_failed", False)
        ]
        assert not invalid_records, [
            {
                "step": record.step,
                "action": record.action,
                "invalid_reasons": record.info.get("invalid_reasons", []),
                "preconditions": record.info.get("preconditions", {}),
            }
            for record in invalid_records
        ]
        return assay_observation or observation, assay_info or step_info
    finally:
        env.close()


def test_year2_process_tasks_reach_assay_with_scripted_agent() -> None:
    crystallization_obs, crystallization_info = _run_scripted_task("reaction-to-crystallization")
    assert crystallization_info["leaderboard_score"] is not None
    assert float(crystallization_obs["crystal_yield"][0]) >= 0.0
    assert float(crystallization_obs["crystal_purity"][0]) >= 0.0

    distillation_obs, distillation_info = _run_scripted_task("reaction-to-distillation")
    assert distillation_info["leaderboard_score"] is not None
    assert float(distillation_obs["distillate_purity"][0]) >= 0.0
    assert float(distillation_obs["distillate_recovery"][0]) >= 0.0


def test_flow_and_electrochemistry_campaigns_produce_process_assays() -> None:
    flow_obs, _ = _run_scripted_task("flow-reaction-optimization")
    electro_obs, _ = _run_scripted_task("electrochemical-conversion")
    assert float(flow_obs["flow_conversion"][0]) >= 0.0
    assert float(electro_obs["electrochemical_selectivity"][0]) >= 0.0
    assert float(electro_obs["energy_efficiency"][0]) >= 0.0


def test_electrolysis_public_info_excludes_unmeasured_electrochemical_truth() -> None:
    env = gym.make("ChemWorld", task_id="electrochemical-conversion", seed=0)
    try:
        env.reset()
        sequence = [
            {"operation": "add_solvent", "volume_L": 0.026, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {
                "operation": "set_potential",
                "potential_V": 1.15,
                "current_mA": 75.0,
                "electrolyte_profile": 1,
            },
            {"operation": "electrolyze", "duration_s": 1800.0},
        ]
        info = {}
        for action in sequence:
            _, _, _, _, info = env.step(action)
        summary = info["state_delta_summary"]
        assert summary["delta_time_s"] > 0.0
        assert summary["delta_cost"] > 0.0
        assert {
            "charge_C",
            "faradaic_charge_C",
            "overpotential_V",
            "equilibrium_potential_V",
            "measured_potential_V",
            "interfacial_potential_V",
            "uncompensated_voltage_drop_V",
            "ohmic_loss_J",
            "total_resistance_ohm",
            "faradaic_efficiency",
        }.isdisjoint(summary)
    finally:
        env.close()
