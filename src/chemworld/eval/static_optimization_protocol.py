"""Shared protocol semantics for fixed-world scientific optimization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.materials import normalize_static_material_information_config
from chemworld.physchem.electrochemical_task_contract import (
    ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE,
    normalize_electrochemical_workflow_mode,
)
from chemworld.world.crystallization_material_family import (
    HISTORICAL_CRYSTALLIZATION_MATERIAL_FAMILY,
    normalize_crystallization_material_family,
)
from chemworld.world.electrochemical_material_family import (
    HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY,
    normalize_electrochemical_material_family,
)
from chemworld.world.scoring import (
    CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
    DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2,
    ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2,
    FLOW_S0_BALANCED_PROCESS_V1,
    PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
    PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
    TASK_DERIVED_SCORING_CONTRACT,
)

STATIC_WORLD_MODE = "static_for_entire_campaign"
ELECTROCHEMICAL_TASK_ID = "electrochemical-conversion"
CRYSTALLIZATION_TASK_ID = "reaction-to-crystallization"
DISTILLATION_TASK_ID = "reaction-to-distillation"
PREDICTIVE_CALL_INTEGRATED = "integrated_final_synthesis"
PREDICTIVE_CALL_SEPARATE = "separate_after_final_commit"
PREDICTIVE_QUERY_HISTORY_LOCAL = (
    "highest_scoring_experiment_with_complete_unseen_one_factor_query_set"
)
PREDICTIVE_QUERY_STANDARDIZED = "standardized_history_independent_anchor_v0.1"


def exploration_experiment_count(protocol: Mapping[str, Any]) -> int:
    """Return the number of complete experiments in the visible S0 campaign."""

    legacy_horizon = protocol.get("horizon")
    campaign = protocol.get("scientific_campaign_budget")
    if isinstance(campaign, Mapping) and "exploration_experiments" in campaign:
        count = _positive_int(
            campaign["exploration_experiments"],
            "scientific_campaign_budget.exploration_experiments",
        )
        if legacy_horizon is not None and _positive_int(legacy_horizon, "horizon") != count:
            raise ValueError("S0 horizon and scientific campaign exploration budget disagree")
        return count
    return _positive_int(legacy_horizon, "horizon")


def static_optimization_workflow_mode(protocol: Mapping[str, Any]) -> str:
    """Return the explicit electrochemical workflow mode for an S0 protocol.

    Non-electrochemical tasks use the single-stage value as an inert executor
    setting. Electrochemical protocols must state their workflow explicitly so
    an omitted field cannot silently revive the historical two-stage recipe.
    """

    tasks = protocol.get("tasks")
    task_ids = {str(item) for item in tasks} if isinstance(tasks, list | tuple) else set()
    executor = protocol.get("executor_contract")
    raw_mode = (
        executor.get("electrochemical_workflow_mode") if isinstance(executor, Mapping) else None
    )
    if ELECTROCHEMICAL_TASK_ID in task_ids and raw_mode is None:
        raise ValueError(
            "electrochemical S0 protocols must explicitly declare "
            "executor_contract.electrochemical_workflow_mode"
        )
    if raw_mode is None:
        return ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
    return normalize_electrochemical_workflow_mode(str(raw_mode))


def static_optimization_material_family_id(protocol: Mapping[str, Any]) -> str:
    """Return the versioned electrochemical material family for one campaign."""

    world_policy = protocol.get("world_policy")
    raw_family = (
        world_policy.get("electrochemical_material_family_id")
        if isinstance(world_policy, Mapping)
        else None
    )
    tasks = protocol.get("tasks")
    task_ids = {str(item) for item in tasks} if isinstance(tasks, list | tuple) else set()
    cell = protocol.get("cell")
    if not task_ids and isinstance(cell, Mapping) and cell.get("task_id"):
        task_ids = {str(cell["task_id"])}
    if ELECTROCHEMICAL_TASK_ID in task_ids and raw_family is None:
        raise ValueError(
            "electrochemical S0 protocols must explicitly declare "
            "world_policy.electrochemical_material_family_id; legacy is reserved "
            "for historical replay"
        )
    family_id = normalize_electrochemical_material_family(raw_family)
    if family_id != HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY and task_ids != {
        ELECTROCHEMICAL_TASK_ID
    }:
        raise ValueError(
            "non-legacy electrochemical material families require exactly the "
            "electrochemical-conversion task"
        )
    return family_id


def static_optimization_crystallization_material_family_id(
    protocol: Mapping[str, Any],
) -> str:
    """Return the versioned reaction-to-crystallization material family."""

    world_policy = protocol.get("world_policy")
    raw_family = (
        world_policy.get("crystallization_material_family_id")
        if isinstance(world_policy, Mapping)
        else None
    )
    tasks = protocol.get("tasks")
    task_ids = {str(item) for item in tasks} if isinstance(tasks, list | tuple) else set()
    cell = protocol.get("cell")
    if not task_ids and isinstance(cell, Mapping) and cell.get("task_id"):
        task_ids = {str(cell["task_id"])}
    family_id = normalize_crystallization_material_family(raw_family)
    if family_id != HISTORICAL_CRYSTALLIZATION_MATERIAL_FAMILY and task_ids != {
        CRYSTALLIZATION_TASK_ID
    }:
        raise ValueError(
            "non-legacy crystallization material families require exactly the "
            "reaction-to-crystallization task"
        )
    return family_id


def static_optimization_scoring_contract_id(protocol: Mapping[str, Any]) -> str:
    """Return the explicit score law for one static optimization campaign."""

    reward = protocol.get("reward_contract")
    raw_contract = reward.get("scoring_contract_id") if isinstance(reward, Mapping) else None
    tasks = protocol.get("tasks")
    task_ids = {str(item) for item in tasks} if isinstance(tasks, list | tuple) else set()
    cell = protocol.get("cell")
    if not task_ids and isinstance(cell, Mapping) and cell.get("task_id"):
        task_ids = {str(cell["task_id"])}
    electrochemical_family_id = static_optimization_material_family_id(protocol)
    crystallization_family_id = static_optimization_crystallization_material_family_id(protocol)
    nonlegacy_electrochemical = (
        ELECTROCHEMICAL_TASK_ID in task_ids
        and electrochemical_family_id != HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY
    )
    nonlegacy_crystallization = (
        CRYSTALLIZATION_TASK_ID in task_ids
        and crystallization_family_id != HISTORICAL_CRYSTALLIZATION_MATERIAL_FAMILY
    )
    if (nonlegacy_electrochemical or nonlegacy_crystallization) and raw_contract is None:
        raise ValueError(
            "non-legacy S0 material protocols must explicitly declare "
            "reward_contract.scoring_contract_id"
        )
    contract_id = TASK_DERIVED_SCORING_CONTRACT if raw_contract is None else str(raw_contract)
    allowed = {
        TASK_DERIVED_SCORING_CONTRACT,
        ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2,
        CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1,
        DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2,
        PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
        PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
        FLOW_S0_BALANCED_PROCESS_V1,
    }
    if contract_id not in allowed:
        raise ValueError(f"unsupported S0 scoring contract ID: {contract_id}")
    if (
        ELECTROCHEMICAL_TASK_ID in task_ids
        and electrochemical_family_id == HISTORICAL_ELECTROCHEMICAL_MATERIAL_FAMILY
        and contract_id != TASK_DERIVED_SCORING_CONTRACT
    ):
        raise ValueError(
            "historical electrochemical replay requires the historical task-derived score"
        )
    if (
        CRYSTALLIZATION_TASK_ID in task_ids
        and crystallization_family_id == HISTORICAL_CRYSTALLIZATION_MATERIAL_FAMILY
        and contract_id != TASK_DERIVED_SCORING_CONTRACT
    ):
        raise ValueError(
            "historical crystallization replay requires the historical task-derived score"
        )
    if contract_id == ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2 and not nonlegacy_electrochemical:
        raise ValueError("electrochemical S0 v2 scoring requires its non-legacy material family")
    if contract_id == CRYSTALLIZATION_S0_BALANCED_PRODUCT_V1 and not nonlegacy_crystallization:
        raise ValueError("crystallization S0 v1 scoring requires its non-legacy material family")
    if contract_id == DISTILLATION_S0_BALANCED_AUDIT_SAFETY_V2 and task_ids != {
        DISTILLATION_TASK_ID
    }:
        raise ValueError(
            "reaction-distillation S0 v2 scoring requires exactly the reaction-to-distillation task"
        )
    if (
        contract_id
        in {
            PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
            PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
        }
        and task_ids != {"partition-discovery"}
    ):
        raise ValueError("partition S0 scoring requires exactly the partition-discovery task")
    if contract_id == FLOW_S0_BALANCED_PROCESS_V1 and task_ids != {"flow-reaction-optimization"}:
        raise ValueError(
            "continuous-flow S0 v1 scoring requires exactly the flow-reaction-optimization task"
        )
    return contract_id


def static_optimization_predictive_call_policy(protocol: Mapping[str, Any]) -> str:
    """Return whether exact predictive queries share the recommendation call."""

    world_understanding = protocol.get("world_understanding")
    if not isinstance(world_understanding, Mapping) or not bool(
        world_understanding.get("predictive_score_enabled", False)
    ):
        return "disabled"
    predictive = world_understanding.get("predictive_validation")
    if not isinstance(predictive, Mapping):
        raise ValueError("predictive world understanding lacks a validation contract")
    policy = str(predictive.get("call_policy", PREDICTIVE_CALL_INTEGRATED))
    if policy not in {PREDICTIVE_CALL_INTEGRATED, PREDICTIVE_CALL_SEPARATE}:
        raise ValueError("predictive validation call_policy is unsupported")
    return policy


def static_optimization_predictive_query_policy(protocol: Mapping[str, Any]) -> str:
    """Return the frozen rule used to choose held-out predictive queries."""

    world_understanding = protocol.get("world_understanding")
    if not isinstance(world_understanding, Mapping) or not bool(
        world_understanding.get("predictive_score_enabled", False)
    ):
        return "disabled"
    predictive = world_understanding.get("predictive_validation")
    if not isinstance(predictive, Mapping):
        raise ValueError("predictive world understanding lacks a validation contract")
    return str(predictive.get("reference_selection_policy", PREDICTIVE_QUERY_HISTORY_LOCAL))


def validate_static_optimization_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject ambiguous or non-static protocols before an S0 run starts."""

    tasks = protocol.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(item, str) and item for item in tasks)
    ):
        raise ValueError("S0 protocol tasks must be a non-empty string list")
    world_policy = protocol.get("world_policy")
    if not isinstance(world_policy, Mapping):
        raise ValueError("S0 protocol lacks world_policy")
    if world_policy.get("mode") != STATIC_WORLD_MODE:
        raise ValueError("S0 runner accepts only a static world policy")
    if list(world_policy.get("interventions", [])):
        raise ValueError("S0 runner rejects world interventions")
    if list(world_policy.get("phase_changes", [])):
        raise ValueError("S0 runner rejects phase changes")
    if world_policy.get("hidden_world_fields_in_public_context") is not False:
        raise ValueError("S0 protocol must hide private world fields")
    validate_development_seed_policy(protocol)
    if "electrochemical_semantic_profile_id" in world_policy:
        raise ValueError(
            "electrochemical semantic profiles were retired; bind a versioned "
            "electrochemical material family instead"
        )
    if protocol.get("shared_calibration_prefix") is not None:
        raise ValueError(
            "shared calibration prefixes were retired; every S0 exploration "
            "experiment must be designed by the evaluated method"
        )
    exploration_experiment_count(protocol)
    campaign = protocol.get("scientific_campaign_budget")
    static_optimization_workflow_mode(protocol)
    material_family_id = static_optimization_material_family_id(protocol)
    crystallization_material_family_id = static_optimization_crystallization_material_family_id(
        protocol
    )
    static_optimization_scoring_contract_id(protocol)
    normalize_static_material_information_config(
        protocol.get("material_information"),
        task_ids=tasks,
        material_family_id=(
            crystallization_material_family_id
            if set(tasks) == {CRYSTALLIZATION_TASK_ID}
            else material_family_id
        ),
    )
    predictive_call_policy = static_optimization_predictive_call_policy(protocol)
    predictive_query_policy = static_optimization_predictive_query_policy(protocol)
    world_understanding = protocol.get("world_understanding")
    if isinstance(world_understanding, Mapping) and bool(world_understanding.get("enabled", False)):
        reference_path = world_understanding.get("reference_path")
        reference_configured = isinstance(reference_path, str) and bool(reference_path.strip())
        declared_scoring = world_understanding.get("declared_scoring_enabled")
        if declared_scoring is True and not reference_configured:
            raise ValueError(
                "Declared world-understanding scoring requires a frozen reference_path"
            )
        if bool(protocol.get("formal_result", False)) and not reference_configured:
            if declared_scoring is not False:
                raise ValueError(
                    "formal S0 without a frozen Declared reference must explicitly set "
                    "world_understanding.declared_scoring_enabled=false"
                )
            if "world_understanding_structural_edge_f1" in set(
                protocol.get("secondary_metrics", [])
            ):
                raise ValueError(
                    "formal S0 cannot report Declared structural F1 without a frozen reference"
                )

    final_synthesis = protocol.get("final_synthesis")
    if isinstance(campaign, Mapping) and "final_synthesis_after_exploration" in campaign:
        expected = bool(campaign["final_synthesis_after_exploration"])
        actual = bool(
            final_synthesis.get("enabled", False) if isinstance(final_synthesis, Mapping) else False
        )
        if expected != actual:
            raise ValueError("scientific campaign and final synthesis contracts disagree")
    if predictive_call_policy == PREDICTIVE_CALL_INTEGRATED:
        if predictive_query_policy != PREDICTIVE_QUERY_HISTORY_LOCAL:
            raise ValueError(
                "integrated predictive validation requires the history-local query policy"
            )
    elif predictive_call_policy == PREDICTIVE_CALL_SEPARATE:
        predictive = protocol["world_understanding"]["predictive_validation"]
        if set(tasks) != {ELECTROCHEMICAL_TASK_ID}:
            raise ValueError("separate standardized predictive validation is electrochemical-only")
        if (
            static_optimization_workflow_mode(protocol)
            != ELECTROCHEMICAL_WORKFLOW_STATIC_SINGLE_STAGE
        ):
            raise ValueError(
                "separate standardized predictive validation requires single-stage electrochemistry"
            )
        if predictive_query_policy != PREDICTIVE_QUERY_STANDARDIZED:
            raise ValueError(
                "separate predictive validation requires the standardized anchor policy"
            )
        if int(predictive.get("additional_model_calls", -1)) != 1:
            raise ValueError("separate predictive validation requires one additional call")
        if predictive.get("recommendation_committed_before_query_visibility") is not True:
            raise ValueError("separate predictive validation must commit the recommendation")
        if predictive.get("prediction_call_can_modify_recommendation") is not False:
            raise ValueError("separate predictive call must not modify the recommendation")
        if not isinstance(final_synthesis, Mapping) or not bool(
            final_synthesis.get("enabled", False)
        ):
            raise ValueError("separate predictive validation requires final synthesis")


def validate_development_seed_policy(
    protocol: Mapping[str, Any],
    *,
    algorithm_seed: int | None = None,
) -> None:
    """Fail closed when a development protocol forbids multi-seed execution."""

    policy = protocol.get("development_seed_policy")
    if policy is None:
        return
    if not isinstance(policy, Mapping):
        raise ValueError("development_seed_policy must be an object")
    if policy.get("multi_seed_execution_allowed") is not False:
        raise ValueError(
            "development_seed_policy must explicitly keep multi-seed execution disabled"
        )
    world_seeds = policy.get("world_seeds")
    algorithm_seeds = policy.get("algorithm_seeds")
    if (
        not isinstance(world_seeds, list)
        or len(world_seeds) != 1
        or isinstance(world_seeds[0], bool)
        or not isinstance(world_seeds[0], int)
    ):
        raise ValueError("single-seed development requires exactly one integer world seed")
    if (
        not isinstance(algorithm_seeds, list)
        or len(algorithm_seeds) != 1
        or isinstance(algorithm_seeds[0], bool)
        or not isinstance(algorithm_seeds[0], int)
    ):
        raise ValueError("single-seed development requires exactly one integer algorithm seed")
    world_policy = protocol.get("world_policy")
    if not isinstance(world_policy, Mapping):
        raise ValueError("S0 protocol lacks world_policy")
    if int(world_policy.get("world_seed", world_seeds[0])) != world_seeds[0]:
        raise ValueError("world seed is outside the single-seed development policy")
    configured_algorithm_seeds = protocol.get("algorithm_seeds")
    if configured_algorithm_seeds is not None and configured_algorithm_seeds != algorithm_seeds:
        raise ValueError(
            "configured algorithm seeds differ from the single-seed development policy"
        )
    if algorithm_seed is not None and int(algorithm_seed) != algorithm_seeds[0]:
        raise ValueError("algorithm seed is outside the single-seed development policy")


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


__all__ = [
    "CRYSTALLIZATION_TASK_ID",
    "DISTILLATION_TASK_ID",
    "ELECTROCHEMICAL_TASK_ID",
    "PREDICTIVE_CALL_INTEGRATED",
    "PREDICTIVE_CALL_SEPARATE",
    "PREDICTIVE_QUERY_HISTORY_LOCAL",
    "PREDICTIVE_QUERY_STANDARDIZED",
    "STATIC_WORLD_MODE",
    "exploration_experiment_count",
    "static_optimization_crystallization_material_family_id",
    "static_optimization_material_family_id",
    "static_optimization_predictive_call_policy",
    "static_optimization_predictive_query_policy",
    "static_optimization_scoring_contract_id",
    "static_optimization_workflow_mode",
    "validate_development_seed_policy",
    "validate_static_optimization_protocol",
]
