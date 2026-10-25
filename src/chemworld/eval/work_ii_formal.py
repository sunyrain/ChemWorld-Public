"""Outcome-blind manifest construction for the Work II formal matrix."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

from chemworld.eval.mechanism_adaptation_execution import load_protocol_object
from chemworld.eval.mechanism_gate_decision import gate_a_execution_contract_binding
from chemworld.eval.mechanism_release import STRUCTURAL_RECEIPT_VERSION
from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.eval.work_ii_ae_formal_cohort import (
    FORMAL_DESIGN_VERSION,
    LEGACY_FORMAL_DESIGN_VERSION,
    load_ae_formal_cohort,
    validate_ae_public_cells,
)
from chemworld.eval.work_ii_c2_admission import (
    C2_LOCI,
    C2_OUTCOME_BLIND_SELECTION_VERSION,
    C2_REQUIRED_CHECKPOINTS,
    C2_REQUIRED_ROUNDS,
    C2_REQUIRED_TASK_COUNTS,
    C2_TASK_ADMISSION_RECEIPT_VERSION,
    build_c2_admission_report,
    c2_admission_sha256,
    c2_outcome_blind_selection_sha256,
    c2_task_admission_receipt_sha256,
    validate_c2_admission_report,
)
from chemworld.eval.work_ii_formal_runtime import validate_formal_runtime_manifest

FORMAL_PREFLIGHT_VERSION = "chemworld-work-ii-formal-matrix-preflight-0.1"
SUPPORTED_FORMAL_ANALYSIS_VERSIONS = {
    LEGACY_FORMAL_DESIGN_VERSION: "chemworld-work-ii-analysis-plan-0.3",
    FORMAL_DESIGN_VERSION: "chemworld-work-ii-analysis-plan-0.3",
}
FORMAL_CELL_VERSION = "chemworld-work-ii-formal-cell-0.1"
FORMAL_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")
FORMAL_SNAPSHOT_STAGES = (
    "pre_evidence",
    "after_experiment_2",
    "after_experiment_4",
    "after_experiment_6",
    "final",
)
FORMAL_CHECKPOINT_EXPERIMENTS = (0, 2, 4, 6, 8)
FORMAL_METHOD_CHECKPOINT_EXPERIMENTS = (2, 4, 6, 8)
FORMAL_COMPLETE_EXPERIMENTS_PER_CELL = 8
FORMAL_BELIEF_CHECKPOINTS_PER_CELL = len(FORMAL_SNAPSHOT_STAGES)
FORMAL_C2_LOCI = ("A_E", *C2_LOCI)
FORMAL_C2_LOCUS_CONTRACT: dict[str, dict[str, Any]] = {
    "A_E": {
        "task_count": 5,
        "complete_experiments_per_cell": 8,
        "checkpoint_complete_experiments": FORMAL_CHECKPOINT_EXPERIMENTS,
        "snapshot_stages": FORMAL_SNAPSHOT_STAGES,
    },
    "A_P": {
        "task_count": C2_REQUIRED_TASK_COUNTS["A_P"],
        "complete_experiments_per_cell": C2_REQUIRED_ROUNDS["A_P"],
        "checkpoint_complete_experiments": C2_REQUIRED_CHECKPOINTS["A_P"],
        "snapshot_stages": (
            "pre_evidence",
            "after_experiment_2",
            "after_experiment_4",
            "after_experiment_7",
            "final",
        ),
    },
    "A_S": {
        "task_count": C2_REQUIRED_TASK_COUNTS["A_S"],
        "complete_experiments_per_cell": C2_REQUIRED_ROUNDS["A_S"],
        "checkpoint_complete_experiments": C2_REQUIRED_CHECKPOINTS["A_S"],
        "snapshot_stages": (
            "pre_evidence",
            "after_experiment_3",
            "after_experiment_6",
            "after_experiment_9",
            "final",
        ),
    },
}
FORMAL_EXPECTED_TOTALS = {
    "tasks": 9,
    "independent_task_world_clusters": 45,
    "participant_cells": 135,
    "provider_sessions": 135,
    "accepted_participant_model_calls_minimum": 135,
    "accepted_participant_model_calls_maximum": 270,
    "provider_attempts_initial_planned": 135,
    "provider_attempts_hard_cap": 270,
    "complete_experiments": 1260,
    "belief_checkpoints": 675,
    "participant_final_recommendations": 135,
    "blind_validation_targets": 270,
    "blind_validation_executions": 810,
}
FORMAL_RECEIPT_VERSION = "chemworld-work-ii-formal-cell-receipt-0.1"
FORMAL_STORE_AUDIT_VERSION = "chemworld-work-ii-formal-store-audit-0.1"
FORMAL_TERMINAL_STATES = frozenset({"completed", "right_censored", "failed"})
DEFAULT_C2_ADMISSION_PLAN = Path(
    "configs/benchmark/work_ii_c2_admission_manifest_v0.1.json"
)

EXPECTED_PARTICIPANT_EXECUTION_CONTRACT: dict[str, Any] = {
    "execution_unit": "task_x_prior_arm_x_world_seed_cell",
    "session_scope": "campaign",
    "accepted_scientific_codex_threads_per_cell": 1,
    "accepted_participant_provider_sessions_per_cell": 1,
    "accepted_participant_model_calls_per_cell_minimum": 1,
    "accepted_participant_model_calls_per_cell_maximum": 2,
    "accepted_turn_continuation_limit": 1,
    "provider_process_attempt_limit_per_cell_attempt": 3,
    "same_session_bindings": [
        "operation_tool_loop",
        "complete_experiments",
        "belief_checkpoints",
        "final_recommendation",
        "provider_receipt",
    ],
    "interaction_contract": {
        "decision_scope": "one_operation_after_each_public_outcome",
        "tool_transport": "host_owned_stdio_mcp",
        "participant_owns_operation_selection": True,
        "host_roles": [
            "schema_validation",
            "transaction_execution",
            "campaign_resource_accounting",
            "hidden_world_execution",
        ],
        "automatic_action_repair": False,
        "automatic_closeout": False,
        "checkpoint_provider_calls": 0,
    },
    "context_and_memory_contract": {
        "context_scope": (
            "one_complete_provider_process_transcript_plus_participant_visible_public_outcomes"
        ),
        "checkpoint_state_schema": "typed_work_ii_belief_snapshot",
        "checkpoint_top_level_fields": [
            "prior_assessment",
            "predictions",
            "law_summary",
            "evidence_ids",
            "next_experiment_intent",
            "overall_confidence",
        ],
        "persistent_workspace_notes_allowed": False,
        "free_text_persistent_memory_allowed": False,
        "bounded_schema_rationale_fields_allowed": True,
        "private_chain_of_thought_retained": False,
    },
    "sampling_contract": {
        "reasoning_effort": "medium",
        "temperature": None,
        "temperature_semantics": "not_exposed_or_set_by_the_codex_harness",
    },
    "timeout_contract_s": {"request": 1200.0, "finalization": 600.0},
    "lifecycle_contract": {
        "explicit_terminate_required_before_final_assay": True,
        "final_assay_closes_completed_experiment": True,
        "explicit_discard_closes_failed_or_abandoned_batch": True,
        "budget_exhaustion_right_censors_open_experiment": True,
        "all_planned_batches_share_one_campaign_resource_card": True,
    },
    "failure_and_retry_contract": {
        "missing_infrastructure_only_resume": True,
        "scientific_or_method_failure_retained": True,
        "persisted_scientific_trajectory_forbids_replacement": True,
        "result_direction_retry_forbidden": True,
    },
    "separate_reported_denominators": [
        "host_provider_process_attempt",
        "provider_session",
        "mcp_tool_call",
        "operation_attempt",
        "committed_operation",
        "complete_experiment",
        "participant_cell",
        "blind_evaluator_execution",
    ],
}

# The v0.1 design is retained as historical evidence and remains readable by
# the validator.  Only the current v0.2 design may authorize the bounded
# same-thread continuation method.
LEGACY_EXPECTED_PARTICIPANT_EXECUTION_CONTRACT = deepcopy(
    EXPECTED_PARTICIPANT_EXECUTION_CONTRACT
)
for _field in (
    "accepted_scientific_codex_threads_per_cell",
    "accepted_participant_model_calls_per_cell_minimum",
    "accepted_participant_model_calls_per_cell_maximum",
    "accepted_turn_continuation_limit",
    "provider_process_attempt_limit_per_cell_attempt",
):
    LEGACY_EXPECTED_PARTICIPANT_EXECUTION_CONTRACT.pop(_field)
LEGACY_EXPECTED_PARTICIPANT_EXECUTION_CONTRACT.update(
    {
        "accepted_scientific_codex_processes_per_cell": 1,
        "accepted_participant_model_calls_per_cell": 1,
    }
)

EXPECTED_REFERENCE_POLICY_CONTRACT: dict[str, Any] = {
    "role": "calibration_or_mechanism_reference_only",
    "participant_formal_denominator": False,
    "participant_information_arms": [
        "opaque_id_only",
        "aligned_property_aware",
        "misindexed_property_matched",
    ],
    "required_semantics_free_calibration_pair": {
        "policy_identity_matched": True,
        "information_conditions": ["id_only", "public_property_vector"],
        "world_and_resource_contract_matched": True,
    },
    "calibration_execution_in_75_participant_cells": False,
    "classical_policy_results_required_for_primary_h3": False,
    "classical_policy_results_required_for_resource_calibrated_interpretation": True,
    "outcome_based_method_arm_deletion_forbidden": True,
}

EXPECTED_METHOD_QUALIFICATION_CONTRACT: dict[str, Any] = {
    "qualification_task_id": "electrochemical-conversion",
    "qualification_world_cohort": "development_and_qualification",
    "qualification_world_seed": 0,
    "qualified_prior_arms": [
        "opaque",
        "aligned_nominal",
        "misindexed_nominal",
    ],
    "qualification_cell_count": 3,
    "complete_experiments_per_cell": FORMAL_COMPLETE_EXPERIMENTS_PER_CELL,
    "belief_checkpoints_per_cell": FORMAL_BELIEF_CHECKPOINTS_PER_CELL,
    "minimum_unique_recipes_per_cell": 6,
    "maximum_participant_selected_exact_repeats_per_cell": 2,
    "resource_calibration_required": True,
    "resource_calibration_status": "pending_w2_26",
    "accepted_scientific_codex_threads_per_cell": 1,
    "accepted_participant_provider_sessions_per_cell": 1,
    "accepted_participant_model_calls_per_cell_minimum": 1,
    "accepted_participant_model_calls_per_cell_maximum": 2,
    "accepted_turn_continuation_limit": 1,
    "provider_process_attempt_limit_per_cell_attempt": 3,
    "maximum_infrastructure_resume_attempts_per_cell": 1,
    "maximum_total_provider_attempts": 6,
    "triplet_failure_semantics": "finish_all_three_arms_then_fail_qualification",
    "qualification_outcome_use": "platform_and_method_acceptance_only",
    "scientific_outcome_selection_forbidden": True,
    "formal_participant_outcomes_before_authorization": 0,
    "real_provider_execution_required": True,
    "same_provider_and_method_as_formal": True,
    "all_cells_must_pass": True,
    "exact_replay_required": True,
    "execution_audit_required": True,
    "currency_approval_required": True,
    "qualification_worlds_excluded_from_formal": True,
}

LEGACY_EXPECTED_METHOD_QUALIFICATION_CONTRACT = deepcopy(
    EXPECTED_METHOD_QUALIFICATION_CONTRACT
)
for _field in (
    "accepted_scientific_codex_threads_per_cell",
    "accepted_participant_model_calls_per_cell_minimum",
    "accepted_participant_model_calls_per_cell_maximum",
    "accepted_turn_continuation_limit",
    "provider_process_attempt_limit_per_cell_attempt",
):
    LEGACY_EXPECTED_METHOD_QUALIFICATION_CONTRACT.pop(_field)
LEGACY_EXPECTED_METHOD_QUALIFICATION_CONTRACT.update(
    {
        "accepted_scientific_codex_processes_per_cell": 1,
        "accepted_participant_model_calls_per_cell": 1,
    }
)

EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT: dict[str, Any] = {
    "evaluation_unit": "participant_cell_x_final_law_summary_x_registered_query_metric",
    "query_source": "same_four_evaluator_held_out_queries_scored_at_belief_checkpoints",
    "truth_source": "shared_task_world_evaluator_truth_pack",
    "required_summary_schema": "chemworld-work-ii-law-summary-0.1",
    "required_query_coverage": "exact_registered_query_metric_set",
    "normalized_error": (
        "mean_over_registered_query_metric_pairs_abs_law_prediction_minus_truth_over_metric_scale"
    ),
    "compression_stability": (
        "law_summary_normalized_error_minus_effective_final_checkpoint_error"
    ),
    "prediction_consistency": (
        "mean_normalized_absolute_difference_between_law_summary_and_final_checkpoint_predictions"
    ),
    "pre_to_summary_improvement": (
        "effective_pre_evidence_error_minus_law_summary_normalized_error"
    ),
    "evaluator_provider_calls": 0,
    "participant_feedback_allowed": False,
    "binary_public_validity_threshold": None,
    "public_interpretation": (
        "descriptive_executability_error_compression_and_consistency_only"
    ),
    "private_transfer_required_for_reusable_law_claim": True,
}

EXPECTED_PRIVATE_CONFIRMATION_CONTRACT: dict[str, Any] = {
    "identity_source": "external_ignored_private_seal",
    "identity_commitment_hash": "canonical_json_sha256_of_complete_seal",
    "task_count": 5,
    "worlds_per_task": 5,
    "prior_arms": ["opaque", "aligned_nominal", "misindexed_nominal"],
    "participant_cell_count": 75,
    "same_participant_method_campaign_resources_and_metrics_as_public": True,
    "private_participant_receives_public_outcomes_or_analysis": False,
    "unseal_requires": [
        "public_75_cell_matrix_terminal",
        "public_confirmatory_analysis_completed_and_hash_bound",
        "public_analysis_and_private_runner_source_frozen",
        "separate_private_currency_ceiling_and_user_execution_signoff",
        "private_seal_commitment_matches_preregistered_hash",
    ],
    "one_shot_policy": {
        "result_direction_rerun_forbidden": True,
        "accepted_or_failed_private_cell_replacement_forbidden": True,
        "missing_infrastructure_only_resume_allowed": True,
        "maximum_infrastructure_resume_attempts_per_cell": 1,
        "all_failures_and_unstarted_cells_retained": True,
    },
    "private_results_may_not_change": [
        "public_hypotheses",
        "public_estimands",
        "analysis_thresholds",
        "participant_method",
        "task_roster",
        "exclusion_or_missingness_rules",
    ],
    "transfer_reporting": (
        "same_public_prediction_law_summary_blind_action_and_failure_metrics_"
        "with_private_world_split_label"
    ),
}

FORMAL_BLOCKING_REQUIREMENTS = (
    "formal currency ceiling is not yet approved",
    "current design and analysis plan explicitly forbid formal execution",
    "current persistent-session method lacks its final qualification receipt",
    "formal execution package lacks its final user authorization receipt",
)


def _validate_environment_binding(root: Path, design: Mapping[str, Any]) -> list[str]:
    """Require the formal design to resolve the exact current Gate A evidence."""

    errors: list[str] = []
    environment = design.get("environment_binding")
    environment = environment if isinstance(environment, Mapping) else {}
    if environment.get("gate_a_evidence_current_required") is not True:
        errors.append("formal design does not require current Gate A evidence")
    for field in ("protocol", "gate_a_plan", "public_decision"):
        relative = environment.get(field)
        if not isinstance(relative, str) or not relative:
            errors.append(f"formal design environment binding lacks {field}")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"formal design environment binding escapes repository: {field}")
            continue
        if not path.is_file():
            errors.append(f"formal design environment binding is missing: {field}")

    current_path = root / "configs/current.json"
    if not current_path.is_file():
        errors.append("current artifact registry is missing")
        return errors
    current = _load_object(current_path)
    mechanism = current.get("mechanism_adaptation")
    mechanism = mechanism if isinstance(mechanism, Mapping) else {}
    if (
        mechanism.get("gate_a_evidence_current") is not True
        or mechanism.get("gate_a_pass") is not True
        or mechanism.get("public_decision_report") != environment.get("public_decision")
    ):
        errors.append("formal design environment does not resolve current passed Gate A evidence")
        return errors

    protocol_path = root / str(environment.get("protocol", ""))
    plan_path = root / str(environment.get("gate_a_plan", ""))
    decision_path = root / str(environment.get("public_decision", ""))
    a2_path = root / str(mechanism.get("a2_structural_receipt", ""))
    a3_path = root / str(mechanism.get("a3_structural_receipt", ""))
    gate_evidence_paths = (protocol_path, plan_path, decision_path, a2_path, a3_path)
    if not all(path.is_file() for path in gate_evidence_paths):
        errors.append("current Gate A runtime-binding evidence is incomplete")
        return errors
    protocol = load_protocol_object(protocol_path)
    plan = _load_object(plan_path)
    decision = _load_object(decision_path)
    a2_receipt = _load_object(a2_path)
    a3_receipt = _load_object(a3_path)
    expected_binding = gate_a_execution_contract_binding(protocol, plan)
    expected_binding_sha256 = expected_binding["binding_sha256"]
    expected_runtime_sha256 = expected_binding["runtime_source_tree_sha256"]
    receipt_rows = (("A2", a2_receipt), ("A3", a3_receipt))
    if any(
        receipt.get("schema_version") != STRUCTURAL_RECEIPT_VERSION
        or receipt.get("execution_contract_binding_sha256") != expected_binding_sha256
        or receipt.get("runtime_source_tree_sha256") != expected_runtime_sha256
        for _, receipt in receipt_rows
    ):
        errors.append("current Gate A certificates do not bind the current runtime semantics")
    if (
        decision.get("a2_structural_receipt_sha256")
        != canonical_json_sha256(a2_receipt)
        or decision.get("a3_structural_receipt_sha256")
        != canonical_json_sha256(a3_receipt)
    ):
        errors.append("current Gate A decision does not bind its structural receipts")
    return errors


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _object(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _string_list(value: object, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list")
    result = [str(item) for item in value]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{label} must be a non-empty unique list")
    return result


def _analysis_design_relation_errors(
    design: Mapping[str, Any],
    analysis: Mapping[str, Any],
    design_relative_path: str,
) -> list[str]:
    """Validate the scientific relations shared by the two frozen inputs.

    The formal manifest binds both complete documents independently.  The
    analysis plan therefore names the current design and states the fields it
    consumes, rather than carrying a second whole-design digest that becomes a
    development-time freshness gate.
    """

    errors: list[str] = []
    design_version = design.get("schema_version")
    design_binding = _object(analysis.get("design_binding"), "analysis.design_binding")
    if design_version == LEGACY_FORMAL_DESIGN_VERSION:
        expected_binding = {
            "path": design_relative_path,
            "sha256": canonical_json_sha256(design),
        }
        if dict(design_binding) != expected_binding:
            errors.append("analysis plan does not bind the current formal design")
    elif dict(design_binding) != {"path": design_relative_path}:
        errors.append("analysis plan does not select the current formal design")

    arms = tuple(_string_list(design.get("prior_arms"), "design.prior_arms"))
    if arms != FORMAL_ARMS:
        errors.append("formal prior-arm order differs from the frozen three-arm contract")

    raw_tasks = design.get("tasks")
    if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
        raise ValueError("design.tasks must be a list")
    tasks = [dict(_object(item, "design task")) for item in raw_tasks]
    world_cohort = _object(design.get("world_cohort"), "world_cohort")
    public = _object(world_cohort.get("public_formal"), "world_cohort.public_formal")
    task_world_seeds = _object(public.get("task_world_seeds"), "task_world_seeds")
    task_ids = [str(task.get("task_id", "")) for task in tasks]
    public_cluster_count = sum(
        len(task_world_seeds.get(task_id, ())) for task_id in task_ids
    )
    campaign_contract = _object(design.get("campaign_contract"), "campaign_contract")
    selection_policy = _object(
        design.get("selection_and_failure_policy"), "selection_and_failure_policy"
    )
    expected_population = {
        "scheduled_public_cells": public_cluster_count * len(arms),
        "independent_task_world_clusters": public_cluster_count,
        "tasks": len(tasks),
        "worlds_per_task": public.get("worlds_per_task"),
        "prior_arms": list(arms),
        "provider_repeats_per_cell": campaign_contract.get("provider_repeats"),
        "all_scheduled_cells_retained": selection_policy.get(
            "failed_scientific_cells_retained_without_replacement"
        ),
    }
    population = _object(analysis.get("analysis_population"), "analysis_population")
    public_metadata_matches_schedule = (
        public.get("independent_world_cluster_count") == public_cluster_count
        and public.get("participant_cell_count")
        == expected_population["scheduled_public_cells"]
        and all(
            len(task_world_seeds.get(task_id, ())) == public.get("worlds_per_task")
            for task_id in task_ids
        )
    )
    if dict(population) != expected_population or not public_metadata_matches_schedule:
        errors.append("analysis population differs from the formal design")

    design_checkpoints = list(
        campaign_contract.get("checkpoint_complete_experiments", ())
    )
    expected_checkpoint_contract = {
        "stage_ids": list(FORMAL_SNAPSHOT_STAGES),
        "complete_experiments": design_checkpoints,
        "semantic_stage_hints_forbidden": True,
    }
    checkpoint_contract = _object(
        analysis.get("checkpoint_contract"), "checkpoint_contract"
    )
    if (
        design_checkpoints != list(FORMAL_CHECKPOINT_EXPERIMENTS)
        or dict(checkpoint_contract) != expected_checkpoint_contract
    ):
        errors.append("analysis checkpoint contract differs from the formal design")

    design_primary = _object(design.get("primary_hypothesis"), "design.primary_hypothesis")
    analysis_primary = _object(
        analysis.get("primary_hypothesis"), "analysis.primary_hypothesis"
    )
    if analysis_primary.get("id") != design_primary.get("id"):
        errors.append("analysis primary hypothesis differs from the formal design")

    implementation = _object(
        analysis.get("analysis_implementation_contract"),
        "analysis_implementation_contract",
    )
    implementation_primary = _object(
        implementation.get("primary_model"),
        "analysis_implementation_contract.primary_model",
    )
    if (
        implementation.get("expected_cluster_count")
        != expected_population["independent_task_world_clusters"]
        or implementation.get("expected_cell_count")
        != expected_population["scheduled_public_cells"]
    ):
        errors.append("analysis implementation denominators differ from its population")

    power = _object(analysis.get("power_design"), "power_design")
    expected_residual_df = public_cluster_count - len(tasks)
    if (
        power.get("independent_clusters") != public_cluster_count
        or power.get("task_fixed_effect_degrees_of_freedom") != len(tasks) - 1
        or power.get("residual_degrees_of_freedom") != expected_residual_df
        or power.get("alpha") != analysis_primary.get("alpha")
        or implementation_primary.get("residual_degrees_of_freedom")
        != expected_residual_df
        or implementation_primary.get("one_sided_alpha")
        != analysis_primary.get("alpha")
        or power.get("secondary_hypotheses_separately_powered") is not False
        or power.get("H4_confirmatory_claim_allowed") is not False
    ):
        errors.append("analysis power design differs from the formal population or model")
    return errors


def build_checkpoint_contract(config: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Materialize the complete public checkpoint contract used by one cell."""

    if arm not in FORMAL_ARMS:
        raise ValueError(f"unknown prior arm: {arm}")
    nominal = arm != "opaque"
    configured = config.get("belief_checkpoint")
    if isinstance(configured, Mapping):
        held_out_queries = [
            dict(_object(item, "held_out_query")) for item in configured["held_out_queries"]
        ]
        metric_ids = _string_list(configured["allowed_metric_ids"], "allowed_metric_ids")
        feature_ids = _string_list(configured["allowed_feature_ids"], "allowed_feature_ids")
        prior_fields = _string_list(configured["allowed_prior_fields"], "allowed_prior_fields")
    else:
        metric_ids = ["selective_product_yield", "energy_efficiency", "safety_risk"]
        feature_ids = [
            "electrolyte_profile",
            "solvent",
            "reagent_amount_mol",
            "potential_V",
            "current_mA",
            "duration_s",
        ]
        prior_fields = ["electrolyte_profile", "solvent"]
        held_out_queries = [
            {
                "query_id": query_id,
                "feature_values": {
                    "electrolyte_profile": electrolyte_profile,
                    "solvent": solvent,
                    "reagent_amount_mol": 0.01,
                    "potential_V": 0.8,
                    "current_mA": 100.0,
                    "duration_s": 1800.0,
                },
                "metric_ids": metric_ids,
            }
            for query_id, electrolyte_profile, solvent in (
                ("q-low", 0, 0),
                ("q-electrolyte", 3, 0),
                ("q-solvent", 0, 3),
                ("q-high", 3, 3),
            )
        ]
    query_metric_contract = {
        str(item["query_id"]): [str(metric) for metric in item.get("metric_ids", metric_ids)]
        for item in held_out_queries
    }
    prediction_pages = [
        {
            "page_id": f"predictions-{index // 4 + 1:03d}",
            "query_metric_contract": {
                query_id: query_metric_contract[query_id]
                for query_id in list(query_metric_contract)[index : index + 4]
            },
        }
        for index in range(0, len(query_metric_contract), 4)
    ]
    law_pages = [
        {
            "page_id": f"laws-{index // 2 + 1:03d}",
            "metric_ids": metric_ids[index : index + 2],
        }
        for index in range(0, len(metric_ids), 2)
    ]
    complete_experiments = int(_object(config["campaign"], "campaign")["complete_experiments"])
    snapshot_stages = [
        str(item)
        for item in config.get(
            "snapshot_stages",
            ["pre_evidence", "post_neutral", "post_discriminating", "final"],
        )
    ]
    checkpoint_experiments = [
        int(item)
        for item in _object(config["campaign"], "campaign")["checkpoint_complete_experiments"]
    ]
    if len(snapshot_stages) < 2 or len(set(snapshot_stages)) != len(snapshot_stages):
        raise ValueError("snapshot_stages must contain at least two unique stage IDs")
    if len(checkpoint_experiments) != len(snapshot_stages):
        raise ValueError("checkpoint experiment counts must match snapshot stages")
    if checkpoint_experiments != sorted(set(checkpoint_experiments)):
        raise ValueError("checkpoint experiment counts must be strictly increasing")
    if checkpoint_experiments[0] != 0 or checkpoint_experiments[-1] != complete_experiments:
        raise ValueError("checkpoint schedule must span pre-evidence through campaign completion")
    return {
        "schema_version": "chemworld-work-ii-campaign-checkpoint-contract-0.1",
        "snapshot_stages": snapshot_stages,
        "checkpoint_complete_experiments": checkpoint_experiments,
        "query_metric_contract": query_metric_contract,
        "held_out_queries": held_out_queries,
        "allowed_feature_ids": feature_ids,
        "allowed_metric_ids": metric_ids,
        "allowed_prior_fields": prior_fields,
        "evidence_catalog": [
            f"experiment-{index}-final-assay" for index in range(1, complete_experiments + 1)
        ],
        "nominal_information_available": nominal,
        "stage_labels_are_checkpoint_ids_only": True,
        "physical_experiment_selection_authority": "participant",
        "snapshot_submission_protocol": {
            "protocol": "staged_pages_v1",
            "prediction_pages": prediction_pages,
            "law_pages": law_pages,
            "submission_order": [
                page["page_id"] for page in (*prediction_pages, *law_pages)
            ],
            "finalize_requires_exact_full_snapshot": True,
            "partial_draft_counts_as_checkpoint": False,
            "participant_payload_auto_repair": False,
        },
    }


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"formal binding is outside the repository: {path}") from error


def _binding(root: Path, relative_path: str) -> dict[str, str]:
    path = root / relative_path
    if not path.is_file():
        raise ValueError(f"missing formal dependency: {relative_path}")
    return {
        "path": relative_path,
        "sha256": file_sha256(path),
        "hash_kind": "file_sha256",
    }


def formal_task_binding_key(c2_locus: str, task_id: str) -> str:
    """Return the locus-qualified task key used by every formal binding."""

    if c2_locus not in FORMAL_C2_LOCI or not task_id:
        raise ValueError("formal task binding requires a canonical locus and task ID")
    return f"{c2_locus}:{task_id}"


def _bound_object(
    root: Path,
    binding: Mapping[str, Any],
    *,
    label: str,
    embedded_field: str | None = None,
    embedded_hash: Any = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Load one repository-local file binding without trusting its path or digest."""

    errors: list[str] = []
    relative = binding.get("path")
    digest = binding.get("sha256")
    if not isinstance(relative, str) or not relative or not isinstance(digest, str):
        return None, [f"{label} binding is incomplete"]
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        return None, [f"{label} binding escapes the repository"]
    if not path.is_file():
        return None, [f"{label} binding is missing: {relative}"]
    if file_sha256(path) != digest:
        return None, [f"{label} binding is stale: {relative}"]
    try:
        payload = _load_object(path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return None, [f"{label} binding cannot be loaded: {error}"]
    if embedded_field is not None:
        value = payload.get(embedded_field)
        if value != binding.get("embedded_sha256") or value != embedded_hash(payload):
            errors.append(f"{label} embedded self-hash is invalid")
    return payload, errors


def _resolve_c2_terminal_task_specs(
    root: Path,
    admission: Mapping[str, Any],
    *,
    formal_runtime_configs: Mapping[tuple[str, str, int], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Resolve A_P/A_S formal tasks only from validated terminal receipts.

    The expansion is all-or-nothing: if any one of the four admitted tasks cannot
    be reconstructed through its receipt, outcome-blind selection, and campaign
    bindings, no C2 extension task is returned.
    """

    errors: list[str] = []
    specs: list[dict[str, Any]] = []
    blocks = admission.get("blocks")
    blocks = blocks if isinstance(blocks, Mapping) else {}
    for locus in C2_LOCI:
        block = blocks.get(locus)
        block = block if isinstance(block, Mapping) else {}
        rows = block.get("task_admissions")
        rows = rows if isinstance(rows, list) else []
        required = C2_REQUIRED_TASK_COUNTS[locus]
        if len(rows) != required:
            errors.append(f"{locus} requires exactly {required} terminal task receipts")
            continue
        locus_specs: list[dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            label = f"{locus} terminal task {index}"
            if not isinstance(row, Mapping) or row.get("passed") is not True:
                errors.append(f"{label} is not a passed terminal admission")
                continue
            task_id = row.get("task_id")
            receipt_binding = row.get("receipt_binding")
            if not isinstance(task_id, str) or not task_id:
                errors.append(f"{label} lacks a task ID")
                continue
            if not isinstance(receipt_binding, Mapping):
                errors.append(f"{label} lacks its receipt binding")
                continue
            receipt, binding_errors = _bound_object(
                root,
                receipt_binding,
                label=f"{label} receipt",
                embedded_field="receipt_sha256",
                embedded_hash=c2_task_admission_receipt_sha256,
            )
            errors.extend(binding_errors)
            if receipt is None:
                continue
            if (
                receipt.get("schema_version") != C2_TASK_ADMISSION_RECEIPT_VERSION
                or receipt.get("terminal_qualification_passed") is not True
                or receipt.get("validation_errors") != []
                or receipt.get("locus") != locus
                or receipt.get("task_id") != task_id
                or receipt.get("complete_experiments_per_cell")
                != C2_REQUIRED_ROUNDS[locus]
                or receipt.get("participant_outcomes_used_for_selection") is not False
                or receipt.get("formal_participant_outcomes_observed") != 0
            ):
                errors.append(f"{label} receipt is not an admissible terminal receipt")
                continue
            config_binding = receipt.get("campaign_config_binding")
            selection_binding = receipt.get("outcome_blind_selection_binding")
            if not isinstance(config_binding, Mapping):
                errors.append(f"{label} lacks its campaign config binding")
                continue
            runtime_binding: Mapping[str, Any] | None = None
            if formal_runtime_configs is not None:
                runtime_row = formal_runtime_configs.get(
                    (locus, task_id, C2_REQUIRED_ROUNDS[locus])
                )
                runtime_binding = (
                    runtime_row.get("formal_campaign_config_binding")
                    if isinstance(runtime_row, Mapping)
                    else None
                )
                if not isinstance(runtime_binding, Mapping) or config_binding.get(
                    "path"
                ) != runtime_binding.get("path"):
                    errors.append(
                        f"{label} campaign config is not the materialized formal runtime config"
                    )
                    continue
            if not isinstance(selection_binding, Mapping):
                errors.append(f"{label} lacks its outcome-blind selection binding")
                continue
            config, config_errors = _bound_object(
                root,
                config_binding,
                label=f"{label} campaign config",
            )
            selection, selection_errors = _bound_object(
                root,
                selection_binding,
                label=f"{label} selection",
                embedded_field="selection_sha256",
                embedded_hash=c2_outcome_blind_selection_sha256,
            )
            errors.extend(config_errors)
            errors.extend(selection_errors)
            if config is None or selection is None:
                continue
            if runtime_binding is not None and canonical_json_sha256(config) != runtime_binding.get(
                "canonical_json_sha256"
            ):
                errors.append(
                    f"{label} campaign config content differs from the formal runtime binding"
                )
                continue
            if config.get("task_id") != task_id:
                errors.append(f"{label} campaign config task identity drifted")
                continue
            if (
                selection.get("schema_version") != C2_OUTCOME_BLIND_SELECTION_VERSION
                or selection.get("locus") != locus
                or selection.get("task_id") != task_id
                or selection.get("selected_before_formal_participant_outcomes") is not True
                or selection.get("formal_participant_outcomes_observed") != 0
                or selection.get("formal_participant_outcomes_used") is not False
                or selection.get("selection_rule_frozen_before_evidence_review") is not True
            ):
                errors.append(f"{label} selection is not outcome blind")
                continue
            locus_specs.append(
                {
                    "c2_locus": locus,
                    "task_id": task_id,
                    "campaign_config": dict(config_binding),
                    "task_admission_receipt": dict(receipt_binding),
                    "outcome_blind_selection": dict(selection_binding),
                }
            )
        if len(locus_specs) == required:
            specs.extend(locus_specs)
    expected_total = sum(C2_REQUIRED_TASK_COUNTS.values())
    keys = [formal_task_binding_key(row["c2_locus"], row["task_id"]) for row in specs]
    if len(specs) != expected_total or len(set(keys)) != expected_total:
        if len(specs) == expected_total:
            errors.append("C2 terminal task roster contains duplicate locus-qualified keys")
        return [], errors
    return specs, errors


def _derive_c2_public_world_seeds(
    *,
    selection_sha256: str,
    c2_locus: str,
    task_id: str,
    namespace_start: int,
    namespace_size: int,
    unavailable: set[int],
) -> list[int]:
    """Derive five outcome-blind public identities from a frozen selection receipt."""

    seeds: list[int] = []
    for world_index in range(1, 6):
        collision_index = 0
        while True:
            payload = (
                "chemworld-work-ii-c2-public-world-v0.1:"
                f"{selection_sha256}:{c2_locus}:{task_id}:"
                f"{world_index}:{collision_index}"
            )
            digest = hashlib.sha256(payload.encode("utf-8")).digest()
            seed = namespace_start + int.from_bytes(digest[:8], "big") % namespace_size
            if seed not in unavailable:
                unavailable.add(seed)
                seeds.append(seed)
                break
            collision_index += 1
    return seeds


def _self_hash(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in payload.items() if key != "preflight_sha256"}
    )


def _cell_key_hash(cell: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in cell.items() if key != "cell_key_sha256"}
    )


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically publish a JSON artifact without ever replacing a prior file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise DuplicateFormalCellError(
                f"immutable formal cell artifact already exists: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


class DuplicateFormalCellError(RuntimeError):
    """A terminal formal cell would be executed or published more than once."""


class InvalidFormalCellReceiptError(RuntimeError):
    """A formal cell receipt does not satisfy its immutable binding."""


class ProviderAttemptLimitError(RuntimeError):
    """A formal cell exhausted its preregistered provider process launch cap."""


class WorkIIFormalCellStore:
    """Write-once terminal receipts plus append-only infrastructure attempts."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]) -> None:
        errors = validate_formal_preflight(manifest)
        if errors:
            raise ValueError("invalid formal manifest: " + "; ".join(errors))
        self.root = Path(root)
        self.receipts = self.root / "terminal_receipts"
        self.infrastructure_attempts = self.root / "infrastructure_attempts"
        self.provider_attempts = self.root / "provider_attempt_receipts"
        cells = manifest.get("cells", [])
        self.cells = {
            str(cell["cell_key_sha256"]): dict(cell) for cell in cells if isinstance(cell, Mapping)
        }
        if len(self.cells) != len(cells):
            raise ValueError("formal manifest contains duplicate cell keys")

    def receipt_path(self, cell_key_sha256: str) -> Path:
        self._cell(cell_key_sha256)
        return self.receipts / f"{cell_key_sha256}.json"

    def has_terminal(self, cell_key_sha256: str) -> bool:
        return self.receipt_path(cell_key_sha256).is_file()

    def write_terminal(
        self,
        cell_key_sha256: str,
        *,
        state: str,
        reason_code: str,
        result: Mapping[str, Any],
    ) -> Path:
        cell = self._cell(cell_key_sha256)
        if state not in FORMAL_TERMINAL_STATES:
            raise ValueError(f"unsupported formal terminal state: {state}")
        expected_prefixes = {
            "completed": ("scientific_completed_",),
            "right_censored": (
                "scientific_right_censored_",
                "method_right_censored_",
            ),
            "failed": ("method_failed_",),
        }[state]
        if not reason_code.startswith(expected_prefixes):
            raise ValueError(f"{state} formal cell reason code has an invalid domain prefix")
        result_payload = dict(result)
        payload = {
            "schema_version": FORMAL_RECEIPT_VERSION,
            "cell_key_sha256": cell_key_sha256,
            "cell": cell,
            "state": state,
            "reason_domain": ("scientific" if reason_code.startswith("scientific_") else "method"),
            "reason_code": reason_code,
            "result": result_payload,
            "result_sha256": canonical_json_sha256(result_payload),
        }
        payload["receipt_sha256"] = canonical_json_sha256(payload)
        target = self.receipt_path(cell_key_sha256)
        _write_json_once(target, payload)
        return target

    def load_terminal(self, cell_key_sha256: str) -> dict[str, Any]:
        path = self.receipt_path(cell_key_sha256)
        payload = _load_object(path)
        self._validate_receipt(payload, expected_key=cell_key_sha256)
        return payload

    def record_infrastructure_failure(
        self,
        cell_key_sha256: str,
        error: BaseException,
        *,
        log_reference: str | None = None,
        log_sha256: str | None = None,
    ) -> Path:
        cell = self._cell(cell_key_sha256)
        if (log_reference is None) != (log_sha256 is None):
            raise ValueError("log reference and digest must be supplied together")
        payload = {
            "schema_version": FORMAL_RECEIPT_VERSION,
            "cell_key_sha256": cell_key_sha256,
            "cell_id": cell["cell_id"],
            "state": "retryable_infrastructure_failure",
            "reason_domain": "infrastructure",
            "reason_code": "infrastructure_cell_attempt_failed",
            "error_type": type(error).__name__,
            "error_message": str(error)[:2000],
            "log_reference": log_reference,
            "log_sha256": log_sha256,
        }
        payload["attempt_sha256"] = canonical_json_sha256(payload)
        target = self.infrastructure_attempts / cell_key_sha256 / f"{uuid4().hex}.json"
        _write_json_once(target, payload)
        return target

    def record_provider_attempt_launch(
        self,
        cell_key_sha256: str,
        *,
        attempt_id: str,
    ) -> Path:
        cell = self._cell(cell_key_sha256)
        existing = sorted((self.provider_attempts / cell_key_sha256).glob("*.json"))
        limit = int(cell["provider_attempt_limit"])
        if len(existing) >= limit:
            raise ProviderAttemptLimitError(
                f"formal cell exhausted provider attempt cap {limit}: {cell['cell_id']}"
            )
        payload = {
            "schema_version": FORMAL_RECEIPT_VERSION,
            "cell_key_sha256": cell_key_sha256,
            "cell_id": cell["cell_id"],
            "attempt_id": attempt_id,
            "attempt_index": len(existing) + 1,
            "attempt_limit": limit,
            "state": "provider_process_launch_authorized",
            "reason_domain": "method",
        }
        payload["attempt_sha256"] = canonical_json_sha256(payload)
        target = self.provider_attempts / cell_key_sha256 / f"{attempt_id}.json"
        _write_json_once(target, payload)
        return target

    def pending_cells(self, *, resume: bool) -> list[dict[str, Any]]:
        audit = self.audit()
        if audit["invalid_receipts"] or audit["unexpected_cell_key_sha256"]:
            raise InvalidFormalCellReceiptError(
                "formal store contains invalid or unexpected terminal receipts"
            )
        if audit["terminal_count"] and not resume:
            raise DuplicateFormalCellError(
                "formal store already contains terminal cells; use missing-only resume"
            )
        completed = set(audit["terminal_cell_key_sha256"])
        attempt_counts = audit["provider_attempt_counts_by_cell_key_sha256"]
        exhausted = [
            cell["cell_id"]
            for key, cell in self.cells.items()
            if key not in completed
            and int(attempt_counts.get(key, 0)) >= int(cell["provider_attempt_limit"])
        ]
        if exhausted:
            raise ProviderAttemptLimitError(
                "missing formal cells exhausted their provider attempt cap: " + ", ".join(exhausted)
            )
        return [dict(cell) for key, cell in self.cells.items() if key not in completed]

    def audit(self) -> dict[str, Any]:
        observed: dict[str, Mapping[str, Any]] = {}
        invalid: list[str] = []
        for path in sorted(self.receipts.glob("*.json")):
            try:
                payload = _load_object(path)
                key = str(payload["cell_key_sha256"])
                self._validate_receipt(payload)
                if path.stem != key or key in observed:
                    raise ValueError("receipt path or uniqueness invariant failed")
                observed[key] = payload
            except (
                KeyError,
                OSError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
                InvalidFormalCellReceiptError,
            ):
                invalid.append(path.as_posix())
        provider_attempt_indices: dict[str, set[int]] = {}
        for path in sorted(self.provider_attempts.glob("*/*.json")):
            try:
                payload = _load_object(path)
                key = str(payload["cell_key_sha256"])
                attempt_id = str(payload["attempt_id"])
                expected_hash = canonical_json_sha256(
                    {name: value for name, value in payload.items() if name != "attempt_sha256"}
                )
                cell = self.cells[key]
                attempt_index = int(payload.get("attempt_index", -1))
                observed_indices = provider_attempt_indices.setdefault(key, set())
                if (
                    payload.get("schema_version") != FORMAL_RECEIPT_VERSION
                    or payload.get("state") != "provider_process_launch_authorized"
                    or payload.get("reason_domain") != "method"
                    or payload.get("attempt_sha256") != expected_hash
                    or path.parent.name != key
                    or path.stem != attempt_id
                    or attempt_index < 1
                    or attempt_index > int(cell["provider_attempt_limit"])
                    or attempt_index in observed_indices
                    or int(payload.get("attempt_limit", -1)) != int(cell["provider_attempt_limit"])
                ):
                    raise ValueError("provider attempt invariant failed")
                observed_indices.add(attempt_index)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        provider_attempt_counts = {
            key: len(indices) for key, indices in provider_attempt_indices.items()
        }
        infrastructure_attempt_count = 0
        recovered: set[str] = set()
        for path in sorted(self.infrastructure_attempts.glob("*/*.json")):
            try:
                payload = _load_object(path)
                key = str(payload["cell_key_sha256"])
                expected_hash = canonical_json_sha256(
                    {name: value for name, value in payload.items() if name != "attempt_sha256"}
                )
                if (
                    payload.get("schema_version") != FORMAL_RECEIPT_VERSION
                    or payload.get("state") != "retryable_infrastructure_failure"
                    or payload.get("reason_domain") != "infrastructure"
                    or payload.get("attempt_sha256") != expected_hash
                    or key not in self.cells
                    or path.parent.name != key
                ):
                    raise ValueError("infrastructure attempt invariant failed")
                infrastructure_attempt_count += 1
                if key in observed:
                    recovered.add(key)
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(path.as_posix())
        expected = set(self.cells)
        observed_keys = set(observed)
        state_counts = {
            state: sum(payload.get("state") == state for payload in observed.values())
            for state in sorted(FORMAL_TERMINAL_STATES)
        }
        report: dict[str, Any] = {
            "schema_version": FORMAL_STORE_AUDIT_VERSION,
            "expected_cell_count": len(expected),
            "terminal_count": len(observed_keys & expected),
            "state_counts": state_counts,
            "terminal_cell_key_sha256": sorted(observed_keys & expected),
            "missing_cell_key_sha256": sorted(expected - observed_keys),
            "unexpected_cell_key_sha256": sorted(observed_keys - expected),
            "invalid_receipts": invalid,
            "infrastructure_attempt_count": infrastructure_attempt_count,
            "provider_attempt_count": sum(provider_attempt_counts.values()),
            "provider_attempt_counts_by_cell_key_sha256": provider_attempt_counts,
            "recovered_infrastructure_failure_count": len(recovered),
            "complete": (
                observed_keys == expected and not invalid and not (observed_keys - expected)
            ),
        }
        report["audit_sha256"] = canonical_json_sha256(report)
        return report

    def _cell(self, cell_key_sha256: str) -> dict[str, Any]:
        try:
            return self.cells[str(cell_key_sha256)]
        except KeyError as error:
            raise ValueError(f"unknown formal cell key: {cell_key_sha256}") from error

    def _validate_receipt(
        self,
        payload: Mapping[str, Any],
        *,
        expected_key: str | None = None,
    ) -> None:
        key = str(payload.get("cell_key_sha256", ""))
        state = str(payload.get("state", ""))
        cell = payload.get("cell")
        result = payload.get("result")
        expected_receipt_hash = canonical_json_sha256(
            {name: value for name, value in payload.items() if name != "receipt_sha256"}
        )
        if (
            payload.get("schema_version") != FORMAL_RECEIPT_VERSION
            or key not in self.cells
            or not isinstance(cell, Mapping)
            or cell != self.cells.get(key)
            or _cell_key_hash(cell) != key
            or state not in FORMAL_TERMINAL_STATES
            or not isinstance(result, Mapping)
            or canonical_json_sha256(result) != payload.get("result_sha256")
            or payload.get("receipt_sha256") != expected_receipt_hash
        ):
            raise InvalidFormalCellReceiptError("invalid formal terminal receipt")
        if expected_key is not None and key != expected_key:
            raise InvalidFormalCellReceiptError(
                "formal terminal receipt does not match the expected cell"
            )


def build_formal_preflight(
    root: Path,
    design_path: Path,
    analysis_path: Path,
    c2_admission_plan_path: Path | None = None,
    formal_runtime_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Build the deterministic C2 public schedule without provider execution.

    A_E is always materialized as the admission-bound 75-cell subblock.  The
    additional 60 A_P/A_S cells appear only when all four terminal task receipts
    can be reconstructed exactly from the C2 admission report.
    """

    root = root.resolve()
    design_path = design_path.resolve()
    analysis_path = analysis_path.resolve()
    c2_admission_plan_path = (
        root / DEFAULT_C2_ADMISSION_PLAN
        if c2_admission_plan_path is None
        else c2_admission_plan_path.resolve()
    )
    design = _load_object(design_path)
    analysis = _load_object(analysis_path)
    prerequisite_errors = _validate_environment_binding(root, design)
    errors: list[str] = []
    design_version = design.get("schema_version")
    expected_analysis_version = SUPPORTED_FORMAL_ANALYSIS_VERSIONS.get(
        design_version
    )
    if expected_analysis_version is None:
        raise ValueError(f"unsupported formal design version: {design_version}")
    if analysis.get("schema_version") != expected_analysis_version:
        raise ValueError(
            "analysis schema is unsupported for the formal design version"
        )

    formal_runtime_manifest: dict[str, Any] | None = None
    formal_runtime_configs: dict[tuple[str, str, int], Mapping[str, Any]] | None = None
    formal_runtime_manifest_binding: dict[str, str] | None = None
    if design_version == FORMAL_DESIGN_VERSION:
        if formal_runtime_manifest_path is None:
            errors.append("current formal design requires an explicit formal runtime manifest")
        else:
            formal_runtime_manifest_path = formal_runtime_manifest_path.resolve()
            try:
                formal_runtime_manifest_path.relative_to(root)
            except ValueError:
                errors.append("formal runtime manifest is outside the repository")
            else:
                try:
                    formal_runtime_manifest = _load_object(formal_runtime_manifest_path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    errors.append(
                        f"formal runtime manifest cannot be loaded: {type(error).__name__}"
                    )
                else:
                    runtime_errors = validate_formal_runtime_manifest(
                        root,
                        formal_runtime_manifest,
                        manifest_path=formal_runtime_manifest_path,
                    )
                    errors.extend(f"formal runtime manifest: {item}" for item in runtime_errors)
                    design_binding = formal_runtime_manifest.get("formal_design_binding")
                    design_binding = design_binding if isinstance(design_binding, Mapping) else {}
                    if design_binding.get("path") != _relative(
                        root, design_path
                    ) or design_binding.get("sha256") != file_sha256(design_path):
                        errors.append(
                            "formal runtime manifest is not bound to the requested formal design"
                        )
                        runtime_errors.append("requested formal design binding differs")
                    if not runtime_errors:
                        rows = formal_runtime_manifest.get("task_configs")
                        rows = rows if isinstance(rows, list) else []
                        formal_runtime_configs = {
                            (
                                str(row.get("locus")),
                                str(row.get("task_id")),
                                int(row.get("rounds", -1)),
                            ): row
                            for row in rows
                            if isinstance(row, Mapping)
                        }
                        formal_runtime_manifest_binding = {
                            "path": _relative(root, formal_runtime_manifest_path),
                            "sha256": file_sha256(formal_runtime_manifest_path),
                            "hash_kind": "file_sha256",
                        }

    design_digest = canonical_json_sha256(design)
    analysis_digest = canonical_json_sha256(analysis)
    errors.extend(
        _analysis_design_relation_errors(
            design,
            analysis,
            _relative(root, design_path),
        )
    )
    law_summary_evaluation_contract = dict(
        _object(
            analysis.get("law_summary_evaluation_contract"),
            "law_summary_evaluation_contract",
        )
    )
    if law_summary_evaluation_contract != EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT:
        errors.append("formal law-summary evaluation contract differs from the frozen analysis")
    law_summary_evaluation_contract_sha256 = canonical_json_sha256(
        law_summary_evaluation_contract
    )
    private_confirmation_contract = dict(
        _object(
            analysis.get("private_confirmation_contract"),
            "private_confirmation_contract",
        )
    )
    if private_confirmation_contract != EXPECTED_PRIVATE_CONFIRMATION_CONTRACT:
        errors.append("private confirmation contract differs from the frozen analysis")
    private_confirmation_contract_sha256 = canonical_json_sha256(
        private_confirmation_contract
    )

    world_cohort = _object(design.get("world_cohort"), "world_cohort")
    development = _object(
        world_cohort.get("development_and_qualification"),
        "world_cohort.development_and_qualification",
    )
    public = _object(world_cohort.get("public_formal"), "world_cohort.public_formal")
    private = _object(
        world_cohort.get("private_confirmation"),
        "world_cohort.private_confirmation",
    )
    task_world_seeds = _object(public.get("task_world_seeds"), "task_world_seeds")
    if design_version == FORMAL_DESIGN_VERSION:
        expected_ae_public, expected_ae_construction, cohort_errors = (
            load_ae_formal_cohort(root, design)
        )
        errors.extend(cohort_errors)
    elif design_version == LEGACY_FORMAL_DESIGN_VERSION:
        expected_ae_public = {
            str(task_id): [int(seed) for seed in seeds]
            for task_id, seeds in task_world_seeds.items()
        }
        expected_ae_construction = {}
    development_seeds = [int(item) for item in development.get("world_seeds", [])]
    public_namespace_start = int(public.get("namespace_start", -1))
    public_namespace_size = int(public.get("namespace_size", -1))
    private_namespace_start = int(private.get("namespace_start", -1))
    private_namespace_size = int(private.get("namespace_size", -1))
    public_namespace_end = public_namespace_start + public_namespace_size
    private_namespace_end = private_namespace_start + private_namespace_size
    namespace_disjoint = (
        public_namespace_end <= private_namespace_start
        or private_namespace_end <= public_namespace_start
    )
    private_commitment = private.get("sealed_identity_commitment_sha256")
    if len(development_seeds) != len(set(development_seeds)):
        errors.append("development/qualification world identities are duplicated")
    if public_namespace_start < 0 or public_namespace_size <= 0:
        errors.append("public formal world namespace is invalid")
    if private_namespace_start < 0 or private_namespace_size <= 0:
        errors.append("private confirmation world namespace is invalid")
    if not namespace_disjoint:
        errors.append("public and private world namespaces overlap")
    if (
        not isinstance(private_commitment, str)
        or len(private_commitment) != 64
        or any(character not in "0123456789abcdef" for character in private_commitment)
    ):
        errors.append("private confirmation identity commitment is invalid")
    if private.get("identities_tracked_in_git") is not False:
        errors.append("private confirmation identities must remain outside Git")
    raw_tasks = design.get("tasks")
    if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, Sequence):
        raise ValueError("design.tasks must be a list")
    tasks = [dict(_object(item, "design task")) for item in raw_tasks]
    if len(tasks) != 5:
        errors.append("formal design must contain exactly five tasks")

    participant_execution_contract = dict(
        _object(
            design.get("participant_execution_contract"),
            "participant_execution_contract",
        )
    )
    expected_participant_execution_contract = (
        EXPECTED_PARTICIPANT_EXECUTION_CONTRACT
        if design_version == FORMAL_DESIGN_VERSION
        else LEGACY_EXPECTED_PARTICIPANT_EXECUTION_CONTRACT
    )
    if participant_execution_contract != expected_participant_execution_contract:
        errors.append("formal participant execution contract differs from the frozen method")
    reference_policy_contract = dict(
        _object(design.get("reference_policy_contract"), "reference_policy_contract")
    )
    if reference_policy_contract != EXPECTED_REFERENCE_POLICY_CONTRACT:
        errors.append("formal reference-policy contract differs from the frozen role")
    method_qualification_contract = dict(
        _object(
            design.get("method_qualification_contract"),
            "method_qualification_contract",
        )
    )
    expected_method_qualification_contract = (
        EXPECTED_METHOD_QUALIFICATION_CONTRACT
        if design_version == FORMAL_DESIGN_VERSION
        else LEGACY_EXPECTED_METHOD_QUALIFICATION_CONTRACT
    )
    if method_qualification_contract != expected_method_qualification_contract:
        errors.append("formal method-qualification contract differs from the frozen gate")
    participant_execution_contract_sha256 = canonical_json_sha256(participant_execution_contract)
    method_qualification_contract_sha256 = canonical_json_sha256(method_qualification_contract)

    cells: list[dict[str, Any]] = []
    task_bindings: list[dict[str, Any]] = []
    provider_contract: dict[str, Any] | None = None
    attempt_contract = dict(
        _object(design.get("provider_attempt_contract"), "provider_attempt_contract")
    )
    expected_attempt_invariants = {
        "attempt_unit": "host_codex_process_launch",
        "initial_attempts_per_cell": 1,
        "maximum_infrastructure_resume_attempts_per_cell": 1,
        "maximum_total_provider_attempts_per_cell": 2,
        "pre_action_restart_limit_within_attempt": 0,
        "any_persisted_trajectory_forbids_replacement": True,
        "retry_after_scientific_operation_forbidden": True,
    }
    if any(
        attempt_contract.get(key) != value
        for key, value in expected_attempt_invariants.items()
    ):
        errors.append("formal provider-attempt contract differs from the frozen cap")
    blind_contract = dict(
        _object(design.get("blind_evaluator_contract"), "blind_evaluator_contract")
    )
    expected_blind_invariants = {
        "participant_final_recommendations_per_cell": 1,
        "recommendation_unit": "one_selected_completed_experiment_index",
        "incumbent_definition": (
            "highest_participant_observed_leaderboard_score_tie_smallest_index"
        ),
        "blind_targets_per_cell": [
            "observed_incumbent",
            "participant_final_recommendation",
        ],
        "blind_replicates_per_target": 3,
        "paired_noise_within_replicate": True,
        "participant_feedback_from_blind_evaluator": False,
        "evaluator_provider_calls": 0,
        "evaluator_trajectory_separate_from_participant": True,
        "evaluator_resources_excluded_from_participant_ledger": True,
    }
    if any(
        blind_contract.get(key) != value
        for key, value in expected_blind_invariants.items()
    ):
        errors.append("formal blind-evaluator contract differs from the frozen denominator")
    truth_contract = dict(
        _object(
            design.get("held_out_evaluator_contract"),
            "held_out_evaluator_contract",
        )
    )
    expected_truth_invariants = {
        "truth_unit": "task_x_world_cluster_x_registered_query",
        "shared_across_prior_arms_and_checkpoints": True,
        "one_frozen_complete_experiment_per_query": True,
        "keyed_observation_coordinate_per_query": True,
        "exact_replay_required": True,
        "failed_truth_executions_retained_without_replacement": True,
        "evaluator_provider_calls": 0,
        "participant_feedback_from_truth_evaluator": False,
        "evaluator_trajectory_separate_from_participant": True,
        "evaluator_resources_excluded_from_participant_ledger": True,
    }
    if any(
        truth_contract.get(key) != value
        for key, value in expected_truth_invariants.items()
    ):
        errors.append("formal held-out evaluator contract differs from the frozen denominator")
    total_query_count = 0
    total_query_metric_count = 0
    evaluator_truth_execution_count = 0
    evaluator_truth_query_metric_count = 0
    public_world_seeds: list[int] = []
    construction_world_seeds = [
        seed for seeds in expected_ae_construction.values() for seed in seeds
    ]
    locus_task_counts = dict.fromkeys(FORMAL_C2_LOCI, 0)

    def add_task_to_schedule(spec: Mapping[str, Any], seeds: list[int]) -> None:
        nonlocal provider_contract
        nonlocal total_query_count, total_query_metric_count
        nonlocal evaluator_truth_execution_count, evaluator_truth_query_metric_count
        locus = str(spec["c2_locus"])
        task_id = str(spec["task_id"])
        task_key = formal_task_binding_key(locus, task_id)
        locus_contract = FORMAL_C2_LOCUS_CONTRACT[locus]
        complete_experiments = int(locus_contract["complete_experiments_per_cell"])
        checkpoints = tuple(locus_contract["checkpoint_complete_experiments"])
        snapshots = tuple(locus_contract["snapshot_stages"])
        config_binding = dict(_object(spec["campaign_config"], f"{task_key}.binding"))
        config_binding["hash_kind"] = "file_sha256"
        relative_config = str(config_binding.get("path"))
        config = _load_object(root / relative_config)
        label = task_key
        if config.get("task_id") != task_id:
            errors.append(f"{label}: campaign task identity mismatch")
        if tuple(config.get("prior_arms", {})) != FORMAL_ARMS:
            errors.append(f"{label}: campaign prior-arm order mismatch")
        opaque_contract = build_checkpoint_contract(config, "opaque")
        aligned_contract = build_checkpoint_contract(config, "aligned_nominal")
        misindexed_contract = build_checkpoint_contract(config, "misindexed_nominal")
        if aligned_contract != misindexed_contract:
            errors.append(f"{label}: informed checkpoint contracts are not matched")
        if tuple(opaque_contract["snapshot_stages"]) != snapshots:
            errors.append(f"{label}: checkpoint stage IDs differ from the locus contract")
        if tuple(opaque_contract["checkpoint_complete_experiments"]) != checkpoints:
            errors.append(f"{label}: checkpoint schedule differs from the locus contract")
        campaign = _object(config["campaign"], f"{label}.campaign")
        method_resources = _object(
            config.get("method_resources"), f"{label}.method_resources"
        )
        execution = _object(config.get("execution"), f"{label}.execution")
        if int(campaign.get("complete_experiments", -1)) != complete_experiments:
            errors.append(f"{label}: formal campaign experiment denominator differs")
        if config.get("episode_mode") != "campaign":
            errors.append(f"{label}: participant session scope is not campaign")
        if design_version == FORMAL_DESIGN_VERSION:
            provider = _object(config.get("provider"), f"{label}.provider")
            if not isinstance(config.get("resource_calibration_card_binding"), Mapping):
                errors.append(f"{label}: formal config lacks its W2-26 task resource card")
            if int(method_resources.get("model_call_limit", -1)) != 2:
                errors.append(
                    f"{label}: model-call limit does not admit one continuation"
                )
            if (
                int(provider.get("accepted_turn_continuation_limit", -1)) != 1
                or int(provider.get("provider_process_attempt_limit", -1)) != 3
            ):
                errors.append(
                    f"{label}: bounded same-thread continuation contract differs"
                )
        elif int(method_resources.get("model_call_limit", -1)) != 1:
            errors.append(f"{label}: legacy model-call limit is not one per cell")
        if int(method_resources.get("operation_limit", -1)) != int(
            campaign.get("operation_attempt_limit", -2)
        ):
            errors.append(f"{label}: method and campaign operation limits differ")
        if int(method_resources.get("complete_experiment_limit", -1)) != complete_experiments:
            errors.append(f"{label}: method complete-experiment limit differs")
        if tuple(method_resources.get("checkpoint_complete_experiments", ())) != checkpoints[1:]:
            errors.append(f"{label}: method checkpoint resource schedule differs")
        qualification = _object(config.get("qualification", {}), f"{label}.qualification")
        if locus == "A_E" and (
            int(qualification.get("minimum_unique_recipes", -1)) != 6
            or int(qualification.get("maximum_exact_repeats", -1)) != 2
        ):
            errors.append(f"{label}: formal recipe-diversity contract differs")
        if locus in C2_LOCI and qualification.get("q2_passed") is not True:
            errors.append(f"{label}: terminal campaign is not bound to passed Q2")
        if (
            int(execution.get("max_concurrency", -1)) != 3
            or int(execution.get("within_cell_concurrency", -1)) != 1
            or execution.get("parallelization_unit") != "same_seed_prior_arm_triplet"
        ):
            errors.append(f"{label}: execution concurrency contract differs")
        provider = dict(_object(config.get("provider"), f"{label}.provider"))
        reduced_provider = {
            key: provider.get(key)
            for key in (
                "id",
                "name",
                "base_url",
                "wire_api",
                "model",
                "reasoning_effort",
                "request_timeout_s",
                "finalization_timeout_s",
            )
        }
        timeout_contract = participant_execution_contract["timeout_contract_s"]
        sampling_contract = participant_execution_contract["sampling_contract"]
        if (
            provider.get("reasoning_effort") != sampling_contract["reasoning_effort"]
            or float(provider.get("request_timeout_s", -1.0))
            != float(timeout_contract["request"])
            or float(provider.get("finalization_timeout_s", -1.0))
            != float(timeout_contract["finalization"])
        ):
            errors.append(f"{label}: provider sampling or timeout contract differs")
        if provider_contract is None:
            provider_contract = reduced_provider
        elif provider_contract != reduced_provider:
            errors.append(f"{label}: provider/model/scaffold axis drift")
        if len(seeds) != 5 or len(set(seeds)) != 5:
            errors.append(f"{label}: public world schedule must contain five unique seeds")
        public_world_seeds.extend(seeds)
        for seed in seeds:
            if not public_namespace_start <= seed < public_namespace_end:
                errors.append(f"{label}: public world seed is outside its namespace")
            if seed in development_seeds:
                errors.append(f"{label}: public world seed overlaps qualification")
            if private_namespace_start <= seed < private_namespace_end:
                errors.append(f"{label}: public world seed enters the private namespace")
        checkpoint_digest = canonical_json_sha256(opaque_contract)
        query_count = len(opaque_contract["query_metric_contract"])
        query_metric_count = sum(
            len(metric_ids)
            for metric_ids in opaque_contract["query_metric_contract"].values()
        )
        evaluator_truth_execution_count += query_count * len(seeds)
        evaluator_truth_query_metric_count += query_metric_count * len(seeds)
        task_binding: dict[str, Any] = {
            "c2_locus": locus,
            "task_id": task_id,
            "task_binding_key": task_key,
            "campaign_config": config_binding,
            "checkpoint_contract_sha256": checkpoint_digest,
            "complete_experiments_per_cell": complete_experiments,
            "checkpoint_complete_experiments": list(checkpoints),
            "held_out_query_count_per_snapshot": query_count,
            "held_out_query_metric_count_per_snapshot": query_metric_count,
        }
        if locus in C2_LOCI:
            task_binding["task_admission_receipt"] = dict(
                _object(spec["task_admission_receipt"], f"{label}.receipt")
            )
            task_binding["outcome_blind_selection"] = dict(
                _object(spec["outcome_blind_selection"], f"{label}.selection")
            )
        task_bindings.append(task_binding)
        locus_task_counts[locus] += 1
        locus_task_index = locus_task_counts[locus]
        for world_index, world_seed in enumerate(seeds, start=1):
            cluster_id = (
                f"work-ii-public-{locus.lower().replace('_', '')}-"
                f"{locus_task_index:02d}-{world_index:02d}"
            )
            for arm_index, arm in enumerate(FORMAL_ARMS, start=1):
                checkpoint = build_checkpoint_contract(config, arm)
                cell: dict[str, Any] = {
                    "schema_version": FORMAL_CELL_VERSION,
                    "schedule_index": len(cells) + 1,
                    "cell_id": f"{cluster_id}-arm-{arm_index:02d}",
                    "world_cluster_id": cluster_id,
                    "c2_locus": locus,
                    "task_id": task_id,
                    "task_binding_key": task_key,
                    "world_index": world_index,
                    "world_seed": world_seed,
                    "world_split": "public_formal",
                    "prior_arm": arm,
                    "campaign_config_path": relative_config,
                    "campaign_config_sha256": config_binding["sha256"],
                    "checkpoint_contract_sha256": canonical_json_sha256(checkpoint),
                    "participant_execution_contract_sha256": (
                        participant_execution_contract_sha256
                    ),
                    "law_summary_evaluation_contract_sha256": (
                        law_summary_evaluation_contract_sha256
                    ),
                    "private_confirmation_contract_sha256": (
                        private_confirmation_contract_sha256
                    ),
                    "complete_experiment_count": complete_experiments,
                    "belief_checkpoint_count": len(checkpoints),
                    "checkpoint_complete_experiments": list(checkpoints),
                    "held_out_query_count_per_snapshot": query_count,
                    "held_out_query_metric_count_per_snapshot": query_metric_count,
                    "provider_session_limit": 1,
                    "provider_attempt_limit": int(
                        attempt_contract.get("maximum_total_provider_attempts_per_cell", -1)
                    ),
                    "provider_repeat": 1,
                    "participant_final_recommendation_count": 1,
                    "blind_validation_target_count": 2,
                    "blind_replicates_per_target": 3,
                    "blind_validation_execution_count": 6,
                    "terminal_states": ["completed", "right_censored", "failed"],
                }
                if locus in C2_LOCI:
                    receipt_binding = _object(
                        spec["task_admission_receipt"], f"{label}.receipt"
                    )
                    selection_binding = _object(
                        spec["outcome_blind_selection"], f"{label}.selection"
                    )
                    cell["task_admission_receipt_binding"] = dict(receipt_binding)
                    cell["outcome_blind_selection_binding"] = dict(selection_binding)
                if design_version == FORMAL_DESIGN_VERSION:
                    cell.update(
                        {
                            "accepted_participant_model_call_minimum": 1,
                            "accepted_participant_model_call_maximum": 2,
                            "accepted_turn_continuation_limit": 1,
                            "provider_process_attempt_limit_per_cell_attempt": 3,
                        }
                    )
                cell["cell_key_sha256"] = _cell_key_hash(cell)
                cells.append(cell)
                total_query_count += query_count * len(checkpoints)
                total_query_metric_count += query_metric_count * len(checkpoints)

    for task in tasks:
        task_id = str(task.get("task_id"))
        runtime_row = (
            formal_runtime_configs.get(("A_E", task_id, 8))
            if formal_runtime_configs is not None
            else None
        )
        runtime_binding = (
            runtime_row.get("formal_campaign_config_binding")
            if isinstance(runtime_row, Mapping)
            else None
        )
        relative_config = (
            str(runtime_binding.get("path"))
            if isinstance(runtime_binding, Mapping)
            else str(task.get("campaign_config"))
        )
        seeds = [int(item) for item in task_world_seeds.get(task_id, [])]
        add_task_to_schedule(
            {
                "c2_locus": "A_E",
                "task_id": task_id,
                "campaign_config": _binding(root, relative_config),
            },
            seeds,
        )

    ae_cells = [cell for cell in cells if cell.get("c2_locus") == "A_E"]
    if design_version == FORMAL_DESIGN_VERSION:
        errors.extend(validate_ae_public_cells(root, design, ae_cells))
    c2_admission = build_c2_admission_report(
        root,
        c2_admission_plan_path,
        design_path,
        ae_cells,
    )
    c2_specs, c2_schedule_errors = _resolve_c2_terminal_task_specs(
        root,
        c2_admission,
        formal_runtime_configs=formal_runtime_configs,
    )
    prerequisite_errors.extend(f"C2 formal schedule: {item}" for item in c2_schedule_errors)
    unavailable_seeds = {
        *development_seeds,
        *construction_world_seeds,
        *public_world_seeds,
    }
    for spec in c2_specs:
        selection_binding = _object(spec["outcome_blind_selection"], "selection binding")
        selection_sha256 = str(selection_binding.get("embedded_sha256", ""))
        seeds = _derive_c2_public_world_seeds(
            selection_sha256=selection_sha256,
            c2_locus=str(spec["c2_locus"]),
            task_id=str(spec["task_id"]),
            namespace_start=public_namespace_start,
            namespace_size=public_namespace_size,
            unavailable=unavailable_seeds,
        )
        add_task_to_schedule(spec, seeds)

    cell_ids = [str(cell["cell_id"]) for cell in cells]
    cell_keys = [str(cell["cell_key_sha256"]) for cell in cells]
    cluster_ids = {str(cell["world_cluster_id"]) for cell in cells}
    schedule_complete = len(c2_specs) == sum(C2_REQUIRED_TASK_COUNTS.values())
    expected_cells = FORMAL_EXPECTED_TOTALS["participant_cells"] if schedule_complete else 75
    expected_clusters = (
        FORMAL_EXPECTED_TOTALS["independent_task_world_clusters"]
        if schedule_complete
        else 25
    )
    if (
        len(cells) != expected_cells
        or len(set(cell_ids)) != expected_cells
        or len(set(cell_keys)) != expected_cells
    ):
        errors.append("formal schedule does not contain the expected unique cells")
    if len(cluster_ids) != expected_clusters:
        errors.append("formal schedule has an invalid independent cluster denominator")
    if (
        len(public_world_seeds) != expected_clusters
        or len(set(public_world_seeds)) != expected_clusters
    ):
        errors.append("public formal world schedule has an invalid identity denominator")
    c2_blockers = list(c2_admission.get("blocking_requirements", []))
    prerequisite_errors.extend(f"C2 admission: {item}" for item in c2_blockers)

    blockers = list(FORMAL_BLOCKING_REQUIREMENTS)
    if (
        design.get("formal_execution_allowed") is True
        or analysis.get("formal_execution_allowed") is True
    ):
        errors.append("pre-registration inputs unexpectedly allow formal execution")
    report: dict[str, Any] = {
        "schema_version": FORMAL_PREFLIGHT_VERSION,
        "status": (
            "failed_execution_blocked"
            if errors or prerequisite_errors
            else "passed_execution_blocked"
        ),
        "formal_result": False,
        "formal_execution_allowed": False,
        "design_binding": {
            "path": _relative(root, design_path),
            "sha256": design_digest,
            "hash_kind": "canonical_json_sha256",
        },
        "analysis_binding": {
            "path": _relative(root, analysis_path),
            "sha256": analysis_digest,
            "hash_kind": "canonical_json_sha256",
        },
        "provider_contract": provider_contract,
        "participant_execution_contract": participant_execution_contract,
        "participant_execution_contract_sha256": (participant_execution_contract_sha256),
        "law_summary_evaluation_contract": law_summary_evaluation_contract,
        "law_summary_evaluation_contract_sha256": (
            law_summary_evaluation_contract_sha256
        ),
        "private_confirmation_contract": private_confirmation_contract,
        "private_confirmation_contract_sha256": (
            private_confirmation_contract_sha256
        ),
        "reference_policy_contract": reference_policy_contract,
        "method_qualification_contract": method_qualification_contract,
        "method_qualification_contract_sha256": method_qualification_contract_sha256,
        "provider_attempt_contract": attempt_contract,
        "blind_evaluator_contract": blind_contract,
        "held_out_evaluator_contract": truth_contract,
        "schedule_policy": {
            "order": "task_then_public_world_then_prior_arm",
            "canonical_c2_locus_order": list(FORMAL_C2_LOCI),
            "schedule_complete": schedule_complete,
            "extension_policy": (
                "all_four_terminal_receipts_or_no_A_P_A_S_formal_cells"
            ),
            "same_world_arm_triplet_max_concurrency": 3,
            "within_cell_concurrency": 1,
            "one_persistent_session_per_cell": True,
            "missing_only_resume": True,
            "accepted_terminal_cells_are_immutable": True,
            "result_direction_early_stopping_forbidden": True,
        },
        "prompt_boundary": {
            "world_seed_exposed_to_participant": False,
            "world_cluster_id_exposed_to_participant": False,
            "prior_arm_label_exposed_to_participant": False,
            "private_identity_exposed_to_participant_or_manifest": False,
            "evaluator_truth_exposed_to_participant": False,
        },
        "world_split_contract": {
            "manifest_split": "public_formal",
            "development_and_qualification_world_seeds": development_seeds,
            "exposed_construction_only": {
                "task_world_seeds": expected_ae_construction,
                "world_identity_count": len(set(construction_world_seeds)),
                "participant_cell_count": 0,
            },
            "public_formal": {
                "namespace_start": public_namespace_start,
                "namespace_size": public_namespace_size,
                "task_world_seeds": expected_ae_public,
                "world_identity_count": len(set(public_world_seeds)),
            },
            "private_confirmation": {
                "namespace_start": private_namespace_start,
                "namespace_size": private_namespace_size,
                "sealed_identity_commitment_sha256": private_commitment,
                "identities_present_in_manifest": False,
            },
            "development_public_identity_disjoint": not bool(
                set(development_seeds) & set(public_world_seeds)
            ),
            "construction_public_identity_disjoint": not bool(
                set(construction_world_seeds) & set(public_world_seeds)
            ),
            "construction_participant_cell_count": 0,
            "public_private_namespace_disjoint": namespace_disjoint,
        },
        "expected_counts": {
            "tasks": len(task_bindings),
            "tasks_by_c2_locus": dict(locus_task_counts),
            "independent_task_world_clusters": len(cluster_ids),
            "participant_cells": len(cells),
            "participant_cells_by_c2_locus": {
                locus: sum(cell["c2_locus"] == locus for cell in cells)
                for locus in FORMAL_C2_LOCI
            },
            "provider_sessions": len(cells),
            "provider_attempts_initial_planned": len(cells),
            "provider_attempts_hard_cap": len(cells)
            * int(attempt_contract["maximum_total_provider_attempts_per_cell"]),
            "provider_repeats_per_cell": 1,
            "complete_experiments": sum(
                int(cell["complete_experiment_count"]) for cell in cells
            ),
            "belief_checkpoints": sum(
                int(cell["belief_checkpoint_count"]) for cell in cells
            ),
            "checkpoint_held_out_queries": total_query_count,
            "checkpoint_held_out_query_metrics": total_query_metric_count,
            "evaluator_truth_executions": evaluator_truth_execution_count,
            "evaluator_truth_query_metrics": evaluator_truth_query_metric_count,
            "participant_final_recommendations": len(cells),
            "blind_validation_targets": len(cells) * 2,
            "blind_validation_executions": len(cells) * 2 * 3,
        }
        | (
            {
                "accepted_participant_model_calls_minimum": len(cells),
                "accepted_participant_model_calls_maximum": len(cells) * 2,
            }
            if design_version == FORMAL_DESIGN_VERSION
            else {}
        ),
        "task_bindings": task_bindings,
        "c2_admission": c2_admission,
        "cells": cells,
        "blocking_requirements": [*blockers, *c2_blockers],
        "prerequisite_errors": prerequisite_errors,
        "errors": errors,
    }
    if design_version == FORMAL_DESIGN_VERSION:
        report["formal_runtime_manifest_binding"] = formal_runtime_manifest_binding
    report["preflight_sha256"] = _self_hash(report)
    return report


def validate_formal_preflight(report: Mapping[str, Any]) -> list[str]:
    """Validate self-hash, schedule uniqueness, and outcome-blind boundaries."""

    errors: list[str] = []
    prerequisite_errors = report.get("prerequisite_errors")
    if not isinstance(prerequisite_errors, list) or any(
        not isinstance(item, str) or not item for item in prerequisite_errors
    ):
        errors.append("formal preflight prerequisite errors are malformed")
    if report.get("schema_version") != FORMAL_PREFLIGHT_VERSION:
        errors.append("unexpected formal preflight schema")
    if report.get("preflight_sha256") != _self_hash(report):
        errors.append("formal preflight self-hash mismatch")
    execution_allowed = report.get("formal_execution_allowed")
    if execution_allowed is True:
        authorization = report.get("authorization_bindings")
        authorization = authorization if isinstance(authorization, Mapping) else {}
        c2_admission = report.get("c2_admission")
        c2_admission = c2_admission if isinstance(c2_admission, Mapping) else {}
        if (
            report.get("status") != "passed_execution_authorized"
            or report.get("blocking_requirements") != []
            or prerequisite_errors != []
            or report.get("errors") != []
            or c2_admission.get("status") != "ready_for_formal_authorization"
            or c2_admission.get("formal_execution_allowed") is not True
            or any(
                not isinstance(authorization.get(field), str)
                or len(str(authorization.get(field))) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in str(authorization.get(field))
                )
                for field in (
                    "base_preflight_sha256",
                    "qualification_receipt_sha256",
                    "preregistration_freeze_receipt_sha256",
                    "formal_cost_contract_sha256",
                    "c2_admission_sha256",
                )
            )
            or authorization.get("c2_admission_sha256")
            != c2_admission.get("admission_sha256")
        ):
            errors.append("formal execution manifest lacks exact authorization bindings")
    elif execution_allowed is False:
        embedded_errors = report.get("errors")
        embedded_errors = embedded_errors if isinstance(embedded_errors, list) else []
        expected_status = (
            "failed_execution_blocked"
            if prerequisite_errors or embedded_errors
            else "passed_execution_blocked"
        )
        c2_admission = report.get("c2_admission")
        c2_admission = c2_admission if isinstance(c2_admission, Mapping) else {}
        expected_blockers = [
            *FORMAL_BLOCKING_REQUIREMENTS,
            *list(c2_admission.get("blocking_requirements", [])),
        ]
        if (
            report.get("status") != expected_status
            or report.get("blocking_requirements") != expected_blockers
            or "authorization_bindings" in report
        ):
            errors.append("formal preflight does not preserve its blocked authorization state")
    else:
        errors.append("formal preflight has an invalid execution authorization state")
    cells = report.get("cells")
    if not isinstance(cells, list):
        errors.append("formal preflight cells are missing")
        return errors
    schedule_policy = report.get("schedule_policy")
    schedule_policy = schedule_policy if isinstance(schedule_policy, Mapping) else {}
    schedule_complete = schedule_policy.get("schedule_complete") is True
    expected_cell_count = (
        FORMAL_EXPECTED_TOTALS["participant_cells"] if schedule_complete else 75
    )
    ids = [cell.get("cell_id") for cell in cells if isinstance(cell, Mapping)]
    keys = [cell.get("cell_key_sha256") for cell in cells if isinstance(cell, Mapping)]
    if (
        len(cells) != expected_cell_count
        or len(set(ids)) != expected_cell_count
        or len(set(keys)) != expected_cell_count
    ):
        errors.append("formal preflight has an invalid unique cell denominator")
    if execution_allowed is True and not schedule_complete:
        errors.append("formal execution cannot authorize an incomplete C2 schedule")
    counts = report.get("expected_counts")
    if not isinstance(counts, Mapping) or counts.get("participant_cells") != len(cells):
        errors.append("formal preflight cell count is inconsistent")
        counts = {}
    task_bindings = report.get("task_bindings")
    task_bindings = task_bindings if isinstance(task_bindings, list) else []
    binding_by_key: dict[str, Mapping[str, Any]] = {}
    for row in task_bindings:
        if not isinstance(row, Mapping):
            errors.append("formal preflight contains a malformed task binding")
            continue
        locus = row.get("c2_locus")
        task_id = row.get("task_id")
        key = row.get("task_binding_key")
        if (
            locus not in FORMAL_C2_LOCI
            or not isinstance(task_id, str)
            or key != formal_task_binding_key(str(locus), task_id)
            or key in binding_by_key
        ):
            errors.append("formal preflight task binding is not locus-qualified and unique")
            continue
        binding_by_key[str(key)] = row
        if locus in C2_LOCI and (
            not isinstance(row.get("task_admission_receipt"), Mapping)
            or not isinstance(row.get("outcome_blind_selection"), Mapping)
        ):
            errors.append(f"formal C2 extension task lacks admission bindings: {key}")
    expected_task_count = FORMAL_EXPECTED_TOTALS["tasks"] if schedule_complete else 5
    if len(binding_by_key) != expected_task_count:
        errors.append("formal preflight task binding denominator is invalid")
    locus_cell_counts = dict.fromkeys(FORMAL_C2_LOCI, 0)
    locus_task_keys: dict[str, set[str]] = {locus: set() for locus in FORMAL_C2_LOCI}
    participant_contract = report.get("participant_execution_contract")
    current_method_contract = (
        participant_contract == EXPECTED_PARTICIPANT_EXECUTION_CONTRACT
    )
    runtime_binding = report.get("formal_runtime_manifest_binding")
    runtime_binding = runtime_binding if isinstance(runtime_binding, Mapping) else None
    embedded_errors = report.get("errors")
    embedded_errors = embedded_errors if isinstance(embedded_errors, list) else []
    if current_method_contract:
        runtime_error_recorded = any(
            isinstance(item, str)
            and (
                item.startswith("formal runtime manifest")
                or item == "current formal design requires an explicit formal runtime manifest"
            )
            for item in embedded_errors
        )
        if runtime_binding is None:
            if not runtime_error_recorded:
                errors.append("current formal preflight lacks its formal runtime manifest binding")
        elif (
            not isinstance(runtime_binding.get("path"), str)
            or not isinstance(runtime_binding.get("sha256"), str)
            or runtime_binding.get("hash_kind") != "file_sha256"
        ):
            errors.append("formal runtime manifest binding is malformed")
    dynamic_counts = {
        "tasks": len(binding_by_key),
        "independent_task_world_clusters": len(
            {
                cell.get("world_cluster_id")
                for cell in cells
                if isinstance(cell, Mapping)
            }
        ),
        "participant_cells": len(cells),
        "provider_sessions": len(cells),
        "provider_attempts_initial_planned": len(cells),
        "provider_attempts_hard_cap": sum(
            int(cell.get("provider_attempt_limit", -1))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "complete_experiments": sum(
            int(cell.get("complete_experiment_count", -1))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "belief_checkpoints": sum(
            int(cell.get("belief_checkpoint_count", -1))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "participant_final_recommendations": sum(
            int(cell.get("participant_final_recommendation_count", -1))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "blind_validation_targets": sum(
            int(cell.get("blind_validation_target_count", -1))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
        "blind_validation_executions": sum(
            int(cell.get("blind_validation_execution_count", -1))
            for cell in cells
            if isinstance(cell, Mapping)
        ),
    }
    if current_method_contract:
        dynamic_counts.update(
            {
                "accepted_participant_model_calls_minimum": len(cells),
                "accepted_participant_model_calls_maximum": len(cells) * 2,
            }
        )
    if any(counts.get(key) != value for key, value in dynamic_counts.items()):
        errors.append("formal preflight dynamic denominators are inconsistent")
    frozen_totals = FORMAL_EXPECTED_TOTALS.copy()
    if not current_method_contract:
        frozen_totals.pop("accepted_participant_model_calls_minimum")
        frozen_totals.pop("accepted_participant_model_calls_maximum")
    if schedule_complete and any(
        dynamic_counts.get(key) != value for key, value in frozen_totals.items()
    ):
        errors.append("complete formal C2 schedule differs from the frozen 135-cell totals")
    participant_contract_hash = report.get("participant_execution_contract_sha256")
    if (
        participant_contract
        not in (
            EXPECTED_PARTICIPANT_EXECUTION_CONTRACT,
            LEGACY_EXPECTED_PARTICIPANT_EXECUTION_CONTRACT,
        )
        or participant_contract_hash
        != canonical_json_sha256(participant_contract)
    ):
        errors.append("formal preflight participant execution contract is invalid")
    if report.get("reference_policy_contract") != EXPECTED_REFERENCE_POLICY_CONTRACT:
        errors.append("formal preflight reference-policy contract is invalid")
    law_summary_contract = report.get("law_summary_evaluation_contract")
    law_summary_contract_hash = report.get("law_summary_evaluation_contract_sha256")
    if (
        law_summary_contract != EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT
        or law_summary_contract_hash
        != canonical_json_sha256(EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT)
    ):
        errors.append("formal preflight law-summary evaluation contract is invalid")
    private_contract = report.get("private_confirmation_contract")
    private_contract_hash = report.get("private_confirmation_contract_sha256")
    if (
        private_contract != EXPECTED_PRIVATE_CONFIRMATION_CONTRACT
        or private_contract_hash
        != canonical_json_sha256(EXPECTED_PRIVATE_CONFIRMATION_CONTRACT)
    ):
        errors.append("formal preflight private-confirmation contract is invalid")
    qualification_contract = report.get("method_qualification_contract")
    qualification_contract_hash = report.get("method_qualification_contract_sha256")
    if (
        qualification_contract
        not in (
            EXPECTED_METHOD_QUALIFICATION_CONTRACT,
            LEGACY_EXPECTED_METHOD_QUALIFICATION_CONTRACT,
        )
        or qualification_contract_hash != canonical_json_sha256(qualification_contract)
        or current_method_contract
        != (qualification_contract == EXPECTED_METHOD_QUALIFICATION_CONTRACT)
    ):
        errors.append("formal preflight method-qualification contract is invalid")
    for cell in cells:
        if not isinstance(cell, Mapping):
            errors.append("formal preflight contains a malformed cell")
            continue
        if cell.get("cell_key_sha256") != _cell_key_hash(cell):
            errors.append(f"formal cell self-hash mismatch: {cell.get('cell_id')}")
        if cell.get("participant_execution_contract_sha256") != participant_contract_hash:
            errors.append(f"formal cell participant contract mismatch: {cell.get('cell_id')}")
        if cell.get("law_summary_evaluation_contract_sha256") != law_summary_contract_hash:
            errors.append(f"formal cell law-summary contract mismatch: {cell.get('cell_id')}")
        if cell.get("private_confirmation_contract_sha256") != private_contract_hash:
            errors.append(
                "formal cell private-confirmation contract mismatch: "
                f"{cell.get('cell_id')}"
            )
        locus = cell.get("c2_locus")
        task_id = cell.get("task_id")
        if locus not in FORMAL_C2_LOCI or not isinstance(task_id, str):
            errors.append(f"formal cell lacks a canonical C2 locus: {cell.get('cell_id')}")
            continue
        task_key = formal_task_binding_key(str(locus), task_id)
        binding = binding_by_key.get(task_key)
        binding_record: Mapping[str, Any] = binding if binding is not None else {}
        contract = FORMAL_C2_LOCUS_CONTRACT[str(locus)]
        locus_cell_counts[str(locus)] += 1
        locus_task_keys[str(locus)].add(task_key)
        if (
            cell.get("task_binding_key") != task_key
            or binding is None
            or cell.get("campaign_config_path")
            != binding_record.get("campaign_config", {}).get("path")
            or cell.get("campaign_config_sha256")
            != binding_record.get("campaign_config", {}).get("sha256")
            or cell.get("complete_experiment_count")
            != contract["complete_experiments_per_cell"]
            or cell.get("belief_checkpoint_count") != 5
            or cell.get("checkpoint_complete_experiments")
            != list(contract["checkpoint_complete_experiments"])
            or cell.get("participant_final_recommendation_count") != 1
            or cell.get("blind_validation_target_count") != 2
            or cell.get("blind_replicates_per_target") != 3
            or cell.get("blind_validation_execution_count") != 6
        ):
            errors.append(f"formal cell denominator or task binding drifted: {cell.get('cell_id')}")
        if locus in C2_LOCI and (
            cell.get("task_admission_receipt_binding")
            != binding_record.get("task_admission_receipt")
            or cell.get("outcome_blind_selection_binding")
            != binding_record.get("outcome_blind_selection")
        ):
            errors.append(f"formal C2 cell lacks exact admission bindings: {cell.get('cell_id')}")
    expected_locus_cells = {
        "A_E": 75,
        "A_P": 30 if schedule_complete else 0,
        "A_S": 30 if schedule_complete else 0,
    }
    expected_locus_tasks = {
        "A_E": 5,
        "A_P": 2 if schedule_complete else 0,
        "A_S": 2 if schedule_complete else 0,
    }
    if locus_cell_counts != expected_locus_cells or {
        locus: len(keys) for locus, keys in locus_task_keys.items()
    } != expected_locus_tasks:
        errors.append("formal C2 locus roster is incomplete or malformed")
    if counts.get("participant_cells_by_c2_locus") != expected_locus_cells:
        errors.append("formal C2 cell denominators are not reported by locus")
    if counts.get("tasks_by_c2_locus") != expected_locus_tasks:
        errors.append("formal C2 task denominators are not reported by locus")
    split = report.get("world_split_contract")
    if not isinstance(split, Mapping):
        errors.append("formal preflight world-split contract is missing")
    else:
        public = split.get("public_formal")
        private = split.get("private_confirmation")
        development_seeds = split.get("development_and_qualification_world_seeds")
        if (
            split.get("manifest_split") != "public_formal"
            or split.get("development_public_identity_disjoint") is not True
            or split.get("public_private_namespace_disjoint") is not True
            or not isinstance(public, Mapping)
            or not isinstance(private, Mapping)
            or not isinstance(development_seeds, list)
        ):
            errors.append("formal preflight world-split contract is not fail-closed")
        else:
            public_start = public.get("namespace_start")
            public_size = public.get("namespace_size")
            private_start = private.get("namespace_start")
            private_size = private.get("namespace_size")
            public_start_int = (
                public_start
                if isinstance(public_start, int) and not isinstance(public_start, bool)
                else None
            )
            public_size_int = (
                public_size
                if isinstance(public_size, int) and not isinstance(public_size, bool)
                else None
            )
            private_start_int = (
                private_start
                if isinstance(private_start, int) and not isinstance(private_start, bool)
                else None
            )
            private_size_int = (
                private_size
                if isinstance(private_size, int) and not isinstance(private_size, bool)
                else None
            )
            if (
                public_start_int is None
                or public_size_int is None
                or private_start_int is None
                or private_size_int is None
                or public_size_int <= 0
                or private_size_int <= 0
            ):
                errors.append("formal preflight world namespaces are invalid")
            else:
                public_end = public_start_int + public_size_int
                private_end = private_start_int + private_size_int
                if not (
                    public_end <= private_start_int or private_end <= public_start_int
                ):
                    errors.append("formal preflight public/private namespaces overlap")
                raw_cell_seeds = [
                    cell.get("world_seed") for cell in cells if isinstance(cell, Mapping)
                ]
                cell_seeds = {
                    seed
                    for seed in raw_cell_seeds
                    if isinstance(seed, int) and not isinstance(seed, bool)
                }
                expected_identities = (
                    FORMAL_EXPECTED_TOTALS["independent_task_world_clusters"]
                    if schedule_complete
                    else 25
                )
                if (
                    len(cell_seeds) != expected_identities
                    or public.get("world_identity_count") != expected_identities
                ):
                    errors.append("formal preflight public identity denominator is invalid")
                if any(
                    not isinstance(seed, int)
                    or isinstance(seed, bool)
                    or not public_start_int <= seed < public_end
                    or private_start_int <= seed < private_end
                    or seed in development_seeds
                    for seed in raw_cell_seeds
                ):
                    errors.append("formal preflight contains a cross-split world identity")
            commitment = private.get("sealed_identity_commitment_sha256")
            if (
                private.get("identities_present_in_manifest") is not False
                or not isinstance(commitment, str)
                or len(commitment) != 64
            ):
                errors.append("formal preflight private identity boundary is invalid")
        if any(
            isinstance(cell, Mapping)
            and (
                cell.get("world_split") != "public_formal"
                or any(
                    field in cell
                    for field in (
                        "private_identity",
                        "private_world_id",
                        "private_world_seed",
                    )
                )
            )
            for cell in cells
        ):
            errors.append("formal preflight cell crossed the public/private boundary")
    prompt = report.get("prompt_boundary")
    if not isinstance(prompt, Mapping) or any(
        prompt.get(key) is not False
        for key in (
            "world_seed_exposed_to_participant",
            "world_cluster_id_exposed_to_participant",
            "prior_arm_label_exposed_to_participant",
            "private_identity_exposed_to_participant_or_manifest",
            "evaluator_truth_exposed_to_participant",
        )
    ):
        errors.append("formal preflight prompt boundary is not fail-closed")
    if report.get("formal_result") is not False:
        errors.append("a preflight cannot be a formal result")
    c2_admission = report.get("c2_admission")
    if not isinstance(c2_admission, Mapping):
        errors.append("formal preflight lacks its C2 admission report")
    else:
        if c2_admission.get("admission_sha256") != c2_admission_sha256(c2_admission):
            errors.append("formal preflight C2 admission self-hash mismatch")
        schedule = c2_admission.get("blocks", {}).get("A_E", {}).get(
            "public_schedule", {}
        )
        ae_cells = [
            cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("c2_locus") == "A_E"
        ]
        if (
            schedule.get("public_schedule_cell_count") != len(ae_cells)
            or schedule.get("public_schedule_sha256")
            != canonical_json_sha256(ae_cells)
        ):
            errors.append("formal preflight C2 admission changed the A_E subblock")
    return errors


def authorize_formal_preflight(
    report: Mapping[str, Any],
    *,
    qualification_receipt: Mapping[str, Any],
    preregistration_freeze_receipt: Mapping[str, Any],
    formal_cost_contract: Mapping[str, Any],
    root: Path | None = None,
) -> dict[str, Any]:
    """Derive the runtime manifest only after every external receipt validates."""

    errors = validate_formal_preflight(report)
    if errors:
        raise ValueError("cannot authorize an invalid formal preflight: " + "; ".join(errors))
    if report.get("formal_execution_allowed") is not False:
        raise ValueError("formal preflight has already crossed the authorization boundary")
    if report.get("prerequisite_errors") != []:
        raise ValueError("formal preflight has unresolved prerequisite failures")
    if report.get("errors") != []:
        raise ValueError("formal preflight has unresolved construction failures")
    c2_admission = report.get("c2_admission")
    c2_admission = c2_admission if isinstance(c2_admission, Mapping) else {}
    if (
        c2_admission.get("status") != "ready_for_formal_authorization"
        or c2_admission.get("formal_execution_allowed") is not True
        or c2_admission.get("blocking_requirements") != []
        or c2_admission.get("evidence_validation_errors") != []
    ):
        raise ValueError("formal preflight lacks complete C2 admission evidence")
    base_hash = report.get("preflight_sha256")
    qualification_hash = canonical_json_sha256(
        {
            key: value
            for key, value in qualification_receipt.items()
            if key != "receipt_sha256"
        }
    )
    cost_hash = canonical_json_sha256(
        {
            key: value
            for key, value in formal_cost_contract.items()
            if key != "formal_cost_contract_sha256"
        }
    )
    freeze_hash = canonical_json_sha256(
        {
            key: value
            for key, value in preregistration_freeze_receipt.items()
            if key != "receipt_sha256"
        }
    )
    freeze_bindings = preregistration_freeze_receipt.get("bindings")
    freeze_bindings = freeze_bindings if isinstance(freeze_bindings, Mapping) else {}
    freeze_qualification = freeze_bindings.get("method_qualification")
    freeze_qualification = (
        freeze_qualification if isinstance(freeze_qualification, Mapping) else {}
    )
    if (
        qualification_receipt.get("schema_version")
        != "chemworld-work-ii-method-qualification-receipt-0.4"
        or qualification_receipt.get("status") != "passed"
        or qualification_receipt.get("formal_execution_authorized") is not False
        or qualification_receipt.get("receipt_sha256") != qualification_hash
        or formal_cost_contract.get("schema_version")
        != "chemworld-work-ii-formal-cost-contract-0.1"
        or formal_cost_contract.get("formal_preflight_sha256") != base_hash
        or formal_cost_contract.get("formal_cost_contract_sha256") != cost_hash
        or preregistration_freeze_receipt.get("schema_version")
        != "chemworld-work-ii-preregistration-freeze-receipt-0.2"
        or preregistration_freeze_receipt.get("status") != "passed_final_freeze"
        or preregistration_freeze_receipt.get("formal_execution_authorized") is not True
        or preregistration_freeze_receipt.get("receipt_sha256") != freeze_hash
        or freeze_bindings.get("formal_preflight_sha256") != base_hash
        or freeze_qualification.get("receipt_sha256") != qualification_hash
        or freeze_qualification.get("manifest_sha256")
        != qualification_receipt.get("qualification_manifest_sha256")
        or preregistration_freeze_receipt.get("formal_currency_budget")
        != formal_cost_contract
    ):
        raise ValueError("formal authorization evidence is invalid or cross-bound incorrectly")
    if root is not None:
        from chemworld.eval.provenance import git_worktree_dirty

        if git_worktree_dirty(root.resolve()):
            raise ValueError("formal authorization requires a clean immutable worktree")
    authorized = dict(report)
    authorized["status"] = "passed_execution_authorized"
    authorized["formal_execution_allowed"] = True
    authorized["blocking_requirements"] = []
    authorized["authorization_bindings"] = {
        "base_preflight_sha256": report.get("preflight_sha256"),
        "qualification_receipt_sha256": qualification_hash,
        "preregistration_freeze_receipt_sha256": freeze_hash,
        "formal_cost_contract_sha256": cost_hash,
        "c2_admission_sha256": c2_admission["admission_sha256"],
    }
    authorized["preflight_sha256"] = _self_hash(authorized)
    authorized_errors = validate_formal_preflight(authorized)
    if authorized_errors:
        raise ValueError(
            "built formal execution manifest is invalid: " + "; ".join(authorized_errors)
        )
    return authorized


def validate_formal_bindings(root: Path, report: Mapping[str, Any]) -> list[str]:
    """Verify every file binding carried by a committed formal preflight."""

    root = root.resolve()
    errors = validate_formal_preflight(report)
    c2 = report.get("c2_admission")
    c2 = c2 if isinstance(c2, Mapping) else {}
    c2_plan_binding = c2.get("plan_binding")
    c2_plan_binding = (
        c2_plan_binding if isinstance(c2_plan_binding, Mapping) else {}
    )
    design_binding = report.get("design_binding")
    design_binding = design_binding if isinstance(design_binding, Mapping) else {}
    c2_plan_path = root / str(c2_plan_binding.get("path", ""))
    design_path = root / str(design_binding.get("path", ""))
    cells = report.get("cells")
    cells = cells if isinstance(cells, list) else []
    if c2_plan_path.is_file() and design_path.is_file():
        ae_cells = [
            cell
            for cell in cells
            if isinstance(cell, Mapping) and cell.get("c2_locus") == "A_E"
        ]
        errors.extend(
            validate_c2_admission_report(
                root,
                c2,
                c2_plan_path,
                design_path,
                ae_cells,
            )
        )
    else:
        errors.append("formal preflight C2 admission bindings are missing")
    bindings: list[Mapping[str, Any]] = []
    for name in ("design_binding", "analysis_binding"):
        candidate = report.get(name)
        if isinstance(candidate, Mapping):
            bindings.append(candidate)
        else:
            errors.append(f"formal preflight lacks {name}")
    runtime_manifest: dict[str, Any] | None = None
    runtime_manifest_path: Path | None = None
    runtime_binding = report.get("formal_runtime_manifest_binding")
    if isinstance(runtime_binding, Mapping):
        bindings.append(runtime_binding)
        relative = runtime_binding.get("path")
        if isinstance(relative, str):
            candidate_path = (root / relative).resolve()
            if candidate_path.is_relative_to(root) and candidate_path.is_file():
                runtime_manifest_path = candidate_path
                try:
                    runtime_manifest = _load_object(candidate_path)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    errors.append(
                        "formal runtime manifest cannot be loaded during binding validation: "
                        f"{type(error).__name__}"
                    )
                else:
                    errors.extend(
                        f"formal runtime manifest: {item}"
                        for item in validate_formal_runtime_manifest(
                            root,
                            runtime_manifest,
                            manifest_path=candidate_path,
                        )
                    )
    elif report.get("participant_execution_contract") == EXPECTED_PARTICIPANT_EXECUTION_CONTRACT:
        errors.append("formal preflight lacks its formal runtime manifest binding")
    rows = report.get("task_bindings")
    if not isinstance(rows, list):
        errors.append("formal preflight lacks task_bindings")
    else:
        for row in rows:
            if not isinstance(row, Mapping):
                errors.append("formal preflight task_bindings contains a malformed row")
                continue
            candidate = row.get("campaign_config")
            if isinstance(candidate, Mapping):
                bindings.append(candidate)
            else:
                errors.append("formal preflight task_bindings contains a malformed binding")
            if row.get("c2_locus") in C2_LOCI:
                for field, embedded_field, hash_function in (
                    (
                        "task_admission_receipt",
                        "receipt_sha256",
                        c2_task_admission_receipt_sha256,
                    ),
                    (
                        "outcome_blind_selection",
                        "selection_sha256",
                        c2_outcome_blind_selection_sha256,
                    ),
                ):
                    extra = row.get(field)
                    if not isinstance(extra, Mapping):
                        errors.append(f"formal C2 task lacks {field}")
                        continue
                    _, extra_errors = _bound_object(
                        root,
                        extra,
                        label=f"formal C2 task {row.get('task_binding_key')} {field}",
                        embedded_field=embedded_field,
                        embedded_hash=hash_function,
                    )
                    errors.extend(extra_errors)
    seen: dict[str, str] = {}
    for binding in bindings:
        relative = binding.get("path")
        digest = binding.get("sha256")
        hash_kind = binding.get("hash_kind")
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or hash_kind not in {"file_sha256", "canonical_json_sha256"}
        ):
            errors.append("formal preflight contains an incomplete file binding")
            continue
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"formal binding escapes the repository: {relative}")
            continue
        if relative in seen and seen[relative] != digest:
            errors.append(f"formal binding has conflicting digests: {relative}")
            continue
        seen[relative] = digest
        if not path.is_file():
            errors.append(f"formal binding is missing: {relative}")
        else:
            actual = (
                file_sha256(path)
                if hash_kind == "file_sha256"
                else canonical_json_sha256(_load_object(path))
            )
            if actual != digest:
                errors.append(f"formal binding digest mismatch: {relative}")
    for cell in report.get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        relative = cell.get("campaign_config_path")
        digest = cell.get("campaign_config_sha256")
        if not isinstance(relative, str) or seen.get(relative) != digest:
            errors.append(f"formal cell campaign binding mismatch: {cell.get('cell_id')}")
    if runtime_manifest is not None and runtime_manifest_path is not None:
        runtime_rows = runtime_manifest.get("task_configs")
        runtime_rows = runtime_rows if isinstance(runtime_rows, list) else []
        runtime_by_key = {
            formal_task_binding_key(str(row.get("locus")), str(row.get("task_id"))): row
            for row in runtime_rows
            if isinstance(row, Mapping)
        }
        scheduled_keys: set[str] = set()
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, Mapping):
                continue
            task_key = row.get("task_binding_key")
            if not isinstance(task_key, str):
                continue
            scheduled_keys.add(task_key)
            runtime_row = runtime_by_key.get(task_key)
            runtime_config = (
                runtime_row.get("formal_campaign_config_binding")
                if isinstance(runtime_row, Mapping)
                else None
            )
            campaign_config = row.get("campaign_config")
            runtime_path = (
                (root / str(runtime_config.get("path"))).resolve()
                if isinstance(runtime_config, Mapping)
                else None
            )
            runtime_digest = (
                canonical_json_sha256(_load_object(runtime_path))
                if runtime_path is not None
                and runtime_path.is_relative_to(root)
                and runtime_path.is_file()
                else None
            )
            if (
                not isinstance(runtime_config, Mapping)
                or not isinstance(campaign_config, Mapping)
                or runtime_config.get("path") != campaign_config.get("path")
                or runtime_digest != runtime_config.get("canonical_json_sha256")
            ):
                errors.append(
                    f"formal task does not consume its materialized runtime config: {task_key}"
                )
        schedule = report.get("schedule_policy")
        schedule = schedule if isinstance(schedule, Mapping) else {}
        if schedule.get("schedule_complete") is True and scheduled_keys != set(runtime_by_key):
            errors.append("complete formal schedule does not consume all nine runtime configs")
    return errors


__all__ = [
    "EXPECTED_LAW_SUMMARY_EVALUATION_CONTRACT",
    "EXPECTED_METHOD_QUALIFICATION_CONTRACT",
    "EXPECTED_PARTICIPANT_EXECUTION_CONTRACT",
    "EXPECTED_PRIVATE_CONFIRMATION_CONTRACT",
    "EXPECTED_REFERENCE_POLICY_CONTRACT",
    "FORMAL_ARMS",
    "FORMAL_BLOCKING_REQUIREMENTS",
    "FORMAL_C2_LOCI",
    "FORMAL_C2_LOCUS_CONTRACT",
    "FORMAL_CELL_VERSION",
    "FORMAL_CHECKPOINT_EXPERIMENTS",
    "FORMAL_EXPECTED_TOTALS",
    "FORMAL_PREFLIGHT_VERSION",
    "FORMAL_RECEIPT_VERSION",
    "FORMAL_SNAPSHOT_STAGES",
    "FORMAL_STORE_AUDIT_VERSION",
    "FORMAL_TERMINAL_STATES",
    "DuplicateFormalCellError",
    "InvalidFormalCellReceiptError",
    "ProviderAttemptLimitError",
    "WorkIIFormalCellStore",
    "authorize_formal_preflight",
    "build_checkpoint_contract",
    "build_formal_preflight",
    "formal_task_binding_key",
    "validate_formal_bindings",
    "validate_formal_preflight",
]
