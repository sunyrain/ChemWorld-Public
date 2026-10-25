"""Fail-closed semantic audit for every confirmatory benchmark component."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.eval.mechanism_adaptation import (
    build_paired_campaign_matrix,
    validate_mechanism_adaptation_protocol,
)
from chemworld.eval.mechanism_relation_graph import (
    validate_diagnostic_relation_graph,
)
from chemworld.eval.provenance import canonical_json_sha256
from chemworld.tasks import CONFIRMATORY_BENCHMARK_TASK_IDS

CONFIRMATORY_TASK_SEMANTICS_AUDIT_VERSION = (
    "chemworld-confirmatory-task-semantics-audit-0.2"
)


def audit_confirmatory_task_semantics(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    relation_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit controls, estimands, blinding, cohorts, and frozen decisions."""

    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check": check_id, "pass": bool(passed), "detail": detail})

    validation_errors = validate_mechanism_adaptation_protocol(protocol)
    add(
        "protocol_schema_valid",
        not validation_errors,
        f"validation_errors={validation_errors}",
    )
    protocol_tasks = tuple(str(item) for item in protocol["design"]["tasks"])
    add(
        "confirmatory_benchmark_task_scope_explicit",
        set(protocol_tasks) == set(CONFIRMATORY_BENCHMARK_TASK_IDS),
        (
            f"formal_protocol_tasks={protocol_tasks}; "
            f"registered_confirmatory_tasks={CONFIRMATORY_BENCHMARK_TASK_IDS}"
        ),
    )

    calibrated = protocol["evaluation_tracks"]["calibrated_online_change"]
    requirement = plan["online_attainability_certificate"]
    add(
        "a3_certifies_reference_policy_attainability_not_participant_agent",
        requirement.get("scientific_name")
        == "A3_online_attainability_certificate"
        and requirement.get("certification_subject")
        == "frozen_reference_diagnostic_policy"
        and requirement.get("participant_agent_evaluation") is False
        and protocol["gates"]["gate_b"].get("evaluation_subject")
        == "participant_agent",
        (
            f"a3_subject={requirement.get('certification_subject')}; "
            f"gate_b_subject={protocol['gates']['gate_b'].get('evaluation_subject')}"
        ),
    )
    add(
        "a3_true_no_change_condition",
        calibrated["truth_change_time_support"] == ["never", 6, 8, 10]
        and requirement["truth_change_time_support"] == ["never", 6, 8, 10]
        and requirement["no_change_condition_included"] is True,
        "The calibrated truth support must contain never, 6, 8, and 10.",
    )
    pairs: dict[str, set[str]] = {}
    pair_clusters: dict[str, set[str]] = {}
    matrix_error: str | None = None
    try:
        matrix = build_paired_campaign_matrix(protocol)
    except ValueError as error:
        matrix = []
        matrix_error = str(error)
    for row in matrix:
        pair_id = str(row["pair_id"])
        pairs.setdefault(pair_id, set()).add(str(row["arm"]))
        pair_clusters.setdefault(pair_id, set()).add(
            str(row["statistical_cluster_id"])
        )
    add(
        "a3_no_change_campaigns_are_matched",
        bool(pairs)
        and all(arms == {"changed", "no_change_twin"} for arms in pairs.values()),
        f"paired_cell_count={len(pairs)}; matrix_error={matrix_error}",
    )
    add(
        "a3_changed_and_no_change_twins_share_statistical_cluster",
        bool(pair_clusters)
        and all(len(cluster_ids) == 1 for cluster_ids in pair_clusters.values()),
        (
            f"paired_cell_count={len(pair_clusters)}; "
            "each changed/no-change pair must contribute one dependent cluster"
        ),
    )
    pairing = plan["paired_phase_design"]
    add(
        "a3_no_change_twins_match_runtime_and_noise_stream",
        pairing.get("common_random_numbers_across_changed_no_change_twins")
        is True
        and pairing.get("identical_reset_rule_across_changed_no_change_twins")
        is True
        and pairing.get("reset_or_instance_identifier_agent_visible") is False
        and pairing.get("pseudo_checkpoint_has_runtime_side_effect") is False
        and pairing.get("campaign_metadata_invariant_across_checkpoint") is True,
        "Changed/never twins differ only by the hidden physical-law intervention.",
    )

    reference = calibrated["reference_sufficiency"]
    add(
        "reference_has_structural_and_predictive_layers",
        reference["basis"]
        == "universal_structural_relation_coverage_and_held_out_predictive_adequacy"
        and reference["relation_closure_controls_certificate"] is True
        and reference["canonical_witness_action_set_controls_certificate"]
        is False
        and reference["all_declared_relations_observed"] is True
        and reference["held_out_old_world_predictive_check_required"] is True,
        f"reference_basis={reference['basis']}",
    )
    add(
        "reference_is_universal_not_truth_conditioned",
        reference["scope"]
        == "universal_all_declared_candidate_families_without_truth_conditioning"
        and requirement["reference_scope"] == reference["scope"],
        f"reference_scope={reference['scope']}",
    )
    predictive_reference = requirement["reference_predictive_adequacy"]
    add(
        "reference_predictive_check_is_within_campaign_cross_fitted",
        predictive_reference.get("cross_fitting")
        == "leave_one_experiment_out_within_campaign_pre_change_only"
        and predictive_reference.get("post_change_observations_allowed")
        is False
        and predictive_reference.get("truth_family_allowed") is False,
        str(predictive_reference.get("source")),
    )
    primary_metrics = set(calibrated["primary_metrics"])
    add(
        "conditional_and_end_to_end_metrics_both_frozen",
        {
            "family_attribution_conditional_on_reference_sufficiency",
            "p_reference_sufficient",
            "p_detection_given_reference_and_changed",
            "p_attribution_given_detection_reference_and_changed",
            "end_to_end_reference_detection_attribution_given_changed",
            "p_no_false_alarm_given_reference_and_never",
            "fpr_horizon_given_never",
        }.issubset(primary_metrics),
        f"primary_metrics={sorted(primary_metrics)}",
    )
    add(
        "changed_and_never_estimands_are_separate",
        "p_attribution_given_detection_reference_and_changed"
        in primary_metrics
        and "p_no_false_alarm_given_reference_and_never"
        in primary_metrics
        and "fpr_horizon_given_never" in primary_metrics,
        "Attribution is defined only for changed campaigns; never controls false alarms.",
    )

    hidden_timing = set(calibrated["agent_visible_timing_contract"]["hidden"])
    add(
        "agent_timing_and_certificate_blinded",
        {
            "minimum_stable_prefix_experiments",
            "change_after_experiments",
            "truth_change_time",
            "reference_sufficiency_certificate",
        }.issubset(hidden_timing)
        and requirement["policy_receives_change_time_support"] is False
        and requirement["policy_receives_minimum_stable_prefix"] is False
        and requirement["policy_receives_reference_certificate"] is False,
        f"hidden_agent_fields={sorted(hidden_timing)}",
    )
    add(
        "changepoint_index_semantics_frozen",
        requirement["changepoint_semantics"]
        == "tau_is_completed_old_world_experiment_count"
        and calibrated["total_experiment_horizon"] >= 18,
        calibrated["changepoint_semantics"],
    )

    partitions = plan["cohort_partition"]
    cohort_namespaces = {
        cohort_id: int(cohort["seed_namespace_start"])
        for cohort_id, cohort in partitions.items()
        if isinstance(cohort, Mapping) and "seed_namespace_start" in cohort
    }
    add(
        "development_a2_a3_private_cohorts_disjoint",
        set(cohort_namespaces)
        in (
            {
                "development",
                "a2_certification",
                "a3_certification",
                "private_confirmation",
            },
            {
                "development",
                "a2_certification",
                "a3_certification",
                "private_environment_confirmation",
                "private_agent_confirmation",
            },
        )
        and len(set(cohort_namespaces.values())) == len(cohort_namespaces)
        and partitions["cross_cohort_world_seed_reuse_allowed"] is False,
        f"seed_namespaces={cohort_namespaces}",
    )
    frozen_criteria = requirement["frozen_pass_criteria"]
    add(
        "a3_pass_criteria_frozen_before_execution",
        {
            "minimum_reference_acquisition_rate_wilson_lower_bound",
            "minimum_changed_detection_recall_cluster_bootstrap_lower_bound",
            "maximum_no_change_false_positive_rate_cluster_bootstrap_upper_bound",
            "minimum_change_detection_auroc_cluster_bootstrap_lower_bound",
            "maximum_integrated_mean_change_probability_brier_score",
            "minimum_attribution_given_detection_and_reference_cluster_bootstrap_lower_bound",
            "minimum_end_to_end_success_rate_cluster_bootstrap_lower_bound",
        }.issubset(frozen_criteria),
        f"criteria={sorted(frozen_criteria)}",
    )
    add(
        "time_resolved_detection_and_right_censoring_frozen",
        frozen_criteria.get(
            "detection_checkpoint_experiments_after_change"
        )
        == [1, 2, 4, 8]
        and frozen_criteria.get(
            "primary_detection_checkpoint_experiments_after_change"
        )
        == 8
        and frozen_criteria.get("undetected_campaign_handling")
        == "right_censored_at_k_8"
        and "equal-weight mean"
        in str(
            frozen_criteria.get(
                "change_probability_brier_aggregation"
            )
        ),
        "AUROC, Brier, recall, horizon FPR, and delay use frozen relative checkpoints.",
    )
    add(
        "a3_gate_intersects_tasks_and_families",
        frozen_criteria.get("gate_aggregation")
        == "intersection_of_overall_each_task_and_each_declared_changed_family"
        and frozen_criteria.get("pooled_micro_average_controls_gate")
        is False,
        str(frozen_criteria.get("gate_aggregation")),
    )
    sample_size = plan.get("sample_size_audit")
    preregistration = plan.get("preregistration")
    add(
        "sample_size_and_preregistration_declared",
        isinstance(sample_size, Mapping)
        and sample_size.get("required") is True
        and int(sample_size.get("recommended_world_seeds_per_family", 0))
        == int(requirement["world_seeds_per_family"])
        and isinstance(preregistration, Mapping)
        and preregistration.get("required_before_a2_or_a3_execution")
        is True,
        (
            f"world_seeds_per_family={requirement['world_seeds_per_family']}; "
            "manifest="
            f"{preregistration.get('manifest') if isinstance(preregistration, Mapping) else None}"
        ),
    )

    feedback = protocol["paired_feedback_design"]
    local_feedback = feedback["local_prefix_reaction_test"]
    campaign_feedback = feedback["full_campaign_utility_test"]
    add(
        "feedback_local_causal_pairing",
        local_feedback["same_public_history_prefix"] is True
        and local_feedback["only_last_feedback_packet_changes"] is True
        and local_feedback["provider_repeats_required"] is True,
        "Local feedback sensitivity changes only the final feedback packet.",
    )
    add(
        "feedback_campaign_pairing",
        campaign_feedback["paired_world_seed"] is True
        and campaign_feedback["paired_initial_state"] is True
        and campaign_feedback["paired_prompt_and_model_settings"] is True,
        "Full-campaign utility uses matched world, state, prompt, and model settings.",
    )

    recovery = protocol["recovery_design"]
    add(
        "recovery_world_effect_and_adaptation_decomposed",
        all(
            recovery.get(key) is True
            for key in (
                "open_loop_iid_action_replay_on_iid_world",
                "open_loop_iid_action_replay_on_shifted_world",
                "frozen_policy_on_shifted_world",
                "adaptive_policy_on_shifted_world",
                "diagnosis_oracle_on_shifted_world",
            )
        )
        and "world_effect" in recovery
        and "adaptation_gain" in recovery,
        "Recovery separates world effect, frozen policy, adaptive policy, and oracle.",
    )
    autonomy = protocol["autonomy_contract"]
    add(
        "autonomy_and_assisted_science_both_reported",
        autonomy["both_scores_mandatory"] is True
        and "autonomous_score" in autonomy
        and "assisted_scientific_score" in autonomy,
        "Procedural autonomy cannot be hidden by assisted closeout.",
    )
    add(
        "diagnostic_relation_graph_frozen_and_bound",
        not validate_diagnostic_relation_graph(protocol, plan, relation_graph),
        (
            f"graph_sha256={relation_graph.get('graph_sha256')}; "
            f"relation_count={relation_graph.get('relation_count')}"
        ),
    )
    add(
        "publication_claim_fails_closed_before_empirical_gates",
        protocol["publication_ready"] is False
        and protocol["benchmark_claim_allowed"] is False
        and plan["formal_benchmark_result"] is False,
        "Design validation cannot enable a benchmark or publication claim.",
    )

    failures = [item for item in checks if not item["pass"]]
    return {
        "schema_version": CONFIRMATORY_TASK_SEMANTICS_AUDIT_VERSION,
        "status": "passed" if not failures else "failed",
        "pass": not failures,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_json_sha256(protocol),
        "gate_a_plan_id": plan["plan_id"],
        "gate_a_plan_sha256": canonical_json_sha256(plan),
        "confirmatory_benchmark_task_ids": list(protocol_tasks),
        "scope": (
            "all mechanism-adaptation confirmatory benchmark components: "
            "integrity, identifiability, change detection, feedback causality, "
            "recovery, and autonomy"
        ),
        "check_count": len(checks),
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "claim_boundary": (
            "A pass certifies protocol semantics and frozen controls only. It does "
            "not certify A2, A3, any participant Agent, or publication readiness."
        ),
    }


__all__ = [
    "CONFIRMATORY_TASK_SEMANTICS_AUDIT_VERSION",
    "audit_confirmatory_task_semantics",
]
