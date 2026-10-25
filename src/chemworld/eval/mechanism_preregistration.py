"""Single-source preregistration manifest for mechanism adaptation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
)

PREREGISTRATION_VERSION = "chemworld-mechanism-preregistration-0.1"
RC24_SCORER_SOURCE_PATHS = (
    "src/chemworld/eval/mechanism_adaptation.py",
    "src/chemworld/eval/mechanism_adaptation_execution.py",
    "src/chemworld/eval/mechanism_gate_decision.py",
    "src/chemworld/eval/mechanism_design_audit.py",
    "src/chemworld/eval/mechanism_relation_graph.py",
)
SCORER_SOURCE_PATHS = (
    "src/chemworld/agents/prompt_context.py",
    "src/chemworld/envs/observation_noise.py",
    "src/chemworld/eval/mechanism_adaptation.py",
    "src/chemworld/eval/mechanism_adaptation_execution.py",
    "src/chemworld/eval/mechanism_gate_decision.py",
    "src/chemworld/eval/mechanism_design_audit.py",
    "src/chemworld/eval/mechanism_release.py",
    "src/chemworld/eval/mechanism_relation_graph.py",
    "src/chemworld/eval/trial_store.py",
)


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    return canonical_json_sha256(payload)


def build_mechanism_preregistration(
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    relation_graph: Mapping[str, Any],
    sample_size_audit: Mapping[str, Any],
    source_commit: str,
) -> dict[str, Any]:
    """Build the immutable A2/A3 execution contract."""

    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise ValueError("source_commit must be a full lowercase Git commit SHA")
    requirement = plan["online_attainability_certificate"]
    criteria = requirement["frozen_pass_criteria"]
    calibrated = protocol["evaluation_tracks"]["calibrated_online_change"]
    release_candidate_match = re.search(
        r"(rc[0-9]+)$",
        str(plan["plan_id"]),
    )
    if release_candidate_match is None:
        raise ValueError("plan_id must end in an RC identifier")
    release_candidate = release_candidate_match.group(1)
    scorer_paths = (
        RC24_SCORER_SOURCE_PATHS
        if release_candidate == "rc24"
        else SCORER_SOURCE_PATHS
    )
    scorer_files = {
        relative: file_sha256(repository_root / relative)
        for relative in scorer_paths
    }
    protocol_source = str(
        protocol.get("protocol_composition", {}).get(
            "source",
            "mechanism_adaptation_v0.3.0.json",
        )
    )
    plan_source = str(
        plan.get("plan_composition", {}).get(
            "source",
            "mechanism_adaptation_gate_a_v0.3.0.json",
        )
    )
    manifest: dict[str, Any] = {
        "schema_version": PREREGISTRATION_VERSION,
        "manifest_id": (
            f"chemworld-mechanism-adaptation-v0.3.0-{release_candidate}-"
            "preregistration"
        ),
        "status": "locked_before_a2_a3_execution",
        "source_commit": source_commit,
        "protocol": {
            "protocol_id": protocol["protocol_id"],
            "canonical_sha256": canonical_json_sha256(protocol),
            "source_path": f"configs/benchmark/{protocol_source}",
        },
        "gate_a_plan": {
            "plan_id": plan["plan_id"],
            "canonical_sha256": canonical_json_sha256(plan),
            "source_path": f"configs/benchmark/{plan_source}",
        },
        "diagnostic_relation_graph": {
            "path": plan["diagnostic_relation_graph"]["report"],
            "graph_sha256": relation_graph["graph_sha256"],
            "relation_count": relation_graph["relation_count"],
        },
        "scorer": {
            "source_file_sha256": scorer_files,
            "source_bundle_sha256": canonical_json_sha256(scorer_files),
        },
        "certification_subjects": {
            "a1": "physical_world_and_intervention_contract",
            "a2": "controlled_oracle_and_decoder",
            "a3": "frozen_reference_diagnostic_policy",
            "gates_b_to_e": "participant_agent",
        },
        "reference_policy": {
            "version": requirement["certificate_schema_version"],
            "action_design": plan["action_library"]["design"],
            "action_design_seed": plan["action_library"]["design_seed"],
            "action_count_per_task": plan["action_library"][
                "action_count_per_task"
            ],
            "certificate_basis": "relation_closure_not_recipe_id_closure",
            "predictive_reference": requirement[
                "reference_predictive_adequacy"
            ],
        },
        "cohorts": plan["cohort_partition"],
        "sample_size": {
            "audit_path": plan["sample_size_audit"]["report"],
            "audit_sha256": sample_size_audit["audit_sha256"],
            "independent_world_clusters_per_family": requirement[
                "world_seeds_per_family"
            ],
            "clusters_per_changed_family_per_change_time": (
                sample_size_audit[
                    "clusters_per_changed_family_per_change_time"
                ]
            ),
            "no_change_clusters_per_task": sample_size_audit[
                "no_change_clusters_per_task"
            ],
            "provider_repeats_per_paired_cell": protocol["design"][
                "provider_repeats_per_paired_cell"
            ],
            "provider_repeats_are_independent_samples": False,
        },
        "timing": {
            "truth_change_time_support": requirement[
                "truth_change_time_support"
            ],
            "changepoint_semantics": requirement[
                "changepoint_semantics"
            ],
            "total_experiment_horizon": calibrated[
                "total_experiment_horizon"
            ],
            "post_change_checkpoints": criteria[
                "detection_checkpoint_experiments_after_change"
            ],
            "primary_post_change_checkpoint": criteria[
                "primary_detection_checkpoint_experiments_after_change"
            ],
            "decision_threshold": criteria[
                "change_decision_probability_threshold"
            ],
            "detection_event": criteria["detection_event"],
            "undetected_campaign_handling": criteria[
                "undetected_campaign_handling"
            ],
        },
        "statistics": {
            "confidence_level": criteria["confidence_level"],
            "cluster_unit": "task_id_and_world_seed",
            "provider_repeat_role": "nested_technical_replicate",
            "cluster_bootstrap_draws": criteria[
                "cluster_bootstrap_draws"
            ],
            "cluster_bootstrap_seed_rule": criteria[
                "cluster_bootstrap_seed_rule"
            ],
            "brier_aggregation": criteria[
                "change_probability_brier_aggregation"
            ],
            "thresholds": {
                key: value
                for key, value in criteria.items()
                if key.startswith("minimum_")
                or key.startswith("maximum_")
            },
            "aggregation_rule": criteria["gate_aggregation"],
            "pooled_micro_average_controls_gate": criteria[
                "pooled_micro_average_controls_gate"
            ],
            "multiple_comparison_policy": protocol["reporting"][
                "multiple_comparison_policy"
            ],
        },
        "paired_no_change_control": {
            key: plan["paired_phase_design"][key]
            for key in (
                "same_hidden_world_seed_across_phases",
                "same_world_seed_reused_across_candidate_twins",
                "common_random_numbers_across_changed_no_change_twins",
                "identical_reset_rule_across_changed_no_change_twins",
                "reset_or_instance_identifier_agent_visible",
                "pseudo_checkpoint_has_runtime_side_effect",
                "campaign_metadata_invariant_across_checkpoint",
            )
        },
        "execution_handling": protocol["execution_handling"],
        "stopping_rule": protocol["execution_handling"]["stopping_rule"],
        "private_confirmation_unseal_condition": protocol[
            "execution_handling"
        ]["private_confirmation_unseal"],
        "state_at_lock": protocol["protocol_state_machine"],
        "mutation_rule": (
            "Any change to a bound hash, threshold, cohort, seed count, scorer, "
            "policy, exclusion, or stopping rule creates a new protocol RC and "
            "cannot reinterpret results collected under this manifest."
        ),
        "publication_ready": False,
    }
    if release_candidate != "rc24":
        manifest["release_candidate"] = release_candidate
        manifest["observation_noise"] = protocol.get(
            "paired_observation_noise"
        )
        manifest["participant_prompt_contract"] = protocol.get(
            "participant_agent_prompt_contract"
        )
        manifest["release_qualification"] = plan.get(
            "release_qualification"
        )
        manifest["analysis_artifacts"] = {
            "decision_and_readiness_state_machine": (
                "src/chemworld/eval/mechanism_release.py"
            ),
            "bootstrap_estimands_tables_and_right_censoring": (
                "src/chemworld/eval/mechanism_adaptation_execution.py"
            ),
            "trial_identity_resume_and_reason_domains": (
                "src/chemworld/eval/trial_store.py"
            ),
        }
    manifest["manifest_sha256"] = _manifest_sha256(manifest)
    return manifest


def validate_mechanism_preregistration(
    manifest: Mapping[str, Any],
    *,
    repository_root: Path,
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    relation_graph: Mapping[str, Any],
    sample_size_audit: Mapping[str, Any],
) -> list[str]:
    """Return binding errors without changing the locked manifest."""

    errors: list[str] = []
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str):
        return ["preregistration source_commit is missing"]
    try:
        expected = build_mechanism_preregistration(
            repository_root=repository_root,
            protocol=protocol,
            plan=plan,
            relation_graph=relation_graph,
            sample_size_audit=sample_size_audit,
            source_commit=source_commit,
        )
    except (KeyError, TypeError, ValueError) as error:
        return [f"cannot reconstruct preregistration: {error}"]
    if dict(manifest) != expected:
        errors.append(
            "preregistration manifest is stale or differs from its bound inputs"
        )
    if manifest.get("manifest_sha256") != _manifest_sha256(manifest):
        errors.append("preregistration manifest_sha256 is invalid")
    return errors


__all__ = [
    "PREREGISTRATION_VERSION",
    "SCORER_SOURCE_PATHS",
    "build_mechanism_preregistration",
    "validate_mechanism_preregistration",
]
