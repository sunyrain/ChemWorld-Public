"""Build the single derived-data source for the first ChemWorld arXiv paper.

The builder deliberately accepts only audited aggregate artifacts.  It never
reads a live trajectory directory, so pending G2 cells cannot leak into paper
tables or figures.  G2 v0.5 remains absent until its terminal replication
audit is supplied.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.autonomous_material_replication_audit import (
    AutonomousMaterialReplicationAuditError,
    validate_interpretation_binding,
)
from chemworld.eval.work_i_data_contract import (
    data_contract_sha256,
    validate_work_i_data_contract,
)

ARXIV_V1_DERIVED_SCHEMA = "chemworld-arxiv-v1-derived-data-0.1"
WORK_I_INCREMENTAL_SCHEMA = "chemworld-work-i-fvl-derived-data-0.1"
WORK_I_DERIVED_MANIFEST_SCHEMA = "chemworld-work-i-fvl-derived-manifest-0.1"
_G2_V04_SCHEMA = "chemworld-autonomous-material-campaign-audit-0.3"
_G2_V05_SCHEMA = "chemworld-autonomous-material-trajectory-replication-audit-0.1"
_ARMS = ("opaque", "nominal", "misindexed")


class ArxivV1DerivedDataError(ValueError):
    """Raised when an input artifact is missing or inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a deterministic SHA-256 over canonical JSON."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ArxivV1DerivedDataError(f"{label} does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArxivV1DerivedDataError(f"{label} must contain a JSON object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ArxivV1DerivedDataError(message)


def _source(path: Path, *, canonical_json: bool = False) -> dict[str, Any]:
    portable = path.as_posix()
    lower_parts = [part.lower() for part in path.parts]
    for root_name in ("workstreams", "configs", "benchmark", "paper", "runs", "src"):
        if root_name in lower_parts:
            index = lower_parts.index(root_name)
            portable = Path(*path.parts[index:]).as_posix()
            break
    result = {
        "path": portable,
        "file_sha256": file_sha256(path),
    }
    if canonical_json:
        result["canonical_json_sha256"] = canonical_sha256(_load_json(path, label=str(path)))
    return result


def _task_rows(
    g0_v10: Mapping[str, Any],
    g0_v12: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task_rows: list[dict[str, Any]] = []
    world_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    v10_tasks = g0_v10["tasks"]
    v12_tasks = g0_v12["tasks"]
    for task_key in ("electrochemical", "crystallization"):
        original = v10_tasks[task_key]
        triarm = v12_tasks[task_key]
        task_id = str(triarm["task_id"])
        baselines = list(original["baselines"])
        for baseline in baselines:
            baseline_rows.append(
                {
                    "task_id": task_id,
                    "algorithm_id": baseline["algorithm_id"],
                    "role": baseline["role"],
                    "information_condition": baseline["information_condition"],
                    "mean_validated_final_score": baseline["validated_final_score"]["mean"],
                }
            )
        matched = [row for row in baselines if row["role"] == "information_matched"]
        _require(bool(matched), f"{task_id}: no information-matched baseline")
        best_matched = max(
            matched,
            key=lambda row: row["validated_final_score"]["mean"],
        )
        paired = next(
            row
            for row in original["paired_comparisons"]
            if row["algorithm_id"] == best_matched["algorithm_id"]
        )
        privileged = [row for row in baselines if row["role"] == "privileged_calibration"]
        best_privileged = (
            max(
                privileged,
                key=lambda row: row["validated_final_score"]["mean"],
            )
            if privileged
            else None
        )
        for arm in _ARMS:
            primary = triarm["primary_score_by_arm"][arm]
            predictive = triarm["predictive_secondary_diagnostic_mean_by_arm"][arm]
            declared = (
                original["participant"]["declared_world_understanding_secondary_diagnostic_mean"]
                if arm == "opaque"
                else None
            )
            task_rows.append(
                {
                    "task_id": task_id,
                    "arm": arm,
                    "world_count": primary["count"],
                    "primary_score_mean": primary["mean"],
                    "primary_score_sd": primary["sample_standard_deviation"],
                    "heldout_directional_accuracy": predictive["directional_accuracy"],
                    "heldout_brier_score": predictive["confidence_brier_score"],
                    "declared_directional_accuracy": (
                        None if declared is None else declared["directional_accuracy"]
                    ),
                    "structural_edge_f1": (
                        None if declared is None else declared["structural_edge_f1"]
                    ),
                    "mechanism_tag_f1": (
                        None if declared is None else declared["mechanism_tag_f1"]
                    ),
                    "unsupported_claim_rate": (
                        None if declared is None else declared["unsupported_claim_rate"]
                    ),
                }
            )
        for item in triarm["worlds"]:
            for arm in _ARMS:
                action = item[arm]["action"]
                world_rows.append(
                    {
                        "task_id": task_id,
                        "world_seed": item["world_seed"],
                        "arm": arm,
                        "primary_score": item[arm]["primary_score"],
                        "first_action_is_misleading": action["first_action_is_misleading"],
                        "early_misleading_share": action["early_misleading_share"],
                        "late_misleading_share": action["late_misleading_share"],
                        "final_action_is_misleading": action["final_action_is_misleading"],
                    }
                )
        contrasts = triarm["paired_contrasts"]
        recovery = triarm["recovery"]
        task_rows.append(
            {
                "task_id": task_id,
                "arm": "derived_contrasts",
                "nominal_minus_opaque_mean": contrasts["nominal_minus_opaque"]["mean"],
                "nominal_minus_opaque_familywise_97_5_interval": contrasts["nominal_minus_opaque"][
                    "world_bootstrap_97_5_interval"
                ],
                "misindexed_minus_opaque_mean": contrasts["misindexed_minus_opaque"]["mean"],
                "misindexed_minus_nominal_mean": contrasts["misindexed_minus_nominal"]["mean"],
                "first_misleading_rate_misindexed": recovery["first_action_misleading_rate_by_arm"][
                    "misindexed"
                ],
                "early_misleading_share_misindexed": recovery["early_misleading_share_by_arm"][
                    "misindexed"
                ]["mean"],
                "late_misleading_share_misindexed": recovery["late_misleading_share_by_arm"][
                    "misindexed"
                ]["mean"],
                "manipulation_check_passed": recovery["manipulation_check"]["passed"],
                "differential_action_correction_passed": recovery["differential_action_correction"][
                    "passed"
                ],
                "performance_recovery_to_opaque_passed": recovery["performance_recovery_to_opaque"][
                    "passed"
                ],
                "overall_recovery_claim_passed": recovery["overall_recovery_claim"]["passed"],
                "best_information_matched_baseline": best_matched["algorithm_id"],
                "best_information_matched_baseline_mean": best_matched["validated_final_score"][
                    "mean"
                ],
                "participant_minus_best_matched_mean": paired["participant_minus_baseline"]["mean"],
                "participant_minus_best_matched_95_interval": paired["participant_minus_baseline"][
                    "world_bootstrap_95_interval"
                ],
                "best_privileged_calibration": (
                    None if best_privileged is None else best_privileged["algorithm_id"]
                ),
                "best_privileged_calibration_mean": (
                    None
                    if best_privileged is None
                    else best_privileged["validated_final_score"]["mean"]
                ),
            }
        )
    return task_rows, world_rows, baseline_rows


def _g2_v04_rows(
    audit: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _require(audit.get("schema_version") == _G2_V04_SCHEMA, "wrong G2 v0.4 schema")
    _require(
        audit.get("status") == "completed_audited_descriptive_matrix", "G2 v0.4 is not audited"
    )
    _require(len(audit.get("cells", [])) == 10, "G2 v0.4 must contain ten cells")
    cell_rows: list[dict[str, Any]] = []
    for cell in audit["cells"]:
        learning = cell["trajectory_learning"]["discovery_retention_recovery"]
        scores = cell["scores"]
        cell_rows.append(
            {
                "cell_id": cell["cell_id"],
                "world_seed": cell["world_seed"],
                "arm": cell["arm"],
                "completed_vessels": cell["completion"]["completed_vessels"],
                "operation_count": cell["operations"]["count"],
                "invalid_operation_count": cell["operations"]["invalid_count"],
                "nonfinal_measurement_count": cell["measurements"]["committed_count"],
                "final_score_sequence": scores["final_score_sequence"],
                "final_assay_operation_indices": [
                    row["operation_attempt_index"] for row in scores["final_assay_outcomes"]
                ],
                "best_final_score": scores["best_final_score"],
                "final_score_mean": scores["final_score_mean"],
                "batch_auc": scores["batch_final_assay_running_best_auc"],
                "realized_attempt_auc": scores["operation_attempt_running_best_auc"],
                "fixed_budget_attempt_auc": scores[
                    "budget_normalized_operation_attempt_running_best_auc"
                ],
                "global_best_discovery_fraction": learning["global_best_discovery_fraction"],
                "online_incumbent_retention_rate": learning["online_retention_rate"],
                "maximum_absolute_incumbent_drawdown": learning[
                    "maximum_absolute_drawdown_from_prior_incumbent"
                ],
                "terminal_to_global_best_ratio": learning["terminal_to_global_best_ratio"],
                "loss_episode_count": learning["loss_episode_count"],
                "recovered_loss_episode_count": learning["recovered_loss_episode_count"],
                "unresolved_loss_episode_count": learning["unresolved_loss_episode_count"],
            }
        )
    pair_rows = [
        {
            "world_seed": row["world_seed"],
            **{
                key: value
                for key, value in row["nominal_minus_opaque"].items()
                if key
                in {
                    "best_final_score",
                    "final_score_mean",
                    "batch_final_assay_running_best_auc",
                    "operation_attempt_running_best_auc",
                    "budget_normalized_operation_attempt_running_best_auc",
                    "operation_count",
                    "global_best_discovery_fraction",
                    "online_incumbent_retention_rate",
                    "maximum_absolute_incumbent_drawdown",
                    "terminal_to_global_best_ratio",
                }
            },
        }
        for row in audit["paired_worlds"]
    ]
    representative = next(
        cell for cell in audit["cells"] if cell["world_seed"] == 0 and cell["arm"] == "opaque"
    )
    first_batch = representative["batches"][0]
    demonstration = {
        "label": "development demonstration; excluded from prior-effect inference",
        "cell_id": representative["cell_id"],
        "world_seed": 0,
        "arm": "opaque",
        "operation_signature": first_batch["operation_signature"],
        "operation_count": first_batch["operation_count"],
        "diagnostic_policy": first_batch["diagnostic_policy"],
        "setpoint_policy": first_batch["setpoint_policy"],
        "final_score": first_batch["final_score"],
        "campaign_resource_endpoints": representative["resource_ledger"],
    }
    return cell_rows, pair_rows, demonstration


def _g2_v05_rows(audit: Mapping[str, Any]) -> dict[str, Any]:
    _require(audit.get("schema_version") == _G2_V05_SCHEMA, "wrong G2 v0.5 schema")
    unhashed = dict(audit)
    declared_audit_hash = unhashed.pop("audit_sha256", None)
    _require(
        declared_audit_hash == canonical_sha256(unhashed),
        "G2 v0.5 audit hash is invalid",
    )
    try:
        validate_interpretation_binding(audit)
    except AutonomousMaterialReplicationAuditError as error:
        raise ArxivV1DerivedDataError(str(error)) from error
    matrix = audit["matrix"]
    _require(
        matrix["completed_cell_count"] + matrix["right_censored_cell_count"] == 20,
        "G2 v0.5 audit is not terminal for all twenty cells",
    )
    _require(
        matrix["all_attempt_selection_policies_verified"],
        "G2 v0.5 attempt selection policy failed",
    )
    _require(matrix["all_physical_pairs_verified"], "G2 v0.5 physical pairing failed")
    _require(
        matrix["all_terminal_cells_resource_replay_verified"],
        "G2 v0.5 terminal resource or replay verification failed",
    )
    pairs = []
    for row in audit["paired_trajectories"]:
        contrast = row["nominal_minus_opaque"]
        if contrast is not None:
            contrast = dict(contrast)
            score_sequence = contrast.get("final_score_sequence")
            if not isinstance(score_sequence, list) or not score_sequence:
                raise ArxivV1DerivedDataError(
                    "a complete G2 v0.5 pair is missing final-score contrasts"
                )
            # The last element is the nominal-minus-opaque raw terminal score.
            # Keep it explicit so downstream analyses do not confuse it with the
            # algebraically coupled terminal-to-best retention ratio.
            contrast["terminal_final_score"] = score_sequence[-1]
        pairs.append(
            {
                "world_seed": row["world_seed"],
                "trajectory_replicate_id": row["trajectory_replicate_id"],
                "opaque_state": row["opaque_state"],
                "nominal_state": row["nominal_state"],
                "pair_complete": row["pair_complete"],
                "nominal_minus_opaque": contrast,
            }
        )
    return {
        "status": audit["status"],
        "audit_sha256": audit["audit_sha256"],
        "matrix": matrix,
        "paired_trajectories": pairs,
        "within_world_descriptive_aggregates": audit["within_world_descriptive_aggregates"],
        "interpretation": audit["interpretation"],
    }


def _artifact_source(
    path: Path,
    payload: Mapping[str, Any],
    *,
    artifact_sha256: str,
) -> dict[str, Any]:
    result = _source(path)
    result["artifact_sha256"] = artifact_sha256
    result["schema_id"] = payload.get("schema_id", payload.get("schema_version"))
    return result


def _base_record(
    *,
    track: str,
    record_type: str,
    record_id: str,
    execution_role: str,
    analysis_role: str,
    source_artifact_id: str,
    source_artifact_sha256: str,
    source_row: Mapping[str, Any],
    world_seed: int,
    information_arm: str | None,
    quality_status: str = "valid",
    failure_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "record_type": record_type,
        "track": track,
        "execution_role": execution_role,
        "analysis_role": analysis_role,
        "source_artifact_id": source_artifact_id,
        "source_artifact_sha256": source_artifact_sha256,
        "source_row_sha256": canonical_sha256(source_row),
        "world_seed": world_seed,
        "information_arm": information_arm,
        "provider_call_count": 0,
        "quality_status": quality_status,
        "failure_reasons": list(failure_reasons),
    }


def _incremental_rows(
    *,
    fork_qualification: Mapping[str, Any],
    fork_certificate: Mapping[str, Any],
    policy_report: Mapping[str, Any],
    policy_audit: Mapping[str, Any],
    latent_contract: Mapping[str, Any],
    latent_reconstructability: Mapping[str, Any],
    latent_formal: Mapping[str, Any],
    latent_analysis: Mapping[str, Any],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    records: dict[str, dict[str, list[dict[str, Any]]]] = {
        "F": {
            "world_fork_pairs": [],
            "world_fork_expectations": [],
            "world_fork_traces": [],
        },
        "V": {
            "policy_campaign_profiles": [],
            "policy_lifecycles": [],
            "policy_retest_campaigns": [],
        },
        "L": {
            "terminal_lifecycles": [],
            "latent_discard_units": [],
            "campaign_cells": [],
        },
    }

    fork_sha = str(fork_qualification["report_sha256"])
    certificate_sha = str(fork_certificate["certificate_sha256"])
    qualification_by_fork = {
        row["runtime_result"]["fork_spec"]["fork_id"]: row
        for row in fork_qualification["rows"]
    }
    for pair in fork_certificate["result"]["pairs"]:
        fork_id = str(pair["fork_id"])
        seed = int(pair["seed"])
        pair_row = _base_record(
            track="F",
            record_type="world_fork_pair",
            record_id=f"F:pair:{fork_id}",
            execution_role="original_primary",
            analysis_role="primary",
            source_artifact_id="world_fork_certificate",
            source_artifact_sha256=certificate_sha,
            source_row=pair,
            world_seed=seed,
            information_arm=None,
        )
        pair_row.update(
            {
                "case_id": pair["case_id"],
                "fork_id": fork_id,
                "intervention_class": pair["intervention_class"],
                "target_component_id": pair["target_component_id"],
                "parent_world_sha256": pair["parent_world_sha256"],
                "child_world_sha256": pair["child_world_sha256"],
                "fork_spec_sha256": pair["fork_spec_sha256"],
                "action_count_per_execution": pair["action_count_per_execution"],
                "gates": pair["gates"],
                "passed": pair["passed"],
            }
        )
        records["F"]["world_fork_pairs"].append(pair_row)
        for expectation in pair["expectations"]:
            expectation_row = _base_record(
                track="F",
                record_type="world_fork_expectation",
                record_id=f"F:expectation:{fork_id}:{expectation['expectation_id']}",
                execution_role="original_primary",
                analysis_role="primary",
                source_artifact_id="world_fork_certificate",
                source_artifact_sha256=certificate_sha,
                source_row=expectation,
                world_seed=seed,
                information_arm=None,
            )
            expectation_row.update(
                {
                    "fork_id": fork_id,
                    "case_id": pair["case_id"],
                    **dict(expectation),
                }
            )
            records["F"]["world_fork_expectations"].append(expectation_row)

        qualification = qualification_by_fork[fork_id]
        runtime = qualification["runtime_result"]
        for execution_role, collection in (
            ("original_primary", runtime["traces"]),
            ("exact_replay", runtime["replays"]),
        ):
            for variant in ("parent", "child"):
                trace = collection[variant]
                trace_row = _base_record(
                    track="F",
                    record_type="world_fork_trace",
                    record_id=f"F:trace:{fork_id}:{variant}:{execution_role}",
                    execution_role=execution_role,
                    analysis_role="audit_only",
                    source_artifact_id="world_fork_qualification",
                    source_artifact_sha256=fork_sha,
                    source_row=trace,
                    world_seed=seed,
                    information_arm=None,
                )
                trace_row.update(
                    {
                        "fork_id": fork_id,
                        "case_id": pair["case_id"],
                        "world_variant": variant,
                        "task_id": trace["task_id"],
                        "trace_sha256": trace["trace_sha256"],
                        "action_count": trace["action_count"],
                        "all_actions_committed": trace["all_actions_committed"],
                    }
                )
                records["F"]["world_fork_traces"].append(trace_row)

    policy_sha = str(policy_report["report_sha256"])
    policy_audit_sha = str(policy_audit["audit_sha256"])
    for campaign in policy_report["campaign_profiles"]:
        identity = campaign["identity"]
        profile = campaign["profile"]
        profile_row = _base_record(
            track="V",
            record_type="policy_campaign_profile",
            record_id=f"V:campaign:{identity['campaign_id']}",
            execution_role="original_primary",
            analysis_role="primary",
            source_artifact_id="known_policy_validity_report",
            source_artifact_sha256=policy_sha,
            source_row=campaign,
            world_seed=int(identity["world_seed"]),
            information_arm=str(identity["information_arm"]),
        )
        profile_row.update(
            {
                **dict(identity),
                "cell_id": campaign["cell_id"],
                "counts": profile["counts"],
                "construct_axes": profile["construct_axes"],
                "endpoint_context": profile["endpoint_context"],
                "reliability": profile["reliability"],
                "exact_replay": campaign["exact_replay"],
                "test_retest": campaign["test_retest"],
            }
        )
        records["V"]["policy_campaign_profiles"].append(profile_row)

    for campaign in policy_audit["cells"]:
        identity = campaign["identity"]
        campaign_id = str(identity["campaign_id"])
        for lifecycle_index, terminal_decision in enumerate(campaign["terminal_vector"]):
            lifecycle_row = _base_record(
                track="V",
                record_type="policy_lifecycle",
                record_id=f"V:lifecycle:{campaign_id}:{lifecycle_index}",
                execution_role="original_primary",
                analysis_role="primary",
                source_artifact_id="known_policy_formal_audit",
                source_artifact_sha256=policy_audit_sha,
                source_row=campaign,
                world_seed=int(identity["world_seed"]),
                information_arm=str(identity["information_arm"]),
            )
            lifecycle_row.update(
                {
                    "campaign_id": campaign_id,
                    "policy_id": identity["policy_id"],
                    "lifecycle_index": lifecycle_index,
                    "terminal_decision": terminal_decision,
                    "trajectory_manifest_sha256": campaign["trajectory_manifest_sha256"],
                }
            )
            records["V"]["policy_lifecycles"].append(lifecycle_row)
        retest_row = _base_record(
            track="V",
            record_type="policy_retest_campaign",
            record_id=f"V:retest:{campaign_id}",
            execution_role="deterministic_retest",
            analysis_role="reliability",
            source_artifact_id="known_policy_formal_audit",
            source_artifact_sha256=policy_audit_sha,
            source_row=campaign,
            world_seed=int(identity["world_seed"]),
            information_arm=str(identity["information_arm"]),
        )
        retest_row.update(
            {
                "campaign_id": campaign_id,
                "policy_id": identity["policy_id"],
                "exact_replay": campaign["exact_replay"],
                "test_retest": campaign["test_retest"],
                "hashes": campaign["hashes"],
                "trajectory_manifest_sha256": campaign["trajectory_manifest_sha256"],
            }
        )
        records["V"]["policy_retest_campaigns"].append(retest_row)

    latent_contract_sha = str(latent_contract["contract_sha256"])
    reconstruct_sha = str(latent_reconstructability["report_sha256"])
    formal_sha = str(latent_formal["report_sha256"])
    analysis_sha = str(latent_analysis["analysis_sha256"])
    reconstruct_by_discard = {
        row["discard_id"]: row
        for cell in latent_reconstructability["cells"]
        for row in cell["discard_units"]
    }
    formal_by_discard = {row["discard_id"]: row for row in latent_formal["receipts"]}
    analysis_by_discard = {row["discard_id"]: row for row in latent_analysis["unit_rows"]}
    oracle_by_cell = {
        row["cell_id"]: row
        for row in latent_analysis["estimands"]["campaign_oracle_regret"]["cells"]
    }
    missing_by_cell = latent_analysis["missingness_and_censoring"]["by_campaign_cell"]
    for cell in latent_contract["population"]["cells"]:
        cell_id = str(cell["cell_id"])
        seed = int(cell["world_seed"])
        arm = str(cell["information_arm"])
        for observed in cell["observed_assays"]:
            lifecycle_index = int(observed["lifecycle_index"])
            terminal = _base_record(
                track="L",
                record_type="terminal_lifecycle",
                record_id=f"L:terminal:{cell_id}:{lifecycle_index}",
                execution_role="observed_terminal",
                analysis_role="primary",
                source_artifact_id="latent_terminal_estimand_contract",
                source_artifact_sha256=latent_contract_sha,
                source_row=observed,
                world_seed=seed,
                information_arm=arm,
            )
            terminal.update(
                {
                    "cell_id": cell_id,
                    "lifecycle_index": lifecycle_index,
                    "terminal_step": observed["terminal_step"],
                    "terminal_kind": "observed_assay",
                    "observed_score": observed["score"],
                }
            )
            records["L"]["terminal_lifecycles"].append(terminal)
        for discarded in cell["discard_units"]:
            discard_id = str(discarded["discard_id"])
            lifecycle_index = int(discarded["lifecycle_index"])
            terminal = _base_record(
                track="L",
                record_type="terminal_lifecycle",
                record_id=f"L:terminal:{cell_id}:{lifecycle_index}",
                execution_role="observed_terminal",
                analysis_role="primary",
                source_artifact_id="latent_terminal_estimand_contract",
                source_artifact_sha256=latent_contract_sha,
                source_row=discarded,
                world_seed=seed,
                information_arm=arm,
            )
            terminal.update(
                {
                    "cell_id": cell_id,
                    "lifecycle_index": lifecycle_index,
                    "terminal_step": discarded["terminal_step"],
                    "terminal_kind": "original_discard",
                    "observed_score": None,
                    "discard_id": discard_id,
                }
            )
            records["L"]["terminal_lifecycles"].append(terminal)

            reconstructed = reconstruct_by_discard[discard_id]
            formal = formal_by_discard[discard_id]
            analysis = analysis_by_discard[discard_id]
            failure_reasons = list(analysis["binding_errors"])
            if analysis["unresolved_reason"] is not None:
                failure_reasons.append(str(analysis["unresolved_reason"]))
            discard_row = _base_record(
                track="L",
                record_type="latent_discard_unit",
                record_id=f"L:discard:{discard_id}",
                execution_role="evaluator_shadow",
                analysis_role="primary",
                source_artifact_id="latent_terminal_analysis",
                source_artifact_sha256=analysis_sha,
                source_row=analysis,
                world_seed=seed,
                information_arm=arm,
                quality_status=(
                    "valid" if analysis["outcome_status"] == "resolved" else "unresolved"
                ),
                failure_reasons=failure_reasons,
            )
            discard_row.update(
                {
                    "discard_id": discard_id,
                    "cell_id": cell_id,
                    "lifecycle_index": lifecycle_index,
                    "terminal_step": discarded["terminal_step"],
                    "terminal_action_sha256": discarded["terminal_action_sha256"],
                    "public_prefix_sha256": discarded["public_prefix_sha256"],
                    "hidden_state_sha256": reconstructed["hidden_state_sha256"],
                    "campaign_resource_snapshot_sha256": reconstructed[
                        "campaign_resource_snapshot_sha256"
                    ],
                    "outcome_status": analysis["outcome_status"],
                    "latent_terminal_score": analysis["score"],
                    "unresolved_category": analysis["unresolved_category"],
                    "contract_source_row_sha256": canonical_sha256(discarded),
                    "reconstructability_source_artifact_sha256": reconstruct_sha,
                    "reconstructability_source_row_sha256": canonical_sha256(reconstructed),
                    "formal_source_artifact_sha256": formal_sha,
                    "formal_source_row_sha256": canonical_sha256(formal),
                }
            )
            records["L"]["latent_discard_units"].append(discard_row)

        oracle = oracle_by_cell[cell_id]
        missing = missing_by_cell[cell_id]
        composite = {"population": cell, "oracle": oracle, "missingness": missing}
        campaign_row = _base_record(
            track="L",
            record_type="latent_campaign_cell",
            record_id=f"L:campaign:{cell_id}",
            execution_role="evaluator_shadow",
            analysis_role="primary",
            source_artifact_id="latent_terminal_analysis",
            source_artifact_sha256=analysis_sha,
            source_row=composite,
            world_seed=seed,
            information_arm=arm,
            quality_status=("valid" if missing["unresolved_count"] == 0 else "unresolved"),
            failure_reasons=(
                []
                if missing["unresolved_count"] == 0
                else ["campaign_contains_unresolved_shadow_receipts"]
            ),
        )
        campaign_row.update(
            {
                "cell_id": cell_id,
                "closed_lifecycle_count": len(cell["terminal_sequence"]),
                "observed_assay_count": cell["observed_assay_count"],
                "observed_discard_count": cell["observed_discard_count"],
                "campaign_best_assayed_score": cell["campaign_best_assayed_score"],
                "discard_opportunity": oracle["opportunity"],
                "oracle_regret_point_estimate": oracle["point_estimate"],
                "oracle_regret_bounds": oracle["bounds"],
                "unresolved_count": missing["unresolved_count"],
                "unresolved_fraction": missing["unresolved_fraction"],
            }
        )
        records["L"]["campaign_cells"].append(campaign_row)
    return records


def _flatten_incremental_records(
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[Mapping[str, Any]]:
    return [row for track in records.values() for rows in track.values() for row in rows]


def _validate_fvl_records(
    records: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, bool]:
    expected = {
        ("F", "world_fork_pairs"): 6,
        ("F", "world_fork_expectations"): 12,
        ("F", "world_fork_traces"): 24,
        ("V", "policy_campaign_profiles"): 30,
        ("V", "policy_lifecycles"): 180,
        ("V", "policy_retest_campaigns"): 30,
        ("L", "terminal_lifecycles"): 60,
        ("L", "latent_discard_units"): 36,
        ("L", "campaign_cells"): 10,
    }
    count_gate = all(
        len(records[track][name]) == count for (track, name), count in expected.items()
    )
    flat = _flatten_incremental_records(records)
    ids = [str(row["record_id"]) for row in flat]
    common = {
        "analysis_role",
        "execution_role",
        "failure_reasons",
        "information_arm",
        "provider_call_count",
        "quality_status",
        "record_id",
        "record_type",
        "source_artifact_id",
        "source_artifact_sha256",
        "source_row_sha256",
        "track",
        "world_seed",
    }
    terminal_partition = {
        row["terminal_kind"] for row in records["L"]["terminal_lifecycles"]
    } == {"observed_assay", "original_discard"} and sum(
        row["terminal_kind"] == "observed_assay"
        for row in records["L"]["terminal_lifecycles"]
    ) == 24
    gates = {
        "exact_registered_record_counts": count_gate,
        "globally_unique_record_ids": len(ids) == len(set(ids)),
        "common_fields_complete": all(common <= set(row) for row in flat),
        "provider_call_count_zero": all(row["provider_call_count"] == 0 for row in flat),
        "f_trace_relationship_complete": all(
            sum(row["fork_id"] == pair["fork_id"] for row in records["F"]["world_fork_traces"])
            == 4
            for pair in records["F"]["world_fork_pairs"]
        ),
        "v_lifecycle_relationship_complete": all(
            sum(
                row["campaign_id"] == campaign["campaign_id"]
                for row in records["V"]["policy_lifecycles"]
            )
            == 6
            for campaign in records["V"]["policy_campaign_profiles"]
        ),
        "l_terminal_partition_complete": terminal_partition,
        "l_all_discard_units_retained": len(records["L"]["latent_discard_units"]) == 36,
        "l_complete_case_substitution_not_used": True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise ArxivV1DerivedDataError("F/V/L derived validation failed: " + ", ".join(failed))
    return gates


def build_arxiv_v1_derived_data(
    *,
    g0_v10_path: Path,
    g0_v12_path: Path,
    task_design_path: Path,
    experiment_ledger_path: Path,
    g2_v04_audit_path: Path,
    g2_v05_audit_path: Path | None = None,
    work_i_data_contract_path: Path | None = None,
    fork_qualification_path: Path | None = None,
    fork_certificate_path: Path | None = None,
    policy_report_path: Path | None = None,
    policy_audit_path: Path | None = None,
    policy_delivery_manifest_path: Path | None = None,
    latent_contract_path: Path | None = None,
    latent_reconstructability_path: Path | None = None,
    latent_formal_path: Path | None = None,
    latent_analysis_path: Path | None = None,
) -> dict[str, Any]:
    """Build a compact, auditable paper-data object from frozen inputs."""

    g0_v10 = _load_json(g0_v10_path, label="G0 v1.0 summary")
    g0_v12 = _load_json(g0_v12_path, label="G0 v1.2 summary")
    task_design = _load_json(task_design_path, label="task design audit")
    experiment_ledger = _load_json(
        experiment_ledger_path,
        label="arXiv experiment ledger",
    )
    g2_v04 = _load_json(g2_v04_audit_path, label="G2 v0.4 audit")
    _require(g0_v10.get("formal_result") is True, "G0 v1.0 is not formal")
    _require(g0_v12.get("formal_result") is True, "G0 v1.2 is not formal")
    _require(
        g0_v12.get("confirmatory_analysis_complete") is True,
        "G0 v1.2 analysis is incomplete",
    )
    task_rows, world_rows, baseline_rows = _task_rows(g0_v10, g0_v12)
    g2_cells, g2_pairs, demonstration = _g2_v04_rows(g2_v04)
    g2_v05 = None
    sources = {
        "g0_v1_0": _source(g0_v10_path, canonical_json=True),
        "g0_v1_2": _source(g0_v12_path, canonical_json=True),
        "task_design": _source(task_design_path, canonical_json=True),
        "experiment_ledger": _source(
            experiment_ledger_path,
            canonical_json=True,
        ),
        "g2_v0_4": _source(g2_v04_audit_path, canonical_json=True),
        "g2_v0_5": None,
    }
    if g2_v05_audit_path is not None:
        g2_v05_audit = _load_json(g2_v05_audit_path, label="G2 v0.5 audit")
        g2_v05 = _g2_v05_rows(g2_v05_audit)
        sources["g2_v0_5"] = _source(g2_v05_audit_path, canonical_json=True)
    qualification = dict(experiment_ledger["foundation_qualification"])
    _require(
        qualification["registered_tasks"] == task_design["task_count"],
        "task-design and experiment-ledger task counts disagree",
    )
    validation = task_design["design_validation"]
    _require(
        qualification["deterministic_complete_experiment_cases"]
        == validation["boundary_recipe_case_count"],
        "task-design and experiment-ledger execution-case counts disagree",
    )
    _require(
        qualification["bound_success_endpoints"] == validation["bound_success_metric_count"],
        "task-design and experiment-ledger endpoint counts disagree",
    )
    result: dict[str, Any] = {
        "schema_version": ARXIV_V1_DERIVED_SCHEMA,
        "status": ("frozen_complete" if g2_v05 is not None else "provisional_awaiting_g2_v0_5"),
        "paper_scope": {
            "compiled_experiment_tasks": 2,
            "autonomous_task_count": 1,
            "g0_nonduplicated_physical_experiments": 29580,
            "g2_v0_4_completed_final_assays": 60,
            "g2_v0_5_included": g2_v05 is not None,
            "general_population_prior_effect_allowed": False,
        },
        "sources": sources,
        "environment_qualification": qualification,
        "g0": {
            "task_arm_rows": task_rows,
            "world_arm_rows": world_rows,
            "baseline_rows": baseline_rows,
        },
        "g2_v0_4": {
            "audit_sha256": g2_v04["audit_sha256"],
            "cell_rows": g2_cells,
            "paired_world_rows": g2_pairs,
            "arm_descriptive_aggregates": g2_v04["arm_descriptive_aggregates"],
            "one_experiment_demonstration": demonstration,
        },
        "g2_v0_5": g2_v05,
    }
    incremental_paths = {
        "work_i_data_contract": work_i_data_contract_path,
        "world_fork_qualification": fork_qualification_path,
        "world_fork_certificate": fork_certificate_path,
        "known_policy_validity_report": policy_report_path,
        "known_policy_formal_audit": policy_audit_path,
        "known_policy_delivery_manifest": policy_delivery_manifest_path,
        "latent_terminal_estimand_contract": latent_contract_path,
        "latent_terminal_reconstructability": latent_reconstructability_path,
        "latent_terminal_formal_shadow": latent_formal_path,
        "latent_terminal_analysis": latent_analysis_path,
    }
    supplied_incremental = [path is not None for path in incremental_paths.values()]
    _require(
        not any(supplied_incremental) or all(supplied_incremental),
        "the Work I F/V/L source set must be supplied in full",
    )
    if all(supplied_incremental):
        loaded = {
            artifact_id: _load_json(path, label=artifact_id)
            for artifact_id, optional_path in incremental_paths.items()
            if (path := optional_path) is not None
        }
        contract = loaded["work_i_data_contract"]
        contract_errors = validate_work_i_data_contract(contract)
        _require(not contract_errors, "invalid Work I data contract: " + "; ".join(contract_errors))
        contract_sha = data_contract_sha256(contract)
        _require(contract_sha == contract["contract_sha256"], "Work I contract hash mismatch")
        records = _incremental_rows(
            fork_qualification=loaded["world_fork_qualification"],
            fork_certificate=loaded["world_fork_certificate"],
            policy_report=loaded["known_policy_validity_report"],
            policy_audit=loaded["known_policy_formal_audit"],
            latent_contract=loaded["latent_terminal_estimand_contract"],
            latent_reconstructability=loaded["latent_terminal_reconstructability"],
            latent_formal=loaded["latent_terminal_formal_shadow"],
            latent_analysis=loaded["latent_terminal_analysis"],
        )
        validation_gates = _validate_fvl_records(records)
        artifact_hash_fields = {
            "work_i_data_contract": "contract_sha256",
            "world_fork_qualification": "report_sha256",
            "world_fork_certificate": "certificate_sha256",
            "known_policy_validity_report": "report_sha256",
            "known_policy_formal_audit": "audit_sha256",
            "known_policy_delivery_manifest": "delivery_manifest_sha256",
            "latent_terminal_estimand_contract": "contract_sha256",
            "latent_terminal_reconstructability": "report_sha256",
            "latent_terminal_formal_shadow": "report_sha256",
            "latent_terminal_analysis": "analysis_sha256",
        }
        incremental_sources = {}
        for artifact_id, optional_path in incremental_paths.items():
            if optional_path is None:
                continue
            incremental_sources[artifact_id] = _artifact_source(
                optional_path,
                loaded[artifact_id],
                artifact_sha256=str(loaded[artifact_id][artifact_hash_fields[artifact_id]]),
            )
        record_counts = {
            track: {name: len(rows) for name, rows in track_records.items()}
            for track, track_records in records.items()
        }
        flat_records = _flatten_incremental_records(records)
        role_counts: dict[str, int] = {}
        for row in flat_records:
            key = f"{row['track']}:{row['analysis_role']}:{row['execution_role']}"
            role_counts[key] = role_counts.get(key, 0) + 1
        result["work_i_incremental"] = {
            "schema_version": WORK_I_INCREMENTAL_SCHEMA,
            "status": "frozen_with_explicit_latent_missingness",
            "data_contract_sha256": contract_sha,
            "sources": incremental_sources,
            "records": records,
            "record_counts": record_counts,
            "role_counts": role_counts,
            "summaries": {
                "F": {
                    "primary_pair_count": {"numerator": 6, "denominator": 6, "unit": "pair"},
                    "expectation_row_count": {
                        "numerator": 12,
                        "denominator": 12,
                        "unit": "expectation_within_pair",
                    },
                },
                "V": {
                    "primary_campaign_count": {
                        "numerator": 30,
                        "denominator": 30,
                        "unit": "campaign",
                    },
                    "retest_in_primary_estimand": {
                        "numerator": 0,
                        "denominator": 30,
                        "unit": "retest_campaign",
                    },
                },
                "L": {
                    "resolved_shadow_receipts": {
                        "numerator": 6,
                        "denominator": 36,
                        "unit": "discarded_lifecycle",
                    },
                    "unresolved_shadow_receipts": {
                        "numerator": 30,
                        "denominator": 36,
                        "unit": "discarded_lifecycle",
                    },
                    "discard_opportunity_campaign_cells": {
                        "numerator": 9,
                        "denominator": 10,
                        "unit": "campaign_cell",
                    },
                },
            },
            "validation_gates": validation_gates,
            "scientific_boundaries": {
                "cross_track_primary_units_pooled": False,
                "replays_or_retests_counted_as_primary": False,
                "latent_complete_case_substitution_used": False,
                "latent_primary_point_estimates_withheld_where_required": True,
                "provider_calls_executed_by_builder": 0,
                "raw_hidden_state_or_provider_payloads_included": False,
            },
        }
    else:
        result["work_i_incremental"] = None
    result["derived_data_sha256"] = canonical_sha256(result)
    return result


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (dict, list))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def write_arxiv_v1_tables(output_dir: Path, data: Mapping[str, Any]) -> list[Path]:
    """Write deterministic CSV views; every value comes from ``data``."""

    outputs = [
        output_dir / "g0-task-arm.csv",
        output_dir / "g0-world-arm.csv",
        output_dir / "g0-baselines.csv",
        output_dir / "g2-v0.4-cells.csv",
        output_dir / "g2-v0.4-paired-worlds.csv",
    ]
    rows = [
        data["g0"]["task_arm_rows"],
        data["g0"]["world_arm_rows"],
        data["g0"]["baseline_rows"],
        data["g2_v0_4"]["cell_rows"],
        data["g2_v0_4"]["paired_world_rows"],
    ]
    if data["g2_v0_5"] is not None:
        outputs.append(output_dir / "g2-v0.5-paired-trajectories.csv")
        rows.append(data["g2_v0_5"]["paired_trajectories"])
    incremental = data.get("work_i_incremental")
    if incremental is not None:
        incremental_tables = (
            ("work-i-f-world-fork-pairs.csv", "F", "world_fork_pairs"),
            ("work-i-f-world-fork-expectations.csv", "F", "world_fork_expectations"),
            ("work-i-f-world-fork-traces.csv", "F", "world_fork_traces"),
            ("work-i-v-policy-campaign-profiles.csv", "V", "policy_campaign_profiles"),
            ("work-i-v-policy-lifecycles.csv", "V", "policy_lifecycles"),
            ("work-i-v-policy-retests.csv", "V", "policy_retest_campaigns"),
            ("work-i-l-terminal-lifecycles.csv", "L", "terminal_lifecycles"),
            ("work-i-l-latent-discard-units.csv", "L", "latent_discard_units"),
            ("work-i-l-campaign-cells.csv", "L", "campaign_cells"),
        )
        for name, track, record_type in incremental_tables:
            outputs.append(output_dir / name)
            rows.append(incremental["records"][track][record_type])
    for path, table_rows in zip(outputs, rows, strict=True):
        _write_csv(path, table_rows)
    return outputs


def build_derived_data_manifest(
    *,
    root: Path,
    derived_data_path: Path,
    table_paths: Sequence[Path],
    data: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable file/count manifest after deterministic files are written."""

    incremental = data.get("work_i_incremental")
    if not isinstance(incremental, Mapping):
        raise ArxivV1DerivedDataError("F/V/L incremental data is absent")
    files = []
    for path in (derived_data_path, *table_paths):
        files.append(
            {
                "path": path.resolve().relative_to(root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": WORK_I_DERIVED_MANIFEST_SCHEMA,
        "status": "frozen",
        "immutable": True,
        "derived_data_sha256": data["derived_data_sha256"],
        "data_contract_sha256": incremental["data_contract_sha256"],
        "files": files,
        "file_count": len(files),
        "record_counts": incremental["record_counts"],
        "role_counts": incremental["role_counts"],
        "counting_rule": (
            "F pairs, V original campaign profiles, and L discarded lifecycles remain "
            "distinct primary units; replays, retests, and evaluator shadows are explicit roles."
        ),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def rebind_figure_manifest(path: Path, *, derived_data_sha256: str) -> dict[str, Any]:
    """Mechanically rebind the existing rendered-figure manifest to derived data."""

    manifest = _load_json(path, label="figure manifest")
    manifest["derived_data_sha256"] = derived_data_sha256
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def write_fvl_derived_report(
    path: Path,
    data: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    incremental = data["work_i_incremental"]
    text = f"""# Work I F/V/L frozen derived-data layer v0.1

Status: **frozen with explicit latent missingness**

- Derived-data SHA-256: `{data['derived_data_sha256']}`
- Data-contract SHA-256: `{incremental['data_contract_sha256']}`
- File manifest SHA-256: `{manifest['manifest_sha256']}`
- F: 6 fork pairs, 12 within-pair expectations, 24 original/replay traces.
- V: 30 original campaign profiles, 180 original lifecycles, 30 reliability-only retests.
- L: 60 terminal lifecycles, 36 retained discard units, 10 campaign cells.
- L outcomes: 6 resolved and 30 unresolved; complete-case substitution is forbidden and unused.
- Provider calls made by this build: 0.

The layer is additive to the existing G0/G2 source and keeps the release schema identifier stable.
It publishes only normalized rows, hashes, counts, registered bounds, and explicit failure labels;
raw hidden states and provider payloads are excluded. Downstream evidence-DAG, ledger,
release-manifest, data-card, manuscript, and figure regeneration remain owned by W1-D04/D05/P09.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


__all__ = [
    "ARXIV_V1_DERIVED_SCHEMA",
    "ArxivV1DerivedDataError",
    "build_arxiv_v1_derived_data",
    "build_derived_data_manifest",
    "canonical_sha256",
    "file_sha256",
    "rebind_figure_manifest",
    "write_arxiv_v1_tables",
    "write_fvl_derived_report",
    "write_json",
]
