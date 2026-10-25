"""Frozen schemas, units, and counting rules for incremental Work I evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from chemworld.eval.latent_terminal_contract import (
    latent_terminal_contract_sha256,
    validate_latent_terminal_contract,
)
from chemworld.eval.latent_terminal_reconstructability import (
    reconstructability_report_sha256,
    validate_reconstructability_report,
)
from chemworld.eval.provenance import canonical_json_sha256, file_sha256

CONTRACT_SCHEMA_ID = "chemworld.work_i_incremental_data_contract"
CONTRACT_SCHEMA_VERSION = "0.1.0"
CONTRACT_ID = "work-i-fvl-incremental-data-contract-v0.1"
CONTRACT_PATH = Path("configs/benchmark/work_i_incremental_data_contract_v0.1.json")
REPORT_PATH = Path("workstreams/arxiv_v1/reports/work-i-incremental-data-contract-v0.1.md")

SOURCE_SPECS = (
    {
        "artifact_id": "world_fork_qualification",
        "path": "workstreams/arxiv_v1/reports/work-i-world-fork-qualification-v0.1.json",
        "role": "immutable_formal_report",
        "hash_field": "report_sha256",
        "hash_mode": "json_ascii_without_hash",
    },
    {
        "artifact_id": "world_fork_certificate",
        "path": "workstreams/arxiv_v1/reports/work-i-world-fork-certificate-v0.1.json",
        "role": "immutable_summary_certificate",
        "hash_field": "certificate_sha256",
        "hash_mode": "json_ascii_core_without_certificate_identity",
    },
    {
        "artifact_id": "known_policy_validity_report",
        "path": "workstreams/arxiv_v1/reports/work-i-known-policy-validity-report-v0.1.json",
        "role": "immutable_formal_report",
        "hash_field": "report_sha256",
        "hash_mode": "repository_canonical_without_hash",
    },
    {
        "artifact_id": "known_policy_delivery_manifest",
        "path": (
            "workstreams/arxiv_v1/reports/work-i-known-policy-validity-report-v0.1.manifest.json"
        ),
        "role": "immutable_delivery_manifest",
        "hash_field": "delivery_manifest_sha256",
        "hash_mode": "repository_canonical_without_hash",
    },
    {
        "artifact_id": "latent_terminal_estimand_contract",
        "path": "configs/benchmark/work_i_latent_terminal_contract_v0.1.json",
        "role": "immutable_protocol_input",
        "hash_field": "contract_sha256",
        "hash_mode": "repository_canonical_without_hash",
    },
    {
        "artifact_id": "latent_terminal_reconstructability",
        "path": (
            "workstreams/arxiv_v1/reports/work-i-latent-terminal-reconstructability-v0.1.json"
        ),
        "role": "immutable_outcome_blind_audit",
        "hash_field": "report_sha256",
        "hash_mode": "repository_canonical_without_hash",
    },
    {
        "artifact_id": "latent_terminal_replay_qualification",
        "path": (
            "workstreams/arxiv_v1/reports/work-i-latent-terminal-replay-qualification-v0.1.json"
        ),
        "role": "immutable_synthetic_qualification",
        "hash_field": "report_sha256",
        "hash_mode": "repository_canonical_without_hash",
    },
)
SOURCE_CODE_PATHS = (
    Path("src/chemworld/eval/work_i_data_contract.py"),
    Path("scripts/build_work_i_data_contract.py"),
)


class WorkIDataContractError(RuntimeError):
    """Raised when a dependency or proposed contract fails closed."""


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkIDataContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise WorkIDataContractError(f"{label} must be a JSON object")
    return payload


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise WorkIDataContractError(f"{key} must be an object")
    return value


def _list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise WorkIDataContractError(f"{key} must be a list")
    return value


def _embedded_hash(payload: Mapping[str, Any], field: str, mode: str) -> str:
    supplied = payload.get(field)
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise WorkIDataContractError(f"invalid embedded hash field: {field}")
    unhashed = deepcopy(dict(payload))
    unhashed.pop(field, None)
    if mode == "json_ascii_core_without_certificate_identity":
        unhashed.pop("certificate_id", None)
    if mode.startswith("json_ascii"):
        computed = hashlib.sha256(
            json.dumps(
                unhashed,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    elif mode == "repository_canonical_without_hash":
        computed = canonical_json_sha256(unhashed)
    else:
        raise WorkIDataContractError(f"unsupported source hash mode: {mode}")
    if computed != supplied:
        raise WorkIDataContractError(f"stale embedded hash field: {field}")
    return supplied


def data_contract_sha256(payload: Mapping[str, Any]) -> str:
    """Return the contract digest excluding its embedded self-hash."""

    unhashed = deepcopy(dict(payload))
    unhashed.pop("contract_sha256", None)
    return canonical_json_sha256(unhashed)


def _load_sources(root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    bindings: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        artifact_id = str(spec["artifact_id"])
        relative = Path(str(spec["path"]))
        path = root / relative
        payload = _read_json_object(path, label=artifact_id)
        hash_field = str(spec["hash_field"])
        hash_mode = str(spec["hash_mode"])
        payloads[artifact_id] = payload
        bindings.append(
            {
                "artifact_id": artifact_id,
                "path": relative.as_posix(),
                "role": spec["role"],
                "embedded_hash_field": hash_field,
                "embedded_sha256": _embedded_hash(payload, hash_field, hash_mode),
                "embedded_hash_mode": hash_mode,
                "file_sha256": file_sha256(path),
            }
        )
    return payloads, bindings


def _validate_source_chain(
    root: Path,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, bool]:
    fork = sources["world_fork_qualification"]
    fork_certificate = sources["world_fork_certificate"]
    policy = sources["known_policy_validity_report"]
    policy_manifest = sources["known_policy_delivery_manifest"]
    latent = sources["latent_terminal_estimand_contract"]
    reconstructability = sources["latent_terminal_reconstructability"]
    replay = sources["latent_terminal_replay_qualification"]

    latent_errors = validate_latent_terminal_contract(latent, root=root)
    reconstructability_errors = validate_reconstructability_report(
        reconstructability,
        root=root,
    )
    fork_design = _mapping(fork_certificate, "design")
    fork_result = _mapping(fork_certificate, "result")
    fork_source = _mapping(fork_certificate, "source")
    policy_estimand = _mapping(policy, "estimand")
    policy_reliability = _mapping(policy, "test_retest_reliability")
    latent_population = _mapping(latent, "population")
    latent_counts = _mapping(latent_population, "counts")
    latent_aggregation = _mapping(latent, "aggregation")
    latent_primary_units = _mapping(latent_aggregation, "primary_overall_units")
    reconstructability_census = _mapping(reconstructability, "census")
    replay_census = _mapping(replay, "census")
    policy_profiles = _list(policy, "campaign_profiles")
    policy_summaries = _mapping(policy, "policy_summaries")

    return {
        "world_fork_reports_pass": (
            fork.get("passed") is True
            and fork_result.get("passed") is True
            and _mapping(fork_certificate, "claim_boundary").get("programmable_world_apparatus")
            is True
        ),
        "world_fork_counts_match": (
            fork.get("pair_count") == fork_design.get("parent_child_pair_count") == 6
            and fork.get("trace_count") == fork_design.get("trace_count") == 24
            and fork.get("provider_call_count") == fork_design.get("provider_call_count") == 0
        ),
        "world_fork_certificate_binds_formal_report": (
            fork_source.get("formal_report_content_sha256") == fork.get("report_sha256")
            and fork_source.get("formal_report_file_sha256")
            == file_sha256(
                root / "workstreams/arxiv_v1/reports/work-i-world-fork-qualification-v0.1.json"
            )
        ),
        "known_policy_report_passes": (
            policy.get("status") == "positive_control_established"
            and _mapping(policy, "evidence_validity").get("status") == "valid"
            and _mapping(policy, "scientific_status").get("established") is True
            and policy.get("formal_world_controller_or_provider_execution") is False
        ),
        "known_policy_counts_match": (
            len(policy_profiles) == policy_estimand.get("primary_campaigns") == 30
            and policy_estimand.get("primary_closed_lifecycles") == 180
            and policy_estimand.get("retest_campaigns") == 30
            and policy_estimand.get("retest_closed_lifecycles") == 180
            and policy_estimand.get("provider_calls") == 0
            and set(policy_summaries)
            == {"assay_all", "measure_then_threshold", "start_then_discard"}
        ),
        "known_policy_retests_are_excluded": (
            policy_estimand.get("retest_in_primary_estimand") is False
            and policy_reliability.get("excluded_from_primary_estimand") is True
            and policy_reliability.get("pair_count") == 30
        ),
        "known_policy_delivery_manifest_passes": (
            policy_manifest.get("status") == "complete"
            and policy_manifest.get("immutable") is True
            and policy_manifest.get("entry_count") == len(_list(policy_manifest, "entries"))
        ),
        "latent_estimand_contract_passes": (
            not latent_errors
            and latent.get("contract_sha256") == latent_terminal_contract_sha256(latent)
        ),
        "latent_population_counts_match": (
            latent_counts.get("cells") == latent_primary_units.get("campaign_cells") == 10
            and latent_counts.get("closed_lifecycles")
            == latent_primary_units.get("all_lifecycles_for_classification")
            == 60
            and latent_counts.get("observed_assays") == 24
            and latent_counts.get("observed_discards")
            == latent_primary_units.get("discarded_lifecycles")
            == 36
            and latent_counts.get("shadow_evaluations_planned") == 36
            and latent_primary_units.get("campaign_cells_with_discard_opportunity") == 9
        ),
        "latent_reconstructability_passes_without_outcomes": (
            not reconstructability_errors
            and reconstructability.get("report_sha256")
            == reconstructability_report_sha256(reconstructability)
            and reconstructability_census.get("reconstructable_unit_count") == 36
            and reconstructability_census.get("shadow_terminal_evaluations_executed") == 0
            and reconstructability_census.get("latent_discard_scores_accessed") == 0
        ),
        "latent_replay_is_synthetic_only": (
            replay.get("status") == "PASS"
            and replay.get("formal_execution_owner") == "W1-L05"
            and replay_census.get("formal_checkpoint_payloads_loaded") == 0
            and replay_census.get("formal_shadow_terminal_evaluations_executed") == 0
            and replay_census.get("formal_latent_discard_scores_accessed") == 0
        ),
    }


def _unit_registry() -> dict[str, dict[str, Any]]:
    return {
        "count": {
            "canonical_unit": "count",
            "json_type": "integer",
            "minimum": 0,
        },
        "ordinal": {
            "canonical_unit": "ordinal",
            "json_type": "integer",
            "minimum": 0,
        },
        "world_seed": {
            "canonical_unit": "seed",
            "json_type": "integer",
            "minimum": 0,
        },
        "dimensionless_fraction": {
            "canonical_unit": "1",
            "json_type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "dimensionless_ratio": {
            "canonical_unit": "1",
            "json_type": "number",
            "minimum": 0.0,
        },
        "normalized_score": {
            "canonical_unit": "1",
            "json_type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "normalized_score_difference": {
            "canonical_unit": "1",
            "json_type": "number",
            "minimum": -1.0,
            "maximum": 1.0,
        },
        "mole": {
            "canonical_unit": "mol",
            "json_type": "number",
        },
        "liter": {
            "canonical_unit": "L",
            "json_type": "number",
        },
        "second": {
            "canonical_unit": "s",
            "json_type": "number",
        },
        "volt": {
            "canonical_unit": "V",
            "json_type": "number",
        },
        "milliampere": {
            "canonical_unit": "mA",
            "json_type": "number",
        },
        "joule": {
            "canonical_unit": "J",
            "json_type": "number",
        },
        "kelvin": {
            "canonical_unit": "K",
            "json_type": "number",
        },
        "pascal": {
            "canonical_unit": "Pa",
            "json_type": "number",
        },
        "currency": {
            "canonical_unit": "currency",
            "json_type": "number",
            "minimum": 0.0,
        },
        "risk": {
            "canonical_unit": "risk",
            "json_type": "number",
            "minimum": 0.0,
        },
        "sha256": {
            "canonical_unit": "sha256_hex",
            "json_type": "string",
            "pattern": "^[0-9a-f]{64}$",
        },
    }


def _common_fields() -> dict[str, dict[str, Any]]:
    return {
        "record_id": {"json_type": "string", "nullable": False},
        "record_type": {"json_type": "string", "nullable": False},
        "track": {
            "json_type": "string",
            "enum": ["F", "V", "L"],
            "nullable": False,
        },
        "execution_role": {
            "json_type": "string",
            "enum": [
                "original_primary",
                "exact_replay",
                "deterministic_retest",
                "synthetic_qualification",
                "evaluator_shadow",
                "observed_terminal",
            ],
            "nullable": False,
        },
        "analysis_role": {
            "json_type": "string",
            "enum": ["primary", "reliability", "qualification", "audit_only"],
            "nullable": False,
        },
        "source_artifact_id": {"json_type": "string", "nullable": False},
        "source_artifact_sha256": {
            "json_type": "string",
            "unit_id": "sha256",
            "nullable": False,
        },
        "source_row_sha256": {
            "json_type": "string",
            "unit_id": "sha256",
            "nullable": False,
        },
        "world_seed": {
            "json_type": "integer",
            "unit_id": "world_seed",
            "nullable": False,
        },
        "information_arm": {"json_type": "string", "nullable": True},
        "provider_call_count": {
            "json_type": "integer",
            "unit_id": "count",
            "nullable": False,
        },
        "quality_status": {
            "json_type": "string",
            "enum": ["valid", "unresolved", "invalid"],
            "nullable": False,
        },
        "failure_reasons": {"json_type": "array[string]", "nullable": False},
    }


def _track_contracts(latent: Mapping[str, Any]) -> dict[str, Any]:
    estimands = _list(latent, "estimands")
    return {
        "F": {
            "track_name": "world_fork_programmability",
            "formal_population": {
                "case_count": 2,
                "world_seed_count_per_case": 3,
                "parent_child_pair_count": 6,
                "world_variants_per_pair": 2,
                "executions_per_variant": 2,
                "trace_count": 24,
                "provider_call_count": 0,
            },
            "record_schemas": {
                "world_fork_pair": {
                    "primary_key": ["fork_id", "world_seed"],
                    "expected_row_count": 6,
                    "analysis_role": "primary",
                    "weighting": "one_parent_child_pair_one_equal_weight",
                    "required_fields": [
                        "case_id",
                        "fork_id",
                        "world_seed",
                        "intervention_class",
                        "target_component_id",
                        "parent_world_sha256",
                        "child_world_sha256",
                        "action_count_per_execution",
                        "gates",
                    ],
                },
                "world_fork_trace": {
                    "primary_key": [
                        "fork_id",
                        "world_seed",
                        "world_variant",
                        "execution_role",
                    ],
                    "expected_row_count": 24,
                    "analysis_role": "audit_only",
                    "allowed_world_variants": ["parent", "child"],
                    "allowed_execution_roles": ["original_primary", "exact_replay"],
                },
                "world_fork_expectation": {
                    "primary_key": ["fork_id", "expectation_id"],
                    "expected_row_count": 12,
                    "analysis_role": "primary",
                    "value_units_by_expectation_id": {
                        "terminal-product-organic-amount": "mole",
                        "assayed-product-organic-fraction": "dimensionless_fraction",
                        "terminal-selective-product-amount": "mole",
                        "assayed-ohmic-efficiency": "dimensionless_fraction",
                    },
                    "relative_delta_unit_id": "dimensionless_ratio",
                },
            },
            "counting_rules": {
                "primary_analysis_unit": "one frozen parent-child pair",
                "trace_replays_in_primary_denominator": False,
                "parent_and_child_are_members_of_one_pair_not_two_independent_pairs": True,
                "expectation_rows_are_repeated_measures_within_pair": True,
                "qualification_or_provider_calls_in_primary_denominator": False,
            },
        },
        "V": {
            "track_name": "known_policy_measurement_validity",
            "formal_population": {
                "world_count": 5,
                "information_arm_count": 2,
                "policy_count": 3,
                "primary_campaign_count": 30,
                "closed_lifecycles_per_campaign": 6,
                "primary_closed_lifecycle_count": 180,
                "deterministic_retest_campaign_count": 30,
                "deterministic_retest_closed_lifecycle_count": 180,
                "provider_call_count": 0,
            },
            "record_schemas": {
                "policy_campaign_profile": {
                    "primary_key": ["campaign_id"],
                    "expected_row_count": 30,
                    "analysis_role": "primary",
                    "weighting": "one_campaign_one_equal_weight_within_policy",
                    "required_identity_fields": [
                        "world_seed",
                        "information_arm",
                        "policy_id",
                        "physical_identity_sha256",
                        "noise_identity_sha256",
                        "material_information_sha256",
                        "resource_card_sha256",
                    ],
                },
                "policy_lifecycle": {
                    "primary_key": ["campaign_id", "lifecycle_index"],
                    "expected_primary_row_count": 180,
                    "analysis_role": "primary",
                    "independent_analysis_unit": False,
                },
                "policy_retest_campaign": {
                    "primary_key": ["campaign_id", "execution_role"],
                    "expected_row_count": 30,
                    "analysis_role": "reliability",
                    "execution_role": "deterministic_retest",
                },
            },
            "metric_units": {
                "assay_fraction": "dimensionless_fraction",
                "discard_fraction": "dimensionless_fraction",
                "closed_lifecycle_fraction": "dimensionless_fraction",
                "measured_lifecycle_fraction": "dimensionless_fraction",
                "continued_after_measurement_fraction": "dimensionless_fraction",
                "threshold_eligible_fraction": "dimensionless_fraction",
                "threshold_decision_concordance": "dimensionless_fraction",
                "global_best_discovery_fraction": "dimensionless_fraction",
                "loss_episode_recovery_rate": "dimensionless_fraction",
                "online_incumbent_retention_rate": "dimensionless_fraction",
                "terminal_to_global_best_ratio": "dimensionless_ratio",
                "maximum_absolute_incumbent_drawdown": "normalized_score_difference",
                "attempted_operations_per_closed_lifecycle": "dimensionless_ratio",
                "committed_operations_per_closed_lifecycle": "dimensionless_ratio",
                "nonfinal_instrument_uses_per_closed_lifecycle": "dimensionless_ratio",
                "post_measure_process_operations_per_closed_lifecycle": ("dimensionless_ratio"),
                "total_cost_per_closed_lifecycle": "currency",
                "total_risk_per_closed_lifecycle": "risk",
                "best_assayed_score": "normalized_score",
                "mean_assayed_score": "normalized_score",
            },
            "counting_rules": {
                "primary_analysis_unit": "one scheduled original campaign profile",
                "campaigns_are_equally_weighted_within_policy": True,
                "lifecycle_rows_pooled_before_profile_construction": False,
                "deterministic_retest_in_primary_estimand": False,
                "retest_role": "same_identity_deterministic_reliability_only",
                "failed_or_incomplete_cells_silently_coerced_to_complete": False,
                "provider_calls_must_be_zero_in_original_and_retest": True,
            },
        },
        "L": {
            "track_name": "discarded_state_latent_terminal_audit",
            "formal_population": {
                "campaign_cell_count": 10,
                "closed_lifecycle_count": 60,
                "observed_assay_count": 24,
                "discarded_lifecycle_count": 36,
                "shadow_evaluation_count": 36,
                "discard_opportunity_cell_count": 9,
                "no_discard_opportunity_cell_ids": ["cell-02"],
                "agent_provider_call_count": 0,
            },
            "record_schemas": {
                "latent_discard_unit": {
                    "primary_key": ["discard_id"],
                    "expected_row_count": 36,
                    "analysis_role": "primary",
                    "required_identity_fields": [
                        "cell_id",
                        "world_seed",
                        "information_arm",
                        "lifecycle_index",
                        "terminal_step",
                        "terminal_action_sha256",
                        "public_prefix_sha256",
                        "hidden_state_sha256",
                        "campaign_resource_snapshot_sha256",
                    ],
                },
                "terminal_lifecycle": {
                    "primary_key": ["cell_id", "lifecycle_index"],
                    "expected_row_count": 60,
                    "terminal_partition": {
                        "observed_assay": 24,
                        "original_discard": 36,
                    },
                    "analysis_role": "primary",
                },
                "latent_campaign_cell": {
                    "primary_key": ["cell_id"],
                    "expected_row_count": 10,
                    "discard_opportunity_row_count": 9,
                    "analysis_role": "primary",
                },
            },
            "estimand_fields": [
                {
                    "estimand_id": row["estimand_id"],
                    "unit": row["unit"],
                    "denominator": row["denominator"],
                    "value_unit_id": (
                        "dimensionless_fraction"
                        if "fraction" in str(row["estimand_id"])
                        or str(row["estimand_id"]).endswith(("precision", "recall"))
                        else "normalized_score"
                        if row["estimand_id"] == "latent_terminal_score"
                        else "normalized_score_difference"
                    ),
                }
                for row in estimands
            ],
            "counting_rules": {
                "primary_analysis_unit": "one frozen discarded lifecycle",
                "one_shadow_result_per_discard_id": True,
                "shadow_result_counts_as_original_agent_experiment": False,
                "shadow_result_counts_as_agent_assay_decision": False,
                "all_36_retained_in_fixed_denominators": True,
                "complete_case_primary_allowed": False,
                "unresolved_units_retained_with_reason_and_bounds": True,
                "zero_denominator_returns_null_with_denominator": True,
                "campaign_oracle_denominator_excludes_only_no_opportunity_cells": True,
                "campaign_oracle_opportunity_cell_count": 9,
            },
        },
    }


def build_work_i_data_contract(root: Path) -> dict[str, Any]:
    """Build the deterministic D01 contract from frozen F/V/L dependencies."""

    resolved = root.resolve()
    sources, source_bindings = _load_sources(resolved)
    source_gates = _validate_source_chain(resolved, sources)
    if not all(source_gates.values()):
        failed = [name for name, passed in source_gates.items() if not passed]
        raise WorkIDataContractError("F/V/L source chain is not freeze-ready: " + ", ".join(failed))
    latent = sources["latent_terminal_estimand_contract"]
    contract: dict[str, Any] = {
        "schema_id": CONTRACT_SCHEMA_ID,
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "status": "frozen",
        "purpose": (
            "Define one auditable schema, unit registry, and counting boundary for "
            "the incremental F/V/L evidence consumed by W1-D03 and later release tasks."
        ),
        "freeze": {
            "task_id": "W1-D01",
            "owner": "codex-1",
            "timing": "before W1-L05 formal shadow outcomes and before W1-D03 assembly",
            "mutable_after_freeze": False,
            "formal_latent_outcomes_read": False,
            "global_derived_data_regenerated": False,
            "global_evidence_dag_regenerated": False,
        },
        "source_bindings": source_bindings,
        "source_code_manifest": {
            path.as_posix(): file_sha256(resolved / path) for path in SOURCE_CODE_PATHS
        },
        "source_validation_gates": source_gates,
        "serialization_contract": {
            "encoding": "UTF-8",
            "canonical_json": "sort_keys=true,separators=(',',':'),ensure_ascii=false",
            "nonfinite_numbers_allowed": False,
            "missing_numeric_representation": None,
            "computational_precision": "preserve source numeric values without display rounding",
            "self_hash_rule": "remove contract_sha256 before canonical SHA-256",
            "source_row_hash_rule": (
                "canonical SHA-256 of the complete immutable source row before projection"
            ),
        },
        "artifact_roles": {
            "allowed": [
                "immutable_protocol_input",
                "immutable_formal_report",
                "immutable_summary_certificate",
                "immutable_delivery_manifest",
                "immutable_outcome_blind_audit",
                "immutable_synthetic_qualification",
            ],
            "rule": (
                "Protocol, formal, replay/retest, qualification, and derived roles "
                "remain explicit and may never be inferred from a filename."
            ),
        },
        "unit_registry": _unit_registry(),
        "common_record_fields": _common_fields(),
        "track_contracts": _track_contracts(latent),
        "cross_track_counting_rules": {
            "primary_units": {
                "F": "parent_child_pair",
                "V": "original_campaign_profile",
                "L": "discarded_lifecycle",
            },
            "never_pool_distinct_primary_units": True,
            "primitive_operations_are_repeated_events_not_independent_samples": True,
            "exact_replays_are_verification_not_additional_primary_units": True,
            "deterministic_retests_are_reliability_not_additional_primary_units": True,
            "synthetic_qualification_is_never_formal_evidence": True,
            "evaluator_shadow_is_not_an_original_agent_decision_or_experiment": True,
            "provider_call_count_is_reported_not_used_as_a_sample_size": True,
            "every_summary_discloses_numerator_denominator_and_unit": True,
            "row_deduplication_key": ["track", "record_type", "record_id"],
            "duplicate_primary_keys": "fatal_validation_error",
        },
        "nullability_and_failure": {
            "only_json_null_represents_missing_numeric_values": True,
            "nan_infinity_or_string_sentinels_forbidden": True,
            "structurally_inapplicable_metrics": (
                "null with an explicit applicability reason; never numeric zero"
            ),
            "zero_denominator": "null value plus exact zero denominator",
            "failed_or_incomplete_unit": (
                "retain identity, quality_status, failure reasons, and registered denominator"
            ),
            "latent_unresolved_unit": (
                "retain all 36 units; withhold registered point estimates where required "
                "and publish frozen worst/best/sharp bounds"
            ),
            "complete_case_substitution_for_registered_primary": False,
        },
        "relationship_constraints": [
            "F: exactly four traces join to each of six fork pairs: parent/child by "
            "original/exact_replay",
            "V: exactly six primary lifecycle rows join to each of 30 original campaign profiles",
            "V: exactly one same-identity retest joins to each original campaign and "
            "remains excluded",
            "L: 24 observed-assay plus 36 original-discard rows partition all 60 "
            "terminal lifecycles",
            "L: exactly one evaluator-shadow result joins to each of 36 discard_id values",
            "L: campaign-oracle summaries use nine discard-opportunity cells; cell-02 remains null",
        ],
        "derived_layer_requirements": {
            "consumer_task": "W1-D03",
            "required_contract_binding": "contract_sha256",
            "required_source_binding_fields": [
                "source_artifact_id",
                "source_artifact_sha256",
                "source_row_sha256",
            ],
            "immutable_manifest_required": True,
            "file_hashes_and_byte_counts_required": True,
            "record_counts_by_track_type_role_required": True,
            "raw_hidden_state_or_provider_payloads_allowed": False,
            "global_artifact_mutation_authorized_by_d01": False,
        },
        "claim_boundary": {
            "freezes_interfaces_not_results": True,
            "does_not_execute_worlds_agents_providers_or_shadow_assays": True,
            "does_not_change_any_frozen_protocol_estimand_threshold_or_outcome": True,
            "does_not_construct_the_final_derived_data_layer": True,
            "does_not_make_cross_track_performance_comparisons": True,
        },
    }
    contract["contract_sha256"] = data_contract_sha256(contract)
    return contract


def validate_work_i_data_contract(
    payload: Mapping[str, Any],
    *,
    root: Path | None = None,
) -> list[str]:
    """Validate both structure and exact frozen dependency/source bindings."""

    errors: list[str] = []
    if payload.get("schema_id") != CONTRACT_SCHEMA_ID:
        errors.append("unexpected schema ID")
    if payload.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append("unexpected schema version")
    if payload.get("contract_id") != CONTRACT_ID or payload.get("status") != "frozen":
        errors.append("unexpected contract identity or status")
    if payload.get("contract_sha256") != data_contract_sha256(payload):
        errors.append("contract self-hash mismatch")
    gates = payload.get("source_validation_gates")
    if not isinstance(gates, Mapping) or not gates or not all(gates.values()):
        errors.append("one or more source-validation gates failed")
    tracks = payload.get("track_contracts")
    if not isinstance(tracks, Mapping) or set(tracks) != {"F", "V", "L"}:
        errors.append("track contracts must be exactly F, V, and L")
    units = payload.get("unit_registry")
    if not isinstance(units, Mapping) or units != _unit_registry():
        errors.append("unit registry differs from the frozen registry")
    common = payload.get("common_record_fields")
    if not isinstance(common, Mapping) or common != _common_fields():
        errors.append("common record schema differs from the frozen schema")
    counting = payload.get("cross_track_counting_rules")
    if not isinstance(counting, Mapping) or any(
        counting.get(field) is not True
        for field in (
            "never_pool_distinct_primary_units",
            "primitive_operations_are_repeated_events_not_independent_samples",
            "exact_replays_are_verification_not_additional_primary_units",
            "deterministic_retests_are_reliability_not_additional_primary_units",
            "synthetic_qualification_is_never_formal_evidence",
            "evaluator_shadow_is_not_an_original_agent_decision_or_experiment",
            "provider_call_count_is_reported_not_used_as_a_sample_size",
            "every_summary_discloses_numerator_denominator_and_unit",
        )
    ):
        errors.append("cross-track counting rules are incomplete")
    nulls = payload.get("nullability_and_failure")
    if not isinstance(nulls, Mapping) or any(
        nulls.get(field) is not expected
        for field, expected in (
            ("only_json_null_represents_missing_numeric_values", True),
            ("nan_infinity_or_string_sentinels_forbidden", True),
            ("complete_case_substitution_for_registered_primary", False),
        )
    ):
        errors.append("nullability or failure rules changed")
    derived = payload.get("derived_layer_requirements")
    if not isinstance(derived, Mapping) or (
        derived.get("consumer_task") != "W1-D03"
        or derived.get("immutable_manifest_required") is not True
        or derived.get("raw_hidden_state_or_provider_payloads_allowed") is not False
        or derived.get("global_artifact_mutation_authorized_by_d01") is not False
    ):
        errors.append("derived-layer boundary changed")
    if root is not None:
        try:
            expected = build_work_i_data_contract(root)
        except WorkIDataContractError as exc:
            errors.append(str(exc))
        else:
            if dict(payload) != expected:
                errors.append("contract differs from deterministic frozen rebuild")
    return list(dict.fromkeys(errors))


__all__ = [
    "CONTRACT_ID",
    "CONTRACT_PATH",
    "CONTRACT_SCHEMA_ID",
    "CONTRACT_SCHEMA_VERSION",
    "REPORT_PATH",
    "SOURCE_CODE_PATHS",
    "SOURCE_SPECS",
    "WorkIDataContractError",
    "build_work_i_data_contract",
    "data_contract_sha256",
    "validate_work_i_data_contract",
]
