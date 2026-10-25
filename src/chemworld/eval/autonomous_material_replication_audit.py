"""Audit fresh Codex trajectory replicates within selected physical worlds."""

# ruff: noqa: RUF001 -- Chinese report strings intentionally use Chinese punctuation.

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceIntegrityError,
    CampaignResourceLedger,
)
from chemworld.data.logging import load_jsonl
from chemworld.eval.autonomous_material_campaign_audit import (
    NOMINAL_ARM,
    OPAQUE_ARM,
    _audit_cell,
    _paired_delta,
)
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)
from chemworld.eval.verify import verify_records

AUTONOMOUS_MATERIAL_REPLICATION_AUDIT_VERSION = (
    "chemworld-autonomous-material-trajectory-replication-audit-0.1"
)
INTERPRETATION_POLICY_VERSION = "chemworld-g2-v0.5-interpretation-policy-0.1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTERPRETATION_POLICY_PATH = Path(
    "configs/benchmark/"
    "g2_autonomous_electrochemical_material_seed1_seed3_r5_v0.5_"
    "interpretation_policy.json"
)
EXPECTED_MANIFEST_VERSION = "chemworld-g2-trajectory-replication-run-0.1"
EXPECTED_WORLD_SEEDS = (1, 3)
EXPECTED_REPLICATE_IDS = ("r01", "r02", "r03", "r04", "r05")
EXPECTED_CONDITIONS = {
    "anonymous_nominal_properties": NOMINAL_ARM,
    "opaque_codes": OPAQUE_ARM,
}
MAXIMUM_PRE_ACTION_PROVIDER_ATTEMPTS = 3
_RIGHT_CENSORED_CLASSIFICATIONS = {
    "terminal_right_censored_provider_failure",
    "terminal_right_censored_method_limit",
}
_PAIR_IDENTITY_FIELDS = (
    "world_seed",
    "world_id",
    "world_family_version",
    "mechanism_hash",
    "material_family_id",
    "material_family_sha256",
    "material_instance_sha256",
    "scoring_contract_hash",
    "workflow_mode",
    "observation_noise_mode",
    "observation_noise_namespace",
    "observation_seed",
    "resource_card_sha256",
    "code_hash",
    "pair_config_sha256",
)
_PAIRED_METRICS = (
    "best_final_score",
    "final_score_mean",
    "batch_final_assay_running_best_auc",
    "operation_attempt_running_best_auc",
    "budget_normalized_operation_attempt_running_best_auc",
    "global_best_discovery_fraction",
    "online_incumbent_retention_rate",
    "maximum_absolute_incumbent_drawdown",
    "terminal_to_global_best_ratio",
    "loss_episode_count",
    "recovered_loss_episode_count",
    "unresolved_loss_episode_count",
    "operation_count",
    "measurement_count",
)


class AutonomousMaterialReplicationAuditError(ValueError):
    """Raised when the fresh-trajectory matrix is not auditable as frozen."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AutonomousMaterialReplicationAuditError(
            f"invalid {label}: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise AutonomousMaterialReplicationAuditError(
            f"{label} must contain a JSON object: {path}"
        )
    return payload


def _portable_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    for root in (_REPOSITORY_ROOT, Path.cwd().resolve()):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _load_interpretation_policy(path: Path) -> dict[str, Any]:
    policy = _load_json_object(path, label="interpretation policy")
    if policy.get("schema_version") != INTERPRETATION_POLICY_VERSION:
        raise AutonomousMaterialReplicationAuditError(
            "unsupported G2 v0.5 interpretation policy schema"
        )
    if policy.get("status") != "outcome-blind-narrative-mapping-supplement":
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 interpretation policy is not the frozen outcome-blind supplement"
        )
    classification = policy.get("classification")
    if not isinstance(classification, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            "interpretation policy classification must be an object"
        )
    core_metrics = classification.get("core_trajectory_metrics")
    endpoint_metrics = classification.get("endpoint_diagnostics")
    if not isinstance(core_metrics, list) or not core_metrics:
        raise AutonomousMaterialReplicationAuditError(
            "interpretation policy must declare core trajectory metrics"
        )
    if not isinstance(endpoint_metrics, list):
        raise AutonomousMaterialReplicationAuditError(
            "interpretation policy endpoint diagnostics must be a list"
        )
    unknown = (set(core_metrics) | set(endpoint_metrics)) - set(_PAIRED_METRICS)
    if unknown:
        raise AutonomousMaterialReplicationAuditError(
            "interpretation policy contains unknown paired metrics: "
            + ", ".join(sorted(unknown))
        )
    branch_ids = [
        branch.get("branch_id")
        for branch in classification.get("branch_precedence", [])
        if isinstance(branch, Mapping)
    ]
    expected_branches = [
        "insufficient_paired_coverage",
        "opposing_world_conditioned_repeatability",
        "frequent_within_world_reversal",
        "metric_specific_or_nonopposing_repeatability",
    ]
    if branch_ids != expected_branches:
        raise AutonomousMaterialReplicationAuditError(
            "interpretation branch precedence is not the frozen four-branch policy"
        )
    return policy


def _resolved_interpretation_policy_path(path: str | Path | None) -> Path:
    policy_path = Path(path or DEFAULT_INTERPRETATION_POLICY_PATH)
    if policy_path.is_absolute() or policy_path.is_file():
        return policy_path
    repository_candidate = _REPOSITORY_ROOT / policy_path
    return repository_candidate if repository_candidate.is_file() else policy_path


def validate_interpretation_binding(
    report: Mapping[str, Any],
    *,
    interpretation_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Fail closed unless an audit is bound to the frozen narrative mapping."""

    policy_file = _resolved_interpretation_policy_path(interpretation_policy_path)
    policy = _load_interpretation_policy(policy_file)
    interpretation = report.get("interpretation")
    if not isinstance(interpretation, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 audit has no interpretation block"
        )
    mapping = interpretation.get("mapping_policy")
    if not isinstance(mapping, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 audit has no interpretation-policy binding"
        )
    expected_mapping = {
        "sha256": file_sha256(policy_file),
        "schema_version": policy["schema_version"],
        "status": policy["status"],
    }
    mismatched = [
        key for key, expected in expected_mapping.items() if mapping.get(key) != expected
    ]
    if mismatched:
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 interpretation-policy binding mismatch: "
            + ", ".join(mismatched)
        )
    selected = interpretation.get("selected_branch")
    if not isinstance(selected, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 audit has no selected interpretation branch"
        )
    branches = {
        branch["branch_id"]: branch
        for branch in policy["classification"]["branch_precedence"]
    }
    branch = branches.get(selected.get("branch_id"))
    if branch is None:
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 audit selected an unknown interpretation branch"
        )
    if selected.get("manuscript_language") != branch.get("manuscript_language"):
        raise AutonomousMaterialReplicationAuditError(
            "G2 v0.5 selected-branch language differs from the frozen policy"
        )
    return {
        "policy_path": _portable_path(policy_file),
        "policy_sha256": expected_mapping["sha256"],
        "branch_id": branch["branch_id"],
        "manuscript_language": branch["manuscript_language"],
    }


def _resolve_under(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AutonomousMaterialReplicationAuditError(f"missing {label}")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise AutonomousMaterialReplicationAuditError(
            f"{label} escapes the replication root: {relative}"
        )
    return candidate


def _canonical_config_hash(config: Mapping[str, Any]) -> str:
    payload = dict(config)
    payload.pop("config_sha256", None)
    return canonical_json_sha256(payload)


def _validate_attempt(
    *,
    manifest_root: Path,
    cell: Mapping[str, Any],
    attempt: Mapping[str, Any],
) -> dict[str, Any]:
    attempt_root = _resolve_under(
        manifest_root,
        attempt.get("attempt_dir"),
        label="attempt_dir",
    )
    config_path = attempt_root / "run_config.json"
    summary_path = attempt_root / "run_summary.json"
    environment_path = attempt_root / "environment_contract.json"
    for path, label in (
        (config_path, "run config"),
        (summary_path, "run summary"),
        (environment_path, "environment contract"),
    ):
        if not path.is_file():
            raise AutonomousMaterialReplicationAuditError(
                f"{cell['cell_id']}/{attempt.get('attempt_id')}: missing {label}"
            )
    config = _load_json_object(config_path, label="attempt run config")
    summary = _load_json_object(summary_path, label="attempt run summary")
    environment = _load_json_object(
        environment_path,
        label="attempt environment contract",
    )
    file_hash_checks = {
        "config": attempt.get("config_sha256") == file_sha256(config_path),
        "summary": attempt.get("summary_sha256") == file_sha256(summary_path),
        "environment": attempt.get("environment_contract_sha256")
        == file_sha256(environment_path),
    }
    failed_file_hashes = [
        name for name, passed in file_hash_checks.items() if not passed
    ]
    if failed_file_hashes:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell['cell_id']}/{attempt.get('attempt_id')}: file hash mismatch: "
            + ", ".join(failed_file_hashes)
        )
    declared_config_hash = config.get("config_sha256")
    checks = {
        "declared_config_hash": declared_config_hash
        == _canonical_config_hash(config),
        "config_cell": config.get("cell") == dict(cell),
        "config_world_seed": config.get("world_seed")
        == int(cell["world_seed"]),
        "config_replicate": config.get("trajectory_replicate_id")
        == str(cell["trajectory_replicate_id"]),
        "config_agent_seed": config.get("agent_seed")
        == int(cell["agent_seed"]),
        "config_arm": config.get("condition_id") == cell["condition_id"],
        "summary_cell": summary.get("cell") == dict(cell),
        "summary_config_hash": summary.get("config_sha256")
        == declared_config_hash,
        "summary_pair_hash": summary.get("pair_config_sha256")
        == config.get("pair_config_sha256"),
        "summary_status": summary.get("run_status")
        == attempt.get("run_status"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell['cell_id']}/{attempt.get('attempt_id')}: identity mismatch: "
            + ", ".join(failed)
        )
    accepted = _accepted_operation_count(summary)
    if accepted != int(attempt.get("accepted_operation_count", 0) or 0):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell['cell_id']}/{attempt.get('attempt_id')}: "
            "accepted-operation count mismatch"
        )
    trajectory_path = attempt_root / "trajectory.jsonl"
    if trajectory_path.is_file():
        records = load_jsonl(trajectory_path)
        if len(records) != accepted:
            raise AutonomousMaterialReplicationAuditError(
                f"{cell['cell_id']}/{attempt.get('attempt_id')}: "
                "trajectory record count mismatch"
            )
        trajectory_hash = file_sha256(trajectory_path)
        if attempt.get("trajectory_sha256") != trajectory_hash:
            raise AutonomousMaterialReplicationAuditError(
                f"{cell['cell_id']}/{attempt.get('attempt_id')}: "
                "trajectory hash mismatch"
            )
        if summary.get("trajectory_sha256") not in {None, trajectory_hash}:
            raise AutonomousMaterialReplicationAuditError(
                f"{cell['cell_id']}/{attempt.get('attempt_id')}: "
                "summary trajectory hash mismatch"
            )
    elif accepted > 0:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell['cell_id']}/{attempt.get('attempt_id')}: "
            "accepted operations exist without trajectory bytes"
        )
    classification = str(attempt.get("classification"))
    expected_classification = _attempt_classification(summary)
    if classification != expected_classification:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell['cell_id']}/{attempt.get('attempt_id')}: "
            "attempt classification mismatch"
        )
    return {
        "attempt_id": attempt.get("attempt_id"),
        "attempt_dir": attempt_root.relative_to(manifest_root).as_posix(),
        "classification": classification,
        "run_status": summary.get("run_status"),
        "accepted_operation_count": accepted,
        "config": config,
        "summary": summary,
        "environment_contract": environment,
        "hashes_verified": True,
    }


def _attempt_classification(summary: Mapping[str, Any]) -> str:
    status = str(summary.get("run_status") or "missing")
    accepted = _accepted_operation_count(summary)
    if status == "completed":
        return "completed"
    if status == "provider_infrastructure_failure" and accepted == 0:
        return "retryable_pre_action_provider_failure"
    if status == "provider_infrastructure_failure" and accepted > 0:
        return "terminal_right_censored_provider_failure"
    if status == "method_resource_limit_exhausted" and accepted > 0:
        return "terminal_right_censored_method_limit"
    return "audit_required"


def _accepted_operation_count(summary: Mapping[str, Any]) -> int:
    raw = summary.get("accepted_operation_count")
    if raw is None:
        behavior = summary.get("behavior")
        if isinstance(behavior, Mapping):
            raw = behavior.get("operation_count")
    return int(raw or 0)


def _validate_state_attempt_policy(
    state: Mapping[str, Any],
    *,
    manifest_root: Path,
) -> dict[str, Any]:
    cell = state.get("cell")
    attempts = state.get("attempts")
    if not isinstance(cell, Mapping) or not isinstance(attempts, list):
        raise AutonomousMaterialReplicationAuditError(
            "each manifest cell requires cell identity and attempts"
        )
    cell_id = str(cell.get("cell_id"))
    if not 1 <= len(attempts) <= MAXIMUM_PRE_ACTION_PROVIDER_ATTEMPTS:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: finalized cell must contain one to three attempts"
        )
    validated = [
        _validate_attempt(
            manifest_root=manifest_root,
            cell=cell,
            attempt=attempt,
        )
        for attempt in attempts
        if isinstance(attempt, Mapping)
    ]
    if len(validated) != len(attempts):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: attempt entries must be objects"
        )
    predecessor_classes = [item["classification"] for item in validated[:-1]]
    if any(
        value != "retryable_pre_action_provider_failure"
        for value in predecessor_classes
    ):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: a non-retryable attempt was selectively replaced"
        )
    final = validated[-1]
    state_name = str(state.get("state"))
    expected_final_classes = {
        "completed": {"completed"},
        "right_censored": _RIGHT_CENSORED_CLASSIFICATIONS,
    }
    if state_name not in expected_final_classes:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: matrix is not finalized; state={state_name}"
        )
    if final["classification"] not in expected_final_classes[state_name]:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: final attempt does not match state={state_name}"
        )
    if state.get("authoritative_attempt_dir") != attempts[-1].get("attempt_dir"):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: authoritative attempt is not the immutable final attempt"
        )
    return {
        "cell": dict(cell),
        "state": state_name,
        "attempt_count": len(validated),
        "pre_action_provider_retry_count": len(validated) - 1,
        "attempts": validated,
        "authoritative": final,
        "selection_policy_verified": True,
    }


def _pair_physical_audit(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> dict[str, Any]:
    left_cell = left["cell"]
    right_cell = right["cell"]
    left_attempt = left["authoritative"]
    right_attempt = right["authoritative"]
    left_environment = left_attempt["environment_contract"]
    right_environment = right_attempt["environment_contract"]
    left_evaluator = left_environment.get("evaluator_identity", {})
    right_evaluator = right_environment.get("evaluator_identity", {})
    left_public = left_environment.get("public_contract", {})
    right_public = right_environment.get("public_contract", {})
    evaluator_fields = (
        "world_id",
        "mechanism_hash",
        "electrochemical_material_family_id",
        "electrochemical_material_family_sha256",
        "electrochemical_material_instance_sha256",
        "observation_noise_mode",
        "observation_noise_namespace",
    )
    public_fields = (
        "task_contract_hash",
        "runtime_profile_hash",
        "scoring_contract_hash",
        "observation_contract_hash",
        "workflow_mode",
    )
    invariants = {
        "world_seed": left_cell["world_seed"] == right_cell["world_seed"],
        "trajectory_replicate_id": left_cell["trajectory_replicate_id"]
        == right_cell["trajectory_replicate_id"],
        "agent_seed": left_cell["agent_seed"] == right_cell["agent_seed"],
        "conditions_are_distinct": left_cell["condition_id"]
        != right_cell["condition_id"],
        "pair_config_sha256": left_attempt["config"].get("pair_config_sha256")
        == right_attempt["config"].get("pair_config_sha256"),
    }
    invariants.update(
        {
            field: left_evaluator.get(field) == right_evaluator.get(field)
            for field in evaluator_fields
        }
    )
    invariants.update(
        {
            field: left_public.get(field) == right_public.get(field)
            for field in public_fields
        }
    )
    return {
        "world_seed": int(left_cell["world_seed"]),
        "trajectory_replicate_id": str(left_cell["trajectory_replicate_id"]),
        "agent_seed": int(left_cell["agent_seed"]),
        "conditions": [left_cell["condition_id"], right_cell["condition_id"]],
        "invariants": invariants,
        "passed": all(invariants.values()),
    }


def _completed_cell_audit(
    *,
    state_audit: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_root: Path,
    expected_vessels: int,
) -> dict[str, Any]:
    cell = state_audit["cell"]
    descriptor = {
        "cell_id": cell["cell_id"],
        "world_seed": int(cell["world_seed"]),
        "arm": cell["condition_id"],
        "material_information": cell["material_information"],
        "run_dir": state_audit["authoritative"]["attempt_dir"],
        "config_path": "run_config.json",
        "summary_path": "run_summary.json",
        "trajectory_path": "trajectory.jsonl",
        "campaign_resource_ledger_path": "campaign_resource_ledger.json",
        "exact_replay_path": "exact_replay.json",
    }
    audited = _audit_cell(
        cell=descriptor,
        manifest=manifest,
        manifest_dir=manifest_root,
        expected_batches=expected_vessels,
        allow_incomplete=False,
    )
    audited.update(
        {
            "trajectory_replicate_id": str(cell["trajectory_replicate_id"]),
            "agent_seed": int(cell["agent_seed"]),
            "pair_order": int(cell["pair_order"]),
            "within_pair_order": int(cell["within_pair_order"]),
            "attempt_count": int(state_audit["attempt_count"]),
            "pre_action_provider_retry_count": int(
                state_audit["pre_action_provider_retry_count"]
            ),
        }
    )
    return audited


def _nested(payload: Mapping[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _reconstruct_right_censored_resource_and_replay(
    *,
    config: Mapping[str, Any],
    trajectory_path: Path,
    cell_id: str,
    expected_vessels: int,
) -> dict[str, Any]:
    records = load_jsonl(trajectory_path)
    if not records:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: action-bearing right-censored trajectory is empty"
        )
    raw_card = config.get("campaign_resource_card")
    if not isinstance(raw_card, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored config lacks campaign resource card"
        )
    try:
        ledger = CampaignResourceLedger(CampaignResourceCard.from_dict(raw_card))
        declared_hashes: list[str] = []
        for index, record in enumerate(records, start=1):
            resources = _nested(
                record,
                "agent_view.tool_json.campaign_state.campaign_resources",
            )
            if not isinstance(resources, Mapping):
                raise CampaignResourceIntegrityError(
                    f"step {index} lacks public campaign resources"
                )
            receipt = resources.get("latest_receipt")
            if not isinstance(receipt, Mapping):
                raise CampaignResourceIntegrityError(
                    f"step {index} lacks campaign resource receipt"
                )
            preflight = receipt.get("preflight")
            outcome_delta = receipt.get("outcome_delta")
            action = record.get("action")
            if (
                not isinstance(preflight, Mapping)
                or not isinstance(outcome_delta, Mapping)
                or not isinstance(action, Mapping)
            ):
                raise CampaignResourceIntegrityError(
                    f"step {index} has malformed action/resource receipt"
                )
            proposed = preflight.get("proposed_delta")
            starts_vessel = bool(
                isinstance(proposed, Mapping)
                and int(proposed.get("vessel_starts", 0)) == 1
            )
            event_id = str(receipt.get("event_id") or "")
            replayed_preflight = ledger.preflight(
                event_id,
                action,
                starts_vessel=starts_vessel,
            )
            if replayed_preflight.to_dict() != dict(preflight):
                raise CampaignResourceIntegrityError(
                    f"step {index} preflight replay mismatch"
                )
            report = outcome_delta.get("report_only")
            if not isinstance(report, Mapping):
                raise CampaignResourceIntegrityError(
                    f"step {index} resource report is missing"
                )
            replayed_delta = ledger.record_outcome(
                event_id,
                action,
                {
                    "operation_committed": receipt.get("operation_committed")
                    is True,
                    "campaign_resource_report_delta": dict(report),
                },
                starts_vessel=starts_vessel,
            )
            if replayed_delta.to_dict() != dict(outcome_delta):
                raise CampaignResourceIntegrityError(
                    f"step {index} outcome replay mismatch"
                )
            declared_hash = resources.get("ledger_sha256")
            if not isinstance(declared_hash, str):
                raise CampaignResourceIntegrityError(
                    f"step {index} lacks public ledger hash"
                )
            replayed_hash = ledger.snapshot()["ledger_sha256"]
            if replayed_hash != declared_hash:
                raise CampaignResourceIntegrityError(
                    f"step {index} public ledger hash replay mismatch"
                )
            declared_hashes.append(declared_hash)
        snapshot = ledger.snapshot()
    except (CampaignResourceIntegrityError, KeyError, TypeError, ValueError) as error:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored campaign resource replay failed: {error}"
        ) from error

    state = snapshot["state"]
    starts = int(state["vessel_starts"])
    final_assays = int(state["final_assays"])
    discarded = int(state.get("discarded_batches", 0))
    closed = final_assays + discarded
    if not 0 <= closed <= starts <= expected_vessels:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: impossible right-censored vessel accounting"
        )
    verification = verify_records(records)
    if not verification.verified:
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored deterministic transition replay failed"
        )
    return {
        "resource_ledger": {
            "verified": True,
            "ledger_sha256": snapshot["ledger_sha256"],
            "all_public_step_hashes_verified": len(declared_hashes) == len(records),
            "operation_attempts": int(state["operation_attempts"]),
            "vessel_starts": starts,
            "final_assays": final_assays,
            "discarded_batches": discarded,
            "closed_batches": closed,
            "started_right_censored_vessels": starts - closed,
            "unstarted_vessel_opportunities": expected_vessels - starts,
            "nonfinal_instrument_uses": int(state["nonfinal_instrument_uses"]),
            "stocks_used": state["stocks_used"],
            "report_only": state["report_only"],
        },
        "exact_replay": {
            "verified": True,
            "trajectory_sha256": file_sha256(trajectory_path),
            "trajectory_record_count": len(records),
            "campaign_resource_ledger_sha256": snapshot["ledger_sha256"],
            **verification.to_dict(),
        },
    }


def _audit_right_censored_provider_sessions(
    summary: Mapping[str, Any],
    *,
    cell_id: str,
) -> dict[str, Any]:
    receipts = summary.get("provider_receipts")
    resources = summary.get("method_resources")
    failure = summary.get("failure")
    if not isinstance(receipts, list) or not all(
        isinstance(receipt, Mapping) for receipt in receipts
    ):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored provider receipts are malformed"
        )
    if not receipts or not isinstance(resources, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored method accounting is incomplete"
        )
    if not isinstance(failure, Mapping):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored provider failure summary is absent"
        )
    completed = receipts[:-1]
    interrupted = receipts[-1]

    def integrity_verified(receipt: Mapping[str, Any]) -> bool:
        return any(
            receipt.get(key) is True
            for key in (
                "experiment_tool_integrity_verified_after_session",
                "lab_tool_integrity_verified_after_session",
                "mcp_tool_integrity_verified_after_session",
            )
        ) and all(
            receipt.get(key) is True
            for key in (
                "experiment_tool_integrity_verified_after_session",
                "lab_tool_integrity_verified_after_session",
                "mcp_tool_integrity_verified_after_session",
            )
            if key in receipt
        )

    completed_failures = [
        index
        for index, receipt in enumerate(completed, start=1)
        if not (
            receipt.get("status") == "completed"
            and receipt.get("return_code") == 0
            and receipt.get("terminal_reason")
            in {"experiment_complete", "batch_discarded"}
            and receipt.get("final_payload_valid") is True
            and receipt.get("usage_complete") is True
            and integrity_verified(receipt)
        )
    ]
    interrupted_valid = (
        interrupted.get("status") != "completed"
        and isinstance(interrupted.get("failure_type"), str)
        and bool(interrupted.get("failure_type"))
        and integrity_verified(interrupted)
    )
    resource_checks = {
        "provider_usage_pending_false": resources.get("provider_usage_pending")
        is False,
        "in_flight_model_calls_zero": int(
            resources.get("in_flight_model_call_count", -1)
        )
        == 0,
        "provider_call_accounting_complete": resources.get(
            "provider_call_accounting_complete"
        )
        is True,
        "model_call_count_matches_receipts": int(
            resources.get("model_call_count", -1)
        )
        == len(receipts),
        "provider_failure_recorded": int(failure.get("provider_failure_count", 0))
        > 0,
    }
    if completed_failures or not interrupted_valid or not all(resource_checks.values()):
        raise AutonomousMaterialReplicationAuditError(
            f"{cell_id}: right-censored provider session audit failed; "
            f"completed_failures={completed_failures}, "
            f"interrupted_valid={interrupted_valid}, checks={resource_checks}"
        )
    return {
        "verified": True,
        "receipt_count": len(receipts),
        "completed_session_count": len(completed),
        "interrupted_session_count": 1,
        "interrupted_failure_type": interrupted.get("failure_type"),
        "completed_session_usage_accounting_complete": True,
        "aggregate_token_accounting_complete": resources.get(
            "provider_token_accounting_complete"
        )
        is True,
        "aggregate_monetary_accounting_complete": resources.get(
            "monetary_accounting_complete"
        )
        is True,
        "resource_checks": resource_checks,
    }


def _right_censored_cell_audit(
    *,
    state_audit: Mapping[str, Any],
    manifest_root: Path,
    expected_vessels: int,
) -> dict[str, Any]:
    cell = state_audit["cell"]
    authoritative = state_audit["authoritative"]
    attempt_root = manifest_root / str(authoritative["attempt_dir"])
    trajectory_path = attempt_root / "trajectory.jsonl"
    config = authoritative["config"]
    summary = authoritative["summary"]
    cell_id = str(cell["cell_id"])
    reconstructed = _reconstruct_right_censored_resource_and_replay(
        config=config,
        trajectory_path=trajectory_path,
        cell_id=cell_id,
        expected_vessels=expected_vessels,
    )
    provider = _audit_right_censored_provider_sessions(summary, cell_id=cell_id)
    return {
        "cell_id": cell_id,
        "world_seed": int(cell["world_seed"]),
        "trajectory_replicate_id": str(cell["trajectory_replicate_id"]),
        "condition_id": str(cell["condition_id"]),
        "agent_seed": int(cell["agent_seed"]),
        "run_status": summary.get("run_status"),
        "accepted_operation_count": int(authoritative["accepted_operation_count"]),
        "attempt_count": int(state_audit["attempt_count"]),
        "provider_sessions": provider,
        **reconstructed,
    }


def _arm(cell: Mapping[str, Any]) -> str:
    return EXPECTED_CONDITIONS[str(cell["condition_id"])]


def _summary(values: Sequence[float]) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    tolerance = 1e-12
    positive = sum(value > tolerance for value in numeric)
    negative = sum(value < -tolerance for value in numeric)
    zero = len(numeric) - positive - negative
    return {
        "n": len(numeric),
        "values": numeric,
        "mean": statistics.fmean(numeric) if numeric else None,
        "median": statistics.median(numeric) if numeric else None,
        "minimum": min(numeric) if numeric else None,
        "maximum": max(numeric) if numeric else None,
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "sign_consistency": (
            max(positive, negative, zero) / len(numeric) if numeric else None
        ),
    }


def _world_summaries(pair_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for world_seed in EXPECTED_WORLD_SEEDS:
        world_rows = [
            row for row in pair_rows if int(row["world_seed"]) == world_seed
        ]
        completed = [row for row in world_rows if row["pair_complete"]]
        result[str(world_seed)] = {
            "planned_pair_count": len(world_rows),
            "completed_pair_count": len(completed),
            "right_censored_pair_count": len(world_rows) - len(completed),
            "paired_metrics": {
                metric: _summary(
                    [row["nominal_minus_opaque"][metric] for row in completed]
                )
                for metric in _PAIRED_METRICS
            },
        }
    return result


def _metric_repeatability_class(
    summary: Mapping[str, Any],
    *,
    threshold: float,
    tolerance: float,
) -> str:
    n = int(summary["n"])
    if n <= 0:
        return "mixed"
    median = float(summary["median"])
    if int(summary["positive_count"]) / n >= threshold and median > tolerance:
        return "directionally_positive"
    if int(summary["negative_count"]) / n >= threshold and median < -tolerance:
        return "directionally_negative"
    if int(summary["zero_count"]) / n >= threshold and abs(median) <= tolerance:
        return "stable_zero"
    return "mixed"


def _select_interpretation_branch(
    world_summaries: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    classification = policy["classification"]
    minimum_pairs = int(classification["minimum_completed_pairs_per_world"])
    threshold = float(classification["directional_consistency_threshold"])
    tolerance = float(classification["zero_tolerance"])
    core_metrics = [str(metric) for metric in classification["core_trajectory_metrics"]]
    endpoint_metrics = [str(metric) for metric in classification["endpoint_diagnostics"]]
    metric_classes: dict[str, dict[str, str]] = {}
    completed_pairs: dict[str, int] = {}
    for world_seed in EXPECTED_WORLD_SEEDS:
        world_key = str(world_seed)
        world = world_summaries[world_key]
        completed_pairs[world_key] = int(world["completed_pair_count"])
        metric_classes[world_key] = {
            metric: _metric_repeatability_class(
                world["paired_metrics"][metric],
                threshold=threshold,
                tolerance=tolerance,
            )
            for metric in (*core_metrics, *endpoint_metrics)
        }

    opposite_metrics = []
    for metric in core_metrics:
        seed_1 = metric_classes["1"][metric]
        seed_3 = metric_classes["3"][metric]
        if {seed_1, seed_3} == {
            "directionally_positive",
            "directionally_negative",
        }:
            opposite_metrics.append(metric)
    mixed_world_metric_count = sum(
        metric_classes[str(world_seed)][metric] == "mixed"
        for world_seed in EXPECTED_WORLD_SEEDS
        for metric in core_metrics
    )

    if any(value < minimum_pairs for value in completed_pairs.values()):
        branch_id = "insufficient_paired_coverage"
    elif len(opposite_metrics) >= 3:
        branch_id = "opposing_world_conditioned_repeatability"
    elif mixed_world_metric_count >= 4:
        branch_id = "frequent_within_world_reversal"
    else:
        branch_id = "metric_specific_or_nonopposing_repeatability"

    branch = next(
        item
        for item in classification["branch_precedence"]
        if item["branch_id"] == branch_id
    )
    return {
        "branch_id": branch_id,
        "manuscript_language": branch["manuscript_language"],
        "descriptive_only": True,
        "completed_pairs_by_world": completed_pairs,
        "minimum_completed_pairs_per_world": minimum_pairs,
        "directional_consistency_threshold": threshold,
        "core_trajectory_metrics": core_metrics,
        "endpoint_diagnostics": endpoint_metrics,
        "world_metric_classifications": metric_classes,
        "opposite_direction_core_metrics": opposite_metrics,
        "opposite_direction_core_metric_count": len(opposite_metrics),
        "mixed_world_by_core_metric_count": mixed_world_metric_count,
        "world_by_core_metric_count": len(EXPECTED_WORLD_SEEDS) * len(core_metrics),
    }


def audit_autonomous_material_trajectory_replication(
    manifest_path: str | Path,
    *,
    expected_vessels_per_cell: int = 6,
    interpretation_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the frozen 2-world x 5-trajectory x 2-arm matrix."""

    manifest_file = Path(manifest_path)
    policy_file = _resolved_interpretation_policy_path(interpretation_policy_path)
    interpretation_policy = _load_interpretation_policy(policy_file)
    manifest_root = manifest_file.resolve().parent
    manifest = _load_json_object(manifest_file, label="replication manifest")
    if manifest.get("schema_version") != EXPECTED_MANIFEST_VERSION:
        raise AutonomousMaterialReplicationAuditError(
            "unsupported replication manifest schema"
        )
    unhashed_manifest = dict(manifest)
    declared_manifest_hash = unhashed_manifest.pop("manifest_sha256", None)
    if declared_manifest_hash != canonical_json_sha256(unhashed_manifest):
        raise AutonomousMaterialReplicationAuditError(
            "replication manifest content hash is invalid"
        )
    if manifest.get("world_seeds") != list(EXPECTED_WORLD_SEEDS):
        raise AutonomousMaterialReplicationAuditError(
            "replication manifest world seeds are not frozen to [1, 3]"
        )
    if manifest.get("trajectory_replicate_ids") != list(
        EXPECTED_REPLICATE_IDS
    ):
        raise AutonomousMaterialReplicationAuditError(
            "replication ids are not frozen to r01 through r05"
        )
    raw_states = manifest.get("cells")
    if not isinstance(raw_states, list) or len(raw_states) != 20:
        raise AutonomousMaterialReplicationAuditError(
            "replication manifest must contain exactly twenty cells"
        )
    state_audits = [
        _validate_state_attempt_policy(state, manifest_root=manifest_root)
        for state in raw_states
        if isinstance(state, Mapping)
    ]
    if len(state_audits) != len(raw_states):
        raise AutonomousMaterialReplicationAuditError(
            "replication cell states must be objects"
        )
    keyed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for state in state_audits:
        cell = state["cell"]
        key = (
            int(cell["world_seed"]),
            str(cell["trajectory_replicate_id"]),
            _arm(cell),
        )
        if key in keyed:
            raise AutonomousMaterialReplicationAuditError(
                f"duplicate world/replicate/arm cell: {key}"
            )
        keyed[key] = state
    expected_keys = {
        (world_seed, replicate_id, arm)
        for world_seed in EXPECTED_WORLD_SEEDS
        for replicate_id in EXPECTED_REPLICATE_IDS
        for arm in (OPAQUE_ARM, NOMINAL_ARM)
    }
    if set(keyed) != expected_keys:
        raise AutonomousMaterialReplicationAuditError(
            "replication world/replicate/arm coverage is incomplete"
        )

    completed_cell_audits: dict[tuple[int, str, str], dict[str, Any]] = {}
    right_censored_cell_audits: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key, state in keyed.items():
        if state["state"] == "completed":
            completed_cell_audits[key] = _completed_cell_audit(
                state_audit=state,
                manifest=manifest,
                manifest_root=manifest_root,
                expected_vessels=expected_vessels_per_cell,
            )
        elif state["state"] == "right_censored":
            right_censored_cell_audits[key] = _right_censored_cell_audit(
                state_audit=state,
                manifest_root=manifest_root,
                expected_vessels=expected_vessels_per_cell,
            )

    pair_rows: list[dict[str, Any]] = []
    for world_seed in EXPECTED_WORLD_SEEDS:
        for replicate_id in EXPECTED_REPLICATE_IDS:
            opaque_state = keyed[(world_seed, replicate_id, OPAQUE_ARM)]
            nominal_state = keyed[(world_seed, replicate_id, NOMINAL_ARM)]
            physical = _pair_physical_audit(opaque_state, nominal_state)
            if not physical["passed"]:
                failed = [
                    name
                    for name, passed in physical["invariants"].items()
                    if not passed
                ]
                raise AutonomousMaterialReplicationAuditError(
                    f"world {world_seed}/{replicate_id}: physical pairing failed: "
                    + ", ".join(failed)
                )
            pair_complete = (
                opaque_state["state"] == nominal_state["state"] == "completed"
            )
            row: dict[str, Any] = {
                "world_seed": world_seed,
                "trajectory_replicate_id": replicate_id,
                "agent_seed": physical["agent_seed"],
                "opaque_state": opaque_state["state"],
                "nominal_state": nominal_state["state"],
                "pair_complete": pair_complete,
                "physical_pairing": physical,
            }
            if pair_complete:
                opaque = completed_cell_audits[
                    (world_seed, replicate_id, OPAQUE_ARM)
                ]
                nominal = completed_cell_audits[
                    (world_seed, replicate_id, NOMINAL_ARM)
                ]
                identity_mismatches = [
                    field
                    for field in _PAIR_IDENTITY_FIELDS
                    if opaque["identity"][field] != nominal["identity"][field]
                ]
                if identity_mismatches:
                    raise AutonomousMaterialReplicationAuditError(
                        f"world {world_seed}/{replicate_id}: audited identity mismatch: "
                        + ", ".join(identity_mismatches)
                    )
                row.update(_paired_delta(nominal, opaque))
            else:
                row["nominal_minus_opaque"] = None
            pair_rows.append(row)

    right_censored_cell_ids = [
        state["cell"]["cell_id"]
        for state in state_audits
        if state["state"] == "right_censored"
    ]
    world_summaries = _world_summaries(pair_rows)
    selected_interpretation = _select_interpretation_branch(
        world_summaries,
        interpretation_policy,
    )
    report: dict[str, Any] = {
        "schema_version": AUTONOMOUS_MATERIAL_REPLICATION_AUDIT_VERSION,
        "status": (
            "completed_audited_fresh_trajectory_replication"
            if not right_censored_cell_ids
            else "completed_audited_fresh_trajectory_replication_with_right_censoring"
        ),
        "formal_result": False,
        "confirmatory_claim_allowed": False,
        "manifest": {
            "path": str(manifest_file),
            "sha256": file_sha256(manifest_file),
            "declared_manifest_sha256": manifest.get("manifest_sha256"),
        },
        "matrix": {
            "world_seeds": list(EXPECTED_WORLD_SEEDS),
            "trajectory_replicate_ids": list(EXPECTED_REPLICATE_IDS),
            "planned_cell_count": 20,
            "completed_cell_count": len(completed_cell_audits),
            "right_censored_cell_count": len(right_censored_cell_ids),
            "right_censored_cell_ids": right_censored_cell_ids,
            "all_attempt_selection_policies_verified": all(
                state["selection_policy_verified"] for state in state_audits
            ),
            "all_physical_pairs_verified": all(
                row["physical_pairing"]["passed"] for row in pair_rows
            ),
            "completed_pair_count": sum(row["pair_complete"] for row in pair_rows),
            "all_terminal_cells_resource_replay_verified": all(
                cell["resource_ledger"]["verified"]
                and cell["exact_replay"]["verified"]
                for cell in (
                    *completed_cell_audits.values(),
                    *right_censored_cell_audits.values(),
                )
            ),
        },
        "attempt_audits": [
            {
                "cell": state["cell"],
                "state": state["state"],
                "attempt_count": state["attempt_count"],
                "pre_action_provider_retry_count": state[
                    "pre_action_provider_retry_count"
                ],
                "attempts": [
                    {
                        key: attempt[key]
                        for key in (
                            "attempt_id",
                            "attempt_dir",
                            "classification",
                            "run_status",
                            "accepted_operation_count",
                            "hashes_verified",
                        )
                    }
                    for attempt in state["attempts"]
                ],
                "selection_policy_verified": state[
                    "selection_policy_verified"
                ],
            }
            for state in state_audits
        ],
        "completed_cells": [
            completed_cell_audits[key] for key in sorted(completed_cell_audits)
        ],
        "right_censored_cells": [
            right_censored_cell_audits[key]
            for key in sorted(right_censored_cell_audits)
        ],
        "paired_trajectories": pair_rows,
        "within_world_descriptive_aggregates": world_summaries,
        "interpretation": {
            "analysis_unit": "fixed physical world by fresh trajectory replicate",
            "fresh_replicates_per_selected_world": 5,
            "selected_world_count": 2,
            "provider_sampling_seed_controlled": False,
            "development_trajectory_included": False,
            "descriptive_only": True,
            "general_world_effect_allowed": False,
            "mapping_policy": {
                "path": _portable_path(policy_file),
                "sha256": file_sha256(policy_file),
                "schema_version": interpretation_policy["schema_version"],
                "status": interpretation_policy["status"],
            },
            "selected_branch": selected_interpretation,
            "caveat": (
                "Seed 1 and seed 3 were selected from development results because "
                "their observed directions differed. Replicates characterize "
                "within-world Codex trajectory variability; they do not turn two "
                "selected worlds into a population-level material-information test."
            ),
        },
    }
    report["audit_sha256"] = canonical_json_sha256(report)
    return report


def _format_optional(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if not math.isfinite(numeric):
        return "—"
    return f"{numeric:.{digits}f}"


def render_autonomous_material_trajectory_replication_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render the replication audit as compact Chinese Markdown."""

    matrix = report["matrix"]
    lines = [
        "# G2 seed 1 / seed 3 fresh trajectory replication 审计",
        "",
        f"状态：`{report['status']}`。完成 {matrix['completed_cell_count']}/20 "
        f"cells，右删失 {matrix['right_censored_cell_count']}。",
        "",
        "| world | replicate | opaque | nominal | Δbest | Δretention | "
        "Δdrawdown | Δterminal/best |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in report["paired_trajectories"]:
        delta = row["nominal_minus_opaque"]
        best = None if delta is None else delta["best_final_score"]
        retention = (
            None if delta is None else delta["online_incumbent_retention_rate"]
        )
        drawdown = (
            None
            if delta is None
            else delta["maximum_absolute_incumbent_drawdown"]
        )
        terminal_ratio = (
            None if delta is None else delta["terminal_to_global_best_ratio"]
        )
        lines.append(
            f"| {row['world_seed']} | {row['trajectory_replicate_id']} | "
            f"{row['opaque_state']} | {row['nominal_state']} | "
            f"{_format_optional(best)} | {_format_optional(retention)} | "
            f"{_format_optional(drawdown)} | {_format_optional(terminal_ratio)} |"
        )
    lines.extend(["", "## 分 world 描述", ""])
    for world_seed in EXPECTED_WORLD_SEEDS:
        world = report["within_world_descriptive_aggregates"][str(world_seed)]
        lines.extend(
            [
                f"### seed {world_seed}",
                "",
                f"完成 pair：{world['completed_pair_count']}/5；右删失 "
                f"{world['right_censored_pair_count']}。",
                "",
                "| metric | median | min | max | +/-/0 | sign consistency |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for metric in _PAIRED_METRICS:
            summary = world["paired_metrics"][metric]
            lines.append(
                f"| {metric} | {_format_optional(summary['median'])} | "
                f"{_format_optional(summary['minimum'])} | "
                f"{_format_optional(summary['maximum'])} | "
                f"{summary['positive_count']}/{summary['negative_count']}/"
                f"{summary['zero_count']} | "
                f"{_format_optional(summary['sign_consistency'])} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Outcome-blind 叙事映射",
            "",
            f"- branch：`{report['interpretation']['selected_branch']['branch_id']}`",
            "- "
            + report["interpretation"]["selected_branch"]["manuscript_language"],
            "- policy SHA-256：`"
            + report["interpretation"]["mapping_policy"]["sha256"]
            + "`",
            "",
            "## 边界",
            "",
            f"- {report['interpretation']['caveat']}",
            "- attempt 只允许在零 accepted action 的 provider infrastructure "
            "failure 后替换；所有动作后失败均保留为右删失。",
            "- 两个 arm 的 Codex provider sampling randomness 不能用本地 "
            "agent_seed 配对或复现；pairing 只对物理世界、观测流、资源和本地"
            "方法 seed 成立。",
            f"- 审计哈希：`{report['audit_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_autonomous_material_trajectory_replication_audit(
    manifest_path: str | Path,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
    expected_vessels_per_cell: int = 6,
    interpretation_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    report = audit_autonomous_material_trajectory_replication(
        manifest_path,
        expected_vessels_per_cell=expected_vessels_per_cell,
        interpretation_policy_path=interpretation_policy_path,
    )
    write_json_atomic(Path(json_path), report)
    output = Path(markdown_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_autonomous_material_trajectory_replication_markdown(report),
        encoding="utf-8",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--expected-vessels", type=int, default=6)
    parser.add_argument(
        "--interpretation-policy",
        type=Path,
        default=DEFAULT_INTERPRETATION_POLICY_PATH,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = write_autonomous_material_trajectory_replication_audit(
        args.manifest,
        json_path=args.json_output,
        markdown_path=args.markdown_output,
        expected_vessels_per_cell=args.expected_vessels,
        interpretation_policy_path=args.interpretation_policy,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "completed_pair_count": report["matrix"][
                    "completed_pair_count"
                ],
                "audit_sha256": report["audit_sha256"],
                "interpretation_branch": report["interpretation"][
                    "selected_branch"
                ]["branch_id"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTONOMOUS_MATERIAL_REPLICATION_AUDIT_VERSION",
    "DEFAULT_INTERPRETATION_POLICY_PATH",
    "INTERPRETATION_POLICY_VERSION",
    "AutonomousMaterialReplicationAuditError",
    "audit_autonomous_material_trajectory_replication",
    "render_autonomous_material_trajectory_replication_markdown",
    "validate_interpretation_binding",
    "write_autonomous_material_trajectory_replication_audit",
]
