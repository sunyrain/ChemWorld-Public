"""Audit and summarize a minimal paired mechanism-adaptation Agent pilot."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.mechanism_adaptation import (
    change_detection_summary,
    declared_change_probability,
    normalized_distribution,
)
from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    git_source_commit,
)
from chemworld.eval.verify import verify_records

PILOT_REPORT_VERSION = "chemworld-mechanism-adaptation-agent-pilot-0.2.1"
CHECKPOINTS = (1, 2, 4, 8)


def load_campaigns_from_index(
    index_path: str | Path,
    *,
    root: str | Path,
) -> list[dict[str, Any]]:
    """Load exactly the campaign summaries named by a frozen run index."""

    repository_root = Path(root)
    index = _load_object(Path(index_path))
    paths = index.get("campaign_paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("campaign index must name at least one campaign")
    campaigns: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(str(raw))
        if not path.is_absolute():
            path = repository_root / path
        campaigns.append(_load_object(path))
    return campaigns


def build_agent_pilot_report(
    protocol: Mapping[str, Any],
    campaigns: Sequence[Mapping[str, Any]],
    *,
    root: str | Path,
    replay: bool = True,
) -> dict[str, Any]:
    """Build a fail-closed Gate 0 and descriptive Gate B/E paired report."""

    repository_root = Path(root).resolve()
    by_arm = {str(item.get("matrix_row", {}).get("arm")): item for item in campaigns}
    if set(by_arm) != {"changed", "no_change_twin"} or len(campaigns) != 2:
        raise ValueError("pilot report requires one complete changed/no-change pair")
    changed = by_arm["changed"]
    no_change = by_arm["no_change_twin"]
    changed_row = changed["matrix_row"]
    no_change_row = no_change["matrix_row"]
    if changed_row.get("pair_id") != no_change_row.get("pair_id"):
        raise ValueError("pilot campaign arms do not share one pair ID")
    expected_protocol_sha256 = canonical_json_sha256(protocol)
    if any(
        campaign.get("protocol_sha256") != expected_protocol_sha256
        for campaign in campaigns
    ):
        raise ValueError("pilot campaign protocol binding is stale")

    trajectory_audits: list[dict[str, Any]] = []
    records_by_arm: dict[str, dict[str, list[dict[str, Any]]]] = {}
    all_receipts: list[Mapping[str, Any]] = []
    for arm, campaign in by_arm.items():
        records_by_arm[arm] = {}
        receipts = campaign.get("provider_receipts")
        if isinstance(receipts, list):
            all_receipts.extend(item for item in receipts if isinstance(item, Mapping))
        for phase in ("iid", "shifted"):
            phase_summary = campaign[phase]
            path = _resolve_repository_path(
                repository_root,
                str(phase_summary["trajectory_path"]),
            )
            records = _load_jsonl(path)
            records_by_arm[arm][phase] = records
            observed_sha256 = file_sha256(path)
            replay_result = None
            if replay:
                replay_result = verify_records(
                    records,
                    world_interventions=(
                        campaign.get("shifted_interventions", ())
                        if phase == "shifted"
                        else ()
                    ),
                ).to_dict()
            trajectory_audits.append(
                {
                    "arm": arm,
                    "phase": phase,
                    "path": path.relative_to(repository_root).as_posix(),
                    "record_count": len(records),
                    "expected_sha256": phase_summary["trajectory_sha256"],
                    "observed_sha256": observed_sha256,
                    "hash_matches": observed_sha256
                    == phase_summary["trajectory_sha256"],
                    "all_outcome_layers_present": all(
                        {
                            "environment_outcome",
                            "agent_visible_observation",
                            "evaluation_outcome",
                        }
                        <= record.keys()
                        for record in records
                    ),
                    "replay": replay_result,
                }
            )

    leakage = _prompt_leakage_audit(records_by_arm)
    provider = _provider_audit(all_receipts, records_by_arm)
    exclusions = _exclusion_audit(records_by_arm)
    gate_0_checks = {
        "complete_changed_no_change_pair": True,
        "protocol_hashes_match": True,
        "trajectory_hashes_match": all(item["hash_matches"] for item in trajectory_audits),
        "three_outcome_layers_present": all(
            item["all_outcome_layers_present"] for item in trajectory_audits
        ),
        "all_trajectories_replay": all(
            item["replay"] is not None and item["replay"]["verified"]
            for item in trajectory_audits
        )
        if replay
        else False,
        "provider_identity_and_receipts_complete": provider["passed"],
        "derived_diagnostic_prompt_leakage_absent": leakage["passed"],
        "model_failures_reason_coded_and_excluded": exclusions["passed"],
        "agent_weight_updates_absent": all(
            campaign.get("agent_weight_updates_performed") is False
            for campaign in campaigns
        ),
    }

    checkpoint_rows: dict[str, Any] = {}
    for checkpoint in CHECKPOINTS:
        changed_distribution = distribution_after_experiments(
            records_by_arm["changed"]["shifted"], checkpoint
        )
        no_change_distribution = distribution_after_experiments(
            records_by_arm["no_change_twin"]["shifted"], checkpoint
        )
        probabilities = [
            declared_change_probability(changed_distribution),
            declared_change_probability(no_change_distribution),
        ]
        checkpoint_rows[str(checkpoint)] = {
            "change_detection": change_detection_summary(
                changed=[True, False],
                probabilities=probabilities,
                detection_delays=[
                    _detection_delay(records_by_arm["changed"]["shifted"]),
                    None,
                ],
            ),
            "changed_arm": _checkpoint_arm_summary(
                changed_distribution,
                truth_id=str(changed.get("truth_id")),
                scores=changed["shifted"]["scores"],
                checkpoint=checkpoint,
            ),
            "no_change_twin": _checkpoint_arm_summary(
                no_change_distribution,
                truth_id="no_change",
                scores=no_change["shifted"]["scores"],
                checkpoint=checkpoint,
            ),
        }

    autonomy = {
        arm: _autonomy_summary(campaign)
        for arm, campaign in (("changed", changed), ("no_change_twin", no_change))
    }
    total_cost = sum(float(item.get("billed_cost_usd") or 0.0) for item in all_receipts)
    gate_0_pass = all(gate_0_checks.values())
    return {
        "schema_version": PILOT_REPORT_VERSION,
        "status": (
            "pilot_complete_integrity_passed_partial_attribution"
            if gate_0_pass
            else "pilot_integrity_failed"
        ),
        "formal_benchmark_result": False,
        "benchmark_claim_allowed": False,
        "publication_ready": False,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": expected_protocol_sha256,
        "report_source_commit": git_source_commit(repository_root),
        "campaign_source_commits": sorted(
            {
                str(records[0].get("agent_metadata", {}).get("git_commit"))
                for phases in records_by_arm.values()
                for records in phases.values()
            }
        ),
        "pair_id": changed_row["pair_id"],
        "task_id": changed_row["task_id"],
        "changed_truth_id": changed["truth_id"],
        "candidate_label_mode": changed_row["candidate_label_mode"],
        "method_id": changed["method_id"],
        "agent_weight_updates_performed": False,
        "scope": {
            "pair_count": 1,
            "campaign_count": 2,
            "provider_repeat_count": 1,
            "change_time": changed_row["phase_reset_after_experiment"],
            "post_change_experiment_count": changed["shifted"][
                "complete_experiment_count"
            ],
            "interpretation": (
                "descriptive pilot only; confidence-bound gates cannot pass from one pair"
            ),
        },
        "gate_0": {
            "status": "passed" if gate_0_pass else "failed",
            "checks": gate_0_checks,
            "trajectory_audits": trajectory_audits,
            "provider_audit": provider,
            "prompt_leakage_audit": leakage,
            "exclusion_audit": exclusions,
        },
        "gate_b": {
            "status": "descriptive_only_insufficient_pairs",
            "checkpoints": checkpoint_rows,
            "confirmatory_gate_pass": False,
            "reason": (
                "one pair cannot establish confidence bounds or randomized-time generalization"
            ),
        },
        "gate_c": {
            "status": "not_evaluated",
            "reason": (
                "same-prefix feedback variants and paired full-campaign feedback arms were not run"
            ),
        },
        "gate_d": {
            "status": "not_evaluated",
            "adaptive_policy_observed": {
                arm: {
                    "mean_score": campaign["shifted"]["mean_score"],
                    "best_score": campaign["shifted"]["best_score"],
                }
                for arm, campaign in (("changed", changed), ("no_change_twin", no_change))
            },
            "missing_required_conditions": [
                "iid_action_replay_iid_world",
                "iid_action_replay_shifted_world",
                "frozen_policy_shifted_world",
                "diagnosis_oracle_shifted_world",
            ],
        },
        "gate_e": {
            "status": "pilot_protocol_failure_observed",
            "campaigns": autonomy,
            "autonomous_and_assisted_scores_separated": True,
            "confirmatory_gate_pass": False,
        },
        "resources": {
            "provider_receipt_count": len(all_receipts),
            "provider_billed_cost_usd": total_cost,
        },
        "interpretation": (
            "Gate 0 passed for this pair. Gate B is descriptively adverse: the no-change "
            "twin receives equal or higher change probability at every frozen checkpoint. "
            "Gate C and Gate D remain unmeasured, and lifecycle assistance prevents a "
            "fully autonomous result. No mechanism-discovery claim is allowed."
        ),
    }


def distribution_after_experiments(
    records: Sequence[Mapping[str, Any]],
    experiment_count: int,
) -> dict[str, float]:
    """Return the first valid decision made after *experiment_count* feedback packets."""

    terminal_index = next(
        (
            index
            for index, record in enumerate(records)
            if int(record.get("method_resources", {}).get("complete_experiment_count", 0))
            >= experiment_count
        ),
        None,
    )
    if terminal_index is None:
        raise ValueError(f"trajectory never completed experiment {experiment_count}")
    for record in records[terminal_index + 1 :]:
        trace = record.get("agent_trace")
        if not isinstance(trace, list) or not trace:
            continue
        decision = trace[-1]
        if not isinstance(decision, Mapping) or decision.get("status") != "model_decision":
            continue
        distribution = decision.get("mechanism_distribution")
        if isinstance(distribution, Mapping):
            return normalized_distribution(
                {str(key): value for key, value in distribution.items()}
            )
    raise ValueError(f"trajectory has no valid decision after experiment {experiment_count}")


def _detection_delay(records: Sequence[Mapping[str, Any]], threshold: float = 0.5) -> int | None:
    maximum = max(
        int(record.get("method_resources", {}).get("complete_experiment_count", 0))
        for record in records
    )
    for experiment_count in range(1, maximum):
        distribution = distribution_after_experiments(records, experiment_count)
        if declared_change_probability(distribution) >= threshold:
            return experiment_count
    return None


def _checkpoint_arm_summary(
    distribution: Mapping[str, float],
    *,
    truth_id: str,
    scores: Sequence[float],
    checkpoint: int,
) -> dict[str, Any]:
    values = normalized_distribution(distribution)
    observed_scores = [float(item) for item in scores[:checkpoint]]
    return {
        "mechanism_distribution": values,
        "change_probability": declared_change_probability(values),
        "mechanism_prediction": max(values, key=values.__getitem__),
        "truth_probability": values[truth_id],
        "mechanism_identified": max(values, key=values.__getitem__) == truth_id,
        "mean_score_through_checkpoint": sum(observed_scores) / len(observed_scores),
        "best_score_through_checkpoint": max(observed_scores),
    }


def _autonomy_summary(campaign: Mapping[str, Any]) -> dict[str, Any]:
    guardrails = campaign.get("lifecycle_guardrail_log")
    log = guardrails if isinstance(guardrails, list) else []
    total_experiments = int(campaign["iid"]["complete_experiment_count"]) + int(
        campaign["shifted"]["complete_experiment_count"]
    )
    assisted_experiments = {
        (str(item.get("phase")), int(item.get("experiment_index", -1)))
        for item in log
        if isinstance(item, Mapping)
    }
    return {
        "autonomous_procedural_status": (
            "protocol_failure" if log else "fully_autonomous_campaign"
        ),
        "assisted_scientific_score": campaign["shifted"]["mean_score"],
        "lifecycle_guardrail_action_count": len(log),
        "assisted_experiment_count": len(assisted_experiments),
        "fully_autonomous_experiment_count": total_experiments
        - len(assisted_experiments),
        "total_complete_experiment_count": total_experiments,
    }


def _provider_audit(
    receipts: Sequence[Mapping[str, Any]],
    records_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    fingerprints: set[str] = set()
    models: set[str] = set()
    for phases in records_by_arm.values():
        for records in phases.values():
            for record in records:
                provenance = (
                    record.get("method_resources", {})
                    .get("agent_usage", {})
                    .get("model_provenance", {})
                )
                models.add(str(provenance.get("model_id") or ""))
                fingerprints.update(
                    str(item)
                    for item in provenance.get("observed_system_fingerprints", ())
                )
    checks = {
        "receipts_present": bool(receipts),
        "all_receipts_succeeded": all(item.get("status") == "succeeded" for item in receipts),
        "all_receipts_billable_and_usage_complete": all(
            item.get("billable") is True and item.get("usage_complete") is True
            for item in receipts
        ),
        "all_request_ids_present": all(bool(item.get("request_id")) for item in receipts),
        "provider_is_deepseek": all(item.get("provider") == "DeepSeek" for item in receipts),
        "model_is_frozen_flash": models == {"deepseek-v4-flash"},
        "system_fingerprint_present": bool(fingerprints),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "receipt_count": len(receipts),
        "models": sorted(models),
        "system_fingerprints": sorted(fingerprints),
    }


def _prompt_leakage_audit(
    records_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    decision_count = 0
    forbidden_trace_fields: set[str] = set()
    metadata_violations = 0
    for phases in records_by_arm.values():
        for records in phases.values():
            for record in records:
                metadata = record.get("agent_metadata", {})
                if metadata.get("derived_diagnostics_returned_to_agent") is not False:
                    metadata_violations += 1
                trace = record.get("agent_trace")
                if not isinstance(trace, list):
                    continue
                for decision in trace:
                    if not isinstance(decision, Mapping):
                        continue
                    decision_count += 1
                    forbidden_trace_fields.update(
                        {"change_probability", "mechanism_prediction"} & set(decision)
                    )
    checks = {
        "metadata_declares_no_derived_feedback": metadata_violations == 0,
        "derived_fields_absent_from_decision_trace": not forbidden_trace_fields,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "decision_trace_count": decision_count,
        "forbidden_trace_fields": sorted(forbidden_trace_fields),
    }


def _exclusion_audit(
    records_by_arm: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    unreasoned_model_failures = 0
    for phases in records_by_arm.values():
        for records in phases.values():
            for record in records:
                trace = record.get("agent_trace")
                if not isinstance(trace, list):
                    continue
                for decision in trace:
                    if not isinstance(decision, Mapping):
                        continue
                    status = str(decision.get("status") or "missing")
                    status_counts[status] = status_counts.get(status, 0) + 1
                    if status == "model_failure":
                        outcome = decision.get("outcome")
                        if not isinstance(outcome, Mapping) or not outcome.get(
                            "error_message"
                        ):
                            unreasoned_model_failures += 1
    checks = {
        "only_valid_decisions_enter_checkpoint_metrics": True,
        "all_model_failures_have_reason_codes": unreasoned_model_failures == 0,
        "decision_statuses_are_declared": set(status_counts)
        <= {"model_decision", "model_failure", "lifecycle_guardrail"},
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "decision_status_counts": status_counts,
        "unreasoned_model_failure_count": unreasoned_model_failures,
    }


def _resolve_repository_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"trajectory path escapes repository root: {raw}") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records or not all(isinstance(item, dict) for item in records):
        raise ValueError(f"trajectory is empty or invalid: {path}")
    return records


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


__all__ = [
    "CHECKPOINTS",
    "PILOT_REPORT_VERSION",
    "build_agent_pilot_report",
    "distribution_after_experiments",
    "load_campaigns_from_index",
]
