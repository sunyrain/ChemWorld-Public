"""Dataset descriptions with independent replay verification."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chemworld.data.datasets import _trajectory_paths, load_dataset_records
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.data.submission import git_commit
from chemworld.data.validation import validate_records

DATASET_CARD_SCHEMA_VERSION = "chemworld-dataset-card-0.4"


def _unique_nonempty(records: list[dict[str, Any]], key: str) -> list[str]:
    return sorted(
        {str(record[key]) for record in records if key in record and record[key] not in {None, ""}}
    )


def _protocol_hashes(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {
        "task_contract_hashes": _unique_nonempty(records, "task_contract_hash"),
        "runtime_profile_hashes": _unique_nonempty(records, "runtime_profile_hash"),
        "mechanism_hashes": _unique_nonempty(records, "mechanism_hash"),
        "scoring_contract_hashes": _unique_nonempty(records, "scoring_contract_hash"),
        "observation_contract_hashes": _unique_nonempty(
            records,
            "observation_contract_hash",
        ),
    }


def _privacy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    sensitive_keys = {"email", "student_id", "name", "participant_name"}
    participant_fields = {"participant_id", "study_id"}
    contains_sensitive_top_level = any(
        bool(sensitive_keys.intersection(record)) for record in records
    )
    contains_participant_id = any(
        bool(participant_fields.intersection(record)) for record in records
    )
    contains_explanations = any(bool(record.get("explanation")) for record in records)
    contains_agent_metadata = any(bool(record.get("agent_metadata")) for record in records)
    return {
        "status": (
            "review_required"
            if contains_sensitive_top_level or contains_participant_id
            else "synthetic_or_submission_provided"
        ),
        "contains_agent_metadata": contains_agent_metadata,
        "contains_explanations": contains_explanations,
        "contains_participant_fields": contains_participant_id,
        "contains_sensitive_top_level_fields": contains_sensitive_top_level,
        "release_note": (
            "Review explanations and agent metadata before publishing human trajectories."
            if contains_explanations or contains_agent_metadata
            else "No obvious human-study fields detected by the dataset-card scan."
        ),
    }


def _records_by_campaign(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        key = str(record.get("campaign_id") or f"seed-{record.get('seed', 'unknown')}")
        grouped.setdefault(key, []).append(record)
    return sorted(grouped.items(), key=lambda item: item[1][0].get("step", 0))


def _replay_verification_summary(path: str | Path) -> dict[str, Any]:
    from chemworld.eval.verify import verify_records

    groups: list[dict[str, Any]] = []
    for trajectory_path in _trajectory_paths(path):
        trajectory_records = load_jsonl(trajectory_path)
        validate_records(trajectory_records)
        for campaign_id, campaign_records in _records_by_campaign(trajectory_records):
            try:
                result = verify_records(campaign_records)
                mismatch_fields = sorted(
                    {
                        str(mismatch.get("field"))
                        for mismatch in result.mismatches
                        if mismatch.get("field") is not None
                    }
                )
                groups.append(
                    {
                        "trajectory_path": str(trajectory_path),
                        "campaign_id": campaign_id,
                        "verified": result.verified,
                        "checked_steps": result.checked_steps,
                        "max_abs_error": result.max_abs_error,
                        "mismatch_count": len(result.mismatches),
                        "mismatch_fields": mismatch_fields,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive card reporting
                groups.append(
                    {
                        "trajectory_path": str(trajectory_path),
                        "campaign_id": campaign_id,
                        "verified": False,
                        "checked_steps": 0,
                        "max_abs_error": None,
                        "mismatch_count": None,
                        "mismatch_fields": [],
                        "error": str(exc),
                    }
                )
    return {
        "checked": True,
        "group_count": len(groups),
        "verified": all(group["verified"] for group in groups),
        "checked_steps": sum(int(group["checked_steps"]) for group in groups),
        "max_abs_error": max(
            (
                float(group["max_abs_error"])
                for group in groups
                if group["max_abs_error"] is not None
            ),
            default=0.0,
        ),
        "mismatch_count": sum(
            int(group["mismatch_count"]) for group in groups if group["mismatch_count"] is not None
        ),
        "groups": groups,
    }


def _outcome_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count recorded attempts and batch endings; a missing assay is never a zero score."""
    counts = dict.fromkeys(
        (
            "experiment_count",
            "final_assay_experiment_count",
            "negative_final_assay_count",
            "discarded_experiment_count",
            "truncated_open_experiment_count",
            "open_experiment_count",
        ),
        0,
    )
    for _, campaign in _records_by_campaign(records):
        batches: dict[int, list[dict[str, Any]]] = {}
        for record in campaign:
            batches.setdefault(int(record.get("experiment_index", 0)), []).append(record)
        for batch in batches.values():
            counts["experiment_count"] += 1
            committed = [r for r in batch if r.get("transaction_status") == "committed"]
            assays = [r for r in committed if r.get("instrument") == "final_assay"]
            if assays:
                counts["final_assay_experiment_count"] += 1
                if any(
                    r.get("raw_signal", {}).get("sample_outcome", {}).get("status")
                    == "negative_result"
                    for r in assays
                ):
                    counts["negative_final_assay_count"] += 1
            elif any(r.get("operation_type") == "discard_batch" for r in committed):
                counts["discarded_experiment_count"] += 1
            elif any(r.get("truncated") is True for r in batch):
                counts["truncated_open_experiment_count"] += 1
            else:
                counts["open_experiment_count"] += 1
    return {
        **counts,
        "operation_count": len(records),
        "committed_action_count": sum(r.get("transaction_status") == "committed" for r in records),
        "rejected_action_count": sum(
            r.get("transaction_status") in {"rolled_back", "validation_failed"} for r in records
        ),
        "unclassified_action_count": sum(
            r.get("transaction_status") not in {"committed", "rolled_back", "validation_failed"}
            for r in records
        ),
        "open_experiment_semantics": "recording ended without final assay, discard, or truncation",
        "negative_final_assay_is_subset_of_final_assay": True,
    }


def dataset_card(path: str | Path) -> dict[str, Any]:
    records = load_dataset_records(path)
    task_ids = sorted(
        {
            str(trajectory_value(record, "benchmark_task_id", None) or record.get("task_id"))
            for record in records
        }
    )
    seeds = sorted({int(record["seed"]) for record in records})
    world_law_versions = sorted({str(record.get("world_family_version")) for record in records})
    env_versions = sorted({str(record.get("env_version")) for record in records})
    replay_summary = _replay_verification_summary(path)
    privacy = _privacy_summary(records)
    commit_hash = git_commit()
    protocol_hashes = _protocol_hashes(records)
    agent_manifests = sorted(
        {str(record.get("agent_metadata", {}).get("agent_name", "unknown")) for record in records}
    )
    return {
        "schema_version": DATASET_CARD_SCHEMA_VERSION,
        "dataset_id": f"chemworld-dataset-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}",
        "task_ids": task_ids,
        "world_law_version": world_law_versions,
        "env_version": env_versions,
        "trajectory_schema_versions": _unique_nonempty(records, "schema_version"),
        "protocol_hashes": protocol_hashes,
        "replay_verification": replay_summary,
        "commit_hash": commit_hash,
        "seeds": seeds,
        "record_count": len(records),
        "outcome_counts": _outcome_counts(records),
        "runtime_semantics_ids": _unique_nonempty(records, "runtime_semantics_id"),
        "agent_manifests": agent_manifests,
        "provenance": {
            "generator": "ChemWorld",
            "commit_hash": commit_hash,
            "source_trajectory_files": [
                trajectory_path.name for trajectory_path in _trajectory_paths(path)
            ],
            "source_schema_versions": _unique_nonempty(records, "schema_version"),
            "agent_manifests": agent_manifests,
            "protocol_hashes": protocol_hashes,
            "replay_verified": replay_summary["verified"],
        },
        "license": "MIT for generated benchmark traces unless submission declares otherwise",
        "privacy_anonymization_status": privacy["status"],
        "privacy": privacy,
        "known_limitations": [
            "virtual semi-mechanistic world",
            "not calibrated to a specific real reaction",
            "private-eval parameters are not included in public datasets",
        ],
        "created_at": datetime.now(UTC).isoformat(),
    }



__all__ = ["dataset_card"]
