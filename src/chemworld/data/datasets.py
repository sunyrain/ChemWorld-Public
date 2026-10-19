"""Dataset export utilities for ChemWorld trajectories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from chemworld.data.logging import load_jsonl, to_builtin
from chemworld.data.submission import git_commit
from chemworld.data.validation import validate_records

DATASET_CARD_SCHEMA_VERSION = "chemworld-dataset-card-0.4"


@dataclass(frozen=True)
class DatasetExportResult:
    output_path: str
    record_count: int
    format: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "record_count": self.record_count,
            "format": self.format,
        }


def _trajectory_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    trajectory_dir = root / "trajectories"
    if trajectory_dir.exists():
        return sorted(trajectory_dir.glob("*.jsonl"))
    return sorted(root.glob("*.jsonl"))


def load_dataset_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for trajectory_path in _trajectory_paths(path):
        trajectory_records = load_jsonl(trajectory_path)
        validate_records(trajectory_records)
        records.extend(trajectory_records)
    if not records:
        raise ValueError(f"No trajectory JSONL files found under {path}")
    return records


def _latest_agent_trace_entry(record: dict[str, Any]) -> dict[str, Any]:
    trace = record.get("agent_trace", [])
    if not isinstance(trace, list) or not trace:
        return {}
    latest = trace[-1]
    return latest if isinstance(latest, dict) else {}


def _agent_trace_memory_note(entry: dict[str, Any]) -> str | None:
    memory_note = entry.get("memory_note")
    if memory_note not in {None, ""}:
        return str(memory_note)
    prompt_input = entry.get("prompt_input", {})
    if not isinstance(prompt_input, dict):
        return None
    memory_summary = prompt_input.get("memory_summary", [])
    if isinstance(memory_summary, list) and memory_summary:
        return str(memory_summary[-1])
    return None


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    observation = record.get("observation", {})
    flags = record.get("constraint_flags", {})
    trace = record.get("agent_trace", [])
    latest_trace = _latest_agent_trace_entry(record)
    return {
        "task_id": record.get("task_id"),
        "run_id": record.get("run_id"),
        "campaign_id": record.get("campaign_id"),
        "experiment_index": record.get("experiment_index"),
        "operation_id": record.get("operation_id"),
        "benchmark_task_id": record.get("benchmark_task_id"),
        "scenario_id": record.get("scenario_id"),
        "world_split": record.get("world_split"),
        "world_family_version": record.get("world_family_version"),
        "env_version": record.get("env_version"),
        "seed": record.get("seed"),
        "step": record.get("step"),
        "operation_type": record.get("operation_type"),
        "instrument": record.get("instrument"),
        "task_contract_hash": record.get("task_contract_hash"),
        "runtime_profile_hash": record.get("runtime_profile_hash"),
        "runtime_semantics_id": record.get("runtime_semantics_id"),
        "transaction_status": record.get("transaction_status"),
        "rollback_reason": record.get("rollback_reason"),
        "terminated": record.get("terminated"),
        "truncated": record.get("truncated"),
        "measurement_cost": record.get("measurement_cost"),
        "sample_consumed": record.get("sample_consumed"),
        "observed_mask": json.dumps(to_builtin(record.get("observed_mask", {})), sort_keys=True),
        "sample_outcome": json.dumps(
            to_builtin(record.get("raw_signal", {}).get("sample_outcome")), sort_keys=True
        ),
        "mechanism_id": record.get("mechanism_id"),
        "mechanism_hash": record.get("mechanism_hash"),
        "scoring_contract_hash": record.get("scoring_contract_hash"),
        "observation_contract_hash": record.get("observation_contract_hash"),
        "reward": record.get("reward"),
        "leaderboard_score": record.get("leaderboard_score"),
        "yield": observation.get("yield"),
        "selectivity": observation.get("selectivity"),
        "conversion": observation.get("conversion"),
        "score": observation.get("score"),
        "purity": observation.get("purity"),
        "recovery": observation.get("recovery"),
        "safety_risk": observation.get("safety_risk"),
        "cost": observation.get("cost"),
        "unsafe": flags.get("unsafe"),
        "precondition_failed": flags.get("precondition_failed"),
        "constitution_failed": flags.get("constitution_failed"),
        "action": json.dumps(to_builtin(record.get("action", {})), sort_keys=True),
        "environment_outcome": json.dumps(
            to_builtin(record.get("environment_outcome", {})),
            sort_keys=True,
        ),
        "agent_visible_observation": json.dumps(
            to_builtin(record.get("agent_visible_observation", {})),
            sort_keys=True,
        ),
        "evaluation_outcome": json.dumps(
            to_builtin(record.get("evaluation_outcome", {})),
            sort_keys=True,
        ),
        "agent_view": json.dumps(to_builtin(record.get("agent_view", {})), sort_keys=True),
        "lab_report": json.dumps(
            to_builtin(record.get("agent_view", {}).get("lab_report", {})),
            sort_keys=True,
        ),
        "agent_trace": json.dumps(to_builtin(record.get("agent_trace", [])), sort_keys=True),
        "agent_trace_step_count": len(trace) if isinstance(trace, list) else 0,
        "agent_trace_prompt_summary": json.dumps(
            to_builtin(latest_trace.get("prompt_input", {})),
            sort_keys=True,
        ),
        "agent_trace_selected_action": json.dumps(
            to_builtin(latest_trace.get("selected_action", {})),
            sort_keys=True,
        ),
        "agent_trace_validation_result": json.dumps(
            to_builtin(latest_trace.get("validator_result", {})),
            sort_keys=True,
        ),
        "agent_trace_observation_summary": json.dumps(
            to_builtin(latest_trace.get("observation_summary", {})),
            sort_keys=True,
        ),
        "agent_trace_reasoning_summary": latest_trace.get("reasoning_summary"),
        "agent_trace_hypothesis_note": latest_trace.get("hypothesis_note"),
        "agent_trace_memory_note": _agent_trace_memory_note(latest_trace),
    }


def export_dataset(
    path: str | Path,
    *,
    output: str | Path,
    format: str,
) -> DatasetExportResult:
    records = load_dataset_records(path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "jsonl":
        if len(_trajectory_paths(path)) == 1 and Path(path).is_file():
            shutil.copyfile(Path(path), output_path)
        else:
            with output_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(to_builtin(record), sort_keys=True) + "\n")
    elif format == "parquet":
        frame = pd.DataFrame(flatten_record(record) for record in records)
        try:
            frame.to_parquet(output_path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet export requires pyarrow or fastparquet. "
                "Install one of them or export with --format jsonl."
            ) from exc
    else:
        raise ValueError("format must be 'jsonl' or 'parquet'")
    return DatasetExportResult(str(output_path), len(records), format)


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
        {str(record.get("benchmark_task_id") or record.get("task_id")) for record in records}
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


__all__ = ["DatasetExportResult", "dataset_card", "export_dataset", "load_dataset_records"]
