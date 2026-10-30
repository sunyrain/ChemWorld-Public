"""Dataset export utilities for ChemWorld trajectories."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from chemworld.data.logging import load_jsonl, to_builtin
from chemworld.data.records import trajectory_value
from chemworld.data.validation import validate_records


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
    observation = trajectory_value(record, 'observation', {})
    flags = record.get("constraint_flags", {})
    trace = record.get("agent_trace", [])
    latest_trace = _latest_agent_trace_entry(record)
    return {
        "task_id": record.get("task_id"),
        "run_id": record.get("run_id"),
        "campaign_id": record.get("campaign_id"),
        "experiment_index": record.get("experiment_index"),
        "operation_id": record.get("operation_id"),
        "benchmark_task_id": trajectory_value(record, 'benchmark_task_id', None),
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
        "reward": trajectory_value(record, 'reward', None),
        "leaderboard_score": trajectory_value(record, 'leaderboard_score', None),
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
        "agent_view": json.dumps(
            to_builtin(trajectory_value(record, "agent_view", {})), sort_keys=True,
        ),
        "lab_report": json.dumps(
            to_builtin(trajectory_value(record, 'agent_view', {}).get("lab_report", {})),
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


__all__ = ["DatasetExportResult", "export_dataset", "load_dataset_records"]
