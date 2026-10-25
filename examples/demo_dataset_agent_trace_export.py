"""Demo: export trajectories with agent-facing views and agent trace fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chemworld.data.datasets import export_dataset, flatten_record
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.runner import run_agent
from chemworld.legacy.agents.llm import ToolUsingLLMStubAgent


def build_demo(root: str | Path = Path("runs") / "demo_agent_trace_dataset") -> dict[str, Any]:
    """Run a deterministic tool-agent and export JSONL plus optional Parquet data."""

    output_root = Path(root)
    trajectory_path = output_root / "trajectories" / "tool_stub_reaction_to_assay_seed0.jsonl"
    dataset_jsonl = output_root / "dataset" / "agent_trace_dataset.jsonl"
    dataset_parquet = output_root / "dataset" / "agent_trace_dataset.parquet"

    run_agent(
        env_id="ChemWorld",
        agent=ToolUsingLLMStubAgent(),
        world_split="public-dev",
        budget=18,
        objective="balanced",
        seed=0,
        task_id="reaction-to-assay",
        output_path=trajectory_path,
    )
    jsonl_result = export_dataset(trajectory_path, output=dataset_jsonl, format="jsonl")

    parquet_status: dict[str, Any]
    try:
        parquet_result = export_dataset(trajectory_path, output=dataset_parquet, format="parquet")
        parquet_status = parquet_result.to_dict()
    except RuntimeError as exc:
        parquet_status = {
            "output_path": str(dataset_parquet),
            "format": "parquet",
            "status": "skipped",
            "reason": str(exc),
        }

    records = load_jsonl(trajectory_path)
    final_record = records[-1]
    flattened = flatten_record(final_record)
    return {
        "trajectory_path": str(trajectory_path),
        "jsonl_export": jsonl_result.to_dict(),
        "parquet_export": parquet_status,
        "record_count": len(records),
        "final_reward": trajectory_value(final_record, 'reward'),
        "agent_trace_step_count": flattened["agent_trace_step_count"],
        "agent_trace_selected_action": flattened["agent_trace_selected_action"],
        "agent_trace_validation_result": flattened["agent_trace_validation_result"],
        "agent_trace_observation_summary": flattened["agent_trace_observation_summary"],
        "agent_trace_memory_note": flattened["agent_trace_memory_note"],
        "lab_report_text": trajectory_value(final_record, 'agent_view')["lab_report"]["text"],
    }


def main() -> None:
    summary = build_demo()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
