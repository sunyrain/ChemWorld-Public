"""Small custom operation agent; a lifecycle example, not a capable optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.tasks import get_task


class FourStepAgent(BaseAgent):
    """Replace act() with your policy; BaseAgent supplies zero-provider metadata."""

    name = "four-step-example"

    def act(self, history: list[HistoryRecord]) -> dict:
        actions: list[dict[str, Any]] = [
            {"operation": "add_solvent", "volume_L": 0.025, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.012},
            {"operation": "terminate"},
            {"operation": "measure", "instrument": "final_assay"},
        ]
        return actions[len(history)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing results are not overwritten")
    task = get_task("reaction-to-assay")
    history = run_agent(
        env_id=task.env_id,
        task_id=task.task_id,
        agent=FourStepAgent(),
        world_split=task.world_split,
        budget=task.budget,
        objective=task.objective,
        seed=0,
        output_path=args.output,
    )
    records = load_jsonl(args.output)
    result = verify_records(records, tolerance=0.0).to_dict()
    complete = (
        len(history) == 4
        and records[-1]["instrument"] == "final_assay"
        and records[-1]["transaction_status"] == "committed"
    )
    print(json.dumps({"formal_result": False, "final_assay_completed": complete, "replay": result}))
    return 0 if complete and result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
