"""Separate electrode OCV and finite Li stocks through the original runtime.

Explicit synthetic capacities, curves and series resistance support one
discharge/rest/partial-recharge sequence with actual action logs and exact replay.
No particle diffusion or thermal chemistry is implied by this reduced model.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.battery_split_ocvr import BatterySplitOCVRSpec
from chemworld.world.battery_world import battery_world_spec
from chemworld.world.composition import WorldCompositionSpec


def default_spec() -> BatterySplitOCVRSpec:
    return BatterySplitOCVRSpec(provenance="synthetic-split-ocvr-example-v1")


def workflow(
    spec: BatterySplitOCVRSpec | None = None,
) -> tuple[WorldCompositionSpec, tuple[dict[str, Any], ...]]:
    spec = default_spec() if spec is None else spec
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": .5},
        {"operation": "wait", "duration_s": 4.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 2.},
        {"operation": "configure_electrical_control", "current_A": -.25},
        {"operation": "wait", "duration_s": 4.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 1.},
        {"operation": "terminate"},
    )


def run(
    output: str | Path, *, spec: BatterySplitOCVRSpec | None = None,
    noise: str = "keyed",
) -> dict[str, str | int | float]:
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve the earlier trajectory; choose a new output path")
    spec = default_spec() if spec is None else spec
    world, actions = workflow(spec)

    class FixedSequence(BaseAgent):
        def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
            return actions[len(history)]

    run_agent(env_id="ChemWorld", agent=FixedSequence(), composition=world.to_dict(),
              world_split="public-dev", budget=80, objective="balanced", seed=4,
              observation_noise_mode=noise, output_path=output)
    records = load_jsonl(output)
    if len(records) != len(actions) or any(
            record["transaction_status"] != "committed" for record in records):
        raise RuntimeError("incomplete Split OCV-R workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output),
            "scope": "synthetic Split OCV-R; two finite Li inventories"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
