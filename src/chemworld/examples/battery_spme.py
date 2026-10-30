"""Offline synthetic SPMe workflow, with public readings and verified JSONL replay.

Run: uv run --no-sync python -m chemworld.examples.battery_spme --output /tmp/spme.jsonl
No provider or PyBaMM is used. This demonstrates native execution, not measured-cell accuracy.
"""

import argparse
import json
from pathlib import Path

from chemworld.agents.base import BaseAgent
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import BatterySPMSpec
from chemworld.world.battery_world import battery_world_spec


def workflow(spec=None):
    spec = spec or BatterySPMSpec(electrolyte=BatteryElectrolyteSpec())
    if spec.electrolyte is None:
        raise ValueError("this example requires the explicit SPMe option")
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": 0.2},
        {"operation": "wait", "duration_s": 5},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0},
        {"operation": "wait", "duration_s": 5},
        {"operation": "configure_electrical_control", "potential_V": 3.75},
        {"operation": "wait", "duration_s": 2},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "terminate"},
    )


def run(output, *, noise="keyed", spec=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve the earlier trajectory; choose a new output path")
    world, actions = workflow(spec)

    class FixedSequence(BaseAgent):
        def act(self, history):
            return actions[len(history)]

    run_agent(env_id="ChemWorld", agent=FixedSequence(), composition=world.to_dict(),
              world_split="public-dev", budget=80, objective="balanced", seed=4,
              observation_noise_mode=noise, output_path=output)
    records = load_jsonl(output)
    if len(records) != len(actions):
        raise RuntimeError(f"incomplete workflow: {len(records)}/{len(actions)} steps")
    if any(record["transaction_status"] != "committed" for record in records):
        raise RuntimeError("workflow contains an uncommitted action; retained in JSONL")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": (spec.model_id if spec is not None else "spme-isothermal-v1"),
            "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print("Executing native SPMe workflow and replay (9 + 9 steps)", flush=True)
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
