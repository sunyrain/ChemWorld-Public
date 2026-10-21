"""Offline many-particle campaign through native equipment, logging and replay."""

import argparse
import json
from pathlib import Path

from chemworld.agents.base import BaseAgent
from chemworld.data.logging import load_jsonl
from chemworld.data.scientific_inputs import InputSource
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def default_spec():
    source = InputSource(
        DataSourceProvenance(
            "synthetic-native-mpm", 0, "Self-authored synthetic two-radius fixture", "1", "CC0-1.0"
        ),
        "synthetic",
    )
    distribution = ParticleSizeDistribution((5e-6, 15e-6), (0.5, 0.5), source)
    return BatteryMPMSpec(
        negative=ParticleSpec(cells=16),
        positive=ParticleSpec(cells=16),
        negative_distribution=distribution,
        positive_distribution=distribution,
    )


def workflow(spec=None):
    spec = spec or default_spec()
    if not isinstance(spec, BatteryMPMSpec):
        raise ValueError("MPM example requires the actual many-particle model")
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

    run_agent(
        env_id="ChemWorld",
        agent=FixedSequence(),
        composition=world.to_dict(),
        world_split="public-dev",
        budget=80,
        objective="balanced",
        seed=4,
        observation_noise_mode=noise,
        output_path=output,
    )
    records = load_jsonl(output)
    if len(records) != len(actions) or any(r["transaction_status"] != "committed" for r in records):
        raise RuntimeError("incomplete MPM workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {
        "model": "mpm-isothermal-v1",
        "steps": len(records),
        "replay_steps": report.checked_steps,
        "noise": noise,
        "trajectory": str(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
