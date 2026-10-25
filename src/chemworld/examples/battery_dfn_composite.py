"""Two negative materials compete locally in a thickness-resolved DFN.

Authored synthetic single-valued OCV and transport inputs. Each material retains
its own particles at each thickness cell; electrolyte stocks remain shared.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chemworld.agents.base import BaseAgent
from chemworld.data.logging import load_jsonl
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.examples.battery_composite import default_spec as particle_spec
from chemworld.physchem.battery_dfn_composite import BatteryCompositeDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.world.battery_world import battery_world_spec


def default_spec(*, radial_cells=8, cells=(4, 3, 4)):
    authored = particle_spec(cells=radial_cells)

    def phase(original, volume):
        return replace(
            original,
            particle=replace(original.particle, active_volume_m3=volume),
            ocv=replace(original.ocv, supported_models=("SPM", "DFN")),
        )

    return BatteryCompositeDFNSpec(
        negative_phases=(
            phase(authored.negative_phases[0], 2.5e-7),
            phase(authored.negative_phases[1], 1e-7),
        ),
        positive_phase=phase(authored.positive_phase, 4e-7),
        electrolyte=BatteryElectrolyteSpec(cells=cells),
        provenance="synthetic-two-negative-one-positive-spatial-composite-DFN-v1",
    )


def workflow(spec=None):
    spec = default_spec() if spec is None else spec
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": 0.12},
        {"operation": "wait", "duration_s": 3.0},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.0},
        {"operation": "wait", "duration_s": 3.0},
        {"operation": "configure_electrical_control", "current_A": -0.1},
        {"operation": "wait", "duration_s": 4.0},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.0},
        {"operation": "wait", "duration_s": 2.0},
        {"operation": "terminate"},
    )


def run(output, *, spec=None, noise="keyed"):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve the earlier trajectory; choose a new output path")
    spec = default_spec() if spec is None else spec
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
        seed=7,
        observation_noise_mode=noise,
        output_path=output,
    )
    records = load_jsonl(output)
    if len(records) != len(actions) or any(
        record["transaction_status"] != "committed" for record in records
    ):
        raise RuntimeError("incomplete composite DFN workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {
        "model": spec.model_id,
        "steps": len(records),
        "replay_steps": report.checked_steps,
        "noise": noise,
        "trajectory": str(output),
        "scope": "synthetic 2+1 phases, resolved thickness/radius and salt, isothermal",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    parser.add_argument("--radial-cells", type=int, default=8)
    parser.add_argument("--thickness-cells", type=int, nargs=3, default=(4, 3, 4))
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                noise=args.noise,
                spec=default_spec(radial_cells=args.radial_cells, cells=args.thickness_cells),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
