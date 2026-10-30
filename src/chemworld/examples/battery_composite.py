"""Authored two-material negative electrode on the existing battery session.

These single-valued curves are synthetic. They do not reproduce the graphite
table or directional silicon curves of Chen2020_composite.
"""

import argparse
import json
from pathlib import Path

from chemworld.agents.base import BaseAgent
from chemworld.data.logging import load_jsonl
from chemworld.data.scientific_inputs import InputSource
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.battery_composite import (
    ActiveMaterialPhaseSpec,
    BatteryCompositeSPMSpec,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def default_spec(*, cells=8):
    source = InputSource(DataSourceProvenance(
        "chemworld:synthetic-composite-example", 0,
        "Authored independent-material transport and single-valued OCV inputs",
        "1", "MIT (ChemWorld-authored example)"), "synthetic")

    def phase(identity, volume, radius, diffusion, cmax, scale, ocv, initial):
        return ActiveMaterialPhaseSpec(
            phase_id=identity,
            particle=ParticleSpec(cells=cells, active_volume_m3=volume, radius_m=radius,
                                  diffusivity_m2_s=diffusion,
                                  maximum_concentration_mol_m3=cmax,
                                  exchange_current_scale_A_m2=scale),
            ocv=ElectrodeOCV("synthetic-"+identity, source, ocv,
                             supported_models=("SPM",)),
            initial_fraction=initial, source=source,
        )

    return BatteryCompositeSPMSpec(
        negative_phases=(
            phase("alpha", 1e-6, 5e-6, 3e-14, 30000., 2., (.25, -.2), .6),
            phase("beta", .35e-6, 4e-6, 1e-14, 50000., 1.8, (.28, -.25), .55),
        ),
        positive_phase=phase("positive", 1.4e-6, 5e-6, 3e-14, 30000., 2., (4.2, -.8), .3),
        provenance="synthetic-two-negative-one-positive-composite-example-v1",
    )


def workflow(spec=None):
    spec = default_spec() if spec is None else spec
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": .12},
        {"operation": "wait", "duration_s": 3.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 3.},
        {"operation": "configure_electrical_control", "current_A": -.1},
        {"operation": "wait", "duration_s": 4.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 2.},
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

    run_agent(env_id="ChemWorld", agent=FixedSequence(), composition=world.to_dict(),
              world_split="public-dev", budget=80, objective="balanced", seed=7,
              observation_noise_mode=noise, output_path=output)
    records = load_jsonl(output)
    if len(records) != len(actions) or any(
            record["transaction_status"] != "committed" for record in records):
        raise RuntimeError("incomplete composite workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "steps": len(records), "replay_steps": report.checked_steps,
            "noise": noise, "trajectory": str(output),
            "scope": "synthetic 2+1 material phases, single-valued OCV, isothermal"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
