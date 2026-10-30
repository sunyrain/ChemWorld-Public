"""Synthetic finite SPM plating through the existing electrical workflow.

Charge, rest and discharge move real active/plated Li in one persistent cell.
The fixed-temperature example is not a material or lifetime calibration.
"""

import argparse
import json
from pathlib import Path

from chemworld.agents.base import BaseAgent
from chemworld.data.logging import load_jsonl
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.physchem.battery_plating import LithiumPlatingSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def default_spec(*, mode="reversible"):
    source = InputSource(DataSourceProvenance(
        "plating-example-synthetic", 0, "Explicit finite-SPM plating example", "1", "CC0-1.0"
    ), "synthetic")
    values = {
        "k_m_s": (1e-9, "m/s"),
        "alpha": (.5, "dimensionless"),
        "ce_mol_m3": (1000., "mol/m^3"),
        "negative_electrode_volume_m3": (2e-6, "m^3"),
    }
    parameters = ScientificTable(
        "plating-example-parameters", "parameters",
        tuple(QuantityColumn(key, unit, "parameter", (0., 1. if key == "alpha" else value*10),
                             "Synthetic example input")
              for key, (value, unit) in values.items()),
        (source,),
        (ScientificRow(
            "example", source.provenance.source_id, "group", "cell", "parameters",
            {key: InputValue(value, unit, None, "Authored example input")
             for key, (value, unit) in values.items()},
        ),),
    )
    particle = ParticleSpec(cells=8, radius_m=5e-6, diffusivity_m2_s=1e-12)
    return BatterySPMSpec(
        negative=particle, positive=particle,
        plating=LithiumPlatingSpec(parameters, "example", mode, initial_plated_mol=4e-5),
        provenance="synthetic-SPM-plating-example-v1",
    )


def workflow(spec=None):
    spec = default_spec() if spec is None else spec
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": -.5},
        {"operation": "wait", "duration_s": 4.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 2.},
        {"operation": "configure_electrical_control", "current_A": .3},
        {"operation": "wait", "duration_s": 4.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 1.},
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
              world_split="public-dev", budget=80, objective="balanced", seed=4,
              observation_noise_mode=noise, output_path=output)
    records = load_jsonl(output)
    if len(records) != len(actions) or any(
            record["transaction_status"] != "committed" for record in records):
        raise RuntimeError("incomplete plating workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "mode": spec.plating.mode, "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output),
            "scope": "synthetic fixed-temperature SPM and two real metal-Li stocks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("reversible", "irreversible"), default="reversible")
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    result = run(args.output, spec=default_spec(mode=args.mode), noise=args.noise)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
