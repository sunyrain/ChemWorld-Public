"""Two-reaction MSMR particles in the existing native battery session.

Inputs are authored synthetic parameters. Reaction occupancies and potentials
are derived from each shell's one lithium inventory; this is not a calibrated
commercial cell, a thermal model, or a mechanistic inference experiment.
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
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.battery_msmr import BatteryMSMRSPMSpec
from chemworld.physchem.battery_msmr_materials import MSMRMaterialSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def material_spec(electrode):
    """Two authored reactions with independent occupancies and exchange currents."""
    source = InputSource(DataSourceProvenance(
        "chemworld:synthetic-msmr-example", 0,
        "Authored two-reaction MSMR example; not measured material parameters",
        "1", "MIT (ChemWorld-authored example)"), "synthetic")
    if electrode not in {"negative", "positive"}:
        raise ValueError("electrode must be negative or positive")
    units = {"X": "dimensionless", "U0_V": "V", "w": "dimensionless",
             "jref_A_m2": "A/m^2", "alpha": "dimensionless"}
    values = ((.55, .18, .8, 2., .45), (.45, .28, 1.3, 1.2, .55)) if electrode == (
        "negative") else ((.6, 4.05, .9, 2., .5), (.4, 4.18, 1.2, 1.5, .45))
    rows = tuple(ScientificRow(
        f"{electrode}-{i}", source.provenance.source_id, electrode, f"{electrode}-{i}",
        "parameters",
        {key: InputValue(value, units[key], None, "Uncertainty not supplied")
         for key, value in zip(units, row, strict=True)}) for i, row in enumerate(values))
    table = ScientificTable("synthetic-msmr-"+electrode, "parameters", tuple(
        QuantityColumn(key, unit, "parameter", (0., 10.),
                       "Authored isothermal MSMR input") for key, unit in units.items()),
        (source,), rows)
    return MSMRMaterialSpec(table, tuple(row.row_id for row in rows), electrode)


def default_spec(*, cells=8):
    particle = ParticleSpec(cells=cells, mesh_exponent=1., diffusivity_m2_s=1e-13)
    return BatteryMSMRSPMSpec(
        negative=particle, positive=particle,
        negative_material=material_spec("negative"),
        positive_material=material_spec("positive"),
        provenance="synthetic-two-reaction-MSMR-SPM-example-v1",
    )


def workflow(spec=None):
    spec = default_spec() if spec is None else spec
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": .2},
        {"operation": "wait", "duration_s": 4.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 2.},
        {"operation": "configure_electrical_control", "current_A": -.15},
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
        raise RuntimeError("incomplete MSMR workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output),
            "scope": "synthetic two-reaction MSMR-SPM; fixed temperature and electrolyte"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
