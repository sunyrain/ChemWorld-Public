"""A finite lithium-metal half-cell using the existing battery session.

All parameters are authored synthetic inputs. The foil's area and conduction
length remain fixed while its actual lithium inventory changes; this example
does not claim measured-material or variable-thickness accuracy.
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
from chemworld.physchem.battery_half_cell import (
    BatteryHalfCellSPMSpec,
    LithiumMetalCounterElectrodeSpec,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def default_spec(*, cells=8, metal_thickness_m=15e-6):
    source = InputSource(DataSourceProvenance(
        "chemworld:synthetic-half-cell-example", 0,
        "Authored finite lithium-metal example; not measured material parameters",
        "1", "MIT (ChemWorld-authored example)"), "synthetic")
    values = {
        "area_m2": (.02, "m^2"),
        "thickness_m": (metal_thickness_m, "m"),
        "conductivity_S_m": (1e7, "S/m"),
        "partial_molar_volume_m3_mol": (13e-6, "m^3/mol"),
        "exchange_current_density_A_m2": (2., "A/m^2"),
        "reference_temperature_K": (298.15, "K"),
        "electrolyte_concentration_mol_m3": (1000., "mol/m^3"),
    }
    table = ScientificTable("synthetic-metal", "parameters", tuple(
        QuantityColumn(key, unit, "parameter", (0., 1e12),
                       "Authored fixed-geometry half-cell input")
        for key, (_, unit) in values.items()), (source,), (
            ScientificRow("metal", source.provenance.source_id, "Li", "Li", "parameters",
                {key: InputValue(value, unit, None, "Uncertainty not supplied")
                 for key, (value, unit) in values.items()}),))
    return BatteryHalfCellSPMSpec(
        metal=LithiumMetalCounterElectrodeSpec(table, "metal"),
        positive=ParticleSpec(cells=cells),
        positive_ocv=ElectrodeOCV("synthetic-positive", source, (4.2, -.8),
                                  supported_models=("SPM",)),
        provenance="synthetic-fixed-geometry-finite-metal-example-v1",
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
        raise RuntimeError("incomplete finite half-cell workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output),
            "initial_metal_mol": spec.metal.initial_amount_mol,
            "scope": "synthetic fixed-geometry finite lithium; no thermal or ageing model"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
