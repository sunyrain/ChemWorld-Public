"""Authored thickness-resolved heat field on the existing battery session.

All inputs are synthetic, with constant net electrode OCV and ideal current
collectors. This is not a calibrated commercial cell or an Arrhenius model.
"""

import argparse
import json
from pathlib import Path

from chemworld.agents.base import BaseAgent
from chemworld.data.logging import load_jsonl
from chemworld.data.scientific_inputs import InputSource
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.physchem.battery_heat_stack import (
    BatteryHeatStackSpec,
    HeatStackCollector,
    HeatStackLayer,
)
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import ParticleSpec
from chemworld.physchem.battery_spatial_thermal import BatterySpatialThermalSPMSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def default_spec(*, cells=8, thermal_cells=3):
    source = InputSource(DataSourceProvenance(
        "chemworld:synthetic-spatial-thermal-example", 0,
        "Authored pouch geometry, constant OCV, fixed transport and thermal properties",
        "1", "MIT (ChemWorld-authored example)"), "synthetic")
    stack = BatteryHeatStackSpec(
        negative=HeatStackLayer(.001, 2e6, .05, thermal_cells, source),
        separator=HeatStackLayer(.0002, 1.5e6, .03, thermal_cells, source),
        positive=HeatStackLayer(.001, 2.5e6, .08, thermal_cells, source),
        negative_collector=HeatStackCollector(20e-6, 3e6, 10., 5., .01, source),
        positive_collector=HeatStackCollector(20e-6, 3e6, 20., 5., .01, source),
        width_m=.1, height_m=.1, edge_heat_transfer_W_m2_K=2.,
        ambient_temperature_K=298.15, source=source,
    )
    particle = ParticleSpec(cells=cells, active_volume_m3=5e-6, radius_m=1e-5,
                            diffusivity_m2_s=1e-13,
                            maximum_concentration_mol_m3=30000.,
                            exchange_current_scale_A_m2=2.)
    return BatterySpatialThermalSPMSpec(
        negative=particle, positive=particle, heat_stack=stack,
        negative_ocv=ElectrodeOCV("constant-negative", source, (.2,),
                                  supported_models=("SPM",)),
        positive_ocv=ElectrodeOCV("constant-positive", source, (4.2,),
                                  supported_models=("SPM",)),
        provenance="synthetic-thickness-thermal-SPM-example-v1",
    )


def workflow(spec=None):
    spec = default_spec() if spec is None else spec
    return battery_world_spec(spec), (
        {"operation": "configure_electrical_control", "current_A": 1.},
        {"operation": "wait", "duration_s": 3.},
        {"operation": "measure", "instrument": "battery_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 3.},
        {"operation": "configure_electrical_control", "current_A": -.8},
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
        raise RuntimeError("incomplete spatial thermal workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "steps": len(records), "replay_steps": report.checked_steps,
            "noise": noise, "trajectory": str(output),
            "scope": ("synthetic pouch-thickness thermal SPM; "
                      "constant OCV, zero entropy, fixed D/BV scale")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
