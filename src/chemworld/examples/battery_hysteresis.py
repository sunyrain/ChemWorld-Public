"""Synthetic directional OCP through the existing battery runtime and replay.

Authored branches illustrate persistent electrode memory; they are not fitted
material data or an independent qualification. Both SPM and MPM remain isothermal.
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
from chemworld.physchem.battery_materials import ElectrodeOCV
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_ocp import ElectrodeOCPSpec
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.world.battery_world import battery_world_spec


def default_spec(*, mode="axen", many_particle=False, cells=8):
    if mode not in {"axen", "current_sigmoid"}:
        raise ValueError("this authored example requires axen or current_sigmoid")
    source = InputSource(DataSourceProvenance(
        "chemworld:synthetic-hysteresis-example", 0,
        "Self-authored OCP branches and parameters; not measured battery data",
        "1", "MIT (ChemWorld-authored example)"), "synthetic")

    def electrode(name, intercept, slope, width, initial_h):
        if mode == "axen":
            values = {"initial_h": initial_h, "lithiation_decay": 20.,
                      "delithiation_decay": 40.}
            units = dict.fromkeys(values, "dimensionless")
        else:
            values = {"collector_area_m2": .02, "sigmoid_slope_m2_A": 100.}
            units = {"collector_area_m2": "m^2", "sigmoid_slope_m2_A": "m^2/A"}
        table = ScientificTable(name+"-ocp", "parameters", tuple(
            QuantityColumn(key, unit, "parameter", (-1e6, 1e6),
                           "Authored synthetic directional OCP parameter")
            for key, unit in units.items()), (source,), (
                ScientificRow("0", source.provenance.source_id, name, name, "parameters",
                    {key: InputValue(value, units[key], None, "Uncertainty not supplied")
                     for key, value in values.items()}),))
        curves = [ElectrodeOCV(name, source, (intercept+offset, slope),
                              supported_models=("SPM", "MPM"))
                  for offset in (-width/2, width/2)]
        return ElectrodeOCPSpec(mode, curves[0], curves[1], table, "0")

    arguments = {"negative": ParticleSpec(cells=cells), "positive": ParticleSpec(cells=cells),
        "negative_ocp": electrode("negative", .2, -.1, .08, .2),
        "positive_ocp": electrode("positive", 4.2, -.8, .12, -.3),
        "provenance": "synthetic-directional-ocp-example-v1"}
    if many_particle:
        distribution = ParticleSizeDistribution((8e-6, 12e-6), (.5, .5), source)
        return BatteryMPMSpec(**arguments, negative_distribution=distribution,
                              positive_distribution=distribution)
    return BatterySPMSpec(**arguments)


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
        raise RuntimeError("incomplete directional OCP workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": spec.model_id, "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output),
            "hysteresis_coordinates": list(spec.hysteresis_keys),
            "scope": "synthetic isothermal directional OCP; no hysteresis energy model"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    parser.add_argument("--mode", choices=("axen", "current_sigmoid"), default="axen")
    parser.add_argument("--many-particle", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise,
        spec=default_spec(mode=args.mode, many_particle=args.many_particle)), indent=2))


if __name__ == "__main__":
    main()
