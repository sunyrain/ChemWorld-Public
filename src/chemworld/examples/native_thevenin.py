"""Offline synthetic circuit: sourced parameters, shared clock, meter and replay.

This authored example is not a measured battery or an independent qualification.
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
from chemworld.physchem.battery_thevenin import CONSTANT_UNITS, TheveninSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.tabulated_functions import TabulatedScalar
from chemworld.world.thevenin_world import thevenin_world_spec


def _table(identity, units, values):
    source = InputSource(DataSourceProvenance("chemworld:synthetic-thevenin-example", 0,
        "Self-authored RC/two-node circuit fixture; not measured material data", "1",
        "MIT (ChemWorld-authored example)"), "synthetic")
    return ScientificTable(identity, "parameters", tuple(
        QuantityColumn(key, unit, "parameter", (-1e9, 1e9), "Authored synthetic circuit fixture")
        for key, unit in units.items()), (source,), tuple(
        ScientificRow(str(i), source.provenance.source_id, identity, str(i), "parameters",
            {key: InputValue(value, units[key], None, "Input uncertainty not supplied")
             for key, value in row.items()}) for i, row in enumerate(values)))


def default_spec():
    def constant(identity, value, unit):
        return TabulatedScalar(_table(identity, {"value": unit}, [{"value": value}]),
                               (), "value", unit)

    values = {"capacity_C": 100., "maximum_current_A": 2., "minimum_voltage_V": .1,
        "maximum_voltage_V": 5., "initial_soc": .7, "minimum_soc": 0., "maximum_soc": 1.,
        "initial_cell_temperature_K": 298.15, "initial_jig_temperature_K": 298.15,
        "minimum_temperature_K": 250., "maximum_temperature_K": 400.,
        "cell_heat_capacity_J_K": 100., "jig_heat_capacity_J_K": 500.,
        "cell_jig_conductance_W_K": .5, "jig_ambient_conductance_W_K": .2,
        "reference_frame_temperature_K": 298.15, "housing_volume_L": .01,
        "initial_rc_0_V": 0., "initial_rc_1_V": 0.}
    units = {**CONSTANT_UNITS, "initial_rc_0_V": "V", "initial_rc_1_V": "V"}
    ocv = _table("synthetic-ocv", {"soc": "dimensionless", "voltage": "V"},
                 [{"soc": 0., "voltage": 3.}, {"soc": 1., "voltage": 4.}])
    return TheveninSpec(_table("synthetic-circuit-constants", units, [values]), "0",
        TabulatedScalar(ocv, (("soc", "soc", "dimensionless"),), "voltage", "V"),
        constant("r0", .1, "Ohm"),
        ((constant("r1", .2, "Ohm"), constant("c1", 5., "F")),
         (constant("r2", .4, "Ohm"), constant("c2", 2.5, "F"))),
        constant("entropy", 1e-4, "V/K"), ScalarControlProgram(((0., 298.15),), "K"))


def workflow(spec=None):
    return thevenin_world_spec(default_spec() if spec is None else spec), (
        {"operation": "configure_electrical_control", "current_A": .2},
        {"operation": "wait", "duration_s": 2.},
        {"operation": "measure", "instrument": "thevenin_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 1.},
        {"operation": "configure_electrical_control", "potential_V": 3.65},
        {"operation": "wait", "duration_s": 1.},
        {"operation": "measure", "instrument": "thevenin_meter"},
        {"operation": "terminate"},
    )


def run(output, *, noise="keyed", spec=None):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve previous trajectories; select a new output path")
    world, actions = workflow(spec)

    class FixedSequence(BaseAgent):
        def act(self, history):
            return actions[len(history)]

    run_agent(env_id="ChemWorld", agent=FixedSequence(), composition=world.to_dict(),
        world_split="public-dev", budget=80, objective="balanced", seed=4,
        observation_noise_mode=noise, output_path=output)
    records = load_jsonl(output)
    if len(records) != len(actions) or any(r["transaction_status"] != "committed" for r in records):
        raise RuntimeError("incomplete circuit workflow; all records retained")
    report = verify_records(records, tolerance=1e-8)
    if not report.verified:
        raise RuntimeError(report.mismatches)
    return {"model": "thevenin-empirical-rc-thermal-v1", "steps": len(records),
            "replay_steps": report.checked_steps, "noise": noise, "trajectory": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
