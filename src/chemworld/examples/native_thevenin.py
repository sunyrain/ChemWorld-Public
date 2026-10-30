"""Offline synthetic circuit: sourced parameters, shared clock, meter and replay.

This authored example is not a measured battery or an independent qualification.
"""

import argparse
import json
from dataclasses import replace
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


def workflow(spec=None, *, include_impedance_analysis=False, include_named_controls=False):
    from chemworld.world.thevenin_analysis import THEVENIN_ANALYSIS

    spec = default_spec() if spec is None else spec
    if include_named_controls and spec.control is None:
        spec = control_spec(spec)
    return thevenin_world_spec(spec,
        include_impedance_analysis=include_impedance_analysis), (
        *(({"operation": "measure", "instrument": THEVENIN_ANALYSIS},)
          if include_impedance_analysis else ()),
        {"operation": "configure_electrical_control", "current_A": .2},
        {"operation": "wait", "duration_s": 2.},
        {"operation": "measure", "instrument": "thevenin_meter"},
        {"operation": "configure_electrical_control", "current_A": 0.},
        {"operation": "wait", "duration_s": 1.},
        *((
            {"operation": "configure_electrical_control", "control_mode": "power",
             "power_W": .5, "branch_lower_A": 0., "branch_upper_A": 1.},
            {"operation": "wait", "duration_s": 1.},
            {"operation": "configure_electrical_control", "control_mode": "resistance",
             "resistance_Ohm": 10.},
            {"operation": "wait", "duration_s": 1.},
            {"operation": "configure_electrical_control", "control_mode": "differential_power",
             "power_W": .4},
            {"operation": "wait", "duration_s": 1.},
            {"operation": "configure_electrical_control", "control_mode": "cccv",
             "potential_V": 3.8, "current_limit_A": -.2},
            {"operation": "wait", "duration_s": 1.},
        ) if include_named_controls else ()),
        {"operation": "configure_electrical_control", "potential_V": 3.65},
        {"operation": "wait", "duration_s": 1.},
        {"operation": "measure", "instrument": "thevenin_meter"},
        {"operation": "terminate"},
    )


def diffusion_spec(*, cells=16, time_constant_s=30.):
    """Authored optional planar SOC diffusion, with explicit corrected physics."""
    from chemworld.physchem.battery_thevenin_diffusion import TheveninDiffusionSpec

    if type(cells) is not int or not 2 <= cells <= 128:
        raise ValueError("the example requires 2..128 integer diffusion cells")
    parameters = _table("synthetic-circuit-diffusion", {"diffusion_time_constant_s": "s"},
                        [{"diffusion_time_constant_s": time_constant_s}])
    diffusion = TheveninDiffusionSpec(parameters, "0", tuple(i/cells for i in range(cells+1)))
    return replace(default_spec(), diffusion=diffusion)


def control_spec(spec=None):
    """Explicit synthetic controller gains and persistent current-memory state."""
    from chemworld.physchem.battery_thevenin_control import TheveninControlSpec

    values = {"initial_current_A": .1, "current_scale_A": 1., "power_scale_W": 1.,
        "resistance_scale_Ohm": 1., "voltage_scale_V": 1.,
        "power_response_time_s": 100., "resistance_response_time_s": 100.,
        "voltage_response_time_s": 1., "antiwindup_time_s": 100./3600.}
    units = {key: ("A" if key.endswith("_A") else "W" if key.endswith("_W")
        else "Ohm" if key.endswith("_Ohm") else "V" if key.endswith("_V") else "s")
        for key in values}
    declaration = TheveninControlSpec(_table("synthetic-controller", units, [values]), "0")
    return replace(default_spec() if spec is None else spec, control=declaration)


def run(output, *, noise="keyed", spec=None, include_impedance_analysis=False,
        include_named_controls=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError("preserve previous trajectories; select a new output path")
    world, actions = workflow(spec, include_impedance_analysis=include_impedance_analysis,
                              include_named_controls=include_named_controls)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    parser.add_argument("--analysis", action="store_true",
        help="Opt-in synthetic stationary model analysis; not a physical EIS acquisition")
    parser.add_argument("--diffusion", action="store_true",
        help="Enable the authored 16-cell planar diffusion example with corrected voltage/heat")
    parser.add_argument("--controls", action="store_true",
        help="Exercise explicit power/load/dynamic/CCCV controls and persistent controller memory")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise,
                         spec=diffusion_spec() if args.diffusion else None,
                         include_impedance_analysis=args.analysis,
                         include_named_controls=args.controls), indent=2))


if __name__ == "__main__":
    main()
