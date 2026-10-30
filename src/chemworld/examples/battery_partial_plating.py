"""Synthetic constant-rate metal deactivation in the original SPM workflow.

The declared rate transfers plated Li to dead Li without an extra electron
current. It is not a calibrated SEI-thickness or material-lifetime mechanism.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.battery_plating import default_spec as base_spec
from chemworld.examples.battery_plating import run as run_sequence
from chemworld.physchem.battery_plating import LithiumPlatingSpec
from chemworld.physchem.data_conflicts import DataSourceProvenance


def default_spec(*, gamma_s_inv=.02):
    spec = base_spec()
    source = InputSource(DataSourceProvenance(
        "partial-plating-example-synthetic", 0,
        "Explicit constant-rate metal deactivation example", "1", "CC0-1.0",
    ), "synthetic")
    table = ScientificTable(
        "partial-plating-example-parameters", "parameters",
        (*spec.plating.parameters.columns,
         QuantityColumn("gamma_s_inv", "1/s", "parameter", (0., 1.),
                        "Authored constant deactivation rate")),
        (source,),
        (ScientificRow(
            "example", source.provenance.source_id, "group", "cell", "parameters",
            {**spec.plating.parameters.rows[0].values,
             "gamma_s_inv": InputValue(gamma_s_inv, "1/s", None,
                                       "Synthetic constant rate, not a material fit")},
        ),),
    )
    return replace(spec, plating=LithiumPlatingSpec(
        table, "example", "partially reversible",
        initial_plated_mol=4e-5, initial_dead_mol=5e-6,
    ), provenance="synthetic-SPM-constant-deactivation-example-v1")


def run(output, *, spec=None, noise="keyed"):
    spec = default_spec() if spec is None else spec
    result = run_sequence(output, spec=spec, noise=noise)
    return {**result, "gamma_s_inv": spec.plating.decay_rate_s_inv,
            "scope": "synthetic isothermal SPM; explicit constant metal deactivation"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gamma", type=float, default=.02, help="Constant rate in 1/s")
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, spec=default_spec(gamma_s_inv=args.gamma),
                         noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
