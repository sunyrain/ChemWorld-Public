"""Synthetic SPM memory driven by main current, including rest-time stripping.

The two operational OCP branches and the metal kinetics are authored examples,
not measured material parameters or a full chemical-energy/lifetime model.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from chemworld.examples.battery_hysteresis import default_spec as hysteresis_spec
from chemworld.examples.battery_plating import default_spec as plating_spec
from chemworld.examples.battery_plating import run as run_sequence


def default_spec():
    base = plating_spec()
    memory = hysteresis_spec()
    return replace(
        base, negative_ocp=memory.negative_ocp, positive_ocp=memory.positive_ocp,
        provenance="synthetic-SPM-Axen-reversible-plating-example-v1",
    )


def run(output, *, spec=None, noise="keyed"):
    spec = default_spec() if spec is None else spec
    result = run_sequence(output, spec=spec, noise=noise)
    return {**result, "hysteresis_coordinates": list(spec.hysteresis_keys),
            "scope": "synthetic isothermal SPM; Axen memory follows main reaction current"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
