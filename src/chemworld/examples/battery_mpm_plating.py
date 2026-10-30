"""Finite particle-distribution plating on the original electrical session.

The two synthetic electrodes have distinct volume-weighted size declarations.
They share one negative metal inventory and the existing charge/rest/discharge
workflow; this example is not a calibrated ageing or lifetime prediction.
"""

import argparse
import json
from pathlib import Path

from chemworld.data.scientific_inputs import InputSource
from chemworld.examples.battery_plating import default_spec as spm_spec
from chemworld.examples.battery_plating import run as run_sequence
from chemworld.examples.battery_plating import workflow as plating_workflow
from chemworld.physchem.battery_mpm import BatteryMPMSpec
from chemworld.physchem.battery_particles import ParticleSizeDistribution
from chemworld.physchem.data_conflicts import DataSourceProvenance


def default_spec(*, mode="reversible"):
    source = InputSource(DataSourceProvenance(
        "mpm-plating-example-synthetic", 0,
        "Explicit synthetic volume-weighted particle radii", "1", "CC0-1.0"), "synthetic")
    return BatteryMPMSpec(
        **{**spm_spec(mode=mode).to_dict(),
           "provenance": "synthetic-MPM-plating-example-v1"},
        negative_distribution=ParticleSizeDistribution(
            (3e-6, 5e-6, 9e-6), (.2, .5, .3), source),
        positive_distribution=ParticleSizeDistribution(
            (4e-6, 7e-6), (.4, .6), source),
    )


def workflow(spec=None):
    spec = default_spec() if spec is None else spec
    if not isinstance(spec, BatteryMPMSpec) or spec.plating is None:
        raise ValueError("this example requires a plating-enabled many-particle model")
    return plating_workflow(spec)


def run(output, *, spec=None, noise="keyed"):
    spec = default_spec() if spec is None else spec
    workflow(spec)
    report = run_sequence(output, spec=spec, noise=noise)
    return {**report, "scope": "synthetic isothermal MPM with one shared pair of metal-Li stocks",
            "particle_counts": list(spec.counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", choices=("reversible", "irreversible"), default="reversible")
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    report = run(args.output, spec=default_spec(mode=args.mode), noise=args.noise)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
