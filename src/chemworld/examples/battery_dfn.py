"""Offline DFN: the same native CC/rest/CV, measurement and exact replay path.

Development model; independent spatial qualification is tracked in CONV-44.
No PyBaMM or provider is used by this example.
"""

import argparse
import json
from pathlib import Path

from chemworld.examples.battery_spme import run as run_battery_workflow
from chemworld.physchem.battery_dfn import BatteryDFNSpec
from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec
from chemworld.physchem.battery_models import ParticleSpec


def run(output, *, noise="keyed", spec=None):
    spec = spec or BatteryDFNSpec(
        negative=ParticleSpec(cells=12), positive=ParticleSpec(cells=12),
        electrolyte=BatteryElectrolyteSpec(cells=(4, 3, 4)))
    if not isinstance(spec, BatteryDFNSpec):
        raise ValueError("the DFN example requires an explicit DFN model")
    return run_battery_workflow(output, noise=noise, spec=spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--noise", choices=("keyed", "sequential"), default="keyed")
    args = parser.parse_args()
    print("Native DFN CC/rest/CV, measurement and replay: 9 + 9 steps", flush=True)
    print(json.dumps(run(args.output, noise=args.noise), indent=2))


if __name__ == "__main__":
    main()
