"""Read fixed-condition QSD rates from the existing harmonic RRKM gas example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.examples.rrkm_gas_network import example_inputs
from chemworld.physchem.energy_grain_reduction import energy_grain_qsd_table
from chemworld.physchem.rrkm_grain_binding import compile_rrkm_grains
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.energy_grain_analysis import analyze_energy_grain_world_state
from chemworld.world.energy_grain import rrkm_grain_world


def example_report() -> dict[str, Any]:
    model, reactor, amounts = example_inputs()
    composition = rrkm_grain_world(model, reactor, amounts)
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=38)
        before = runtime._state.to_dict()
        report = analyze_energy_grain_world_state(runtime._state, composition)
        assert runtime._state.to_dict() == before
        return {
            "current_world": report,
            "condition_table": energy_grain_qsd_table(
                compile_rrkm_grains(model),
                temperatures_K=(400.0, 600.0, 900.0),
                bath_pressures_Pa=(1e3, 1e4, 1e5),
                bath_mole_fractions={"He": 5 / 6, "Ar": 1 / 6},
            ),
            "state_unchanged": True,
        }
    finally:
        runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(example_report(), stream, indent=2, allow_nan=False)
    print(f"QSD report written to {args.output}; original world unchanged", flush=True)


if __name__ == "__main__":
    main()
