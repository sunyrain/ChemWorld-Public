"""Read-only steady analysis of two original finite, thermally connected reactors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec
from chemworld.physchem.gas_reactor_network import (
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
    gas_inventory_key,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_network_steady import analyze_gas_network_steady_world
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def example_world() -> Any:
    declaration = {
        "phases": [
            {
                "name": "gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "species": ["A", "B"],
                "elements": ["H"],
            }
        ],
        "units": {"length": "m", "quantity": "mol", "activation-energy": "J/mol"},
        "species": [
            {
                "name": name,
                "composition": {"H": 2},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": h,
                    "s0": 40,
                    "cp0": 32,
                    "T-min": 200,
                    "T-max": 2000,
                },
            }
            for name, h in (("A", 0), ("B", -2000))
        ],
        "reactions": [{"equation": "A <=> B", "rate-constant": {"A": 2, "b": 0, "Ea": 0}}],
    }
    network = convert_mechanism(declaration)
    spec = GasReactorNetworkSpec(
        (GasReactorNodeSpec("hot", 1), GasReactorNodeSpec("cold", 1)),
        (),
        700,
        "hot",
        (GasNodeEnergySpec("hot", 700, 1), GasNodeEnergySpec("cold", 400, 1)),
        (GasThermalLinkSpec("contact", "hot", "cold", 0.1),),
    )
    return gas_reactor_network_world(
        spec, network, {n: {"A": 0.015, "B": 0.005} for n in ("hot", "cold")}
    )


def example_scales() -> dict[str, float]:
    return {gas_inventory_key(n, s): 0.02 for n in ("hot", "cold") for s in ("A", "B")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve existing analysis output")
    composition = example_world()
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=35)
        before = runtime._state.to_dict()
        result = analyze_gas_network_steady_world(
            runtime._state, composition, amount_scales_mol=example_scales()
        )
        result["input_world_unchanged"] = runtime._state.to_dict() == before
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "world_advanced": False,
                    "original_rate_evaluations": result["original_rate_evaluations"],
                }
            )
        )
        if result["status"] != "steady_state_found" or not result["input_world_unchanged"]:
            raise RuntimeError(result.get("failure"))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
