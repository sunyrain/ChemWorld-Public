"""Read local reaction responses through a real finite two-vessel world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.physchem.gas_network_sensitivity import GasSensitivityParameter
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_network_sensitivity import analyze_gas_network_world_state
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def example_inputs() -> tuple[Any, Any, Any]:
    payload = {
        "units": {"length": "m", "quantity": "kmol", "activation-energy": "J/kmol"},
        "phases": [
            {
                "name": "profile-gas",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": ["H"],
                "species": ["A", "B"],
            }
        ],
        "species": [
            {
                "name": name,
                "composition": {"H": 2},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{h} J/mol",
                    "s0": "40 J/mol/K",
                    "cp0": "32 J/mol/K",
                    "T-min": 150,
                    "T-max": 2000,
                },
            }
            for name, h in (("A", 0), ("B", -10000))
        ],
        "reactions": [{"equation": "A => B", "rate-constant": {"A": 0.7, "b": 0, "Ea": 0}}],
    }
    network = convert_mechanism(payload)
    spec = GasReactorNetworkSpec(
        (GasReactorNodeSpec("source", 1), GasReactorNodeSpec("receiver", 3)),
        (GasFlowConnectionSpec("feed", "source", "receiver", coefficient=0.003),),
        700,
        "source",
    )
    amounts = {"source": {"A": 0.02, "B": 0.01}, "receiver": {"A": 0.003, "B": 0.002}}
    return spec, network, amounts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve old reports; choose a new output path")
    spec, network, amounts = example_inputs()
    composition = gas_reactor_network_world(spec, network, amounts)
    runtime = ChemWorldRuntime(composition=composition)
    try:
        runtime.reset(seed=7)
        runtime.execute({"operation": "wait", "duration_s": 0.25})
        result = analyze_gas_network_world_state(
            runtime._state,
            composition,
            (GasSensitivityParameter("reaction", network.reactions[0].reaction_id, "A"),),
            (0.0, 0.1, 0.3),
            progress=lambda p: print(json.dumps(p), flush=True),
        )
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({key: result[key] for key in ("status", "work", "world_consumer")}))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
