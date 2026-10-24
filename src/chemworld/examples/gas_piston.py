"""Finite-load gas piston using original gas inventory, world clock and heat device."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_network_pressure import GasPressureControlSpec
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def mechanism_document() -> dict[str, Any]:
    """Authored two-species ideal chemistry; not a real material parameter set."""
    return {
        "units": {"length": "m", "quantity": "mol", "activation-energy": "J/mol"},
        "phases": [
            {
                "name": "piston",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": ["H"],
                "species": ["A", "B"],
            }
        ],
        "species": [
            {
                "name": name,
                "composition": {"H": atoms},
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": f"{h} J/mol",
                    "s0": "30 J/mol/K",
                    "cp0": f"{cp} J/mol/K",
                    "T-min": 200.0,
                    "T-max": 2000.0,
                },
            }
            for name, atoms, h, cp in (("A", 2, 0.0, 30.0), ("B", 4, -9000.0, 44.0))
        ],
        "reactions": [{"equation": "B => 2 A", "rate-constant": {"A": 0.3, "b": 0.0, "Ea": 0.0}}],
    }


def example_inputs(*, thermal: bool = False, connected: bool = False) -> tuple[Any, ...]:
    network = convert_mechanism(mechanism_document())
    nodes: tuple[GasReactorNodeSpec, ...] = (GasReactorNodeSpec("piston", 1.0),)
    edges: tuple[GasFlowConnectionSpec, ...] = ()
    temperatures = {
        "piston": 600.0,
        "feed": 700.0 if thermal else 600.0,
        "receiver": 500.0 if thermal else 600.0,
    }
    if connected:
        nodes += (
            GasReactorNodeSpec("feed", 1.0, chemistry_enabled=False),
            GasReactorNodeSpec("receiver", 2.0, chemistry_enabled=False),
        )
        edges = (
            GasFlowConnectionSpec("inlet", "feed", "piston", coefficient=0.0006),
            GasFlowConnectionSpec("outlet", "piston", "receiver", coefficient=0.0004),
        )
    spec = GasReactorNetworkSpec(
        nodes,
        edges,
        600.0,
        "piston",
        energy_nodes=tuple(
            GasNodeEnergySpec(
                n.vessel_id,
                temperatures[n.vessel_id],
                0.5,
                minimum_temperature_K=300.0,
                maximum_temperature_K=1200.0,
            )
            for n in nodes
        )
        if thermal
        else (),
        pressure_control=GasPressureControlSpec("piston", 100000.0, 0.4, 2.0),
    )
    pressures = {"piston": 100000.0, "feed": 150000.0, "receiver": 50000.0}
    fractions = {"piston": 0.2, "feed": 0.7, "receiver": 0.1}
    amounts = {
        n.vessel_id: {
            s: x * pressures[n.vessel_id] * n.volume_L / (1000 * R * temperatures[n.vessel_id])
            for s, x in (("A", fractions[n.vessel_id]), ("B", 1 - fractions[n.vessel_id]))
        }
        for n in nodes
    }
    return spec, network, amounts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thermal", action="store_true")
    parser.add_argument("--connected", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain attempts; choose a fresh output")
    runtime = ChemWorldRuntime(
        composition=gas_reactor_network_world(
            *example_inputs(thermal=args.thermal, connected=args.connected)
        )
    )
    try:
        runtime.reset(seed=35)
        runtime.execute({"operation": "select_vessel", "vessel": "piston"})
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.2})
        args.output.write_text(
            json.dumps(
                {"info": dict(outcome.info), "state": runtime._state.to_dict()},
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": outcome.info["transaction_status"],
                    "time_s": runtime._state.ledger.time_s,
                    "volume_L": runtime._state.volume_L,
                }
            ),
            flush=True,
        )
        if outcome.info["transaction_status"] != "committed":
            raise RuntimeError(outcome.info)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
