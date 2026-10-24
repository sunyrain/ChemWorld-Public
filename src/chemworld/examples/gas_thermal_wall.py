"""Independent finite wall cooling and reaction in the original gas world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.gas_thermal_wall import GAS_WALL_TYPE, GasThermalWallSpec, GasWallContact
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def mechanism_document(*, reactive: bool = True) -> dict[str, Any]:
    """Two synthetic same-formula species; no real-material prediction is claimed."""
    return {
        "units": {"length": "m", "quantity": "mol", "activation-energy": "J/mol"},
        "phases": [
            {
                "name": "wall-example",
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
                    "s0": "30 J/mol/K",
                    "cp0": "30 J/mol/K",
                    "T-min": 200.0,
                    "T-max": 2000.0,
                },
            }
            for name, h in (("A", 0), ("B", -12000))
        ],
        "reactions": [{"equation": "A => B", "rate-constant": {"A": 4.0, "b": 0.0, "Ea": 8000.0}}]
        if reactive
        else [],
    }


def example_inputs(*, connected: bool = False, reactive: bool = True) -> tuple[Any, ...]:
    # Keep a valid imported mechanism. An inert test disables the original
    # node's chemistry instead of passing an unsupported empty mechanism.
    network = convert_mechanism(mechanism_document())
    nodes: tuple[GasReactorNodeSpec, ...] = (
        GasReactorNodeSpec("reactor", 1.0, chemistry_enabled=reactive),
    )
    edges: tuple[GasFlowConnectionSpec, ...] = ()
    amounts = {"reactor": {"A": 0.02, "B": 0.005}}
    temperatures = {"reactor": 450.0, "feed": 700.0, "receiver": 350.0}
    contacts: tuple[GasWallContact, ...] = (GasWallContact("reactor", 0.15),)
    if connected:
        nodes += (
            GasReactorNodeSpec("feed", 1.0, chemistry_enabled=False),
            GasReactorNodeSpec("receiver", 2.0, chemistry_enabled=False),
        )
        edges = (
            GasFlowConnectionSpec("in", "feed", "reactor", coefficient=0.0003),
            GasFlowConnectionSpec("out", "reactor", "receiver", coefficient=0.0002),
        )
        amounts.update({"feed": {"A": 0.03, "B": 0.01}, "receiver": {"A": 0.01, "B": 0.01}})
        contacts += (GasWallContact("receiver", 0.05),)
    spec = GasReactorNetworkSpec(
        nodes,
        edges,
        450.0,
        "reactor",
        energy_nodes=tuple(
            GasNodeEnergySpec(
                n.vessel_id,
                temperatures[n.vessel_id],
                0.5,
                minimum_temperature_K=250.0,
                maximum_temperature_K=1200.0,
            )
            for n in nodes
        ),
        thermal_wall=GasThermalWallSpec(
            "wall",
            contacts,
            800.0,
            2.0,
            loss_conductance_W_K=0.02,
            emissivity=0.8,
            radiating_area_m2=0.001,
            minimum_temperature_K=250.0,
            maximum_temperature_K=1200.0,
        ),
    )
    return spec, network, amounts


def wall_temperature(runtime: ChemWorldRuntime) -> float:
    return next(
        s.values["temperature_K"]
        for local in (runtime._state, *runtime._state.inactive_vessels.values())
        for s in local.state_slots.slots
        if s.spec.state_type == GAS_WALL_TYPE
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--connected", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("retain previous attempts; select a fresh output")
    runtime = ChemWorldRuntime(
        composition=gas_reactor_network_world(*example_inputs(connected=args.connected))
    )
    try:
        runtime.reset(seed=35)
        history = []
        for _ in range(2):
            outcome = runtime.execute({"operation": "wait", "duration_s": 0.2})
            history.append({"info": dict(outcome.info), "state": runtime._state.to_dict()})
            if outcome.info["transaction_status"] != "committed":
                break
        args.output.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        print(
            json.dumps(
                {
                    "completed": len(history),
                    "time_s": runtime._state.ledger.time_s,
                    "gas_temperature_K": runtime._state.temperature_K,
                    "wall_temperature_K": wall_temperature(runtime),
                }
            ),
            flush=True,
        )
        if len(history) != 2 or any(
            h["info"]["transaction_status"] != "committed" for h in history
        ):
            raise RuntimeError(history[-1]["info"].get("outcome_diagnostics"))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
