"""Finite catalytic source/receiver with original slots and one shared clock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec, GasThermalLinkSpec
from chemworld.physchem.gas_network_surface import GasSurfaceAttachment
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def example_inputs(*, size: int = 1, thermal: bool = False, both: bool = False) -> tuple[Any, ...]:
    document = authored_declaration(site_occupancy=size)
    bundle = convert_surface_mechanism(
        document,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1400,
    )
    attachments = tuple(
        GasSurfaceAttachment(name, name + ":catalyst", bundle.surface, 0.1, 0.2)
        for name in (("source", "receiver") if both else ("source",))
    )
    spec = GasReactorNetworkSpec(
        (GasReactorNodeSpec("source", 1), GasReactorNodeSpec("receiver", 3)),
        (GasFlowConnectionSpec("feed", "source", "receiver", "mass-flow", 6e-6),),
        700,
        "source",
        energy_nodes=(
            GasNodeEnergySpec(
                "source",
                700,
                support_heat_capacity_J_K=1,
                minimum_temperature_K=300,
                maximum_temperature_K=1400,
            ),
            GasNodeEnergySpec(
                "receiver",
                500,
                support_heat_capacity_J_K=2,
                minimum_temperature_K=300,
                maximum_temperature_K=1400,
            ),
        )
        if thermal
        else (),
        heat_links=(GasThermalLinkSpec("wall", "source", "receiver", 0.01),) if thermal else (),
        surfaces=attachments,
    )
    amounts = {"source": {"A": 0.02, "B": 0.01}, "receiver": {"A": 0.003, "B": 0.002}}
    theta = (
        {"Pt": 0.8, "HA": 0.1, "HB": 0.1}
        if size == 1
        else {
            "Pt": 0.55,
            "HA": 0.2,
            "HB": 0.15,
            "HAA": 0.1,
        }
    )
    initial = {
        a.owner_id: {
            s: value * 1e-6 / bundle.surface.site_occupancies[s] for s, value in theta.items()
        }
        for a in attachments
    }
    return document, spec, bundle.gas_network, amounts, initial


def example_world(**kwargs: Any) -> Any:
    _, spec, network, amounts, surfaces = example_inputs(**kwargs)
    return gas_reactor_network_world(spec, network, amounts, surface_amounts_by_owner=surfaces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--thermal", action="store_true")
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve attempts; choose a new report path")
    runtime = ChemWorldRuntime(
        composition=example_world(size=args.size, thermal=args.thermal, both=args.both)
    )
    try:
        runtime.reset(seed=35)
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.1})
        report = {"info": dict(outcome.info), "state": runtime._state.to_dict()}
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(
            json.dumps(
                {
                    "transaction": outcome.info["transaction_status"],
                    "time_s": runtime._state.ledger.time_s,
                }
            )
        )
        if outcome.info["transaction_status"] != "committed":
            raise RuntimeError(outcome.info)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
