"""Finite feed and product vessels connected through a reacting gas column."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.gas_spatial_transport import GasSpatialTransportSpec
from chemworld.physchem.gas_transport import gas_transport_from_network
from chemworld.physchem.mechanism_import import convert_mechanism
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def example_inputs(
    *, cells: int = 4, non_equimolar: bool = False, flux_model: str = "stefan-maxwell"
) -> tuple[Any, ...]:
    """Authored synthetic chemistry and transport, not a real material parameter set."""
    document = {
        "units": {"length": "m", "quantity": "mol", "activation-energy": "J/mol"},
        "phases": [
            {
                "name": "finite-flow-column",
                "thermo": "ideal-gas",
                "kinetics": "gas",
                "elements": ["H"],
                "species": ["A", "B"],
            }
        ],
        "species": [
            {
                "name": name,
                "composition": {"H": 4 if name == "B" and non_equimolar else 2},
                "transport": {
                    "model": "gas",
                    "geometry": "linear",
                    "diameter": 3.5,
                    "well-depth": 100.0,
                },
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
            for name, h, cp in (("A", 0.0, 30.0), ("B", -9000.0, 44.0))
        ],
        "reactions": [
            {
                "equation": "B => 2 A" if non_equimolar else "A => B",
                "rate-constant": {"A": 0.3, "b": 0.0, "Ea": 0.0},
            }
        ],
    }
    network = convert_mechanism(document)
    mesh = DiffusionMesh.planar(0.04, cells, area_m2=0.002)
    names = tuple(f"cell-{i:03d}" for i in range(cells))
    spatial = GasSpatialTransportSpec(
        mesh,
        names,
        gas_transport_from_network(network, thermal_diffusion="off"),
        flux_model,
        boundary_mode="finite-flow",
    )
    nodes = (
        GasReactorNodeSpec("feed", 1.0, chemistry_enabled=False),
        *(
            GasReactorNodeSpec(name, float(v) * 1000)
            for name, v in zip(names, mesh.cell_volumes_m3, strict=True)
        ),
        GasReactorNodeSpec("receiver", 2.0, chemistry_enabled=False),
    )
    spec = GasReactorNetworkSpec(
        nodes,
        (
            GasFlowConnectionSpec("inlet", "feed", names[0], coefficient=0.0004),
            GasFlowConnectionSpec("outlet", names[-1], "receiver", coefficient=0.0004),
        ),
        600.0,
        names[0],
        spatial_transport=spatial,
    )
    fractions = {
        "feed": 0.8,
        "receiver": 0.1,
        **{name: 0.1 + 0.6 * (i + 0.5) / cells for i, name in enumerate(names)},
    }
    pressures = {"feed": 160000.0, "receiver": 40000.0}
    amounts = {
        node.vessel_id: {
            s: fraction
            * pressures.get(node.vessel_id, 100000.0)
            * node.volume_L
            / (1000 * R * spec.temperature_K)
            for s, fraction in (
                ("A", fractions[node.vessel_id]),
                ("B", 1 - fractions[node.vessel_id]),
            )
        }
        for node in nodes
    }
    return spec, network, amounts


def example_world(**kwargs: Any) -> Any:
    return gas_reactor_network_world(*example_inputs(**kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--non-equimolar", action="store_true")
    parser.add_argument("--cells", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("preserve attempts; choose a fresh output path")
    runtime = ChemWorldRuntime(
        composition=example_world(cells=args.cells, non_equimolar=args.non_equimolar)
    )
    try:
        runtime.reset(seed=36)
        outcome = runtime.execute({"operation": "wait", "duration_s": 0.2})
        payload = {"info": dict(outcome.info), "state": runtime._state.to_dict()}
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
        print(
            json.dumps(
                {
                    "transaction": outcome.info["transaction_status"],
                    "time_s": runtime._state.ledger.time_s,
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
