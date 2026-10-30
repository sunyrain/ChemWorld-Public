"""Authored harmonic RRKM competing products in the original finite gas world."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.energy_grain_specs import (
    EnergyGrainBathGas,
    EnergyGrainState,
    EnergyTransferBath,
    InternalModeBaseline,
    R,
)
from chemworld.physchem.gas_reactor_network import (
    GasFlowConnectionSpec,
    GasReactorNetworkSpec,
    GasReactorNodeSpec,
)
from chemworld.physchem.rrkm_density_states import RRKMEnergyGrid, RRKMHarmonicState
from chemworld.physchem.rrkm_grain_binding import (
    RRKMGrainSpec,
    RRKMProductChannel,
    rrkm_grain_network,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.energy_grain import rrkm_grain_world


def example_inputs(*, flowing: bool = False) -> tuple[Any, ...]:
    """Synthetic 30-mode C6H6 formula card; no benzene parameter prediction claim."""
    source = DataSourceProvenance(
        "authored-rrkm38", 0, "Explicit synthetic harmonic test model", "2026-10-24", "CC0-1.0"
    )
    well = RRKMHarmonicState("A", (1000.0,) * 30, -15000.0, "electronic", 1, source)
    first = RRKMHarmonicState("TS-P", (1000.0,) * 29, -2500.0, "electronic", 1, source)
    second = RRKMHarmonicState("TS-Q", (1000.0,) * 28 + (2000.0,), -2000.0, "electronic", 1, source)
    model = RRKMGrainSpec(
        "harmonic-rrkm-example",
        "C6H6",
        InternalModeBaseline(30.0, 0.0, 40.0, well.mode_ids),
        well,
        RRKMEnergyGrid(1000.0, 14),
        (EnergyTransferBath("He", 3000.0, 3000.0), EnergyTransferBath("Ar", 1000.0, 4000.0)),
        tuple(EnergyGrainBathGas(s, s, 2.5 * R, 0.0, 20.0) for s in ("He", "Ar")),
        (
            RRKMProductChannel("to-P", first, -500.0, 1, EnergyGrainState("P", -4000.0)),
            RRKMProductChannel("to-Q", second, -700.0, 2, EnergyGrainState("Q", -2000.0)),
        ),
    )
    network = rrkm_grain_network(model)
    nodes: tuple[GasReactorNodeSpec, ...] = (GasReactorNodeSpec("cell", 1.0),)
    amounts = {"cell": dict.fromkeys(network.species_ids, 0.0)}
    amounts["cell"].update({"A:grain:0": 0.0009, "A:grain:14": 0.0001, "He": 0.01, "Ar": 0.002})
    connections: tuple[GasFlowConnectionSpec, ...] = ()
    if flowing:
        nodes += (GasReactorNodeSpec("feed", 1.0, chemistry_enabled=False),)
        amounts["feed"] = dict.fromkeys(network.species_ids, 0.0)
        amounts["feed"]["He"] = 0.02
        connections = (GasFlowConnectionSpec("bath-feed", "feed", "cell", coefficient=0.0002),)
    reactor = GasReactorNetworkSpec(nodes, connections, 600.0, "cell")
    return model, reactor, amounts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flowing", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous attempts")
    runtime = ChemWorldRuntime(composition=rrkm_grain_world(*example_inputs(flowing=args.flowing)))
    try:
        runtime.reset(seed=38)
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
                    "products_mol": {s: runtime._state.species_amounts[s] for s in ("P", "Q")},
                }
            ),
            flush=True,
        )
        if outcome.info["transaction_status"] != "committed":
            raise RuntimeError(
                {k: outcome.info.get(k) for k in ("error_message", "outcome_diagnostics")}
            )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
