"""Grow a synthetic finite-library gas mechanism, then use the original world.

The library is authored numerical demonstration data, not a real-material fit or
an RMG database. Four parameter entries do not prescribe enumeration results:
the original molecular graph recipes must discover each reaction first.
"""

from __future__ import annotations

import json
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    GrowthCondition,
    MechanismGrowthLibrary,
    ReactionLibraryEntry,
)
from chemworld.physchem.molecular_graph import GraphAtom, GraphBond, MolecularGraph
from chemworld.physchem.reaction_family import ReactionFamily, elementary_reaction_families
from chemworld.physchem.reaction_generation import KineticsBinding, SpeciesDataBinding
from chemworld.physchem.reaction_network_specs import RateLawSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import rate_coefficient_unit


def example_growth_inputs(
    *,
    sample_spacing_s: float = 0.25,
) -> tuple[MechanismGrowthLibrary, tuple[ReactionFamily, ...], tuple[GrowthCondition, ...]]:
    source = DataSourceProvenance(
        "synthetic-growth38",
        0,
        "Authored finite graph/parameter demonstration",
        version="1",
        license="CC0-1.0",
    )

    def graph(
        name: str,
        carbons: list[tuple[int, int]],
        bonds: list[tuple[int, int, int]],
        *,
        hydrogen: bool = False,
    ) -> MolecularGraph:
        atoms, edges = [], []
        for i, (hs, radicals) in enumerate(carbons):
            atoms.append(GraphAtom(str(i), "H" if hydrogen else "C", 0, radicals, 0))
            for j in range(hs):
                key = f"{i}H{j}"
                atoms.append(GraphAtom(key, "H", 0, 0, 0))
                edges.append(GraphBond(str(i), key, 1))
        edges.extend(GraphBond(str(a), str(b), n) for a, b, n in bonds)
        return MolecularGraph(
            name, tuple(atoms), tuple(edges), 1 + sum(a.radical_electrons for a in atoms), source
        )

    graphs = {
        "ethyl": graph("ethyl", [(2, 1), (3, 0)], [(0, 1, 1)]),
        "ethene": graph("ethene", [(2, 0), (2, 0)], [(0, 1, 2)]),
        "H": graph("H", [(0, 1)], [], hydrogen=True),
        "H2": graph("H2", [(1, 0)], [], hydrogen=True),
        "ethane": graph("ethane", [(3, 0), (3, 0)], [(0, 1, 1)]),
        "butane": graph(
            "butane", [(3, 0), (2, 0), (2, 0), (3, 0)], [(0, 1, 1), (1, 2, 1), (2, 3, 1)]
        ),
    }
    species = tuple(
        SpeciesDataBinding(
            value,
            SpeciesSpec(name, value.formula, phase="gas"),
            {
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [200.0, 2000.0],
                    "data": [[4.0, 0.0, 0.0, 0.0, 0.0, 100.0 * len(value.atoms), 2.0]],
                },
                "composition": value.composition,
                "reference_pressure_Pa": 101325.0,
            },
            source,
        )
        for name, value in graphs.items()
    )
    rows = (
        ("beta_scission", {"ethyl": 1}, {"ethene": 1, "H": 1}, 0.4, 3000),
        ("radical_recombination", {"ethyl": 2}, {"butane": 1}, 0.04, 0),
        ("radical_recombination", {"ethyl": 1, "H": 1}, {"ethane": 1}, 8.0, 2000),
        ("radical_recombination", {"H": 2}, {"H2": 1}, 4.0, 0),
    )
    rates = tuple(
        ReactionLibraryEntry(
            family,
            reactants,
            products,
            KineticsBinding(
                f"declared-parameter-{i}",
                RateLawSpec(f"declared-rate-{i}", "arrhenius", {"A": a, "b": 0.0, "Ea": ea}),
                rate_coefficient_unit(sum(reactants.values())),
                "J/mol",
                "effective_reaction",
                1.0,
                source,
            ),
        )
        for i, (family, reactants, products, a, ea) in enumerate(rows)
    )
    count = round(2 / sample_spacing_s)
    if count < 1 or abs(count * sample_spacing_s - 2) > 1e-12:
        raise ValueError("example sample spacing must divide two seconds")
    conditions = tuple(
        GrowthCondition(
            f"T{t}-n{n}",
            {"ethyl": n},
            t,
            1.0,
            tuple(i * sample_spacing_s for i in range(count + 1)),
        )
        for t in (400.0, 800.0)
        for n in (0.1, 0.4)
    )
    return MechanismGrowthLibrary(species, rates), elementary_reaction_families()[1:], conditions


def run_example() -> dict[str, Any]:
    from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    library, families, conditions = example_growth_inputs()
    growth = grow_mechanism(library, families, ("ethyl",), conditions)
    if not growth.complete:
        raise RuntimeError(f"example growth stopped: {growth.status}: {growth.reasons}")
    condition = conditions[0]
    spec = GasReactorNetworkSpec(
        (GasReactorNodeSpec("reactor", 1.0),), (), condition.temperature_K, "reactor"
    )
    original = ChemWorldRuntime(
        composition=gas_reactor_network_world(
            spec,
            growth.network,
            {"reactor": dict(condition.initial_amounts_mol)},
        )
    )
    restored = None
    try:
        original.reset(seed=38)
        first = original.execute({"operation": "wait", "duration_s": 0.5})
        if first.info["transaction_status"] != "committed":
            raise RuntimeError(str(first.info))
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(original.snapshot().to_dict()))
        )
        for runtime in (original, restored):
            outcome = runtime.execute({"operation": "wait", "duration_s": 0.5})
            if outcome.info["transaction_status"] != "committed":
                raise RuntimeError(str(outcome.info))
        return {
            "growth": growth.to_dict(),
            "world_time_s": original._state.ledger.time_s,
            "world_amounts_mol": dict(original._state.species_amounts),
            "snapshot_replay_equal": original._state.to_dict() == restored._state.to_dict(),
        }
    finally:
        original.close()
        if restored is not None:
            restored.close()


if __name__ == "__main__":
    output = run_example()
    print(json.dumps({key: value for key, value in output.items() if key != "growth"}, indent=2))
