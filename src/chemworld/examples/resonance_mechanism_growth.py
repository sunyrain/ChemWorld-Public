"""Finite resonance-aware growth with explicit synthetic parameters and real worlds.

The original recombination recipe is restricted by declared H/carbon-radical
patterns. Both butenyl resonance forms react; the two constitutional butenes
remain distinct. These CC0 numerical declarations are not experimental kinetics.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    GrowthCondition,
    MechanismGrowthLibrary,
    ReactionLibraryEntry,
)
from chemworld.physchem.molecular_graph import (
    AtomPattern,
    GraphAtom,
    GraphBond,
    MolecularGraph,
    MolecularPattern,
)
from chemworld.physchem.reaction_family import ReactionFamily, elementary_reaction_families
from chemworld.physchem.reaction_generation import (
    GenerationPolicy,
    KineticsBinding,
    SpeciesDataBinding,
)
from chemworld.physchem.reaction_network_specs import RateLawSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import rate_coefficient_unit

SOURCE = DataSourceProvenance(
    "authored-resonance-growth38E",
    0,
    "Explicit Lewis forms and synthetic finite two-channel parameters",
    version="1",
    license="CC0-1.0",
)


def _chain(
    name: str, hydrogens: tuple[int, ...], orders: tuple[int, ...], radical: int | None = None
) -> MolecularGraph:
    atoms, bonds = [], []
    for i, count in enumerate(hydrogens):
        key = f"{name}:C{i}"
        atoms.append(GraphAtom(key, "C", 0, int(i == radical), 0))
        for j in range(count):
            h = f"{name}:H{i}.{j}"
            atoms.append(GraphAtom(h, "H", 0, 0, 0))
            bonds.append(GraphBond(key, h, 1))
    bonds.extend(
        GraphBond(f"{name}:C{i}", f"{name}:C{i + 1}", order) for i, order in enumerate(orders)
    )
    return MolecularGraph(name, tuple(atoms), tuple(bonds), 1 if radical is None else 2, SOURCE)


def example_resonance_inputs(
    *, representative: int = 0, temperature_K: float = 500.0, per_path_factor: float = 1.0
) -> tuple[MechanismGrowthLibrary, tuple[ReactionFamily, ...], tuple[GrowthCondition, ...]]:
    """Use either explicitly authored resonance representative of the same R ID."""
    if representative not in (0, 1):
        raise ValueError("example representative is zero or one")
    graphs = {
        "H": MolecularGraph("atomic-H", (GraphAtom("free-H", "H", 0, 1, 0),), (), 2, SOURCE),
        "R": _chain(
            f"radical-representative-{representative}",
            (3, 1, 1, 2),
            (1, 1, 2) if representative == 0 else (1, 2, 1),
            1 if representative == 0 else 3,
        ),
        "butene1": _chain("bound-butene-1", (3, 2, 1, 2), (1, 1, 2)),
        "butene2": _chain("bound-butene-2", (3, 1, 1, 3), (1, 2, 1)),
    }
    original = elementary_reaction_families()[1]
    family = replace(
        original,
        family_id="declared_H_carbon_recombination",
        reactant_patterns=(
            MolecularPattern((AtomPattern("*1", elements=("H",), radicals=(1,), charges=(0,)),)),
            MolecularPattern((AtomPattern("*2", elements=("C",), radicals=(1,), charges=(0,)),)),
        ),
        unordered_reactants=False,
        source=SOURCE,
    )
    species = tuple(
        SpeciesDataBinding(
            graph,
            SpeciesSpec(name, graph.formula, phase="gas"),
            {
                "thermo": {
                    "model": "NASA7",
                    "temperature-ranges": [200.0, 2000.0],
                    "data": [
                        [3.0 * len(graph.atoms), 0.0, 0.0, 0.0, 0.0, 100.0 * len(graph.atoms), 2.0]
                    ],
                },
                "composition": graph.composition,
                "reference_pressure_Pa": 101325.0,
            },
            SOURCE,
        )
        for name, graph in graphs.items()
    )
    rates = tuple(
        ReactionLibraryEntry(
            family.family_id,
            {"H": 1, "R": 1},
            {name: 1},
            KineticsBinding(
                "declared-" + name,
                RateLawSpec("rate-" + name, "arrhenius", {"A": rate, "b": 0.0, "Ea": 0.0}),
                rate_coefficient_unit(2),
                "J/mol",
                "effective_reaction" if per_path_factor == 1 else "per_declared_path",
                per_path_factor,
                SOURCE,
            ),
        )
        for name, rate in (("butene1", 2.0), ("butene2", 3.0))
    )
    conditions = (
        GrowthCondition(
            "finite-H-plus-R", {"H": 0.2, "R": 0.1}, temperature_K, 1.0, (0.0, 0.2, 0.5, 1.0, 2.0)
        ),
    )
    return MechanismGrowthLibrary(species, rates), (family,), conditions


def run_example(*, representative: int = 0) -> dict[str, Any]:
    from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    library, families, conditions = example_resonance_inputs(representative=representative)
    growth = grow_mechanism(
        library,
        families,
        ("H", "R"),
        conditions,
        policy=CoreEdgeGrowthPolicy(generation=GenerationPolicy(allylic_resonance=True)),
    )
    if not growth.complete:
        raise RuntimeError(f"resonance growth stopped: {growth.status}: {growth.reasons}")
    condition = conditions[0]
    world = gas_reactor_network_world(
        GasReactorNetworkSpec(
            (GasReactorNodeSpec("reactor", 1.0),), (), condition.temperature_K, "reactor"
        ),
        growth.network,
        {"reactor": dict(condition.initial_amounts_mol)},
    )
    original, resumed = ChemWorldRuntime(composition=world), None
    try:
        original.reset(seed=38)
        first = original.execute({"operation": "wait", "duration_s": 0.5})
        if first.info["transaction_status"] != "committed":
            raise RuntimeError(str(first.info))
        snapshot = json.loads(json.dumps(original.snapshot().to_dict(), allow_nan=False))
        resumed = ChemWorldRuntime.from_snapshot(snapshot)
        for runtime in (original, resumed):
            result = runtime.execute({"operation": "wait", "duration_s": 0.5})
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(str(result.info))
        return {
            "growth": growth.to_dict(),
            "world_time_s": original._state.ledger.time_s,
            "world_amounts_mol": dict(original._state.species_amounts),
            "snapshot_replay_equal": original.snapshot().to_dict() == resumed.snapshot().to_dict(),
        }
    finally:
        original.close()
        if resumed is not None:
            resumed.close()


if __name__ == "__main__":
    print(json.dumps({k: v for k, v in run_example().items() if k != "growth"}, indent=2))
