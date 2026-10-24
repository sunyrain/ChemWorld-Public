"""Explicit 5-hexenyl cyclization through original graph, growth and gas worlds.

Numerical data are authored software fixtures, not measured cyclization kinetics.
"""

from __future__ import annotations

import json
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.gas_network_energy_specs import GasNodeEnergySpec
from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
from chemworld.physchem.mechanism_growth_specs import MechanismGrowthLibrary, ReactionLibraryEntry
from chemworld.physchem.molecular_graph import GraphAtom, GraphBond, MolecularGraph
from chemworld.physchem.reaction_family import extended_radical_reaction_families
from chemworld.physchem.reaction_generation import (
    KineticsBinding,
    SpeciesDataBinding,
    bind_generated_network,
    generate_reactions,
)
from chemworld.physchem.reaction_network_specs import RateLawSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import rate_coefficient_unit
from chemworld.world.gas_reactor_network import gas_reactor_network_world

SOURCE = DataSourceProvenance(
    "authored-radical-topology-37C",
    0,
    "Authored topology and finite numerical fixtures; not measured molecular parameters",
    version="1",
    license="CC0-1.0",
)


def example_graphs() -> dict[str, MolecularGraph]:
    graphs = {}
    for name, radical in (("hexenyl", 1), ("cyclopentylmethyl", 6)):
        atoms, bonds = [], []
        for i, hydrogens in enumerate((2, 2, 2, 2, 1, 2), 1):
            atoms.append(GraphAtom(f"C{i}", "C", 0, int(i == radical), 0))
            for j in range(hydrogens):
                h = f"H{i}_{j}"
                atoms.append(GraphAtom(h, "H", 0, 0, 0))
                bonds.append(GraphBond(f"C{i}", h, 1))
        bonds.extend(
            GraphBond(f"C{i}", f"C{i + 1}", 2 if i == 5 and radical == 1 else 1)
            for i in range(1, 6)
        )
        if radical == 6:
            bonds.append(GraphBond("C1", "C5", 1))
        graphs[name] = MolecularGraph(name, tuple(atoms), tuple(bonds), 2, SOURCE)
    return graphs


def example_library() -> MechanismGrowthLibrary:
    graphs = example_graphs()
    species = []
    for name, h0 in (("hexenyl", 50000.0), ("cyclopentylmethyl", 45000.0)):
        species.append(
            SpeciesDataBinding(
                graphs[name],
                SpeciesSpec(name, graphs[name].formula, phase="gas"),
                {
                    "thermo": {
                        "model": "constant-cp",
                        "T0": 298.15,
                        "h0": h0,
                        "s0": 250.0,
                        "cp0": 110.0,
                        "T-min": 300.0,
                        "T-max": 1400.0,
                    },
                    "composition": graphs[name].composition,
                    "reference_pressure_Pa": 101325.0,
                    "units": {"energy": "J", "quantity": "mol"},
                },
                SOURCE,
            )
        )
    family = extended_radical_reaction_families()[-1]
    kinetics = KineticsBinding(
        "cyclization",
        RateLawSpec("cyclization", "arrhenius", {"A": 2.0, "b": 0.0, "Ea": 2000.0}),
        rate_coefficient_unit(1),
        "J/mol",
        "effective_reaction",
        1.0,
        SOURCE,
    )
    return MechanismGrowthLibrary(
        tuple(species),
        (
            ReactionLibraryEntry(
                family.family_id, {"hexenyl": 1}, {"cyclopentylmethyl": 1}, kinetics
            ),
        ),
    )


def example_network() -> Any:
    from dataclasses import replace

    library = example_library()
    family = extended_radical_reaction_families()[-1]
    generated = generate_reactions({"hexenyl": example_graphs()["hexenyl"]}, (family,))
    if len(generated.reactions) != 1:
        raise ValueError("authored cyclization requires one generated product channel")
    return bind_generated_network(
        generated,
        library.species,
        (replace(library.reactions[0].kinetics, reaction_id=generated.reactions[0].reaction_id),),
        network_id="authored-5-exo-cyclization",
        reference_temperature_K=500.0,
    )


def example_world(*, temperature_K: float = 500.0, isothermal: bool = True) -> Any:
    energy = (
        ()
        if isothermal
        else (
            GasNodeEnergySpec(
                "reactor",
                temperature_K,
                1.0,
                minimum_temperature_K=300.0,
                maximum_temperature_K=1400.0,
            ),
        )
    )
    spec = GasReactorNetworkSpec(
        (GasReactorNodeSpec("reactor", 1.0),), (), temperature_K, "reactor", energy_nodes=energy
    )
    return gas_reactor_network_world(spec, example_network(), {"reactor": {"hexenyl": 0.02}})


def run_example() -> dict[str, Any]:
    from chemworld.physchem.mechanism_growth import grow_mechanism
    from chemworld.physchem.mechanism_growth_specs import GrowthCondition
    from chemworld.runtime import ChemWorldRuntime

    family = extended_radical_reaction_families()[-1]
    condition = GrowthCondition(
        "cyclization-500K", {"hexenyl": 0.02}, 500.0, 1.0, (0.0, 0.25, 0.5, 1.0)
    )
    growth = grow_mechanism(example_library(), (family,), ("hexenyl",), (condition,))
    runtime = ChemWorldRuntime(composition=example_world())
    try:
        runtime.reset(seed=37)
        result = runtime.execute({"operation": "wait", "duration_s": 1.0})
        return {
            "growth": growth.to_dict(),
            "transaction": result.info["transaction_status"],
            "amounts_mol": runtime._state.species_amounts,
            "time_s": runtime._state.ledger.time_s,
        }
    finally:
        runtime.close()


if __name__ == "__main__":
    print(json.dumps(run_example(), indent=2))
