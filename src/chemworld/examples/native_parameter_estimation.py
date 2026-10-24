"""Authored CC0 graph/group/rule demonstration; these are not material parameters."""

from __future__ import annotations

import json
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.kinetics_rate_rules import (
    DeclaredArrhenius,
    KineticEstimationLibrary,
    KineticRoleTree,
    KineticRuleEntry,
    KineticTrainingEntry,
)
from chemworld.physchem.mechanism_parameter_estimation import prepare_parameterized_mechanism
from chemworld.physchem.molecular_graph import (
    AtomPattern,
    BondPattern,
    GraphAtom,
    GraphBond,
    MolecularGraph,
    MolecularPattern,
)
from chemworld.physchem.reaction_family import elementary_reaction_families
from chemworld.physchem.reaction_generation import SpeciesDataBinding
from chemworld.physchem.reaction_network_specs import SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import rate_coefficient_unit
from chemworld.physchem.thermo_group_additivity import (
    GroupAdditivityLibrary,
    GroupThermoData,
    MolecularThermoContext,
    RootedGroupNode,
)

SOURCE = DataSourceProvenance(
    "authored-functional-37B",
    0,
    "CC0 authored group/rule values for software qualification; not measured material data",
    version="1",
    license="CC0-1.0",
)
GRID = (298.15, 400.0, 600.0, 1000.0, 1500.0)


def example_graphs() -> dict[str, MolecularGraph]:
    def graph(
        name: str,
        rows: list[tuple[int, int]],
        edges: list[tuple[int, int]],
        *,
        hydrogen: bool = False,
    ) -> MolecularGraph:
        atoms, bonds = [], []
        for i, (hs, radicals) in enumerate(rows):
            atoms.append(GraphAtom(str(i), "H" if hydrogen else "C", 0, radicals, 0))
            for j in range(hs):
                key = f"{i}H{j}"
                atoms.append(GraphAtom(key, "H", 0, 0, 0))
                bonds.append(GraphBond(str(i), key, 1))
        bonds.extend(GraphBond(str(a), str(b), 1) for a, b in edges)
        return MolecularGraph(
            name, tuple(atoms), tuple(bonds), 1 + sum(a.radical_electrons for a in atoms), SOURCE
        )

    return {
        "H": graph("H", [(0, 1)], [], hydrogen=True),
        "H2": graph("H2", [(1, 0)], [], hydrogen=True),
        "methane": graph("methane", [(4, 0)], []),
        "methyl": graph("methyl", [(3, 1)], []),
        "ethane": graph("ethane", [(3, 0), (3, 0)], [(0, 1)]),
        "propane": graph("propane", [(3, 0), (2, 0), (3, 0)], [(0, 1), (1, 2)]),
    }


def example_parameter_inputs() -> tuple[
    dict, tuple, GroupAdditivityLibrary, KineticEstimationLibrary
]:
    graphs = example_graphs()

    def data(h: float, s: float, cp: tuple[float, ...]) -> GroupThermoData:
        return GroupThermoData(GRID, cp, h, s, SOURCE)

    def group(name: str, carbons: int, value: GroupThermoData) -> RootedGroupNode:
        atoms = [AtomPattern("center", elements=("C",), charges=(0,), radicals=(0,))]
        for i in range(4):
            atoms.append(
                AtomPattern(
                    str(i), elements=("C" if i < carbons else "H",), charges=(0,), radicals=(0,)
                )
            )
        pattern = MolecularPattern(
            tuple(atoms), tuple(BondPattern("center", str(i), (1,)) for i in range(4))
        )
        return RootedGroupNode(name, "carbon", pattern, "center", value)

    groups = (
        RootedGroupNode(
            "carbon",
            None,
            MolecularPattern(
                (AtomPattern("center", elements=("C",), charges=(0,), radicals=(0,)),)
            ),
            "center",
        ),
        group("methane-group", 0, data(-75000, 210, (35, 40, 50, 65, 75))),
        group("terminal-group", 1, data(-40000, 120, (25, 30, 40, 50, 60))),
        group("secondary-group", 2, data(-20000, 50, (20, 25, 30, 40, 50))),
    )
    hbi = (
        RootedGroupNode(
            "carbon-HBI",
            None,
            MolecularPattern(
                (AtomPattern("center", elements=("C",), charges=(0,), radicals=(1,)),)
            ),
            "center",
            data(200000, -5, (-2, -2, -2, -2, -2)),
        ),
    )
    zero = data(0, 0, (0, 0, 0, 0, 0))
    contexts = tuple(
        MolecularThermoContext(graphs[name], sigma, 1, zero, SOURCE, True)
        for name, sigma in (("methane", 12), ("methyl", 6), ("ethane", 18), ("propane", 18))
    )
    elementary = tuple(
        SpeciesDataBinding(
            graphs[name],
            SpeciesSpec(name, graphs[name].formula, phase="gas"),
            {
                "thermo": {
                    "model": "constant-cp",
                    "T0": 298.15,
                    "h0": h,
                    "s0": s,
                    "cp0": cp,
                    "T-min": GRID[0],
                    "T-max": GRID[-1],
                },
                "composition": graphs[name].composition,
                "reference_pressure_Pa": 101325,
                "units": {"energy": "J", "quantity": "mol"},
            },
            SOURCE,
        )
        for name, h, s, cp in (("H", 200000, 110, 20.8), ("H2", 0, 130, 29.1))
    )
    thermo = GroupAdditivityLibrary(
        groups, hbi, contexts, elementary, "authored-common-Hf-standard-S", 101325, SOURCE
    )
    nodes = tuple(
        RootedGroupNode(
            name,
            parent,
            MolecularPattern(
                (AtomPattern("center", elements=elements, charges=(0,), radicals=(1,)),)
            ),
            "center",
        )
        for name, parent, elements in (
            ("R", None, ("C", "H")),
            ("H", "R", ("H",)),
            ("C", "R", ("C",)),
        )
    )
    family = elementary_reaction_families()[1]

    def parameters(A: float, b: float = 0, ea: float = 0) -> DeclaredArrhenius:
        return DeclaredArrhenius(
            A,
            b,
            ea,
            rate_coefficient_unit(2, temperature_exponent=b),
            (300, 1500),
            "effective_reaction",
            1,
        )

    roles = tuple(KineticRoleTree(label, nodes) for label in ("*1", "*2"))
    rules = (
        KineticRuleEntry(
            "fallback-RR", family.family_id, ("R", "R"), parameters(0.5, 0.5, 1000), SOURCE
        ),
        KineticRuleEntry("rule-CC", family.family_id, ("C", "C"), parameters(1), SOURCE),
    )
    kinetic = KineticEstimationLibrary(
        {family.family_id: roles},
        rules,
        (
            KineticTrainingEntry(
                "training-HC",
                family.family_id,
                (graphs["H"], graphs["methyl"]),
                (graphs["methane"],),
                parameters(2),
                SOURCE,
            ),
        ),
        (
            KineticTrainingEntry(
                "lib-HH",
                family.family_id,
                (graphs["H"], graphs["H"]),
                (graphs["H2"],),
                parameters(1),
                SOURCE,
            ),
        ),
    )
    return {"H": graphs["H"], "methyl": graphs["methyl"]}, (family,), thermo, kinetic


def prepare_example() -> Any:
    seeds, families, thermo, kinetic = example_parameter_inputs()
    return prepare_parameterized_mechanism(
        seeds,
        families,
        thermo,
        kinetic,
        network_id="authored-automatic-radical-recombination",
        reference_temperature_K=500,
    )


def run_example() -> dict[str, Any]:
    from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
    from chemworld.physchem.mechanism_growth import grow_mechanism
    from chemworld.physchem.mechanism_growth_specs import GrowthCondition
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.gas_reactor_network import gas_reactor_network_world

    prepared = prepare_example()
    if not prepared.complete:
        raise RuntimeError(str(prepared.reasons))
    _, families, _, _ = example_parameter_inputs()
    condition = GrowthCondition(
        "authored-500K", {"H": 0.1, "methyl": 0.1}, 500, 1, (0, 0.25, 0.5, 0.75, 1)
    )
    growth = grow_mechanism(prepared.growth_library, families, ("H", "methyl"), (condition,))
    if not growth.complete:
        raise RuntimeError(str(growth.reasons))
    spec = GasReactorNetworkSpec((GasReactorNodeSpec("reactor", 1),), (), 500, "reactor")
    runtime = ChemWorldRuntime(
        composition=gas_reactor_network_world(
            spec, growth.network, {"reactor": dict(condition.initial_amounts_mol)}
        )
    )
    restored = None
    try:
        runtime.reset(seed=37)
        step = runtime.execute({"operation": "wait", "duration_s": 0.5})
        if step.info["transaction_status"] != "committed":
            raise RuntimeError(str(step.info))
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for target in (runtime, restored):
            step = target.execute({"operation": "wait", "duration_s": 0.5})
            if step.info["transaction_status"] != "committed":
                raise RuntimeError(str(step.info))
        return {
            "preparation": prepared.to_dict(),
            "growth": growth.to_dict(),
            "time_s": runtime._state.ledger.time_s,
            "amounts_mol": dict(runtime._state.species_amounts),
            "snapshot_replay_equal": runtime._state.to_dict() == restored._state.to_dict(),
        }
    finally:
        runtime.close()
        if restored:
            restored.close()


if __name__ == "__main__":
    result = run_example()
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"preparation", "growth"}}, indent=2
        )
    )
