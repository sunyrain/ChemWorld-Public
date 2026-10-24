"""Fixed 40-unit / 213-comparison automatic-parameter development block."""

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from rmg37b_reference import COUNTS, kinetic, thermo, trajectory

from chemworld.examples.native_parameter_estimation import (
    example_graphs,
    example_parameter_inputs,
    prepare_example,
    run_example,
)
from chemworld.physchem.kinetics_rate_rules import estimate_reaction_kinetics
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    GrowthCondition,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
)
from chemworld.physchem.mechanism_parameter_estimation import prepare_parameterized_mechanism
from chemworld.physchem.molecular_graph import GraphAtom, GraphBond
from chemworld.physchem.reaction_generation import GenerationPolicy
from chemworld.physchem.reaction_network import BatchSolverOptions
from chemworld.physchem.reaction_rate_laws import arrhenius_k
from chemworld.physchem.thermo_group_additivity import estimate_group_thermochemistry
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml


def compare(name, actual, expected, atol=2e-6, rtol=2e-9):
    return {
        "quantity": name,
        "actual": float(actual),
        "expected": float(expected),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "absolute_error": abs(float(actual) - float(expected)),
        "passed": math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol),
    }


def exact(name, condition):
    return compare(name, int(condition), 1, 0, 0)


def estimate_model(name, library=None, graph=None):
    if library is None:
        library = example_parameter_inputs()[2]
    estimated = estimate_group_thermochemistry(graph or example_graphs()[name], name, library)
    if not estimated.success:
        raise ValueError(estimated.failure_reason)
    return estimated, species_thermo_from_cantera_yaml(
        species_id=name, **estimated.binding.thermo_declaration
    )


def thermo_case(name, T):
    estimated, model = estimate_model(name)
    state = model.evaluate(T)
    expected = thermo(name, T)
    counts = [
        estimated.trace["group_counts"].get(key, 0)
        for key in ("methane-group", "terminal-group", "secondary-group")
    ]
    counts.append(estimated.trace["hbi_counts"].get("carbon-HBI", 0))
    actual = {
        "Cp": state.cp_J_mol_K,
        "H": state.enthalpy_J_mol,
        "S": state.entropy_J_mol_K,
        "G": state.gibbs_J_mol,
    }
    return [
        *(
            compare("count-" + str(i), value, COUNTS[name][i], 0, 0)
            for i, value in enumerate(counts)
        ),
        *(
            compare(key, value, expected[key], 2e-8 if key in {"S", "Cp"} else 2e-6)
            for key, value in actual.items()
        ),
    ]


def selected_rate(tier, prepared=None, library=None):
    prepared = prepared or prepare_example()
    library = library or example_parameter_inputs()[3]
    required = {
        "library": ["H", "H"],
        "training": ["CH3", "H"],
        "rule": ["CH3", "CH3"],
        "ancestor": ["CH3", "CH3"],
    }[tier]
    candidate = next(
        r
        for r in prepared.generation.reactions
        if sorted(
            prepared.generation.species[key].formula
            for key, number in r.reactants.items()
            for _ in range(number)
        )
        == required
    )
    if tier == "ancestor":
        library = replace(library, rules=tuple(r for r in library.rules if r.entry_id != "rule-CC"))
    selected = estimate_reaction_kinetics(candidate, prepared.generation.species, library)
    if not selected.success:
        raise ValueError(selected.failure_reason)
    return selected


def kinetic_case(tier, T):
    selected = selected_rate(tier)
    code = {"library": 0, "training": 1, "rule": 2, "ancestor": 3}
    entry = {
        "library": "lib-HH",
        "training": "training-HC",
        "rule": "rule-CC",
        "ancestor": "fallback-RR",
    }
    return [
        compare(
            "k",
            arrhenius_k(selected.binding.rate_law.parameters, T),
            kinetic(tier, T),
            2e-12,
            2e-10,
        ),
        compare("tier", code[selected.trace["tier"]], code[tier], 0, 0),
        exact("entry", selected.trace["entry_id"] == entry[tier]),
        compare(
            "distance", selected.trace["ancestor_distance"], 2 if tier == "ancestor" else 0, 0, 0
        ),
        compare("degeneracy", selected.binding.degeneracy_factor, 1, 0, 0),
    ]


def project(amounts, prepared):
    return {
        graph.formula: amounts.get(key, 0) for key, graph in prepared.generation.species.items()
    }


def balances(values):
    return (
        values["CH3"] + values["CH4"] + 2 * values["C2H6"],
        values["H"] + 2 * values["H2"] + 3 * values["CH3"] + 4 * values["CH4"] + 6 * values["C2H6"],
    )


def integration_case(index):
    prepared = prepare_example()
    if index == 0:
        return [
            exact("complete", prepared.complete),
            compare("species", len(prepared.network.species), 5, 0, 0),
            compare("reactions", len(prepared.network.reactions), 3, 0, 0),
            compare("thermo bindings", len(prepared.growth_library.species), 5, 0, 0),
            compare("rate bindings", len(prepared.growth_library.reactions), 3, 0, 0),
            exact("element balance", prepared.network.check_element_balance()),
        ]
    if index == 1:
        result = prepared.network.integrate_batch(
            {"H": 0.1, "methyl": 0.1},
            volume_L=1,
            temperature_K=500,
            duration_s=1,
            evaluation_times_s=(0, 0.25, 0.5, 0.75, 1),
            solver_options=BatchSolverOptions(method="DOP853", rtol=1e-11, atol_mol=1e-13),
        )
        rows = []
        for i, t in enumerate(result.times_s):
            actual = project(
                {key: result.amounts_mol[j][i] for j, key in enumerate(result.species_ids)},
                prepared,
            )
            rows.extend(
                compare(f"{key}@{t}", value, trajectory(t)[key], 2e-8, 2e-7)
                for key, value in actual.items()
            )
        c, h = balances(actual)
        return [
            *rows,
            compare("C balance", c, 0.1, 2e-9, 0),
            compare("H balance", h, 0.4, 2e-9, 0),
            exact("provenance", bool(prepared.network.metadata["parameter_estimation"])),
        ]
    if index == 2:
        families = example_parameter_inputs()[1]
        condition = GrowthCondition(
            "fixed", {"H": 0.1, "methyl": 0.1}, 500, 1, (0, 0.25, 0.5, 0.75, 1)
        )
        before = json.dumps(condition.to_dict(), sort_keys=True)
        args = (prepared.growth_library, families, ("H", "methyl"), (condition,))
        full = grow_mechanism(*args)
        stopped = grow_mechanism(*args, resources=GrowthResourceLimits(maximum_rounds=1))
        checkpoint = MechanismGrowthCheckpoint.from_dict(
            json.loads(json.dumps(stopped.checkpoint.to_dict()))
        )
        resumed = grow_mechanism(*args, checkpoint=checkpoint)
        return [
            exact("complete", full.complete and resumed.complete),
            compare("core species", len(full.network.species), 5, 0, 0),
            compare("reactions", len(full.network.reactions), 3, 0, 0),
            exact("resume", resumed.network.to_dict() == full.network.to_dict()),
            exact(
                "actual solve",
                full.checkpoint.state["condition_solves"] > 0 and full.checkpoint.state["nfev"] > 0,
            ),
            exact("initial inventory", before == json.dumps(condition.to_dict(), sort_keys=True)),
        ]
    world = run_example()
    values = project(world["amounts_mol"], prepared)
    c, h = balances(values)
    return [
        *(compare(key, value, trajectory(1)[key], 2e-8, 2e-7) for key, value in values.items()),
        compare("elapsed", world["time_s"], 1, 1e-12, 0),
        exact("exact replay", world["snapshot_replay_equal"]),
        compare("C balance", c, 0.1, 2e-9, 0),
        compare("H balance", h, 0.4, 2e-9, 0),
    ]


def invariant_case(index):
    library = example_parameter_inputs()[2]
    graph = example_graphs()["propane"]
    if index == 0:
        mapping = {a.atom_id: "rename-" + str(i) for i, a in enumerate(reversed(graph.atoms))}
        graph = replace(
            graph,
            atoms=tuple(replace(a, atom_id=mapping[a.atom_id]) for a in graph.atoms),
            bonds=tuple(
                replace(b, first=mapping[b.first], second=mapping[b.second]) for b in graph.bonds
            ),
        )
    elif index == 1:
        graph = replace(
            graph, atoms=tuple(reversed(graph.atoms)), bonds=tuple(reversed(graph.bonds))
        )
    elif index == 2:
        library = replace(
            library,
            groups=tuple(reversed(library.groups)),
            contexts=tuple(reversed(library.contexts)),
        )
    else:
        prepared = prepare_example()
        rates = example_parameter_inputs()[3]
        rates = replace(
            rates,
            rules=tuple(reversed(rates.rules)),
            training=tuple(reversed(rates.training)),
            library=tuple(reversed(rates.library)),
        )
        originals, others = [], []
        for tier in ("library", "training", "rule"):
            originals.append(selected_rate(tier, prepared))
            others.append(selected_rate(tier, prepared, rates))
        return [
            *(
                compare(
                    "k-" + str(i),
                    arrhenius_k(b.binding.rate_law.parameters, 500),
                    arrhenius_k(a.binding.rate_law.parameters, 500),
                    2e-12,
                    2e-10,
                )
                for i, (a, b) in enumerate(zip(originals, others, strict=True))
            ),
            exact(
                "tier and entry trace", [r.trace for r in originals] == [r.trace for r in others]
            ),
        ]
    estimated, model = estimate_model("propane", library, graph)
    actual, expected = model.evaluate(600), thermo("propane", 600)
    return [
        compare("H", actual.enthalpy_J_mol, expected["H"]),
        compare("S", actual.entropy_J_mol_K, expected["S"], 2e-8),
        compare("Cp", actual.cp_J_mol_K, expected["Cp"], 2e-8),
        exact(
            "counts", estimated.trace["group_counts"] == {"terminal-group": 2, "secondary-group": 1}
        ),
    ]


def rejection_case(index):
    seeds, families, library, rates = example_parameter_inputs()
    graph = example_graphs()["methyl"]
    if index == 0:
        library = replace(library, groups=(library.groups[0],))
    elif index == 1:
        library = replace(
            library, groups=(*library.groups, replace(library.groups[1], node_id="duplicate"))
        )
    elif index == 2:
        library = replace(library, contexts=())
    elif index == 3:
        library = replace(library, hbi_groups=())
    elif index == 4:
        graph = example_graphs()["propane"]
        removed = {"0H2", "2H2"}
        graph = replace(
            graph,
            atoms=tuple(a for a in graph.atoms if a.atom_id not in removed),
            bonds=(
                *tuple(b for b in graph.bonds if not {b.first, b.second} & removed),
                GraphBond("0", "2", 1),
            ),
        )
    elif index == 5:
        graph = replace(
            graph, multiplicity=1, atoms=(GraphAtom("0", "C", 1, 0, 0), *graph.atoms[1:])
        )
    elif index == 6:
        graph = replace(graph, atoms=(replace(graph.atoms[0], isotope=13), *graph.atoms[1:]))
    elif index == 7:
        graph = example_graphs()["ethane"]
        removed = {"0H2", "1H2"}
        graph = replace(
            graph,
            multiplicity=3,
            atoms=tuple(
                replace(a, radical_electrons=1) if a.element == "C" else a
                for a in graph.atoms
                if a.atom_id not in removed
            ),
            bonds=tuple(b for b in graph.bonds if not {b.first, b.second} & removed),
        )
    elif index in (8, 9, 10):
        if index == 8:
            rates = replace(
                rates, rules=(*rates.rules, replace(rates.rules[1], entry_id="conflict"))
            )
        if index == 9:
            rates = replace(rates, rules=(), training=(), library=())
        kwargs = {"generation_policy": GenerationPolicy(maximum_reactions=1)} if index == 10 else {}
        result = prepare_parameterized_mechanism(
            seeds,
            families,
            library,
            rates,
            network_id="reject",
            reference_temperature_K=500,
            **kwargs,
        )
        return [
            {
                **exact("incomplete with reason", not result.complete and bool(result.reasons)),
                "diagnostic": list(result.reasons),
            }
        ]
    else:
        try:
            prepare_example().network.reaction_rates({}, volume_L=1, temperature_K=200)
        except ValueError as exc:
            return [
                {
                    **exact("explicit exported rejection", "kinetic validity" in str(exc)),
                    "diagnostic": str(exc),
                }
            ]
        return [exact("explicit exported rejection", False)]
    result = estimate_group_thermochemistry(graph, "rejected", library)
    return [
        {
            **exact("unestimated with reason", not result.success and bool(result.failure_reason)),
            "diagnostic": result.failure_reason,
        }
    ]


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    definitions = [
        *(
            {"kind": "thermo", "args": [name, T], "comparisons": 8}
            for name in ("methane", "ethane", "propane", "methyl")
            for T in (300, 600, 1000)
        ),
        *(
            {"kind": "kinetic", "args": [tier, T], "comparisons": 5}
            for tier in ("library", "training", "rule", "ancestor")
            for T in (500, 1000)
        ),
        *(
            {"kind": "integration", "args": [i], "comparisons": count}
            for i, count in enumerate((6, 28, 6, 9))
        ),
        *({"kind": "invariant", "args": [i], "comparisons": 4} for i in range(4)),
        *({"kind": "rejection", "args": [i], "comparisons": 1} for i in range(12)),
    ]
    (output / "units.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    functions = {
        "thermo": thermo_case,
        "kinetic": kinetic_case,
        "integration": integration_case,
        "invariant": invariant_case,
        "rejection": rejection_case,
    }
    units, comparisons = [], []
    started = time.monotonic()
    print("start units=0/40 planned_comparisons=213", flush=True)
    for i, definition in enumerate(definitions, 1):
        record = {"unit": i, **definition}
        try:
            rows = functions[definition["kind"]](*definition["args"])
            if len(rows) != definition["comparisons"]:
                raise RuntimeError("comparison denominator changed")
            comparisons.extend({"unit": i, **row} for row in rows)
            record.update(passed=all(row["passed"] for row in rows), executed=len(rows))
        except Exception as exc:
            record.update(passed=False, executed=0, exception=f"{type(exc).__name__}: {exc}")
        units.append(record)
        elapsed = time.monotonic() - started
        print(
            f"progress units={i}/40 rate={i / max(elapsed, 1e-9):.2f}/s "
            f"eta={(40 - i) * elapsed / i:.1f}s status={'PASS' if record['passed'] else 'FAIL'}",
            flush=True,
        )
    failed_units = [row for row in units if not row["passed"]]
    failed_comparisons = [row for row in comparisons if not row["passed"]]
    summary = {
        "mode": "development",
        "planned_units": 40,
        "passed_units": 40 - len(failed_units),
        "failed_units": len(failed_units),
        "planned_comparisons": 213,
        "executed_comparisons": len(comparisons),
        "passed_comparisons": len(comparisons) - len(failed_comparisons),
        "failed_comparisons": len(failed_comparisons),
        "unexecuted_comparisons": 213 - len(comparisons),
        "unit_failures": failed_units,
        "comparison_failures": failed_comparisons,
        "elapsed_seconds": time.monotonic() - started,
        "parameter_scope": "CC0 authored functional values; no real RMG database qualification",
    }
    for name, rows in (("unit_results", units), ("comparisons", comparisons)):
        (output / (name + ".jsonl")).write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8"
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
