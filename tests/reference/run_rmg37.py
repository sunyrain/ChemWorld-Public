"""Fixed small-graph oracle and native ODE acceptance; no upstream RMG dependency."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from itertools import permutations, product
from math import exp, isclose
from pathlib import Path

from rmg37_fixtures import bindings, molecule, structures

from chemworld.physchem.molecular_graph import (
    GraphBond,
    MolecularGraph,
    enumerate_allylic_resonance,
    molecular_isomorphic,
)
from chemworld.physchem.reaction_family import elementary_reaction_families
from chemworld.physchem.reaction_generation import bind_generated_network, generate_reactions
from chemworld.physchem.reaction_network import BatchSolverOptions, ReactionNetworkSpec


def nuclear_electronic_key(atom):
    return atom.element, atom.isotope, atom.charge, atom.radical_electrons, atom.lone_pairs


def brute_isomorphic(left, right):
    """Independent exact enumeration, deliberately without a matching library."""
    if left.multiplicity != right.multiplicity:
        return False
    groups_left, groups_right = defaultdict(list), defaultdict(list)
    for atom in left.atoms:
        groups_left[nuclear_electronic_key(atom)].append(atom.atom_id)
    for atom in right.atoms:
        groups_right[nuclear_electronic_key(atom)].append(atom.atom_id)
    if {key: len(ids) for key, ids in groups_left.items()} != {
        key: len(ids) for key, ids in groups_right.items()
    }:
        return False
    keys = list(groups_left)
    target = {(frozenset((b.first, b.second)), b.order) for b in right.bonds}
    for choices in product(*(permutations(groups_right[key]) for key in keys)):
        mapping = {
            old: new
            for key, selected in zip(keys, choices, strict=True)
            for old, new in zip(groups_left[key], selected, strict=True)
        }
        edges = {(frozenset((mapping[b.first], mapping[b.second])), b.order) for b in left.bonds}
        if edges == target:
            return True
    return False


def permuted(graph):
    mapping = {a.atom_id: f"renamed-{i}" for i, a in enumerate(reversed(graph.atoms))}
    return replace(
        graph,
        atoms=tuple(replace(a, atom_id=mapping[a.atom_id]) for a in reversed(graph.atoms)),
        bonds=tuple(
            GraphBond(mapping[b.first], mapping[b.second], b.order) for b in reversed(graph.bonds)
        ),
    )


class Record:
    def __init__(self, unit):
        self.unit, self.comparisons = unit, []

    def check(self, label, actual, expected, *, numeric=False):
        passed = (
            isclose(actual, expected, rel_tol=2e-7, abs_tol=2e-10)
            if numeric
            else actual == expected
        )
        self.comparisons.append(
            {"label": label, "actual": actual, "expected": expected, "passed": passed}
        )


def identity(record, left, right, expected):
    oracle = brute_isomorphic(left, right)
    record.check("independent_oracle", oracle, expected)
    record.check("native_vs_oracle", molecular_isomorphic(left, right), oracle)
    restored = MolecularGraph.from_dict(json.loads(json.dumps(right.to_dict())))
    record.check("codec_identity", brute_isomorphic(restored, right), True)


def family_case(record, output, family, seeds, expected, expected_paths):
    known = structures()
    report = generate_reactions(
        {key: known[key] for key in seeds}, elementary_reaction_families()[family : family + 1]
    )
    (output / f"{record.unit}-generation.json").write_text(
        json.dumps(report.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
    )
    record.check("complete", report.complete, True)
    record.check("rejected", list(report.rejected), [])
    actual = []
    for reaction in report.reactions:
        names = {}
        for key, count in reaction.products.items():
            matches = [
                name
                for name, graph in known.items()
                if brute_isomorphic(report.species[key], graph)
            ]
            record.check(f"{reaction.reaction_id}/{key}/unique_product_identity", len(matches), 1)
            names[matches[0] if len(matches) == 1 else "unknown"] = count
        actual.append(names)
        record.check(
            reaction.reaction_id + "/charge",
            sum(report.species[k].charge * n for k, n in reaction.reactants.items()),
            sum(report.species[k].charge * n for k, n in reaction.products.items()),
        )
        for index, path in enumerate(reaction.paths):
            rows = path["atom_mapping"]
            left = {(r["reactant_index"], r["reactant_atom"]) for r in rows}
            right = {(r["product_index"], r["product_atom"]) for r in rows}
            expected_count = sum(
                len(report.species[k].atoms) * n for k, n in reaction.reactants.items()
            )
            record.check(
                f"{reaction.reaction_id}/{index}/bijective_nuclei",
                [len(left), len(right), len(rows)],
                [expected_count] * 3,
            )
            identities = []
            for row in rows:
                a = next(
                    a
                    for a in report.species[row["reactant_species"]].atoms
                    if a.atom_id == row["reactant_atom"]
                )
                b = next(
                    a
                    for a in report.species[row["product_species"]].atoms
                    if a.atom_id == row["product_atom"]
                )
                identities.append((a.element, a.isotope) == (b.element, b.isotope))
            record.check(
                f"{reaction.reaction_id}/{index}/nuclear_identities", all(identities), True
            )

    def canonical(rows):
        return sorted(json.dumps(row, sort_keys=True) for row in rows)

    record.check("complete_product_set", canonical(actual), canonical(expected))
    record.check(
        "unique_edit_paths", sorted(len(r.paths) for r in report.reactions), expected_paths
    )


def resonance_case(record, truncated):
    original = structures()["allyl"]
    other = molecule(
        "allyl-shifted",
        [("C", 0, 0, 0, 3), ("C", 0, 0, 0, 1), ("C", 0, 0, 0, 1), ("C", 0, 1, 0, 2)],
        [(0, 1, 1), (1, 2, 2), (2, 3, 1)],
        multiplicity=2,
    )
    result = enumerate_allylic_resonance(original, maximum_forms=1 if truncated else 64)
    record.check("complete", result.complete, not truncated)
    record.check("form_count", len(result.forms), 1 if truncated else 2)
    for index, form in enumerate(result.forms):
        record.check(f"form_{index}", brute_isomorphic(form, (original, other)[index]), True)
        record.check(
            f"nuclei_{index}",
            sorted(a.atom_id for a in form.atoms),
            sorted(a.atom_id for a in original.atoms),
        )


def trajectory_case(record, output, family, seed):
    report = generate_reactions(
        {seed: structures()[seed]}, elementary_reaction_families()[family : family + 1]
    )
    species, rates = bindings(report)
    network = bind_generated_network(
        report, species, rates, network_id=record.unit, reference_temperature_K=500
    )
    payload = json.dumps(network.to_dict(), indent=2, allow_nan=False)
    (output / f"{record.unit}-network.json").write_text(payload, encoding="utf-8")
    network = ReactionNetworkSpec.from_dict(json.loads(payload))
    result = network.integrate_batch(
        {seed: 0.1},
        volume_L=1,
        temperature_K=500,
        duration_s=2,
        evaluation_times_s=[0, 0.5, 1, 2],
        solver_options=BatchSolverOptions(method="DOP853", rtol=1e-10, atol_mol=1e-13),
    )
    record.check("conservation", network.check_conservation(), True)
    record.check("time_points", list(result.times_s), [0, 0.5, 1, 2])
    for t, values in zip(result.times_s, zip(*result.amounts_mol, strict=True), strict=True):
        expected = 0.1 / (1 + 2 * 0.2 * 0.1 * t) if family == 1 else 0.1 * exp(-0.2 * t)
        actual = dict(zip(result.species_ids, values, strict=True))
        record.check(f"{t}/{seed}", actual[seed], expected, numeric=True)
        for key, count in report.reactions[0].products.items():
            record.check(
                f"{t}/{key}",
                actual[key],
                count * (0.1 - expected) / (2 if family == 1 else 1),
                numeric=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    graphs = structures()
    cases = []
    for key in ("H", "H2", "methane", "methyl", "ethane", "ethene", "ethanol", "ether"):
        cases.append(
            (
                f"positive-{key}",
                lambda r, key=key: identity(r, graphs[key], permuted(graphs[key]), True),
            )
        )
    isotope = replace(
        graphs["methane"],
        atoms=tuple(
            replace(a, isotope=13) if a.element == "C" else a for a in graphs["methane"].atoms
        ),
    )
    cation = molecule("cation", [("C", 1, 0, 0, 3)], [])
    oxygen = molecule("oxygen", [("O", 0, 2, 2, 0)], [], multiplicity=3)
    for name, left, right in (
        ("constitutional", graphs["ethanol"], graphs["ether"]),
        ("isotope", graphs["methane"], isotope),
        ("charge-radical", graphs["methyl"], cation),
        ("spin", oxygen, replace(oxygen, multiplicity=1)),
    ):
        cases.append((f"negative-{name}", lambda r, a=left, b=right: identity(r, a, b, False)))
    for name, family, seeds, products, paths in (
        ("abstraction", 0, ("methane", "H"), [{"methyl": 1, "H2": 1}], [4]),
        ("recombination", 1, ("methyl",), [{"ethane": 1}], [1]),
        ("ethyl-scission", 2, ("ethyl",), [{"ethene": 1, "H": 1}], [3]),
        (
            "propyl-scission",
            2,
            ("propyl",),
            [{"ethene": 1, "methyl": 1}, {"propene": 1, "H": 1}],
            [1, 2],
        ),
    ):
        cases.append(
            (
                f"family-{name}",
                lambda r, f=family, s=seeds, p=products, n=paths: family_case(
                    r, args.output, f, s, p, n
                ),
            )
        )
    cases.extend(
        [
            ("resonance-complete", lambda r: resonance_case(r, False)),
            ("resonance-truncated", lambda r: resonance_case(r, True)),
            ("trajectory-recombination", lambda r: trajectory_case(r, args.output, 1, "methyl")),
            ("trajectory-scission", lambda r: trajectory_case(r, args.output, 2, "ethyl")),
        ]
    )
    start, records = time.monotonic(), []
    for i, (name, execute) in enumerate(cases, 1):
        record, error = Record(name), None
        try:
            execute(record)
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        records.append(
            {
                "unit": name,
                "comparisons": record.comparisons,
                "error": error,
                "passed": error is None and all(c["passed"] for c in record.comparisons),
            }
        )
        elapsed = max(time.monotonic() - start, 1e-9)
        print(
            f"RMG37 {i}/{len(cases)} {name} passed={records[-1]['passed']} "
            f"throughput={i / elapsed:.1f} units/s "
            f"ETA={(len(cases) - i) * elapsed / i:.1f}s",
            flush=True,
        )
    comparisons = [c for r in records for c in r["comparisons"]]
    summary = {
        "status": "passed" if all(r["passed"] for r in records) else "failed",
        "evidence_mode": "development",
        "units_total": len(records),
        "units_passed": sum(r["passed"] for r in records),
        "comparisons_total": len(comparisons),
        "comparisons_passed": sum(c["passed"] for c in comparisons),
        "failures": [r for r in records if not r["passed"]],
        "reference": (
            "Independent brute-force graph permutations, authored product graphs and "
            "closed-form ODEs; not upstream RMG numerical equivalence"
        ),
        "trajectory_tolerances": {"relative": 2e-7, "absolute_mol": 2e-10},
        "elapsed_s": time.monotonic() - start,
    }
    (args.output / "records.json").write_text(
        json.dumps(records, indent=2, allow_nan=False), encoding="utf-8"
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
