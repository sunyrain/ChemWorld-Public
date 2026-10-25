"""Fixed 16-unit / 90-comparison 38E development block; retain every attempt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from chemworld.examples.resonance_mechanism_growth import example_resonance_inputs
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    CoreEdgeGrowthPolicy,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
    MechanismGrowthLibrary,
)
from chemworld.physchem.molecular_graph import (
    GraphAtom,
    GraphBond,
    MolecularGraph,
    enumerate_allylic_resonance,
)
from chemworld.physchem.molecular_identity import molecular_identity_mapping
from chemworld.physchem.reaction_generation import GenerationPolicy, generate_reactions
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from reference.resonance_growth38_cases import CASES, declared_forms, mapped_network, renamed
from reference.resonance_growth38_reference import (
    analytic_amounts,
    labelled_state,
    path_checks,
    signature,
)
from reference.rmg37_fixtures import SOURCE, structures

R = 8.31446261815324


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def policy(limit=64):
    return CoreEdgeGrowthPolicy(
        generation=GenerationPolicy(allylic_resonance=True, maximum_resonance_forms=limit)
    )


def growth(representative=0, factor=1.0, **kwargs):
    library, families, conditions = example_resonance_inputs(
        representative=representative, per_path_factor=factor
    )
    return grow_mechanism(library, families, ("H", "R"), conditions, policy=policy(), **kwargs)


def labelled_fingerprint(graph, mapping=None):
    atoms, bonds, spin = labelled_state(graph)
    mapping = mapping or {key: key for key in atoms}
    return repr(
        (
            sorted((mapping[k], v) for k, v in atoms.items()),
            sorted((tuple(sorted((mapping[a], mapping[b]))), v) for (a, b), v in bonds.items()),
            spin,
        )
    )


class Block:
    def __init__(self, output, original_source):
        self.output, self.original_source = output, original_source
        self.records, self.details, self.current = [], [], None

    def check(self, name, actual, expected=True, *, numeric=False, rtol=2e-7, atol=2e-10):
        actual, expected = plain(actual), plain(expected)
        passed = (
            bool(np.allclose(actual, expected, rtol=rtol, atol=atol))
            if numeric
            else actual == expected
        )
        self.current["comparisons"].append(
            {"name": name, "passed": passed, "actual": actual, "expected": expected}
        )

    def identity(self, name):
        expected = declared_forms(name)
        result = enumerate_allylic_resonance(expected[0])
        self.check("complete_labelled_closure_count", len(result.forms), len(expected))
        self.check(
            "independent_entire_labelled_states",
            sorted(labelled_fingerprint(g.to_dict()) for g in result.forms),
            sorted(labelled_fingerprint(g.to_dict()) for g in expected),
        )
        self.check("explicit_completeness", result.complete)
        valid = []
        for form in expected:
            target = renamed(expected[0])
            mapping = molecular_identity_mapping(form, target, allylic_resonance=True)
            valid.append(
                mapping is not None
                and any(
                    labelled_fingerprint(candidate.to_dict(), mapping)
                    == labelled_fingerprint(target.to_dict())
                    for candidate in expected
                )
            )
        self.check("reachable_complete_nuclear_maps", valid, [True] * len(expected))
        report = generate_reactions(
            {f"form{i}": form for i, form in enumerate(expected)},
            (),
            policy=GenerationPolicy(allylic_resonance=True),
        )
        self.check(
            "one_species_for_all_seed_forms",
            [report.complete, len(report.species), len(set(report.seed_aliases.values()))],
            [True, 1, 1],
        )
        limited = enumerate_allylic_resonance(expected[0], maximum_forms=len(expected) - 1)
        self.check(
            "bounded_not_falsely_complete",
            [limited.complete, len(limited.forms)],
            [False, len(expected) - 1],
        )

    def negatives(self, _):
        forms = declared_forms("butenyl")
        self.check("exact_graphs_remain_different", molecular_identity_mapping(*forms) is None)
        library, _, _ = example_resonance_inputs()
        self.check(
            "real_positional_isomers_not_merged",
            molecular_identity_mapping(
                library.species_by_id["butene1"].graph,
                library.species_by_id["butene2"].graph,
                allylic_resonance=True,
            )
            is None,
        )

        def isotope(atom):
            return replace(
                forms[0],
                atoms=tuple(
                    replace(a, isotope=2) if a.atom_id == atom else a for a in forms[0].atoms
                ),
            )

        first, second = isotope("H0.0"), isotope("H3.0")
        self.check(
            "same_formula_different_isotopomers",
            molecular_identity_mapping(first, second, allylic_resonance=True) is None
            and first.composition == second.composition,
        )
        radical = structures()["methyl"]
        cation = replace(
            radical,
            multiplicity=1,
            atoms=tuple(
                replace(a, charge=1, radical_electrons=0) if a.element == "C" else a
                for a in radical.atoms
            ),
        )
        self.check(
            "electronic_charge_not_merged",
            molecular_identity_mapping(radical, cation, allylic_resonance=True) is None,
        )
        oxygen = MolecularGraph(
            "O2",
            (GraphAtom("a", "O", 0, 1, 2), GraphAtom("b", "O", 0, 1, 2)),
            (GraphBond("a", "b", 1),),
            3,
            SOURCE,
        )
        self.check(
            "spin_multiplicity_not_merged",
            molecular_identity_mapping(
                oxygen, replace(oxygen, multiplicity=1), allylic_resonance=True
            )
            is None,
        )
        mapping = molecular_identity_mapping(first, renamed(first), allylic_resonance=True)
        self.check(
            "isotope_retained_positive_mapping",
            mapping is not None
            and labelled_fingerprint(first.to_dict(), mapping)
            == labelled_fingerprint(renamed(first).to_dict()),
        )

    def conflict(self, kind):
        library, families, conditions = example_resonance_inputs()
        before = plain(library.to_dict())
        error = ""
        with patch.object(
            ReactionNetworkSpec, "integrate_batch", side_effect=AssertionError("unexpected solve")
        ) as solve:
            try:
                if kind == "rate":
                    old = library.reactions[0]
                    changed = replace(
                        old,
                        kinetics=replace(
                            old.kinetics,
                            source=replace(
                                old.kinetics.source, source_id="conflicting-rate-source"
                            ),
                        ),
                    )
                    invalid = replace(library, reactions=(*library.reactions, changed))
                else:
                    alias = example_resonance_inputs(representative=1)[0].species_by_id["R"]
                    declaration = plain(alias.thermo_declaration)
                    if kind == "different-thermo":
                        declaration["thermo"]["data"][0][5] += 500
                    alias = replace(
                        alias,
                        species=replace(
                            alias.species, species_id="R" if kind == "same-id" else "R-other"
                        ),
                        thermo_declaration=declaration,
                        source=replace(alias.source, source_id="second-thermochemistry-source"),
                    )
                    invalid = replace(library, species=(*library.species, alias))
                grow_mechanism(invalid, families, ("H", "R"), conditions, policy=policy())
            except ValueError as exc:
                error = str(exc)
        self.check("identity_conflict_refused", bool(error))
        required = (
            "reaction keys"
            if kind == "rate"
            else "unique species"
            if kind == "same-id"
            else "resonance_identity_conflict"
        )
        self.check(
            "diagnostic_names_contract",
            required in error
            and (
                kind in ("rate", "same-id")
                or ("R-other" in error
                and "second-thermochemistry-source" in error)
            ),
        )
        self.check("no_solver_before_conflict", solve.call_count, 0)
        self.check("input_not_replaced", plain(library.to_dict()), before)
        self.details.append({"unit": self.current["name"], "failure_diagnostic": error})

    def truncation(self, _):
        library, families, conditions = example_resonance_inputs()
        error = ""
        with patch.object(
            ReactionNetworkSpec, "integrate_batch", side_effect=AssertionError("unexpected solve")
        ) as solve:
            try:
                grow_mechanism(library, families, ("H", "R"), conditions, policy=policy(1))
            except ValueError as exc:
                error = str(exc)
        self.check(
            "incomplete_identity_is_not_inequality", error, "resonance_identity_search_truncated"
        )
        self.check("no_solver_on_incomplete_identity", solve.call_count, 0)
        forms = declared_forms("butenyl")
        report = generate_reactions({"a": forms[0], "b": forms[1]}, (), policy=policy(1).generation)
        self.check("incomplete_static_generation", report.complete, False)
        self.check("truncation_reason_retained", list(report.truncated_reasons), [error])

    def aliases(self, _):
        library, families, _ = example_resonance_inputs()
        seeds = {key: library.species_by_id[key].graph for key in ("H", "R")}
        baseline = generate_reactions(seeds, families, policy=policy().generation)
        alias = example_resonance_inputs(representative=1)[0].species_by_id["R"].graph
        actual = generate_reactions({**seeds, "alias": alias}, families, policy=policy().generation)
        self.check(
            "alias_does_not_double_paths",
            [r.to_dict() for r in actual.reactions],
            [r.to_dict() for r in baseline.reactions],
        )
        self.check(
            "independent_product_isomer_set",
            sorted(
                signature(actual.species[k].to_dict()) for r in actual.reactions for k in r.products
            ),
            sorted(
                signature(library.species_by_id[k].graph.to_dict()) for k in ("butene1", "butene2")
            ),
        )
        bound = growth().network
        checks = path_checks(bound.to_dict(), [f.to_dict() for f in families])
        self.check(
            "actual_forms_nuclear_maps_recipe",
            checks,
            [{"nuclear_bijection": True, "recipe_replay": True, "topology_coordinates": True}] * 2,
        )
        self.check(
            "path_counts_are_not_species_or_rate_multipliers",
            [
                len(actual.species),
                len(actual.reactions),
                sum(len(r.paths) for r in actual.reactions),
            ],
            [4, 2, 2],
        )

    def mapped(self, kind):
        network, families = mapped_network(kind)
        checks = path_checks(network.to_dict(), [f.to_dict() for f in families])
        rows = network.reactions[0].metadata["atom_paths"][0]["atom_mapping"]
        self.check(
            "explicit_occurrences",
            sorted({r["reactant_index"] for r in rows}),
            [0, 1] if kind == "occurrences" else [0],
        )
        self.check("complete_nuclear_bijection", [r["nuclear_bijection"] for r in checks], [True])
        self.check("independent_recipe_replay", [r["recipe_replay"] for r in checks], [True])
        self.check(
            "original_topology_remapped", [r["topology_coordinates"] for r in checks], [True]
        )
        self.details.append({"unit": self.current["name"], "network": network.to_dict()})

    def growth(self, config):
        representative, factor = config
        library, families, conditions = example_resonance_inputs(
            representative=representative, per_path_factor=factor
        )
        result = growth(representative, factor)
        network = result.network
        self.details.append({"unit": self.current["name"], "growth": result.to_dict()})
        self.check(
            "complete_bound_species_and_stable_reactions",
            [result.complete, sorted(network.species_ids), sorted(network.reaction_ids)],
            [True, ["H", "R", "butene1", "butene2"], sorted(r.key for r in library.reactions)],
        )
        history = result.checkpoint.state["history"]
        self.check(
            "real_core_edge_promotion",
            [h["core_ids"] for h in history],
            [["H", "R"], ["H", "R", "butene1", "butene2"]],
        )
        self.check(
            "passive_edge_flux",
            [
                [s["edge_production_mol_L_s"][k] for k in ("butene1", "butene2")]
                for s in history[0]["conditions"][0]["samples"]
            ],
            [[0.04 * factor, 0.06 * factor]] * 5,
            numeric=True,
        )
        samples = history[-1]["conditions"][0]["samples"]
        keys = ("H", "R", "butene1", "butene2")
        self.check(
            "analytic_five_time_core_trajectory",
            [[s["core_amounts_mol"][k] for k in keys] for s in samples],
            [
                [analytic_amounts(t, factor=factor)[k] for k in keys]
                for t in conditions[0].sample_times_s
            ],
            numeric=True,
        )
        self.check(
            "bound_actual_form_replay",
            path_checks(network.to_dict(), [f.to_dict() for f in families]),
            [{"nuclear_bijection": True, "recipe_replay": True, "topology_coordinates": True}] * 2,
        )
        self.check(
            "declared_not_observed_degeneracy",
            sorted(
                (r.rate_law.parameters["A"], r.metadata["declared_degeneracy_factor"])
                for r in network.reactions
            ),
            [(2 * factor, factor), (3 * factor, factor)],
        )
        self.check(
            "original_network_and_library_codec",
            [
                ReactionNetworkSpec.from_dict(plain(network.to_dict())).to_dict()
                == network.to_dict(),
                MechanismGrowthLibrary.from_dict(plain(library.to_dict())).to_dict()
                == library.to_dict(),
            ],
            [True, True],
        )
        stopped = growth(representative, factor, resources=GrowthResourceLimits(maximum_rounds=1))
        resumed = growth(
            representative,
            factor,
            checkpoint=MechanismGrowthCheckpoint.from_dict(plain(stopped.checkpoint.to_dict())),
        )
        self.check(
            "resource_stop_exact_continuation",
            [
                stopped.status,
                resumed.network.to_dict() == network.to_dict(),
                resumed.checkpoint.state["history"] == history,
            ],
            ["resource_limit", True, True],
        )
        self.world(network, conditions[0], factor)

    def world(self, network, condition, factor):
        from chemworld.physchem.gas_reactor_network import GasReactorNetworkSpec, GasReactorNodeSpec
        from chemworld.runtime import ChemWorldRuntime
        from chemworld.world.gas_reactor_network import gas_reactor_network_world

        world = gas_reactor_network_world(
            GasReactorNetworkSpec(
                (GasReactorNodeSpec("reactor", 1.0),), (), condition.temperature_K, "reactor"
            ),
            network,
            {"reactor": dict(condition.initial_amounts_mol)},
        )
        runtime = ChemWorldRuntime(composition=world)
        resumed = None
        trajectory = []
        try:
            runtime.reset(seed=38)
            for dt in (0.2, 0.3, 0.5, 1.0):
                receipt = runtime.execute({"operation": "wait", "duration_s": dt})
                if receipt.info["transaction_status"] != "committed":
                    raise RuntimeError(str(receipt.info))
                state = runtime._state
                trajectory.append(
                    [
                        *(state.species_amounts[k] for k in ("H", "R", "butene1", "butene2")),
                        state.temperature_K,
                        state.ledger.time_s,
                        state.ledger.energy_jacket_J,
                    ]
                )
                if len(trajectory) == 2:
                    snapshot = plain(runtime.snapshot().to_dict())
                    resumed = ChemWorldRuntime.from_snapshot(snapshot)
            expected = []
            for t in (0.2, 0.5, 1.0, 2.0):
                amounts = analytic_amounts(t, factor=factor)
                expected.append(
                    [
                        *(amounts[k] for k in ("H", "R", "butene1", "butene2")),
                        condition.temperature_K,
                        t,
                        R * condition.temperature_K * (0.1 - amounts["R"]),
                    ]
                )
            self.check(
                "original_world_inventory_temperature_clock_and_first_law",
                trajectory,
                expected,
                numeric=True,
                rtol=2e-7,
                atol=2e-7,
            )
            for dt in (0.5, 1.0):
                resumed.execute({"operation": "wait", "duration_s": dt})
            expected_snapshot = plain(runtime.snapshot().to_dict())
            same = plain(resumed.snapshot().to_dict()) == expected_snapshot
            stem = self.current["name"]
            saved = self.output / (stem + "-snapshot.json")
            saved.write_text(json.dumps(snapshot, indent=2), encoding="utf8")
            target = self.output / (stem + "-fresh.json")
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(saved),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (self.output / (stem + "-fresh.log")).write_text(
                process.stdout + process.stderr, encoding="utf8"
            )
            fresh = json.loads(target.read_text(encoding="utf8")) if target.exists() else None
            self.check(
                "complete_snapshot_same_and_fresh_process",
                [same, process.returncode, fresh == expected_snapshot],
                [True, 0, True],
            )
            self.details.append(
                {"unit": stem, "world_trajectory": trajectory, "final_snapshot": expected_snapshot}
            )
        finally:
            runtime.close()
            if resumed is not None:
                resumed.close()

    def compatibility(self, _):
        from chemworld.examples.native_mechanism_growth import example_growth_inputs

        library, families, conditions = example_growth_inputs()

        def payload(generate, grow, policy_class, library_class):
            return plain(
                {
                    "generation": generate(
                        {s.species.species_id: s.graph for s in library.species}, families
                    ).to_dict(),
                    "growth": grow(library, families, ("ethyl",), conditions).to_dict(),
                    "policy": policy_class().to_dict(),
                    "library": library_class.from_dict(plain(library.to_dict())).to_dict(),
                }
            )

        current = payload(
            generate_reactions, grow_mechanism, CoreEdgeGrowthPolicy, MechanismGrowthLibrary
        )
        names = (
            "reaction_generation",
            "mechanism_growth_specs",
            "mechanism_growth_binding",
            "mechanism_growth",
        )
        saved = {name: sys.modules["chemworld.physchem." + name] for name in names}
        try:
            for name in names:
                full = "chemworld.physchem." + name
                module = types.ModuleType(full)
                module.__package__ = "chemworld.physchem"
                module.__file__ = str(self.original_source / (name + ".py"))
                sys.modules[full] = module
                exec(
                    compile(
                        Path(module.__file__).read_text(encoding="utf8"), module.__file__, "exec"
                    ),
                    module.__dict__,
                )
            old_generation = sys.modules["chemworld.physchem.reaction_generation"]
            old_growth = sys.modules["chemworld.physchem.mechanism_growth"]
            old_specs = sys.modules["chemworld.physchem.mechanism_growth_specs"]
            original = payload(
                old_generation.generate_reactions,
                old_growth.grow_mechanism,
                old_specs.CoreEdgeGrowthPolicy,
                old_specs.MechanismGrowthLibrary,
            )
        finally:
            for name, module in saved.items():
                sys.modules["chemworld.physchem." + name] = module
        for key in ("generation", "growth", "policy", "library"):
            self.check("exact_original_false_" + key, current[key], original[key])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        from chemworld.runtime import ChemWorldRuntime

        runtime = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text(encoding="utf8")))
        try:
            for dt in (0.5, 1.0):
                result = runtime.execute({"operation": "wait", "duration_s": dt})
                if result.info["transaction_status"] != "committed":
                    raise RuntimeError(str(result.info))
            with args.output.open("x", encoding="utf8") as stream:
                json.dump(runtime.snapshot().to_dict(), stream, indent=2, allow_nan=False)
        finally:
            runtime.close()
        return
    if args.original_source is None:
        parser.error("--original-source must contain the four preserved pre-38E modules")
    args.output.mkdir(parents=True, exist_ok=False)
    block = Block(args.output, args.original_source)
    start = time.perf_counter()
    for index, (name, kind, data, count) in enumerate(CASES, 1):
        block.current = {"name": name, "comparisons": [], "planned_comparisons": count}
        print(
            f"38E {index - 1}/{len(CASES)} complete; starting {name}; "
            f"elapsed={time.perf_counter() - start:.2f}s",
            flush=True,
        )
        try:
            getattr(block, kind)(data)
            if len(block.current["comparisons"]) != count:
                raise AssertionError("fixed comparison denominator changed")
        except Exception:
            block.current["exception"] = traceback.format_exc()
            while len(block.current["comparisons"]) < count:
                block.current["comparisons"].append(
                    {
                        "name": "not_completed_after_exception",
                        "passed": False,
                        "status": "unexecuted",
                    }
                )
        block.current["passed"] = (
            all(c["passed"] for c in block.current["comparisons"])
            and "exception" not in block.current
        )
        block.records.append(block.current)
        elapsed = time.perf_counter() - start
        print(
            f"38E {index}/{len(CASES)} {name}: {'PASS' if block.current['passed'] else 'FAIL'}; "
            f"throughput={index / max(elapsed, 0.001):.2f} units/s; "
            f"ETA={(len(CASES) - index) * elapsed / index:.1f}s",
            flush=True,
        )
    comparisons = [c for r in block.records for c in r["comparisons"]]
    summary = {
        "mode": "development",
        "scope": "existing finite resonance identity through original core-edge/world",
        "units": len(CASES),
        "passed_units": sum(r["passed"] for r in block.records),
        "comparisons": sum(c[3] for c in CASES),
        "passed_comparisons": sum(c["passed"] for c in comparisons),
        "failed_comparisons": sum(
            not c["passed"] and c.get("status") != "unexecuted" for c in comparisons
        ),
        "unexecuted_comparisons": sum(c.get("status") == "unexecuted" for c in comparisons),
        "elapsed_s": time.perf_counter() - start,
        "failed_units": [r["name"] for r in block.records if not r["passed"]],
        "reference": (
            "authored labelled Lewis forms, exhaustive nuclear permutations, "
            "independent recipe replay and analytic kinetics; no RMG execution"
        ),
    }
    for filename, payload in (
        ("records.json", block.records),
        ("details.json", block.details),
        ("summary.json", summary),
    ):
        (args.output / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8"
        )
    print(json.dumps(summary, indent=2), flush=True)
    if summary["failed_units"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
