"""Fixed 24-unit / 136-comparison authored topology-to-world development block."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from dataclasses import replace
from pathlib import Path

import cantera as ct
import numpy as np
from rmg37_fixtures import bindings
from rmg37c_cases import cases
from rmg37c_reference import (
    TIMES,
    R,
    energy,
    maps_valid,
    network_path_validation,
    paths_valid,
    product_signatures,
    reference_history,
    reference_phase,
    signature,
)

from chemworld.examples.native_intramolecular_reactions import (
    example_graphs,
    example_library,
    example_network,
    example_world,
)
from chemworld.physchem.mechanism_growth import grow_mechanism
from chemworld.physchem.mechanism_growth_specs import (
    GrowthCondition,
    GrowthResourceLimits,
    MechanismGrowthCheckpoint,
    MechanismGrowthLibrary,
)
from chemworld.physchem.reaction_family import ReactionFamily, extended_radical_reaction_families
from chemworld.physchem.reaction_generation import bind_generated_network, generate_reactions
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.runtime import ChemWorldRuntime


def json_copy(value):
    return json.loads(json.dumps(value))


class Block:
    def __init__(self, output):
        output.mkdir(parents=True, exist_ok=False)
        self.output, self.rows, self.units, self.details, self.case = output, [], [], [], {}

    def check(self, name, actual, expected, *, rtol=0.0, atol=0.0):
        exact = isinstance(expected, (bool, str, dict))
        if exact:
            passed = json_copy(actual) == json_copy(expected)
            error = None
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            passed = bool(
                a.shape == b.shape
                and np.all(np.isfinite(a))
                and np.allclose(a, b, rtol=rtol, atol=atol)
            )
            error = float(np.max(np.abs(a - b))) if a.size and a.shape == b.shape else None
        self.rows.append(
            {
                "unit": self.case["name"],
                "quantity": name,
                "passed": passed,
                "actual": actual,
                "expected": expected,
                "rtol": rtol,
                "atol": atol,
                "maximum_absolute_error": error,
            }
        )

    def structures(self):
        c = self.case
        family = extended_radical_reaction_families()[c["family"]]
        initial = {k: v.to_dict() for k, v in c["seeds"].items()}
        report = generate_reactions(c["seeds"], (family,))
        self.details.append({"unit": c["name"], "generation": report.to_dict()})
        identity = product_signatures(report) == sorted(signature(g) for g in c["expected"])
        paths = sum(len(r.paths) for r in report.reactions)
        if c["kind"] == "product":
            self.check("complete", report.complete, True)
            self.check("reaction_count", len(report.reactions), len(c["expected"]))
            self.check("independent_product_identity", identity, True)
            self.check("full_nuclear_bijections", maps_valid(report), True)
            self.check("distinct_edit_paths", paths, c["paths"])
        elif c["kind"] == "topology":
            self.check("complete", report.complete, True)
            self.check("reaction_count", len(report.reactions), len(c["expected"]))
            self.check("source_unchanged", {k: v.to_dict() for k, v in c["seeds"].items()}, initial)
            diagnostic = any("shorter original-graph route" in r["reason"] for r in report.rejected)
            self.check(
                "topology_or_identity", diagnostic if c.get("shorter_route") else identity, True
            )
        else:
            self.check("distinct_edit_paths", paths, c["paths"])
            self.check("full_nuclear_bijections", maps_valid(report), True)
            self.check("original_path_evidence", paths_valid(report, c["family"]), True)
            self.check(
                "family_json",
                ReactionFamily.from_dict(json_copy(family.to_dict())).to_dict(),
                family.to_dict(),
            )
            if c["family"] == 0:
                values = []
                for factor in (1, 3):
                    species, rates = bindings(report, rate=2.0, factor=factor)
                    network = bind_generated_network(
                        report,
                        species,
                        rates,
                        network_id="explicit-degeneracy",
                        reference_temperature_K=500,
                    )
                    values.append(network.reactions[0].rate_law.parameters["A"])
                self.check("declared_not_observed_degeneracy", values, [2.0, 6.0])
            else:
                self.check("independent_isotopic_products", identity, True)

    def world(self):
        c = self.case
        runtime = ChemWorldRuntime(
            composition=example_world(temperature_K=c["temperature"], isothermal=c["isothermal"])
        )
        restored = None
        trajectory, temperatures, elements, balances, times, commits = [], [], [], [], [], []
        reference_n, reference_t = reference_history(c["temperature"], c["isothermal"])
        initial_energy = energy([0.02, 0], c["temperature"], support=0 if c["isothermal"] else 1)
        try:
            runtime.reset(seed=37)
            for i in range(len(TIMES)):
                if i:
                    reply = runtime.execute({"operation": "wait", "duration_s": 0.25})
                    commits.append(reply.info["transaction_status"])
                state = runtime._state
                amounts = [
                    state.species_amounts.get(k, 0.0) for k in ("hexenyl", "cyclopentylmethyl")
                ]
                trajectory.append(amounts)
                temperatures.append(state.temperature_K)
                times.append(state.ledger.time_s)
                elements.append([6 * sum(amounts), 11 * sum(amounts)])
                delta = (
                    energy(amounts, state.temperature_K, support=0 if c["isothermal"] else 1)
                    - initial_energy
                )
                bath = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
                balances.append(
                    [delta - bath, bath - (-5000 * amounts[1] if c["isothermal"] else 0.0)]
                )
            self.check(
                "inventory_trajectory", trajectory, reference_n.tolist(), rtol=3e-6, atol=2e-10
            )
            self.check("temperature_trajectory", temperatures, reference_t, rtol=2e-7, atol=2e-5)
            self.check("first_law_and_bath", balances, [[0.0, 0.0]] * 5, atol=2e-6)
            self.check("element_inventory", elements, [[0.12, 0.22]] * 5, atol=2e-10)
            self.check("time", times, TIMES.tolist(), atol=1e-12)
            self.check("four_commits", {"values": commits}, {"values": ["committed"] * 4})
            snapshot = json_copy(runtime.snapshot().to_dict())
            checkpoint = self.output / (c["name"] + "-snapshot.json")
            checkpoint.write_text(json.dumps(snapshot), encoding="utf-8")
            restored = ChemWorldRuntime.from_snapshot(snapshot)
            statuses = [
                world.execute({"operation": "wait", "duration_s": 0.25}).info["transaction_status"]
                for world in (runtime, restored)
            ]
            expected_snapshot = json_copy(runtime.snapshot().to_dict())
            self.check(
                "exact_json_continuation",
                {"statuses": statuses, "snapshot": restored.snapshot().to_dict()},
                {"statuses": ["committed"] * 2, "snapshot": expected_snapshot},
            )
            target = self.output / (c["name"] + "-fresh.json")
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(checkpoint),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (self.output / (c["name"] + "-fresh.log")).write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            fresh = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
            self.check(
                "exact_fresh_process_continuation",
                {"exit": process.returncode, "snapshot": fresh},
                {"exit": 0, "snapshot": expected_snapshot},
            )
            network, gas = example_network(), reference_phase(c["temperature"])
            native = [
                next(
                    iter(
                        network.reaction_rates(
                            {"hexenyl": 1.0}, volume_L=1.0, temperature_K=c["temperature"]
                        ).values()
                    )
                )
            ]
            expected = [float(gas.forward_rate_constants[0])]
            for key, name in (("hexenyl", "R"), ("cyclopentylmethyl", "P")):
                value = network.imported_gas_thermochemistry[key].evaluate(c["temperature"])
                native.extend([value.enthalpy_J_mol, value.cp_J_mol_K])
                expected.extend(
                    [
                        gas.species(name).thermo.h(c["temperature"]) / 1000,
                        gas.species(name).thermo.cp(c["temperature"]) / 1000,
                    ]
                )
            self.check("public_rate_and_thermo", native, expected, rtol=2e-12, atol=1e-10)
            generated = generate_reactions(
                {"hexenyl": example_graphs()["hexenyl"]}, extended_radical_reaction_families()[-1:]
            )
            paths = json_copy(generated.reactions[0].paths)
            for path in paths:
                for row in path["atom_mapping"]:
                    row["product_species"] = "cyclopentylmethyl"
            meta = network.reactions[0].metadata
            self.check(
                "mapped_metadata_and_network_codec",
                {
                    "paths": meta["atom_paths"],
                    "count": meta["observed_edit_paths"],
                    "factor": meta["declared_degeneracy_factor"],
                    "family": meta["family_id"],
                    "codec": ReactionNetworkSpec.from_dict(json_copy(network.to_dict())).to_dict(),
                },
                {
                    "paths": paths,
                    "count": 1,
                    "factor": 1.0,
                    "family": "radical_cyclization_5_exo",
                    "codec": network.to_dict(),
                },
            )
            self.details.append(
                {
                    "unit": c["name"],
                    "trajectory": trajectory,
                    "temperature_K": temperatures,
                    "network": network.to_dict(),
                    "balances_J": balances,
                }
            )
        finally:
            runtime.close()
            if restored is not None:
                restored.close()

    def growth(self):
        c = self.case
        library, families = example_library(), extended_radical_reaction_families()[-1:]
        conditions = (
            GrowthCondition(
                "cyclization", {"hexenyl": 0.02}, c["temperature"], 1.0, (0.0, 0.25, 0.5, 1.0)
            ),
        )
        budgets = GrowthResourceLimits(
            maximum_rounds=8,
            maximum_condition_solves=40,
            maximum_core_species=12,
            maximum_edge_candidates=32,
        )
        result = grow_mechanism(library, families, ("hexenyl",), conditions, resources=budgets)
        self.details.append({"unit": c["name"], "growth": result.to_dict()})
        self.check("complete", result.complete, True)
        self.check(
            "core_species",
            {"ids": sorted(result.network.species_ids)},
            {"ids": ["cyclopentylmethyl", "hexenyl"]},
        )
        self.check(
            "stoichiometry_and_mapped_paths",
            {
                "reactions": len(result.network.reactions),
                "stoich": [r.stoichiometry for r in result.network.reactions],
                "paths": network_path_validation(result.network),
            },
            {
                "reactions": 1,
                "stoich": [{"hexenyl": -1.0, "cyclopentylmethyl": 1.0}],
                "paths": {
                    "nuclear_bijections": True,
                    "original_paths": True,
                    "path_counts": [1],
                    "identities": {k: signature(v) for k, v in example_graphs().items()},
                },
            },
        )
        samples = result.checkpoint.state["history"][-1]["conditions"][0]["samples"]
        amounts = [
            [s["core_amounts_mol"].get(k, 0.0) for k in ("hexenyl", "cyclopentylmethyl")]
            for s in samples
        ]
        k = 2 * np.exp(-2000 / (R * c["temperature"]))
        expected = [
            [0.02 * np.exp(-k * t), 0.02 * (1 - np.exp(-k * t))]
            for t in conditions[0].sample_times_s
        ]
        self.check("original_growth_ODE", amounts, expected, rtol=2e-7, atol=2e-11)
        self.check(
            "library_JSON",
            MechanismGrowthLibrary.from_dict(json_copy(library.to_dict())).to_dict(),
            library.to_dict(),
        )
        stopped = grow_mechanism(
            library,
            families,
            ("hexenyl",),
            conditions,
            resources=replace(budgets, maximum_rounds=1),
        )
        checkpoint = MechanismGrowthCheckpoint.from_dict(json_copy(stopped.checkpoint.to_dict()))
        resumed = grow_mechanism(
            library, families, ("hexenyl",), conditions, resources=budgets, checkpoint=checkpoint
        )
        self.details.append(
            {"unit": c["name"], "stopped": stopped.to_dict(), "resumed": resumed.to_dict()}
        )
        self.check(
            "bounded_checkpoint_exact_resume",
            {
                "stop": stopped.status,
                "complete": resumed.complete,
                "network": resumed.network.to_dict(),
                "history": resumed.checkpoint.state["history"],
                "condition_solves": resumed.checkpoint.state["condition_solves"],
            },
            {
                "stop": "resource_limit",
                "complete": True,
                "network": result.network.to_dict(),
                "history": result.checkpoint.state["history"],
                "condition_solves": result.checkpoint.state["condition_solves"],
            },
        )


def run(output, canary=False):
    fixed = cases()
    if canary:
        fixed = [fixed[i] for i in (0, 18, 22)]
    block, started = Block(output), time.monotonic()
    planned = sum(c["expected_comparisons"] for c in fixed)
    for index, case in enumerate(fixed, 1):
        block.case = case
        before, error = len(block.rows), None
        try:
            if case["kind"] == "world":
                block.world()
            elif case["kind"] == "growth":
                block.growth()
            else:
                block.structures()
        except Exception:
            error = traceback.format_exc()
        selected = block.rows[before:]
        passed = (
            error is None
            and len(selected) == case["expected_comparisons"]
            and all(r["passed"] for r in selected)
        )
        record = {
            "unit": case["name"],
            "kind": case["kind"],
            "passed": passed,
            "error": error,
            "planned_comparisons": case["expected_comparisons"],
            "executed": len(selected),
            "passed_comparisons": sum(r["passed"] for r in selected),
        }
        block.units.append(record)
        with (output / "unit_results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        with (output / "comparisons.jsonl").open("a", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row) + "\n")
        elapsed = time.monotonic() - started
        print(
            f"37C {index}/{len(fixed)} {case['name']} {'PASS' if passed else 'FAIL'} "
            f"comparisons={len(block.rows)}/{planned} "
            f"rate={index / max(elapsed, 0.001):.2f} units/s "
            f"ETA={(len(fixed) - index) * elapsed / index:.1f}s {error or ''}",
            flush=True,
        )
    summary = {
        "schema": "chemworld-rmg37c-development-v1",
        "reference": "authored graph oracle; Cantera " + ct.__version__ + " / SciPy scalar ODE",
        "planned_units": len(fixed),
        "units_passed": sum(r["passed"] for r in block.units),
        "planned_comparisons": planned,
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(r["passed"] for r in block.rows),
        "unexecuted_comparisons": planned - len(block.rows),
        "failures": [r for r in block.units if not r["passed"]],
        "failed_comparisons": [r for r in block.rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "passed": all(r["passed"] for r in block.units) and len(block.rows) == planned,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "details.json").write_text(json.dumps(block.details, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        try:
            reply = runtime.execute({"operation": "wait", "duration_s": 0.25})
            if reply.info["transaction_status"] != "committed":
                raise ValueError(str(reply.info))
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(runtime.snapshot().to_dict(), stream)
        finally:
            runtime.close()
    else:
        summary = run(args.output, args.canary)
        print(
            json.dumps(
                {k: v for k, v in summary.items() if k not in {"failures", "failed_comparisons"}}
            ),
            flush=True,
        )
        raise SystemExit(0 if summary["passed"] else 1)
