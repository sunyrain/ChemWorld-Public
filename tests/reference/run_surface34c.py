"""Fixed 24-unit / 376-comparison multidentate qualification with all failures retained."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import cantera as ct
import numpy as np
from run_surface34b import Block as ComparisonBlock
from surface34c_reference import (
    AREA,
    COVERAGES,
    TIMES,
    R,
    cases,
    elements,
    energy,
    finite_history,
    reference,
    sizes,
    thermal_endpoint,
)

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration, example_world
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    solve_catalytic_steady_state,
)
from chemworld.physchem.catalysis_network import (
    catalytic_inventory_key,
    catalytic_system,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.surface_mechanism_import import (
    ImportedSurfaceMechanism,
    convert_surface_mechanism,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.catalysis_analysis import analyze_catalytic_world

POLICY = RootPolicy(tolerance=1e-10, maximum_iterations=80, maximum_backtracks=40)


def native(case):
    payload = authored_declaration(case["density"], site_occupancy=case["size"])
    bundle = convert_surface_mechanism(
        payload,
        surface_phase="surface",
        source=SOURCE,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    )
    spec = bundle.surface
    scales = spec.coverage_amount_scales_mol({"surface": AREA * case["density"]})
    slot = make_catalytic_slot(
        "surface",
        "reactor",
        spec,
        capacities_mol={"surface": AREA * case["density"]},
        initial_amounts_mol={
            k: t * scales[k] for k, t in zip(spec.adsorbates, COVERAGES, strict=True)
        },
    )
    total = 1e5 * 0.001 / (R * case["temperature"])
    state = pack_catalytic_state(
        spec,
        (slot,),
        {"A": 0.7 * total, "B": 0.3 * total},
        temperature_K=case["temperature"],
        volume_L=1.0,
    )
    return payload, bundle, slot, state


def amounts(state, spec, slot):
    return np.array(
        [state.amounts_mol[k] for k in spec.gas_species]
        + [state.amounts_mol[catalytic_inventory_key(slot, k)] for k in spec.adsorbates]
    )


def world_amounts(state):
    slot = state.state_slots.slots[0]
    return np.array(
        [state.species_amounts[k] for k in ("A", "B")]
        + [slot.values[k] for k in ("Pt", "HA", "HB", "HAA")]
    )


class Block(ComparisonBlock):
    def static(self):
        c = self.case
        payload, bundle, slot, state = native(c)
        spec = bundle.surface
        gas, surface = reference(payload, c["temperature"])
        rates = evaluate_catalytic_rates(state, slot, spec)
        self.compare(
            "coverage",
            [rates.activities[k] for k in spec.adsorbates],
            surface.coverages.tolist(),
            rtol=2e-12,
            atol=1e-16,
        )
        self.compare(
            "surface_standard_concentration",
            [c["density"] / spec.site_occupancies[k] for k in spec.adsorbates],
            (surface.concentrations * 1000 / surface.coverages).tolist(),
            rtol=2e-12,
            atol=1e-16,
        )
        for key, function in (
            ("cp_J_mol_K", lambda s, t: s.thermo.cp(t)),
            ("enthalpy_J_mol", lambda s, t: s.thermo.h(t)),
            ("entropy_J_mol_K", lambda s, t: s.thermo.s(t)),
            ("gibbs_J_mol", lambda s, t: s.thermo.h(t) - t * s.thermo.s(t)),
        ):
            self.compare(
                key,
                [getattr(s.evaluate(c["temperature"]), key) for s in spec.species],
                [
                    function(s, c["temperature"]) / 1000
                    for s in (*gas.species(), *surface.species())
                ],
                rtol=2e-9,
                atol=2e-6,
            )
        for label, actual, expected in (
            (
                "forward",
                list(rates.forward_rates_mol_s.values()),
                surface.forward_rates_of_progress,
            ),
            (
                "reverse",
                list(rates.reverse_rates_mol_s.values()),
                surface.reverse_rates_of_progress,
            ),
        ):
            self.compare(label, actual, (expected * 1000 * AREA).tolist())
        derivative = np.array(
            [rates.amount_rates_mol_s[k] for k in spec.gas_species]
            + [rates.amount_rates_mol_s[catalytic_inventory_key(slot, k)] for k in spec.adsorbates]
        )
        expected = (
            np.concatenate(
                (surface.get_net_production_rates(gas), surface.get_net_production_rates(surface))
            )
            * 1000
            * AREA
        )
        self.compare("molecular_derivatives", derivative.tolist(), expected.tolist())
        self.compare(
            "element_derivatives",
            elements(derivative, gas, surface).tolist(),
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-16,
        )
        self.compare(
            "weighted_site_derivative",
            float(derivative[2:] @ sizes(surface)),
            0.0,
            rtol=0.0,
            atol=1e-16,
        )
        restored = ImportedSurfaceMechanism.from_dict(json.loads(json.dumps(bundle.to_dict())))
        again = evaluate_catalytic_rates(state, slot, restored.surface)
        self.compare(
            "codec_exact_rates",
            {"forward": again.forward_rates_mol_s, "reverse": again.reverse_rates_mol_s},
            {"forward": rates.forward_rates_mol_s, "reverse": rates.reverse_rates_mol_s},
        )

    def history(self):
        c = self.case
        payload, bundle, slot, initial = native(c)
        spec = bundle.surface
        gas, surface = reference(payload, c["temperature"])
        expected = finite_history(payload, c["temperature"])
        solved = catalytic_system(
            spec, (slot,), support_heat_capacity_J_K=1.0, isothermal=True
        ).advance(initial, 1.0, evaluation_times_s=TIMES)
        for t, state, ref in zip(TIMES, solved.trajectory, expected, strict=True):
            actual = amounts(state, spec, slot)
            prefix = f"t={t}:"
            self.compare(
                prefix + "gas",
                actual[:2].tolist(),
                ref["amounts"][:2].tolist(),
                rtol=3e-5,
                atol=2e-11,
            )
            self.compare(
                prefix + "surface_molecules",
                actual[2:].tolist(),
                ref["amounts"][2:].tolist(),
                rtol=3e-5,
                atol=2e-11,
            )
            self.compare(
                prefix + "elements",
                elements(actual, gas, surface).tolist(),
                ref["elements"].tolist(),
                rtol=2e-8,
                atol=1e-12,
            )
            self.compare(
                prefix + "weighted_sites",
                float(actual[2:] @ sizes(surface)),
                AREA * c["density"],
                rtol=2e-8,
                atol=1e-12,
            )
            self.compare(
                prefix + "temperature", state.temperature_K, c["temperature"], rtol=0, atol=1e-9
            )
            self.compare(
                prefix + "total_U",
                energy(actual, state.temperature_K, gas, surface),
                ref["energy"],
                rtol=3e-6,
                atol=2e-5,
            )
            self.details.append(
                {
                    "unit": c["name"],
                    "time_s": t,
                    "native_mol": actual.tolist(),
                    "reference_mol": ref["amounts"].tolist(),
                }
            )

    def world(self):
        c = self.case
        composition = example_world(site_occupancy=c["size"], isothermal=c["isothermal"])
        runtime, restored = ChemWorldRuntime(composition=composition), None
        payload = authored_declaration(site_occupancy=c["size"])
        gas, surface = reference(payload, 700)
        ref = thermal_endpoint(payload, 700, c["isothermal"])
        try:
            runtime.reset(seed=34)
            initial = runtime._state
            reply = runtime.execute({"operation": "wait", "duration_s": 1})
            state = runtime._state
            actual = world_amounts(state)
            self.compare("commit", reply.info["transaction_status"], "committed")
            self.compare(
                "molecular_endpoint",
                actual.tolist(),
                ref["amounts"].tolist(),
                rtol=3e-5,
                atol=2e-11,
            )
            self.compare(
                "temperature", state.temperature_K, float(ref["temperature"]), rtol=3e-7, atol=3e-5
            )
            delta = energy(actual, state.temperature_K, gas, surface) - energy(
                world_amounts(initial), initial.temperature_K, gas, surface
            )
            q = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
            self.compare("independent_first_law", delta, q, rtol=0.0, atol=3e-6)
            self.compare(
                "weighted_sites",
                float(actual[2:] @ sizes(surface)),
                AREA * c["density"],
                rtol=2e-8,
                atol=1e-12,
            )
            self.compare(
                "elements",
                elements(actual, gas, surface).tolist(),
                elements(world_amounts(initial), gas, surface).tolist(),
                rtol=2e-8,
                atol=1e-12,
            )
            before = json.loads(json.dumps(runtime.snapshot().to_dict()))
            query = analyze_catalytic_world(composition, state, target_species="B", policy=POLICY)
            self.compare("world_query_status", query["status"], "converged")
            self.compare("query_snapshot_unchanged", runtime.snapshot().to_dict(), before)
            restored = ChemWorldRuntime.from_snapshot(before)
            for instance in (runtime, restored):
                result = instance.execute({"operation": "wait", "duration_s": 1})
                if result.info["transaction_status"] != "committed":
                    raise ValueError("snapshot continuation action failed")
            self.compare(
                "in_process_exact_snapshot",
                runtime.snapshot().to_dict(),
                restored.snapshot().to_dict(),
            )
            source, target = (
                self.path / (c["name"] + "-snapshot.json"),
                self.path / (c["name"] + "-resumed.json"),
            )
            source.write_text(json.dumps(before), encoding="utf8")
            fresh = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(source),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (self.path / (c["name"] + "-process.log")).write_text(
                fresh.stdout + fresh.stderr, encoding="utf8"
            )
            if fresh.returncode:
                raise ValueError("fresh process continuation failed: " + fresh.stderr)
            self.compare(
                "fresh_process_exact_snapshot",
                json.loads(target.read_text(encoding="utf8")),
                runtime.snapshot().to_dict(),
            )
            self.details.append(
                {
                    "unit": c["name"],
                    "native_mol": actual.tolist(),
                    "reference_mol": ref["amounts"].tolist(),
                    "native_T": state.temperature_K,
                    "reference_T": ref["temperature"],
                    "delta_U": delta,
                    "Q": q,
                    "reference_nfev": ref["nfev"],
                    "query": query,
                }
            )
        finally:
            runtime.close()
            if restored:
                restored.close()

    def structure(self):
        c = self.case
        payload, bundle, _slot, _initial = native(c)
        spec = bundle.surface
        seed = dict(zip(spec.adsorbates, COVERAGES, strict=True))
        before = copy.deepcopy(seed)
        boundary = CatalyticAnalysisConditions(700, {"A": 7e4, "B": 3e4}, {"surface": 1e-6})
        root = solve_catalytic_steady_state(spec, boundary, seed, target_species="B", policy=POLICY)
        self.details.append({"unit": c["name"], "root": root})
        gas, surface = reference(payload, 700)
        surface.advance_coverages(
            1e4, rtol=1e-9, atol=1e-18, max_steps=100000, max_error_test_failures=20
        )
        residual = surface.get_net_production_rates(surface) * sizes(surface) / surface.site_density
        self.details[-1]["reference_coverages"] = surface.coverages.tolist()
        self.details[-1]["reference_residual"] = residual.tolist()
        if max(abs(residual)) > 1e-6:
            raise ValueError("source steady reference residual exceeds fixed limit")
        if root["status"] != "converged":
            raise ValueError("native conserved steady root unavailable: " + root["failure_reason"])
        self.compare(
            "steady_coverages",
            [root["coverages"][k] for k in spec.adsorbates],
            surface.coverages.tolist(),
            rtol=2e-4,
            atol=2e-7,
        )
        self.compare(
            "steady_gas_rates",
            [root["gas_rates_mol_s"][k] for k in spec.gas_species],
            (surface.get_net_production_rates(gas) * 1000 * AREA).tolist(),
            rtol=2e-3,
            atol=2e-10,
        )
        self.compare(
            "weighted_sites",
            sum(
                root["adsorbate_amounts_mol"][k] * spec.site_occupancies[k] for k in spec.adsorbates
            ),
            1e-6,
            rtol=2e-8,
            atol=1e-12,
        )
        self.compare("rank", root["steady_chart"]["reachable_dimension"], 3, rtol=0, atol=0)
        rejected = False
        try:
            make_catalytic_slot(
                "bad",
                "reactor",
                spec,
                capacities_mol={"surface": 1e-6},
                initial_amounts_mol={k: t * 1e-6 for k, t in seed.items()},
            )
        except ValueError:
            rejected = True
        self.compare("molecular_sum_is_not_capacity", rejected, True)
        self.compare("initial_unchanged", seed, before)


def run(path, canary=False):
    block, started = Block(path), time.monotonic()
    fixed = cases()
    if canary:
        fixed = [fixed[i] for i in (0, 12, 18)]
    for i, case in enumerate(fixed, 1):
        block.case = case
        offset, error = len(block.rows), None
        try:
            getattr(block, "static" if case["kind"] == "state" else case["kind"])()
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        selected = block.rows[offset:]
        passed = (
            error is None
            and len(selected) == case["expected_comparisons"]
            and all(r["passed"] for r in selected)
        )
        block.units.append(
            {
                **case,
                "passed": passed,
                "error": error,
                "comparisons_executed": len(selected),
                "comparisons_passed": sum(r["passed"] for r in selected),
            }
        )
        elapsed = time.monotonic() - started
        print(
            f"34C {i}/{len(fixed)} {case['name']} {'PASS' if passed else 'FAIL'} "
            f"throughput={i / max(elapsed, 1e-9):.3f} units/s "
            f"ETA={(len(fixed) - i) * elapsed / i:.1f}s",
            flush=True,
        )
        summary = {
            "schema": "chemworld-surface34c-development-v1",
            "mode": "development",
            "cantera_version": ct.__version__,
            "units_total": len(fixed),
            "units_completed": len(block.units),
            "units_passed": sum(u["passed"] for u in block.units),
            "comparisons_expected": sum(u["expected_comparisons"] for u in fixed),
            "comparisons_executed": len(block.rows),
            "comparisons_passed": sum(r["passed"] for r in block.rows),
            "units": block.units,
            "failed_comparisons": [r for r in block.rows if not r["passed"]],
            "elapsed_s": elapsed,
        }
        for filename, value in (
            ("summary.json", summary),
            ("comparisons.json", block.rows),
            ("details.json", block.details),
        ):
            (path / filename).write_text(json.dumps(value, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        world = ChemWorldRuntime.from_snapshot(json.loads(args.resume.read_text(encoding="utf8")))
        try:
            reply = world.execute({"operation": "wait", "duration_s": 1})
            if reply.info["transaction_status"] != "committed":
                raise ValueError("fresh process wait did not commit")
            args.output.write_text(json.dumps(world.snapshot().to_dict()), encoding="utf8")
        finally:
            world.close()
    else:
        run(args.output, args.canary)
