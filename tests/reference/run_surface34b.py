"""Fixed 52-unit / 552-comparison real Cantera development qualification."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import cantera as ct
import numpy as np
from surface34b_cases import TIMES, cases, payload_for, reference, source_payload

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration, example_world
from chemworld.physchem.catalysis_network import (
    catalytic_system,
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.surface_mechanism_import import (
    ImportedSurfaceMechanism,
    convert_surface_mechanism,
)
from chemworld.runtime import ChemWorldRuntime

UPSTREAM = DataSourceProvenance(
    "cantera-ptcombust-surface",
    0,
    "Cantera 3.2.0 distributed ptcombust.yaml; Deutschmann et al., 1996; public data API export",
    "Cantera-3.2.0/Pt-1.2",
    "Cantera-distribution-BSD-3-Clause; original attribution retained",
)


def native(payload, source, temperature, pressure, coverage):
    bundle = convert_surface_mechanism(
        payload,
        surface_phase=payload["phases"][1]["name"],
        source=source,
        minimum_temperature_K=300,
        maximum_temperature_K=1900,
    )
    spec = bundle.surface
    site = spec.site_types[0]
    capacity = 0.1 * spec.site_densities_mol_m2[site]
    slot = make_catalytic_slot(
        "surface",
        "reactor",
        spec,
        capacities_mol={site: capacity},
        initial_amounts_mol={
            k: capacity * c for k, c in zip(spec.adsorbates, coverage, strict=True)
        },
    )
    gas_total = pressure * 0.001 / (ct.gas_constant / 1000 * temperature)
    fractions = (
        [0.7, 0.3]
        if len(spec.gas_species) == 2
        else [1 / len(spec.gas_species)] * len(spec.gas_species)
    )
    physical = pack_catalytic_state(
        spec,
        (slot,),
        {k: gas_total * x for k, x in zip(spec.gas_species, fractions, strict=True)},
        temperature_K=temperature,
        volume_L=1.0,
    )
    return bundle, slot, physical


def flatten_native(state, spec, slot):
    return [state.amounts_mol[k] for k in spec.gas_species] + [
        state.amounts_mol[f"{slot.spec.slot_id}:{k}"] for k in spec.adsorbates
    ]


def elements(amounts, gas, surface):
    species = [*gas.species(), *surface.species()]
    names = sorted({key for s in species for key in s.composition})
    return [
        sum(n * s.composition.get(key, 0) for n, s in zip(amounts, species, strict=True))
        for key in names
    ]


class Block:
    def __init__(self, path):
        self.path = path
        path.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.units = []
        self.case = {}
        self.details = []

    def compare(self, name, actual, expected, *, rtol=2e-7, atol=1e-16):
        exact = isinstance(expected, (bool, str, dict))
        if exact:
            # The contract is exact JSON continuation, not tuple-vs-list Python types.
            passed = json.loads(json.dumps(actual)) == json.loads(json.dumps(expected))
            error = None
        else:
            a, b = np.asarray(actual), np.asarray(expected)
            passed = bool(
                a.shape == b.shape
                and np.all(np.isfinite(a))
                and np.allclose(a, b, rtol=rtol, atol=atol)
            )
            error = float(np.max(np.abs(a - b))) if a.size else 0.0
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

    def state(self, payload, source):
        c = self.case
        bundle, slot, state = native(
            payload, source, c["temperature"], c["pressure"], c["coverage"]
        )
        spec = bundle.surface
        gas, surface = reference(payload, c["temperature"], c["pressure"], c["coverage"])
        # Species thermo is at the declared standard pressure. Cantera phase
        # standard_entropies_R / standard_gibbs_RT instead include current P.
        for key, function in (
            ("cp_J_mol_K", lambda s, t: s.thermo.cp(t)),
            ("enthalpy_J_mol", lambda s, t: s.thermo.h(t)),
            ("entropy_J_mol_K", lambda s, t: s.thermo.s(t)),
            ("gibbs_J_mol", lambda s, t: s.thermo.h(t) - t * s.thermo.s(t)),
        ):
            actual = [getattr(s.evaluate(c["temperature"]), key) for s in spec.species]
            expected = [
                function(s, c["temperature"]) / 1000 for s in (*gas.species(), *surface.species())
            ]
            self.compare(key, actual, expected, rtol=2e-9, atol=2e-6)
        rates = evaluate_catalytic_rates(state, slot, spec)
        for key, actual, expected in (
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
            (
                "net",
                [
                    rates.forward_rates_mol_s[k] - rates.reverse_rates_mol_s[k]
                    for k in rates.forward_rates_mol_s
                ],
                surface.net_rates_of_progress,
            ),
        ):
            self.compare(key, actual, (expected * 1000 * 0.1).tolist())
        actual = [rates.amount_rates_mol_s[k] for k in spec.gas_species] + [
            rates.amount_rates_mol_s[f"{slot.spec.slot_id}:{k}"] for k in spec.adsorbates
        ]
        expected = (
            np.concatenate(
                (surface.get_net_production_rates(gas), surface.get_net_production_rates(surface))
            )
            * 1000
            * 0.1
        )
        self.compare("material_derivatives", actual, expected.tolist())
        restored = ImportedSurfaceMechanism.from_dict(json.loads(json.dumps(bundle.to_dict())))
        again = evaluate_catalytic_rates(state, slot, restored.surface)
        self.compare("json_rate_equality", again.forward_rates_mol_s, rates.forward_rates_mol_s)

    def history(self):
        c = self.case
        payload = authored_declaration(c["density"])
        bundle, slot, initial = native(payload, SOURCE, c["temperature"], 1e5, (0.8, 0.1, 0.1))
        spec = bundle.surface
        gas, surface = reference(payload, c["temperature"], 1e5, (0.8, 0.1, 0.1))
        reactor = ct.IdealGasReactor(gas, energy="off", volume=0.001, clone=False)
        rs = ct.ReactorSurface(surface, reactor, A=0.1, clone=False)
        network = ct.ReactorNet([reactor])
        network.rtol, network.atol, network.max_steps = 1e-11, 1e-20, 100000
        system = catalytic_system(spec, (slot,), support_heat_capacity_J_K=1.0, isothermal=True)
        trajectory = system.advance(initial, 1.0, evaluation_times_s=TIMES)
        states = trajectory.trajectory
        # The original coupled API returns sampled physical states, including t0.
        for at, state in zip(TIMES, states, strict=True):
            if at:
                network.advance(at)
            actual = flatten_native(state, spec, slot)
            expected_gas = (reactor.phase.concentrations * 1000 * reactor.volume).tolist()
            expected_surface = (rs.coverages * surface.site_density * 1000 * 0.1).tolist()
            expected = expected_gas + expected_surface
            prefix = f"t={at}:"
            self.compare(prefix + "gas", actual[:2], expected_gas, rtol=3e-5, atol=2e-11)
            self.compare(prefix + "surface", actual[2:], expected_surface, rtol=3e-5, atol=2e-11)
            self.compare(
                prefix + "elements",
                elements(actual, gas, surface),
                elements(expected, gas, surface),
                rtol=2e-8,
                atol=1e-12,
            )
            self.compare(
                prefix + "sites", sum(actual[2:]), 0.1 * c["density"], rtol=2e-8, atol=1e-12
            )
            self.compare(prefix + "T", state.temperature_K, c["temperature"], rtol=0, atol=1e-9)
            self.details.append(
                {"unit": c["name"], "time_s": at, "native_mol": actual, "reference_mol": expected}
            )

    def energy(self, isothermal):
        runtime = ChemWorldRuntime(composition=example_world(isothermal=isothermal))
        try:
            runtime.reset(seed=34)
            initial = runtime._state
            first = runtime.execute({"operation": "wait", "duration_s": 1})
            final = runtime._state
            self.compare("committed", first.info["transaction_status"], "committed")
            self.compare(
                "finite_gas_changed", final.species_amounts != initial.species_amounts, True
            )
            self.compare("T_behavior", final.temperature_K == initial.temperature_K, isothermal)
            gas, surface = reference(authored_declaration(), 700, 1e5, (0.8, 0.1, 0.1))

            def energy(state):
                t = state.temperature_K
                gas.TP, surface.TP = (t, state.pressure_Pa), (t, state.pressure_Pa)
                ns = state.state_slots.slots[0].values
                ng = [state.species_amounts[k] for k in gas.species_names]
                ad = [ns[k] for k in surface.species_names]
                ug = (gas.standard_enthalpies_RT - 1) * ct.gas_constant * t / 1000
                us = surface.standard_enthalpies_RT * ct.gas_constant * t / 1000
                return float(np.dot(ng, ug) + np.dot(ad, us) + 1.0 * (t - 298.15))

            delta = energy(final) - energy(initial)
            q = final.ledger.energy_jacket_J - final.ledger.heat_loss_J
            self.compare("independent_first_law", delta, q, rtol=2e-8, atol=2e-7)
            self.compare("Q_behavior", abs(q) > 1e-8, isothermal)
            slots = final.state_slots.slots[0].values
            self.compare(
                "site_sum", sum(slots[k] for k in ("Pt", "HA", "HB")), 1e-6, rtol=2e-8, atol=1e-12
            )
            self.details.append(
                {
                    "unit": self.case["name"],
                    "delta_U_J": delta,
                    "Q_J": q,
                    "final_temperature_K": final.temperature_K,
                }
            )
        finally:
            runtime.close()

    def snapshot(self):
        runtime, restored = ChemWorldRuntime(composition=example_world()), None
        try:
            runtime.reset(seed=34)
            first = runtime.execute({"operation": "wait", "duration_s": 1})
            self.compare("first_commit", first.info["transaction_status"], "committed")
            snap = json.loads(json.dumps(runtime.snapshot().to_dict()))
            path = self.path / "snapshot.json"
            path.write_text(json.dumps(snap), encoding="utf-8")
            restored = ChemWorldRuntime.from_snapshot(snap)
            for world, label in ((runtime, "original"), (restored, "restored")):
                reply = world.execute({"operation": "wait", "duration_s": 1})
                self.compare(label + "_commit", reply.info["transaction_status"], "committed")
            self.compare("exact_in_process", runtime._state.to_dict(), restored._state.to_dict())
            target = self.path / "fresh-process.json"
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(path),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (self.path / "fresh-process.log").write_text(
                result.stdout + result.stderr, encoding="utf-8"
            )
            self.compare("process_exit", result.returncode, 0, rtol=0, atol=0)
            self.compare(
                "exact_fresh_process",
                json.loads(target.read_text(encoding="utf-8")),
                runtime._state.to_dict(),
            )
        finally:
            runtime.close()
            if restored:
                restored.close()

    def rollback(self):
        runtime = ChemWorldRuntime(composition=example_world())
        try:
            runtime.reset(seed=34)
            first = runtime.execute({"operation": "wait", "duration_s": 1})
            self.compare("first_commit", first.info["transaction_status"], "committed")
            before = runtime._state.to_dict()
            bad = copy.deepcopy(runtime.snapshot().to_dict())
            slot = next(iter(bad["world_state"]["state_slots"].values()))
            slot["values"]["Pt"] *= 0.9
            rejected = False
            try:
                broken = ChemWorldRuntime.from_snapshot(bad)
                broken.close()
            except ValueError:
                rejected = True
            self.compare("incomplete_restore_rejected", rejected, True)
            reply = runtime.execute({"operation": "wait", "duration_s": -1})
            self.compare(
                "invalid_action_rejected", reply.info.get("transaction_status") != "committed", True
            )
            after = runtime._state.to_dict()
            self.compare("material_rollback", after["phases"], before["phases"])
            self.compare("site_rollback", after["state_slots"], before["state_slots"])
            self.compare("clock_rollback", runtime._state.ledger.time_s, 1.0, rtol=0, atol=0)
        finally:
            runtime.close()


def run(path):
    block, started = Block(path), time.monotonic()
    fixed = cases()
    source = source_payload()
    (path / "units.json").write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    (path / "ptcombust-surface-input.json").write_text(
        json.dumps(source, indent=2), encoding="utf-8"
    )
    for i, case in enumerate(fixed, 1):
        block.case = case
        before, error = len(block.rows), None
        try:
            if case["kind"] == "state":
                block.state(
                    payload_for(case, source), SOURCE if case["source"] == "authored" else UPSTREAM
                )
            elif case["kind"] == "history":
                block.history()
            elif case["kind"] in {"isothermal", "adiabatic"}:
                block.energy(case["kind"] == "isothermal")
            elif case["kind"] == "snapshot":
                block.snapshot()
            else:
                block.rollback()
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        selected = block.rows[before:]
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
            f"34B {i}/52 {case['name']} {'PASS' if passed else 'FAIL'} "
            f"comparisons={len(block.rows)}/552 rate={i / max(elapsed, 0.001):.2f} units/s "
            f"ETA={(52 - i) * elapsed / i:.1f}s {error or ''}",
            flush=True,
        )
        with (path / "unit_results.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(block.units[-1]) + "\n")
        with (path / "comparisons.jsonl").open("a", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row) + "\n")
    summary = {
        "schema": "chemworld-surface34b-development-v1",
        "reference": "Cantera " + ct.__version__,
        "planned_units": 52,
        "units_passed": sum(r["passed"] for r in block.units),
        "planned_comparisons": 552,
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(r["passed"] for r in block.rows),
        "unexecuted_comparisons": 552 - len(block.rows),
        "failures": [r for r in block.units if not r["passed"]],
        "failed_comparisons": [r for r in block.rows if not r["passed"]],
        "elapsed_s": time.monotonic() - started,
        "passed": all(r["passed"] for r in block.units) and len(block.rows) == 552,
    }
    (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (path / "details.json").write_text(json.dumps(block.details, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        runtime = ChemWorldRuntime.from_snapshot(
            json.loads(args.resume.read_text(encoding="utf-8"))
        )
        try:
            runtime.execute({"operation": "wait", "duration_s": 1})
            args.output.write_text(json.dumps(runtime._state.to_dict()), encoding="utf-8")
        finally:
            runtime.close()
    else:
        result = run(args.output)
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in {"failures", "failed_comparisons"}},
                indent=2,
            )
        )
        raise SystemExit(0 if result["passed"] else 1)
