"""Fixed 20-unit / 212-comparison development qualification; retains every failure."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from math import exp
from pathlib import Path

import ase
import numpy as np
from hindered_adsorbate53_reference import (
    ADS_INITIAL,
    CAPACITY,
    GAS_INITIAL,
    TIMES,
    R,
    ReferenceThermo,
    cases,
    rate_values,
    reference_history,
    total_energy,
)

from chemworld.examples.native_hindered_adsorbate import (
    example_adsorbate,
    example_mechanism,
    example_world,
)
from chemworld.physchem.catalysis_network import (
    evaluate_catalytic_rates,
    make_catalytic_slot,
    pack_catalytic_state,
)
from chemworld.runtime import ChemWorldRuntime


class Block:
    def __init__(self, path):
        path.mkdir(parents=True, exist_ok=False)
        self.path, self.rows, self.units, self.details, self.case = path, [], [], [], {}

    def compare(self, quantity, actual, expected, *, rtol=0, atol=0):
        exact = isinstance(expected, (bool, str, dict))
        if exact:
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
                "quantity": quantity,
                "passed": passed,
                "actual": actual,
                "expected": expected,
                "rtol": rtol,
                "atol": atol,
                "maximum_absolute_error": error,
            }
        )

    def thermo(self):
        c = self.case
        model = example_adsorbate(c["profile"])
        t = c["temperature"]
        native, reference = model.evaluate(t), ReferenceThermo(c["profile"]).values(t)
        for name, actual, atol, rtol in (
            ("U", native.enthalpy_J_mol, 0.05, 2e-6),
            ("S", native.entropy_J_mol_K, 2e-4, 2e-6),
            ("G", native.gibbs_J_mol, 0.2, 2e-6),
            ("Cv", native.cp_J_mol_K, 2e-3, 1e-5),
            ("ZPE", model.zero_point_energy_J_mol, 0.01, 2e-6),
        ):
            self.compare(name, actual, reference[name], rtol=rtol, atol=atol)
        h = 0.01
        values = [model.evaluate(t + k * h).gibbs_J_mol for k in (-2, -1, 1, 2)]
        negative_derivative = -(values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * h)
        self.compare(
            "fixed_site_G_derivative",
            negative_derivative,
            native.entropy_J_mol_K,
            rtol=2e-6,
            atol=1e-4,
        )
        self.details.append({"unit": c["name"], "reference": reference})

    def rates(self):
        c = self.case
        spec = example_mechanism(binding_J_mol=c["binding"])
        slot = make_catalytic_slot(
            "surface",
            "reactor",
            spec,
            capacities_mol={"A": CAPACITY},
            initial_amounts_mol={"CO_site": ADS_INITIAL},
        )
        state = pack_catalytic_state(
            spec,
            (slot,),
            {"CO": GAS_INITIAL},
            temperature_K=c["temperature"],
            volume_L=1,
        )
        actual = evaluate_catalytic_rates(state, slot, spec)
        expected = rate_values(ReferenceThermo(binding=c["binding"]), c["temperature"], ADS_INITIAL)
        self.compare(
            "delta_G", actual.delta_g_J_mol["adsorb"], expected["delta"], rtol=2e-6, atol=0.2
        )
        self.compare(
            "equilibrium_activity_ratio",
            exp(-actual.delta_g_J_mol["adsorb"] / (R * c["temperature"])),
            expected["equilibrium"],
            rtol=1e-4,
            atol=1e-9,
        )
        forward, reverse = (
            actual.forward_rates_mol_s["adsorb"],
            actual.reverse_rates_mol_s["adsorb"],
        )
        for name, value, ref in (
            ("forward", forward, expected["forward"]),
            ("reverse", reverse, expected["reverse"]),
            ("net", forward - reverse, expected["forward"] - expected["reverse"]),
        ):
            self.compare(name, value, ref, rtol=1e-4, atol=1e-12)
        rate = expected["forward"] - expected["reverse"]
        self.compare(
            "inventory_derivatives",
            [actual.amount_rates_mol_s[k] for k in ("CO", f"{slot.spec.slot_id}:CO_site")],
            [-rate, rate],
            rtol=1e-4,
            atol=1e-12,
        )

    def world(self):
        c = self.case
        expected, evaluations = reference_history(c["temperature"], c["isothermal"])
        runtime = ChemWorldRuntime(
            composition=example_world(
                temperature_K=c["temperature"],
                isothermal=c["isothermal"],
            )
        )
        restored, checkpoint, final, owned = None, None, None, []
        snapshots = self.path / c["name"]
        snapshots.mkdir()
        observed = []
        try:
            runtime.reset(seed=53)
            previous = 0
            for at, ref in zip(TIMES, expected, strict=True):
                if at:
                    reply = runtime.execute({"operation": "wait", "duration_s": at - previous})
                    if reply.info.get("transaction_status") != "committed":
                        raise RuntimeError("world wait failed: " + str(reply.info))
                state = runtime._state
                slot = state.state_slots.slots[0]
                gas, ads = state.species_amounts["CO"], slot.values["CO_site"]
                heat = state.ledger.energy_jacket_J - state.ledger.heat_loss_J
                prefix = f"t={at}:"
                self.compare(prefix + "gas", gas, ref["gas"], rtol=3e-5, atol=2e-11)
                self.compare(prefix + "adsorbed", ads, ref["adsorbed"], rtol=3e-5, atol=2e-11)
                self.compare(
                    prefix + "temperature",
                    state.temperature_K,
                    ref["temperature"],
                    rtol=1e-7,
                    atol=2e-5,
                )
                self.compare(prefix + "Q", heat, ref["Q"], rtol=3e-5, atol=2e-6)
                self.compare(
                    prefix + "elements",
                    [gas + ads, gas + ads],
                    ref["elements"],
                    rtol=1e-9,
                    atol=1e-12,
                )
                owned.append((slot.values["capacity:A"], ads))
                observed.append(
                    {
                        "time_s": at,
                        "gas": gas,
                        "adsorbed": ads,
                        "temperature": state.temperature_K,
                        "Q": heat,
                    }
                )
                if at == 2:
                    checkpoint = json.loads(json.dumps(runtime.snapshot().to_dict()))
                    (snapshots / "checkpoint.json").write_text(
                        json.dumps(checkpoint), encoding="utf-8"
                    )
                previous = at
            final = runtime._state
            reference = ReferenceThermo()
            delta_u = total_energy(reference, final.temperature_K, owned[-1][1]) - total_energy(
                reference,
                c["temperature"],
                ADS_INITIAL,
            )
            heat = final.ledger.energy_jacket_J - final.ledger.heat_loss_J
            self.compare("independent_first_law", delta_u, heat, rtol=3e-5, atol=2e-6)
            self.compare(
                "finite_owned_sites",
                {
                    "capacity": [row[0] for row in owned],
                    "nonnegative_occupied": all(row[1] >= 0 for row in owned),
                    "nonnegative_vacancy": all(row[1] <= row[0] for row in owned),
                },
                {
                    "capacity": [CAPACITY] * 5,
                    "nonnegative_occupied": True,
                    "nonnegative_vacancy": True,
                },
            )
            restored = ChemWorldRuntime.from_snapshot(checkpoint)
            for duration in (2, 4):
                reply = restored.execute({"operation": "wait", "duration_s": duration})
                if reply.info.get("transaction_status") != "committed":
                    raise RuntimeError("restored wait failed: " + str(reply.info))
            self.compare("exact_in_process", restored._state.to_dict(), final.to_dict())
            target = snapshots / "fresh-state.json"
            process = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    str(Path(__file__).resolve()),
                    "--resume",
                    str(snapshots / "checkpoint.json"),
                    "--output",
                    str(target),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            (snapshots / "fresh-process.log").write_text(
                process.stdout + process.stderr, encoding="utf-8"
            )
            fresh = (
                json.loads(target.read_text(encoding="utf-8"))
                if process.returncode == 0
                else {
                    "process_exit": process.returncode,
                    "error": process.stderr,
                }
            )
            self.compare("exact_fresh_process", fresh, final.to_dict())
            self.details.append(
                {
                    "unit": c["name"],
                    "reference_evaluations": evaluations,
                    "reference": expected,
                    "native": observed,
                    "delta_U_J": delta_u,
                }
            )
        finally:
            runtime.close()
            if restored is not None:
                restored.close()


def run(path):
    block, start = Block(path), time.monotonic()
    fixed = cases()
    (path / "units.json").write_text(json.dumps(fixed, indent=2), encoding="utf-8")
    for number, case in enumerate(fixed, 1):
        print(f"53B starting {number}/20 {case['name']}", flush=True)
        block.case = case
        before, error = len(block.rows), None
        try:
            getattr(block, case["kind"])()
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
        selected = block.rows[before:]
        passed = (
            error is None
            and len(selected) == case["expected_comparisons"]
            and all(row["passed"] for row in selected)
        )
        block.units.append(
            {
                **case,
                "passed": passed,
                "error": error,
                "comparisons_executed": len(selected),
                "comparisons_passed": sum(row["passed"] for row in selected),
            }
        )
        elapsed = time.monotonic() - start
        print(
            f"53B {number}/20 {'PASS' if passed else 'FAIL'} comparisons={len(block.rows)}/212 "
            f"rate={number / max(elapsed, 0.001):.2f} units/s "
            f"ETA={(20 - number) * elapsed / number:.1f}s "
            f"{error or ''}",
            flush=True,
        )
        with (path / "units.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(block.units[-1]) + "\n")
        with (path / "comparisons.jsonl").open("a", encoding="utf-8") as handle:
            for row in selected:
                handle.write(json.dumps(row) + "\n")
    summary = {
        "schema": "chemworld-hindered53B-development-v1",
        "reference": "ASE " + ase.__version__,
        "reference_constants": "CODATA2014; physical SI conversion; fixed-site entropy adapter",
        "planned_units": 20,
        "units_passed": sum(row["passed"] for row in block.units),
        "planned_comparisons": 212,
        "comparisons_executed": len(block.rows),
        "comparisons_passed": sum(row["passed"] for row in block.rows),
        "unexecuted_comparisons": 212 - len(block.rows),
        "failures": [row for row in block.units if not row["passed"]],
        "failed_comparisons": [row for row in block.rows if not row["passed"]],
        "elapsed_s": time.monotonic() - start,
        "passed": len(block.rows) == 212 and all(row["passed"] for row in block.units),
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
            for duration in (2, 4):
                reply = runtime.execute({"operation": "wait", "duration_s": duration})
                if reply.info.get("transaction_status") != "committed":
                    raise RuntimeError(str(reply.info))
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
