"""Fixed 20-unit / 156-comparison qualification; no changed selection after failure."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import cantera as ct
import numpy as np
from catalysis54b_reference import (
    CAPACITY,
    analytic,
    cases,
    pt_payload_and_boundary,
    source_reference,
)

from chemworld.examples.native_imported_surface import SOURCE, authored_declaration, example_world
from chemworld.examples.native_surface_steady import (
    branching_declaration,
    branching_mechanism,
    branching_seeds,
)
from chemworld.physchem.catalysis_analysis import (
    CatalyticAnalysisConditions,
    catalytic_steady_states,
    solve_catalytic_steady_state,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.catalysis_analysis import analyze_catalytic_world

POLICY = RootPolicy(tolerance=1e-10, maximum_iterations=80, maximum_backtracks=40)
SOURCE_POLICY = RootPolicy(tolerance=1e-7, maximum_iterations=80, maximum_backtracks=40)


def eig(values):
    if isinstance(values, list) and values and isinstance(values[0], dict):
        values = [complex(v["real"], v["imag"]) for v in values]
    return np.array([[z.real, z.imag] for z in sorted(values, key=lambda z: (z.real, z.imag))])


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class Block:
    def __init__(self, path):
        path.mkdir(parents=True, exist_ok=False)
        self.path, self.rows, self.units, self.details, self.case = path, [], [], [], {}

    def compare(self, quantity, actual, expected, *, rtol=0.0, atol=0.0):
        if isinstance(expected, (str, bool)):
            passed, error = actual == expected, None
        elif isinstance(expected, dict):
            passed = actual == expected
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
                "quantity": quantity,
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
                "rtol": rtol,
                "atol": atol,
                "maximum_absolute_error": error,
            }
        )

    def authored(self):
        c = self.case
        spec = branching_mechanism(disconnected=c["disconnected"])
        seeds = branching_seeds(disconnected=c["disconnected"])
        boundary = CatalyticAnalysisConditions(
            c["temperature"], {"A": c["pressure"], "B": 0.0, "C": 0.0}, {"surface": CAPACITY}
        )
        solved = catalytic_steady_states(
            spec,
            boundary,
            seeds,
            target_species="B",
            policy=POLICY,
            product_species=("B", "C"),
            element_basis="H",
        )
        self.details.append({"unit": c["name"], "native": solved})
        self.compare("status", bool(solved["roots"]), True)
        if not solved["roots"]:
            raise ValueError("no native analytic root")
        root = solved["roots"][0]
        ref = analytic(c["disconnected"], c["temperature"], c["pressure"])
        for quantity, actual, expected, atol in (
            (
                "coverages",
                [root["coverages"][k] for k in spec.adsorbates],
                [ref["coverages"][k] for k in spec.adsorbates],
                2e-10,
            ),
            (
                "gas",
                [root["gas_rates_mol_s"][k] for k in spec.gas_species],
                [ref["gas"][k] for k in spec.gas_species],
                2e-14,
            ),
            ("forward", list(root["forward_rates_mol_s"].values()), ref["forward"], 2e-14),
            ("reverse", list(root["reverse_rates_mol_s"].values()), ref["reverse"], 2e-14),
        ):
            self.compare(quantity, actual, expected, rtol=2e-7, atol=atol)
        self.compare(
            "spectrum",
            eig(root["eigenvalues_s_inv"]),
            eig(ref["eigenvalues"]),
            rtol=2e-5,
            atol=2e-5,
        )
        self.compare(
            "selectivity",
            [root["selectivity"]["fractions"][k] for k in ("B", "C")],
            [ref["selectivity"][k] for k in ("B", "C")],
            rtol=2e-7,
            atol=2e-10,
        )
        self.compare("stability", root["stability"], "stable")
        self.compare(
            "invariant_totals",
            root["conserved_totals_at_root_mol"],
            root["steady_chart"]["conserved_totals_mol"],
            atol=2e-16,
        )
        self.compare("same_class_seed_successes", solved["converged_attempts"], 3)

    def source(self):
        c = self.case
        payload, pressures = pt_payload_and_boundary(c["temperature"], c["feed"])
        spec = convert_surface_mechanism(
            payload,
            surface_phase="Pt_surf",
            source=SOURCE,
            minimum_temperature_K=300,
            maximum_temperature_K=1900,
        ).surface
        capacity = payload["phases"][1]["site-density"] * 1000 * 0.1
        seed = dict.fromkeys(spec.adsorbates, 1 / len(spec.adsorbates))
        boundary = CatalyticAnalysisConditions(c["temperature"], pressures, {"Pt_surf": capacity})
        root = solve_catalytic_steady_state(
            spec, boundary, seed, target_species="CO2", policy=SOURCE_POLICY
        )
        self.details.append({"unit": c["name"], "native": root})
        self.compare("status", root["status"], "converged")
        ref = source_reference(payload, c["temperature"], pressures, seed)
        self.details[-1]["reference_residual_s_inv"] = ref["residual"]
        self.details[-1]["reference"] = {**ref, "eigenvalues": eig(ref["eigenvalues"])}
        if root["status"] != "converged":
            raise ValueError("native source root did not converge")
        self.compare(
            "coverages",
            [root["coverages"][k] for k in spec.adsorbates],
            [ref["coverages"][k] for k in spec.adsorbates],
            rtol=2e-4,
            atol=2e-7,
        )
        for quantity, actual, expected in (
            (
                "gas",
                [root["gas_rates_mol_s"][k] for k in spec.gas_species],
                [ref["gas"][k] for k in spec.gas_species],
            ),
            ("forward", list(root["forward_rates_mol_s"].values()), ref["forward"]),
            ("reverse", list(root["reverse_rates_mol_s"].values()), ref["reverse"]),
        ):
            self.compare(quantity, actual, expected, rtol=2e-3, atol=2e-10)
        self.compare(
            "spectrum", eig(root["eigenvalues_s_inv"]), eig(ref["eigenvalues"]), rtol=5e-3, atol=1.0
        )
        self.compare(
            "physical_residuals",
            [max(abs(v) for v in root["surface_residual_s_inv"]), max(abs(ref["residual"]))],
            [0.0, 0.0],
            atol=1e-6,
        )
        self.compare(
            "site_total", sum(root["adsorbate_amounts_mol"].values()), capacity, atol=2e-16
        )

    def structural(self):
        c = self.case
        spec = branching_mechanism(disconnected=c.get("disconnected", False))
        seed = branching_seeds(disconnected=c.get("disconnected", False))[0]
        boundary = CatalyticAnalysisConditions(
            700, {"A": 5e4, "B": 0.0, "C": 0.0}, {"surface": CAPACITY}
        )
        if c["kind"] == "basis":
            a = solve_catalytic_steady_state(
                spec, boundary, seed, target_species="B", policy=POLICY
            )
            b = solve_catalytic_steady_state(
                spec,
                boundary,
                seed,
                target_species="B",
                policy=POLICY,
                coordinate_order=tuple(reversed(spec.adsorbates)),
            )
            self.compare(
                "same_root",
                [a["coverages"][k] for k in spec.adsorbates],
                [b["coverages"][k] for k in spec.adsorbates],
                rtol=2e-7,
                atol=2e-10,
            )
            self.compare(
                "same_spectrum",
                eig(a["eigenvalues_s_inv"]),
                eig(b["eigenvalues_s_inv"]),
                rtol=2e-5,
                atol=2e-5,
            )
            self.compare(
                "rank", a["steady_chart"]["reachable_dimension"], 4 if c["disconnected"] else 3
            )
            self.compare(
                "same_totals",
                a["conserved_totals_at_root_mol"],
                b["conserved_totals_at_root_mol"],
                atol=2e-16,
            )
        elif c["kind"] == "incompatible":
            altered = {**seed, "I": 0.11, "E": seed["E"] - 0.01}
            before = deepcopy(seed)
            reason = ""
            try:
                catalytic_steady_states(spec, boundary, [seed, altered], target_species="B")
            except ValueError as exc:
                reason = str(exc)
            self.compare("same_site_total", sum(seed.values()), sum(altered.values()), atol=1e-15)
            self.compare("rejected_class", "compatibility class" in reason, True)
            self.compare("initial_unchanged", seed, before)
            root = solve_catalytic_steady_state(
                spec, boundary, seed, target_species="B", policy=POLICY
            )
            self.compare("single_class_status", root["status"], "converged")
        else:
            payload = branching_declaration()
            payload["reactions"] = payload["reactions"][:1]
            spec = convert_surface_mechanism(
                payload,
                surface_phase="surface",
                source=SOURCE,
                minimum_temperature_K=300,
                maximum_temperature_K=1900,
            ).surface
            boundary = CatalyticAnalysisConditions(
                700, {"A": 0.0, "B": 0.0, "C": 0.0}, {"surface": CAPACITY}
            )
            root = solve_catalytic_steady_state(
                spec,
                boundary,
                seed,
                target_species="B",
                policy=POLICY,
                product_species=("B", "C"),
                element_basis="H",
            )
            self.compare("rank", root["steady_chart"]["reachable_dimension"], 1)
            self.compare(
                "zero_mode_retained",
                root["stability"] == "marginal_or_unresolved"
                and root["eigenvalues_s_inv"] == [{"real": 0.0, "imag": 0.0}],
                True,
            )
            self.compare("zero_selectivity", root["selectivity"]["status"], "unavailable")
            self.compare("initial_preserved", root["coverages"], seed)

    def world(self):
        composition = example_world(isothermal=self.case["isothermal"])
        runtime = ChemWorldRuntime(composition=composition)
        try:
            runtime.reset(seed=34)
            reply = runtime.execute({"operation": "wait", "duration_s": 1})
            if reply.info["transaction_status"] != "committed":
                raise ValueError("actual finite-world wait did not commit")
            before = deepcopy(runtime.snapshot().to_dict())
            state = runtime._state
            ledger, amounts = deepcopy(state.ledger.to_dict()), deepcopy(state.to_dict())
            root = analyze_catalytic_world(composition, state, target_species="B", policy=POLICY)
            self.details.append({"unit": self.case["name"], "native": root})
            self.compare("status", root["status"], "converged")
            ref = source_reference(
                authored_declaration(),
                state.temperature_K,
                root["conditions"]["gas_partial_pressures_Pa"],
                root["initial_coverages"],
            )
            self.compare(
                "reference_coverages",
                [root["coverages"][k] for k in ref["coverages"]],
                list(ref["coverages"].values()),
                rtol=2e-4,
                atol=2e-7,
            )
            self.compare("snapshot_unchanged", runtime.snapshot().to_dict(), before)
            self.compare("ledger_unchanged", state.ledger.to_dict(), ledger)
            self.compare("inventory_unchanged", state.to_dict(), amounts)
            self.compare(
                "read_only_boundary",
                root["world_query"]["boundary"],
                "read-only; held-current-gas-pressure surface analysis",
            )
        finally:
            runtime.close()

    def run(self, selected):
        started = time.monotonic()
        for i, case in enumerate(selected, 1):
            self.case = case
            offset = len(self.rows)
            error = None
            try:
                if case["kind"] == "analytic":
                    self.authored()
                elif case["kind"] == "source":
                    self.source()
                elif case["kind"] == "world":
                    self.world()
                else:
                    self.structural()
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            completed = self.rows[offset:]
            passed = (
                error is None
                and len(completed) == case["expected_comparisons"]
                and all(r["passed"] for r in completed)
            )
            self.units.append(
                {
                    **case,
                    "passed": passed,
                    "completed_comparisons": len(completed),
                    "failure": error,
                }
            )
            elapsed = time.monotonic() - started
            print(
                f"54B {i}/{len(selected)} {case['name']} {'PASS' if passed else 'FAIL'} "
                f"throughput={i / max(elapsed, 1e-9):.3f} units/s "
                f"ETA={(len(selected) - i) * elapsed / i:.1f}s",
                flush=True,
            )
            self.save(started)

    def save(self, started):
        summary = {
            "schema": "chemworld-catalysis54b-development-v1",
            "cantera_version": ct.__version__,
            "units_total": len(self.units),
            "units_passed": sum(u["passed"] for u in self.units),
            "comparisons_expected": sum(u["expected_comparisons"] for u in self.units),
            "comparisons_executed": len(self.rows),
            "comparisons_passed": sum(r["passed"] for r in self.rows),
            "elapsed_s": time.monotonic() - started,
            "units": self.units,
            "failed_comparisons": [r for r in self.rows if not r["passed"]],
        }
        for name, value in (
            ("summary.json", summary),
            ("comparisons.json", self.rows),
            ("details.json", self.details),
        ):
            (self.path / name).write_text(
                json.dumps(value, indent=2, default=plain), encoding="utf8"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    selected = cases()
    if args.canary:
        selected = [selected[i] for i in (0, 8, 18)]
    Block(args.output).run(selected)
