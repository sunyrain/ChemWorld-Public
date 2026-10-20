"""Fixed NATIVE-EXT-32 development block; append every attempt to one summary."""

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from itertools import product
from math import log10, sqrt
from pathlib import Path
from time import perf_counter

from native_aqueous_reference import ReaktoroCarbonateReference

from chemworld.physchem.aqueous_mineral import (
    WATER_MOLAR_MASS_KG_MOL,
    CarbonateEquilibriumSpec,
    carbonate_feed,
    element_amounts,
    solve_carbonate_equilibrium,
)


def conditions():
    model = CarbonateEquilibriumSpec()
    result = [
        (
            f"grid-C{c:g}-Ca{ca:g}-NaCl{salt:g}",
            model,
            carbonate_feed(
                water_kg=1,
                sodium_bicarbonate_mol=c,
                calcium_chloride_mol=ca,
                sodium_chloride_mol=salt,
            ),
            False,
        )
        for c, ca, salt in product((1e-4, 1e-3, 1e-2), (1e-4, 1e-3, 1e-2), (0, 0.01, 0.05))
    ]
    for reagent in ("hydrochloric_acid_mol", "sodium_hydroxide_mol"):
        result.append(
            (
                reagent,
                model,
                carbonate_feed(
                    water_kg=1,
                    sodium_bicarbonate_mol=0.001,
                    calcium_chloride_mol=0.001,
                    sodium_chloride_mol=0.01,
                    **{reagent: 0.001},
                ),
                False,
            )
        )
    result.append(
        (
            "mineral-excluded",
            replace(model, allow_calcite=False),
            carbonate_feed(
                water_kg=1,
                sodium_bicarbonate_mol=0.01,
                calcium_chloride_mol=0.01,
                sodium_chloride_mol=0.01,
            ),
            False,
        )
    )
    for n in (1e-6, 0.001):
        result.append(
            (f"finite-mineral-{n:g}", model, carbonate_feed(water_kg=1, calcite_mol=n), False)
        )
    result.append(
        ("ionic-domain-rejection", model, carbonate_feed(water_kg=1, sodium_chloride_mol=0.2), True)
    )
    return result


def native_checks(model, initial, result):
    totals = element_amounts(initial)
    normalized = max(
        abs(v) / max(abs(totals[k]), 1e-12) for k, v in result.element_residuals_mol.items()
    )
    residual = max(abs(v) for v in result.mass_action_log10_residuals.values() if v is not None)
    return {
        "element_and_carbon": normalized <= 1e-10,
        "charge": abs(result.charge_residual_eq) <= 1e-11,
        "mass_action": residual <= 1e-8,
        "nonnegative": min(result.amounts_mol.values()) >= 0,
        "complementarity": (
            not model.allow_calcite
            or (
                abs(result.saturation_ratio - 1) <= 1e-8
                if result.calcite_mol > 0
                else result.saturation_ratio <= 1 + 1e-8
            )
        ),
        "ionic_domain": result.ionic_strength_mol_kg <= model.maximum_ionic_strength_mol_kg,
    }, {
        "maximum_normalized_element_residual": normalized,
        "maximum_mass_action_log10_residual": residual,
    }


def compare(native, reference):
    amounts, water = reference["amounts_mol"], reference["solvent_mass_kg"]
    species_errors = {}
    for k, n in amounts.items():
        if k == "Calcite":
            continue
        a, b = native.amounts_mol[k] / native.solvent_mass_kg, n / water
        species_errors[k] = abs(a - b) / max(abs(a), abs(b), 1e-12)
    pH_error = abs(native.pH - reference["pH"])
    ionic_error = abs(native.ionic_strength_mol_kg - reference["ionic_strength_mol_kg"]) / max(
        native.ionic_strength_mol_kg, reference["ionic_strength_mol_kg"], 1e-12
    )
    mineral_error = abs(native.calcite_mol - amounts["Calcite"])
    return {
        "reference_pH": pH_error <= 1e-5,
        "reference_species": max(species_errors.values()) <= 2e-5,
        "reference_ionic_strength": ionic_error <= 1e-6,
        "reference_mineral": mineral_error
        <= 1e-8 + 1e-5 * max(native.calcite_mol, amounts["Calcite"]),
    }, {
        "pH_absolute_difference": pH_error,
        "species_molality_relative_differences": species_errors,
        "ionic_strength_relative_difference": ionic_error,
        "mineral_absolute_difference_mol": mineral_error,
    }


def analytical():
    outcomes = []
    for name in ("water", "spectator", "acid", "base"):
        model = CarbonateEquilibriumSpec(davies_A=0 if name in {"acid", "base"} else 0.509)
        feed = carbonate_feed(
            water_kg=1,
            calcium_chloride_mol=0.001 if name == "spectator" else 0,
            hydrochloric_acid_mol=0.001 if name == "acid" else 0,
            sodium_hydroxide_mol=0.001 if name == "base" else 0,
        )
        if name in {"acid", "base"}:
            # Exact finite-water quadratic, derived independently from charge
            # and water dissociation. The minor ion consumes actual water.
            kw, mw, excess = 1e-14, WATER_MOLAR_MASS_KG_MOL, 0.001
            a, b = 1 - kw * mw * mw, excess + 2 * kw * mw
            minor = 2 * kw / (b + sqrt(b * b + 4 * a * kw))
            water = 1 - mw * minor
            h = (excess + minor) / water if name == "acid" else minor / water
            expected = -log10(h)
        else:
            expected = 7
        result = solve_carbonate_equilibrium(model, feed)
        error = abs(result.pH - expected)
        outcomes.append(
            {
                "id": name,
                "pH": result.pH,
                "expected_pH": expected,
                "absolute_error": error,
                "passed": error <= 1e-8,
            }
        )
    return outcomes


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reaktoro-prefix", type=Path)
    parser.add_argument("--without-reference", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workstreams/world_foundation/NATIVE_AQUEOUS_MINERAL_RESULTS.json"),
    )
    args = parser.parse_args()
    summary = (
        json.loads(args.output.read_text(encoding="utf-8"))
        if args.output.exists()
        else {
            "schema": "native-aqueous-development-v1",
            "mode": "development",
            "acceptance_note": "NATIVE_AQUEOUS_MINERAL_ACCEPTANCE.md",
            "attempts": [],
        }
    )
    block = {
        "kind": "fixed_33_condition_block",
        "attempt": len(summary["attempts"]) + 1,
        "denominator": 33,
        "reference_required": not args.without_reference,
        "conditions": [],
        "failures": [],
        "complete": False,
    }
    summary["attempts"].append(block)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save()
    cache = {}
    start = perf_counter()
    for index, (name, model, feed, reject) in enumerate(conditions(), 1):
        unit = {
            "id": name,
            "input_amounts_mol": feed,
            "spec": model.to_dict(),
            "expected_rejection": reject,
        }
        began = perf_counter()
        try:
            if reject:
                try:
                    solve_carbonate_equilibrium(model, feed)
                except ValueError as exc:
                    unit["checks"] = {"ionic_domain_rejected": "ionic-strength" in str(exc)}
                    unit["rejection"] = str(exc)
                else:
                    unit["checks"] = {"ionic_domain_rejected": False}
            else:
                native = solve_carbonate_equilibrium(model, feed)
                checks, metrics = native_checks(model, feed, native)
                unit.update(native=native.to_dict(), checks=checks, metrics=metrics)
                if not args.without_reference:
                    if model.allow_calcite not in cache:
                        cache[model.allow_calcite] = ReaktoroCarbonateReference(
                            model, args.reaktoro_prefix
                        )
                    reference = cache[model.allow_calcite].solve(feed)
                    compared, differences = compare(native, reference)
                    unit["reference"] = reference
                    checks.update(compared)
                    metrics.update(differences)
            unit["passed"] = all(unit["checks"].values())
        except Exception as exc:
            unit.update(passed=False, exception=repr(exc), traceback=traceback.format_exc())
        unit["elapsed_s"] = perf_counter() - began
        block["conditions"].append(unit)
        if not unit["passed"]:
            block["failures"].append(name)
        save()
        elapsed = perf_counter() - start
        print(
            f"carbonate {index}/33 {name}: {'PASS' if unit['passed'] else 'FAIL'}; "
            f"{index / max(elapsed, 1e-9):.2f} conditions/s; "
            f"ETA {elapsed / index * (33 - index):.1f}s",
            flush=True,
        )
    block["analytical_limits"] = analytical()
    block["analytical_denominator"] = 4
    block["failures"].extend(
        "analytical:" + u["id"] for u in block["analytical_limits"] if not u["passed"]
    )
    block.update(
        complete=True,
        elapsed_s=perf_counter() - start,
        passed_conditions=sum(u["passed"] for u in block["conditions"]),
        passed_analytical=sum(u["passed"] for u in block["analytical_limits"]),
    )
    save()
    code = int(bool(block["failures"]))
    print(
        json.dumps(
            {
                k: block[k]
                for k in (
                    "attempt",
                    "passed_conditions",
                    "denominator",
                    "passed_analytical",
                    "analytical_denominator",
                    "failures",
                    "elapsed_s",
                )
            }
        ),
        flush=True,
    )
    if cache:
        # The Windows 2.13.0 Python activity callback remains retained inside
        # the extension at shutdown. Results are flushed above; avoid its
        # observed finalization-only GIL crash. Never mask a computation error.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
