"""Fixed finite-calcite kinetic block; retain every attempt and failed unit."""

import argparse
import json
import os
import sys
import traceback
from dataclasses import replace
from itertools import product
from math import log
from pathlib import Path
from time import perf_counter

from check_native_aqueous import compare
from native_mineral_reference import ReaktoroMineralReference
from scipy.optimize import brentq

from chemworld.physchem.aqueous_mineral import (
    CarbonateEquilibriumSpec,
    carbonate_feed,
    element_amounts,
    solve_carbonate_equilibrium,
)
from chemworld.physchem.mineral_kinetics import (
    MineralGeometry,
    MineralKineticsSpec,
    advance_mineral,
)

TIMES = (0, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6, 4e6)


def prepare(aqueous, target):
    calls = []

    def feed(c):
        return carbonate_feed(
            water_kg=1,
            sodium_bicarbonate_mol=c,
            calcium_chloride_mol=0.001,
            sodium_chloride_mol=0.01,
            calcite_mol=0.01,
        )

    def residual(c):
        result = solve_carbonate_equilibrium(aqueous, feed(c), fixed_calcite_mol=0.01)
        calls.append(
            {
                "NaHCO3_mol": c,
                "Omega": result.saturation_ratio,
                "pH": result.pH,
                "ionic_strength_mol_kg": result.ionic_strength_mol_kg,
            }
        )
        return result.saturation_ratio - target

    c = brentq(residual, 0, 0.03, xtol=1e-15, rtol=1e-14)
    result = solve_carbonate_equilibrium(aqueous, feed(c), fixed_calcite_mol=0.01)
    return {
        "target_Omega": target,
        "NaHCO3_mol": c,
        "input_amounts_mol": feed(c),
        "initial": result.to_dict(),
        "root_evaluations": calls,
    }


def checks_for_point(feed, point, reference):
    state = point.aqueous
    original = element_amounts(feed)
    actual = element_amounts(state.amounts_mol)
    element_error = max(abs(actual[k] - n) / max(n, 1e-12) for k, n in original.items())
    mass_action = max(abs(v) for v in state.mass_action_log10_residuals.values() if v is not None)
    _, metrics = compare(state, reference["aqueous"])
    area_error = abs(point.area_m2 - reference["area_m2"]) / max(
        point.area_m2, reference["area_m2"], 1e-12
    )
    checks = {
        "nonnegative": min(state.amounts_mol.values()) >= 0 and point.area_m2 >= 0,
        "elements_and_carbon": element_error <= 1e-10,
        "charge": abs(state.charge_balance_error_eq) <= 1e-11,
        "aqueous_mass_action": mass_action <= 1e-8,
        "ionic_domain": state.ionic_strength_mol_kg <= 0.1,
        "reference_pH": metrics["pH_absolute_difference"] <= 5e-4,
        "reference_molality": max(metrics["species_molality_relative_differences"].values())
        <= 1e-3,
        "reference_mineral": metrics["mineral_absolute_difference_mol"]
        <= 1e-8 + 5e-5 * max(state.calcite_mol, reference["aqueous"]["amounts_mol"]["Calcite"]),
        "reference_area": area_error <= 1e-3,
    }
    metrics.update(
        normalized_element_residual=element_error,
        mass_action_log10_residual=mass_action,
        area_relative_difference=area_error,
    )
    return {
        "time_s": point.time_s,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
    }


def early_limit(model, aqueous, feed, geometry):
    report = advance_mineral(model, aqueous, feed, geometry, 0.01)
    initial, final = report.trajectory[0].aqueous, report.final_aqueous
    n, omega, dt = feed["Calcite"], initial.saturation_ratio, 0.01
    extent = final.calcite_mol - n
    tangent = -geometry.area(n) * model.rate_constant_mol_m2_s * (1 - omega) * dt
    radius = (
        n ** (1 / 3)
        - geometry.reference_area_m2
        * model.rate_constant_mol_m2_s
        / (3 * geometry.reference_amount_mol ** (2 / 3))
        * (1 - omega)
        * dt
    ) ** 3 - n
    linear_error, radius_error = (abs(extent - x) / max(abs(x), 1e-12) for x in (tangent, radius))
    passed = (
        abs(extent) <= 1e-12
        if abs(omega - 1) <= model.saturation_tolerance
        else linear_error <= 1e-4 and radius_error <= 1e-4
    )
    return {
        "time_s": dt,
        "actual_extent_mol": extent,
        "constant_area_extent_mol": tangent,
        "constant_drive_radius_extent_mol": radius,
        "linear_relative_error": linear_error,
        "radius_relative_error": radius_error,
        "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--reaktoro-prefix", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workstreams/world_foundation/NATIVE_MINERAL_KINETICS_RESULTS.json"),
    )
    args = parser.parse_args()
    data = json.loads(args.output.read_text(encoding="utf-8"))
    block = {
        "kind": "fixed_12_trajectory_block",
        "attempt": len(data["attempts"]) + 1,
        "trajectory_denominator": 12,
        "point_denominator": 120,
        "early_limit_denominator": 9,
        "times_s": TIMES,
        "trajectories": [],
        "preparations": [],
        "failures": [],
        "complete": False,
    }
    data["attempts"].append(block)

    def save():
        args.output.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save()
    aqueous, model = CarbonateEquilibriumSpec(), MineralKineticsSpec()
    start = perf_counter()
    try:
        for target in (0.1, 1, 10):
            unit = prepare(aqueous, target)
            block["preparations"].append(unit)
            save()
            print(f"Prepared Omega={target:g}, NaHCO3={unit['NaHCO3_mol']:.12g} mol", flush=True)
    except Exception as exc:
        block.update(preparation_error=repr(exc), traceback=traceback.format_exc())
        save()
        raise
    prepared = {u["target_Omega"]: u["input_amounts_mol"] for u in block["preparations"]}
    cases = [
        (f"area-{a:g}-Omega-{omega:g}", model, MineralGeometry(a, 0.01), prepared[omega], "grid")
        for a, omega in product((0.0025, 0.01, 0.04), (0.1, 1, 10))
    ]
    cases.extend(
        [
            (
                "finite-exhaustion",
                model,
                MineralGeometry(0.04, 0.01),
                carbonate_feed(water_kg=1, calcite_mol=1e-6),
                "exhaustion",
            ),
            (
                "supersaturated-no-seed",
                model,
                MineralGeometry(0, 0),
                {**prepared[10], "Calcite": 0.0},
                "seedless",
            ),
            (
                "zero-rate-hold",
                replace(model, rate_constant_mol_m2_s=0),
                MineralGeometry(),
                prepared[0.1],
                "inert",
            ),
        ]
    )
    reference = ReaktoroMineralReference(aqueous, args.reaktoro_prefix)
    for index, (name, kinetics, geometry, feed, kind) in enumerate(cases, 1):
        unit = {
            "id": name,
            "kind": kind,
            "kinetic_spec": kinetics.to_dict(),
            "geometry": geometry.to_dict(),
            "input_amounts_mol": feed,
            "points": [],
        }
        began = perf_counter()
        print(f"Mineral {index}/12 starting {name}", flush=True)
        try:
            native = advance_mineral(
                kinetics, aqueous, feed, geometry, TIMES[-1], evaluation_times_s=TIMES
            )
            unit["native"] = native.to_dict()
            ref = reference.advance(
                feed, kinetics.rate_constant_mol_m2_s, geometry.to_dict(), TIMES
            )
            unit["reference"] = ref
            for a, b in zip(native.trajectory, ref["trajectory"], strict=True):
                unit["points"].append(checks_for_point(feed, a, b))
            extra = {"clock": native.actual_duration_s == TIMES[-1]}
            if kind == "grid":
                early = early_limit(kinetics, aqueous, feed, geometry)
                unit["early_limit"] = early
                exact = solve_carbonate_equilibrium(aqueous, feed)
                end = native.final_aqueous
                unit["equilibrium_endpoint"] = exact.to_dict()
                extra.update(
                    early_limit=early["passed"],
                    equilibrium_mineral=abs(end.calcite_mol - exact.calcite_mol)
                    <= 1e-8 + 1e-5 * max(end.calcite_mol, exact.calcite_mol),
                    equilibrium_pH=abs(end.pH - exact.pH) <= 1e-5,
                    equilibrium_omega=end.saturation_ratio > 0
                    and abs(log(end.saturation_ratio)) <= 1e-8,
                )
            elif kind == "exhaustion":
                extra.update(
                    exhausted=native.final_aqueous.calcite_mol == 0,
                    area_removed=native.trajectory[-1].area_m2 == 0,
                )
            elif kind == "seedless":
                extra.update(
                    no_seed=all(
                        p.aqueous.calcite_mol == 0 and p.area_m2 == 0 for p in native.trajectory
                    )
                )
            else:
                extra.update(
                    zero_rate_hold=all(
                        p.aqueous.calcite_mol == feed["Calcite"] for p in native.trajectory
                    )
                )
            unit.update(
                extra_checks=extra,
                passed=all(p["passed"] for p in unit["points"]) and all(extra.values()),
            )
        except Exception as exc:
            unit.update(passed=False, exception=repr(exc), traceback=traceback.format_exc())
        unit["elapsed_s"] = perf_counter() - began
        block["trajectories"].append(unit)
        if not unit["passed"]:
            block["failures"].append(name)
        save()
        elapsed = perf_counter() - start
        print(
            f"Mineral {index}/12 {name}: {'PASS' if unit['passed'] else 'FAIL'}; "
            f"{index / elapsed:.3f} trajectories/s; ETA {elapsed / index * (12 - index):.1f}s",
            flush=True,
        )
    points = [p for u in block["trajectories"] for p in u["points"]]
    early = [u["early_limit"] for u in block["trajectories"] if "early_limit" in u]
    block.update(
        complete=True,
        passed_trajectories=sum(u["passed"] for u in block["trajectories"]),
        completed_point_comparisons=len(points),
        passed_points=sum(p["passed"] for p in points),
        failed_points=sum(not p["passed"] for p in points),
        not_completed_points=120 - len(points),
        passed_early_limits=sum(p["passed"] for p in early),
        elapsed_s=perf_counter() - start,
    )
    save()
    print(
        json.dumps(
            {
                k: block[k]
                for k in (
                    "attempt",
                    "passed_trajectories",
                    "passed_points",
                    "failed_points",
                    "not_completed_points",
                    "passed_early_limits",
                    "failures",
                    "elapsed_s",
                )
            }
        ),
        flush=True,
    )
    code = int(bool(block["failures"]))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)  # see the documented Reaktoro Windows callback-finalization boundary


if __name__ == "__main__":
    main()
