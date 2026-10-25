"""Frozen 32-unit molecular/solid bridge qualification; every output is create-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from traceback import format_exc

from chemworld.physchem.calphad_equilibrium import calphad_tp_equilibrium
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_flash import MolecularDensityBasin
from chemworld.physchem.molecular_solid_boundary import (
    MolecularElementReferenceShift,
    MolecularSolidBoundaryPath,
    MolecularSolidBoundarySpec,
    evaluate_molecular_solid_boundary,
    solve_molecular_solid_boundary_path,
)
from chemworld.physchem.multifluid import MultifluidBinaryPair, MultifluidModel
from reference.calphad_equilibrium_models import ELEMENTS, native_spec, solution
from reference.molecular28_solid_reference import PREF, SHIFTS, TREF, R, cases, mapping, reference


def declaration(case):
    atoms = mapping(case)
    ids = tuple(atoms)
    fluids = tuple(
        HelmholtzFluid(
            k,
            "",
            0.055845,
            R,
            TREF,
            PREF / (R * TREF),
            100.0,
            2000.0,
            1e9,
            (
                HelmholtzTerm("IdealGasHelmholtzLead", {"a1": -1.0, "a2": 0.0}),
                HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5}),
            ),
            (),
            "authored parent28 S ideal-gas equation",
        )
        for k in ids
    )
    model = MultifluidModel(
        fluids,
        tuple(
            MultifluidBinaryPair((a, b), 1.0, 1.0, 1.0, 1.0, 0.0, (), "explicit ideal residual")
            for i, a in enumerate(ids)
            for b in ids[i + 1 :]
        ),
        R,
        "source_equal",
        "parent28 S independent same-equation comparison",
    )
    elements = tuple(sorted({e for row in atoms.values() for e in row}))
    phase = "PURE" if len(elements) == 1 else "SOLID"
    tdb = ELEMENTS + solution(phase, elements, (0,) * len(elements))
    if case["kind"] == "nonideal":
        tdb += f"\nPARAMETER L(SOLID,FE,NI;0) 100 {case['omega']}; 2500 N !\n"
    solids = native_spec({"tdb": tdb, "phases": [phase]})
    shifts = tuple(
        MolecularElementReferenceShift(
            e,
            *(SHIFTS[e] if case.get("shifted") else (0.0, 0.0, 0.0)),
            PREF,
            "authored explicit S reference conversion",
        )
        for e in elements
    )
    return MolecularSolidBoundarySpec(
        model.caloric_model(), solids, atoms, shifts, "parent28 frozen S interface reference"
    )


def compare(case):
    expected, spec = reference(case), declaration(case)
    record = {"case": case, "reference": expected, "checks": {}}
    checks = record["checks"]

    def check(name, error, tolerance):
        checks[name] = {
            "error": float(error),
            "tolerance": tolerance,
            "pass": bool(error <= tolerance),
        }

    if case["kind"] == "path":
        basin = MolecularDensityBasin("gas", (1.0, 1000.0), 40.0)
        path = (
            MolecularSolidBoundaryPath(250.0, 350.0, PREF, PREF, basin)
            if case["coordinate"] == "T"
            else MolecularSolidBoundaryPath(300.0, 300.0, 0.5 * PREF, 2 * PREF, basin)
        )
        out = solve_molecular_solid_boundary_path(spec, case["amounts"], path, phase_name="PURE")
        record["native"] = out
        check("success", not out["success"], 0)
        if out["boundary"]:
            check("T_K", abs(out["boundary"]["temperature_K"] - 300.0), 2e-6)
            check("P_Pa", abs(out["boundary"]["pressure_Pa"] - PREF), 1e-4)
            check("tangent", abs(out["tangent_distance_J_mol_atoms"]), 2e-5)
    else:
        result = evaluate_molecular_solid_boundary(
            spec, case["T"], case["P"] / (R * case["T"]), case["amounts"]
        )
        record["native"] = result.to_dict()
        for k, value in expected["molecular"].items():
            check(f"molecular_mu:{k}", abs(result.molecular_potentials_J_mol[k] - value), 2e-8)
        for e, value in expected["stock"].items():
            check(f"stock:{e}", abs(result.element_amounts_mol[e] - value), 1e-12)
        if expected["potentials"] is None:
            check(
                "refusal",
                result.success
                or result.failure_reason != "molecular_state_not_reaction_compatible",
                0,
            )
            check("no_invented_tangent", result.element_potentials_J_mol_atoms is not None, 0)
        else:
            check("success", not result.success, 0)
            if result.success:
                for e, value in expected["potentials"].items():
                    check(
                        f"element_mu:{e}",
                        abs(result.element_potentials_J_mol_atoms[e] - value),
                        2e-8,
                    )
                check(
                    "minimum_distance",
                    abs(result.minimum_distance_J_mol_atoms - expected["minimum"]),
                    2e-5,
                )
                candidate = min(result.candidates, key=lambda c: c.tangent_distance_J_mol_atoms)
                error = min(
                    max(abs(candidate.element_fractions.get(e, 0) - x) for e, x in xs.items())
                    for xs in expected["candidate_compositions"]
                )
                check("candidate_composition", error, 2e-6)
            if case["kind"] == "support":
                n = case["amounts"]["b"]
                if n == 0:
                    check(
                        "exact_absence",
                        result.element_amounts_mol["NI"] != 0
                        or "b" in result.molecular_potentials_J_mol,
                        0,
                    )
                else:
                    check(
                        "trace_stock_relative", abs(result.element_amounts_mol["NI"] - n) / n, 1e-12
                    )
            if case["kind"] == "consumer" and result.success:
                seeds = tuple(c.calphad_seed() for c in result.candidates)
                check(
                    "zero_seed_allocation",
                    any(p.amount_mol_atoms != 0 or p.fraction != 0 for p in seeds),
                    0,
                )
                solid = calphad_tp_equilibrium(
                    spec.solids,
                    case["T"],
                    case["P"],
                    result.element_amounts_mol,
                    initial_phases=seeds,
                )
                record["original42"] = solid.to_dict()
                check("consumer_success", not solid.success, 0)
                check(
                    "seed_accepted",
                    not any(a["accepted"] for a in solid.diagnostics["warm_seed_attempts"]),
                    0,
                )
                if solid.success:
                    for e, value in expected["stock"].items():
                        check(
                            f"consumer_stock:{e}",
                            abs(
                                sum(p.element_amounts_mol.get(e, 0.0) for p in solid.phases) - value
                            ),
                            1e-10,
                        )
                        check(
                            f"consumer_mu:{e}",
                            abs(solid.element_potentials_J_mol[e] - expected["potentials"][e]),
                            3e-4,
                        )
                    check("consumer_phases", abs(len(solid.phases) - 1), 0)
                    check(
                        "consumer_composition",
                        max(
                            abs(solid.phases[0].element_fractions[e] - x)
                            for e, x in expected["candidate_compositions"][0].items()
                        ),
                        2e-6,
                    )
    record["pass"] = all(c["pass"] for c in checks.values())
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units = cases()
    start = monotonic()
    state = {"completed": 0, "current": "initializing"}
    stop = Event()

    def heartbeat():
        while not stop.wait(30):
            elapsed, count = monotonic() - start, state["completed"]
            rate = count / elapsed
            print(
                f"S heartbeat {count}/32 unit={state['current']} rate={rate:.3f}/s "
                f"ETA={(32 - count) / rate if rate else 'unknown'}",
                flush=True,
            )

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    records = []
    try:
        with (args.output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for case in units:
                state["current"] = case["id"]
                unit_start = monotonic()
                try:
                    row = compare(case)
                except Exception:
                    row = {"case": case, "pass": False, "exception": format_exc(), "checks": {}}
                row["elapsed_s"] = monotonic() - unit_start
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                records.append(row)
                state["completed"] = len(records)
                elapsed = monotonic() - start
                rate = len(records) / elapsed
                print(
                    f"S {len(records)}/32 {case['id']} pass={row['pass']} "
                    f"rate={rate:.3f}/s ETA={(32 - len(records)) / rate:.1f}s",
                    flush=True,
                )
    finally:
        stop.set()
        thread.join()
    failures = [
        {
            "id": r["case"]["id"],
            "exception": r.get("exception"),
            "failed_checks": {k: v for k, v in r["checks"].items() if not v["pass"]},
        }
        for r in records
        if not r["pass"]
    ]
    summary = {
        "scope": "development_same_equation_solid_bridge",
        "fixed_units": 32,
        "attempted": len(records),
        "completed": len(records),
        "passed": len(records) - len(failures),
        "failed": len(failures),
        "comparisons": sum(len(r["checks"]) for r in records),
        "failed_comparisons": sum(len(r["failed_checks"]) for r in failures),
        "failures": failures,
        "elapsed_s": monotonic() - start,
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
