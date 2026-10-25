"""Fixed 67-unit general molecular equilibrium development qualification.

Run from the repository with PYTHONPATH=tests. Every invocation uses a new output
directory and retains every failed unit. --only is explicitly nonqualifying debug.
"""

from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path
from threading import Event, Thread
from time import monotonic
from traceback import format_exc

import numpy as np
from thermo.eos_mix import PRMIX

from chemworld.physchem.molecular_critical import solve_molecular_critical
from chemworld.physchem.molecular_eos import molecular_eos_state
from chemworld.physchem.molecular_flash import MolecularDensityBasin, MolecularFlashPhase
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseSeed,
    solve_molecular_multiphase,
)
from chemworld.physchem.molecular_phase_boundary import (
    MolecularBoundarySeed,
    MolecularPhaseBoundarySpec,
    solve_molecular_azeotrope,
    solve_molecular_phase_boundary,
    trace_molecular_phase_envelope,
)
from reference.molecular28_general_cases import native_spec, pr_model
from reference.molecular28_general_reference import (
    FAMILIES,
    MATERIALS,
    azeotrope_reference,
    mixture_critical_reference,
    pure_nearcritical_reference,
    regular_state,
    thermo_boundary,
    thermo_state,
    vlle_state,
)


def cases():
    rows = []

    def add(prefix, family, temperature, z, pressure, coordinate):
        rows.append(
            {
                "id": f"{prefix}-{len(rows) + 1:02d}",
                "kind": "closure",
                "family": family,
                "T": temperature,
                "z": z,
                "P": pressure,
                "coordinate": coordinate,
            }
        )

    for t in (280.0, 300.0, 320.0):
        for z in (0.4, 0.6):
            for coordinate in ("TP", "PH", "PS", "UV"):
                add("G-PR", "vle", t, (z, 1 - z), 6e5, coordinate)
    for coordinate in ("TV", "Pmass_density", "Ph_mass", "Ps_mass"):
        add("G-MASS", "vle", 300.0, (0.6, 0.4), 6e5, coordinate)
    for t, z in [
        (300.0, (1 / 3,) * 3),
        (300.0, (0.49, 0.49, 0.02)),
        (300.0, (0.98, 0.01, 0.01)),
        (300.0, (0.5, 0.5, 0.0)),
        (300.0, (0.5 - 5e-16, 0.5 - 5e-16, 1e-15)),
        (350.0, (1 / 3,) * 3),
        (400.0, (1 / 3,) * 3),
        (600.0, (1 / 3,) * 3),
    ]:
        add("G-REG", "regular", t, z, 1e5, "TP")
    for coordinate in ("PH", "PS", "UV"):
        add("G-REG-INV", "regular", 300.0, (1 / 3,) * 3, 1e5, coordinate)
    for t in (280.0, 300.0, 330.0):
        for pressure in (1e6, 3e6, 6e6):
            add("G-VLLE", "vlle", t, (0.3, 0.3, 0.4), pressure, "TP")
    for coordinate in ("PH", "PS", "UV"):
        add("G-VLLE-INV", "vlle", 300.0, (0.3, 0.3, 0.4), 3e6, coordinate)
    for t in (290.0, 310.0):
        for kind in ("bubble", "dew"):
            for fixed in ("T", "P"):
                rows.append(
                    {
                        "id": f"G-BOUND-{t:g}-{kind}-{fixed}",
                        "kind": "boundary",
                        "T": t,
                        "boundary": kind,
                        "fixed": fixed,
                    }
                )
    rows += [{"id": "G-AZEO", "kind": "azeotrope"}]
    rows += [{"id": f"G-NEAR-{f}", "kind": "nearcritical", "fraction": f} for f in (0.99, 0.9999)]
    rows += [
        {"id": "G-CRIT", "kind": "critical"},
        {"id": "G-TRACE-forward", "kind": "forward"},
        {"id": "G-TRACE-backward", "kind": "backward"},
    ]
    for phase in ("gas", "liquid"):
        add("G-META", "vle", 300.0, (0.6, 0.4), 6e5, "TP")
        rows[-1]["metastable"] = phase
    assert len(rows) == 67
    return rows


def check(name, error, limit):
    value = float(error)
    return {
        "name": name,
        "error": value,
        "limit": limit,
        "passed": bool(np.isfinite(value) and value <= limit),
    }


def planned_comparisons(case):
    if case["kind"] == "closure":
        return (
            12
            + (2 if case["family"] == "vlle" else 0)
            + sum(1 if x == 0 else 2 if x < 1e-12 else 0 for x in case["z"])
        )
    return {
        "boundary": 9,
        "azeotrope": 8,
        "nearcritical": 8,
        "critical": 3,
        "forward": 263,
        "backward": 271,
    }[case["kind"]]


def state_comparisons(result, ref, feed):
    checks = [check("phase_count", abs(len(result.phases) - len(ref["amounts"])), 0)]
    if not result.success or not checks[0]["passed"]:
        return checks
    ids = tuple(feed)
    order = min(
        permutations(range(len(result.phases))),
        key=lambda order: sum(
            abs(result.phases[j].component_amounts_mol[k] - ref["amounts"][i][n])
            for i, j in enumerate(order)
            for n, k in enumerate(ids)
        ),
    )
    checks += [
        check(
            "phase_amount",
            max(
                abs(result.phases[j].component_amounts_mol[k] - ref["amounts"][i][n])
                for i, j in enumerate(order)
                for n, k in enumerate(ids)
            ),
            2e-6,
        ),
        check(
            "density_relative",
            max(
                abs(result.phases[j].state.molar_density_mol_m3 / ref["densities"][i] - 1)
                for i, j in enumerate(order)
            ),
            3e-5,
        ),
    ]
    for attr, key, tolerance, relative in (
        ("temperature_K", "T", 0.002, False),
        ("pressure_Pa", "P", 2e-5, True),
        ("enthalpy_J", "H", 0.03, False),
        ("entropy_J_K", "S", 3e-4, False),
        ("internal_energy_J", "U", 0.03, False),
        ("volume_m3", "V", 3e-5, True),
    ):
        actual = getattr(result, attr)
        checks.append(
            check(
                key, abs(actual / ref[key] - 1) if relative else abs(actual - ref[key]), tolerance
            )
        )
    for attr, limit in (
        ("material_balance_error_mol", 2e-10),
        ("chemical_potential_error_RT", 3e-7),
        ("pressure_relative_error", 3e-7),
    ):
        checks.append(check(attr, getattr(result, attr), limit))
    for key, amount in feed.items():
        if amount == 0:
            checks.append(
                check(
                    "exact_absence_" + key,
                    max(abs(p.component_amounts_mol[key]) for p in result.phases),
                    0,
                )
            )
        elif amount < 1e-12:
            recovered = sum(p.component_amounts_mol[key] for p in result.phases)
            checks.append(check("trace_conservation_" + key, abs(recovered / amount - 1), 1e-12))
            checks.append(
                check(
                    "trace_positive_all_phases_" + key,
                    sum(p.component_amounts_mol[key] <= 0 for p in result.phases),
                    0,
                )
            )
    if "source_error_inner" in ref:
        checks += [check(k, ref[k], 1e-7) for k in ("source_error_inner", "source_error_outer")]
    return checks


def closure(case):
    family, t, p, z = (case[k] for k in ("family", "T", "P", "z"))
    ref = (
        regular_state(z, t, p)
        if family == "regular"
        else vlle_state(t, p, z)
        if family == "vlle"
        else thermo_state(z, T=t, P=p)
    )
    spec = native_spec(family)
    feed = dict(zip(spec.model.component_ids, z, strict=True))
    coordinates = case["coordinate"]
    keys = {
        "Pmass_density": ("P", "mass_density"),
        "Ph_mass": ("P", "h_mass"),
        "Ps_mass": ("P", "s_mass"),
    }.get(coordinates, tuple(coordinates))
    values = ref | {
        "mass_density": ref["mass"] / ref["V"],
        "h_mass": ref["H"] / ref["mass"],
        "s_mass": ref["S"] / ref["mass"],
    }
    seed = MolecularMultiphaseSeed(t + 2, p * 1.01)
    if "metastable" in case:
        material = [MATERIALS[k] for k in FAMILIES["vle"]]
        source = PRMIX(
            Tcs=[v[0] for v in material],
            Pcs=[v[1] for v in material],
            omegas=[v[2] for v in material],
            zs=list(z),
            T=t,
            P=p,
        )
        rho = 1 / (source.V_g if case["metastable"] == "gas" else source.V_l)
        phase = MolecularFlashPhase(
            case["metastable"],
            feed.copy(),
            molecular_eos_state(spec.model.residual_model, t, rho, z),
        )
        seed = MolecularMultiphaseSeed(t, p, (phase,))
        ref["metastable_initial_density"] = rho
    result = solve_molecular_multiphase(spec, feed, inputs={k: values[k] for k in keys}, seed=seed)
    checks = state_comparisons(result, ref, feed)
    return {
        "success": result.success and all(c["passed"] for c in checks),
        "reference": ref,
        "native": result.to_dict(),
        "comparisons": checks,
        "failure_reason": result.failure_reason,
    }


def boundary_spec(family, kind, composition, *, pure=False):
    if pure:
        lb, vb, tb, pb = (100.0, 15000.0), (100.0, 15000.0), (300.0, 390.0), (1e6, 5e6)
    elif family == "azeotrope":
        lb, vb, tb, pb = (10000.0, 40000.0), (1.0, 1000.0), (300.0, 400.0), (1e4, 1e6)
    elif family == "envelope":
        lb, vb, tb, pb = (8000.0, 30000.0), (100.0, 15000.0), (200.0, 250.0), (1e6, 6.35e6)
    else:
        lb, vb, tb, pb = (5000.0, 20000.0), (1.0, 6000.0), (250.0, 400.0), (1e4, 3e6)
    return MolecularPhaseBoundarySpec(
        pr_model(family, pure=pure),
        tuple(composition),
        kind,
        MolecularDensityBasin("liquid", lb, sum(lb) / 2),
        MolecularDensityBasin("gas", vb, sum(vb) / 2),
        tb,
        pb,
        stability_mode="local_coexistence",
    )


def boundary_seed(ref, kind):
    return MolecularBoundarySeed(
        ref["T"] + 1,
        1.01 * ref["rhoL"],
        0.99 * ref["rhoV"],
        tuple(ref["y"] if kind == "bubble" else ref["x"]),
    )


def boundary_comparisons(result, ref):
    if not result.success:
        return []
    liquid, vapor = result.liquid, result.vapor
    checks = [
        check("T", abs(liquid.temperature_K - ref["T"]), 0.002),
        check("P", abs(liquid.pressure_Pa / ref["P"] - 1), 2e-5),
        check("rhoL", abs(liquid.molar_density_mol_m3 / ref["rhoL"] - 1), 3e-5),
        check("rhoV", abs(vapor.molar_density_mol_m3 / ref["rhoV"] - 1), 3e-5),
        check(
            "x", max(abs(x - y) for x, y in zip(liquid.mole_fractions, ref["x"], strict=True)), 2e-6
        ),
        check(
            "y", max(abs(x - y) for x, y in zip(vapor.mole_fractions, ref["y"], strict=True)), 2e-6
        ),
        check("chemical_potential_RT", result.chemical_potential_error_RT, 3e-7),
        check("pressure_relative", result.pressure_relative_error, 3e-7),
    ]
    if "source_chemical_potential_error_RT" in ref:
        checks.append(
            check("source_chemical_potential_RT", ref["source_chemical_potential_error_RT"], 3e-7)
        )
    return checks


def boundary(case):
    kind = case.get("boundary", "bubble")
    if case["kind"] == "azeotrope":
        ref = azeotrope_reference()
        spec = boundary_spec("azeotrope", kind, (0.6, 0.4))
        seed = MolecularBoundarySeed(351.0, 1.01 * ref["rhoL"], 0.99 * ref["rhoV"], (0.58, 0.42))
        result = solve_molecular_azeotrope(spec, seed, temperature_K=350.0)
    elif case["kind"] == "nearcritical":
        ref = pure_nearcritical_reference(case["fraction"])
        spec = boundary_spec("vle", kind, (1.0,), pure=True)
        result = solve_molecular_phase_boundary(
            spec, boundary_seed(ref, kind), temperature_K=ref["T"]
        )
    else:
        ref = thermo_boundary((0.6, 0.4), kind, T=case["T"])
        spec = boundary_spec("vle", kind, (0.6, 0.4))
        coordinate = (
            {"temperature_K": ref["T"]} if case["fixed"] == "T" else {"pressure_Pa": ref["P"]}
        )
        result = solve_molecular_phase_boundary(spec, boundary_seed(ref, kind), **coordinate)
    checks = boundary_comparisons(result, ref)
    return {
        "success": result.success and all(c["passed"] for c in checks),
        "reference": ref,
        "native": result.to_dict(),
        "comparisons": checks,
        "failure_reason": result.failure_reason,
    }


def critical():
    ref = mixture_critical_reference()
    result = solve_molecular_critical(
        pr_model("envelope"),
        (0.8, 0.2),
        initial_temperature_K=230.0,
        initial_density_mol_m3=10000.0,
        temperature_bounds_K=(200.0, 260.0),
        density_bounds_mol_m3=(5000.0, 18000.0),
    )
    checks = []
    if result.success:
        actual = result.response
        checks = [
            check("T", abs(actual["temperature_K"] - ref["T"]), 0.002),
            check("rho", abs(actual["molar_density_mol_m3"] / ref["rho"] - 1), 2e-4),
            check("P", abs(actual["pressure_Pa"] / ref["P"] - 1), 3e-5),
        ]
    return {
        "success": ref["success"] and result.success and all(c["passed"] for c in checks),
        "reference": ref,
        "native": result.to_dict(),
        "comparisons": checks,
        "failure_reason": result.failure_reason,
    }


def envelope(case, context):
    spec = boundary_spec("envelope", "dew", (0.8, 0.2))
    initial_ref = thermo_boundary((0.8, 0.2), "dew", family="envelope", P=5e6)
    if case["kind"] == "forward":
        initial = solve_molecular_phase_boundary(
            spec, boundary_seed(initial_ref, "dew"), pressure_Pa=5e6
        )
        if not initial.success:
            return {
                "success": False,
                "native": initial.to_dict(),
                "reference": initial_ref,
                "comparisons": [],
                "failure_reason": "initial_boundary_failed",
            }
        direction = 1
    else:
        if "forward" not in context:
            return {
                "success": False,
                "comparisons": [],
                "failure_reason": "missing_forward_dependency",
            }
        initial = context["forward"].points[-1]
        direction = 1  # At the high-P side of the T fold, increasing T traces backwards.
    result = trace_molecular_phase_envelope(
        spec, initial, steps=28, arclength_step=0.025, direction=direction
    )
    if case["kind"] == "forward":
        context["forward"] = result
    refs, checks = [], [check("completed_steps", abs(result.completed_steps - 28), 0)]
    for index, point in enumerate(result.points):
        ref = thermo_boundary((0.8, 0.2), "dew", family="envelope", P=point.liquid.pressure_Pa)
        refs.append(ref)
        checks += [
            dict(c, name=f"point{index}/{c['name']}") for c in boundary_comparisons(point, ref)
        ]
    if case["kind"] == "forward":
        checks.append(
            check(
                "temperature_turn",
                not any(t["coordinate"] == "temperature_K" for t in result.turning_intervals),
                0,
            )
        )
    else:
        checks += [
            dict(c, name="return/" + c["name"])
            for c in boundary_comparisons(result.points[-1], initial_ref)
        ]
    return {
        "success": result.success and all(c["passed"] for c in checks),
        "reference": refs,
        "native": result.to_dict(),
        "comparisons": checks,
        "failure_reason": result.termination_reason if not result.success else "",
    }


def execute(case, context):
    if case["kind"] == "closure":
        return closure(case)
    if case["kind"] in {"boundary", "azeotrope", "nearcritical"}:
        return boundary(case)
    if case["kind"] == "critical":
        return critical()
    return envelope(case, context)


def json_safe(value):
    """Preserve nonfinite failure diagnostics as explicit JSON tags, not NaN."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_diagnostic": str(value)}
    return value


def run(output, *, only=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    matrix = cases()
    selected = [c for c in matrix if only is None or only in c["id"]]
    (output / "MATRIX.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    start = monotonic()
    state = {"completed": 0, "current": "starting"}
    stop = Event()

    def progress():
        elapsed = monotonic() - start
        completed = state["completed"]
        throughput = completed / elapsed if elapsed else 0
        eta = (len(selected) - completed) / throughput if throughput else None
        print(
            json.dumps(
                {
                    "stage": "general-qualification" if only is None else "debug-only",
                    "completed": completed,
                    "total": len(selected),
                    "current": state["current"],
                    "elapsed_s": round(elapsed, 1),
                    "units_per_min": round(throughput * 60, 3),
                    "eta_s": None if eta is None else round(eta, 1),
                }
            ),
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    records, context = [], {}
    try:
        with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
            for case in selected:
                state["current"] = case["id"]
                begin = monotonic()
                try:
                    record = execute(case, context)
                except Exception as error:
                    record = {
                        "success": False,
                        "comparisons": [],
                        "failure_reason": repr(error),
                        "traceback": format_exc(),
                    }
                record.update(case=case, elapsed_s=monotonic() - begin)
                records.append(record)
                stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
                stream.flush()
                state["completed"] += 1
                progress()
    finally:
        stop.set()
        thread.join()
    failures = [
        {
            "id": r["case"]["id"],
            "reason": r.get("failure_reason", ""),
            "failed_comparisons": [c for c in r["comparisons"] if not c["passed"]],
        }
        for r in records
        if not r["success"]
    ]
    comparisons = [c for r in records for c in r["comparisons"]]
    summary = {
        "mode": "development_qualification" if only is None else "nonqualifying_debug",
        "fixed_units": 67,
        "attempted": len(records),
        "completed": len(records),
        "passed": sum(r["success"] for r in records),
        "failed": len(failures),
        "comparisons_attempted": len(comparisons),
        "comparisons_planned": sum(planned_comparisons(c) for c in selected),
        "comparisons_not_evaluated": sum(planned_comparisons(c) for c in selected)
        - len(comparisons),
        "comparisons_passed": sum(c["passed"] for c in comparisons),
        "failures": failures,
        "elapsed_s": monotonic() - start,
        "passed_block": only is None and len(records) == 67 and not failures,
    }
    (output / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="nonqualifying debug subset; never a qualification pass")
    args = parser.parse_args()
    print(json.dumps(run(args.output, only=args.only), indent=2), flush=True)
