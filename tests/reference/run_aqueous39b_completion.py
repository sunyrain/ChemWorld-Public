"""Fixed 62-unit / 448-comparison explicit-authority HKF completion block."""

from __future__ import annotations

import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from threading import Event, Thread
from time import monotonic

from aqueous39b_cases import DIELECTRIC, TP
from aqueous39b_reference import PR, TR, M, density, hkf_state, mp, water_g
from run_aqueous39b import (
    comparison,
    dielectric_case,
    integration_case,
    reference_case,
    rejection_case,
    species_case,
)
from scipy.optimize import brentq

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord, VesselRecord
from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_hkf_sources import (
    hkf_aqueous_database,
    hkf_reference_basis,
    hkf_source_records,
    hkf_source_species,
)
from chemworld.physchem.aqueous_phase_properties import aqueous_phase_properties
from chemworld.physchem.aqueous_reaction_equilibrium import (
    AqueousEquilibriumSpec,
    solve_aqueous_reaction_equilibrium,
)
from chemworld.runtime.aqueous_phase_equilibrium import equilibrate_aqueous_phase_inventory

R = 8.31446261815324


@lru_cache(maxsize=6)
def independent_standards(T, P):
    """Independent 50-digit equations, using only the original public numeric data."""
    basis = {"H": 65.340, "O": 102.576, "Na": 51.46, "Cl": 111.5395, "charge": -65.340}
    states = {k: hkf_state(v, T, P, basis,
                          reference_authority="gibbs_entropy" if k == "OH-" else "strict")
              for k, v in hkf_source_records().items()}
    mm, t, p = M(".018015268"), M(str(T)), M(str(P))
    sr = -mp.diff(lambda x: mm * water_g(x, PR), TR)
    hr = mm * water_g(TR, PR) + TR * sr
    g = mm * water_g(t, p) + M(-285830) - hr - t * (M("69.95") - sr)
    s = -mp.diff(lambda x: mm * water_g(x, p), t) + M("69.95") - sr
    states["H2O"] = {"G": float(g), "H": float(g + t * s), "V": float(mm / density(t, p))}
    return states


def dissociation_case(T, P, salt):
    db = hkf_aqueous_database(hydroxide_reference_authority="gibbs_entropy")
    spec = AqueousEquilibriumSpec(db, ("water-dissociation",))
    feed = {"H2O": 55.5, "Na+": salt, "Cl-": salt, "H+": 0.0, "OH-": 0.0}
    ref = independent_standards(T, P)
    delta = {k: ref["H+"][k] + ref["OH-"][k] - ref["H2O"][k] for k in ("G", "H", "V")}

    def affinity(log_extent):
        extent = math.exp(log_extent)
        water = 55.5 - extent
        return (delta["G"] / (R * T) + 2 * math.log(extent / (water * .018015))
                + (2 * salt + 2 * extent) / water)

    extent = math.exp(brentq(affinity, math.log(1e-25), math.log(.99 * 55.5), xtol=1e-12))
    result = solve_aqueous_reaction_equilibrium(spec, feed, temperature_K=T, pressure_Pa=P)
    if not result.success:
        return [], {"root": result.to_dict(), "reference_extent_mol": extent}
    props = aqueous_phase_properties(db, feed, temperature_K=T, pressure_Pa=P)
    ledger = PhaseLedger({
        "aqueous": PhaseRecord("aqueous", "tank", "aqueous", props.volume_m3 * 1000, feed,
                               metadata={"carrier_representation": "species_inventory"}),
        "other": PhaseRecord("other", "other-tank", "solid", .1, {"inert": 1}),
    })
    vessel = VesselRecord("tank", "HKF-TP", 3, 623.15, 1e8, temperature_K=T, pressure_Pa=P)
    before = ledger.to_dict()
    candidate = equilibrate_aqueous_phase_inventory(
        ledger, spec, "aqueous", vessel=vessel, temperature_K=T, pressure_Pa=P)
    if not candidate.success:
        return [], {"root": result.to_dict(), "inventory": candidate.to_dict(),
                    "reference_extent_mol": extent}
    checks = [
        comparison("root success", int(result.success), 1, 0, 0),
        comparison("H+ amount", result.amounts_mol["H+"], extent, 2e-12, 2e-6),
        comparison("OH- amount", result.amounts_mol["OH-"], extent, 2e-12, 2e-6),
        comparison("water amount", result.amounts_mol["H2O"], 55.5 - extent, 2e-9, 0),
        comparison("charge", result.charge_residual_mol, 0, 2e-9, 0),
        comparison("affinity", result.reaction_residuals["water-dissociation"], 0, 2e-8, 0),
        comparison("inventory success", int(candidate.success), 1, 0, 0),
        comparison("heat", candidate.heat_duty_J, extent * delta["H"], 2e-5),
        comparison("work", candidate.boundary_work_J, P * extent * delta["V"], 2e-8),
        comparison("delta U", candidate.final_properties.internal_energy_J
                   - candidate.initial_properties.internal_energy_J,
                   extent * (delta["H"] - P * delta["V"]), 2e-5),
        comparison("immutable input", int(ledger.to_dict() == before), 1, 0, 0),
        comparison("other owner", int(candidate.ledger.phases["other"].to_dict()
                   == before["other"]), 1, 0, 0),
    ]
    return checks, {"root": result.to_dict(), "inventory": candidate.to_dict(),
                    "reference_extent_mol": extent, "independent_standards": ref}


def authority_case(index):
    if index:
        db = hkf_aqueous_database(hydroxide_reference_authority="gibbs_entropy")
        payload = json.loads(json.dumps(db.to_dict(), allow_nan=False))
        clone = AqueousDatabase.from_dict(payload)
        a, b = db.standard_states(473.15, 1e7), clone.standard_states(473.15, 1e7)
        return [comparison("full codec", int(clone.to_dict() == payload), 1, 0, 0),
                *(comparison(k + " G", b[k].gibbs_J_mol, a[k].gibbs_J_mol) for k in a)]
    raw = hkf_source_records()
    rejected = False
    try:
        hkf_source_species("OH-")
    except ValueError:
        rejected = True
    th = hkf_source_species("OH-", reference_authority="gibbs_entropy")
    state, data = th.evaluate_tp(298.15, 1e5), th.reference_diagnostics()
    shift = hkf_reference_basis().entropy_shift({"H": 1, "O": 1}, -1)
    expected_g = -37595 * 4.184 - 298.15 * shift
    return [comparison("strict source rejection", int(rejected), 1, 0, 0),
            comparison("raw source unchanged", int(raw == hkf_source_records()
                       and th.formation_enthalpy_ref_J_mol == -54977 * 4.184
                       and th.formation_gibbs_ref_J_mol == -37595 * 4.184
                       and th.entropy_ref_J_mol_K == -2.56 * 4.184), 1, 0, 0),
            comparison("working G", state.gibbs_J_mol, expected_g),
            comparison("working H", state.enthalpy_J_mol, expected_g + 298.15 * -2.56 * 4.184),
            comparison("raw residual", data["source_triple_residual_J_mol"], 12.484976, 2e-8, 0),
            comparison("raw triple remains inconsistent", int(not data["source_triple_consistent"]),
                       1, 0, 0)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    if output.exists():
        raise ValueError("choose a new immutable output path")
    jobs = [*( ("dielectric", list(row), 7, lambda row=row: dielectric_case(*row))
               for row in DIELECTRIC),
            *(("species", [k, *tp], 8, lambda k=k, tp=tp: species_case(
                k, *tp, reference_authority="gibbs_entropy" if k == "OH-" else "strict"))
              for k in ("Na+", "Cl-", "H+", "OH-") for tp in TP),
            *(("reference", [i], 3, lambda i=i: reference_case(i)) for i in range(4)),
            *(("integration", [i], 6, lambda i=i: integration_case(i)) for i in range(4)),
            *(("rejection", [i], 1, lambda i=i: rejection_case(i)) for i in range(8)),
            *(("dissociation", [*tp, salt], 12, lambda tp=tp, salt=salt:
                dissociation_case(*tp, salt)) for tp in TP for salt in (0., .1)),
            *(("authority", [i], 6, lambda i=i: authority_case(i)) for i in range(2))]
    assert len(jobs) == 62 and sum(row[2] for row in jobs) == 448
    rows, stopped, started = [], Event(), monotonic()

    def heartbeat():
        while not stopped.wait(30):
            elapsed = monotonic() - started
            rate = len(rows) / elapsed
            print(f"39B completion {len(rows)}/62; {rate:.3f}/s; "
                  f"ETA={(62-len(rows))/rate if rate else -1:.1f}s", flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for kind, args, count, run in jobs:
            row = {"kind": kind, "args": args, "comparisons_scheduled": count}
            try:
                value = run()
                checks, details = value if kind == "dissociation" else (value, {})
                row.update(comparisons=checks, details=details,
                           passed=len(checks) == count and all(c["passed"] for c in checks))
            except Exception as exc:
                row.update(comparisons=[], passed=False, exception=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            print(f"39B {len(rows)}/62 {kind} {args}: {row['passed']}", flush=True)
        summary = {"mode": "development", "units_scheduled": 62,
                   "units_passed": sum(r["passed"] for r in rows), "comparisons_scheduled": 448,
                   "comparisons_executed": sum(len(r["comparisons"]) for r in rows),
                   "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
                   "failures": [{k: v for k, v in r.items() if k != "details"}
                                for r in rows if not r["passed"]],
                   "elapsed_s": monotonic() - started,
                   "source_diagnostics": hkf_source_species(
                       "OH-", reference_authority="gibbs_entropy").reference_diagnostics()}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"summary": summary, "units": rows},
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        stopped.set()
        thread.join()


if __name__ == "__main__":
    main()
