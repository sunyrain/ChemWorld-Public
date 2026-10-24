"""Execute the fixed 12-condition/two-grid independent 42A block with progress."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from math import exp, log
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np
from calphad_equilibrium_models import cases, native_spec
from scipy.optimize import brentq, linear_sum_assignment

from chemworld.physchem.calphad_equilibrium import calphad_tp_equilibrium

R = 8.3145


def compare(result, reference):
    names = ("phase_identity_multiplicity", "phase_compositions", "phase_fractions", "G", "H", "mu")
    checks = dict.fromkeys(names, False)
    errors = {}
    if not result.success or not reference["success"]:
        return checks, errors
    phases = [p for p in result.phases if p.fraction > 1e-6]
    refs = reference["phases"]
    if len(phases) == len(refs):
        costs = np.array([[sum(abs(p.element_fractions[e]-q["element_fractions"][e])
                         for e in p.element_fractions) if p.phase_name == q["phase_name"] else 1e6
                           for q in refs] for p in phases])
        rows, columns = linear_sum_assignment(costs)
        checks["phase_identity_multiplicity"] = bool(np.max(costs[rows, columns]) < 1e6)
        if checks["phase_identity_multiplicity"]:
            errors["phase_compositions"] = max(abs(phases[i].element_fractions[e]-
                refs[j]["element_fractions"][e]) for i, j in zip(rows, columns, strict=True)
                for e in phases[i].element_fractions)
            errors["phase_fractions"] = max(abs(phases[i].fraction-refs[j]["fraction"])
                for i, j in zip(rows, columns, strict=True))
            checks["phase_compositions"] = errors["phase_compositions"] <= 2e-5
            checks["phase_fractions"] = errors["phase_fractions"] <= 2e-5
    errors["G"] = abs(result.total_gibbs_J-reference["G"])
    errors["H"] = abs(result.total_enthalpy_J-reference["H"])
    errors["mu"] = max(abs(mu-reference["mu"][e])
                       for e, mu in result.element_potentials_J_mol.items())
    checks.update(G=errors["G"] <= .02, H=errors["H"] <= .1, mu=errors["mu"] <= .1)
    return {k: bool(v) for k, v in checks.items()}, errors


def analytic(case, result):
    if case["id"] not in {"binary-gap", "binary-hot", "ternary-three", "ternary-hot"}:
        return None
    if not result.success:
        return {"passed": False, "error": None}
    if case["id"] == "binary-gap":
        x = brentq(lambda x: R*case["T"]*log(x/(1-x))+12000*(1-2*x), 1e-10, .5-1e-8)
        expected = [x, 1-x]
        observed = sorted(p.element_fractions["FE"] for p in result.phases)
        error = max(abs(a-b) for a, b in zip(expected, observed, strict=True))
    elif case["id"] == "binary-hot":
        error = max(abs(p.element_fractions["FE"]-.5) for p in result.phases)
    else:
        ratio = exp(12000/(R*case["T"]))
        a, b = ratio/(ratio+2), 1/(ratio+2)
        error = max(abs(x-(a if p.phase_name.startswith(e+"_") else b))
                    for p in result.phases for e, x in p.element_fractions.items())
    return {"passed": error <= 2e-6, "error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-python", type=Path, default=Path(tempfile.gettempdir()) /
                        "chemworld-conv42-reference-0.11.2/Scripts/python.exe")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("retain completed evidence; use a new output path")
    fixed = cases()
    completed, stage, started = 0, "independent reference", monotonic()
    done = Event()

    def heartbeat():
        while not done.wait(30):
            elapsed = monotonic()-started
            print(f"42A {stage}: {completed}/24 native runs, elapsed={elapsed:.1f}s", flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        reference_run = subprocess.run([str(args.reference_python),
            str(Path(__file__).with_name("calphad_equilibrium_pycalphad.py"))],
            input=json.dumps(fixed), text=True, stdout=subprocess.PIPE, check=True)
        reference = json.loads(reference_run.stdout)
        assert reference["version"] == "0.11.2"
        references = {r["id"]: r for r in reference["cases"]}
        rows = []
        stage = "native equilibrium"
        for case in fixed:
            for intervals in (8, 16):
                start = monotonic()
                result = calphad_tp_equilibrium(native_spec(case, intervals), case["T"],
                                                101325., case["feed"])
                checks, errors = compare(result, references[case["id"]])
                rows.append({"condition": case["id"], "grid_intervals": intervals,
                    "elapsed_s": monotonic()-start, "reference_checks": checks,
                    "reference_errors": errors, "analytic": analytic(case, result),
                    "native": result.to_dict()})
                completed += 1
                elapsed = monotonic()-started
                print(f"42A {completed}/24 {case['id']} grid={intervals}: "
                    f"success={result.success} checks={sum(checks.values())}/6 "
                    f"rate={completed/elapsed:.3f}/s ETA={(24-completed)*elapsed/completed:.1f}s",
                    flush=True)
        comparisons = [v for row in rows for v in row["reference_checks"].values()]
        exact = [row["analytic"]["passed"] for row in rows if row["analytic"] is not None]
        payload = {"mode": "development", "fixed_conditions": fixed, "reference": reference,
            "summary": {"conditions": 12, "native_runs": 24,
                "successful_native_runs": sum(row["native"]["success"] for row in rows),
                "reference_comparisons": len(comparisons), "reference_passed": sum(comparisons),
                "analytic_comparisons": len(exact), "analytic_passed": sum(exact),
                "failures": [{"condition": row["condition"], "grid": row["grid_intervals"],
                    "reason": row["native"]["failure_reason"],
                    "failed_checks": [k for k, v in row["reference_checks"].items() if not v]}
                    for row in rows if not all(row["reference_checks"].values())],
                "elapsed_s": monotonic()-started}, "runs": rows}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                         allow_nan=False)+"\n",
                               encoding="utf-8")
        print(json.dumps(payload["summary"], indent=2), flush=True)
    finally:
        done.set()
        thread.join()


if __name__ == "__main__":
    main()
