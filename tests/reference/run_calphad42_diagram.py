"""Fixed 42C block: independent scalar phase boundaries and original stock."""

import argparse
import json
import os
import threading
import time
from dataclasses import replace
from math import log
from pathlib import Path

from calphad42_diagram_cases import diagram_spec
from calphad42_diagram_reference import R, binodal, equilibrium, interaction, surface
from scipy.optimize import brentq

from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_phase_diagram import scan_calphad_phase_diagram
from chemworld.runtime.calphad_phase_diagram import select_calphad_diagram_inventory_point
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget

NAMES = (
    "phase_identity",
    "fractions",
    "compositions",
    "G_J_mol",
    "H_J_mol",
    "mu_J_mol",
    "element_balance",
    "tie_line_x",
    "boundary_T_K",
    "topology",
)
LIMITS = (0, 2e-6, 2e-6, 0.02, 0.05, 0.02, 2e-10, 2e-5, 0.005, 0)


def expected_windows(family):
    if family == "CROSSING":
        return []
    if family == "REGULAR":
        return [(450.0, 12000 * 0.5 / (R * log(3)))]

    def residual(t):
        return interaction(family, t)[0] / (R * t) - 2 * log(3)

    return [(brentq(residual, 550, 700), brentq(residual, 700, 850))]


def compare_map(result, family):
    error = dict.fromkeys(NAMES, 0.0)
    failures = []

    def add(name, value):
        error[name] = max(error[name], float(value))

    for point in result.scan_points:
        t, z = point["temperature_K"], point["composition"]
        actual, expected = point["equilibrium"], equilibrium(family, t, z)
        if not point["success"]:
            for name in NAMES[:7]:
                error[name] = None
            failures.append({"kind": "scan", "T": t, "z": z, "reason": actual["failure_reason"]})
            continue
        # Failed scheduled metrics remain failed, never removed from denominators.
        saved = {name for name, value in error.items() if value is None}
        for name in saved:
            error[name] = 0.0
        phases = sorted(actual["phases"], key=lambda p: p["element_fractions"].get("FE", 0.0))
        refs = expected["phases"]
        if expected["nonunique"]:
            add("phase_identity", any(p["phase_name"] not in {"ALPHA", "BETA"} for p in phases))
            add("fractions", abs(sum(p["fraction"] for p in phases) - 1))
            add("compositions", max(abs(p["element_fractions"].get("FE", 0.0) - z) for p in phases))
            expected_h = sum(
                p["fraction"] * surface(family, t, z, p["phase_name"])["H"] for p in phases
            )
        else:
            add("phase_identity", [p["phase_name"] for p in phases] != [p["phase"] for p in refs])
            if len(phases) != len(refs):
                add("fractions", 1.0)
                add("compositions", 1.0)
            else:
                add(
                    "fractions",
                    max(
                        abs(a["fraction"] - b["fraction"])
                        for a, b in zip(phases, refs, strict=True)
                    ),
                )
                add(
                    "compositions",
                    max(
                        abs(a["element_fractions"].get("FE", 0.0) - b["x"])
                        for a, b in zip(phases, refs, strict=True)
                    ),
                )
            expected_h = expected["H"]
        add("G_J_mol", abs(actual["total_gibbs_J"] - expected["G"]))
        add("H_J_mol", abs(actual["total_enthalpy_J"] - expected_h))
        add(
            "mu_J_mol",
            max(
                abs(actual["element_potentials_J_mol"][e] - mu) for e, mu in expected["mu"].items()
            ),
        )
        add(
            "element_balance",
            max(
                abs(
                    sum(p["amount_mol_atoms"] * p["element_fractions"].get(e, 0.0) for p in phases)
                    - value
                )
                for e, value in (("FE", z), ("NI", 1 - z))
            ),
        )
        for name in saved:
            error[name] = None
    for point in result.boundaries:
        if not point["success"]:
            error["tie_line_x"] = None
            failures.append(
                {"kind": "anchor", "T": point["temperature_K"], "reason": point["failure_reason"]}
            )
        elif error["tie_line_x"] is not None:
            add(
                "tie_line_x",
                max(
                    abs(a - b)
                    for a, b in zip(
                        point["compositions"], binodal(family, point["temperature_K"]), strict=True
                    )
                ),
            )
    endpoint_temperatures = {
        "REGULAR": [12000 / (2 * R)],
        "LOOP": [550.0, 850.0],
        "CROSSING": [700.0],
    }[family]
    expected_kind = "nonunique_first_order" if family == "CROSSING" else "critical"
    add("topology", len(result.endpoints) != len(endpoint_temperatures))
    for point, expected_t in zip(result.endpoints, endpoint_temperatures, strict=True):
        if not point["success"]:
            error["boundary_T_K"] = None
            failures.append({"kind": "endpoint", "reason": point.get("failure_reason")})
        elif error["boundary_T_K"] is not None:
            add("boundary_T_K", abs(point["temperature_K"] - expected_t))
        add("topology", point["kind"] != expected_kind or not point["success"])
    expected = expected_windows(family)
    window = result.windows[0]
    actual = [w["temperature_bounds_K"] for w in window["intervals"]]
    add("topology", not window["success"] or len(actual) != len(expected))
    if not window["success"] or len(actual) != len(expected):
        error["boundary_T_K"] = None
    elif error["boundary_T_K"] is not None:
        for a, b in zip(actual, expected, strict=True):
            add("boundary_T_K", max(abs(x - y) for x, y in zip(a, b, strict=True)))
    add(
        "topology",
        not result.complete or len(result.branches) != (0 if family == "CROSSING" else 1),
    )
    for branch in result.branches:
        add("topology", not branch["complete"] or bool(branch["breaks"]))
        kinds = sorted(
            path.get("termination", {}).get("kind", "unresolved") for path in branch["paths"]
        )
        add(
            "topology",
            kinds
            != (
                ["critical", "declared_domain_end"]
                if family == "REGULAR"
                else ["critical", "critical"]
            ),
        )
    comparisons = [
        {
            "metric": name,
            "maximum_absolute_error": error[name],
            "limit": limit,
            "passed": error[name] is not None and error[name] <= limit,
        }
        for name, limit in zip(NAMES, LIMITS, strict=True)
    ]
    trials = [
        {
            "attempt": i,
            "T": a["temperature_K"],
            "z": a["composition"],
            "purpose": a["purpose"],
            "reason": a["equilibrium"]["failure_reason"],
        }
        for i, a in enumerate(result.attempts)
        if not a["equilibrium"]["success"]
    ]
    return comparisons, failures, trials


def inventory_unit(name, diagram):
    mismatch, exhausted = name == "inventory_mismatch", name == "inventory_exhausted"
    t = 800 if name == "inventory_single" else 600
    z = 0.3 if mismatch else 0.25
    ledger = PhaseLedger(
        {
            "feed": PhaseRecord("feed", "vessel", "solid", 0.02, {"FE": 2 * z, "NI": 2 * (1 - z)}),
            "other": PhaseRecord("other", "elsewhere", "solid", 0.01, {"CR": 3.0}),
        }
    )
    before = ledger.to_dict()
    if exhausted:
        spec = diagram.specification
        diagram = scan_calphad_phase_diagram(
            replace(spec, policy=replace(spec.policy, maximum_evaluations=1))
        )
        index = next(i for i, p in enumerate(diagram.scan_points) if not p["success"])
    else:
        index = next(
            i
            for i, p in enumerate(diagram.scan_points)
            if p["temperature_K"] == t and p["composition"] == 0.25
        )
    result, rejection = None, ""
    try:
        result = select_calphad_diagram_inventory_point(
            ledger,
            diagram,
            index,
            vessel_id="vessel",
            source_phase_ids=("feed",),
            phase_targets=(
                CalphadPhaseTarget("left", "ALPHA", 0.007),
                CalphadPhaseTarget("right", "ALPHA", 0.009),
            ),
        )
    except ValueError as exc:
        rejection = str(exc)
    rejected = mismatch or exhausted
    output = result.ledger if result is not None else ledger
    actual = result.equilibrium.phases if result is not None else ()
    expected = equilibrium("REGULAR", t, z)["phases"]
    amount_error = 0.0
    if rejected:
        identity = set(output.phases) == {"feed", "other"}
        amount_error = max(
            abs(output.phases["feed"].species_amounts_mol[e] - value)
            for e, value in (("FE", 2 * z), ("NI", 2 * (1 - z)))
        )
    else:
        identity = len(actual) == len(expected) and all(p.phase_name == "ALPHA" for p in actual)
        if len(actual) != len(expected):
            amount_error = 1.0
        else:
            amount_error = max(
                abs(p.element_amounts_mol.get(e, 0.0) - 2 * r["fraction"] * value)
                for p, r in zip(actual, expected, strict=True)
                for e, value in (("FE", r["x"]), ("NI", 1 - r["x"]))
            )
    total_error = max(
        abs(output.total_amounts_mol()[e] - n) for e, n in ledger.total_amounts_mol().items()
    )
    checks = {
        "success_or_rejection": bool(rejection)
        if rejected
        else result is not None and result.success,
        "phase_identity": identity,
        "actual_element_amounts": amount_error <= 2e-10,
        "total_stock": total_error <= 2e-10,
        "original_ledger_unchanged": ledger.to_dict() == before,
        "unrelated_phase_preserved": output.phases["other"] == ledger.phases["other"],
        "no_physical_time_or_heat": diagram.to_dict()["physical_time_s"] is None
        and diagram.to_dict()["heat_duty_J"] is None
        and (
            result is None
            or result.to_dict()["energy_scope"]
            == "TP_material_boundary; no_heat_duty_or_time_advance"
        ),
        "failure_diagnostics": (
            "composition" in rejection
            if mismatch
            else "unresolved" in rejection
            if exhausted
            else bool(result.equilibrium.diagnostics["rounds"])
            if result
            else False
        ),
    }
    return (
        {
            "id": name,
            "comparisons": [{"metric": key, "passed": value} for key, value in checks.items()],
            "amount_error_mol": amount_error,
            "total_error_mol": total_error,
            "expected_rejection": rejection,
            "passed": all(checks.values()),
        },
        {
            "before": before,
            "after": output.to_dict(),
            "rejection": rejection,
            "result": result.to_dict() if result is not None else None,
            "budget_diagram": diagram.to_dict() if exhausted else None,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    directory = Path(os.environ["TEMP"]) / "ChemWorld-42C" / args.tag
    directory.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {"completed": 0, "TP_evaluations": 0, "stage": "initializing"}
    stop = threading.Event()

    def progress():
        elapsed = max(time.monotonic() - start, 1e-9)
        completed = status["completed"]
        print(
            json.dumps(
                {
                    **status,
                    "units_total": 16,
                    "elapsed_s": round(elapsed, 2),
                    "units_per_s": completed / elapsed,
                    "ETA_s": (16 - completed) * elapsed / completed if completed else None,
                }
            ),
            flush=True,
        )

    def heartbeat():
        while not stop.wait(30):
            progress()

    def callback(row):
        status.update(row)
        if row["TP_evaluations"] % 20 == 0:
            progress()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    units, diagrams, details = [], {}, []
    try:
        for family in ("REGULAR", "LOOP", "CROSSING"):
            for policy in (0, 1):
                for direction in (1, -1):
                    name = f"{family}-policy{policy}-direction{direction}"
                    status.update(stage=name, TP_evaluations=0)
                    progress()
                    result = scan_calphad_phase_diagram(
                        diagram_spec(family, policy), direction=direction, progress=callback
                    )
                    comparisons, failures, trials = compare_map(result, family)
                    raw = directory / f"{name}.json"
                    with raw.open("x", encoding="utf-8") as handle:
                        json.dump(result.to_dict(), handle, allow_nan=False)
                    units.append(
                        {
                            "id": name,
                            "comparisons": comparisons,
                            "passed": all(c["passed"] for c in comparisons),
                            "failed_requests": failures,
                            "failed_trials": trials,
                            "TP_calls": len(result.attempts),
                            "scan_points": len(result.scan_points),
                            "anchors": len(result.boundaries),
                            "endpoints": len(result.endpoints),
                            "windows": len(result.windows),
                        }
                    )
                    details.append(str(raw))
                    if family == "REGULAR" and policy == 0 and direction == 1:
                        diagrams["inventory"] = result
                    status["completed"] += 1
                    progress()
        for name in (
            "inventory_split",
            "inventory_single",
            "inventory_mismatch",
            "inventory_exhausted",
        ):
            status["stage"] = name
            unit, raw = inventory_unit(name, diagrams["inventory"])
            path = directory / f"{name}.json"
            with path.open("x", encoding="utf-8") as handle:
                json.dump(raw, handle, allow_nan=False)
            details.append(str(path))
            units.append(unit)
            status["completed"] += 1
            progress()
    finally:
        stop.set()
        thread.join()
    summary = {
        "scope": "development; original neutral monatomic binary substitutional CEF; finite search",
        "fixed_units": 16,
        "executed_units": len(units),
        "passed_units": sum(u["passed"] for u in units),
        "fixed_comparisons": 152,
        "executed_comparisons": sum(len(u["comparisons"]) for u in units),
        "passed_comparisons": sum(c["passed"] for u in units for c in u["comparisons"]),
        "requested_scans": 300,
        "requested_anchors": 40,
        "requested_endpoints": 16,
        "requested_windows": 12,
        "elapsed_s": time.monotonic() - start,
        "raw_outputs": details,
        "units": units,
    }
    with (directory / "SUMMARY.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(
        json.dumps(
            {
                "summary": str(directory / "SUMMARY.json"),
                "passed_units": summary["passed_units"],
                "passed_comparisons": summary["passed_comparisons"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
