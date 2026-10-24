"""Execute the predeclared finite 42D two-resolution block with all failures."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np
from scipy.optimize import linear_sum_assignment

from chemworld.examples.calphad_ternary import (
    build_calphad_ternary_example,
    ternary_example_spec,
    ternary_tdb,
)
from chemworld.foundation.state_ledgers import PhaseLedger, PhaseRecord
from chemworld.physchem.calphad_ternary_diagram import (
    CalphadTernaryPath,
    CalphadTernaryResult,
    scan_calphad_ternary_diagram,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.calphad_phase_inventory import CalphadPhaseTarget
from chemworld.runtime.calphad_ternary_inventory import select_calphad_ternary_inventory_point
from chemworld.world.calphad_ternary_analysis import analyze_calphad_ternary_world

FAMILIES = ("preferred", "excluded", "ideal", "regular")
RESOLUTIONS = (3, 6)
REQUESTED = ((0.3, 0.3, 0.4), (0.7, 0.2, 0.1), (0.15, 0.75, 0.1))


def fixed_spec(family, resolution):
    return replace(
        ternary_example_spec(family, resolution),
        requested_compositions=REQUESTED,
        paths=(
            CalphadTernaryPath(
                "declared-FE-to-NI",
                (
                    (1.0, 0.0, 0.0),
                    REQUESTED[0],
                    (1 / 3, 1 / 3, 1 / 3),
                    REQUESTED[1],
                    (0.0, 1.0, 0.0),
                ),
            ),
        ),
    )


def compare(point, reference):
    checks = dict.fromkeys(
        (
            "phase_identity_multiplicity",
            "phase_compositions",
            "phase_fractions",
            "G",
            "H",
            "mu_active",
        ),
        False,
    )
    errors = {}
    native = point["equilibrium"]
    if not native["success"] or not reference["success"]:
        return checks, errors
    phases = [p for p in native["phases"] if p["fraction"] > 1e-6]
    refs = reference["phases"]
    if len(phases) == len(refs):
        costs = np.array(
            [
                [
                    sum(
                        abs(p["element_fractions"].get(e, 0.0) - q["element_fractions"].get(e, 0.0))
                        for e in ("FE", "NI", "CR")
                    )
                    if p["phase_name"] == q["phase_name"]
                    else 1e6
                    for q in refs
                ]
                for p in phases
            ]
        )
        rows, cols = linear_sum_assignment(costs)
        checks["phase_identity_multiplicity"] = bool(np.max(costs[rows, cols]) < 1e6)
        if checks["phase_identity_multiplicity"]:
            errors["phase_compositions"] = max(
                abs(
                    phases[i]["element_fractions"].get(e, 0.0)
                    - refs[j]["element_fractions"].get(e, 0.0)
                )
                for i, j in zip(rows, cols, strict=True)
                for e in ("FE", "NI", "CR")
            )
            errors["phase_fractions"] = max(
                abs(phases[i]["fraction"] - refs[j]["fraction"])
                for i, j in zip(rows, cols, strict=True)
            )
            checks["phase_compositions"] = errors["phase_compositions"] <= 2e-5
            checks["phase_fractions"] = errors["phase_fractions"] <= 2e-5
    errors["G"] = abs(native["total_gibbs_J"] - reference["G"])
    errors["H"] = abs(native["total_enthalpy_J"] - reference["H"])
    errors["mu_active"] = max(
        abs(value - reference["mu"][e]) for e, value in native["element_potentials_J_mol"].items()
    )
    checks.update(G=errors["G"] <= 0.02, H=errors["H"] <= 0.1, mu_active=errors["mu_active"] <= 0.1)
    return {k: bool(v) for k, v in checks.items()}, errors


def reference_style(point):
    eq = point["equilibrium"]
    return {
        "success": eq["success"],
        "phases": eq["phases"],
        "G": eq["total_gibbs_J"],
        "H": eq["total_enthalpy_J"],
        "mu": eq["element_potentials_J_mol"],
    }


def map_checks(diagram, refs):
    geometry = diagram.geometry
    ref_signatures = [sorted(p["phase_name"] for p in r.get("phases", [])) for r in refs]
    expected_brackets = [
        e["point_indices"]
        for e in geometry["grid_edges"]
        if all(refs[i]["success"] for i in e["point_indices"])
        and ref_signatures[e["point_indices"][0]] != ref_signatures[e["point_indices"][1]]
    ]
    tie_ok = True
    for tie in geometry["tie_geometry"]:
        x = [sum(v["fraction"] * v["composition"][i] for v in tie["vertices"]) for i in range(3)]
        tie_ok &= (
            max(
                abs(a - b)
                for a, b in zip(x, diagram.points[tie["point_index"]]["composition"], strict=True)
            )
            <= 2e-9
        )
    path = geometry["paths"][0]
    ids = path["point_indices"]
    return {
        "finite_mesh_counts": (
            len(geometry["grid_cells"]) == diagram.specification.grid_intervals**2
            and len(diagram.points) == (13 if diagram.specification.grid_intervals == 3 else 31)
        ),
        "boundary_brackets_match_reference": [
            e["point_indices"] for e in geometry["boundary_brackets"]
        ]
        == expected_brackets,
        "ties_obey_lever_rule": bool(tie_ok),
        "path_preserves_adjacency": [e["point_indices"] for e in path["edges"]]
        == [list(p) for p in pairwise(ids)],
        "finite_json_and_no_clock": (
            CalphadTernaryResult.from_dict(
                json.loads(json.dumps(diagram.to_dict(), allow_nan=False))
            ).to_dict()
            == diagram.to_dict()
            and diagram.to_dict()["physical_time_s"] is None
            and diagram.to_dict()["heat_duty_J"] is None
        ),
    }


def inventory_unit(diagram):
    original = PhaseLedger(
        {
            "feed": PhaseRecord(
                "feed", "reactor", "solid", 0.00002, {"FE": 0.0006, "NI": 0.0006, "CR": 0.0008}
            ),
            "other": PhaseRecord("other", "other", "solid", 0.001, {"CU": 0.1}),
        }
    )
    clone = CalphadTernaryResult.from_dict(json.loads(json.dumps(diagram.to_dict())))
    index = next(p["index"] for p in clone.points if p["composition"] == list(REQUESTED[0]))
    for p in clone.points[index]["equilibrium"]["phases"]:
        p["amount_mol_atoms"] = 1000000
        p["fraction"] = 1000000
    before = original.to_dict()
    targets = tuple(
        CalphadPhaseTarget(f"{m.phase_name}-{j}", m.phase_name, 0.01)
        for m in diagram.specification.equilibrium.models
        for j in range(3)
    )
    result = select_calphad_ternary_inventory_point(
        original,
        clone,
        index,
        vessel_id="reactor",
        source_phase_ids=("feed",),
        phase_targets=targets,
    )
    actual = result.ledger.total_amounts_mol()
    expected = original.total_amounts_mol()
    return {
        "checks": {
            "original_TP_success": result.success,
            "actual_atoms_conserved": max(abs(actual.get(k, 0) - v) for k, v in expected.items())
            <= 1e-10,
            "original_unchanged": original.to_dict() == before,
            "unrelated_owner_preserved": result.ledger.phases["other"] == original.phases["other"],
            "cached_amounts_ignored": result.success
            and abs(sum(p.amount_mol_atoms for p in result.equilibrium.phases) - 0.002) <= 1e-10,
        },
        "result": result.to_dict(),
    }


def world_unit(resolution):
    checks = dict.fromkeys(
        (
            "construct",
            "read_only_query",
            "exact_actual_point",
            "original_ideal_phase",
            "heat_committed",
            "heat_and_time",
            "actual_atoms",
            "snapshot_exact",
            "continued_query",
            "finite_json",
        ),
        False,
    )
    payload = {"checks": checks}
    session = replay = None
    try:
        composition = build_calphad_ternary_example()
        session = ChemWorldRuntime(composition=composition)
        session.reset(seed=42)
        checks["construct"] = True
        before = session._state.to_dict()
        report = analyze_calphad_ternary_world(
            composition,
            session._state,
            grid_intervals=resolution,
            progress=lambda p: print(f"world grid={resolution}", json.dumps(p), flush=True),
        )
        query = report["world_query"]
        point = report["diagram"]["points"][query["actual_point_index"]]
        checks["read_only_query"] = before == session._state.to_dict()
        checks["exact_actual_point"] = (
            point["success"] and point["composition"] == query["actual_composition"]
        )
        checks["original_ideal_phase"] = point["phase_signature"] == ["ALPHA"]
        action = session.execute(
            {"operation": "heat", "target_temperature_K": 800.0, "duration_s": 1.0}
        )
        payload["heat_info"] = action.info
        checks["heat_committed"] = action.info["transaction_status"] == "committed"
        checks["heat_and_time"] = (
            session._state.ledger.time_s == 1
            and abs(session._state.ledger.energy_jacket_J - 0.05) <= 1e-8
            and session._state.temperature_K > before["temperature_K"]
        )
        checks["actual_atoms"] = (
            max(
                abs(session._state.species_amounts[e] - v)
                for e, v in {"FE": 0.0003, "NI": 0.0003, "CR": 0.0004}.items()
            )
            <= 1e-10
        )
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        actions = [
            runtime.execute({"operation": "wait", "duration_s": 1.0}).info
            for runtime in (session, replay)
        ]
        payload["continuation_info"] = actions
        checks["snapshot_exact"] = (
            all(a["transaction_status"] == "committed" for a in actions)
            and session._state.to_dict() == replay._state.to_dict()
        )
        after = analyze_calphad_ternary_world(
            composition,
            session._state,
            grid_intervals=resolution,
            progress=lambda p: print(f"world after grid={resolution}", json.dumps(p), flush=True),
        )
        after_point = after["diagram"]["points"][after["world_query"]["actual_point_index"]]
        checks["continued_query"] = (
            after_point["success"] and after["world_query"]["state_time_s"] == 2
        )
        payload.update(before_query=report, after_query=after, final_state=session._state.to_dict())
        json.dumps(payload, allow_nan=False)
        checks["finite_json"] = True
    except Exception as exc:
        payload["error"] = repr(exc)
    finally:
        if session is not None:
            session.close()
        if replay is not None:
            replay.close()
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    stage = "native sections"
    completed = 0
    done = Event()

    def heartbeat():
        while not done.wait(30):
            print(
                f"42D {stage}; units={completed}/14 elapsed={monotonic() - started:.1f}s",
                flush=True,
            )

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        diagrams = {}
        reference_inputs = []
        for family in FAMILIES:
            tdb, allowed = ternary_tdb(family)
            for resolution in RESOLUTIONS:
                key = f"{family}-{resolution}"
                diagram = scan_calphad_ternary_diagram(
                    fixed_spec(family, resolution),
                    progress=lambda p, key=key: print(key, json.dumps(p), flush=True),
                )
                diagrams[key] = diagram
                (args.output / f"{key}.json").write_text(
                    json.dumps(diagram.to_dict(), indent=2, allow_nan=False), encoding="utf8"
                )
                reference_inputs.extend(
                    {
                        "id": f"{key}:{p['index']}",
                        "tdb": tdb,
                        "phases": allowed,
                        "T": diagram.specification.temperature_K,
                        "P": diagram.specification.pressure_Pa,
                        "feed": dict(
                            zip(diagram.specification.components, p["composition"], strict=True)
                        ),
                    }
                    for p in diagram.points
                )
                completed += 1
        stage = "independent pycalphad"
        run = subprocess.run(
            [
                str(args.reference_python),
                str(Path(__file__).with_name("calphad42_ternary_pycalphad.py")),
            ],
            input=json.dumps(reference_inputs),
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        )
        reference = json.loads(run.stdout)
        (args.output / "reference.json").write_text(
            json.dumps(reference, indent=2, allow_nan=False), encoding="utf8"
        )
        refs = {r["id"]: r for r in reference["cases"]}
        rows = []
        maps = []
        for key, diagram in diagrams.items():
            point_refs = [refs[f"{key}:{p['index']}"] for p in diagram.points]
            for point, ref in zip(diagram.points, point_refs, strict=True):
                checks, errors = compare(point, ref)
                balance = (
                    max(
                        abs(
                            sum(
                                p["fraction"] * p["element_fractions"].get(e, 0.0)
                                for p in point["equilibrium"]["phases"]
                            )
                            - x
                        )
                        for e, x in zip(
                            diagram.specification.components, point["composition"], strict=True
                        )
                    )
                    if point["success"]
                    else None
                )
                rows.append(
                    {
                        "id": f"{key}:{point['index']}",
                        "checks": checks,
                        "errors": errors,
                        "conserved": balance is not None and balance <= 2e-9,
                        "balance_error": balance,
                        "native_success": point["success"],
                        "failure_reason": point["equilibrium"]["failure_reason"],
                    }
                )
            maps.append({"id": key, "checks": map_checks(diagram, point_refs)})
        convergence = []
        for family in FAMILIES:
            fine = {tuple(p["composition"]): p for p in diagrams[f"{family}-6"].points}
            for point in diagrams[f"{family}-3"].points:
                checks, errors = compare(point, reference_style(fine[tuple(point["composition"])]))
                convergence.append(
                    {
                        "family": family,
                        "composition": point["composition"],
                        "checks": checks,
                        "errors": errors,
                    }
                )
        stage = "actual inventory and thermal world"
        inventories = []
        for family in FAMILIES:
            inventories.append({"id": family, **inventory_unit(diagrams[f"{family}-6"])})
            completed += 1
            print(f"42D inventory {family}: {completed}/14", flush=True)
        worlds = []
        for resolution in RESOLUTIONS:
            worlds.append({"grid_intervals": resolution, **world_unit(resolution)})
            completed += 1
            print(f"42D world grid={resolution}: {completed}/14", flush=True)
        groups = {
            "reference": rows,
            "map_geometry": maps,
            "shared_resolution": convergence,
            "inventory": inventories,
            "world": worlds,
        }
        counts = {
            name: {
                "passed": sum(v for row in group for v in row["checks"].values()),
                "total": sum(len(row["checks"]) for row in group),
            }
            for name, group in groups.items()
        }
        failures = [
            {
                "group": name,
                "id": row.get("id", row.get("family", row.get("grid_intervals"))),
                "composition": row.get("composition"),
                "failed_checks": [k for k, v in row["checks"].items() if not v],
                "error": row.get("error", row.get("failure_reason")),
            }
            for name, group in groups.items()
            for row in group
            if not all(row["checks"].values())
        ]
        failures.extend(
            {"group": "conservation", "id": row["id"], "balance_error": row["balance_error"]}
            for row in rows
            if not row["conserved"]
        )
        signatures = Counter(
            "+".join(p["phase_signature"]) for d in diagrams.values() for p in d.points
        )
        summary = {
            "mode": "development",
            "units": completed,
            "declared_units": 14,
            "native_nodes": len(rows),
            "native_success": sum(row["native_success"] for row in rows),
            "conservation_passed": sum(row["conserved"] for row in rows),
            "counts": counts,
            "observed_assemblages": dict(signatures),
            "failures": failures,
            "elapsed_s": monotonic() - started,
        }
        payload = {
            "summary": summary,
            "groups": groups,
            "reference_version": reference["version"],
            "scope": "finite two-resolution development block; no global map qualification",
        }
        (args.output / "results.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf8"
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        done.set()
        thread.join()


if __name__ == "__main__":
    main()
