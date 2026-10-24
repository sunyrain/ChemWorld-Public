"""Fixed 26-unit, 90-observation, 948-comparison spatial development block."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aqueous40d_reference import TIMES, history, self_check

from chemworld.examples.native_aqueous_reactive_transport import example_inputs
from chemworld.physchem.aqueous_reactive_transport import (
    advance_aqueous_reactive_transport,
    prepare_aqueous_transport_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.aqueous_reactive_transport_world import build_aqueous_reactive_transport_world


def comparison(name, actual, expected=0.0, atol=0.0, rtol=0.0):
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    error = np.abs(a - e)
    allowed = atol + rtol * np.abs(e)
    return {
        "quantity": name,
        "maximum_absolute_error": float(np.max(error)),
        "absolute_tolerance": atol,
        "relative_tolerance": rtol,
        "passed": bool(np.all(error <= allowed)),
    }


def exact(name, result):
    return comparison(name, int(bool(result)), 1)


def concentrations(spec, amounts):
    return {
        k: np.array(
            [
                amounts[v][k] / spec.mesh.cell_volumes_m3[i]
                for i, v in enumerate(spec.cell_vessel_ids)
            ]
        )
        for k in amounts[spec.anchor_vessel_id]
    }


def invariant_error(kind, spec, before, after):
    values = []

    def totals(n):
        sums = {k: sum(row[k] for row in n.values()) for k in n[spec.anchor_vessel_id]}
        if kind == "salt":
            return [sums["M+"], sums["A-"]]
        return [
            sums["A-"] + sums["HA"] + sums.get("S", 0),
            sums["H+"] + sums["HA"] + sums.get("S", 0),
        ]

    values.extend(np.array(totals(before)) - totals(after))
    values.extend(before[v]["H2O"] - after[v]["H2O"] for v in spec.cell_vessel_ids)
    return max(abs(x) for x in values)


def numeric_case(kind, condition, cells):
    spec, initial = example_inputs(kind, condition, cells)
    reference = history(kind, condition, cells)
    result = advance_aqueous_reactive_transport(spec, initial, 100, evaluation_times_s=TIMES)
    rows = []
    charges = {s.species_id: s.charge for s in spec.equilibrium.database.species}
    for point, expected in zip(result.trajectory, reference, strict=True):
        actual = concentrations(spec, point["amounts_by_vessel"])
        faces = list(point["faces"].values())
        local = [
            comparison(
                "all cell species concentrations",
                [actual[k] for k in actual],
                [expected["concentrations"][k] for k in actual],
                2e-5,
                2e-6,
            ),
            comparison(
                "all face potentials",
                [f["potential_difference_V"] for f in faces],
                expected["potential_V"],
                2e-7,
            ),
            comparison(
                "all face species rates",
                [[f["amount_rates_mol_s"][k] for f in faces] for k in spec.solute_ids],
                [expected["rates_mol_s"][k] for k in spec.solute_ids],
                1e-13,
                1e-5,
            ),
            comparison(
                "cell charge",
                [sum(charges[k] * n[k] for k in n) for n in point["amounts_by_vessel"].values()],
                0,
                1e-13,
            ),
            comparison(
                "face current", [f["current_molar_charge_rate_mol_s"] for f in faces], 0, 1e-18
            ),
            comparison(
                "all conserved totals and finite water",
                invariant_error(kind, spec, initial, point["amounts_by_vessel"]),
                0,
                2e-11,
            ),
            comparison(
                "diffusion dissipation",
                sum(f["dissipation_W"] for f in faces),
                expected["dissipation_W"],
                2e-10,
                2e-5,
            ),
            exact("nonnegative face dissipation", all(f["dissipation_W"] >= -1e-14 for f in faces)),
        ]
        if kind != "salt":
            affinities = []
            for n in point["amounts_by_vessel"].values():
                mu = spec.equilibrium.database.chemical_potentials(
                    n, temperature_K=spec.temperature_K, pressure_Pa=spec.pressure_Pa
                ).chemical_potentials_J_mol
                affinities.append(mu["H+"] + mu["A-"] - mu["HA"])
            local += [
                comparison("fast reaction affinity", affinities, 0, 2e-5),
                comparison(
                    "same source bath Q", point["bath_heat_J"], expected["bath_heat_J"], 2e-6
                ),
                comparison(
                    "local first law",
                    [
                        d["bath_heat_J"]
                        + d["transported_enthalpy_J"]
                        - d["internal_energy_change_J"]
                        for d in point["cells"].values()
                    ],
                    0,
                    2e-9,
                ),
            ]
        rows.extend({"time_s": point["time_s"], **r} for r in local)
    convergence = None
    if kind != "slow":
        continuum = history(kind, condition, cells, continuous=True)[-1]["concentrations"]
        a = concentrations(spec, result.final["amounts_by_vessel"])
        convergence = max(float(np.max(abs(a[k] - continuum[k]))) for k in a)
    return rows, {"native": result.to_dict(), "continuum_concentration_error_mol_m3": convergence}


def local_states(runtime):
    return {runtime._state.vessel_id: runtime._state, **runtime._state.inactive_vessels}


def runtime_case(index):
    spec, n = example_inputs("slow")
    if index in (0, 1, 7):
        runtime = ChemWorldRuntime(composition=build_aqueous_reactive_transport_world(spec, n))
        restored = None
        try:
            runtime.reset(seed=40)
            if index == 7:
                before = runtime._state
                with patch(
                    "chemworld.physchem.aqueous_reactive_transport.solve_aqueous_reaction_equilibrium",
                    side_effect=ValueError("fixed forced fast closure failure"),
                ):
                    step = runtime.execute({"operation": "wait", "duration_s": 1})
                return [
                    exact(
                        "explicit failed transaction",
                        step.info["transaction_status"] != "committed",
                    ),
                    exact(
                        "anchor stocks", runtime._state.species_amounts == before.species_amounts
                    ),
                    exact(
                        "other stocks", runtime._state.inactive_vessels == before.inactive_vessels
                    ),
                    exact("clock rollback", runtime._state.ledger.time_s == before.ledger.time_s),
                    exact(
                        "heat rollback",
                        runtime._state.ledger.energy_jacket_J == before.ledger.energy_jacket_J,
                    ),
                    exact("phase ownership rollback", runtime._state.phases == before.phases),
                ], {"step": step.info}
            if index == 1:
                first = runtime.execute({"operation": "wait", "duration_s": 50})
                restored = ChemWorldRuntime.from_snapshot(
                    json.loads(json.dumps(runtime.snapshot().to_dict()))
                )
                for target in (runtime, restored):
                    last = target.execute({"operation": "wait", "duration_s": 50})
                return [
                    exact("first wait committed", first.info["transaction_status"] == "committed"),
                    exact("continuation committed", last.info["transaction_status"] == "committed"),
                    exact(
                        "exact full snapshot continuation",
                        runtime._state.to_dict() == restored._state.to_dict(),
                    ),
                    comparison("shared clock", runtime._state.ledger.time_s, 100, 1e-12),
                    exact(
                        "all cell elapsed",
                        all(
                            runtime._state.vessel_elapsed_s[v] == 100 for v in spec.cell_vessel_ids
                        ),
                    ),
                    exact(
                        "no persistent invariant duplicate",
                        all(not s.state_slots.slots for s in local_states(runtime).values()),
                    ),
                ], {"final": runtime._state.to_dict()}
            step = runtime.execute({"operation": "wait", "duration_s": 100})
            locals_ = local_states(runtime)
            values = concentrations(
                spec, {v: locals_[v].species_amounts for v in spec.cell_vessel_ids}
            )
            expected = history("slow", 0, 4)[-1]
            diagnostic = step.info
            rows = [
                exact("real transaction committed", step.info["transaction_status"] == "committed"),
                comparison(
                    "all world species",
                    [values[k] for k in values],
                    [expected["concentrations"][k] for k in values],
                    2e-5,
                    2e-6,
                ),
                comparison("single shared clock", runtime._state.ledger.time_s, 100, 1e-12),
                exact(
                    "original phase and finite water",
                    all(
                        locals_[v].phases.phases["aqueous"].species_amounts_mol
                        == locals_[v].species_amounts
                        and locals_[v].species_amounts["H2O"] == n[v]["H2O"]
                        for v in spec.cell_vessel_ids
                    ),
                ),
                comparison(
                    "world single bath Q",
                    runtime._state.ledger.energy_jacket_J,
                    expected["bath_heat_J"],
                    2e-6,
                ),
                exact(
                    "spatial redistribution occurred",
                    abs(
                        sum(locals_["cell-00"].species_amounts[k] for k in ("S", "HA", "A-"))
                        - sum(n["cell-00"][k] for k in ("S", "HA", "A-"))
                    )
                    > 1e-10,
                ),
            ]
            return rows, {"step": diagnostic, "final": runtime._state.to_dict()}
        finally:
            runtime.close()
            if restored:
                restored.close()
    # Declaration/initial-condition failures preserve the original inputs.
    if index == 2:
        n["cell-00"]["H+"] *= 2
    if index == 5:
        n["cell-00"]["H+"] = n["cell-00"]["A-"] = 0
    if index == 6:
        for cell in n.values():
            cell["H+"] = cell["A-"] = 0
    before_n, before_spec = (
        json.dumps(n, sort_keys=True),
        json.dumps(spec.to_dict(), sort_keys=True),
    )
    failure = ""
    try:
        if index == 3:
            db = spec.equilibrium.database
            altered = replace(
                db,
                species=tuple(
                    replace(s, standard_state=replace(s.standard_state, molar_volume_m3_mol=1e-6))
                    if s.species_id == "HA"
                    else s
                    for s in db.species
                ),
            )
            replace(spec, equilibrium=replace(spec.equilibrium, database=altered))
        elif index == 4:
            replace(spec, temperature_K=400)
        else:
            prepare_aqueous_transport_state(spec, n)
    except ValueError as exc:
        failure = str(exc)
    return [
        exact("declared failure", bool(failure)),
        exact("readable reason", len(failure) > 10),
        exact("input species unchanged", json.dumps(n, sort_keys=True) == before_n),
        exact(
            "input declaration unchanged", json.dumps(spec.to_dict(), sort_keys=True) == before_spec
        ),
        exact(
            "finite water unchanged", all(n[v]["H2O"] == json.loads(before_n)[v]["H2O"] for v in n)
        ),
        exact(
            "mesh unchanged", list(spec.mesh.faces_m) == json.loads(before_spec)["mesh"]["faces_m"]
        ),
    ], {"failure_reason": failure}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    reference_check = self_check()
    definitions = [
        *(
            {"kind": kind, "condition": c, "cells": n, "comparisons": 40 if kind == "salt" else 55}
            for kind in ("salt", "fast", "slow")
            for c in (0, 1)
            for n in (4, 8, 16)
        ),
        *({"kind": "world", "index": i, "comparisons": 6} for i in range(8)),
    ]
    (output / "units.json").write_text(json.dumps(definitions, indent=2), encoding="utf-8")
    units, comparisons, details = [], [], []
    started = time.monotonic()
    print("start units=0/26 observations=90 comparisons=948 reference_self_check=PASS", flush=True)
    for i, definition in enumerate(definitions, 1):
        record = {"unit": i, **definition}
        try:
            rows, detail = (
                runtime_case(definition["index"])
                if definition["kind"] == "world"
                else numeric_case(definition["kind"], definition["condition"], definition["cells"])
            )
            if len(rows) != definition["comparisons"]:
                raise ValueError("fixed comparison denominator mismatch")
            comparisons.extend({"unit": i, **row} for row in rows)
            details.append({"unit": i, **detail})
            record.update(passed=all(r["passed"] for r in rows), executed=len(rows))
        except Exception as exc:
            record.update(passed=False, executed=0, exception=f"{type(exc).__name__}: {exc}")
        units.append(record)
        elapsed = time.monotonic() - started
        print(
            f"progress units={i}/26 rate={i / max(elapsed, 1e-9):.3f}/s "
            f"eta={(26 - i) * elapsed / i:.1f}s "
            f"status={'PASS' if record['passed'] else 'FAIL'}",
            flush=True,
        )
    failures = [r for r in comparisons if not r["passed"]]
    summary = {
        "mode": "development",
        "planned_units": 26,
        "passed_units": sum(r["passed"] for r in units),
        "failed_units": sum(not r["passed"] for r in units),
        "planned_observations": 90,
        "planned_comparisons": 948,
        "executed_comparisons": len(comparisons),
        "passed_comparisons": len(comparisons) - len(failures),
        "failed_comparisons": len(failures),
        "unexecuted_comparisons": 948 - len(comparisons),
        "unit_failures": [r for r in units if not r["passed"]],
        "comparison_failures": failures,
        "reference_self_check": reference_check,
        "elapsed_seconds": time.monotonic() - started,
        "data_scope": "authored CC0; no material accuracy or full reactive transport qualification",
    }
    for name, rows in (("unit_results", units), ("comparisons", comparisons), ("details", details)):
        (output / (name + ".jsonl")).write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows), encoding="utf-8"
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
