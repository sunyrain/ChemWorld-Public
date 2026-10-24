"""Run the prospective 32A block with immutable output and per-unit progress."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np
from fixed_bed_cases import IDS, cases, initial_data, native_world
from fixed_bed_reference import reference_case

from chemworld.foundation.state import WorldState
from chemworld.physchem.fixed_bed import FIXED_BED_POLICY
from chemworld.runtime.fixed_bed import advance_fixed_bed


def native_case(case, cells, tight):
    start = monotonic()
    spec, state = native_world(case, cells)
    policy = replace(FIXED_BED_POLICY, rtol=1e-8 if tight else 1e-6, atol=1e-12 if tight else 1e-10)
    profiles, diagnostics, heat = [], [], 0.0
    success, stop = True, None
    names = [*spec.cell_phase_ids, "source", "collector", "purge"]
    for stage, (inlet, duration) in enumerate(case["stages"]):
        result = advance_fixed_bed(
            state,
            spec,
            duration,
            inlet_phase_id=inlet,
            outlet_phase_id="collector",
            policy=policy,
            evaluation_times_s=np.linspace(0.0, duration, int(duration) + 1),
        )
        diagnostics.append(dict(result.diagnostics))
        if not result.success:
            success = False
            break
        for profile in result.outputs["trajectory"]:
            if profiles and abs(profiles[-1]["time"] - profile["time_s"]) < 1e-10:
                continue
            gas = np.array(
                [
                    [
                        profile["gas_amounts_mol"].get(
                            p, state.phases.phases[p].species_amounts_mol
                        )[k]
                        for k in IDS
                    ]
                    for p in names
                ]
            )
            profiles.append(
                {
                    "time": profile["time_s"],
                    "gas": gas,
                    "bound": np.array(
                        [[q[k] for k in IDS[:2]] for q in profile["bound_amounts_mol"]]
                    ),
                    "heat": heat + profile["bath_heat_J"],
                }
            )
        state = result.state
        heat = state.ledger.energy_jacket_J
        if case.get("snapshot") and stage == 0:
            state = WorldState.from_dict(json.loads(json.dumps(state.to_dict())))
        if result.outcome == "stopped":
            stop = result.diagnostics["stopping_event"]
            break
    return {
        "success": success,
        "profiles": profiles,
        "diagnostics": diagnostics,
        "stop": stop,
        "end": state.ledger.time_s,
        "elapsed_s": monotonic() - start,
        "names": names,
        "cells": cells,
        "tight": tight,
        "condition": case["id"],
    }


def compare(case, native, reference):
    keys = (
        "gas_trajectory",
        "bound_trajectory",
        "pressure",
        "bath_heat",
        "component_balance",
        "energy_balance",
        "outlet_composition",
        "stop",
    )
    checks = dict.fromkeys(keys, False)
    if not native["success"] or not reference["success"]:
        return checks, {}
    n = native["cells"]
    volumes, original_gas, original_bound = initial_data(case, n)
    scale = np.maximum(original_gas.sum(axis=1), 1e-5)[:, None]
    original = original_gas.sum(axis=0)
    original[:2] += original_bound.sum(axis=0)
    errors = dict.fromkeys(keys, 0.0)
    for row in native["profiles"]:
        if row["time"] > reference["end"] + 1e-4:
            continue
        ref = reference["evaluate"](min(row["time"], reference["end"]))
        errors["gas_trajectory"] = max(
            errors["gas_trajectory"], float(np.max(np.abs(row["gas"] - ref["gas"]) / scale))
        )
        errors["bound_trajectory"] = max(
            errors["bound_trajectory"],
            float(np.max(np.abs(row["bound"] - ref["bound"]) / (0.001 / n))),
        )
        pressure = (
            row["gas"].sum(axis=1) * 8.31446261815324 * case["temperature"] / (volumes * 0.001)
        )
        errors["pressure"] = max(errors["pressure"], float(np.max(abs(pressure - ref["pressure"]))))
        errors["bath_heat"] = max(errors["bath_heat"], abs(row["heat"] - ref["heat"]))
        total = row["gas"].sum(axis=0)
        total[:2] += row["bound"].sum(axis=0)
        errors["component_balance"] = max(
            errors["component_balance"], float(np.max(np.abs(total - original)))
        )
    final = native["profiles"][-1]
    ref = reference["evaluate"](min(final["time"], reference["end"]))
    x, xr = final["gas"][n - 1], ref["gas"][n - 1]
    errors["outlet_composition"] = float(np.max(np.abs(x / x.sum() - xr / xr.sum())))
    errors["energy_balance"] = max(
        abs(d["energy_balance_residual_J"]) for d in native["diagnostics"]
    )
    errors["stop"] = abs(native["end"] - reference["end"])
    tolerances = (3e-5, 3e-5, 0.5, 0.002, 1e-10, 1e-7, 2e-5, 1e-4)
    checks = {
        key: errors[key] <= tolerance for key, tolerance in zip(keys, tolerances, strict=True)
    }
    checks["stop"] &= native["stop"] == reference["stop"]
    return checks, errors


def spatial_error(native, reference):
    if not native["success"] or not reference["success"]:
        return None
    error = 0.0
    n = native["cells"]
    for row in native["profiles"]:
        if row["time"] > reference["end"]:
            continue
        ref = reference["evaluate"](row["time"])
        x, xr = row["gas"][n - 1], ref["gas"][15]
        error = max(
            error,
            float(np.max(abs(x / x.sum() - xr / xr.sum()))),
            float(np.max(abs(row["bound"].sum(axis=0) - ref["bound"].sum(axis=0)) / 0.001)),
        )
    return error


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("retain completed block; choose a new output path")
    started, completed, stage = monotonic(), 0, "initialization"
    done = Event()

    def heartbeat():
        while not done.wait(30):
            elapsed = monotonic() - started
            print(f"32A {stage}: {completed}/32 native units, elapsed={elapsed:.1f}s", flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    rows, references, spatial, temporal = [], [], [], []
    try:
        for case in cases():
            refs = {}
            for cells in (4, 8, 16):
                stage = f"reference {case['id']} grid={cells}"
                try:
                    refs[cells] = reference_case(case, cells)
                    references.append(
                        {
                            "condition": case["id"],
                            "cells": cells,
                            **plain(refs[cells]),
                            "final": plain(refs[cells]["evaluate"](refs[cells]["end"])),
                        }
                    )
                except Exception as exc:
                    refs[cells] = None
                    references.append({"condition": case["id"], "cells": cells, "error": repr(exc)})
                print(f"32A reference {case['id']} grid={cells} complete", flush=True)
            current = {}
            for cells in (4, 8):
                for tight in (False, True):
                    stage = f"native {case['id']} grid={cells} tight={tight}"
                    result = native_case(case, cells, tight)
                    checks, errors = (
                        compare(case, result, refs[cells])
                        if refs[cells] is not None
                        else (
                            dict.fromkeys(
                                (
                                    "gas_trajectory",
                                    "bound_trajectory",
                                    "pressure",
                                    "bath_heat",
                                    "component_balance",
                                    "energy_balance",
                                    "outlet_composition",
                                    "stop",
                                ),
                                False,
                            ),
                            {},
                        )
                    )
                    result.update(checks=checks, errors=errors)
                    result["spatial_error"] = (
                        spatial_error(result, refs[16]) if refs[16] is not None else None
                    )
                    rows.append(result)
                    current[cells, tight] = result
                    completed += 1
                    elapsed = monotonic() - started
                    print(
                        f"32A {completed}/32 {stage}: {sum(checks.values())}/8 "
                        f"rate={completed / elapsed:.3f}/s "
                        f"ETA={(32 - completed) * elapsed / completed:.1f}s",
                        flush=True,
                    )
                loose, tight_result = current[cells, False], current[cells, True]
                low = max(
                    loose["errors"].get("gas_trajectory", float("inf")),
                    loose["errors"].get("bound_trajectory", float("inf")),
                )
                high = max(
                    tight_result["errors"].get("gas_trajectory", float("inf")),
                    tight_result["errors"].get("bound_trajectory", float("inf")),
                )
                temporal.append(
                    {
                        "condition": case["id"],
                        "cells": cells,
                        "passed": bool(np.isfinite(high) and high <= low + 1e-7),
                        "loose_error": low if np.isfinite(low) else None,
                        "tight_error": high if np.isfinite(high) else None,
                    }
                )
            coarse, fine = (current[n, True]["spatial_error"] for n in (4, 8))
            spatial.append(
                {
                    "condition": case["id"],
                    "coarse_error": coarse,
                    "fine_error": fine,
                    "passed": coarse is not None
                    and fine is not None
                    and fine <= 1.02 * coarse + 1e-6,
                }
            )
        all_checks = [value for row in rows for value in row["checks"].values()]
        summary = {
            "conditions": 8,
            "native_runs": 32,
            "reference_runs": 24,
            "native_successful": sum(row["success"] for row in rows),
            "reference_successful": sum(row.get("success", False) for row in references),
            "reference_failures": [
                {"condition": row["condition"], "cells": row["cells"], "error": row.get("error")}
                for row in references
                if not row.get("success", False)
            ],
            "comparisons": len(all_checks),
            "passed": sum(all_checks),
            "spatial": [sum(s["passed"] for s in spatial), len(spatial)],
            "temporal": [sum(s["passed"] for s in temporal), len(temporal)],
            "failures": [
                {
                    "condition": row["condition"],
                    "cells": row["cells"],
                    "tight": row["tight"],
                    "failed_checks": [k for k, v in row["checks"].items() if not v],
                    "error": row["diagnostics"][-1].get("error"),
                }
                for row in rows
                if not all(row["checks"].values())
            ],
            "elapsed_s": monotonic() - started,
        }
        payload = plain(
            {
                "mode": "development",
                "fixed_cases": cases(),
                "summary": summary,
                "references": references,
                "runs": rows,
                "spatial_checks": spatial,
                "temporal_checks": temporal,
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(plain(summary), indent=2), flush=True)
    finally:
        done.set()
        thread.join()


if __name__ == "__main__":
    main()
