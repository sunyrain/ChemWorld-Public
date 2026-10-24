"""Fixed 52E 16-unit/128-comparison development runner; retains every failure."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from sbml_finite_time52_cases import cases, model
from sbml_finite_time52_reference import reference

from chemworld.physchem.sbml_finite_time import sbml_finite_time_stability
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.sbml_finite_time import analyze_sbml_finite_time_world
from chemworld.world.sbml_world import sbml_world

OFFSETS = (0.2, 0.8, 2.0)


def numeric(name, actual, expected):
    a, b = np.asarray(actual), np.asarray(expected)
    return {
        "name": name,
        "passed": bool(np.all(np.isfinite(a)) and np.allclose(a, b, atol=1e-8, rtol=3e-7)),
        "actual": a.tolist(),
        "expected": b.tolist(),
        "atol": 1e-8,
        "rtol": 3e-7,
        "max_absolute_error": float(np.max(np.abs(a - b))),
    }


def exact(name, checks):
    return {"name": name, "passed": all(checks.values()), "conditions": checks}


def run_unit(kind, scaled, scales, record):
    source = model(kind, scaled=scaled)
    spec = SBMLSystemSpec(source)
    options = {"sample_offsets_s": OFFSETS, "amount_scales_mol": scales}
    report = sbml_finite_time_stability(spec, spec.frame.initial_values, start_time_s=0, **options)
    record["source_report"], record["source_model"] = report, spec.to_dict()
    assert report["status"] == "complete", report["failure"]
    rows = report["records"]
    refs = [reference(kind, t, scales) for t in OFFSETS]
    comparisons = record["comparisons"]
    comparisons.append(
        numeric(
            "original-amounts",
            [[r["coordinates"]["N:" + s] for s in report["species_ids"]] for r in rows],
            [r["amounts"] for r in refs],
        )
    )
    comparisons.append(
        numeric(
            "singular-growth",
            [r["singular_growth_factors"] for r in rows],
            [r["growth"] for r in refs],
        )
    )
    comparisons.append(
        numeric(
            "finite-time-rates",
            [r["finite_time_rates_s_inverse"] for r in rows],
            [r["rates"] for r in refs],
        )
    )
    basis = np.asarray(report["basis_scaled_amounts"])
    comparisons.append(
        numeric(
            "compatibility-class-tangent",
            [basis @ np.asarray(r["reduced_tangent_map"]) @ basis.T for r in rows],
            [r["transfer"] for r in refs],
        )
    )
    comparisons.append(
        exact(
            "rank-and-conservation",
            {
                "rank": report["reachable_dimension"] == refs[0]["rank"],
                "excluded_modes": all(
                    r["conserved_zero_modes_excluded"] == len(scales) - refs[0]["rank"]
                    for r in rows
                ),
                "conservation": all(
                    r["conservation_error"] < 1e-10 and r["tangent_conservation_error"] < 1e-10
                    for r in rows
                ),
                "initial_totals": all(
                    np.max(
                        np.abs(
                            np.asarray(r["conserved_totals"]) - report["initial_conserved_totals"]
                        ),
                        initial=0,
                    )
                    < 1e-10
                    for r in rows
                ),
            },
        )
    )
    composition = sbml_world(source)
    runtime, restored = ChemWorldRuntime(composition=composition), None
    try:
        runtime.reset(seed=52)
        before = runtime._state.to_dict()
        current = analyze_sbml_finite_time_world(runtime._state, composition, **options)
        record["world_report"] = current
        comparisons.append(
            exact(
                "world-read-only",
                {
                    "same_source_result": {k: v for k, v in current.items() if k != "world_query"}
                    == report,
                    "state_unchanged": runtime._state.to_dict() == before,
                    "resources_uncharged": current["world_query"]["live_resources_charged"]
                    is False,
                },
            )
        )
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        rebuilt = SBMLSystemSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
        comparisons.append(
            exact(
                "source-and-world-json",
                {
                    "world_replay": analyze_sbml_finite_time_world(
                        restored._state, composition, **options
                    )
                    == current,
                    "source_replay": sbml_finite_time_stability(
                        rebuilt, rebuilt.frame.initial_values, start_time_s=0, **options
                    )
                    == report,
                    "report_json": json.loads(json.dumps(current, allow_nan=False)) == current,
                },
            )
        )
        comparisons.append(
            exact(
                "window-interpretation",
                {
                    "completed": report["completed_windows"] == report["requested_windows"] == 3,
                    "resolved": all(all(r["resolved_singular_values"]) for r in rows),
                    "amplification": all(
                        np.isclose(
                            r["maximum_amplification"], expected["growth"][0], atol=1e-8, rtol=3e-7
                        )
                        for r, expected in zip(rows, refs, strict=True)
                    ),
                    "growth_flag": all(
                        (r["maximum_amplification"] > 1) == (expected["growth"][0] > 1)
                        for r, expected in zip(rows, refs, strict=True)
                    ),
                    "finite_window_only": "no asymptotic" in report["interpretation"],
                    "future_controls": current["world_query"]["future_controls_predicted"] is False,
                },
            )
        )
    finally:
        runtime.close()
        if restored:
            restored.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records, started = [], time.perf_counter()
    for index, (name, kind, scaled, scales) in enumerate(cases(), 1):
        record = {"unit": name, "comparisons": []}
        begin = time.perf_counter()
        try:
            run_unit(kind, scaled, scales, record)
        except Exception as error:
            record["exception"] = str(error)
            record["traceback"] = traceback.format_exc()
        while len(record["comparisons"]) < 8:
            record["comparisons"].append(
                {"name": f"unresolved-slot-{len(record['comparisons']) + 1}", "passed": False}
            )
        record["passed"] = all(c["passed"] for c in record["comparisons"])
        record["elapsed_s"] = time.perf_counter() - begin
        records.append(record)
        (args.output / f"{index:02}-{name}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.perf_counter() - started
        print(
            f"52E {index}/16; pass={record['passed']}; {index / elapsed:.2f} unit/s; "
            f"ETA={(16 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    summary = {
        "mode": "development",
        "units": 16,
        "comparisons": 128,
        "units_passed": sum(r["passed"] for r in records),
        "comparisons_passed": sum(c["passed"] for r in records for c in r["comparisons"]),
        "elapsed_s": time.perf_counter() - started,
        "failures": [
            {"unit": r["unit"], "comparison": c["name"], "exception": r.get("exception")}
            for r in records
            for c in r["comparisons"]
            if not c["passed"]
        ],
        "source_analysis_rhs": sum(
            r.get("source_report", {}).get("rhs_evaluations", 0) for r in records
        ),
        "scope": (
            "finite-time metric-dependent singular growth; no asymptotic Lyapunov or chaos claim"
        ),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
