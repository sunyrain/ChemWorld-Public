"""Fixed 28-unit/224-comparison positive-domain and refusal completion."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from threading import Event, Thread
from time import monotonic

import numpy as np

from chemworld.examples.molecular_cascade import molecular_cascade_example
from chemworld.physchem.molecular_cascade import solve_molecular_cascade
from reference.molecular_cascade32_reference import steady
from reference.run_molecular_cascade32 import check, steady_unit, world_unit


def refusal(temperatures):
    spec, _, _, feeds = molecular_cascade_example(temperatures_K=temperatures, flow_ratio=1.25)
    before = spec.to_dict()
    reference = steady(temperatures, 1.25)
    native = solve_molecular_cascade(spec.cascade, *feeds)
    rows = {}
    check(
        rows,
        "independent_negative_flow",
        not reference["success"]
        and min(f for s in reference["stages"] for f in s["rates"]) < 0
        and reference["component_balance_error"] <= 1e-10,
        {
            "flows": [s["rates"] for s in reference["stages"]],
            "component_error": reference["component_balance_error"],
        },
    )
    check(
        rows,
        "native_refuses_unphysical_solution",
        not native.success and not native.diagnostics.get("stages"),
        native.failure_reason,
    )
    check(
        rows,
        "unchanged_declaration_and_finite_budget",
        before == spec.to_dict()
        and native.diagnostics["root_policy"]["maximum_iterations"] == 100
        and native.diagnostics.get("root", {}).get("iterations", 101) <= 100,
        native.diagnostics.get("root"),
    )
    return {"checks": rows, "native": native.to_dict(), "reference": reference}


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    units, completed = [], []
    conditions = []
    for n in (1, 2, 4, 6):
        for program in ("uniform", "linear"):
            ts = tuple(
                map(
                    float,
                    (300.0,) * n
                    if program == "uniform"
                    else np.linspace(300, 302, n)
                    if n > 1
                    else (302.0,),
                )
            )
            name = f"N{n}-{program}"
            conditions.append((name, ts))
            for ratio in (0.5, 0.75):
                units.append(
                    (f"{name}-ratio{ratio}", 10, lambda ts=ts, ratio=ratio: steady_unit(ts, ratio))
                )
    for kind in ("normal", "source", "receiver_resume", "power"):
        units.append((f"world-{kind}", 10, lambda kind=kind: world_unit(kind, output, ratio=0.75)))
    units += [(f"refuse-{name}-ratio1.25", 3, lambda ts=ts: refusal(ts)) for name, ts in conditions]
    began, stop, live = monotonic(), Event(), {"stage": "starting"}

    def status():
        elapsed = max(monotonic() - began, 1e-9)
        count = len(completed)
        return {
            **live,
            "completed": count,
            "total": 28,
            "elapsed_s": elapsed,
            "units_per_s": count / elapsed,
            "eta_s": (28 - count) * elapsed / count if count else None,
        }

    def heartbeat():
        while not stop.wait(30):
            print(json.dumps(status()), flush=True)

    thread = Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        for name, count, function in units:
            live["stage"] = name
            print(json.dumps(status()), flush=True)
            try:
                row = function()
                assert len(row["checks"]) == count
            except Exception:
                row = {"checks": {}, "exception": traceback.format_exc()}
            row.update(unit=name, declared_comparisons=count)
            (output / f"{name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            # The original helper identifies unavailable calculations explicitly.
            for c in row["checks"].values():
                c["assessed"] = c.get("detail") != "calculation unavailable"
            completed.append(
                {
                    k: v
                    for k, v in row.items()
                    if k not in {"native", "declaration", "info", "first_info"}
                }
            )
            print(
                json.dumps(
                    {**status(), "passed": sum(c["passed"] for c in row["checks"].values())}
                ),
                flush=True,
            )
    finally:
        stop.set()
        thread.join()
    checks = [c for r in completed for c in r["checks"].values()]
    summary = {
        "scope": "development_valid_feed_cascade_and_explicit_domain_refusals",
        "declared_units": 28,
        "executed_units": len(completed),
        "passed_units": sum(
            len(r["checks"]) == r["declared_comparisons"]
            and all(c["passed"] for c in r["checks"].values())
            for r in completed
        ),
        "declared_comparisons": 224,
        "assessed_comparisons": sum(c["assessed"] for c in checks),
        "passed_comparisons": sum(c["passed"] for c in checks),
        "unexecuted_comparisons": 224 - sum(c["assessed"] for c in checks),
        "elapsed_s": monotonic() - began,
        "units": completed,
        "failures": [
            {"unit": r["unit"], "check": name, **c}
            for r in completed
            for name, c in r["checks"].items()
            if not c["passed"]
        ],
        "exceptions": [
            {"unit": r["unit"], "exception": r["exception"]} for r in completed if "exception" in r
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "units"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
