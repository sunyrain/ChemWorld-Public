"""Fixed 24-unit/536-comparison 49E block with retained failures and progress."""

import argparse
import json
import math
import sys
import threading
import time
import traceback
from dataclasses import replace
from functools import partial
from pathlib import Path

from chemworld.examples.gac_design import gac_design_example
from chemworld.physchem.aqueous_adsorption import aqueous_adsorption_profile
from chemworld.physchem.gac_design import evaluate_gac_design
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_adsorption import unpack_aqueous_adsorption
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.gac_design import analyze_gac_design_world
from reference.gac_design49_fixtures import METRICS, commit, numerical_cases, world_case
from reference.gac_design49_reference import upstream


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def compare(name, actual, expected, *, atol=1e-9, rtol=2e-7):
    allowance = atol + rtol * abs(expected)
    error = abs(actual - expected)
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "allowance": allowance,
        "error_over_allowance": error / allowance,
        "passed": math.isfinite(actual) and error <= allowance,
    }


def equality(name, actual, expected):
    return {"name": name, "actual": actual, "expected": expected, "passed": actual == expected}


def numeric(spec, feed):
    reference = upstream(spec, feed)
    native = evaluate_gac_design(spec, feed)
    floors = {
        "average_concentration_ratio": 1e-11,
        "film_transfer_m_s": 1e-14,
        "surface_diffusion_m2_s": 1e-22,
        "carbon_usage_kg_s": 1e-13,
        "average_outlet_target_kg_s": 1e-13,
    }
    checks = [
        compare(key, native["metrics"][key], reference["metrics"][key], atol=floors.get(key, 1e-9))
        for key in METRICS
    ]
    for index, (a, b) in enumerate(
        zip(native["breakthrough"], reference["breakthrough"], strict=True)
    ):
        for key in ("time_s", "concentration_ratio"):
            checks.append(
                compare(
                    f"curve:{index}:{key}",
                    a[key],
                    b[key],
                    atol=1e-11 if key == "concentration_ratio" else 1e-9,
                )
            )
    return {"checks": checks, "native": native, "upstream": reference}


def actual_world(shared, advanced):
    runtime, spec = world_case(shared, advanced)
    before = encoded(runtime.snapshot().to_dict())
    state = runtime._state
    states = {**state.inactive_vessels, state.vessel_id: state.replace(inactive_vessels={})}
    anchor = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    record, original, inventory = unpack_aqueous_adsorption(anchor)
    requested = record.settings["flow_control"]["requested_flow_m3_s"]
    point = aqueous_adsorption_profile(original, inventory, requested)
    volume = original.volume_m3(inventory.amounts_mol[0])
    concentration = inventory.amounts_mol[0]["A"] * original.masses_kg_mol["A"] / volume
    q = point["actual_flow_m3_s"]
    available = max(0, volume - original.pickup_volume_L / 1000)
    report = analyze_gac_design_world(state, spec)
    feed, query = report["feed"], report["world_query"]
    expected = {
        "C": (feed["concentration_kg_m3"], concentration),
        "Q": (feed["volumetric_flow_m3_s"], q),
        "V": (query["feed_volume_m3"], volume),
        "T": (feed["temperature_K"], original.temperature_K),
        "P": (feed["pressure_Pa"], original.pressure_Pa),
        "available": (query["available_feed_volume_m3"], available),
        "duration": (query["feed_duration_before_pickup_s"], available / q),
    }
    checks = [compare(k, a, b, atol=1e-10, rtol=1e-9) for k, (a, b) in expected.items()]
    checks.append(equality("coverage", query["source_covers_replacement"], advanced))
    checks.append(
        equality("complete_state_unchanged", encoded(runtime.snapshot().to_dict()) == before, True)
    )
    checks.append(equality("strict_json", json.loads(encoded(report)) == report, True))
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.from_dict(json.loads(before)))
    checks.append(
        equality("restored_query", analyze_gac_design_world(restored._state, spec) == report, True)
    )
    checks.append(equality("actual_pump_limited", 0 < q < requested, True))
    return {
        "checks": checks,
        "native": report,
        "world_snapshot_before": json.loads(before),
        "world_snapshot_after": runtime.snapshot().to_dict(),
    }


def rejected(which):
    spec, feed = gac_design_example()
    mutations = {
        "concentration": lambda: evaluate_gac_design(spec, replace(feed, concentration_kg_m3=11.0)),
        "temperature": lambda: evaluate_gac_design(spec, replace(feed, temperature_K=350.0)),
        "pressure": lambda: evaluate_gac_design(spec, replace(feed, pressure_Pa=3e6)),
        "biot": lambda: evaluate_gac_design(
            replace(spec, empirical=replace(spec.empirical, biot_bounds=(1e-9, 1e-8))), feed
        ),
        "ratio": lambda: replace(spec, replacement_concentration_ratio=0.999),
        "undersized": lambda: evaluate_gac_design(
            replace(spec, bed=replace(spec.bed, empty_bed_contact_time_s=1e-6)), feed
        ),
    }
    if which == "stopped":
        runtime, design = world_case(blocked=True)
        call = partial(analyze_gac_design_world, runtime._state, design)
        fragment = "positive actual"
    elif which == "rinse":
        runtime, design = world_case()
        commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="rinse")
        call = partial(analyze_gac_design_world, runtime._state, design)
        fragment = "not rinse"
    else:
        call = mutations[which]
        fragment = {"biot": "Biot", "ratio": "replacement ratio", "undersized": "minimum EBCT"}.get(
            which, which
        )
    try:
        call()
    except ValueError as exc:
        return {
            "checks": [equality("intended_rejection", fragment in str(exc), True)],
            "error": str(exc),
        }
    return {"checks": [equality("intended_rejection", False, True)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.reference_directory))
    work = [
        (name, 40, lambda s=spec, f=feed: numeric(s, f)) for name, spec, feed in numerical_cases()
    ]
    work += [
        (f"world-shared-{s}-advanced-{a}", 12, lambda s=s, a=a: actual_world(s, a))
        for s in (False, True)
        for a in (False, True)
    ]
    work += [
        (f"reject-{name}", 1, lambda name=name: rejected(name))
        for name in (
            "concentration",
            "temperature",
            "pressure",
            "biot",
            "ratio",
            "undersized",
            "stopped",
            "rinse",
        )
    ]
    report = {
        "mode": "development",
        "expected_units": 24,
        "expected_comparisons": 536,
        "planned_units": [{"id": name, "comparisons": n} for name, n, _ in work],
        "scalar_metrics": list(METRICS),
        "attempt": args.output.name,
        "units": [],
    }
    started = time.monotonic()
    stopped = threading.Event()

    def progress():
        elapsed = time.monotonic() - started
        count = len(report["units"])
        rate = count / elapsed if elapsed else 0
        eta = (24 - count) / rate if rate else None
        print(
            f"49E units={count}/24 elapsed={elapsed:.1f}s rate={rate:.3f}/s ETA={eta}", flush=True
        )

    def heartbeat():
        while not stopped.wait(25):
            progress()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    for name, planned, call in work:
        begin = time.monotonic()
        unit = {"id": name, "expected_comparisons": planned, "checks": []}
        try:
            unit.update(call())
        except Exception:
            unit["failure"] = traceback.format_exc()
        unit["elapsed_s"] = time.monotonic() - begin
        unit["executed_comparisons"] = len(unit["checks"])
        unit["passed_comparisons"] = sum(c["passed"] for c in unit["checks"])
        unit["passed"] = unit["passed_comparisons"] == planned and "failure" not in unit
        (args.output / f"{name}.json").write_text(
            json.dumps(unit, indent=2, allow_nan=False), encoding="utf-8"
        )
        report["units"].append(unit)
        print(
            f"{name}: {unit['passed_comparisons']}/{planned} elapsed={unit['elapsed_s']:.3f}s",
            flush=True,
        )
    stopped.set()
    thread.join()
    report.update(
        elapsed_s=time.monotonic() - started,
        passed_units=sum(u["passed"] for u in report["units"]),
        executed_comparisons=sum(u["executed_comparisons"] for u in report["units"]),
        passed_comparisons=sum(u["passed_comparisons"] for u in report["units"]),
    )
    report["passed"] = report["passed_units"] == 24 and report["passed_comparisons"] == 536
    (args.output / "RESULTS.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    progress()
    print(
        {k: v for k, v in report.items() if k not in {"units", "planned_units", "scalar_metrics"}},
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
