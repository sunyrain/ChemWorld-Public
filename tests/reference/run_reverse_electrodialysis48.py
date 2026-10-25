"""Frozen 24-unit / 240-comparison passive periodic RED development block."""

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from chemworld.examples.reverse_electrodialysis import example_report
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.reverse_electrodialysis import (
    ReverseElectrodialysisSpec,
    advance_reverse_electrodialysis,
    reverse_electrodialysis_inventory,
    reverse_electrodialysis_point,
)
from chemworld.runtime import ChemWorldRuntime


def exact(name, value, **details):
    return {"name": name, "passed": bool(value), **details}


def numeric(name, actual, expected, *, atol=1e-9, rtol=0.003):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return exact(
        name,
        np.all(np.isfinite(a)) and np.allclose(a, b, atol=atol, rtol=rtol),
        actual=a.tolist(),
        expected=b.tolist(),
        atol=atol,
        rtol=rtol,
        maximum_absolute_error=float(np.max(np.abs(a - b), initial=0)),
    )


def vector(ns, ids):
    return [[n[k] for k in ids] for n in ns]


def charge(spec, ns):
    return [sum(spec.properties.charges[k] * n[k] for k in spec.properties.ion_ids) for n in ns]


def profiles(spec, row, *, native):
    return [
        [[r[k] for k in spec.properties.ion_ids] for r in row[m]["pore_concentrations_mol_m3"]]
        if native
        else row[m]["pore_concentrations_mol_m3"]
        for m in ("aem", "cem")
    ]


def point_checks(spec, ns, case, native, independent):
    from reference.electrodialysis48_reference import F, R

    ions = spec.properties.ion_ids
    circuit = native["circuit_residual_V"]
    analytic = True
    details = {}
    if case["kind"] == "analytic":
        cd, cc = [n["Na+"] / (n["H2O"] * 18e-6) for n in ns]
        expected = (
            2
            * R
            * spec.cell_pair.temperature_K
            / F
            * (math.asinh(100 / (2 * cd)) - math.asinh(100 / (2 * cc)))
        )
        analytic = numeric("analytic", native["voltage_V"], expected)["passed"]
        details = {"analytic_finite_charge_voltage_V": expected}
    if case.get("equal"):
        analytic = max(abs(v) for v in native["diluate_rates_mol_s"].values()) < 1e-10
        analytic &= abs(native["voltage_V"]) < 1e-9
    return [
        exact("converged", native["status"] == "converged"),
        numeric(
            "terminal-IV",
            [native[k] for k in ("current_A", "voltage_V")],
            [independent[k] for k in ("current_A", "voltage_V")],
        ),
        numeric(
            "all-ion-flow",
            [native["diluate_rates_mol_s"][k] for k in ions],
            [independent["diluate_rates_mol_s"][k] for k in ions],
        ),
        numeric(
            "water-flow",
            [native[m]["water_flux_m_s"] for m in ("aem", "cem")],
            [independent[m]["water_flux_m_s"] for m in ("aem", "cem")],
        ),
        numeric("external-load-power", native["load_power_W"], independent["load_power_W"]),
        numeric(
            "all-pore-concentrations",
            profiles(spec, native, native=True),
            profiles(spec, independent, native=False),
        ),
        exact(
            "charge-and-material",
            max(
                map(
                    abs,
                    charge(
                        spec, (native["diluate_rates_mol_s"], native["concentrate_rates_mol_s"])
                    ),
                )
            )
            < 1e-10
            and all(
                abs(native["diluate_rates_mol_s"][k] + native["concentrate_rates_mol_s"][k]) < 1e-10
                for k in spec.properties.component_ids
            ),
        ),
        exact(
            "circuit-Gdot-and-analytic-limit",
            (circuit is None or abs(circuit) < 1e-9)
            and abs(native["terminal_load_power_residual_W"]) < 1e-9
            and abs(native["free_energy_power_residual_W"]) < 1e-9
            and analytic,
            **details,
        ),
        numeric(
            "independent-dissipation-Gdot",
            [native[k] for k in ("dissipation_W", "gibbs_energy_rate_W")],
            [independent[k] for k in ("dissipation_W", "gibbs_energy_rate_W")],
        ),
        exact(
            "finite-JSON-and-declaration-codec",
            json.loads(json.dumps(native, allow_nan=False)) == native
            and ReverseElectrodialysisSpec.from_dict(spec.to_dict()).to_dict() == spec.to_dict(),
        ),
    ]


def run_point(spec, ns, case, record):
    from reference.reverse_electrodialysis48_reference import point

    before = json.dumps([spec.to_dict(), ns], sort_keys=True)
    start = time.monotonic()
    native = reverse_electrodialysis_point(spec, ns, case["resistance"])
    native_s = time.monotonic() - start
    independent = point(spec.cell_pair, ns, case["resistance"])
    record.update(
        native=native,
        reference=independent,
        native_elapsed_s=native_s,
        reference_elapsed_s=time.monotonic() - start - native_s,
    )
    checks = point_checks(spec, ns, case, native, independent)
    checks[-1]["passed"] &= before == json.dumps([spec.to_dict(), ns], sort_keys=True)
    record["comparisons"] = checks


def run_grid(spec, ns, case, record):
    from reference.reverse_electrodialysis48_reference import point

    levels = []
    for segments in (4, 8, 16):
        pair = replace(
            spec.cell_pair,
            aem=replace(spec.cell_pair.aem, segments=segments),
            cem=replace(spec.cell_pair.cem, segments=segments),
        )
        selected = replace(spec, cell_pair=pair)
        native = reverse_electrodialysis_point(selected, ns, 100)
        independent = point(pair, ns, 100)
        terminal_error = max(
            abs(native[k] - independent[k]) / abs(independent[k])
            for k in ("current_A", "voltage_V")
        )
        a, b = (
            np.asarray(profiles(selected, native, native=True)),
            np.asarray(profiles(selected, independent, native=False)),
        )
        profile_error = float(np.max(np.abs(a - b) / np.abs(b)))
        levels.append(
            {
                "segments": segments,
                "native": native,
                "reference": independent,
                "terminal_relative_error": terminal_error,
                "profile_relative_error": profile_error,
            }
        )
    record["levels"] = levels
    checks = point_checks(spec, ns, case, levels[-1]["native"], levels[-1]["reference"])
    checks[0] = exact(
        "all-grids-converged", all(r["native"]["status"] == "converged" for r in levels)
    )
    for slot, key in ((1, "terminal_relative_error"), (5, "profile_relative_error")):
        errors = [r[key] for r in levels]
        checks[slot] = exact(
            key + "-decreases-4-8-16",
            errors[0] > errors[1] > errors[2] and errors[2] < 0.003,
            errors=errors,
        )
    record["comparisons"] = checks


def run_finite(spec, ns, case, record):
    from reference.reverse_electrodialysis48_reference import finite

    ramp = case.get("ramp")
    program = ScalarControlProgram(
        ((0, case["resistance"]),) if not ramp else ((0, ramp[0]), (ramp[1], case["resistance"])),
        "Ohm",
    )
    initial = reverse_electrodialysis_inventory(spec, *ns)
    start = time.monotonic()
    solved, receipt = advance_reverse_electrodialysis(
        spec,
        initial,
        program,
        5,
        progress=lambda t, count: print(f"native time={t:.3f}s rhs={count}", flush=True),
    )
    native_s = time.monotonic() - start
    independent = finite(
        spec.cell_pair,
        ns,
        case["resistance"],
        5,
        ramp=ramp,
        progress=lambda c: print(c, flush=True),
    )
    final = solved.final_state
    record.update(
        native={"advance": asdict(solved), "receipt": receipt},
        reference=independent,
        native_elapsed_s=native_s,
        reference_elapsed_s=time.monotonic() - start - native_s,
    )
    duration = numeric(
        "duration-and-physical-stop", solved.actual_duration_s, independent["actual_duration_s"]
    )
    duration["passed"] &= solved.stopping_event == independent["stopping_event"]
    duration.update(native_stop=solved.stopping_event, reference_stop=independent["stopping_event"])
    record["comparisons"] = [
        duration,
        numeric(
            "all-finite-amounts",
            vector(final.amounts_mol, spec.properties.component_ids),
            vector(independent["amounts_mol"], spec.properties.component_ids),
        ),
        numeric("original-liquid-H", final.enthalpies_J, independent["enthalpies_J"]),
        numeric(
            "generated-work",
            receipt["load_energy_delivered_J"],
            independent["load_energy_delivered_J"],
        ),
        numeric(
            "bath-heat-input",
            receipt["bath_heat_into_liquid_J"],
            independent["load_energy_delivered_J"],
        ),
        numeric(
            "independent-free-energy-decrease",
            receipt["free_energy_decrease_J"],
            independent["free_energy_decrease_J"],
        ),
        numeric(
            "independent-irreversible-dissipation",
            receipt["free_energy_decrease_minus_load_J"],
            independent["dissipation_integral_J"],
        ),
        exact(
            "closed-component-inventory",
            max(map(abs, solved.component_residuals_mol.values())) < 1e-10,
        ),
        numeric("each-tank-charge", charge(spec, final.amounts_mol), [0, 0], rtol=0, atol=1e-10),
        exact(
            "first-law-and-finite-JSON",
            abs(solved.enthalpy_residual_J) < 1e-8
            and receipt["electrical_work_into_liquid_J"] == -receipt["load_energy_delivered_J"]
            and bool(json.dumps(record["native"], allow_nan=False)),
        ),
    ]


def world_amounts(state):
    return state["species_amounts"], state["inactive_vessels"]["concentrate"]["species_amounts"]


def run_world(spec, ns, case, record, output):
    from reference.reverse_electrodialysis48_reference import finite

    consumer = example_report(spec, ns, shared_clock=case["shared"])
    record["consumer"] = consumer
    if not consumer["complete"]:
        failed = consumer["transactions"][-1]
        raise RuntimeError("world operation did not commit: " + failed["action"]["operation"])
    first = finite(spec.cell_pair, ns, 100, 2, progress=lambda c: print(c, flush=True))
    second = finite(
        spec.cell_pair,
        first["amounts_mol"],
        200,
        2,
        ramp=(100, 1),
        progress=lambda c: print(c, flush=True),
    )
    record["reference"] = {"first": first, "second": second}
    work = first["load_energy_delivered_J"] + second["load_energy_delivered_J"]
    prefix = output / (case["name"] + "-prefix.json")
    fresh = output / (case["name"] + "-fresh.json")
    prefix.write_text(
        json.dumps(
            {"snapshot": consumer["prefix_snapshot"], "action": consumer["tail_action"]},
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        str(Path(__file__).resolve()),
        "--resume",
        str(prefix),
        "--output",
        str(fresh),
    ]
    child = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
    record["fresh_process"] = {
        "command": command,
        "returncode": child.returncode,
        "stdout": child.stdout,
        "stderr": child.stderr,
    }
    fresh_data = json.loads(fresh.read_text(encoding="utf-8")) if fresh.exists() else {}
    initial, final = consumer["initial_state"], consumer["final_state"]
    ledger = final["ledger"]
    measured = {
        c["channel_id"]: c
        for c in consumer["transactions"][3]["info"]["native_observation"]["channels"]
    }
    final_ns = world_amounts(final)
    thermal = [
        final["thermal"]["reactor"]["heat_loss_J"],
        final["inactive_vessels"]["concentrate"]["thermal"]["concentrate"]["heat_loss_J"],
    ]
    conservation = [
        sum(n[k] for n in final_ns) - sum(n[k] for n in ns) for k in spec.properties.component_ids
    ]
    record["comparisons"] = [
        exact(
            "all-real-actions-committed",
            all(t["info"]["transaction_status"] == "committed" for t in consumer["transactions"])
            and all(t["transaction_status"] == "committed" for t in consumer["tail_transactions"]),
        ),
        exact(
            "actual-meter-and-four-second-clock",
            ledger["time_s"] == 4
            and measured["terminal_potential"]["value"] == consumer["prefix_terminal"]["voltage_V"]
            and measured["electrical_current"]["value"] == consumer["prefix_terminal"]["current_A"]
            and all(c["sampled_at"]["time_s"] == 2 for c in measured.values()),
        ),
        numeric(
            "independent-final-inventories",
            vector(final_ns, spec.properties.component_ids),
            vector(second["amounts_mol"], spec.properties.component_ids),
        ),
        numeric("signed-work-once", ledger["flow_electrical_work_J"], -work),
        numeric("signed-bath-once", ledger["heat_loss_J"], -work),
        exact(
            "no-electric-refund-or-duplicate-heat",
            ledger["cost"] >= initial["ledger"]["cost"] and ledger["energy_jacket_J"] == 0,
        ),
        numeric("two-original-bath-owners", thermal, [-work / 2, -work / 2]),
        exact("exact-complete-JSON-resume", consumer["exact_json_resume"]),
        exact(
            "fresh-process-complete-snapshot",
            child.returncode == 0
            and fresh_data.get("snapshot") == consumer["final_snapshot"]
            and fresh_data.get("transaction_status") == "committed",
        ),
        exact(
            "material-charge-and-first-law",
            max(map(abs, conservation)) < 1e-10
            and max(map(abs, charge(spec, final_ns))) < 1e-10
            and abs(ledger["flow_electrical_work_J"] - ledger["heat_loss_J"]) < 1e-8,
        ),
    ]


def run_stops(spec, ns, case, record):
    cases = (
        [("ion_minimum:0", spec)]
        if case["kind"] == "ion-stop"
        else [
            (
                "current_compliance",
                replace(spec, cell_pair=replace(spec.cell_pair, maximum_current_A=1e-20)),
            ),
            (
                "voltage_compliance",
                replace(spec, cell_pair=replace(spec.cell_pair, maximum_voltage_V=1e-20)),
            ),
            ("load_power_limit", replace(spec, maximum_load_power_W=1e-20)),
            ("bath_heating_limit", replace(spec, maximum_bath_heating_W=1e-20)),
        ]
    )
    rows = []
    for stop, selected in cases:
        initial = reverse_electrodialysis_inventory(selected, *ns)
        solved, receipt = advance_reverse_electrodialysis(
            selected, initial, ScalarControlProgram(((0, 100),), "Ohm"), 5
        )
        rows.append(
            {
                "expected_stop": stop,
                "model": selected.to_dict(),
                "advance": asdict(solved),
                "receipt": receipt,
            }
        )
    record["subcases"] = rows
    checks = [
        ("expected-stop", lambda r: r["advance"]["stopping_event"] == r["expected_stop"]),
        ("zero-duration", lambda r: r["advance"]["final_state"]["time_s"] == 0),
        (
            "unchanged-n",
            lambda r: (
                r["advance"]["final_state"]["amounts_mol"]
                == r["advance"]["initial_state"]["amounts_mol"]
            ),
        ),
        (
            "unchanged-H",
            lambda r: (
                r["advance"]["final_state"]["enthalpies_J"]
                == r["advance"]["initial_state"]["enthalpies_J"]
            ),
        ),
        ("zero-load-work", lambda r: r["receipt"]["load_energy_delivered_J"] == 0),
        ("zero-bath", lambda r: r["receipt"]["bath_heat_into_liquid_J"] == 0),
        (
            "conservation",
            lambda r: max(map(abs, r["advance"]["component_residuals_mol"].values())) < 1e-10,
        ),
        (
            "charge",
            lambda r: (
                max(map(abs, charge(spec, r["advance"]["final_state"]["amounts_mol"]))) < 1e-10
            ),
        ),
        ("first-law", lambda r: abs(r["advance"]["enthalpy_residual_J"]) < 1e-8),
        ("finite-JSON", lambda r: bool(json.dumps(r, allow_nan=False))),
    ]
    record["comparisons"] = [
        exact(
            name,
            all(fn(r) for r in rows),
            subcases=[{"stop": r["expected_stop"], "passed": bool(fn(r))} for r in rows],
        )
        for name, fn in checks
    ]


def run(output):
    from reference.reverse_electrodialysis48_cases import CASES, build

    output.mkdir(parents=True, exist_ok=False)
    rows, started = [], time.monotonic()
    for i, case in enumerate(CASES):
        record = {"unit": case["name"], "case": case, "comparisons": []}
        print(f"48E starting {i + 1}/24 {case['name']}", flush=True)
        before = time.monotonic()
        try:
            spec, ns = build(case)
            record.update(model=spec.to_dict(), initial_amounts=ns)
            if case["kind"] in {"point", "analytic"}:
                run_point(spec, ns, case, record)
            elif case["kind"] == "grid":
                run_grid(spec, ns, case, record)
            elif case["kind"] == "finite":
                run_finite(spec, ns, case, record)
            elif case["kind"] == "world":
                run_world(spec, ns, case, record, output)
            else:
                run_stops(spec, ns, case, record)
        except Exception as error:
            record.update(
                exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
            )
        while len(record["comparisons"]) < 10:
            record["comparisons"].append(
                exact("unexecuted-slot-" + str(len(record["comparisons"])), False)
            )
        assert len(record["comparisons"]) == 10
        record.update(
            elapsed_s=time.monotonic() - before,
            passed=all(c["passed"] for c in record["comparisons"]),
        )
        rows.append(record)
        (output / f"{i + 1:02}-{case['name']}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.monotonic() - started
        print(
            f"48E completed {i + 1}/24 passed={record['passed']} {record.get('exception', '')}; "
            f"throughput={(i + 1) / elapsed:.2f} unit/s ETA={(23 - i) * elapsed / (i + 1):.1f}s",
            flush=True,
        )
    summary = {
        "evidence_class": "synthetic-development",
        "units": 24,
        "comparisons": 240,
        "units_passed": sum(r["passed"] for r in rows),
        "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
        "elapsed_s": time.monotonic() - started,
        "unit_results": [
            {
                "unit": r["unit"],
                "passed": r["passed"],
                "elapsed_s": r["elapsed_s"],
                "comparisons_passed": sum(c["passed"] for c in r["comparisons"]),
            }
            for r in rows
        ],
        "failures": [
            {"unit": r["unit"], "comparison": c, "exception": r.get("exception")}
            for r in rows
            for c in r["comparisons"]
            if not c["passed"]
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(
        json.dumps({k: summary[k] for k in ("units_passed", "comparisons_passed", "elapsed_s")}),
        flush=True,
    )


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.resume:
        payload = json.loads(args.resume.read_text(encoding="utf-8"))
        runtime = ChemWorldRuntime.from_snapshot(payload["snapshot"])
        try:
            result = runtime.execute(payload["action"])
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(
                    {
                        "transaction_status": result.info["transaction_status"],
                        "snapshot": runtime.snapshot().to_dict(),
                    },
                    stream,
                    allow_nan=False,
                )
        finally:
            runtime.close()
    else:
        run(args.output)


if __name__ == "__main__":
    main()
