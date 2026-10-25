"""Fixed 31D development block; individual results and every failure persist."""

import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Event, Thread

from gas_shaft31_cases import case, document, dynamic_cases, native
from gas_shaft31_reference import Reference
from run_gas_shaft31_canary import execute

from chemworld.examples.gas_shaft_machine import gas_shaft_machine_world
from chemworld.physchem.gas_reactor_network import (
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.physchem.gas_shaft_machine import gas_shaft_operating_point
from chemworld.runtime import ChemWorldRuntime

DYNAMIC = (
    "amounts",
    "T",
    "P",
    "U",
    "Hin",
    "Hout",
    "W",
    "S",
    "Q_loss",
    "event",
    "first_law",
    "elements",
)
STATIC = ("temperatures", "enthalpies", "flow", "power", "entropy")
INVARIANT = ("amounts", "T", "P", "W", "ports")
WORLD = (
    "one_clock",
    "shaft_sign",
    "no_piston_work",
    "separate_heat",
    "snapshot_equal",
    "fresh_process_equal",
)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf8")


def max_error(left, right):
    if isinstance(left, dict):
        return max((max_error(left[k], right[k]) for k in left), default=0.0)
    if isinstance(left, (list, tuple)):
        return max((max_error(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return abs(float(left) - float(right))


def trajectory(card):
    spec, network, amounts = native(card)
    state = pack_gas_reactor_network_state(spec, network, amounts)
    totals = dict.fromkeys(("Hin", "Hout", "W", "S", "Q", "loss"), 0.0)
    delta_u = dict.fromkeys(("source", "receiver"), 0.0)
    elapsed, samples, reports, resources = 0.0, [], [], []
    stopping = None
    for target in card["sample_times_s"] or [card["duration_s"]]:
        result, report = advance_gas_reactor_network(spec, network, state, target - elapsed)
        state = result.final_state
        elapsed += result.actual_duration_s
        stopping = result.stopping_event
        edge = report["connections"]["shaft-feed"]
        for key, value in {
            "Hin": edge["enthalpy_J"],
            "Hout": edge["delivered_enthalpy_J"],
            "W": edge["shaft_work_J"],
            "S": edge["entropy_generated_J_K"],
            "Q": report["bath_heat_J"],
            "loss": report["heat_loss_J"],
        }.items():
            totals[key] += value
        for name in delta_u:
            delta_u[name] += report["nodes"][name]["internal_energy_change_J"]
        samples.append(
            {
                "time_s": elapsed,
                "amounts": {
                    n: [report["nodes"][n]["amounts_mol"][s] for s in ("A", "B")] for n in delta_u
                },
                "temperatures": {n: report["nodes"][n]["temperature_K"] for n in delta_u},
                "pressures": {n: report["nodes"][n]["pressure_Pa"] for n in delta_u},
                "delta_U": dict(delta_u),
                **totals,
            }
        )
        reports.append(report)
        resources.append(dict(result.solver_diagnostic))
        if stopping is not None:
            break
    return {
        "samples": samples,
        "stopping_event": stopping,
        "reports": reports,
        "resources": resources,
    }


def dynamic_unit(card, row, _):
    row["input"] = card
    row["reference"] = reference = Reference(card).solve()
    row["native"] = actual = trajectory(card)
    pairs = list(zip(actual["samples"], reference["samples"], strict=True))
    errors = {}
    for name, fields, tolerance in (
        ("amounts", ("amounts",), 1e-9),
        ("T", ("temperatures",), 1e-5),
        ("P", ("pressures",), 0.005),
        ("U", ("delta_U",), 2e-5),
        ("Hin", ("Hin",), 2e-5),
        ("Hout", ("Hout",), 2e-5),
        ("W", ("W",), 2e-5),
        ("S", ("S",), 1e-7),
        ("Q_loss", ("Q", "loss"), 2e-5),
    ):
        error = max(max_error(a[f], b[f]) for a, b in pairs for f in fields)
        errors[name] = {"maximum_absolute_error": error, "tolerance": tolerance}
        row["checks"][name] = error <= tolerance
    time_error = max(abs(a["time_s"] - b["time_s"]) for a, b in pairs)
    row["checks"]["event"] = (
        actual["stopping_event"] == reference["stopping_event"] == card.get("expected_event")
        and time_error <= 2e-6
    )
    errors["event"] = {"maximum_time_error_s": time_error, "tolerance_s": 2e-6}
    first_law, elements = True, True
    for r in actual["reports"]:
        e = r["connections"]["shaft-feed"]
        scale = sum(
            abs(v)
            for v in (
                e["enthalpy_J"],
                e["delivered_enthalpy_J"],
                r["bath_heat_J"],
                r["heat_loss_J"],
            )
        )
        first_law &= abs(r["energy_residual_J"]) <= 2e-6 + 5e-6 * scale
        first_law &= all(
            abs(n["energy_residual_J"]) <= 2e-6 + 5e-6 * scale for n in r["nodes"].values()
        )
        first_law &= (
            abs(e["delivered_enthalpy_J"] - e["enthalpy_J"] - e["shaft_work_J"])
            <= 2e-7 + 2e-7 * scale
        )
        elements &= max((abs(v) for v in r["element_residuals_mol"].values()), default=0.0) <= 1e-10
    row["checks"].update(first_law=bool(first_law), elements=bool(elements))
    row["errors"] = errors


def static_unit(card, row, _):
    row["input"] = card
    spec, network, _ = native(card)
    p2 = 200000.0 if spec.shaft_machine.mode == "compressor" else 50000.0
    row["native"] = a = gas_shaft_operating_point(
        spec.shaft_machine,
        network.imported_gas_thermochemistry,
        inlet_temperature_K=400.0,
        inlet_pressure_Pa=100000.0,
        outlet_pressure_Pa=p2,
        mole_fractions={"A": 0.7, "B": 0.3},
        requested_molar_rate_mol_s=0.001,
    )
    import numpy as np

    row["reference"] = b = Reference(card).point(
        400.0, 100000.0, p2, np.array([0.7, 0.3]), request=0.001
    )
    errors = {
        "temperatures": max(
            abs(a["isentropic_temperature_K"] - b["T_s"]),
            abs(a["outlet_temperature_K"] - b["T_out"]),
        ),
        "enthalpies": max(
            abs(a[k] - b[f])
            for k, f in (
                ("inlet_enthalpy_J_mol", "h_in"),
                ("outlet_enthalpy_J_mol", "h_out"),
                ("shaft_work_J_mol", "w"),
            )
        ),
        "flow": abs(a["molar_rate_mol_s"] - b["flow"]),
        "power": abs(a["shaft_power_W"] - b["W"]),
        "entropy": abs(a["entropy_generation_W_K"] - b["S"]),
    }
    tolerances = dict(zip(STATIC, (1e-7, 1e-5, 1e-12, 1e-8, 1e-8), strict=True))
    row["errors"] = {
        k: {"maximum_absolute_error": e, "tolerance": tolerances[k]} for k, e in errors.items()
    }
    row["checks"].update({k: e <= tolerances[k] for k, e in errors.items()})
    row["checks"]["entropy"] &= a["entropy_generation_W_K"] >= -1e-8 and b["S"] >= -1e-8


def invariant_unit(card, row, _):
    base = case(mode=card["mode"])
    base["sample_times_s"] = [1.0]
    alternative = copy.deepcopy(base)
    if card["kind"] == "anchor":
        alternative["spec"]["anchor_vessel_id"] = "receiver"
        alternative["spec"]["temperature_K"] = 350.0
    else:
        alternative["document"] = document(shifts={"A": 10000.0, "B": -7000.0})
    row["input"] = [base, alternative]
    row["baseline"] = first = trajectory(base)
    row["alternative"] = second = trajectory(alternative)
    a, b = first["samples"][-1], second["samples"][-1]
    offset = 0.0
    if card["kind"] == "reference":
        moved = first["reports"][-1]["connections"]["shaft-feed"]["amounts_mol"]
        offset = moved["A"] * 10000.0 - moved["B"] * 7000.0
    errors = {
        "amounts": max_error(a["amounts"], b["amounts"]),
        "T": max_error(a["temperatures"], b["temperatures"]),
        "P": max_error(a["pressures"], b["pressures"]),
        "W": abs(a["W"] - b["W"]),
        "ports": max(abs(b[f] - a[f] - offset) for f in ("Hin", "Hout")),
    }
    tolerances = dict(zip(INVARIANT, (1e-10, 1e-6, 0.005, 2e-6, 2e-6), strict=True))
    row["errors"] = {
        k: {"maximum_absolute_error": e, "tolerance": tolerances[k]} for k, e in errors.items()
    }
    row["checks"].update({k: e <= tolerances[k] for k, e in errors.items()})


def world_unit(card, row, folder):
    mode, heating = card["mode"], card["heating"]
    row["input"] = card
    session = replay = None
    try:
        session = ChemWorldRuntime(composition=gas_shaft_machine_world(mode, with_heater=heating))
        session.reset(seed=31)
        if heating:
            execute(session, "select_vessel", vessel="source")
            execute(session, "queue_lab_heat", target_temperature_K=600.0, duration_s=1.0)
        execute(session, "select_vessel", vessel="receiver")
        result = execute(session, "wait", duration_s=1.0)
        row["wait_info"] = result.info
        ledger = session._state.ledger
        row["checks"].update(
            one_clock=ledger.time_s == 1.0,
            shaft_sign=ledger.flow_shaft_work_J > 0
            if mode == "compressor"
            else ledger.flow_shaft_work_J < 0,
            no_piston_work=ledger.boundary_work_J == 0.0,
            separate_heat=abs(ledger.energy_jacket_J - (2.0 if heating else 0.0)) <= 1e-8,
        )
        payload = json.loads(json.dumps(session.snapshot().to_dict(), allow_nan=False))
        replay = ChemWorldRuntime.from_snapshot(payload)
        source, target = (
            folder / (row["id"] + "-snapshot.json"),
            folder / (row["id"] + "-fresh.json"),
        )
        save(source, payload)
        code = (
            "import json; from pathlib import Path; "
            "from chemworld.runtime import ChemWorldRuntime; "
            f"p=json.loads(Path({str(source)!r}).read_text(encoding='utf8')); "
            "r=ChemWorldRuntime.from_snapshot(p); "
            "z=r.execute({'operation':'wait','duration_s':1.}); "
            "assert z.info['transaction_status']=='committed',z.info; "
            f"p=Path({str(target)!r}); "
            "p.write_text(json.dumps(r._state.to_dict()),encoding='utf8'); r.close()"
        )
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        row["child"] = {
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
        }
        if child.returncode:
            raise RuntimeError("fresh-process continuation failed")
        for runtime in (session, replay):
            execute(runtime, "wait", duration_s=1.0)
        row["checks"]["snapshot_equal"] = session._state.to_dict() == replay._state.to_dict()
        row["checks"]["fresh_process_equal"] = json.loads(
            target.read_text(encoding="utf8")
        ) == json.loads(json.dumps(session._state.to_dict()))
        row["final_state"] = session._state.to_dict()
    finally:
        if session:
            session.close()
        if replay:
            replay.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    units = [(c["id"], dynamic_unit, c, DYNAMIC) for c in dynamic_cases()]
    for family in ("constant", "nasa"):
        for mode in ("compressor", "expander"):
            for eta in (1.0, 0.6):
                c = case(family, mode)
                c["spec"]["shaft_machine"].update(
                    isentropic_efficiency=eta, maximum_shaft_power_W=0.5
                )
                units.append((f"static-{family}-{mode}-{eta}", static_unit, c, STATIC))
    for mode in ("compressor", "expander"):
        for kind in ("anchor", "reference"):
            units.append(
                (
                    f"invariance-{mode}-{kind}",
                    invariant_unit,
                    {"mode": mode, "kind": kind},
                    INVARIANT,
                )
            )
    for mode, heating in (("compressor", False), ("expander", False), ("compressor", True)):
        units.append(
            (f"world-{mode}-heat-{heating}", world_unit, {"mode": mode, "heating": heating}, WORLD)
        )
    start, done, rows = time.monotonic(), Event(), []

    def progress():
        elapsed = time.monotonic() - start
        rate = len(rows) / elapsed
        print(
            f"31D units={len(rows)}/31 "
            f"checks={sum(sum(r['checks'].values()) for r in rows)}/270 "
            f"elapsed={elapsed:.1f}s rate={rate:.3f}/s "
            f"ETA={(31 - len(rows)) / rate if rate else -1:.1f}s",
            flush=True,
        )

    def heartbeat():
        while not done.wait(30):
            progress()

    Thread(target=heartbeat, daemon=True).start()
    for identity, run, card, checks in units:
        row = {"id": identity, "checks": dict.fromkeys(checks, False), "failures": []}
        print("31D start", identity, flush=True)
        unit_start = time.monotonic()
        try:
            run(card, row, args.output)
        except Exception as exc:
            row["failures"].append({"error": repr(exc), "traceback": traceback.format_exc()})
        row["elapsed_s"] = time.monotonic() - unit_start
        row["passed"] = all(row["checks"].values()) and not row["failures"]
        rows.append(row)
        save(args.output / f"{identity}.json", row)
        save(
            args.output / "results.json",
            {
                "development_only": True,
                "cantera_version": "3.2.0",
                "units_planned": 31,
                "checks_planned": 270,
                "units_completed": len(rows),
                "units_passed": sum(r["passed"] for r in rows),
                "checks_passed": sum(sum(r["checks"].values()) for r in rows),
                "elapsed_s": time.monotonic() - start,
                "rows": rows,
            },
        )
        progress()
    done.set()
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
