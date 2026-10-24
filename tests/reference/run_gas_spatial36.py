"""Fixed 24-unit sealed spatial gas development block; never replaces evidence."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from math import exp, isclose, log, pi, sin
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import gas_spatial36_cases as cases
import gas_spatial36_reference as oracle

from chemworld.physchem.gas_reactor_network import (
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


def json_default(value):
    return value.tolist() if hasattr(value, "tolist") else value.item()


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, kind="exact"):
        limits = {
            "n": (5e-6, 3e-10),
            "P": (5e-6, 2e-3),
            "J": (8e-6, 5e-6),
            "balance": (0.0, 3e-10),
            "time": (1e-5, 2e-6),
            "velocity": (5e-6, 1e-9),
        }
        passed = (
            actual == expected
            if kind == "exact"
            else isclose(actual, expected, rel_tol=limits[kind][0], abs_tol=limits[kind][1])
        )
        self.checks.append(
            {
                "label": label,
                "actual": actual,
                "expected": expected,
                "kind": kind,
                "passed": bool(passed),
            }
        )


def check_balances(record, card, actual):
    rows = list(actual["nodes"].values())
    total = sum(sum(r["amounts_mol"].values()) for r in rows)
    for name, row in actual["nodes"].items():
        record.check(
            f"{name}:uniform-density",
            sum(row["amounts_mol"].values()) - total / card["cells"],
            0.0,
            "balance",
        )
        record.check(f"{name}:first-law", row["energy_residual_J"], 0.0, "J")
        record.check(f"{name}:fixed-T", row["temperature_K"], card["temperature"])
    for key, value in actual["element_residuals_mol"].items():
        record.check(f"element:{key}", value, 0.0, "balance")
    record.check("group-first-law", actual["energy_residual_J"], 0.0, "J")
    record.check("node-bath-sum", sum(r["bath_heat_J"] for r in rows), actual["bath_heat_J"], "J")
    record.check(
        "zero-boundary",
        actual["spatial"]["boundary_species_rates_mol_s"],
        {s: dict.fromkeys(card["species"], 0.0) for s in ("left", "right")},
    )
    record.check(
        "low-Mach-domain",
        actual["spatial"]["maximum_mach"] <= card["maximum_mach"] * (1 + 1e-9),
        True,
    )


def physical(record, card, *, analytic=False):
    spec, network, amounts = cases.native(card)
    initial = pack_gas_reactor_network_state(spec, network, amounts)
    event = card["id"] == "pressure-event"
    reference = oracle.solve(card, cases.TIMES, event)
    record.evidence.update(input=card, reference=reference, native=[])
    for sample in reference["samples"]:
        result, actual = advance_gas_reactor_network(
            spec, network, initial, 1 if event else sample["time_s"]
        )
        record.evidence["native"].append({"diagnostic": actual, "solver": result.solver_diagnostic})
        record.check("time", result.actual_duration_s, sample["time_s"], "time")
        check_balances(record, card, actual)
        for k, name in enumerate(spec.spatial_transport.cell_vessel_ids):
            row = actual["nodes"][name]
            for species, amount in sample["amounts"][k].items():
                record.check(f"{name}:{species}", row["amounts_mol"][species], amount, "n")
            record.check(f"{name}:P", row["pressure_Pa"], sample["pressure_Pa"], "P")
            record.check(f"{name}:bath", row["bath_heat_J"], sample["bath"][k], "J")
        for j, (identity, _, _) in enumerate(spec.spatial_transport.topology):
            native, ref = actual["connections"][identity], sample["faces"][j]
            face = actual["spatial"]["faces"][identity]
            for species in card["species"]:
                index = list(card["species"]).index(species)
                record.check(
                    f"{identity}:total:{species}",
                    native["amounts_mol"][species],
                    ref["amounts"][species],
                    "n",
                )
                record.check(
                    f"{identity}:diffusion:{species}",
                    native["diffusion_mol"][species],
                    ref["diffusion"][species],
                    "n",
                )
                record.check(
                    f"{identity}:rate:{species}",
                    face["amount_rates_mol_s"][species],
                    ref["rates"]["total"][index],
                    "n",
                )
            record.check(f"{identity}:bulk", native["bulk_mol"], ref["bulk"], "n")
            record.check(f"{identity}:H", native["enthalpy_J"], ref["H"], "J")
            record.check(
                f"{identity}:diffusion-H", native["diffusion_enthalpy_J"], ref["diffusion_H"], "J"
            )
            record.check(
                f"{identity}:molar-velocity",
                face["molar_velocity_m_s"],
                ref["rates"]["molar_velocity"],
                "velocity",
            )
            record.check(
                f"{identity}:mass-velocity",
                face["mass_velocity_m_s"],
                ref["rates"]["mass_velocity"],
                "velocity",
            )
        if analytic:
            check_analytic(record, card, actual, result.actual_duration_s)
        if event:
            record.check("event-kind", result.stopping_event, "spatial:maximum_pressure")
            record.check(
                "analytic-stop",
                result.actual_duration_s,
                -log(2 - card["maximum_pressure"] / card["pressure"]) / card["kr"],
                "time",
            )


def check_analytic(record, card, actual, t):
    volume = card["length"] * card["area"] / card["cells"]
    n0 = card["pressure"] * volume / (cases.R * card["temperature"])
    if card["id"] == "pure":
        expected = {"I": n0}
    elif card["id"] == "association":
        a = n0 / (1 + 2 * card["kf"] * n0 / volume * t)
        expected = {"A": a, "B": (n0 - a) / 2, "I": 0.0}
    else:
        b = n0 * exp(-card["kr"] * t)
        expected = {"A": 2 * (n0 - b), "B": b, "I": 0.0}
    for name, row in actual["nodes"].items():
        q = 0.0
        k = int(name.split("-")[-1])
        for species, n in expected.items():
            record.check(f"analytic:{name}:{species}", row["amounts_mol"][species], n, "n")
            u = (
                card["species"][species]["h"]
                + card["species"][species]["cp"] * (card["temperature"] - 298.15)
                - cases.R * card["temperature"]
            )
            q += (n - n0 * card["fractions"][k][species]) * u
        record.check(f"analytic:{name}:bath", row["bath_heat_J"], q, "J")


def grid(record, cells):
    card = cases.case("grid", cells=cells)
    spec, network, amounts = cases.native(card)
    initial = pack_gas_reactor_network_state(spec, network, amounts)
    d = oracle.diffusivity(card, "A", "B", card["pressure"])
    errors, outputs = [], []
    for theta in (0.02, 0.1, 0.3):
        t = theta * card["length"] ** 2 / d
        result, actual = advance_gas_reactor_network(spec, network, initial, t)
        check_balances(record, card, actual)
        for k, name in enumerate(spec.spatial_transport.cell_vessel_ids):
            row = actual["nodes"][name]["amounts_mol"]
            fraction = row["A"] / sum(row.values())
            expected = 0.5 + 0.25 * cells / pi * (
                sin(pi * (k + 1) / cells) - sin(pi * k / cells)
            ) * exp(-pi * pi * theta)
            errors.append(abs(fraction - expected) / 0.25)
        outputs.append(
            {"theta": theta, "time_s": t, "diagnostic": actual, "solver": result.solver_diagnostic}
        )
    record.evidence.update(input=card, maximum_normalized_error=max(errors), native=outputs)
    if cells == 32:
        record.check("32-cell-continuum-error", max(errors) <= 3.5e-4, True)
    return max(errors)


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    assert result.info["transaction_status"] == "committed", {
        k: result.info.get(k) for k in ("rollback_reason", "error_message", "outcome_diagnostics")
    }
    return result


CONTINUATION = [
    {"operation": "wait", "duration_s": 0.3},
    {"operation": "select_vessel", "vessel": "cell-001"},
    {"operation": "wait", "duration_s": 0.5},
]


def world(record, kind, directory):
    card = cases.case()
    spec, network, amounts = cases.native(card)
    composition = gas_reactor_network_world(spec, network, amounts)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=7)
        execute(runtime, "select_vessel", vessel="cell-002")
        execute(runtime, "wait", duration_s=0.2)
        snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
        record.evidence["snapshot"] = snapshot
        if kind == "rollback":
            before = runtime._state.to_dict()
            with patch(
                "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network",
                side_effect=RuntimeError("36B injected failure"),
            ):
                result = runtime.execute({"operation": "wait", "duration_s": 0.4})
            after = runtime._state.to_dict()
            record.evidence.update(before=before, after=after)
            record.check("failure-outcome", result.info["operation_outcome"], "numerical_failure")
            record.check("zero-duration", result.info["actual_duration_s"], 0.0)
            record.check(
                "retained-operation-cost",
                after["ledger"]["cost"],
                before["ledger"]["cost"] + 0.03,
                "balance",
            )
            before, after = json.loads(json.dumps(before)), json.loads(json.dumps(after))
            for key in ("ledger", "process"):
                for field in ("cost", "risk"):
                    before[key].pop(field)
                    after[key].pop(field)
            record.check("exact-physical-rollback", after, before)
            return
        if kind == "json":
            restored = ChemWorldRuntime.from_snapshot(snapshot)
        for action in CONTINUATION:
            execute(runtime, **action)
            if restored:
                execute(restored, **action)
                record.check("exact-json", restored._state.to_dict(), runtime._state.to_dict())
        if kind == "process":
            source, target = directory / "snapshot.json", directory / "continued.json"
            source.write_text(json.dumps(snapshot), encoding="utf-8")
            code = (
                "import json; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                f"p=Path({str(source)!r}); "
                "r=ChemWorldRuntime.from_snapshot(json.loads(p.read_text(encoding='utf-8'))); "
                f"results=[r.execute(a) for a in {CONTINUATION!r}]; "
                "assert all(z.info['transaction_status']=='committed' for z in results); "
                f"p=Path({str(target)!r}); "
                "p.write_text(json.dumps(r._state.to_dict()),encoding='utf-8'); r.close()"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            record.check(
                "exact-fresh-process",
                json.loads(target.read_text(encoding="utf-8")),
                json.loads(json.dumps(runtime._state.to_dict())),
            )
        if kind == "world":
            states = local_states(runtime._state)
            record.check("one-clock", runtime._state.vessel_elapsed_s, dict.fromkeys(states, 1.0))
            record.check("heat-unavailable", "queue_lab_heat" in composition.task.operations, False)
            for name, state in states.items():
                record.check(
                    f"{name}:phase-owner",
                    state.phases.phases["gas"].species_amounts_mol,
                    state.species_amounts,
                )
                record.check(f"{name}:evolved", state.species_amounts != amounts[name], True)
                record.check(f"{name}:temperature-owner", state.temperature_K, card["temperature"])
                record.check(
                    f"{name}:actual-P",
                    state.pressure_Pa,
                    sum(state.species_amounts.values())
                    * cases.R
                    * state.temperature_K
                    * 1000
                    / state.volume_L,
                    "P",
                )
                record.check(
                    f"{name}:local-heat",
                    state.thermal.vessels[name].energy_jacket_J,
                    state.ledger.energy_jacket_J,
                    "J",
                )
                for slot in state.state_slots.slots:
                    record.check(
                        f"{name}:no-extra-stock",
                        any(f.species_id or f.unit in {"K", "Pa"} for f in slot.spec.fields),
                        False,
                    )
            record.check(
                "group-heat",
                runtime._state.ledger.energy_jacket_J,
                sum(s.thermal.vessels[n].energy_jacket_J for n, s in states.items()),
                "J",
            )
        record.evidence["final"] = runtime._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("Choose unused filenames; all earlier attempts are retained")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        (f"T-{model}-{kind}", cases.case(kind, model), "trajectory")
        for model in ("stefan-maxwell", "mixture-averaged")
        for kind in ("binary", "ternary", "equimolar", "non-equimolar")
    ]
    entries += [(f"G-{n}", n, "grid") for n in (4, 8, 16, 32)]
    entries += [
        (f"L-{name}", cases.limit(name), "analytic")
        for name in ("association", "dissociation", "pure", "pressure-event")
    ]
    entries += [
        (f"P-{t}-{p}", cases.case(temperature=t, pressure=p), "trajectory")
        for t in (400, 700)
        for p in (50000, 100000)
    ]
    entries += [(f"W-{kind}", None, kind) for kind in ("world", "json", "process", "rollback")]
    records, grid_errors, started = [], [], time.perf_counter()
    heartbeat_stop = Event()
    stage = {"unit": "starting", "completed": 0}

    def heartbeat():
        count = 0
        while not heartbeat_stop.wait(30):
            count += 1
            elapsed = time.perf_counter() - started
            completed = stage["completed"]
            eta = "unknown" if not completed else f"{(24 - completed) * elapsed / completed:.1f}s"
            print(
                f"36B alive {count}: {stage['unit']}; {completed}/24 completed; "
                f"{completed / elapsed:.3f} units/s; ETA {eta}",
                flush=True,
            )

    Thread(target=heartbeat, daemon=True).start()
    for index, (name, card, kind) in enumerate(entries, 1):
        stage["unit"] = name
        record = Record(name)
        try:
            if kind == "grid":
                grid_errors.append(grid(record, card))
                if len(grid_errors) > 1:
                    record.check("grid-error-ratio", grid_errors[-2] / grid_errors[-1] >= 3, True)
            elif kind in {"trajectory", "analytic"}:
                physical(record, card, analytic=kind == "analytic")
            else:
                directory = args.output.parent / (args.output.stem + "-" + name)
                directory.mkdir(exist_ok=False)
                world(record, kind, directory)
        except Exception:
            record.evidence["exception"] = traceback.format_exc()
        failures = [c for c in record.checks if not c["passed"]]
        records.append(
            {
                "unit": name,
                "passed": not failures and "exception" not in record.evidence,
                "checks": record.checks,
                "failures": failures,
                "evidence": record.evidence,
            }
        )
        elapsed = time.perf_counter() - started
        stage["completed"] = index
        print(
            f"36B {index}/24 {name}: {'PASS' if records[-1]['passed'] else 'FAIL'}; "
            f"{index / elapsed:.2f} units/s; ETA {(24 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    heartbeat_stop.set()
    summary = {
        "scope": "36B fixed development block",
        "units": 24,
        "passed_units": sum(r["passed"] for r in records),
        "comparisons": sum(len(r["checks"]) for r in records),
        "passed_comparisons": sum(sum(c["passed"] for c in r["checks"]) for r in records),
        "elapsed_s": time.perf_counter() - started,
        "grid_errors": grid_errors,
        "results": [
            {"unit": r["unit"], "passed": r["passed"], "comparisons": len(r["checks"])}
            for r in records
        ],
        "failures": [
            {
                "unit": r["unit"],
                "checks": r["failures"],
                "exception": r["evidence"].get("exception"),
            }
            for r in records
            if not r["passed"]
        ],
    }
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, default=json_default),
        encoding="utf-8",
    )
    args.summary.write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"results", "failures"}}),
        flush=True,
    )
    return 0 if summary["passed_units"] == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
