"""Fixed 24-unit 35B development qualification, with all failures retained."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from math import exp, isclose
from pathlib import Path
from unittest.mock import patch

import gas_network35_energy_cases as cases
import gas_network35_energy_reference as oracle

from chemworld.physchem.gas_reactor_network import (
    advance_gas_reactor_network,
    pack_gas_reactor_network_state,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, kind="exact"):
        tolerances = {
            "n": (4e-6, 3e-10),
            "T": (4e-6, 3e-6),
            "P": (5e-6, 2e-3),
            "J": (6e-6, 5e-6),
            "balance": (0, 3e-10),
            "time": (1e-5, 2e-6),
        }
        for key in ("actual", "expected"):
            value = actual if key == "actual" else expected
            if hasattr(value, "item"):
                if key == "actual":
                    actual = value.item()
                else:
                    expected = value.item()
        passed = (
            actual == expected
            if kind == "exact"
            else isclose(actual, expected, rel_tol=tolerances[kind][0], abs_tol=tolerances[kind][1])
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


def compare(record, actual, expected, prefix):
    for name, row in expected["nodes"].items():
        for species, value in row["amounts_mol"].items():
            record.check(
                f"{prefix}:{name}:{species}",
                actual["nodes"][name]["amounts_mol"][species],
                value,
                "n",
            )
        for field, value in row.items():
            if field != "amounts_mol":
                kind = "T" if field == "temperature_K" else "P" if field == "pressure_Pa" else "J"
                record.check(f"{prefix}:{name}:{field}", actual["nodes"][name][field], value, kind)
        record.check(
            f"{prefix}:{name}:first-law", actual["nodes"][name]["energy_residual_J"], 0, "J"
        )
    for name, row in expected["edges"].items():
        for species, value in row["amounts_mol"].items():
            record.check(
                f"{prefix}:{name}:transferred:{species}",
                actual["connections"][name]["amounts_mol"][species],
                value,
                "n",
            )
        record.check(
            f"{prefix}:{name}:transported-H",
            actual["connections"][name]["enthalpy_J"],
            row["enthalpy_J"],
            "J",
        )
    for name, value in expected["links"].items():
        record.check(
            f"{prefix}:{name}:conducted-Q", actual["heat_links"][name]["heat_J"], value, "J"
        )
    record.check(f"{prefix}:group-first-law", actual["energy_residual_J"], 0, "J")
    for element, value in actual["element_residuals_mol"].items():
        record.check(f"{prefix}:{element}-balance", value, 0, "balance")


def physical(record, card, *, event=False, analytic=False):
    spec, network, device = cases.native(card)
    state = pack_gas_reactor_network_state(spec, network, card["amounts"])
    duration = 3 if event else 1
    reference = oracle.solve(card, duration, None if event else cases.TIMES, event)
    record.evidence.update(input=card, reference=reference, native=[])
    initial_total = sum(sum(n.values()) for n in card["amounts"].values())
    for sample in reference["samples"]:
        # Each sample is an independently solved interval from the fixed initial state.
        requested = duration if event else sample["time_s"]
        result, diagnostic = advance_gas_reactor_network(
            spec, network, state, requested, thermal_device=device
        )
        record.evidence["native"].append(
            {"diagnostic": diagnostic, "solver": result.solver_diagnostic}
        )
        compare(record, diagnostic, sample, str(sample["time_s"]))
        record.check(
            "total-moles", sum(result.final_state.amounts_mol.values()), initial_total, "balance"
        )
        record.check("actual-time", result.actual_duration_s, sample["time_s"], "time")
        if event:
            record.check("stopped", result.stopping_event is not None, True)
            expected_event = {
                "event-inventory": "minimum_amount",
                "event-high-T": "maximum_temperature",
                "event-low-T": "minimum_temperature",
                "event-pressure": "maximum_pressure",
            }
            record.check(
                "event-kind", result.stopping_event.split(":")[-1], expected_event[card["id"]]
            )
        if analytic:
            analytic_compare(record, card, diagnostic, sample["time_s"])


def analytic_compare(record, card, actual, t):
    node = card["nodes"][0]
    capacity = node["support_J_K"] + 0.02 * (32 - cases.R)
    expected_t, expected_a, heat, loss = node["temperature_K"], 0.02, 0.0, 0.0
    if card["id"] == "constant-power":
        heat = 5 * t
        expected_t += heat / capacity
    elif card["id"] == "newton-cooling":
        expected_t = 300 + 300 * exp(-0.2 * t / capacity)
        loss = capacity * (600 - expected_t)
    elif card["id"] == "closed-exotherm":
        expected_a *= exp(-0.7 * t)
        expected_t += (0.02 - expected_a) * 10000 / capacity
    else:
        c2 = 2 + 0.02 * (32 - cases.R)
        equilibrium = (capacity * 600 + c2 * 300) / (capacity + c2)
        delta = 300 * exp(-0.4 * (1 / capacity + 1 / c2) * t)
        expected_t = equilibrium + c2 * delta / (capacity + c2)
        other_t = equilibrium - capacity * delta / (capacity + c2)
        record.check("analytic:other:T", actual["nodes"]["other"]["temperature_K"], other_t, "T")
        record.check(
            "analytic:link:Q",
            actual["heat_links"]["wall"]["heat_J"],
            capacity * (600 - expected_t),
            "J",
        )
    row = actual["nodes"]["reactor"]
    record.check("analytic:reactor:T", row["temperature_K"], expected_t, "T")
    record.check("analytic:reactor:A", row["amounts_mol"]["A"], expected_a, "n")
    record.check("analytic:reactor:B", row["amounts_mol"]["B"], 0.02 - expected_a, "n")
    record.check("analytic:heater", row["bath_heat_J"], heat, "J")
    record.check("analytic:loss", row["heat_loss_J"], loss, "J")


def session():
    card = cases.case("loop", "heating")
    spec, network, heater = cases.native(card)
    runtime = ChemWorldRuntime(
        composition=gas_reactor_network_world(
            spec, network, card["amounts"], anchor_thermal_device=replace(heater, enabled=False)
        )
    )
    runtime.reset(seed=7)
    return runtime


def execute(runtime, operation, **fields):
    result = runtime.execute({"operation": operation, **fields})
    assert result.info["transaction_status"] == "committed", {
        k: result.info.get(k) for k in ("transaction_status", "rollback_reason", "error_message")
    }
    return result


def begin(runtime):
    execute(runtime, "select_vessel", vessel="reactor")
    execute(runtime, "queue_lab_heat", target_temperature_K=900, duration_s=1)
    execute(runtime, "select_vessel", vessel="source")
    execute(runtime, "wait", duration_s=0.25)


CONTINUATION = [
    {"operation": "wait", "duration_s": 0.75},
    {"operation": "select_vessel", "vessel": "collector"},
    {"operation": "wait", "duration_s": 0.2},
]


def world(record, kind, directory):
    runtime, restored = session(), None
    try:
        begin(runtime)
        before = runtime.snapshot().to_dict()
        record.evidence["snapshot"] = before
        if kind == "rollback":
            old = runtime._state.to_dict()
            with patch(
                "chemworld.runtime.gas_reactor_network.advance_gas_reactor_network",
                side_effect=RuntimeError("35B injected solver failure"),
            ):
                result = runtime.execute({"operation": "wait", "duration_s": 0.5})
            new = runtime._state.to_dict()
            record.check("failure-outcome", result.info["operation_outcome"], "numerical_failure")
            record.check("zero-physical-time", result.info["actual_duration_s"], 0)
            record.check(
                "retained-operation-cost",
                new["ledger"]["cost"],
                old["ledger"]["cost"] + 0.03,
                "balance",
            )
            record.evidence.update(before=old, after=new)
            # Original rejected operation penalties remain; all physical state is exact.
            old, new = json.loads(json.dumps(old)), json.loads(json.dumps(new))
            for key in ("ledger", "process"):
                for field in ("cost", "risk"):
                    old[key].pop(field)
                    new[key].pop(field)
            record.check("physical-queue-time-rollback", new, old)
            return
        if kind == "json":
            restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(before)))
        for action in CONTINUATION:
            execute(runtime, **action)
            if restored:
                execute(restored, **action)
                record.check(
                    "exact-json-continuation", restored._state.to_dict(), runtime._state.to_dict()
                )
        if kind == "process":
            source, target = directory / "snapshot.json", directory / "continued.json"
            source.write_text(json.dumps(before), encoding="utf-8")
            code = (
                "import json; from pathlib import Path; "
                "from chemworld.runtime import ChemWorldRuntime; "
                f"p=Path({str(source)!r}); "
                "r=ChemWorldRuntime.from_snapshot(json.loads(p.read_text(encoding='utf-8'))); "
                f"z=[r.execute(a) for a in {CONTINUATION!r}]; "
                "assert all(x.info['transaction_status']=='committed' for x in z); "
                f"p=Path({str(target)!r}); "
                "p.write_text(json.dumps(r._state.to_dict()),encoding='utf-8'); r.close()"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            record.check(
                "exact-fresh-process",
                json.loads(target.read_text(encoding="utf-8")),
                json.loads(json.dumps(runtime._state.to_dict())),
            )
        if kind == "queue":
            states = local_states(runtime._state)
            record.check(
                "one-completion", [j["sequence"] for j in runtime._state.lab_clock.completed], [0]
            )
            record.check("empty-running", runtime._state.lab_clock.running, ())
            record.check("empty-queued", runtime._state.lab_clock.queued, ())
            record.check("one-clock", runtime._state.vessel_elapsed_s, dict.fromkeys(states, 1.2))
            record.check("one-heater-integral", runtime._state.ledger.energy_jacket_J, 20, "J")
            record.check(
                "heater-disabled",
                states["reactor"]
                .equipment.equipment["reactor:heater"]
                .settings["thermal_device"]["enabled"],
                False,
            )
            for name, state in states.items():
                for slot in state.state_slots.slots:
                    record.check(
                        f"{name}:no-persistent-temperature",
                        any(f.unit == "K" for f in slot.spec.fields),
                        False,
                    )
                record.check(
                    f"{name}:original-phase-stock",
                    state.phases.phases["gas"].species_amounts_mol,
                    state.species_amounts,
                )
                record.check(
                    f"{name}:local-thermal",
                    state.thermal.vessels[name].energy_jacket_J,
                    20 if name == "reactor" else 0,
                    "J",
                )
            old_clock = runtime._state.lab_clock.to_dict()
            denied = runtime.execute(
                {"operation": "queue_lab_heat", "target_temperature_K": 800, "duration_s": 1}
            )
            record.check("member-heat-rejected", denied.info["operation_outcome"], "rejected")
            record.check("member-queue-unchanged", runtime._state.lab_clock.to_dict(), old_clock)
        record.evidence["final_state"] = runtime._state.to_dict()
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
        raise FileExistsError("Retain prior attempts; choose unused output and summary names")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for topology in ("serial", "loop"):
        for mode in ("inert", "adiabatic", "heating", "cooling"):
            entries.append((f"{topology}-{mode}", cases.case(topology, mode), "trajectory"))
    for name in ("constant-power", "newton-cooling", "two-body-conduction", "closed-exotherm"):
        entries.append((name, cases.analytic(name), "analytic"))
    for name in ("inventory", "high-T", "low-T", "pressure"):
        entries.append((f"event-{name}", cases.event(name)[0], "event"))
    for name in ("mass", "valve", "reverse-valve", "reversed-T"):
        entries.append((name, cases.flow_case(name), "trajectory"))
    entries.extend(
        (f"world-{kind}", None, kind) for kind in ("queue", "json", "process", "rollback")
    )
    records, start = [], time.perf_counter()
    for index, (name, card, kind) in enumerate(entries, 1):
        record = Record(name)
        try:
            if card is not None:
                physical(record, card, event=kind == "event", analytic=kind == "analytic")
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
        elapsed = time.perf_counter() - start
        print(
            f"35B {index}/24 {name}: {'PASS' if records[-1]['passed'] else 'FAIL'}; "
            f"{index / elapsed:.2f} units/s; ETA {(24 - index) * elapsed / index:.1f}s",
            flush=True,
        )
    summary = {
        "scope": "35B development, fixed 24 units",
        "units": 24,
        "passed_units": sum(r["passed"] for r in records),
        "comparisons": sum(len(r["checks"]) for r in records),
        "passed_comparisons": sum(sum(c["passed"] for c in r["checks"]) for r in records),
        "elapsed_s": time.perf_counter() - start,
        "failures": [
            {
                "unit": r["unit"],
                "comparisons": r["failures"],
                "exception": r["evidence"].get("exception"),
            }
            for r in records
            if not r["passed"]
        ],
        "results": [
            {"unit": r["unit"], "passed": r["passed"], "comparisons": len(r["checks"])}
            for r in records
        ],
    }
    args.output.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8"
    )
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in summary.items() if k not in {"results", "failures"}}),
        flush=True,
    )
    return 0 if summary["passed_units"] == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
