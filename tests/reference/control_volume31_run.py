"""Run the fixed four-world/forty-check parent 31 closure without new physics."""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from chemworld.physchem.coupled_system import assemble_coupled_rates
from chemworld.physchem.gas_reactor_network import (
    gas_bath_key,
    gas_enthalpy_key,
    gas_inventory_key,
    gas_reactor_network_system,
    gas_transfer_key,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_reactor_network import unpack_gas_network


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def compare(name, actual, expected, *, rtol=2e-6, atol=2e-10):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    tolerance = atol + rtol * np.abs(b)
    return {
        "name": name,
        "passed": bool(np.all(np.isfinite(a)) and np.all(np.abs(a - b) <= tolerance)),
        "actual": a.tolist(),
        "expected": b.tolist(),
        "rtol": rtol,
        "atol": atol,
        "max_absolute_error": float(np.max(np.abs(a - b), initial=0)),
        "maximum_tolerance_ratio": float(np.max(np.abs(a - b) / tolerance, initial=0)),
    }


def check(name, value, **detail):
    return {"name": name, "passed": bool(value), **detail}


def grouped(name, rows):
    return {"name": name, "passed": all(r["passed"] for r in rows), "measurements": rows}


def read(runtime, network):
    before = canonical(runtime.snapshot().to_dict())
    state = runtime._state
    _, spec, packed = unpack_gas_network(state, network)
    system = gas_reactor_network_system(spec, network)
    rates = assemble_coupled_rates(
        packed,
        processes=system.processes,
        thermodynamics=system.thermodynamics,
        heat_flux=system.heat_flux,
        ports=system.ports,
        isothermal=system.isothermal,
    )
    states = {state.vessel_id: state, **state.inactive_vessels}
    thermal = {
        n: local.thermal.to_dict().get(n, {}).get("energy_jacket_J", 0.0)
        for n, local in states.items()
    }
    return {
        "time_s": state.ledger.time_s,
        "amounts": {n: dict(s.species_amounts) for n, s in states.items()},
        "rates_mol_s": {
            n: {k: rates.amount_rates_mol_s[gas_inventory_key(n, k)] for k in network.species_ids}
            for n in states
        },
        "transfers_mol": {
            e.connection_id: {
                s: packed.coordinates[gas_transfer_key(e.connection_id, s)]
                for s in network.species_ids
            }
            for e in spec.connections
        },
        "flow_rates_mol_s": {
            e.connection_id: {
                s: rates.coordinate_rates[gas_transfer_key(e.connection_id, s)]
                for s in network.species_ids
            }
            for e in spec.connections
        },
        "enthalpy_J": {
            e.connection_id: packed.coordinates[gas_enthalpy_key(e.connection_id)]
            for e in spec.connections
        },
        "bath_J": {n: packed.coordinates[gas_bath_key(n)] for n in states},
        "thermal_J": thermal,
        "pressure_Pa": {n: s.pressure_Pa for n, s in states.items()},
        "temperature_K": {n: s.temperature_K for n, s in states.items()},
        "vessel_elapsed_s": dict(state.vessel_elapsed_s),
        "local_times_s": {n: s.ledger.time_s for n, s in states.items()},
        "phase_owned": all(
            s.phases.to_dict()["gas"]["species_amounts_mol"] == s.species_amounts
            for s in states.values()
        ),
        "ledger": state.ledger.to_dict(),
        "snapshot_unchanged_by_read": canonical(runtime.snapshot().to_dict()) == before,
        "snapshot": before,
    }


def numbers(mapping):
    return [mapping[k][s] for k in sorted(mapping) for s in sorted(mapping[k])]


def comparisons(card, rows, transactions, replay, fresh):
    from reference.control_volume31_reference import evaluate

    reference = [evaluate(card, t) for t in card["times_s"]]
    details = [[] for _ in range(10)]
    for i, (native, expected) in enumerate(zip(rows, reference, strict=True)):
        t = expected["time_s"]
        details[0].append(
            compare(
                f"clock-{i}",
                [
                    native["time_s"],
                    *native["local_times_s"].values(),
                    *native["vessel_elapsed_s"].values(),
                ],
                t,
                rtol=0,
                atol=1e-8,
            )
        )
        details[1].append(
            compare(f"all-n-{i}", numbers(native["amounts"]), numbers(expected["amounts"]))
        )
        details[2].extend(
            (
                compare(
                    f"all-P-{i}",
                    list(native["pressure_Pa"].values()),
                    [expected["pressure_Pa"][n] for n in native["pressure_Pa"]],
                    atol=1e-3,
                ),
                compare(
                    f"all-T-{i}",
                    list(native["temperature_K"].values()),
                    card["temperature_K"],
                    rtol=0,
                    atol=1e-10,
                ),
                check(f"unique-phase-{i}", native["phase_owned"]),
            )
        )
        details[3].extend(
            (
                compare(
                    f"physical-dn-{i}",
                    numbers(native["rates_mol_s"]),
                    numbers(expected["rates_mol_s"]),
                    atol=1e-10,
                ),
                compare(
                    f"control-total-dn-{i}",
                    sum(native["rates_mol_s"]["control"].values()),
                    0,
                    rtol=0,
                    atol=1e-10,
                ),
            )
        )
        details[5].append(
            compare(
                f"cumulative-transfer-{i}",
                numbers(native["transfers_mol"]),
                numbers(expected["transfers_mol"]),
            )
        )
        details[5].append(
            compare(
                f"actual-outlet-mixture-{i}",
                [native["flow_rates_mol_s"][e][s] for e in ("out-1", "out-2") for s in ("A", "B")],
                [
                    expected["flow_rates_mol_s"][e][s]
                    for e in ("out-1", "out-2")
                    for s in ("A", "B")
                ],
                atol=1e-10,
            )
        )
        details[5].append(
            compare(
                f"split-1-to-3-{i}",
                [native["flow_rates_mol_s"]["out-2"][s] for s in ("A", "B")],
                [3 * native["flow_rates_mol_s"]["out-1"][s] for s in ("A", "B")],
                rtol=0,
                atol=1e-10,
            )
        )
        balance, reactions = [], []
        for node in card["amounts"]:
            for species in ("A", "B"):
                port = sum(
                    native["transfers_mol"][edge][species] * ((target == node) - (source == node))
                    for edge, source, target, _ in card["edges"]
                )
                reaction = (
                    expected["reaction_extent_mol"] * (1 if species == "B" else -1)
                    if node == "control"
                    else 0
                )
                balance.append(
                    native["amounts"][node][species] - card["amounts"][node][species] - port
                )
                reactions.append(reaction)
        details[6].extend(
            (
                compare(f"node-material-balance-{i}", balance, reactions, atol=2e-10),
                compare(
                    f"closed-H-element-{i}",
                    2 * sum(sum(v.values()) for v in native["amounts"].values()),
                    2 * sum(sum(v.values()) for v in card["amounts"].values()),
                    rtol=0,
                    atol=1e-9,
                ),
            )
        )
        for field in ("enthalpy_J", "bath_J"):
            details[7].append(
                compare(
                    f"{field}-{i}",
                    [native[field][k] for k in sorted(native[field])],
                    [expected[field][k] for k in sorted(expected[field])],
                    atol=1e-7,
                )
            )
        details[7].append(
            compare(
                f"original-local-thermal-{i}",
                [native["thermal_J"][k] for k in sorted(native["thermal_J"])],
                [expected["bath_J"][k] for k in sorted(expected["bath_J"])],
                atol=1e-7,
            )
        )
        details[7].append(
            compare(
                f"aggregate-bath-once-{i}",
                native["ledger"]["energy_jacket_J"],
                sum(expected["bath_J"].values()),
                atol=1e-7,
            )
        )
        h = {s: card["h0_J_mol"][s] + 32 * (card["temperature_K"] - 298.15) for s in ("A", "B")}
        residuals, tolerances = [], []
        for node in card["amounts"]:
            du = sum(
                (h[s] - card["R"] * card["temperature_K"])
                * (native["amounts"][node][s] - card["amounts"][node][s])
                for s in h
            )
            hin = sum(
                native["enthalpy_J"][edge] for edge, _, target, _ in card["edges"] if target == node
            )
            hout = sum(
                native["enthalpy_J"][edge] for edge, source, _, _ in card["edges"] if source == node
            )
            residuals.append(du - native["bath_J"][node] - hin + hout)
            tolerances.append(1e-7 + 2e-6 * (abs(native["bath_J"][node]) + abs(hin) + abs(hout)))
        details[7].append(
            check(
                f"node-first-laws-{i}",
                all(abs(r) <= tol for r, tol in zip(residuals, tolerances, strict=True)),
                residuals_J=residuals,
                tolerances_J=tolerances,
            )
        )
        details[7].append(
            check(
                f"global-first-law-{i}",
                abs(sum(residuals)) <= 1e-7 + 2e-6 * sum(abs(v) for v in native["bath_J"].values()),
                residual_J=sum(residuals),
                tolerance_J=1e-7 + 2e-6 * sum(abs(v) for v in native["bath_J"].values()),
            )
        )
        details[7].append(
            check(
                f"no-undeclared-power-{i}",
                all(
                    native["ledger"][k] == 0
                    for k in (
                        "flow_heat_J",
                        "flow_shaft_work_J",
                        "flow_pump_work_J",
                        "flow_electrical_work_J",
                        "boundary_work_J",
                    )
                ),
            )
        )
        details[8].append(check(f"rate-read-is-pure-{i}", native["snapshot_unchanged_by_read"]))
    final = rows[-1]
    details[0].append(
        check(
            "all-actions-committed",
            all(
                t["info"]["transaction_status"] == "committed"
                and t["info"]["operation_outcome"] == "success"
                for t in transactions
            ),
        )
    )
    details[4] = [
        compare(
            "steady-fraction",
            final["amounts"]["control"]["A"] / card["inventory_mol"],
            reference[-1]["steady_A_fraction"],
            rtol=0,
            atol=1e-6,
        ),
        compare(
            "steady-accumulation",
            list(final["rates_mol_s"]["control"].values()),
            0,
            rtol=0,
            atol=1e-8,
        ),
        compare(
            "unchanged-control-holdup",
            [sum(r["amounts"]["control"].values()) for r in rows],
            card["inventory_mol"],
            rtol=0,
            atol=2e-10,
        ),
    ]
    details[8].append(
        check(
            "complete-in-process-snapshot",
            replay["transaction_status"] == "committed" and replay["snapshot"] == final["snapshot"],
        )
    )
    details[9].append(
        check(
            "complete-fresh-process-snapshot",
            fresh.get("transaction_status") == "committed"
            and fresh.get("snapshot") == final["snapshot"],
        )
    )
    names = (
        "actual-commits-and-clocks",
        "all-finite-inventories",
        "pressure-temperature-and-phase-owner",
        "same-control-volume-dynamic-rates",
        "zero-accumulation-limit",
        "ports-and-nonselective-split",
        "node-and-global-material-balance",
        "source-H-bath-and-first-law",
        "full-JSON-resume-and-pure-query",
        "fresh-process-full-resume",
    )
    return [grouped(name, detail) for name, detail in zip(names, details, strict=True)], reference


def run_unit(case, output, record):
    from reference.control_volume31_cases import build

    card, network, spec, composition = build(case)
    record.update(card=card, specification=spec.to_dict(), composition=composition.to_dict())
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=31)
        rows = [read(runtime, network)]
        transactions = []
        for i, t in enumerate(card["times_s"][1:], 1):
            action = {"operation": "wait", "duration_s": t - runtime._state.ledger.time_s}
            print(f"{case['name']} checkpoint {i}/4 target={t:.6f}s", flush=True)
            result = runtime.execute(action)
            transactions.append({"action": action, "info": result.info})
            record["transactions"] = transactions
            if result.info["transaction_status"] != "committed":
                raise RuntimeError(
                    f"checkpoint {i} did not commit: {result.info.get('outcome_diagnostics')}"
                )
            rows.append(read(runtime, network))
            if i == 3:
                prefix = rows[-1]["snapshot"]
                restored = ChemWorldRuntime.from_snapshot(prefix)
        record["checkpoints"] = rows
        tail = transactions[-1]["action"]
        replay_result = restored.execute(tail)
        replay = {
            "transaction_status": replay_result.info["transaction_status"],
            "snapshot": canonical(restored.snapshot().to_dict()),
        }
        record["in_process_replay"] = replay
        prefix_path, child_path = (
            output / (case["name"] + "-prefix.json"),
            output / (case["name"] + "-child.json"),
        )
        prefix_path.write_text(
            json.dumps({"snapshot": prefix, "action": tail}, allow_nan=False), encoding="utf-8"
        )
        command = [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(Path(__file__).resolve()),
            "--resume",
            str(prefix_path),
            "--output",
            str(child_path),
        ]
        child = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60)
        record["child_process"] = {
            "command": command,
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
        }
        fresh = json.loads(child_path.read_text(encoding="utf-8")) if child_path.exists() else {}
        record["comparisons"], record["reference"] = comparisons(
            card, rows, transactions, replay, fresh
        )
    finally:
        runtime.close()
        if restored is not None:
            restored.close()


def run(output):
    from reference.control_volume31_cases import CASES

    output.mkdir(parents=True, exist_ok=False)
    rows, start = [], time.monotonic()
    for i, case in enumerate(CASES):
        record = {"unit": case["name"], "comparisons": []}
        unit_start = time.monotonic()
        print(f"31 closure starting {i + 1}/4 {case['name']}", flush=True)
        try:
            run_unit(case, output, record)
        except Exception as error:
            record.update(
                exception=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
            )
        while len(record["comparisons"]) < 10:
            record["comparisons"].append(
                check("unexecuted-slot-" + str(len(record["comparisons"])), False)
            )
        record.update(
            elapsed_s=time.monotonic() - unit_start,
            passed=all(c["passed"] for c in record["comparisons"]),
        )
        rows.append(record)
        (output / f"{i + 1:02}-{case['name']}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.monotonic() - start
        print(
            f"31 closure {i + 1}/4 pass={record['passed']} {record.get('exception', '')} "
            f"throughput={(i + 1) / elapsed:.2f}/s ETA={(3 - i) * elapsed / (i + 1):.1f}s",
            flush=True,
        )
    summary = {
        "evidence_class": "synthetic-development",
        "units": 4,
        "comparisons": 40,
        "units_passed": sum(r["passed"] for r in rows),
        "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
        "elapsed_s": time.monotonic() - start,
        "unit_results": [
            {
                "unit": r["unit"],
                "elapsed_s": r["elapsed_s"],
                "passed": r["passed"],
                "comparisons": r["comparisons"],
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
