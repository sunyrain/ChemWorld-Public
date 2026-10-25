"""Complete 37-unit declared half-cell block; every attempt/failure is retained."""

import argparse
import json
import math
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from scipy.optimize import brentq

from chemworld.physchem.catalysis_network import catalytic_inventory_key
from chemworld.physchem.chemostatted_catalysis import (
    chemostatted_catalytic_system,
    chemostatted_readout,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.chemostatted_catalysis import unpack_chemostatted_catalysis
from chemworld.world.chemostatted_catalysis_world import chemostatted_catalysis_world
from reference.chemostatted_catalysis54_cases import fixed_cases, physical_case
from reference.chemostatted_catalysis54_reference import independent


def execute(runtime, action):
    result = runtime.execute(action)
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    if action["operation"] == "measure" and result.info["measurement_status"] != "valid":
        raise RuntimeError(
            {
                "measurement_status": result.info["measurement_status"],
                "diagnostics": result.info["outcome_diagnostics"],
            }
        )
    return result


def world_unit(case, directory):
    spec, boundary, slot, initial = physical_case(case)
    world = chemostatted_catalysis_world(
        spec,
        boundary,
        {"hydrogen": 1e-6},
        capacities_mol={"metal": 1e-6},
        initial_adsorbates_mol={k: slot.values[k] for k in spec.adsorbates},
        potential_V=-0.2,
        support_heat_capacity_J_K=0.1,
    )
    runtime = ChemWorldRuntime(composition=world)
    runtime.reset()
    execute(
        runtime, {"operation": "configure_electrical_control", "potential_V": case["potential"]}
    )
    execute(runtime, {"operation": "wait", "duration_s": 5})
    prefix = json.loads(json.dumps(runtime.snapshot().to_dict()))
    (directory / "prefix.json").write_text(json.dumps(prefix, indent=2), encoding="utf-8")
    clone = ChemWorldRuntime.from_snapshot(prefix)
    actions = (
        {"operation": "wait", "duration_s": 15},
        {"operation": "measure", "instrument": "electrical_meter"},
        {"operation": "measure", "instrument": "chemostatted_coverage"},
    )
    observations = []
    for action in actions:
        result = execute(runtime, action)
        execute(clone, action)
        if action["operation"] == "measure":
            observations.append(
                {
                    "time_s": runtime._state.ledger.time_s,
                    "observation": result.observation.to_dict(),
                }
            )
    (directory / "observations.json").write_text(
        json.dumps(observations, indent=2), encoding="utf-8"
    )
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    (directory / "snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    same = json.loads(json.dumps(clone.snapshot().to_dict())) == snapshot
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/chemostatted_catalysis54_resume.py",
            str(directory / "prefix.json"),
            str(directory / "continued.json"),
        ],
        capture_output=True,
        text=True,
        timeout=50,
    )
    (directory / "child.json").write_text(
        json.dumps(
            {"returncode": child.returncode, "stdout": child.stdout, "stderr": child.stderr},
            indent=2,
        ),
        encoding="utf-8",
    )
    fresh = (
        child.returncode == 0
        and json.loads((directory / "continued.json").read_text(encoding="utf-8")) == snapshot
    )
    _, spec, boundary, slot, final, _, _ = unpack_chemostatted_catalysis(runtime._state)
    return spec, boundary, slot, initial, final, runtime._state.ledger, same, fresh, observations


def native_equilibrium(case):
    spec, boundary, slot, initial = physical_case(case)
    return brentq(
        lambda potential: (
            chemostatted_readout(spec, slot, boundary, potential, initial).applied_current_A
        ),
        boundary.minimum_potential_V,
        boundary.maximum_potential_V,
        xtol=1e-14,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cases = fixed_cases()
    if args.canary:
        cases = [c for c in cases if c["mode"] == "world"]
    rows = []
    numeric_errors = {}
    started = time.monotonic()
    status = {"complete": 0, "current": "starting"}
    done = threading.Event()

    def heartbeat():
        while not done.wait(25):
            print(
                f"54C alive {status['complete']}/{len(cases)} current={status['current']} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    for case in cases:
        unit = directory / case["unit"]
        unit.mkdir()
        status["current"] = case["unit"]
        checks = []

        def numeric(label, actual, expected, atol, rtol=3e-7, checks=checks):
            absolute_error = abs(actual - expected)
            numeric_errors[label] = max(numeric_errors.get(label, 0.0), absolute_error)
            checks.append(
                {
                    "label": label,
                    "actual": actual,
                    "expected": expected,
                    "absolute_error": absolute_error,
                    "absolute_tolerance": atol,
                    "relative_tolerance": rtol,
                    "passed": math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                }
            )

        def boolean(label, condition, checks=checks):
            checks.append({"label": label, "passed": bool(condition)})

        error = None
        try:
            spec, boundary, slot, initial = physical_case(case)
            (unit / "input.json").write_text(
                json.dumps(
                    {"case": case, "mechanism": spec.to_dict(), "boundary": boundary.to_dict()},
                    indent=2,
                ),
                encoding="utf-8",
            )
            if case["mode"] == "equilibrium":
                read = chemostatted_readout(spec, slot, boundary, case["potential"], initial)
                numeric("equilibrium_current_A", read.applied_current_A, 0, 1e-14)
                key = "electron-proton-capture"
                numeric(
                    "detailed_balance",
                    read.forward_rates_mol_s[key],
                    read.reverse_rates_mol_s[key],
                    1e-20,
                )
                numeric("actual_affinity_J_mol", read.affinities_J_mol[key], 0, 1e-8)
                native_root = native_equilibrium(case)
                numeric("native_equilibrium_V", native_root, case["potential"], 1e-12)
                neighbor_pH = case["pH"] + 1 if case["pH"] == 1 else case["pH"] - 1
                slope = (native_equilibrium({**case, "pH": neighbor_pH}) - native_root) / (
                    neighbor_pH - case["pH"]
                )
                numeric(
                    "nernst_slope_V_per_pH",
                    slope,
                    -8.31446261815324 * case["temperature"] * math.log(10) / 96485.33212,
                    1e-13,
                )
            else:
                if case["mode"] == "world":
                    spec, boundary, slot, initial, final, ledger, same, fresh, observations = (
                        world_unit(case, unit)
                    )
                    duration = ledger.time_s
                    bath = ledger.energy_jacket_J
                    boolean("full_JSON_same_process", same)
                    boolean("full_JSON_fresh_process", fresh)
                    numeric(
                        "ledger_work_once",
                        ledger.flow_electrical_work_J,
                        final.coordinates["electrical_work_J"],
                        1e-12,
                    )
                    numeric("original_in_situ_elapsed_s", duration, 22.0, 1e-12, rtol=0)
                    boolean("two_measurements", len(observations) == 2)
                    for observation, target_time in zip(observations, (21.0, 22.0), strict=True):
                        numeric(
                            "observation_time_s", observation["time_s"], target_time, 1e-12, rtol=0
                        )
                        channels = {
                            c["channel_id"]: c
                            for c in observation["observation"]["native"]["channels"]
                        }
                        expected_at_measure = independent(
                            case["kind"],
                            case["temperature"],
                            case["pH"],
                            case["potential"],
                            target_time,
                        )
                        if target_time == 21:
                            from reference.chemostatted_catalysis54_reference import coefficients

                            kf, kr, _, _ = coefficients(
                                case["kind"], case["temperature"], case["potential"]
                            )
                            theta = expected_at_measure["theta"]
                            current = (
                                -96485.33212
                                * 1e-6
                                * (kf * 10 ** (-case["pH"]) * (1 - theta) - kr * theta)
                            )
                            numeric(
                                "observed_current_A",
                                channels["electrical_current"]["value"],
                                current,
                                1e-10,
                            )
                            numeric(
                                "observed_potential_V",
                                channels["terminal_potential"]["value"],
                                case["potential"],
                                1e-12,
                            )
                        else:
                            key = "hydroxyl_bound" if case["kind"] == "hydroxylation" else "H_ads"
                            numeric(
                                "observed_coverage",
                                channels["coverage_" + key]["value"],
                                expected_at_measure["theta"],
                                3e-10,
                            )
                    reference_duration = 22.0
                else:
                    system = chemostatted_catalytic_system(
                        spec, slot, boundary, case["potential"], support_heat_capacity_J_K=0.1
                    )
                    report = system.advance(initial, case["duration"])
                    final, duration, bath = (
                        report.final_state,
                        report.actual_duration_s,
                        report.final_state.boundary_heat_J,
                    )
                    boolean("finite_stopping_event", report.stopping_event == case["event"])
                    numeric("original_U_residual", report.energy_balance_residual_J, 0, 3e-6)
                    for key, value in report.conservation_residuals.items():
                        numeric(
                            "conservation:" + key, value, 0, 1e-8 if key.endswith("_C") else 1e-12
                        )
                    reference_duration = (
                        0
                        if case["event"]
                        in {"electrode_current_limit", "bath_heating_limit", "bath_cooling_limit"}
                        else case["duration"]
                    )
                expected = independent(
                    case["kind"],
                    case["temperature"],
                    case["pH"],
                    case["potential"],
                    reference_duration,
                    pressure_limit=boundary.maximum_pressure_Pa
                    if case["event"] == "gas_pressure_limit"
                    else None,
                )
                read = chemostatted_readout(spec, slot, boundary, case["potential"], final)
                key = "hydroxyl_bound" if case["kind"] == "hydroxylation" else "H_ads"
                numeric("coverage", read.coverages[key], expected["theta"], 3e-10)
                numeric("gas_mol", final.amounts_mol["hydrogen"], expected["gas_mol"], 3e-15)
                numeric("actual_time_s", duration, expected["elapsed_s"], 3e-6)
                for name in ("faradaic_charge_C", "electrical_work_J"):
                    numeric(name, final.coordinates[name], expected[name], 1e-10)
                for name in boundary.solutes:
                    numeric(
                        "real_chemostat:" + name,
                        final.amounts_mol[name],
                        initial.amounts_mol[name],
                        1e-20,
                    )
                pin = final.exchanged_mol.get("reservoir:proton:in", 0) - final.exchanged_mol.get(
                    "reservoir:proton:out", 0
                )
                hin = final.exchange_energy_J.get(
                    "reservoir:proton:in", 0
                ) - final.exchange_energy_J.get("reservoir:proton:out", 0)
                numeric("real_proton_boundary_mol", pin, expected["proton_net_in_mol"], 3e-15)
                numeric("real_proton_boundary_H_J", hin, expected["proton_net_enthalpy_J"], 1e-10)
                numeric("bath_heat_J", bath, expected["bath_heat_J"], 1e-9)
                numeric(
                    "faraday_proton_relation",
                    final.coordinates["faradaic_charge_C"],
                    -96485.33212 * pin,
                    1e-9,
                )
                final_ads = {
                    k: final.amounts_mol[catalytic_inventory_key(slot, k)] for k in spec.adsorbates
                }
                boolean("actual_finite_sites", sum(final_ads.values()) <= 1e-6 * (1 + 1e-9))
                (unit / "physical.json").write_text(
                    json.dumps(
                        {
                            "n_mol": dict(final.amounts_mol),
                            "electrical": dict(final.coordinates),
                            "boundary_mol": dict(final.exchanged_mol),
                            "boundary_H_J": dict(final.exchange_energy_J),
                            "bath_J": bath,
                            "expected": expected,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
        except Exception:
            error = traceback.format_exc()
        row = {
            "unit": case["unit"],
            "passed": error is None and all(c["passed"] for c in checks),
            "checks": len(checks),
            "failed_checks": [c for c in checks if not c["passed"]],
            "error": error,
        }
        (unit / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        (unit / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        status["complete"] += 1
        elapsed = time.monotonic() - started
        print(
            f"54C {len(rows)}/{len(cases)} {case['unit']} {'PASS' if row['passed'] else 'FAIL'} "
            f"rate={len(rows) / elapsed:.2f}/s "
            f"ETA={(len(cases) - len(rows)) * elapsed / len(rows):.1f}s",
            flush=True,
        )
    done.set()
    result = {
        "mode": "development-canary" if args.canary else "development-fixed-block",
        "total": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "checks": sum(r["checks"] for r in rows),
        "checks_passed": sum(r["checks"] - len(r["failed_checks"]) for r in rows),
        "maximum_absolute_errors": numeric_errors,
        "elapsed_s": time.monotonic() - started,
        "units": rows,
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "units"}), flush=True)
    return 0 if result["total"] == result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
