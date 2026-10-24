"""Fixed 20-unit 49F development block with actual upstream and full continuation."""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from chemworld.physchem.suspension_treatment import (
    advance_suspension_treatment,
    suspension_treatment_inventory,
    suspension_volume,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.suspension_treatment import unpack_suspension_treatment
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.suspension_treatment_world import suspension_treatment_world_spec
from reference.suspension_treatment49_cases import fixed_cases, independent_flows


def execute(runtime, operation, **values):
    result = runtime.execute({"operation": operation, **values})
    if result.info["transaction_status"] != "committed":
        raise RuntimeError(result.info)
    return result


def world_unit(spec, feed, flow, duration, shared, directory):
    payload = suspension_treatment_world_spec(spec, feed, shared_clock=shared).to_dict()
    payload["task"]["operations"].append("select_vessel")
    runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
    runtime.reset()
    execute(runtime, "set_flow_rate", flow_rate_mL_min=flow * 6e7)
    execute(runtime, "select_vessel", vessel=spec.sludge_vessel_id)
    if not shared:
        execute(runtime, "select_vessel", vessel="reactor")
    operation = "wait" if shared else "run_flow"
    execute(runtime, operation, duration_s=5)
    prefix = json.loads(json.dumps(runtime.snapshot().to_dict()))
    (directory / "prefix.json").write_text(json.dumps(prefix, indent=2), encoding="utf-8")
    execute(runtime, operation, duration_s=duration - 5)
    final_snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    (directory / "snapshot.json").write_text(json.dumps(final_snapshot, indent=2), encoding="utf-8")
    restored = ChemWorldRuntime.from_snapshot(prefix)
    execute(restored, operation, duration_s=duration - 5)
    in_process = json.loads(json.dumps(restored.snapshot().to_dict())) == final_snapshot
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/suspension_treatment49_resume.py",
            str(directory / "prefix.json"),
            str(directory / "continued.json"),
            operation,
            str(duration - 5),
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
    fresh_process = (
        child.returncode == 0
        and json.loads((directory / "continued.json").read_text(encoding="utf-8")) == final_snapshot
    )
    current = runtime._state
    states = {current.vessel_id: current.replace(inactive_vessels={}), **current.inactive_vessels}
    anchor = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    _, _, final = unpack_suspension_treatment(anchor)
    ledger = current.ledger
    diagnostic = {
        "pump_work_J": ledger.flow_pump_work_J,
        "mixing_work_J": ledger.flow_shaft_work_J,
        "hydraulic_work_J": ledger.flow_pump_work_J * spec.pump.efficiency,
        "motor_loss_J": ledger.flow_pump_work_J * (1 - spec.pump.efficiency),
        "bath_heating_J": ledger.energy_jacket_J,
        "bath_cooling_J": ledger.heat_loss_J - ledger.flow_pump_work_J * (1 - spec.pump.efficiency),
    }
    diagnostic["enthalpy_residual_J"] = (
        sum(final.enthalpies_J)
        - sum(suspension_treatment_inventory(spec, feed).enthalpies_J)
        - diagnostic["hydraulic_work_J"]
        - diagnostic["mixing_work_J"]
        - diagnostic["bath_heating_J"]
        + diagnostic["bath_cooling_J"]
    )
    return final, diagnostic, ledger.time_s, in_process, fresh_process, states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--upstream-path", default=os.path.join(os.environ["TEMP"], "ChemWorld-49E-reference")
    )
    parser.add_argument(
        "--canary", action="store_true", help="installed local/shared canaries only"
    )
    args = parser.parse_args()
    sys.path.insert(0, args.upstream_path)
    from reference.suspension_treatment49_reference import upstream

    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    cases = fixed_cases()
    if args.canary:
        cases = [c for c in cases if c[5] is not None]
    rows, status, started, stop = (
        [],
        {"completed": 0, "current": "starting"},
        time.monotonic(),
        threading.Event(),
    )

    def progress():
        while not stop.wait(25):
            print(
                f"49F alive {status['completed']}/{len(cases)} current={status['current']} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    threading.Thread(target=progress, daemon=True).start()
    for name, spec, feed, flow, duration, shared, reference, event in cases:
        status["current"] = name
        unit = directory / name
        unit.mkdir()
        checks = []

        def numeric(label, actual, expected, atol=3e-9, rtol=3e-7, checks=checks):
            checks.append(
                {
                    "label": label,
                    "actual": actual,
                    "expected": expected,
                    "passed": math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol),
                }
            )

        def boolean(label, condition, checks=checks):
            checks.append({"label": label, "passed": bool(condition)})

        error, upstream_constraints = None, 0
        try:
            initial = suspension_treatment_inventory(spec, feed)
            (unit / "input.json").write_text(
                json.dumps(
                    {
                        "spec": spec.to_dict(),
                        "feed": {
                            "liquid_mol": dict(feed.liquid_mol),
                            "dispersed_mol": feed.dispersed_mol,
                            "floc_mol": feed.floc_mol,
                        },
                        "flow_m3_s": flow,
                        "duration_s": duration,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            q, volume, inlet, product, sludge = independent_flows(spec, feed, flow)
            expected_dt = duration
            if q > 0:
                expected_dt = min(
                    duration,
                    (volume - spec.pickup_volume_L / 1000) / q,
                    spec.capacities_L[1] / 1000 / suspension_volume(spec, product),
                    spec.capacities_L[2] / 1000 / suspension_volume(spec, sludge),
                )
            if event in {"no_forward_flow", "bath_cooling_limit"}:
                expected_dt = 0
            if shared is None:
                result, diagnostic = advance_suspension_treatment(spec, initial, flow, duration)
                final, dt = result.final_state, result.actual_duration_s
                boolean("stopping_event", result.stopping_event == event)
                numeric("original_U_residual", result.energy_balance_residual_J, 0, 3e-6)
                for element, residual in result.conservation_residuals.items():
                    numeric("element:" + element, residual, 0, 3e-9)
                numeric(
                    "actual_pump_flow",
                    diagnostic["initial_operating_point"]["actual_flow_m3_s"],
                    q,
                    1e-16,
                )
                native = diagnostic["initial_operating_point"]
            else:
                final, diagnostic, dt, same, fresh, states = world_unit(
                    spec, feed, flow, duration, shared, unit
                )
                boolean("exact_JSON_continuation", same)
                boolean("fresh_process_JSON_continuation", fresh)
                for tank_name, tank in states.items():
                    boolean(
                        "phase_identity:" + tank_name,
                        set(tank.phases.phases) == {"liquid", "dispersed", "floc"},
                    )
                    boolean(
                        "total_H_owner:" + tank_name,
                        len(tank.state_slots.slots) == 1
                        and tank.state_slots.slots[0].spec.phase_id is None,
                    )
                native = None
            numeric("actual_duration", dt, expected_dt, 3e-6)
            for i, rates in enumerate(({k: -v for k, v in inlet.items()}, product, sludge)):
                for k, rate in rates.items():
                    numeric(
                        f"original_n:{i}:{k}",
                        final.amounts_mol[i][k],
                        initial.amounts_mol[i][k] + rate * expected_dt,
                        3e-10,
                    )
            numeric("material_H_residual", diagnostic["enthalpy_residual_J"], 0, 3e-6)
            numeric(
                "pump_work_partition",
                diagnostic["pump_work_J"],
                diagnostic["hydraulic_work_J"] + diagnostic["motor_loss_J"],
                3e-9,
            )
            boolean(
                "mixing_work_positive_iff_flowing",
                (diagnostic["mixing_work_J"] > 0) == (q > 0 and dt > 0),
            )
            if reference:
                ref = upstream(spec, feed, q)
                (unit / "upstream.json").write_text(json.dumps(ref, indent=2), encoding="utf-8")
                upstream_constraints = len(ref["constraints"])
                numeric(
                    "upstream:tss_loss",
                    native["converted_particle_mol_s"]
                    * spec.material.particle.molecular_weight_kg_mol,
                    ref["tss_loss_kg_s"],
                    1e-15,
                )
                for key in (
                    "rapid_basin_volume_each_m3",
                    "rapid_basin_volume_total_m3",
                    "floc_basin_volume_m3",
                    "paddle_speed_m_s",
                    "rapid_mixing_power_W",
                    "floc_mixing_power_W",
                    "mixing_power_W",
                ):
                    numeric("upstream:" + key, native[key], ref[key], 1e-12)
                mapping = {
                    "H2O": "liquid:water",
                    "TDS": "liquid:glucose",
                    "TSS": "dispersed",
                    "Sludge": "floc",
                }
                for target in ("product", "sludge"):
                    for aggregate, key in mapping.items():
                        numeric(
                            f"upstream:{target}:{aggregate}",
                            native[target + "_mol_s"][key] * spec.molecular_weights[key],
                            ref[target + "_kg_s"][aggregate],
                            1e-15,
                        )
            (unit / "diagnostic.json").write_text(
                json.dumps(diagnostic, indent=2), encoding="utf-8"
            )
        except Exception:
            error = traceback.format_exc()
        row = {
            "unit": name,
            "passed": error is None and all(c["passed"] for c in checks),
            "checks": len(checks),
            "failed_checks": [c for c in checks if not c["passed"]],
            "upstream_constraints": upstream_constraints,
            "error": error,
        }
        (unit / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
        (unit / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        status["completed"] += 1
        elapsed = time.monotonic() - started
        print(
            f"49F {len(rows)}/{len(cases)} {name}: {'PASS' if row['passed'] else 'FAIL'} "
            f"rate={len(rows) / elapsed:.2f}/s "
            f"ETA={(len(cases) - len(rows)) * elapsed / len(rows):.1f}s",
            flush=True,
        )
    stop.set()
    summary = {
        "mode": "development-canary" if args.canary else "development-fixed-block",
        "total": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "checks": sum(r["checks"] for r in rows),
        "upstream_constraints": sum(r["upstream_constraints"] for r in rows),
        "elapsed_s": time.monotonic() - started,
        "units": rows,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "units"}), flush=True)
    return 0 if summary["passed"] == summary["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
