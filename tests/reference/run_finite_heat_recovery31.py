"""Execute the fixed 36-unit independent finite heat-recovery block."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from math import isclose
from pathlib import Path

from finite_heat_recovery31_cases import (
    IDS,
    fixed_cases,
    initial_case,
    molar_volumes,
    reference,
    setup,
    spec_case,
)

from chemworld.physchem.finite_heat_recovery import (
    advance_finite_heat_recovery,
    heat_recovery_profile,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.finite_heat_recovery import (
    heat_recovery_record,
    heat_recovery_vessels,
    unpack_heat_recovery,
)
from chemworld.world.finite_heat_recovery_world import finite_heat_recovery_world_spec

ATOMS = {"water": {"H": 2, "O": 1}, "ethanol": {"C": 2, "H": 6, "O": 1}}


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, atol=None):
        passed = bool(
            actual == expected
            if atol is None
            else isclose(actual, expected, rel_tol=3e-7, abs_tol=atol)
        )
        self.checks.append(
            {
                "quantity": label,
                "actual": actual,
                "expected": expected,
                "atol": atol,
                "rtol": None if atol is None else 3e-7,
                "passed": passed,
            }
        )


def check_inventory(record, prefix, spec, before, amounts, enthalpies, expected):
    volumes = molar_volumes(spec)
    record.check(prefix + "/tank_count", len(amounts), 4)
    for i, n in enumerate(amounts):
        for k in IDS:
            record.check(f"{prefix}/tank{i}/{k}", n[k], expected["amounts"][i][k], 1e-10)
        record.check(f"{prefix}/tank{i}/H_J", enthalpies[i], expected["enthalpies_J"][i], 1e-7)
        record.check(
            f"{prefix}/tank{i}/T_K",
            spec.at_ph(i, n, enthalpies[i]),
            expected["temperatures_K"][i],
            1e-6,
        )
        record.check(
            f"{prefix}/tank{i}/volume_m3",
            sum(n[k] * volumes[k] for k in IDS),
            sum(expected["amounts"][i][k] * volumes[k] for k in IDS),
            1e-10,
        )
    for side in range(2):
        rows = range(2 * side, 2 * side + 2)
        for k in IDS:
            record.check(
                f"{prefix}/circuit{side}/conserved_{k}",
                sum(amounts[i][k] for i in rows),
                sum(before[i][k] for i in rows),
                1e-10,
            )
        for atom in ("C", "H", "O"):
            record.check(
                f"{prefix}/circuit{side}/element_{atom}",
                sum(amounts[i][k] * ATOMS[k].get(atom, 0) for i in rows for k in IDS),
                sum(before[i][k] * ATOMS[k].get(atom, 0) for i in rows for k in IDS),
                1e-10,
            )


def physical(record, case):
    identity, spec, composition, reverse, flows, duration, variable = case
    amounts, temperatures = setup(spec, composition, reverse)
    initial = initial_case(spec, composition, reverse)
    record.evidence.update(
        {
            "case": identity,
            "spec": spec.to_dict(),
            "initial_amounts_mol": amounts,
            "initial_temperatures_K": temperatures,
            "requested_flows_m3_s": flows,
            "requested_duration_s": duration,
        }
    )
    expected = reference(spec, amounts, temperatures, flows, duration, variable=variable)
    result, diagnostic = advance_finite_heat_recovery(spec, initial, flows, duration)
    record.evidence.update(
        {
            "case": identity,
            "spec": spec.to_dict(),
            "initial_amounts_mol": amounts,
            "initial_temperatures_K": temperatures,
            "requested_flows_m3_s": flows,
            "requested_duration_s": duration,
            "reference": expected,
            "actual_amounts_mol": result.final_state.amounts_mol,
            "actual_enthalpies_J": result.final_state.enthalpies_J,
            "diagnostic": diagnostic,
            "solver": result.solver_diagnostic,
        }
    )
    record.check("stopping_event", result.stopping_event, expected["stop"])
    record.check("actual_duration_s", result.actual_duration_s, expected["actual_duration_s"], 1e-6)
    record.check(
        "heat_left_to_right_W",
        diagnostic["heat_left_to_right_W"],
        expected["duty_left_to_right_W"],
        1e-7,
    )
    record.check("conductance_closure_W_K", diagnostic["conductance_residual_W_K"], 0, 1e-7)
    for i, name in enumerate(("left", "right")):
        record.check(
            name + "/actual_flow_m3_s",
            diagnostic["circuits"][name]["actual_flow_m3_s"],
            expected["source_flows_m3_s"][i],
            1e-10,
        )
        record.check(
            name + "/outlet_temperature_K",
            diagnostic["outlet_temperatures_K"][i],
            expected["outlet_temperatures_K"][i],
            1e-6,
        )
        receipt = diagnostic["circuit_receipts"][name]
        for species in IDS:
            withdrew = (
                initial.amounts_mol[2 * i][species] - result.final_state.amounts_mol[2 * i][species]
            )
            collected = (
                result.final_state.amounts_mol[2 * i + 1][species]
                - initial.amounts_mol[2 * i + 1][species]
            )
            record.check(
                name + "/withdrawal_receipt/" + species,
                receipt["transferred_mol"][species],
                withdrew,
                1e-10,
            )
            record.check(
                name + "/collection_receipt/" + species,
                receipt["transferred_mol"][species],
                collected,
                1e-10,
            )
        record.check(
            name + "/motor_loss_J",
            receipt["motor_loss_J"],
            expected["circuit_motor_loss_J"][i],
            1e-7,
        )
    check_inventory(
        record,
        "finite",
        spec,
        amounts,
        result.final_state.amounts_mol,
        result.final_state.enthalpies_J,
        expected,
    )
    for key in ("shaft_work_J", "motor_loss_J"):
        record.check(key, diagnostic[key], expected[key], 1e-7)
    record.check(
        "exchange_integral_J",
        diagnostic["exchanged_heat_left_to_right_J"],
        expected["duty_left_to_right_W"] * expected["actual_duration_s"],
        1e-7,
    )
    record.check("native_energy_residual_J", result.enthalpy_residual_J, 0, 1e-7)
    record.check(
        "total_liquid_and_body_energy_boundary_J",
        sum(result.final_state.enthalpies_J) - sum(initial.enthalpies_J),
        diagnostic["shaft_work_J"] - diagnostic["motor_loss_J"],
        1e-7,
    )
    for species, residual in result.component_residuals_mol.items():
        record.check("native_component_residual/" + species, residual, 0, 1e-10)


def execute(record, runtime, action, label):
    result = runtime.execute(action)
    record.check(label + "/transaction", result.info["transaction_status"], "committed")
    record.evidence.setdefault("actions", []).append(
        {"label": label, "action": action, "info": result.info}
    )


def world(record, reverse, shared):
    spec = spec_case()
    amounts, temperatures = setup(spec, 1, reverse)
    composition = finite_heat_recovery_world_spec(spec, amounts, temperatures, shared_clock=shared)
    whole, divided = (ChemWorldRuntime(composition=composition) for _ in range(2))
    operation = "wait" if shared else "run_flow"
    record.evidence.update(
        {
            "spec": spec.to_dict(),
            "initial_amounts_mol": amounts,
            "initial_temperatures_K": temperatures,
            "shared_clock": shared,
        }
    )
    for label, runtime in (("whole", whole), ("divided", divided)):
        runtime.reset(seed=31)
        for side, value in (("left", 8), ("right", 12)):
            execute(
                record,
                runtime,
                {"operation": "set_flow_rate", "flow_path": side, "flow_rate_mL_min": value},
                label + "/set-" + side,
            )
        record.check(
            label + "/independent_controls",
            heat_recovery_record(runtime._state).settings["flow_controls"],
            {"left": 8 / 6e7, "right": 12 / 6e7},
        )
    execute(record, whole, {"operation": operation, "duration_s": 15}, "whole/15s")
    for dt in (4, 6):
        execute(record, divided, {"operation": operation, "duration_s": dt}, f"divided/{dt}s")
    snapshot = json.loads(json.dumps(divided.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    future = {"operation": operation, "duration_s": 5}
    execute(record, divided, future, "divided/5s")
    execute(record, restored, future, "restored/5s")
    record.check(
        "exact_json_continuation",
        restored.snapshot().to_dict() == divided.snapshot().to_dict(),
        True,
    )
    with tempfile.TemporaryDirectory(prefix="chemworld31B-child-") as folder:
        incoming, outgoing = Path(folder) / "input.json", Path(folder) / "output.json"
        incoming.write_text(json.dumps({"snapshot": snapshot, "action": future}), encoding="utf-8")
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--resume-input",
                str(incoming),
                "--resume-output",
                str(outgoing),
            ],
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )
        record.evidence["child"] = {
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
        }
        record.check("child_exit", child.returncode, 0)
        if child.returncode == 0:
            response = json.loads(outgoing.read_text(encoding="utf-8"))
            record.check("child_committed", response["status"], "committed")
            record.check(
                "child_exact_future",
                response["snapshot"] == json.loads(json.dumps(divided.snapshot().to_dict())),
                True,
            )
            record.evidence["child_snapshot"] = response["snapshot"]
    expected = reference(spec, amounts, temperatures, (8 / 6e7, 12 / 6e7), 15)
    for label, runtime in (("whole", whole), ("divided", divided)):
        state = runtime._state
        _, _, inventory = unpack_heat_recovery(state)
        check_inventory(
            record, label, spec, amounts, inventory.amounts_mol, inventory.enthalpies_J, expected
        )
        for key, value in {
            "flow_pump_work_J": expected["shaft_work_J"],
            "heat_loss_J": expected["motor_loss_J"],
            "energy_jacket_J": 0.0,
            "flow_heat_J": 0.0,
            "flow_electrical_work_J": 0.0,
        }.items():
            record.check(label + "/group_ledger/" + key, getattr(state.ledger, key), value, 1e-7)
        record.check(label + "/group_time_s", state.ledger.time_s, 15, 1e-6)
        for i, name in enumerate(heat_recovery_vessels(state, spec)):
            local = state if name == state.vessel_id else state.inactive_vessels[name]
            record.check(
                label + "/phase/" + name,
                local.phases.phases["liquid"].species_amounts_mol,
                local.species_amounts,
            )
            slots = local.state_slots.slots
            record.check(label + "/H_slots/" + name, len(slots), 2)
            record.check(
                label + "/H_owner_kinds/" + name,
                sorted(s.spec.owner_kind for s in slots),
                ["material", "vessel"],
            )
            record.check(
                label + "/owned_H/" + name,
                sum(s.values["enthalpy_J"] for s in slots),
                expected["enthalpies_J"][i],
                1e-7,
            )
            record.check(
                label + "/owner_locations/" + name, all(s.location_id == name for s in slots), True
            )
            thermal = local.thermal.vessels[name]
            motor = expected["circuit_motor_loss_J"][i // 2] if i % 2 else 0.0
            record.check(label + "/motor_thermal_owner/" + name, thermal.heat_loss_J, motor, 1e-7)
            record.check(label + "/no_external_utility/" + name, thermal.energy_jacket_J, 0, 1e-7)
            record.check(label + "/vessel_time/" + name, state.vessel_elapsed_s[name], 15, 1e-6)
        before = runtime.snapshot().to_dict()
        profile1 = heat_recovery_profile(spec, inventory, (8 / 6e7, 12 / 6e7))
        profile2 = heat_recovery_profile(spec, inventory, (8 / 6e7, 12 / 6e7))
        record.check(label + "/pure_profile_repeat", profile1, profile2)
        record.check(
            label + "/profile_snapshot_unchanged", runtime.snapshot().to_dict() == before, True
        )
        record.evidence[label] = {"snapshot": before, "profile": profile1, "reference": expected}
    whole.close()
    divided.close()
    restored.close()


def run(output, summary):
    if output.exists() or summary.exists():
        raise ValueError("choose new evidence paths; retain every previous attempt")
    records, started = [], time.perf_counter()

    def unit(name, operation):
        record = Record(name)
        try:
            operation(record)
        except Exception as exc:
            record.evidence["exception"] = {"type": type(exc).__name__, "message": str(exc)}
            record.check("execution_completed", False, True)
        records.append(record)
        elapsed = time.perf_counter() - started
        print(
            f"stage=reference units={len(records)}/36 throughput={len(records) / elapsed:.2f}/s "
            f"eta_s={(36 - len(records)) * elapsed / len(records):.1f} unit={name}",
            flush=True,
        )

    for case in fixed_cases():
        unit(case[0], lambda r, c=case: physical(r, c))
    for reverse in (False, True):
        for shared in (False, True):
            unit(
                f"world-reverse{reverse}-shared{shared}",
                lambda r, a=reverse, b=shared: world(r, a, b),
            )
    checks = [c for r in records for c in r.checks]
    failures = [{"unit": r.name, **c} for r in records for c in r.checks if not c["passed"]]
    report = {
        "mode": "development",
        "design": "FINITE_HEAT_RECOVERY_EXPERIMENT.md",
        "unit_count": 36,
        "completed_units": len(records),
        "passed_units": sum(bool(r.checks) and all(c["passed"] for c in r.checks) for r in records),
        "comparison_count": len(checks),
        "passed_comparisons": sum(c["passed"] for c in checks),
        "failures": failures,
        "elapsed_s": time.perf_counter() - started,
        "units": [
            {"unit": r.name, "comparisons": r.checks, "evidence": r.evidence} for r in records
        ],
    }
    compact = {
        **{k: v for k, v in report.items() if k != "units"},
        "full_artifact_path": str(output.resolve()),
        "units": [
            {
                "unit": r.name,
                "comparison_count": len(r.checks),
                "passed_comparisons": sum(c["passed"] for c in r.checks),
                "failures": [c for c in r.checks if not c["passed"]],
                **({"exception": r.evidence["exception"]} if "exception" in r.evidence else {}),
            }
            for r in records
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in ((output, report), (summary, compact)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    print(
        json.dumps({k: v for k, v in report.items() if k not in {"units", "failures"}}), flush=True
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--resume-input", type=Path)
    parser.add_argument("--resume-output", type=Path)
    args = parser.parse_args()
    if args.resume_input is not None:
        data = json.loads(args.resume_input.read_text(encoding="utf-8"))
        runtime = ChemWorldRuntime.from_snapshot(data["snapshot"])
        result = runtime.execute(data["action"])
        with args.resume_output.open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "status": result.info["transaction_status"],
                    "snapshot": runtime.snapshot().to_dict(),
                },
                stream,
                allow_nan=False,
            )
        runtime.close()
    else:
        if args.output is None or args.summary is None:
            parser.error("--output and --summary are required for the fixed block")
        raise SystemExit(run(args.output, args.summary))
