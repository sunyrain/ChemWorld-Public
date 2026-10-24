"""Fixed 32-unit independent split/utility/economics development comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from math import isclose
from pathlib import Path

from zero_order_treatment49_cases import (
    analytic,
    economic_policy,
    fixed_cases,
    inventory_case,
    molar_volumes,
    spec_case,
)

from chemworld.physchem.water_treatment_economics import evaluate_water_treatment_economics
from chemworld.physchem.zero_order_treatment import advance_zero_order_treatment
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.zero_order_treatment import (
    analyze_treatment_world_economics,
    treatment_record,
    treatment_vessels,
    unpack_treatment,
)
from chemworld.world.zero_order_treatment_world import zero_order_treatment_world_spec

ATOMS = {
    "water": {"H": 2, "O": 1},
    "ethanol": {"C": 2, "H": 6, "O": 1},
    "acetone": {"C": 3, "H": 6, "O": 1},
}


class Record:
    def __init__(self, name):
        self.name, self.checks, self.evidence = name, [], {}

    def check(self, label, actual, expected, *, atol=None):
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

    def receipt(self, prefix, actual, expected):
        self.check(prefix + "/fields", sorted(actual), sorted(expected))
        for key, expected_value in expected.items():
            if isinstance(expected_value, dict):
                self.check(
                    prefix + "/" + key + "/keys", sorted(actual[key]), sorted(expected_value)
                )
                for stage, value in expected_value.items():
                    self.check(f"{prefix}/{key}/{stage}", actual[key][stage], value, atol=1e-10)
            elif key == "schema":
                self.check(prefix + "/" + key, actual[key], expected_value)
            else:
                atol = 1e-6 if key == "elapsed_s" else 1e-10 if key.endswith("m3") else 1e-7
                self.check(prefix + "/" + key, actual[key], expected_value, atol=atol)


def inventory_checks(record, prefix, spec, initial, actual, enthalpies, expected):
    volumes = molar_volumes(spec)
    record.check(prefix + "/tank_count", len(actual), len(expected["amounts"]))
    for i, (amounts, reference) in enumerate(zip(actual, expected["amounts"], strict=True)):
        for key, value in reference.items():
            record.check(f"{prefix}/tank{i}/{key}", amounts[key], value, atol=1e-10)
        record.check(f"{prefix}/tank{i}/H_J", enthalpies[i], expected["enthalpies"][i], atol=1e-7)
        record.check(
            f"{prefix}/tank{i}/volume_m3",
            sum(amounts[k] * volumes[k] for k in volumes),
            sum(reference[k] * volumes[k] for k in volumes),
            atol=1e-10,
        )
    for key in volumes:
        record.check(
            f"{prefix}/conserved_component/{key}",
            sum(n[key] for n in actual),
            sum(n[key] for n in initial.amounts_mol),
            atol=1e-10,
        )
    for element in ("C", "H", "O"):
        record.check(
            f"{prefix}/conserved_element/{element}",
            sum(n[k] * ATOMS[k].get(element, 0) for n in actual for k in volumes),
            sum(n[k] * ATOMS[k].get(element, 0) for n in initial.amounts_mol for k in volumes),
            atol=1e-10,
        )


def physical_unit(record, case):
    name, spec, composition, flow, duration = case
    initial = inventory_case(spec, composition)
    expected = analytic(spec, initial, flow, duration)
    result, diagnostic = advance_zero_order_treatment(spec, initial, flow, duration)
    receipt = diagnostic["receipt"]
    record.evidence.update(
        {
            "case": name,
            "spec": spec.to_dict(),
            "initial_amounts_mol": initial.amounts_mol,
            "initial_enthalpies_J": initial.enthalpies_J,
            "requested_flow_m3_s": flow,
            "requested_duration_s": duration,
            "reference": expected,
            "diagnostic": diagnostic,
            "final_amounts_mol": result.final_state.amounts_mol,
            "final_enthalpies_J": result.final_state.enthalpies_J,
            "solver": result.solver_diagnostic,
        }
    )
    record.check("stop", result.stopping_event, expected["stop"])
    record.check("actual_duration_s", result.actual_duration_s, expected["dt"], atol=1e-6)
    record.check(
        "actual_pump_flow_m3_s",
        diagnostic["profile"]["actual_flow_m3_s"],
        expected["flow"],
        atol=1e-10,
    )
    record.check(
        "actual_pump_pressure_Pa",
        diagnostic["profile"]["pressure_rise_Pa"],
        expected["pressure"],
        atol=1e-7,
    )
    inventory_checks(
        record,
        "inventory",
        spec,
        initial,
        result.final_state.amounts_mol,
        result.final_state.enthalpies_J,
        expected,
    )
    record.receipt("receipt", receipt, expected["receipt"])
    for species, residual in result.component_residuals_mol.items():
        record.check("native_component_residual/" + species, residual, 0, atol=1e-10)
    record.check("native_enthalpy_residual_J", result.enthalpy_residual_J, 0, atol=1e-7)
    record.check(
        "actual_collected_volume_balance_m3",
        receipt["feed_volume_m3"],
        receipt["product_volume_m3"] + sum(receipt["reject_volumes_m3"].values()),
        atol=1e-10,
    )
    record.check(
        "actual_liquid_energy_boundary_J",
        diagnostic["liquid_enthalpy_change_J"],
        receipt["pump_work_J"]
        + receipt["additional_electrical_work_J"]
        - receipt["motor_loss_J"]
        + receipt["bath_heat_into_liquid_J"],
        atol=1e-7,
    )
    return receipt


def economic_unit(record, spec, receipt, discount):
    policy = economic_policy(spec, discount)
    before = json.loads(json.dumps(receipt))
    report = evaluate_water_treatment_economics(receipt, policy)
    electricity = (receipt["pump_work_J"] + receipt["additional_electrical_work_J"]) / 3.6e6
    electricity_cost = electricity * 0.12
    handling = sum(receipt["reject_volumes_m3"].values()) * 2.0
    variable = electricity_cost + handling
    crf = 1 / 20 if discount == 0 else discount / (1 - (1 + discount) ** -20)
    factor = 4000 * 3600 / receipt["elapsed_s"]
    product = factor * receipt["product_volume_m3"]
    total = 600 * crf + 600 * 0.03 + 40 + factor * variable
    expected = {
        "observed_electricity_kWh": electricity,
        "observed_electricity_cost": electricity_cost,
        "observed_concentrate_handling_estimate": handling,
        "observed_variable_cost_estimate": variable,
        "installed_capital": 600.0,
        "capital_recovery_factor_per_year": crf,
        "annual_capital_charge": 600 * crf,
        "annual_fixed_operating_cost": 58.0,
        "projected_annual_variable_cost": factor * variable,
        "projected_annual_product_m3": product,
        "projected_annual_total_cost": total,
        "levelized_product_cost_per_m3": total / product if product else None,
    }
    record.evidence.update(
        {"receipt": receipt, "policy": policy.to_dict(), "actual": report, "reference": expected}
    )
    for key, value in expected.items():
        record.check(key, report[key], value, atol=None if value is None else 1e-8)
    record.check("schema", report["schema"], "native-water-treatment-economics-v1")
    record.check("currency", report["currency"], "USD")
    record.check("price_base_year", report["price_base_year"], 2026)
    record.check("receipt_read_only", receipt, before)
    record.check("original_receipt_preserved", report["actual_observed_receipt"], receipt)
    record.check("repeat_read_only", evaluate_water_treatment_economics(receipt, policy), report)


def command(record, runtime, action, label):
    result = runtime.execute(action)
    record.check(label + "/transaction", result.info["transaction_status"], "committed")
    record.evidence.setdefault("actions", []).append(
        {"label": label, "action": action, "info": result.info}
    )


def world_unit(record, stages, shared):
    spec = spec_case(stages, 318.15)
    initial = inventory_case(spec)
    composition = zero_order_treatment_world_spec(
        spec, initial.amounts_mol[0], collectors=initial.amounts_mol[1:], shared_clock=shared
    )
    whole, segmented = (ChemWorldRuntime(composition=composition) for _ in range(2))
    operation = "wait" if shared else "run_flow"
    record.evidence.update(
        {"spec": spec.to_dict(), "shared_clock": shared, "advance_operation": operation}
    )
    for label, runtime in (("whole", whole), ("segmented", segmented)):
        runtime.reset(seed=49)
        record.check(
            label + "/preload_excluded",
            treatment_record(runtime._state).settings["treatment_totals"]["product_volume_m3"],
            0,
        )
        command(
            record, runtime, {"operation": "set_flow_rate", "flow_rate_mL_min": 10}, label + "/set"
        )
    command(record, whole, {"operation": operation, "duration_s": 15}, "whole/15s")
    for dt in (4, 6):
        command(record, segmented, {"operation": operation, "duration_s": dt}, f"segmented/{dt}s")
    snapshot = json.loads(json.dumps(segmented.snapshot().to_dict()))
    resumed = ChemWorldRuntime.from_snapshot(snapshot)
    future = {"operation": operation, "duration_s": 5}
    command(record, segmented, future, "segmented/final5s")
    command(record, resumed, future, "restored/final5s")
    record.check(
        "json_snapshot_exact_future", resumed.snapshot().to_dict(), segmented.snapshot().to_dict()
    )
    with tempfile.TemporaryDirectory(prefix="chemworld-49A-resume-") as directory:
        input_path, output_path = Path(directory) / "input.json", Path(directory) / "output.json"
        input_path.write_text(
            json.dumps({"snapshot": snapshot, "action": future}), encoding="utf-8"
        )
        child = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--resume-input",
                str(input_path),
                "--resume-output",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )
        record.evidence["child_process"] = {
            "returncode": child.returncode,
            "stdout": child.stdout,
            "stderr": child.stderr,
        }
        record.check("independent_process_exit", child.returncode, 0)
        if child.returncode == 0:
            child_output = json.loads(output_path.read_text(encoding="utf-8"))
            record.check("independent_process_committed", child_output["status"], "committed")
            record.check(
                "independent_process_exact_future",
                child_output["snapshot"],
                json.loads(json.dumps(segmented.snapshot().to_dict())),
            )
    expected = analytic(spec, initial, duration=15)
    for label, runtime in (("whole", whole), ("segmented", segmented)):
        state = runtime._state
        _, _, inventory = unpack_treatment(state)
        inventory_checks(
            record, label, spec, initial, inventory.amounts_mol, inventory.enthalpies_J, expected
        )
        receipt = treatment_record(state).settings["treatment_totals"]
        record.receipt(label + "/receipt", receipt, expected["receipt"])
        ledger = state.ledger
        for key, value in {
            "time_s": 15.0,
            "flow_pump_work_J": receipt["pump_work_J"],
            "flow_electrical_work_J": receipt["additional_electrical_work_J"],
            "heat_loss_J": receipt["motor_loss_J"] - receipt["bath_heat_into_liquid_J"],
            "energy_jacket_J": 0.0,
        }.items():
            record.check(
                label + "/original_ledger/" + key,
                getattr(ledger, key),
                value,
                atol=1e-6 if key == "time_s" else 1e-7,
            )
        for i, name in enumerate(treatment_vessels(state, spec)):
            tank = state if name == state.vessel_id else state.inactive_vessels[name]
            record.check(
                label + "/phase/" + name,
                tank.phases.phases["liquid"].species_amounts_mol,
                tank.species_amounts,
            )
            slots = tank.state_slots.slots
            record.check(label + "/H_owner_count/" + name, len(slots), 2)
            for slot in slots:
                record.check(label + "/H_location/" + slot.spec.slot_id, slot.location_id, name)
                target = expected["enthalpies"][i] if slot.spec.owner_kind == "material" else 0.0
                record.check(
                    label + "/H/" + slot.spec.slot_id, slot.values["enthalpy_J"], target, atol=1e-7
                )
            if shared:
                record.check(
                    label + "/shared_clock/" + name, state.vessel_elapsed_s[name], 15, atol=1e-6
                )
        before = runtime.snapshot().to_dict()
        report = analyze_treatment_world_economics(state, economic_policy(spec))
        repeated = analyze_treatment_world_economics(state, economic_policy(spec))
        record.check(label + "/economic_consumer_repeat", report, repeated)
        record.check(
            label + "/economic_consumer_no_snapshot_write", runtime.snapshot().to_dict(), before
        )
        record.check(
            label + "/economic_consumer_no_ledger_cost_write", state.ledger.cost, ledger.cost
        )
        record.evidence[label] = {"snapshot": before, "economics": report}
    whole.close()
    segmented.close()
    resumed.close()


def run_block(output, summary):
    if output.exists() or summary.exists():
        raise ValueError("choose new evidence paths; do not overwrite any earlier attempt")
    records, receipts, started = [], {}, time.perf_counter()

    def unit(name, operation):
        record = Record(name)
        try:
            operation(record)
        except Exception as error:
            record.evidence["exception"] = {"type": type(error).__name__, "message": str(error)}
            record.check("execution_completed", False, True)
        records.append(record)
        elapsed = time.perf_counter() - started
        print(
            f"stage=reference units={len(records)}/32 throughput={len(records) / elapsed:.2f}/s "
            f"eta_s={(32 - len(records)) * elapsed / len(records):.1f} unit={name}",
            flush=True,
        )

    cases = fixed_cases()
    for case in cases:

        def physical(record, selected=case):
            receipts[selected[0]] = physical_unit(record, selected)

        unit(case[0], physical)
    for index in (17, 23):
        for discount in (0.0, 0.08):
            case = cases[index]
            unit(
                f"economics-{case[0]}-discount{discount}",
                lambda r, c=case, d=discount: economic_unit(r, c[1], receipts[c[0]], d),
            )
    for stages in (1, 3):
        for shared in (False, True):
            unit(
                f"world-{stages}-shared{shared}", lambda r, n=stages, s=shared: world_unit(r, n, s)
            )
    checks = [c for r in records for c in r.checks]
    failures = [{"unit": r.name, **c} for r in records for c in r.checks if not c["passed"]]
    report = {
        "mode": "development",
        "design": "ZERO_ORDER_TREATMENT_EXPERIMENT.md",
        "unit_count": 32,
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
        raise SystemExit(run_block(args.output, args.summary))
