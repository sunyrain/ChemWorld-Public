"""Fixed four-unit/24-comparison passive RED real-consumer canary."""

import argparse
import importlib.util
import json
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

from chemworld.examples.reverse_electrodialysis import example_report
from chemworld.foundation.lab_clock import LabClockSpec
from chemworld.physchem.electrodialysis import electrodialysis_point
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.reverse_electrodialysis import (
    ReverseElectrodialysisSpec,
    advance_reverse_electrodialysis,
    reverse_electrodialysis_inventory,
    reverse_electrodialysis_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.electrodialysis_world import electrodialysis_world_spec


def exact(name, value):
    return {"name": name, "passed": bool(value)}


def numeric(name, actual, expected, *, atol=1e-9, rtol=0.003):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    return {
        "name": name,
        "passed": bool(np.all(np.isfinite(a)) and np.allclose(a, b, atol=atol, rtol=rtol)),
        "actual": a.tolist(),
        "expected": b.tolist(),
        "atol": atol,
        "rtol": rtol,
        "maximum_absolute_error": float(np.max(np.abs(a - b), initial=0)),
    }


def fixtures(ions):
    from reference.electrodialysis48_fixtures import amounts, spec

    pair = spec(ions)
    return ReverseElectrodialysisSpec(pair, 1e6, 1.0, 1.0), (
        amounts(pair, dilution=0.5),
        amounts(pair, dilution=1),
    )


def run_unit(index, record, baseline):
    from reference.reverse_electrodialysis48_reference import finite, point

    spec, ns = fixtures(3 if index == 0 else 4)
    record["model"], record["initial_amounts"] = spec.to_dict(), ns
    if index == 0:
        native = reverse_electrodialysis_point(spec, ns, 100)
        record["native"] = native
        independent = point(spec.cell_pair, ns, 100)
        record["reference"] = independent
        record["comparisons"] = [
            exact("converged", native["status"] == "converged"),
            numeric(
                "terminal-IV",
                [native["current_A"], native["voltage_V"]],
                [independent["current_A"], independent["voltage_V"]],
            ),
            numeric(
                "all-species-flow",
                list(native["diluate_rates_mol_s"].values()),
                list(independent["diluate_rates_mol_s"].values()),
            ),
            numeric("external-load-power", native["load_power_W"], independent["load_power_W"]),
            exact(
                "circuit-and-free-energy",
                native["load_power_W"] > 0
                and abs(native["circuit_residual_V"]) < 1e-9
                and abs(native["free_energy_power_residual_W"]) < 1e-9,
            ),
            exact(
                "each-tank-neutral",
                all(
                    abs(
                        sum(
                            spec.properties.charges[k] * native[field][k]
                            for k in spec.properties.ion_ids
                        )
                    )
                    < 1e-10
                    for field in ("diluate_rates_mol_s", "concentrate_rates_mol_s")
                ),
            ),
        ]
    elif index == 1:
        initial = reverse_electrodialysis_inventory(spec, *ns)
        solved, receipt = advance_reverse_electrodialysis(
            spec, initial, ScalarControlProgram(((0, 100),), "Ohm"), 5
        )
        record["native"] = {
            "receipt": receipt,
            "amounts": solved.final_state.amounts_mol,
            "diagnostics": solved.solver_diagnostic,
        }
        independent = finite(spec.cell_pair, ns, 100, 5, progress=lambda c: print(c, flush=True))
        record["reference"] = independent
        record["comparisons"] = [
            exact("full-duration", solved.actual_duration_s == 5 and solved.stopping_event is None),
            numeric(
                "all-finite-amounts",
                [list(n.values()) for n in solved.final_state.amounts_mol],
                [list(n.values()) for n in independent["amounts_mol"]],
            ),
            numeric(
                "generated-energy",
                receipt["load_energy_delivered_J"],
                independent["load_energy_delivered_J"],
            ),
            exact("conservation", max(map(abs, solved.component_residuals_mol.values())) < 1e-10),
            exact(
                "first-law",
                abs(solved.enthalpy_residual_J) < 1e-8
                and receipt["electrical_work_into_liquid_J"] < 0
                and receipt["bath_heat_into_liquid_J"] > 0,
            ),
            numeric(
                "free-energy-dissipation",
                receipt["free_energy_decrease_minus_load_J"],
                independent["dissipation_integral_J"],
            ),
        ]
    elif index == 2:
        consumer = example_report(spec, ns)
        record["consumer"] = consumer
        if not consumer["complete"]:
            failed = consumer["transactions"][-1]
            raise RuntimeError(
                "world operation "
                + failed["action"]["operation"]
                + " did not commit: "
                + failed["info"].get("error_message", "unknown")
            )
        initial, final = consumer["initial_state"], consumer["final_state"]
        prefix = consumer["prefix_snapshot"]
        measured = {
            c["channel_id"]: c
            for c in consumer["transactions"][3]["info"]["native_observation"]["channels"]
        }
        record["comparisons"] = [
            exact(
                "actions-committed",
                all(
                    r["info"]["transaction_status"] == "committed" for r in consumer["transactions"]
                )
                and all(
                    r["transaction_status"] == "committed" for r in consumer["tail_transactions"]
                ),
            ),
            exact(
                "meter-shared-time",
                final["ledger"]["time_s"] == 4
                and measured["terminal_potential"]["value"]
                == consumer["prefix_terminal"]["voltage_V"]
                and measured["electrical_current"]["value"]
                == consumer["prefix_terminal"]["current_A"]
                and measured["electrical_current"]["sampled_at"]["time_s"] == 2,
            ),
            exact(
                "generated-work-and-no-refund",
                final["ledger"]["flow_electrical_work_J"] < 0
                and final["ledger"]["cost"] >= initial["ledger"]["cost"],
            ),
            numeric(
                "bath-electric-one-settlement",
                final["ledger"]["flow_electrical_work_J"],
                final["ledger"]["heat_loss_J"],
                atol=1e-8,
                rtol=0,
            ),
            exact(
                "json-exact-resume",
                consumer["exact_json_resume"]
                and json.loads(json.dumps(prefix, allow_nan=False)) == prefix,
            ),
            exact(
                "finite-consumer",
                final["species_amounts"] != initial["species_amounts"]
                and final["ledger"]["energy_jacket_J"] == 0
                and consumer["prefix_terminal"]["voltage_V"]
                * consumer["prefix_terminal"]["current_A"]
                < 0,
            ),
        ]
    else:
        module_spec = importlib.util.spec_from_file_location(
            "_red_pre_edit_ed", baseline / "electrodialysis.py"
        )
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        old_cc = module.electrodialysis_point(spec.cell_pair, ns, 0.001, "A")
        old_cv = module.electrodialysis_point(spec.cell_pair, ns, 0.2, "V")
        record["old_points"] = {"cc": old_cc, "cv": old_cv}
        new_cc = electrodialysis_point(spec.cell_pair, ns, 0.001, "A")
        new_cv = electrodialysis_point(spec.cell_pair, ns, 0.2, "V")
        original_world = electrodialysis_world_spec(spec.cell_pair, *ns)
        payload = original_world.to_dict()
        payload["physics"] = replace(
            original_world.physics, lab_clock=LabClockSpec(vessel_capacity=2)
        ).to_dict()
        payload["components"].append(
            {
                "id": "legacy-clock",
                "kind": "lab_schedule",
                "vessel": "reactor",
                "parameters": {"duration_range_s": [1e-6, 14400]},
            }
        )
        runtime = ChemWorldRuntime(composition=WorldCompositionSpec.from_dict(payload))
        restored = None
        try:
            runtime.reset(seed=48)
            configured = runtime.execute(
                {"operation": "configure_electrical_control", "current_A": 0.001}
            )
            advanced = runtime.execute({"operation": "wait", "duration_s": 0.5})
            snapshot = runtime.snapshot().to_dict()
            restored = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(snapshot)))
            original_tail = runtime.execute({"operation": "wait", "duration_s": 0.5})
            restored_tail = restored.execute({"operation": "wait", "duration_s": 0.5})
            record["legacy_transactions"] = [
                configured.info,
                advanced.info,
                original_tail.info,
                restored_tail.info,
            ]
            record["comparisons"] = [
                exact("legacy-CC-full-point", old_cc == new_cc),
                exact("legacy-CV-full-point", old_cv == new_cv),
                exact(
                    "legacy-codec",
                    module.ElectrodialysisSpec.from_dict(spec.cell_pair.to_dict()).to_dict()
                    == spec.cell_pair.to_dict(),
                ),
                exact(
                    "legacy-configure-wait",
                    configured.info["transaction_status"]
                    == advanced.info["transaction_status"]
                    == "committed",
                ),
                exact("legacy-positive-work", runtime._state.ledger.flow_electrical_work_J > 0),
                exact(
                    "legacy-exact-resume",
                    runtime.snapshot().to_dict() == restored.snapshot().to_dict()
                    and original_tail.info["transaction_status"]
                    == restored_tail.info["transaction_status"]
                    == "committed",
                ),
            ]
        finally:
            runtime.close()
            if restored:
                restored.close()


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows, started = [], time.monotonic()
    for index, name in enumerate(
        ("loaded-point", "finite-four-ion", "actual-shared-world", "legacy-powered-ed")
    ):
        record = {"unit": name, "comparisons": []}
        print(f"48E canary starting {index + 1}/4 {name}", flush=True)
        try:
            run_unit(index, record, args.baseline)
        except Exception as error:
            record.update(exception=str(error), traceback=traceback.format_exc())
        while len(record["comparisons"]) < 6:
            record["comparisons"].append(
                exact("unexecuted-slot-" + str(len(record["comparisons"])), False)
            )
        record["passed"] = all(c["passed"] for c in record["comparisons"])
        rows.append(record)
        (args.output / f"{index + 1:02}-{name}.json").write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        elapsed = time.monotonic() - started
        print(
            f"48E canary {index + 1}/4 pass={record['passed']}; "
            f"{(index + 1) / elapsed:.2f} unit/s; ETA={(3 - index) * elapsed / (index + 1):.1f}s",
            flush=True,
        )
    summary = {
        "units": 4,
        "comparisons": 24,
        "units_passed": sum(r["passed"] for r in rows),
        "comparisons_passed": sum(c["passed"] for r in rows for c in r["comparisons"]),
        "elapsed_s": time.monotonic() - started,
        "failures": [
            {"unit": r["unit"], "comparison": c["name"], "exception": r.get("exception")}
            for r in rows
            for c in r["comparisons"]
            if not c["passed"]
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps({k: summary[k] for k in ("units_passed", "comparisons_passed", "elapsed_s")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
