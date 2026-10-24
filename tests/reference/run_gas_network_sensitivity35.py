"""Prospective 12-unit/144-slot original gas-network response qualification."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import replace
from pathlib import Path

import gas_network_sensitivity35_cases as cases
import gas_network_sensitivity35_reference as reference
import numpy as np

from chemworld.foundation import WorldState
from chemworld.physchem.gas_network_sensitivity import (
    GasSensitivityParameter,
    GasSensitivityPolicy,
    gas_network_trajectory_sensitivities,
    gas_sensitivity_observables,
)
from chemworld.physchem.gas_reactor_network import pack_gas_reactor_network_state
from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.gas_network_sensitivity import analyze_gas_network_world_state
from chemworld.runtime.gas_reactor_network import unpack_gas_network
from chemworld.runtime.scheduling import local_states
from chemworld.world.gas_reactor_network import gas_reactor_network_world

ATOL = {"mol": 2e-7, "K": 0.02, "Pa": 0.2, "J": 0.003}


def snapshot(state):
    return {
        "amounts": dict(state.amounts_mol),
        "temperature": state.temperature_K,
        "time": state.time_s,
        "coordinates": dict(state.coordinates),
    }


def work_matches(result):
    work = result["work"]
    advances = [a for c in result["trajectories"] for a in c["advances"]]
    return len(advances) == work["attempted_advances"] == work["completed_advances"] + work[
        "failed_advances"
    ] and all(
        work["reported_" + field] == sum((a["solver"] or {}).get(field, 0) or 0 for a in advances)
        for field in ("nfev", "njev", "nlu")
    )


class Unit:
    def __init__(self, index):
        self.index, self.last = index, time.monotonic()
        self.record = {
            "unit": index,
            "checks": [
                {"label": label, "status": "not_executed", "passed": False}
                for label in (cases.NUMERICAL_SLOTS if index <= 8 else cases.CONTRACT_SLOTS[index])
            ],
            "analyses": [],
            "exceptions": [],
        }

    def check(self, label, actual, expected=True):
        row = next(c for c in self.record["checks"] if c["label"] == label)
        row.update(
            status="evaluated", actual=actual, expected=expected, passed=bool(actual == expected)
        )

    def compare(self, label, rows, expected, units):
        details = []
        for index, (row, other) in enumerate(zip(rows, expected, strict=True)):
            for key, value in row.items():
                tolerance = ATOL[units[key]] + 0.003 * abs(other[key])
                details.append(
                    {
                        "sample": index,
                        "key": key,
                        "actual": value,
                        "expected": other[key],
                        "tolerance": tolerance,
                        "passed": abs(value - other[key]) <= tolerance,
                    }
                )
        self.check(label, all(d["passed"] for d in details))
        next(c for c in self.record["checks"] if c["label"] == label)["values"] = details

    def progress(self, message):
        if time.monotonic() - self.last >= 20:
            print(json.dumps({"unit": self.index, **message}), flush=True)
            self.last = time.monotonic()

    def run(self, raw, bindings=None, times=cases.TIMES, **options):
        spec, network, device = cases.native(raw)
        state = pack_gas_reactor_network_state(spec, network, raw["amounts"])
        before = snapshot(state)
        parameters = tuple(
            GasSensitivityParameter(*b) for b in (bindings or cases.bindings(raw["unit_index"]))
        )
        result = gas_network_trajectory_sensitivities(
            spec,
            network,
            state,
            parameters,
            times,
            thermal_device=options.pop("thermal_device", device),
            progress=self.progress,
            **options,
        )
        self.record["analyses"].append(result)
        return result, before == snapshot(state)


def numerical(unit):
    raw = cases.card(unit.index)
    result, unchanged = unit.run(raw)
    expected = reference.reference(raw, cases.bindings(unit.index), cases.TIMES)
    unit.record["reference"] = expected
    base = [row["values"] for row in result["trajectories"][0]["records"]]
    unit.compare("baseline", base, expected["baseline"], result["units"])
    coarse, fine, truth = [], [], []
    for response, independent in zip(result["responses"], expected["derivatives"], strict=True):
        for row, other in zip(response["records"], independent, strict=True):
            coarse.append({k: v["coarse_dy_dlnp"] for k, v in row["coefficients"].items()})
            fine.append({k: v["fine_dy_dlnp"] for k, v in row["coefficients"].items()})
            truth.append(other)
    unit.compare("coarse", coarse, truth, result["units"])
    unit.compare("fine", fine, truth, result["units"])
    unit.check(
        "step_convergence",
        all(
            abs(f[key] - t[key]) <= abs(c[key] - t[key]) + ATOL[result["units"][key]]
            for c, f, t in zip(coarse, fine, truth, strict=True)
            for key in f
        ),
    )
    width = 8 * len(raw["nodes"]) + 3 * len(raw["edges"]) + len(raw["heat_links"])
    unit.check(
        "units",
        len(result["units"]) == width
        and all(v in ATOL for v in result["units"].values())
        and all(p["parameter_unit"] for p in result["parameters"]),
    )
    coefficients = [
        v
        for response in result["responses"]
        for row in response["records"]
        for v in row["coefficients"].values()
    ]
    unit.check(
        "nonzero_normalization",
        all(
            v["normalized"] == v["fine_dy_dlnp"] / v["baseline"]
            for v in coefficients
            if v["baseline"] != 0
        ),
    )
    unit.check(
        "zero_normalization",
        any(v["baseline"] == 0 for v in coefficients)
        and all(v["normalized"] is None for v in coefficients if v["baseline"] == 0),
    )
    elemental = [
        sum(
            row[f"node:{n['id']}:amount:A"] * (1 if raw.get("dimer") else 2)
            + row[f"node:{n['id']}:amount:B"] * 2
            for n in raw["nodes"]
        )
        for row in fine
    ]
    unit.check("element_closure", max(abs(x) for x in elemental) <= 2e-7)
    first_law = []
    for row in fine:
        for node in raw["nodes"]:
            name = node["id"]
            incoming = sum(
                row[f"edge:{edge['id']}:enthalpy"]
                for edge in raw["edges"]
                if edge["target"] == name
            )
            outgoing = sum(
                row[f"edge:{edge['id']}:enthalpy"]
                for edge in raw["edges"]
                if edge["source"] == name
            )
            first_law.append(
                row[f"node:{name}:delta_U"]
                - row[f"node:{name}:bath"]
                + row[f"node:{name}:loss"]
                - row[f"node:{name}:conducted"]
                - incoming
                + outgoing
            )
    unit.check("first_law", max(abs(x) for x in first_law) <= 0.006)
    planned = 1 + 4 * len(cases.bindings(unit.index))
    unit.check(
        "denominators",
        result["work"]["completed_trajectories"] == planned
        and result["work"]["evaluated_time_slots"] == planned * 5
        and len(fine) * width == 5 * width * len(cases.bindings(unit.index)),
    )
    unit.check(
        "work",
        work_matches(result)
        and result["work"]["attempted_advances"] == planned * 4
        and result["work"]["reported_nfev"] > 0,
    )
    unit.check("immutable", unchanged)


def boundary(unit):
    raw = cases.card(3)
    raw["amounts"]["source"]["B"] = 0.0
    result, unchanged = unit.run(raw, times=(0.0, 0.1))
    initial = result["responses"][0]["records"][0]["coefficients"]
    final = result["responses"][0]["records"][1]["coefficients"]
    unit.check("zero_y", initial["node:source:amount:B"]["normalized"] is None)
    unit.check("zero_dy", initial["node:source:amount:A"]["normalized"], 0.0)
    unit.check(
        "response_signs",
        final["node:source:amount:A"]["fine_dy_dlnp"]
        < 0
        < final["node:source:amount:B"]["fine_dy_dlnp"],
    )
    # Disabled edge is authored through the original immutable spec.
    valve = cases.card(5)
    spec, network, _ = cases.native(valve)
    state = pack_gas_reactor_network_state(spec, network, valve["amounts"])
    parameter = GasSensitivityParameter("flow", "feed", "coefficient")
    disabled_spec = replace(spec, connections=(replace(spec.connections[0], enabled=False),))
    disabled = gas_network_trajectory_sensitivities(
        disabled_spec, network, state, (parameter,), (0.0, 0.1)
    )
    unit.record["analyses"].append(disabled)
    unit.check(
        "disabled_valve",
        all(
            v["fine_dy_dlnp"] == 0
            for row in disabled["responses"][0]["records"]
            for v in row["coefficients"].values()
        ),
    )
    valve["amounts"] = {"source": {"A": 0.003, "B": 0.002}, "receiver": {"A": 0.08, "B": 0.02}}
    adverse, _ = unit.run(valve, times=(0.0, 0.1))
    unit.check(
        "adverse_valve",
        adverse["status"] == "piecewise_finite_difference_uncertified"
        and all(
            v["fine_dy_dlnp"] == 0
            for row in adverse["responses"][0]["records"]
            for v in row["coefficients"].values()
        ),
    )
    valve = cases.card(5)
    valve["amounts"]["receiver"] = {"A": 0.054, "B": 0.036}
    equal, _ = unit.run(valve, times=(0.0, 0.1))
    unit.check("valve_kink", equal["smoothness"] == "piecewise_not_certified")
    rejected = []
    for label, altered, pars in (
        (
            "zero_parameter",
            replace(spec, connections=(replace(spec.connections[0], coefficient=0),)),
            (parameter,),
        ),
        ("unknown_parameter", spec, (GasSensitivityParameter("flow", "absent", "coefficient"),)),
        ("duplicate_parameter", spec, (parameter, parameter)),
    ):
        failure = None
        try:
            gas_network_trajectory_sensitivities(altered, network, state, pars, (0.0, 0.1))
        except ValueError as error:
            failure = str(error)
        rejected.append({"label": label, "failure": failure})
        unit.check(label, failure is not None)
    unit.record["rejections"] = rejected
    unit.check("units", all(v["unit_dy_dp"] for v in final.values()))
    unit.check(
        "no_filling", result["trajectories"][0]["records"][0]["values"]["node:source:amount:B"], 0.0
    )
    unit.check("immutable", unchanged)


def events(unit):
    raw = cases.card(2)
    raw["nodes"][0]["minimum_mol"] = 0.015
    result, unchanged = unit.run(raw, times=(0.0, 4.0, 5.0, 6.0))
    trajectories = result["trajectories"]
    unit.check("analytic_event", abs(trajectories[0]["actual_end_time_s"] - 5) <= 2e-6)
    unit.check("baseline_endpoint", trajectories[0]["status"], "stopped")
    for label, indices in (("coarse_endpoints", (1, 2)), ("fine_endpoints", (3, 4))):
        unit.check(
            label,
            all(
                abs(
                    trajectories[i]["actual_end_time_s"]
                    - 5 * np.exp(-trajectories[i]["log_offset"])
                )
                <= 2e-6
                for i in indices
            ),
        )
    response = result["responses"][0]["records"]
    unit.check(
        "pre_event",
        abs(response[1]["coefficients"]["node:source:amount:A"]["fine_dy_dlnp"] + 0.008) < 2e-7,
    )
    unit.check("post_event", all(row["coefficients"] is None for row in response[2:]))
    unit.check("event_topology", result["status"], "incomplete_finite_difference")
    unit.check("no_extrapolation", all(c["records"][-1]["values"] is None for c in trajectories))
    unit.check("denominator", result["work"]["planned_time_slots"], 20)
    unit.check("root_reason", all(c["stopping_event"] is not None for c in trajectories))
    unit.check("work", work_matches(result))
    unit.check("immutable", unchanged)


def world(unit, directory):
    raw = cases.card(8)
    spec, network, _ = cases.native(raw)
    device = ThermalDeviceSpec(
        setpoint_K=900, minimum_setpoint_K=200, maximum_setpoint_K=1500, enabled=False
    )
    composition = gas_reactor_network_world(
        spec, network, raw["amounts"], anchor_thermal_device=device
    )
    runtime = ChemWorldRuntime(composition=composition)
    parameters = (GasSensitivityParameter("flow", "feed", "coefficient"),)
    try:
        runtime.reset(seed=7)
        runtime.execute({"operation": "wait", "duration_s": 0.25})
        runtime.execute({"operation": "select_vessel", "vessel": "source"})
        before = runtime._state.to_dict()
        result = analyze_gas_network_world_state(
            runtime._state, composition, parameters, (0.0, 0.1, 0.3), progress=unit.progress
        )
        unit.record["analyses"].append(result)
        unit.check("current_time", result["start_time_s"], 0.25)
        unit.check("anchor", result["world_consumer"]["anchor_vessel"], "reactor")
        unit.check("ledger", runtime._state.to_dict()["ledger"], before["ledger"])
        unit.check("snapshot", runtime._state.to_dict(), before)
        restored = WorldState.from_dict(json.loads(json.dumps(before)))
        replay = analyze_gas_network_world_state(restored, composition, parameters, (0.0, 0.1, 0.3))
        unit.check("json", replay, result)
        input_path, output_path = directory / "world-input.json", directory / "world-output.json"
        input_path.write_text(
            json.dumps({"state": before, "composition": composition.to_dict()}), encoding="utf-8"
        )
        script = """import json,sys
from pathlib import Path
from chemworld.foundation import WorldState
from chemworld.world.composition import WorldCompositionSpec
from chemworld.physchem.gas_network_sensitivity import GasSensitivityParameter
from chemworld.runtime.gas_network_sensitivity import analyze_gas_network_world_state
p=json.loads(Path(sys.argv[1]).read_text())
r=analyze_gas_network_world_state(WorldState.from_dict(p['state']),WorldCompositionSpec.from_dict(p['composition']),(GasSensitivityParameter('flow','feed','coefficient'),),(0.,.1,.3))
Path(sys.argv[2]).write_text(json.dumps(r,allow_nan=False))
"""
        process = subprocess.run(
            [sys.executable, "-c", script, str(input_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        unit.record["child_process"] = {
            "returncode": process.returncode,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        unit.check(
            "new_process", process.returncode == 0 and json.loads(output_path.read_text()) == result
        )
        unit.check(
            "connections",
            result["world_consumer"]["connection_ids"],
            ["feed", "product", "recycle"],
        )
        locals_ = local_states(runtime._state)
        anchor = locals_["reactor"].replace(
            inactive_vessels={key: value for key, value in locals_.items() if key != "reactor"}
        )
        _, _, origin = unpack_gas_network(anchor, network)
        action = runtime.execute({"operation": "wait", "duration_s": 0.1})
        locals_ = local_states(runtime._state)
        anchor = locals_["reactor"].replace(
            inactive_vessels={key: value for key, value in locals_.items() if key != "reactor"}
        )
        _, _, after = unpack_gas_network(anchor, network)
        actual = gas_sensitivity_observables(spec, network, after, origin)[0]
        modeled = result["trajectories"][0]["records"][1]["values"]
        n_t = [key for key in actual if ":amount:" in key or key.endswith(":temperature")]
        others = [key for key in actual if key not in n_t]
        unit.check(
            "baseline_n_T",
            action.info["operation_outcome"] == "success"
            and all(abs(actual[key] - modeled[key]) <= ATOL[result["units"][key]] for key in n_t),
        )
        unit.check(
            "baseline_P_heat",
            all(abs(actual[key] - modeled[key]) <= ATOL[result["units"][key]] for key in others),
        )
        runtime.execute({"operation": "select_vessel", "vessel": "reactor"})
        runtime.execute(
            {"operation": "queue_lab_heat", "duration_s": 1, "target_temperature_K": 900}
        )
        queued = runtime._state.to_dict()
        refused = False
        try:
            analyze_gas_network_world_state(runtime._state, composition, parameters, (0.0, 0.1))
        except ValueError as error:
            refused = "scheduled-history" in str(error)
            unit.record["queue_refusal"] = str(error)
        unit.check(
            "no_live_charge",
            refused
            and runtime._state.to_dict() == queued
            and result["world_consumer"]["live_operations_executed"] == 0,
        )
        unit.check("work", work_matches(result))
        unit.check(
            "denominator",
            [result["work"]["completed_trajectories"], result["work"]["completed_advances"]],
            [5, 10],
        )
    finally:
        runtime.close()


def failures(unit):
    raw = cases.card(3)
    result, unchanged = unit.run(raw, policy=GasSensitivityPolicy(max_advances=1))
    unit.check("budget", result["work"]["attempted_advances"], 1)
    unit.check("parameter_denominator", len(result["responses"]), 2)
    unit.check(
        "unattempted",
        result["work"]["unavailable_time_slots"] > 0
        and all(len(c["records"]) == 5 for c in result["trajectories"]),
    )
    spec, network, _ = cases.native(raw)
    state = pack_gas_reactor_network_state(spec, network, raw["amounts"])
    parameter = (GasSensitivityParameter("flow", "feed", "coefficient"),)
    rejected = []
    for label in ("nonfinite", "overflow", "spatial", "thermo_parameter", "time_grid", "step"):
        failure = None
        try:
            if label == "nonfinite":
                replace(spec.connections[0], coefficient=float("inf"))
            elif label == "overflow":
                altered = replace(
                    spec,
                    connections=(replace(spec.connections[0], coefficient=sys.float_info.max),),
                )
                gas_network_trajectory_sensitivities(altered, network, state, parameter, (0.0, 0.1))
            elif label == "spatial":
                import gas_spatial36_cases

                spatial_card = gas_spatial36_cases.case()
                spatial_spec, spatial_network, spatial_amounts = gas_spatial36_cases.native(
                    spatial_card
                )
                spatial_state = pack_gas_reactor_network_state(
                    spatial_spec, spatial_network, spatial_amounts
                )
                gas_network_trajectory_sensitivities(
                    spatial_spec, spatial_network, spatial_state, parameter, (0.0, 0.1)
                )
            elif label == "thermo_parameter":
                GasSensitivityParameter("thermochemistry", "A", "enthalpy")
            elif label == "time_grid":
                gas_network_trajectory_sensitivities(spec, network, state, parameter, (1.0, 0.5))
            else:
                GasSensitivityPolicy(log_step=0)
        except ValueError as error:
            failure = str(error)
        rejected.append({"label": label, "failure": failure})
        unit.check(label, failure is not None)
    unit.record["rejections"] = rejected
    thermal = cases.card(8)
    failed, same = unit.run(
        thermal,
        bindings=(("flow", "feed", "coefficient"),),
        times=(0.0, 0.1),
        thermal_device=ThermalDeviceSpec(
            setpoint_K=1800, minimum_setpoint_K=200, maximum_setpoint_K=2000
        ),
    )
    unit.check(
        "forward_failure",
        failed["work"]["failed_advances"] == 5
        and len(failed["failures"]) == 5
        and work_matches(failed),
    )
    unit.check("strict_json", bool(json.dumps(failed, allow_nan=False)))
    unit.check("immutable", unchanged and same)


def summarize(payload, full_evidence):
    """Derive a readable report without repeating the preserved raw trajectories."""
    summary = json.loads(json.dumps(payload, allow_nan=False))
    for unit in summary["units"]:
        reference = unit.get("reference")
        if reference:
            unit["reference"] = {k: v for k, v in reference.items()
                                 if k not in {"baseline", "derivatives"}}
        for analysis in unit["analyses"]:
            analysis.pop("responses", None)
            for candidate in analysis["trajectories"]:
                candidate.pop("records", None)
                advances = candidate.pop("advances")
                candidate["advance_status_counts"] = {
                    status: sum(a["status"] == status for a in advances)
                    for status in sorted({a["status"] for a in advances})
                }
                candidate["failed_advances"] = [
                    a for a in advances if a["status"] not in {"completed", "not_attempted"}
                ]
        for check in unit["checks"]:
            if check["label"] in {"snapshot", "ledger", "json"} and check["passed"]:
                check["actual"], check["expected"] = "exactly_equal", "exactly_equal"
            values = check.pop("values", [])
            if values:
                check["value_count"] = len(values)
                check["failed_values"] = [v for v in values if not v["passed"]]
                check["worst_by_quantity"] = {
                    key: max((v for v in values if v["key"] == key),
                             key=lambda v: abs(v["actual"] - v["expected"]) / v["tolerance"])
                    for key in sorted({v["key"] for v in values})
                }
    summary["full_evidence"] = str(full_evidence)
    summary["summary_scope"] = (
        "All comparison outcomes, all failing values, worst normalized error per quantity, "
        "candidate status/events and resource counts. Full vectors and trajectories remain "
        "in full_evidence; summarization does not execute the experiment."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.summary.exists():
        raise ValueError("preserve attempts; choose new output paths")
    payload = {
        "schema": "chemworld-35C-fixed-development-qualification-1",
        "planned_units": 12,
        "planned_comparisons": 144,
        "units": [],
    }
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="chemworld-35C-") as temporary:
        for index in range(1, 13):
            unit = Unit(index)
            print(
                json.dumps(
                    {
                        "stage": "unit_start",
                        "unit": index,
                        "completed_units": index - 1,
                        "planned_units": 12,
                    }
                ),
                flush=True,
            )
            try:
                if index <= 8:
                    numerical(unit)
                elif index == 9:
                    boundary(unit)
                elif index == 10:
                    events(unit)
                elif index == 11:
                    world(unit, Path(temporary))
                else:
                    failures(unit)
            except Exception as error:
                unit.record["exceptions"].append(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            payload["units"].append(unit.record)
            args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            elapsed = time.monotonic() - start
            print(
                json.dumps(
                    {
                        "stage": "unit_complete",
                        "unit": index,
                        "completed_units": index,
                        "planned_units": 12,
                        "passed": sum(c["passed"] for c in unit.record["checks"]),
                        "comparisons": 12,
                        "elapsed_s": elapsed,
                        "units_per_s": index / elapsed,
                        "eta_s": elapsed * (12 - index) / index,
                    }
                ),
                flush=True,
            )
    checks = [c for u in payload["units"] for c in u["checks"]]
    payload.update(
        elapsed_s=time.monotonic() - start,
        attempted_units=12,
        passed_units=sum(all(c["passed"] for c in u["checks"]) for u in payload["units"]),
        evaluated_comparisons=sum(c["status"] == "evaluated" for c in checks),
        passed_comparisons=sum(c["passed"] for c in checks),
        failed_comparisons=sum(c["status"] == "evaluated" and not c["passed"] for c in checks),
        unexecuted_comparisons=sum(c["status"] == "not_executed" for c in checks),
    )
    payload["failures"] = [
        {"unit": u["unit"], **c} for u in payload["units"] for c in u["checks"] if not c["passed"]
    ]
    analyses = [a for u in payload["units"] for a in u["analyses"]]
    payload["total_work"] = {
        key: sum(a["work"][key] for a in analyses) for key in analyses[0]["work"]
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    summary = summarize(payload, args.output)
    args.summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {k: v for k, v in payload.items() if k not in {"units", "failures", "total_work"}}
        ),
        flush=True,
    )
    if payload["passed_comparisons"] != 144:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
