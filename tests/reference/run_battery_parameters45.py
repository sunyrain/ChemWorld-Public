"""Frozen Chen2020 comparison tooling; execution requires an explicit coordinator GO.

With no --execute this only prints the prospective plan. Every executed route
uses a fresh subprocess and immutable compressed checkpoints. The assessment
always reports the complete denominator, including absent and failed histories.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping
from contextlib import suppress
from copy import deepcopy
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
from battery_parameters45_cases import (
    CASE_IDS,
    PROFILE_R_OVER_R,
    RESPONSE_ORDER,
    THRESHOLD_CANDIDATES,
    case_data,
    native_spec,
    routes,
    stage_times,
)

# These prospective values are fixed for the experiment note, before execution.
THRESHOLDS = deepcopy(THRESHOLD_CANDIDATES)
ROUTE_CHECKS = (
    "available", "execution", "identity", "mesh", "released-initial-state", "stage-count",
)
STAGE_CHECKS = (
    "present", "schema", "boundary-continuity", "requested-observations", "stopping-identity",
    "stopping-value", "hardware-current", "imposed-control", "particle-lithium",
    "negative-faradaic", "positive-faradaic", "physical-domain", "raw-state-retention",
    "all-original-times-retained",
)
METRICS = (*RESPONSE_ORDER, "negative_profile_fraction", "positive_profile_fraction")
PAIR_CHECKS = ("both-present", "common-time-support", "termination-kind", "stop-time",
               *("common:" + name for name in METRICS),
               *("endpoint:" + name for name in METRICS))
COMPARISONS = (
    ("cross-coarse", ("native", "coarse"), ("source", "coarse"), 1.),
    ("cross-fine", ("native", "fine"), ("source", "fine"), 1.),
    ("native-refinement", ("native", "coarse"), ("native", "fine"), .5),
    ("source-refinement", ("source", "coarse"), ("source", "fine"), .5),
)


def json_value(value):
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    return value


def route_id(route):
    return "/".join(route[k] for k in ("unit", "implementation", "grid_level"))


def plan():
    stages = sum(len(case_data(i)["stages"]) for i in CASE_IDS)
    observations = sum(s["samples"] for i in CASE_IDS for s in case_data(i)["stages"])
    return {
        "schema": "parameters45-plan-v1", "units": list(CASE_IDS), "routes": routes(),
        "histories": 24, "authored_stages": stages * 4,
        "authored_observation_slots": observations * 4,
        "observation_semantics": "Slots after a true cutoff are not required observations. "
        "All returned original points, exact boundaries, event endpoints and tails are retained.",
        "comparisons": len(COMPARISONS) * len(CASE_IDS),
        "comparison_stages": stages * len(COMPARISONS),
        "route_check_names": ROUTE_CHECKS, "stage_check_names": STAGE_CHECKS,
        "comparison_stage_check_names": PAIR_CHECKS,
        "check_denominator": 24 * len(ROUTE_CHECKS) + stages * 4 * len(STAGE_CHECKS)
        + stages * len(COMPARISONS) * len(PAIR_CHECKS),
        "thresholds": deepcopy(THRESHOLDS),
        "comparison_edges": COMPARISONS,
        "cases": [case_data(i) for i in CASE_IDS],
        "execution_authorized_by_this_plan": False,
    }


def archive_new(path, value):
    """Exclusive creation and decoded-byte verification; never overwrite an attempt."""
    payload = json.dumps(json_value(value), separators=(",", ":"), allow_nan=False).encode()
    with path.open("xb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as archive:
        archive.write(payload)
    with gzip.open(path, "rb") as archive:
        recovered = archive.read()
    if recovered != payload:
        raise OSError("compressed history did not reproduce its original bytes")
    return {"path": str(path), "uncompressed_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_archive(path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def write_new(path, value):
    with path.open("x") as stream:
        json.dump(json_value(value), stream, indent=2, allow_nan=False)
        stream.write("\n")


def _state_snapshot(state):
    return {field.name: json_value(getattr(state, field.name)) for field in fields(state)}


def native_time(data, radial_cells, *, progress_callback=None):
    """Advance the authentic source-bound builder through the existing residual path."""
    import chemworld.physchem.coupled_residual as residual_module
    from chemworld.physchem.battery_models import (
        F,
        battery_control_current,
        battery_readout,
        battery_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, radial_cells)
    state, segments = spec.initial_state(), []
    contract = {
        "implementation": "native", "spec": spec.to_dict(), "F_C_mol": F,
        "binding_validated_against_actual_spec": spec.parameter_binding.validate_projection(spec),
        "response_order": RESPONSE_ORDER,
        "mesh": {name: {"class": type(getattr(spec, name).mesh).__name__,
                        "nodes_m": list(getattr(spec, name).mesh.centers_m),
                        "edges_m": list(getattr(spec, name).mesh.faces_m),
                        "cell_volumes_m3": list(getattr(spec, name).mesh.cell_volumes_m3),
                        "cells": radial_cells, "radius_m": getattr(spec, name).radius_m,
                        "mesh_exponent": getattr(spec, name).mesh_exponent}
                 for name in ("negative", "positive")},
        "residual_observation": "Wrap original solve_residual once and return its report unchanged;"
        "original report points include every exposed state, rate and algebraic coordinate. "
        "Hidden internal solver steps are not exposed.",
    }
    progress = {"contract": contract, "case": deepcopy(data), "segments": segments}

    def publish(**extra):
        progress.update(extra)
        if progress_callback is not None:
            progress_callback(progress)

    def record(current_state, program, system):
        current = battery_control_current(spec, current_state, program)
        reading = battery_readout(spec, current_state, current)
        scalars = {
            "voltage_V": reading.voltage_V, "current_A": current,
            "charge_C": current_state.coordinates["discharged_charge_C"],
            "work_J": current_state.coordinates["delivered_work_J"],
        }
        profiles = {}
        for name in ("negative", "positive"):
            particle = getattr(spec, name)
            amounts = np.asarray([current_state.amounts_mol[k] for k in particle.keys(name)])
            concentration = amounts / particle.mesh.cell_volumes_m3
            profiles[name] = {"amounts_mol": amounts.tolist(),
                              "concentration_mol_m3": concentration.tolist(),
                              "fraction": (concentration
                                           / particle.maximum_concentration_mol_m3).tolist()}
            scalars[name + "_Li_mol"] = float(amounts.sum())
            scalars[name + "_surface_fraction"] = getattr(reading, name + "_surface_fraction")
            scalars[name + "_average_fraction"] = getattr(reading, name + "_average_fraction")
            scalars[name + "_ocp_V"] = spec.ocv(name, scalars[name + "_surface_fraction"])
            scalars[name + "_overpotential_V"] = getattr(reading, name + "_overpotential_V")
        return {"time_s": current_state.time_s,
                "row": [scalars[name] for name in RESPONSE_ORDER], "profiles": profiles,
                "native_state": _state_snapshot(current_state),
                "native_events": {event.event_id: float(event.value(current_state))
                                  for event in system.all_events}}

    for index, stage in enumerate(data["stages"]):
        boundary, captured = None, []
        try:
            program = ScalarControlProgram(((0., stage["target"]),), stage["unit"])
            system = battery_system(spec, program)
            boundary = record(state, program, system)
            publish(active_stage=index, active_request=deepcopy(stage), boundary_t0=boundary,
                    phase="before-native-integration")
            original_solver = residual_module.solve_residual

            def observe(residual, *args, original_solver=original_solver,
                        captured=captured, **kwargs):
                report = original_solver(residual, *args, **kwargs)
                captured.append({
                    "variables": [asdict(v) for v in residual.variables],
                    "equations": [asdict(e) for e in residual.equations],
                    "differential_indices": residual.differential.tolist(),
                    "algebraic_indices": residual.algebraic.tolist(),
                    "points": [{"relative_time_s": p.time_s, "y": p.state.tolist(),
                                "ydot": p.rate.tolist()} for p in report.points],
                    "events": [{"name": name, "relative_time_s": p.time_s,
                                "y": p.state.tolist(), "ydot": p.rate.tolist()}
                               for name, p in report.events],
                    "success": report.success, "failure_reason": report.failure_reason,
                    "stopping_event": report.stopping_event,
                    "diagnostics": json_value(report.diagnostics),
                })
                publish(phase="native-residual-report-returned", pending_residual_reports=captured)
                return report

            with patch.object(residual_module, "solve_residual", observe):
                result = system.advance(state, stage["duration_s"],
                                        evaluation_times_s=stage_times(stage))
            states = list(result.trajectory)
            if not states or states[-1].time_s != result.final_state.time_s:
                states.append(result.final_state)
            publish(phase="native-solution-returned",
                    pending_native_states=[_state_snapshot(s) for s in states])
            records = [record(s, program, system) for s in states]
            segment = {
                "index": index, "request": deepcopy(stage), "boundary_t0": boundary,
                "requested_start_time_s": state.time_s,
                "requested_end_time_s": state.time_s + stage["duration_s"],
                "requested_relative_times_s": stage_times(stage), "records": records,
                "original_reported_times_s": [s.time_s for s in states],
                "original_residual_reports": captured,
                "actual_start_time_s": state.time_s,
                "actual_end_time_s": result.final_state.time_s,
                "termination": "final time" if result.outcome == "success"
                else result.stopping_event,
                "outcome": result.outcome, "solver": json_value(result.solver_diagnostic),
                "native_solver_policy": asdict(system.policy),
                "conservation_residuals": dict(result.conservation_residuals),
                "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
                "energy_balance_residual_J": result.energy_balance_residual_J,
                "actual_exchanges": json_value(result.actual_exchanges),
            }
            segments.append(segment)
            state = result.final_state
            progress.pop("pending_residual_reports", None)
            progress.pop("pending_native_states", None)
            publish(phase="native-stage-captured")
            if result.outcome != "success":
                break
        except Exception as exc:
            progress.update(failed_stage=index, failed_request=deepcopy(stage),
                            failed_boundary_t0=boundary, last_completed_time_s=state.time_s)
            exc.partial_history = progress
            raise
    return {"contract": contract, "case": deepcopy(data), "segments": segments,
            "actual_stages": len(segments), "requested_stages": len(data["stages"])}


def _stop_kind(value):
    text = str(value).lower().replace("-", " ")
    if "final time" in text:
        return "complete"
    for kind in ("minimum", "maximum"):
        if kind + " voltage" in text:
            return kind + "-voltage"
    return str(value)


def raw_completed_segments(trace):
    """Only returned raw segments, never the classified zero-duration boundary view."""
    if not isinstance(trace, Mapping):
        return []
    segments = trace.get("segments")
    if isinstance(segments, (tuple, list)):
        return list(segments)
    partial = trace.get("partial_history")
    if isinstance(partial, Mapping) and isinstance(partial.get("segments"), (tuple, list)):
        return list(partial["segments"])
    return []


def raw_saved_counts(histories):
    segments = [segment for trace in histories for segment in raw_completed_segments(trace)]
    return {"saved_raw_stages": len(segments),
            "saved_raw_observations": sum(len(s["records"]) for s in segments
                                           if isinstance(s, Mapping)
                                           and isinstance(s.get("records"), (tuple, list)))}


def _finite_array(value, shape):
    try:
        array = np.asarray(value)
        return array.shape == shape and array.dtype.kind in "fiu" and np.isfinite(array).all()
    except (TypeError, ValueError, OverflowError):
        return False


def _source_record_complete(record, contract, *, boundary=False):
    """Validate retained original evidence without creating or repairing any state."""
    if not isinstance(record, Mapping) or not isinstance(contract, Mapping):
        return False
    try:
        rhs, alg = contract["rhs_size"], contract["algebraic_size"]
        if type(rhs) is not int or type(alg) is not int or rhs <= 0 or alg <= 0:
            return False
        time_s = record["time_s"]
        if (isinstance(time_s, bool) or not isinstance(time_s, (int, float))
                or not math.isfinite(time_s) or time_s < 0):
            return False
        for name, shape in (("row", (len(RESPONSE_ORDER),)), ("source_y", (rhs + alg,)),
                            ("source_rhs", (rhs,)), ("source_algebraic", (alg,))):
            if not _finite_array(record.get(name), shape):
                return False
        if boundary and (record.get("differential_state_unchanged_by_boundary_root") is not True
                         or record.get("boundary_operation")
                         != "source algebraic root only; no integration"):
            return False
        for electrode in ("negative", "positive"):
            cells = contract["mesh"][electrode]["cells"]
            if type(cells) is not int or cells < 2:
                return False
            if not _finite_array(record["profiles"][electrode]["fraction"], (cells,)):
                return False
        event_names = [event["name"] for event in contract["all_events"]]
        if not event_names or not isinstance(record.get("source_events"), Mapping):
            return False
        if set(record["source_events"]) != set(event_names):
            return False
        for name in event_names:
            values = record["source_events"][name]
            if not isinstance(values, (tuple, list, np.ndarray)) or not len(values):
                return False
            if not _finite_array(values, (len(values),)):
                return False
        for name in ("source_particle_Li_mol", "source_total_Li_including_electrolyte_mol"):
            if not _finite_array([record.get(name)], (1,)):
                return False
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def _initial_source_stop(trace, partial):
    """Recognize a real source initialization event, never a numerical exception."""
    if not isinstance(trace, Mapping) or not isinstance(partial, Mapping):
        return None
    error = trace.get("exception", {})
    if not isinstance(error, Mapping) or not isinstance(error.get("message"), str):
        return None
    stage, request = partial.get("failed_stage"), partial.get("failed_request")
    if type(stage) is not int or stage < 0 or not isinstance(request, Mapping):
        return None
    if (not {"duration_s", "target", "unit", "samples", "expected"} <= request.keys()
            or not isinstance(request["unit"], str) or request["unit"] not in ("A", "V")
            or not _finite_array([request["duration_s"], request["target"]], (2,))
            or request["duration_s"] <= 0 or type(request["samples"]) is not int
            or request["samples"] < 2 or not isinstance(request["expected"], str)):
        return None
    boundary = partial.get("failed_boundary_t0")
    contract = partial.get("contract")
    message = error.get("message", "").lower()
    if (error.get("type") != "SolverError" or "initial conditions" not in message
            or "events" not in message
            or not _source_record_complete(boundary, contract, boundary=True)):
        return None
    completed = raw_completed_segments(trace)
    if stage != len(completed):
        return None
    if completed:
        try:
            previous = completed[-1]["records"][-1]
            if (not _source_record_complete(previous, contract)
                    or boundary["time_s"] != previous["time_s"]
                    or not np.array_equal(np.asarray(boundary["source_y"])[:contract["rhs_size"]],
                                          np.asarray(previous["source_y"])[:contract["rhs_size"]])):
                return None
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return None
    elif boundary["time_s"] != 0:
        return None
    original = contract.get("original_events", [])
    if not isinstance(original, (tuple, list)):
        return None
    for event in original:
        if not isinstance(event, Mapping) or not isinstance(event.get("name"), str):
            return None
        name = event["name"]
        values = boundary.get("source_events", {}).get(name, [])
        if (event.get("type") == "TERMINATION" and name.lower() in message and values
                and all(math.isfinite(v) for v in values) and min(values) <= 0
                and _stop_kind(name) in {"minimum-voltage", "maximum-voltage"}):
            return {"index": partial["failed_stage"], "request": partial["failed_request"],
                    "boundary_t0": boundary, "records": [boundary],
                    "original_reported_times_s": [],
                    "actual_start_time_s": boundary["time_s"],
                    "actual_end_time_s": boundary["time_s"],
                    "termination": name, "physical_initial_stop": True,
                    "original_exception": error}
    return None


def _view(trace):
    if not isinstance(trace, dict):
        return {}, [], False
    partial = trace.get("partial_history") or {}
    if not isinstance(partial, dict):
        partial = {}
    contract = trace.get("contract", partial.get("contract", {}))
    if not isinstance(contract, dict):
        contract = {}
    segments = raw_completed_segments(trace)
    try:
        initial_stop = _initial_source_stop(trace, partial)
    except (AttributeError, KeyError, TypeError, ValueError):
        initial_stop = None
    if initial_stop is not None:
        segments.append(initial_stop)
    return contract, segments, initial_stop is not None


def _stage_view(segment, contract):
    boundary = segment["boundary_t0"]
    records = [boundary, *(r for r in segment["records"] if r["time_s"] > boundary["time_s"])]
    times = np.asarray([r["time_s"] for r in records], dtype=float)
    rows = np.asarray([r["row"] for r in records], dtype=float)
    if (rows.shape != (len(times), len(RESPONSE_ORDER)) or not len(times)
            or not np.isfinite(rows).all() or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError("invalid original scalar rows or stage times")
    profiles = {}
    for electrode in ("negative", "positive"):
        nodes = np.asarray(contract["mesh"][electrode]["nodes_m"], dtype=float)
        fractions = np.asarray([r["profiles"][electrode]["fraction"] for r in records])
        radius = contract["mesh"][electrode]["radius_m"]
        if (fractions.shape != (len(times), len(nodes)) or not np.isfinite(fractions).all()
                or not np.all(np.diff(nodes) > 0)):
            raise ValueError("invalid original radial profiles")
        target = np.asarray(PROFILE_R_OVER_R)
        if target[0] < nodes[0] / radius or target[-1] > nodes[-1] / radius:
            raise ValueError("comparison radii would require radial extrapolation")
        profiles[electrode] = {
            "original": fractions,
            "common_r": np.asarray([np.interp(target, nodes / radius, row) for row in fractions]),
        }
    return {"times": times, "rows": rows, "profiles": profiles, "raw": segment,
            "records": records, "stop": _stop_kind(segment["termination"])}


def _metric_limit(name, factor):
    values = THRESHOLDS["cross_implementation"]
    if name.endswith("_Li_mol"):
        key = "electrode_Li_mol"
    elif name.endswith("_surface_fraction"):
        key = "surface_fraction"
    elif name.endswith("_average_fraction"):
        key = "average_fraction"
    elif name.endswith("_profile_fraction"):
        key = "radial_profile_fraction"
    elif name.endswith("_V"):
        key = "voltage_V"
    else:
        key = name
    return factor * values[key]


def _check(name, passed, **details):
    return {"name": name, "passed": bool(passed), **json_value(details)}


def _error_norm(left, right):
    difference = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if not difference.size or not np.isfinite(difference).all():
        raise ValueError("empty or nonfinite comparison")
    return float(np.max(np.abs(difference)))


def _stage_checks(view, previous, original, stage, start, implementation, contract):
    result = dict.fromkeys(STAGE_CHECKS, False)
    if view is None:
        return result
    result.update(present=True, schema=True)
    raw, times, rows = view["raw"], view["times"], view["rows"]
    end, initial_stop = times[-1], raw.get("physical_initial_stop", False)
    result["boundary-continuity"] = abs(times[0] - start) <= 1e-8
    if previous is not None:
        result["boundary-continuity"] &= (
            abs(times[0] - previous["times"][-1]) <= 1e-8
            and _error_norm(rows[0, 2:6], previous["rows"][-1, 2:6]) <= 1e-10
            and all(np.array_equal(view["profiles"][e]["original"][0],
                                   previous["profiles"][e]["original"][-1])
                    for e in ("negative", "positive")))
    wanted = np.asarray(stage_times(stage)) + start
    required = wanted[wanted <= end + 1e-8]
    result["requested-observations"] = (
        len(required) > 0 and end <= start + stage["duration_s"] + 1e-8
        and all(np.min(abs(times - t)) <= 1e-8 for t in required)
        and abs(end - raw["actual_end_time_s"]) <= 1e-8)
    expected = stage["expected"]
    result["stopping-identity"] = view["stop"] == expected
    if expected == "complete":
        result["stopping-value"] = abs(end - start - stage["duration_s"]) <= 1e-8
    elif expected in {"minimum-voltage", "maximum-voltage"}:
        cutoff = 2.5 if expected == "minimum-voltage" else 4.2
        tolerance = THRESHOLDS[implementation + "_voltage_event_V"] if implementation == "native" \
            else THRESHOLDS["source_original_voltage_event_V"]
        result["stopping-value"] = (rows[-1, 0] <= cutoff if cutoff == 2.5
                                    else rows[-1, 0] >= cutoff) if initial_stop \
            else abs(rows[-1, 0] - cutoff) <= tolerance
    result["hardware-current"] = np.max(abs(rows[:, 1])) <= 10 + THRESHOLDS["hardware_current_A"]
    if stage["unit"] == "A":
        result["imposed-control"] = np.max(abs(rows[:, 1] - stage["target"])) <= 1e-7
    else:
        result["imposed-control"] = (np.max(abs(rows[:, 0] - stage["target"]))
                                     <= THRESHOLDS["CV_voltage_V"]
                                     and np.max(abs(rows[:, 1])) < 10 - 1e-7)
    result["particle-lithium"] = np.max(abs(rows[:, 4] + rows[:, 5]
                                           - original[4] - original[5])) \
        <= THRESHOLDS["particle_Li_closure_mol"]
    faraday = contract["F_C_mol"]
    for column, sign, name in ((4, 1, "negative-faradaic"), (5, -1, "positive-faradaic")):
        result[name] = np.max(abs(faraday * (rows[:, column] - original[column])
                                  + sign * (rows[:, 2] - original[2]))) \
            <= THRESHOLDS["Faraday_closure_C"]
    result["physical-domain"] = (
        np.min(rows[:, 6:8]) >= .01 - 2e-8 and np.max(rows[:, 6:8]) <= .99 + 2e-8
        and np.min(rows[:, 8:10]) >= 0 and np.max(rows[:, 8:10]) <= 1
        and all(np.min(view["profiles"][e]["original"]) >= -1e-10
                and np.max(view["profiles"][e]["original"]) <= 1 + 1e-10
                for e in ("negative", "positive"))
        and (initial_stop or (np.min(rows[:, 0]) >= 2.5 - 2e-5
                             and np.max(rows[:, 0]) <= 4.2 + 2e-5)))
    if implementation == "source":
        size = contract["rhs_size"] + contract["algebraic_size"]
        result["raw-state-retention"] = (
            _source_record_complete(raw["boundary_t0"], contract, boundary=True)
            and all(_source_record_complete(r, contract) for r in raw["records"]))
        if not initial_stop:
            result["raw-state-retention"] &= (
                len(raw["original_all_ys"]) == len(raw["original_all_ts"])
                and all(np.asarray(y).shape == (size, len(t))
                        for y, t in zip(raw["original_all_ys"], raw["original_all_ts"],
                                        strict=True)))
    else:
        reports = raw["original_residual_reports"]
        result["raw-state-retention"] = bool(reports) and all(
            all(len(p["y"]) == len(report["variables"]) == len(p["ydot"])
                and np.isfinite(np.asarray(p["y"], float)).all()
                and np.isfinite(np.asarray(p["ydot"], float)).all()
                for p in report["points"]) and report["points"]
            for report in reports) and all("native_state" in r for r in view["records"])
    result["all-original-times-retained"] = (
        initial_stop or raw["original_reported_times_s"] == [r["time_s"] for r in raw["records"]])
    return result


def _route_assessment(route, trace):
    identity, implementation = (route[k] for k in ("unit", "implementation"))
    data, key = case_data(identity), route_id(route)
    checks, views, errors = [], [], []
    route_results = dict.fromkeys(ROUTE_CHECKS, False)
    route_results["available"] = isinstance(trace, dict)
    trace = trace if isinstance(trace, dict) else {}
    contract, segments, initial_stop = _view(trace or {})
    route_results["execution"] = bool(trace) and ("exception" not in trace or initial_stop)
    route_results["stage-count"] = len(segments) == len(data["stages"])
    try:
        if implementation == "source":
            manifest = contract["source_parameter_manifest"]
            route_results["identity"] = (
                contract["pybamm_version"] == "25.4.2" and contract["casadi_version"] == "3.6.7"
                and contract["parameter_updates"] == {} and len(manifest) == 94
                and sum(p["kind"] == "number" for p in manifest.values()) == 86
                and sum(p["kind"] == "function" for p in manifest.values()) == 6
                and manifest["Nominal cell capacity [A.h]"]["value"] == 5
                and manifest["Current function [A]"]["value"] == 5
                and len(contract["all_events"]) == len(contract["original_events"]) + 4
                and all(e in contract["all_events"] for e in contract["original_events"]))
        else:
            from chemworld.physchem.battery_models import battery_spec_from_dict

            spec = battery_spec_from_dict(contract["spec"])
            route_results["identity"] = (
                spec.parameter_binding.validate_projection(spec)
                and spec.parameter_binding.source_nominal_capacity_Ah == 5
                and spec.maximum_current_A == 10 and spec.negative.cells == route["radial_cells"])
        mesh_valid = True
        for electrode, radius in (("negative", 5.86e-6), ("positive", 5.22e-6)):
            mesh = contract["mesh"][electrode]
            edges, nodes = np.asarray(mesh["edges_m"]), np.asarray(mesh["nodes_m"])
            expected = np.linspace(0, 1, route["radial_cells"] + 1)
            if implementation == "native":
                expected = np.asarray(getattr(spec, electrode).mesh.faces_m) / radius
                expected_nodes = np.asarray(getattr(spec, electrode).mesh.centers_m) / radius
            else:
                expected_nodes = (expected[1:] + expected[:-1]) / 2
            mesh_valid &= (mesh["cells"] == route["radial_cells"]
                           and mesh["radius_m"] == radius and len(nodes) == route["radial_cells"]
                           and edges.shape == expected.shape
                           and _error_norm(edges / radius, expected) <= 1e-14
                           and _error_norm(nodes / radius, expected_nodes) <= 1e-14
                           and (getattr(spec, electrode).mesh_exponent == 2 if implementation
                                == "native" else mesh["class"] == "Uniform1DSubMesh")
                           and np.all(np.diff(nodes) > 0))
        route_results["mesh"] = bool(mesh_valid)
    except Exception as exc:
        errors.append({"phase": "contract", "type": type(exc).__name__, "message": str(exc)})
    original = None
    for index, stage in enumerate(data["stages"]):
        view = None
        if index < len(segments):
            try:
                view = _stage_view(segments[index], contract)
                if original is None:
                    original = view["rows"][0]
            except Exception as exc:
                errors.append({"phase": "stage-schema", "stage": index,
                               "type": type(exc).__name__, "message": str(exc)})
        views.append(view)
        start = sum(s["duration_s"] for s in data["stages"][:index])
        try:
            results = _stage_checks(view, views[index - 1] if index else None, original,
                                    stage, start, implementation, contract)
        except Exception as exc:
            results = dict.fromkeys(STAGE_CHECKS, False)
            errors.append({"phase": "stage-check", "stage": index,
                           "type": type(exc).__name__, "message": str(exc)})
        checks.extend(_check(f"{key}/stage-{index}/{name}", results[name]) for name in STAGE_CHECKS)
    if original is not None:
        expected_li = (.065 * 1.58 * 8.52e-5 * .75 * 29866,
                       .065 * 1.58 * 7.56e-5 * .665 * 17038)
        route_results["released-initial-state"] = (
            views[0]["times"][0] == 0 and _error_norm(original[2:4], (0, 0)) <= 1e-10
            and _error_norm(original[4:6], expected_li) <= 1e-12
            and all(np.max(abs(views[0]["profiles"][e]["original"][0] - theta)) <= 1e-12
                    for e, theta in (("negative", 29866 / 33133), ("positive", 17038 / 63104))))
    checks[:0] = [_check(key + "/" + name, route_results[name]) for name in ROUTE_CHECKS]
    return {"route": route, "passed": all(c["passed"] for c in checks), "checks": checks,
            "errors": errors, "physical_initial_source_stop": initial_stop,
            "raw_completed_stages": len(raw_completed_segments(trace)),
            "available_physical_stages": sum(v is not None for v in views)}, views


def _sample(view, times):
    scalar = {name: np.interp(times, view["times"], view["rows"][:, i])
              for i, name in enumerate(RESPONSE_ORDER)}
    for electrode in ("negative", "positive"):
        values = view["profiles"][electrode]["common_r"]
        scalar[electrode + "_profile_fraction"] = np.column_stack([
            np.interp(times, view["times"], values[:, j]) for j in range(values.shape[1])])
    return scalar


def _compare_stage(name, left, right, stage, start, factor):
    results = {key: {"passed": False} for key in PAIR_CHECKS}
    results["both-present"]["passed"] = left is not None and right is not None
    tail, error = None, None
    if left is not None and right is not None:
        try:
            end = min(left["times"][-1], right["times"][-1])
            requested = np.asarray(stage_times(stage)) + start
            common = requested[(requested >= max(left["times"][0], right["times"][0]) - 1e-8)
                               & (requested <= end + 1e-8)]
            results["common-time-support"] = {"passed": len(common) > 0,
                                               "count": len(common), "end_s": float(end)}
            results["termination-kind"] = {"passed": left["stop"] == right["stop"],
                                             "left": left["stop"], "right": right["stop"]}
            stop_error = abs(left["times"][-1] - right["times"][-1])
            stop_limit = factor * THRESHOLDS["cross_implementation"]["event_time_s"]
            results["stop-time"] = {"passed": stop_error <= stop_limit,
                                    "max_abs": float(stop_error), "limit": stop_limit}
            values = (_sample(left, common), _sample(right, common))
            endpoints = (_sample(left, [left["times"][-1]]),
                         _sample(right, [right["times"][-1]]))
            for prefix, pair in (("common:", values), ("endpoint:", endpoints)):
                for metric in METRICS:
                    norm = _error_norm(pair[0][metric], pair[1][metric])
                    limit = _metric_limit(metric, factor)
                    results[prefix + metric] = {"passed": norm <= limit, "max_abs": norm,
                                                 "limit": limit, "count": pair[0][metric].size}
            tail = {
                side: {"actual_end_s": float(view["times"][-1]),
                       "points_beyond_paired_interval": int(np.sum(view["times"] > end)),
                       "unpaired_duration_s": float(view["times"][-1] - end),
                       "actual_endpoint_row": view["rows"][-1].tolist(),
                       "raw_termination": view["raw"]["termination"]}
                for side, view in (("left", left), ("right", right))}
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
    checks = [_check(name + "/" + key, **results[key]) for key in PAIR_CHECKS]
    return {"checks": checks, "passed": all(c["passed"] for c in checks),
            "tail_audit": tail, "assessment_error": error}


def assess_histories(histories):
    """Pure finite-data assessment; missing/failed routes retain their full denominator."""
    result = {"schema": "parameters45-assessment-v1", "routes": [], "comparisons": [],
              "checks": [], "thresholds": deepcopy(THRESHOLDS)}
    for identity in CASE_IDS:
        views, data = {}, case_data(identity)
        for route in (r for r in routes() if r["unit"] == identity):
            report, normalized = _route_assessment(route, histories.get(route_id(route)))
            result["routes"].append(report)
            result["checks"].extend(report["checks"])
            views[(route["implementation"], route["grid_level"])] = normalized
        for label, left, right, factor in COMPARISONS:
            comparison = {"unit": identity, "kind": label, "left": left, "right": right,
                          "different_meshes_declared": label.startswith("cross"), "stages": []}
            start = 0.
            for index, stage in enumerate(data["stages"]):
                paired = _compare_stage(f"{identity}/{label}/stage-{index}",
                                        views[left][index], views[right][index], stage, start,
                                        factor)
                comparison["stages"].append(paired)
                result["checks"].extend(paired["checks"])
                start += stage["duration_s"]
            comparison["passed"] = all(s["passed"] for s in comparison["stages"])
            result["comparisons"].append(comparison)
    expected = plan()
    result.update(check_denominator=expected["check_denominator"],
                  evaluated_checks=len(result["checks"]),
                  failed_checks=sum(not c["passed"] for c in result["checks"]),
                  available_histories=sum(isinstance(histories.get(route_id(r)), dict)
                                          for r in routes()),
                  history_denominator=24, comparison_denominator=24,
                  comparison_stage_denominator=48)
    result["all_passed"] = (result["evaluated_checks"] == result["check_denominator"]
                             and result["failed_checks"] == 0)
    result["units"] = [{"unit": identity,
                        "passed": all(c["passed"] for c in result["checks"]
                                      if c["name"].startswith(identity + "/"))}
                       for identity in CASE_IDS]
    return result


def _worker(request_path, directory):
    request = json.loads(request_path.read_text())
    route, data = request["route"], request["case"]
    checkpoints = []

    def checkpoint(value):
        path = directory / f"progress-{len(checkpoints):04d}.json.gz"
        checkpoints.append(archive_new(path, value))
        print(json.dumps({"route": route_id(route), "checkpoint": str(path),
                          "phase": value.get("phase"), "active_stage": value.get("active_stage")}),
              flush=True)

    started = time.monotonic()
    try:
        if route["implementation"] == "native":
            trace = native_time(data, route["radial_cells"], progress_callback=checkpoint)
        else:
            from battery_parameters45_pybamm import reference_time

            trace = reference_time(data, route["radial_cells"], progress_callback=checkpoint)
    except Exception as exc:
        trace = {"exception": {"type": type(exc).__name__, "message": str(exc),
                               "traceback": traceback.format_exc()},
                 "partial_history": getattr(exc, "partial_history", None)}
    trace["execution"] = {"route": route, "elapsed_s": time.monotonic() - started,
                           "checkpoints": checkpoints}
    archive_new(directory / "final.json.gz", trace)
    return 0 if "exception" not in trace else 1


def _latest_partial(directory):
    unreadable = []
    for path in sorted(directory.glob("progress-*.json.gz"), reverse=True):
        try:
            return read_archive(path), unreadable
        except Exception as exc:
            unreadable.append({"path": str(path), "type": type(exc).__name__, "message": str(exc)})
    return None, unreadable


def _terminate_route_group(process):
    """Terminate every descendant in the isolated POSIX process group."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, sig)
        if sig == signal.SIGTERM:
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
    # SIGKILL is sent even if the leader already exited: a descendant may have
    # ignored SIGTERM. Reap the leader after terminating the entire group.
    process.wait(timeout=5)


def execute_route(route, directory, budget_s, *, progress):
    if os.name != "posix":
        raise RuntimeError("route process-group budgets require POSIX; "
                           "this platform is unsupported")
    directory.mkdir()
    request_path = directory / "request.json"
    write_new(request_path, {"route": route, "case": case_data(route["unit"])})
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-request",
               str(request_path), "--worker-directory", str(directory)]
    started, previous_count, heartbeat = time.monotonic(), -1, 0.
    timed_out = False
    with (directory / "stdout.log").open("x") as stdout:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                count = len(list(directory.glob("progress-*.json.gz")))
                if count != previous_count or elapsed - heartbeat >= 20:
                    progress({"event": "route-progress", "route": route_id(route),
                              "elapsed_s": elapsed, "checkpoint_count": count,
                              "budget_s": budget_s})
                    heartbeat, previous_count = elapsed, count
                if elapsed >= budget_s:
                    timed_out = True
                    break
                time.sleep(.2)
        finally:
            _terminate_route_group(process)
    final_path = directory / "final.json.gz"
    try:
        if timed_out or not final_path.exists():
            raise RuntimeError("route exceeded its independent budget" if timed_out
                               else "worker exited without its final artifact")
        trace = read_archive(final_path)
        artifact = {"path": str(final_path),
                    "archive_sha256": hashlib.sha256(final_path.read_bytes()).hexdigest()}
    except Exception as exc:
        partial, unreadable = _latest_partial(directory)
        trace = {"exception": {"type": "RouteTimeout" if timed_out else type(exc).__name__,
                               "message": str(exc)}, "partial_history": partial,
                 "worker_returncode": process.returncode,
                 "unreadable_checkpoints": unreadable}
        artifact = archive_new(directory / "supervisor-failure.json.gz", trace)
    artifact.update(route=route, worker_returncode=process.returncode, timed_out=timed_out,
                    elapsed_s=time.monotonic() - started, budget_s=budget_s,
                    process_group_id=process.pid,
                    budget_scope="POSIX isolated session/process group; TERM then KILL; "
                    "leader reaped",
                    exception=trace.get("exception"),
                    all_artifacts=[str(p) for p in sorted(directory.iterdir())])
    return trace, artifact


def _provenance(protocol):
    import casadi
    import pybamm

    import chemworld

    if (pybamm.__version__, casadi.__version__) != ("25.4.2", "3.6.7"):
        raise RuntimeError("runner requires pinned PyBaMM25.4.2 and CasADi3.6.7")
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    own = Path(__file__).parent
    source = Path(pybamm.__file__).parent
    paths = [own / name for name in ("run_battery_parameters45.py", "battery_parameters45_cases.py",
                                    "battery_parameters45_pybamm.py")]
    paths += [repo / "src/chemworld/physchem" / name for name in
              ("battery_models.py", "battery_materials.py", "battery_parameter_sets.py")]
    paths += [repo / "src/chemworld/data/parameters/battery_chen2020_pybamm_25_4_2.json", protocol,
              source / "input/parameters/lithium_ion/Chen2020.py",
              source / "parameters/lithium_ion_parameters.py",
              source / "models/submodels/external_circuit/function_control_external_circuit.py",
              source / "models/submodels/external_circuit/discharge_throughput.py"]
    # Missing provenance is a preflight failure, rather than an unreported gap.
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python": sys.version, "platform": platform.platform(),
            "chemworld_module": chemworld.__file__, "pybamm_module": pybamm.__file__,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("pybamm", "casadi", "numpy", "scipy")},
            "files_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--route", action="append", default=[])
    parser.add_argument("--route-budget-s", type=float, default=900.)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-directory", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_request is not None:
        return _worker(args.worker_request, args.worker_directory)
    prospective = plan()
    selected = [r for r in routes() if not args.route or route_id(r) in args.route]
    if len(set(args.route)) != len(args.route) or set(args.route) - {route_id(r) for r in routes()}:
        raise ValueError("unknown or duplicate route selection")
    if not math.isfinite(args.route_budget_s) or args.route_budget_s <= 0:
        raise ValueError("route budget must be finite and positive")
    if not args.execute:
        print(json.dumps(json_value({**prospective, "selected_routes": selected}), indent=2))
        return 0
    if args.output_dir is None or args.protocol is None:
        raise ValueError("execution requires a new output directory and the fixed experiment note")
    if os.name != "posix":
        raise RuntimeError("route process-group budgets require POSIX; "
                           "this platform is unsupported")
    output, protocol = args.output_dir.resolve(), args.protocol.resolve(strict=True)
    provenance = _provenance(protocol)
    repo = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    if output.is_relative_to(repo.resolve()):
        raise ValueError("raw scientific histories must remain outside the Git checkout")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "plan.json", prospective)
    write_new(output / "provenance.json", provenance)
    histories, artifacts, started = {}, [], time.monotonic()
    with (output / "progress.jsonl").open("x") as journal:
        def progress(event):
            record = {"total_elapsed_s": time.monotonic() - started, **event}
            line = json.dumps(json_value(record), separators=(",", ":"))
            journal.write(line + "\n")
            journal.flush()
            print(line, flush=True)

        for index, route in enumerate(selected):
            progress({"event": "route-start", "route": route_id(route),
                      "attempted": index, "selected_denominator": len(selected),
                      "full_denominator": 24})
            directory = output / route_id(route).replace("/", "-")
            try:
                trace, artifact = execute_route(route, directory, args.route_budget_s,
                                                progress=progress)
            except Exception as exc:
                partial, unreadable = (_latest_partial(directory) if directory.exists()
                                       else (None, []))
                trace = {"exception": {"type": type(exc).__name__, "message": str(exc),
                                       "traceback": traceback.format_exc()},
                         "partial_history": partial, "unreadable_checkpoints": unreadable}
                artifact = {"route": route, "exception": trace["exception"], "timed_out": False,
                            "elapsed_s": None, "supervisor_exception": True}
                try:
                    artifact["retained_failure"] = archive_new(
                        output / (directory.name + "-supervisor-exception.json.gz"), trace)
                except Exception as storage_error:
                    artifact["storage_exception"] = str(storage_error)
                progress({"event": "supervisor-exception-retained", **artifact})
            histories[route_id(route)] = trace
            artifacts.append(artifact)
            progress({"event": "route-finish", "route": route_id(route),
                      "attempted": index + 1, "exception": artifact["exception"],
                      "timed_out": artifact["timed_out"], "route_elapsed_s": artifact["elapsed_s"]})
    assessment = assess_histories(histories)
    write_new(output / "assessment.json", assessment)
    manifest = {"schema": "parameters45-manifest-v1", "attempt": args.attempt,
                "mode": "qualification" if len(selected) == 24 else "development-subset",
                "protocol": str(protocol), "provenance": provenance, "plan": prospective,
                "artifacts": artifacts, "attempted_histories": len(selected),
                "returned_histories": sum("exception" not in h for h in histories.values()),
                "timeouts": sum(a["timed_out"] for a in artifacts),
                "exceptions": sum("exception" in h for h in histories.values()),
                "physical_initial_source_stops": sum(r["physical_initial_source_stop"]
                                                      for r in assessment["routes"]),
                **raw_saved_counts(histories.values()),
                "available_physical_stages": sum(r["available_physical_stages"]
                                                  for r in assessment["routes"]),
                "evaluated_checks": assessment["evaluated_checks"],
                "check_denominator": assessment["check_denominator"],
                "failed_checks": assessment["failed_checks"], "units": assessment["units"],
                "elapsed_s": time.monotonic() - started,
                "all_passed": assessment["all_passed"] and len(selected) == 24,
                "archive_scope": "All route originals, checkpoints, worker stdout and supervisor "
                "failures remain immutable. No failed attempt is replaced or removed."}
    write_new(output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("attempted_histories", "returned_histories",
                                             "timeouts", "failed_checks", "check_denominator",
                                             "all_passed")}),
          flush=True)
    return 0 if manifest["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
