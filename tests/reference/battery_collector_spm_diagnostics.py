"""Read-only post hoc diagnostics for archived collector SPM attempts.

This module does not run a solver or the qualification assessor.  It adds no
acceptance threshold and never replaces the original structural result.  Run as
``uv run --no-sync python tests/reference/battery_collector_spm_diagnostics.py
INPUT.json[.gz] OUTPUT.json`` only after the caller has documented the analysis.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from copy import deepcopy
from itertools import pairwise
from pathlib import Path

import numpy as np
from battery_collector_spm_cases import GRIDS, LAYOUTS, NAMES, case_data, route_plan


def _issue(issues, path, reason):
    issues.append({"path": path, "reason": reason})


def _nonfinite(value, path, issues):
    if isinstance(value, dict):
        if "__nonfinite__" in value:
            _issue(issues, path, "archived nonfinite: " + str(value["__nonfinite__"]))
        for key, child in value.items():
            _nonfinite(child, f"{path}.{key}", issues)
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _nonfinite(child, f"{path}[{index}]", issues)
    elif isinstance(value, (float, int)) and not np.isfinite(value):
        _issue(issues, path, "nonfinite number: " + str(value))


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite__": str(value)}
    return value


def _array(value, path, issues, shape=None):
    try:
        array = np.asarray(value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise ValueError("missing/nonfinite numeric data")
        if shape is not None and array.shape != shape:
            raise ValueError(f"shape {array.shape}, expected {shape}")
        return array
    except (TypeError, ValueError) as error:
        _issue(issues, path, str(error))
        return None


def _stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {"available": False, "reason": "empty data"}
    if not np.all(np.isfinite(values)):
        return {"available": False, "reason": "nonfinite derived data"}
    scale = float(np.max(abs(values)))
    rms = scale * float(np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0
    return {
        "available": True,
        "count": int(values.size),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "max_abs": float(np.max(abs(values))),
        "rms": rms,
        "first": float(values.flat[0]),
        "last": float(values.flat[-1]),
    }


def _required(mapping, keys, path, issues):
    if not isinstance(mapping, dict):
        _issue(issues, path, "missing or non-object record")
        return {}
    for key in keys:
        if key not in mapping:
            _issue(issues, f"{path}.{key}", "missing required field")
    return mapping


def recover_trace(route, kind):
    """Recover successful subtraces even when their paired implementation failed."""
    if isinstance(route.get(kind), dict):
        return route[kind], "returned trace"
    partial = route.get("partial_history")
    if not isinstance(partial, dict):
        return None, "not recorded"
    completed = partial.get("completed_traces", {})
    if isinstance(completed, dict) and isinstance(completed.get(kind), dict):
        return completed[kind], "completed trace retained in failed pair"
    if partial.get("stage") == kind:
        context = partial.get("failed_implementation_context")
        if isinstance(context, dict):
            return {
                "segments": context.get("completed_segments", []),
                "initialization": context.get("consistent_initialization"),
                "pending_solution": context.get("pending_solution"),
                "failure_context": context,
            }, "failed implementation partial history"
    return None, "not reached or not recorded"


# Compatibility for callers that inspected the initial private helper during development.
_recover = recover_trace


def _profile(profile, kind, name, nt, grid, data, path, issues):
    profile = _required(
        profile,
        (
            "shell_amounts_mol",
            "reaction_current_A",
            "average_fraction",
            "surface_fraction",
            "z_nodes_m",
            "r_edges_m",
        ),
        path,
        issues,
    )
    nz, nr = grid["z"], grid["r"]
    shell_shape = (nt, nz, nr) if kind == "native" else (nt, nr, nz)
    shells = _array(
        profile.get("shell_amounts_mol"), path + ".shell_amounts_mol", issues, shell_shape
    )
    if shells is not None and kind == "source":
        shells = shells.transpose(0, 2, 1)
    z = _array(profile.get("z_nodes_m"), path + ".z_nodes_m", issues, (nz,))
    radial = _array(profile.get("r_edges_m"), path + ".r_edges_m", issues, (nr + 1,))
    if radial is not None and np.any(np.diff(radial) <= 0):
        _issue(issues, path + ".r_edges_m", "non-increasing radial mesh")
    if kind == "native":
        edges = np.linspace(0.0, data["height_m"], nz + 1)
        edge_origin = "uniform native geometry from archived height and grid"
    else:
        edges = _array(profile.get("z_edges_m"), path + ".z_edges_m", issues, (nz + 1,))
        edge_origin = "original source z_edges_m"
    if edges is not None and np.any(np.diff(edges) <= 0):
        _issue(issues, path + ".z_edges_m", "non-increasing spatial mesh")
        edges = None
    result = {"shells": shells, "z": z, "edges": edges, "edge_origin": edge_origin}
    for key in ("reaction_current_A", "average_fraction", "surface_fraction"):
        result[key] = _array(profile.get(key), f"{path}.{key}", issues, (nt, nz))
    if shells is not None:
        result["cell_Li_mol"] = shells.sum(axis=2)
        result["Li_mol"] = shells.sum(axis=(1, 2))
    return result


def _segment(segment, kind, data, grid, index, path, issues):
    segment = _required(
        segment,
        (
            "times_s",
            "values",
            "end_time_s",
            "outcome",
            "termination",
            "raw_profiles",
            "source_boundary_voltage_V",
        ),
        path,
        issues,
    )
    times = _array(segment.get("times_s"), path + ".times_s", issues)
    if times is None or times.ndim != 1 or len(times) == 0:
        _issue(issues, path + ".times_s", "missing/non-vector/empty time axis")
        return {"available": False}, None
    if np.any(np.diff(times) <= 0):
        _issue(issues, path + ".times_s", "non-increasing time axis")
        return {"available": False}, None
    nt = len(times)
    waveform = data["waveform"]
    expected = waveform[index] if index < len(waveform) else None
    start = sum(row[0] for row in waveform[:index])
    coverage = {
        "samples": nt,
        "t0_only": nt == 1,
        "recorded_start_s": float(times[0]),
        "recorded_end_s": float(times[-1]),
        "outcome": segment.get("outcome"),
        "termination": segment.get("termination"),
        "expected_start_s": start,
    }
    if expected:
        duration, target, mode = expected
        coverage.update(
            expected_end_s=start + duration,
            expected_samples=data["samples_per_segment"],
            start_offset_s=float(times[0] - start),
            end_offset_s=float(times[-1] - start - duration),
        )
        if nt == data["samples_per_segment"]:
            coverage["authored_sample_time_difference_s"] = _stats(
                times - start - np.linspace(0, duration, nt)
            )
    values = _array(segment.get("values"), path + ".values", issues, (nt, 4))
    boundary = _array(
        segment.get("source_boundary_voltage_V"), path + ".source_boundary_voltage_V", issues, (nt,)
    )
    if kind == "native":
        required = ("raw_amounts_mol", "raw_coordinates", "raw_algebraic_values")
    else:
        required = (
            "source_state_vectors",
            "source_initial_rhs",
            "source_initial_algebraic_residual",
            "source_fields",
        )
    _required(segment, required, path, issues)
    profiles = segment.get("raw_profiles", {})
    profiles = _required(profiles, ("negative", "positive"), path + ".raw_profiles", issues)
    normalized = {
        name: _profile(
            profiles.get(name), kind, name, nt, grid, data, path + ".raw_profiles." + name, issues
        )
        for name in ("negative", "positive")
    }
    if kind == "source":
        fields = _required(
            segment.get("source_fields"),
            (
                "Negative current collector potential [V]",
                "Positive current collector potential [V]",
                "Current collector current density [A.m-2]",
                "Negative electrode current density [A.m-2]",
                "Positive electrode current density [A.m-2]",
                "Local voltage [V]",
                "source_boundary_voltage_V",
                "through_current_A",
            ),
            path + ".source_fields",
            issues,
        )
        states = _array(segment.get("source_state_vectors"), path + ".source_state_vectors", issues)
        if states is not None and (states.ndim != 2 or states.shape[0] != nt):
            _issue(issues, path + ".source_state_vectors", "full state time/shape mismatch")
    else:
        fields = {}
        caches = segment.get("raw_algebraic_values")
        if not isinstance(caches, list) or len(caches) != nt or any(row != {} for row in caches):
            _issue(issues, path + ".raw_algebraic_values", "missing/noncanonical persistent cache")
    for name, profile in normalized.items():
        for field, source_name in (
            ("collector_potential_V", f"{name.title()} current collector potential [V]"),
            ("local_voltage_V", "Local voltage [V]"),
        ):
            original = profiles.get(name)
            value = (
                original.get(field)
                if kind == "native" and isinstance(original, dict)
                else (fields.get(source_name))
            )
            profile[field] = _array(value, f"{path}.{name}.{field}", issues, (nt, grid["z"]))
    metrics = {}
    faraday = data.get("_faraday_C_mol")
    for name, sign in (("negative", 1), ("positive", -1)):
        profile = normalized[name]
        if "Li_mol" in profile:
            metrics[name + "_Li_mol"] = _stats(profile["Li_mol"])
            if values is not None and faraday is not None:
                metrics[name + "_Faraday_residual_C"] = _stats(
                    faraday * (profile["Li_mol"] - profile["Li_mol"][0])
                    + sign * (values[:, 2] - values[0, 2])
                )
        q = profile["reaction_current_A"]
        if q is not None and values is not None:
            metrics[name + "_sum_q_minus_signed_I_A"] = _stats(q.sum(axis=1) - sign * values[:, 1])
    if all("Li_mol" in normalized[name] for name in normalized):
        total = sum(profile["Li_mol"] for profile in normalized.values())
        metrics["total_Li_drift_mol"] = _stats(total - total[0])
    qn, qp = (normalized[name]["reaction_current_A"] for name in normalized)
    if qn is not None and qp is not None:
        metrics["local_negative_plus_positive_q_A"] = _stats(qn + qp)
    if values is not None:
        for column, label in enumerate(
            (
                "native_physical_terminal_voltage_V"
                if kind == "native"
                else "source_raw_boundary_voltage_V",
                "current_A",
                "charge_C",
                "work_J",
            )
        ):
            metrics[label] = _stats(values[:, column])
        if expected and mode == "V":
            metrics["CV_delta_W_minus_target_delta_Q_J"] = _stats(
                values[:, 3] - values[0, 3] - target * (values[:, 2] - values[0, 2])
            )
            metrics["CV_recorded_voltage_minus_target_V"] = _stats(values[:, 0] - target)
            metrics["CV_note"] = (
                "Target-voltage identity is descriptive only: finite current limiting and "
                "different voltage definitions can produce a nonzero value. No new pass rule."
            )
    if kind == "native":
        for key, column in (("discharged_charge_C", 2), ("delivered_work_J", 3)):
            records = segment.get("raw_coordinates")
            if not isinstance(records, list) or len(records) != nt:
                _issue(issues, path + ".raw_coordinates", "missing/truncated QW history")
                break
            raw = _array(
                [row.get(key) if isinstance(row, dict) else None for row in records],
                path + ".raw_coordinates." + key,
                issues,
                (nt,),
            )
            if raw is not None and values is not None:
                metrics["raw_" + key + "_minus_values"] = _stats(raw - values[:, column])
        amounts = segment.get("raw_amounts_mol")
        if not isinstance(amounts, list) or len(amounts) != nt:
            _issue(issues, path + ".raw_amounts_mol", "missing/truncated inventory history")
        else:
            for name in normalized:
                keys = profiles.get(name, {}).get("inventory_keys")
                try:
                    flat = [key for cell in keys for key in cell]
                    if len(flat) != grid["z"] * grid["r"] or len(set(flat)) != len(flat):
                        raise ValueError("wrong/duplicate inventory keys")
                    raw = np.asarray([[row[key] for key in flat] for row in amounts], dtype=float)
                    if not np.all(np.isfinite(raw)):
                        raise ValueError("nonfinite raw inventory")
                    shells = normalized[name]["shells"]
                    if shells is not None:
                        metrics[name + "_raw_inventory_minus_shell_mol"] = _stats(
                            raw - shells.reshape(nt, -1)
                        )
                except (TypeError, KeyError, ValueError) as error:
                    _issue(issues, path + ".raw_amounts_mol." + name, str(error))
    return {
        "available": True,
        "coverage": coverage,
        "metrics": metrics,
    }, {"times": times, "values": values, "boundary": boundary, "profiles": normalized}


def _trace(trace, kind, origin, data, grid, path, issues):
    if trace is None:
        _issue(issues, path, origin)
        return {"coverage": "missing", "origin": origin, "segments": []}, []
    segments = trace.get("segments")
    if not isinstance(segments, list):
        _issue(issues, path + ".segments", "missing/non-list segments")
        segments = []
    records, arrays = [], []
    for key, expected in (("implementation", kind), ("unit", data["id"]), ("grid", grid)):
        if key in trace and trace[key] != expected:
            _issue(issues, path + "." + key, "trace metadata mismatch")
    if "inputs" in trace and json.dumps(trace["inputs"], sort_keys=True) != json.dumps(
        data,
        sort_keys=True,
    ):
        _issue(issues, path + ".inputs", "trace does not match fixed authored inputs")
    context = deepcopy(data)
    faraday = _array(trace.get("faraday_C_mol"), path + ".faraday_C_mol", issues, ())
    if faraday is not None and faraday > 0:
        context["_faraday_C_mol"] = float(faraday)
    elif faraday is not None:
        _issue(issues, path + ".faraday_C_mol", "nonpositive Faraday constant")
    for index, segment in enumerate(segments):
        try:
            record, array = _segment(
                segment, kind, context, grid, index, f"{path}.segments[{index}]", issues
            )
        except Exception as error:  # Keep all other segments and the failure visible.
            _issue(issues, f"{path}.segments[{index}]", f"{type(error).__name__}: {error}")
            record, array = {"available": False, "diagnostic_exception": str(error)}, None
        records.append(record)
        arrays.append(array)
    expected_count = len(data["waveform"])
    if len(segments) != expected_count:
        _issue(issues, path + ".segments", f"{len(segments)} recorded, {expected_count} expected")
    coverage = "partial"
    initialization = trace.get("initialization")
    if not segments:
        coverage = "initialization_only" if isinstance(initialization, dict) else "no_samples"
    elif all(r.get("coverage", {}).get("t0_only", False) for r in records):
        coverage = "t0_only"
    elif len(records) == expected_count and all(
        r.get("coverage", {}).get("samples") == data["samples_per_segment"]
        and r["coverage"]["outcome"] == "success"
        for r in records
    ):
        coverage = "complete_sample_records"
    result = {"coverage": coverage, "origin": origin, "segments": records}
    # A failed initializer's raw observations are retained verbatim, including
    # per-observation read errors.  It is never counted as a solved history.
    if initialization is not None:
        result["initialization_only_observations"] = deepcopy(initialization)
    if trace.get("pending_solution") is not None:
        result["unextracted_pending_solution"] = {
            key: deepcopy(value)
            for key, value in trace["pending_solution"].items()
            if key != "source_state_vectors"
        }
        _issue(issues, path + ".pending_solution", "raw solution exists but profiles not extracted")
    continuity = []
    for previous, current in pairwise(arrays):
        if previous is None or current is None:
            continuity.append({"available": False})
            continue
        entry = {"time_gap_s": float(current["times"][0] - previous["times"][-1])}
        if previous["values"] is not None and current["values"] is not None:
            entry["charge_jump_C"] = float(current["values"][0, 2] - previous["values"][-1, 2])
            entry["work_jump_J"] = float(current["values"][0, 3] - previous["values"][-1, 3])
        for name in ("negative", "positive"):
            left, right = previous["profiles"][name], current["profiles"][name]
            if left["shells"] is not None and right["shells"] is not None:
                entry[name + "_shell_inventory_jump_mol"] = _stats(
                    right["shells"][0] - left["shells"][-1]
                )
        continuity.append(entry)
    result["segment_continuity"] = continuity
    return result, arrays


def common_interval_sums(left, left_edges, right, right_edges):
    """Sum *whole cells* between exact common edges; never interpolate q.

    The common domain endpoints must coincide.  Missing inner common edges
    simply yield wider intervals, potentially only the full-domain current.
    """
    left_edges, right_edges = np.asarray(left_edges), np.asarray(right_edges)
    if left_edges[0] != right_edges[0] or left_edges[-1] != right_edges[-1]:
        raise ValueError("spatial domain endpoints differ")
    edges = np.intersect1d(left_edges, right_edges)
    if len(edges) < 2:
        raise ValueError("no complete common spatial interval")
    a, b = [], []
    for lo, hi in pairwise(edges):
        il, ir = np.searchsorted(left_edges, (lo, hi))
        jl, jr = np.searchsorted(right_edges, (lo, hi))
        a.append(left[:, il:ir].sum(axis=1))
        b.append(right[:, jl:jr].sum(axis=1))
    return edges, np.column_stack(a), np.column_stack(b)


def _compare_segments(left, right, label):
    if not left and not right:
        return [{"available": False, "comparison": label, "reason": "both histories missing"}]
    comparisons = []
    for index in range(max(len(left), len(right))):
        record = {"segment": index, "comparison": label}
        if index >= len(left) or index >= len(right) or left[index] is None or right[index] is None:
            record.update(available=False, reason="missing comparable segment")
            comparisons.append(record)
            continue
        a, b = left[index], right[index]
        time, ia, ib = np.intersect1d(a["times"], b["times"], return_indices=True)
        record.update(
            available=bool(len(time)),
            exact_common_samples=len(time),
            left_samples=len(a["times"]),
            right_samples=len(b["times"]),
            all_samples_exactly_matched=(len(time) == len(a["times"]) == len(b["times"])),
            left_matched_fraction=len(time) / len(a["times"]),
            right_matched_fraction=len(time) / len(b["times"]),
            time_alignment="exact shared times only; no temporal interpolation",
        )
        if not len(time):
            record["reason"] = "no exact common observation time"
            comparisons.append(record)
            continue
        record["time_range_s"] = [float(time[0]), float(time[-1])]
        metrics = record["differences_left_minus_right"] = {}
        if a["values"] is not None and b["values"] is not None:
            names = (
                "voltage_definition_gap_V"
                if label == "native_source_same_grid"
                else "same_definition_voltage_V",
                "current_A",
                "charge_C",
                "work_J",
            )
            for column, name in enumerate(names):
                metrics[name] = _stats(a["values"][ia, column] - b["values"][ib, column])
        if a["boundary"] is not None and b["boundary"] is not None:
            metrics["source_boundary_diagnostic_V"] = _stats(a["boundary"][ia] - b["boundary"][ib])
        record["spatial_aggregation"] = {}
        for name in ("negative", "positive"):
            pa, pb = a["profiles"][name], b["profiles"][name]
            if pa["edges"] is None or pb["edges"] is None:
                record["spatial_aggregation"][name] = {"available": False, "reason": "missing mesh"}
                continue
            for field in ("reaction_current_A", "cell_Li_mol"):
                va, vb = pa.get(field), pb.get(field)
                if va is None or vb is None:
                    metrics[name + "_" + field] = {"available": False, "reason": "missing field"}
                    continue
                try:
                    edges, aa, bb = common_interval_sums(va[ia], pa["edges"], vb[ib], pb["edges"])
                    record["spatial_aggregation"][name] = {
                        "available": True,
                        "common_edges_m": edges.tolist(),
                        "interval_count": len(edges) - 1,
                        "left_original_cell_count": len(pa["edges"]) - 1,
                        "right_original_cell_count": len(pb["edges"]) - 1,
                        "full_domain_only": len(edges) == 2,
                        "definition": "sum of whole original cells over exact common intervals",
                    }
                    metrics[name + "_" + field] = _stats(aa - bb)
                except ValueError as error:
                    record["spatial_aggregation"][name] = {"available": False, "reason": str(error)}
            z_a, z_b = pa["z"], pb["z"]
            same_z = z_a is not None and z_b is not None and np.array_equal(z_a, z_b)
            record.setdefault("nodal_comparison_coverage", {})[name] = {
                "all_nodes_exactly_matched": same_z,
                "exact_common_nodes": (
                    len(np.intersect1d(z_a, z_b)) if z_a is not None and z_b is not None else 0
                ),
                "note": "Nodal quantities compared only when all recorded z nodes match exactly",
            }
            if same_z:
                for field in (
                    "average_fraction",
                    "surface_fraction",
                    "collector_potential_V",
                    "local_voltage_V",
                ):
                    if pa[field] is not None and pb[field] is not None:
                        metrics[name + "_same_z_" + field] = _stats(pa[field][ia] - pb[field][ib])
        comparisons.append(record)
    return comparisons


def _compare(left, right, label):
    """Isolate malformed comparisons so subsequent segments/routes remain visible."""
    result = []
    for index in range(max(len(left), len(right), 1)):
        try:
            record = _compare_segments(left[index : index + 1], right[index : index + 1], label)[0]
        except Exception as error:
            record = {
                "available": False,
                "comparison": label,
                "diagnostic_exception": f"{type(error).__name__}: {error}",
            }
        record["segment"] = index
        result.append(record)
    return result


def diagnose(report):
    """Return diagnostics without mutating input or reclassifying qualification."""
    issues = []
    _nonfinite(report, "$", issues)
    units = report.get("units", [])
    if not isinstance(units, list):
        _issue(issues, "$.units", "missing/non-list units")
        units = []
    by_id, duplicates = {}, []
    for index, unit in enumerate(units):
        if not isinstance(unit, dict) or unit.get("id") not in NAMES:
            _issue(issues, f"$.units[{index}]", "unknown/malformed unit retained")
        elif unit["id"] in by_id:
            _issue(
                issues, f"$.units[{index}]", "duplicate unit retained; first used for diagnostics"
            )
            duplicates.append(deepcopy(unit))
        else:
            by_id[unit["id"]] = (index, unit)
    output = {
        "analysis": "offline descriptive diagnostics; no new qualification thresholds",
        "voltage_conventions": {
            "native": "values[:,0] is physical terminal/tab voltage including series drop",
            "source": "values[:,0] is original PyBaMM finite-volume boundary Voltage [V]",
            "comparison": "direct voltage gap is not an error between identical observables",
            "boundary": "native source-boundary diagnostic versus original source boundary voltage",
        },
        "mesh_note": "z/r are jointly refined; differences do not establish convergence order",
        "coverage_note": (
            "complete_sample_records means counts/outcomes present; time offsets are reported "
            "separately, not accepted using a new tolerance"
        ),
        "expected_units": len(NAMES),
        "expected_routes": 48,
        "original_summary": deepcopy({k: v for k, v in report.items() if k != "units"}),
        "original_unit_outcomes": [
            {k: deepcopy(v) for k, v in u.items() if k != "routes"}
            if isinstance(u, dict)
            else deepcopy(u)
            for u in units
        ],
        "unexpected_units": [
            deepcopy(u) for u in units if not isinstance(u, dict) or u.get("id") not in NAMES
        ],
        "duplicate_units": duplicates,
        "units": [],
        "issues": issues,
    }
    for name in NAMES:
        index, unit = by_id.get(name, (None, {}))
        path = f"$.units[{index}]" if index is not None else "$.missing." + name
        if index is None:
            _issue(issues, path, "missing unit")
        data = case_data(name)
        expected_names = {r[0] for r in route_plan(name)}
        raw_routes = unit.get("routes", {})
        if not isinstance(raw_routes, dict):
            _issue(issues, path + ".routes", "missing/non-object routes")
            raw_routes = {}
        item = {"id": name, "routes": {}, "coarse_fine": {}, "unexpected_routes": {}}
        arrays = {}
        for route_name, kind, layout, grid_name in route_plan(name):
            rp = path + ".routes." + route_name
            raw = raw_routes.get(route_name)
            local = []
            entry = {"kind": kind, "issues": local}
            item["routes"][route_name] = entry
            if not isinstance(raw, dict):
                _issue(local, rp, "missing/non-object route")
                entry["coverage"] = "missing"
                continue
            entry["exception"] = deepcopy(
                {
                    key: raw[key]
                    for key in ("exception", "exception_type", "traceback")
                    if key in raw
                }
            )
            _nonfinite(raw, rp, local)
            for key, expected in (
                ("unit", name),
                ("grid", GRIDS[grid_name]),
                ("layout", LAYOUTS[layout]),
            ):
                if key in raw and raw[key] != expected:
                    _issue(local, rp + "." + key, "metadata mismatch")
            if kind == "refusal":
                entry["coverage"] = "refusal_not_a_history"
                entry["record"] = deepcopy(raw)
                continue
            arrays[route_name] = {}
            for implementation in ("native", "source"):
                trace, origin = recover_trace(raw, implementation)
                try:
                    result, norm = _trace(
                        trace,
                        implementation,
                        origin,
                        data,
                        GRIDS[grid_name],
                        rp + "." + implementation,
                        local,
                    )
                except Exception as error:
                    _issue(
                        local,
                        rp + "." + implementation,
                        f"diagnostic exception {type(error).__name__}: {error}",
                    )
                    result, norm = {"coverage": "invalid", "diagnostic_exception": str(error)}, []
                entry[implementation] = result
                arrays[route_name][implementation] = norm
            entry["same_grid_comparison"] = _compare(
                arrays[route_name]["native"],
                arrays[route_name]["source"],
                "native_source_same_grid",
            )
        for extra in set(raw_routes) - expected_names:
            _issue(issues, path + ".routes." + extra, "unexpected route retained")
            item["unexpected_routes"][extra] = deepcopy(raw_routes[extra])
        if data["expected"] != "refusals":
            for layout in LAYOUTS:
                for implementation in ("native", "source"):
                    item["coarse_fine"][layout + "_" + implementation] = _compare(
                        arrays.get(f"history_{layout}_coarse", {}).get(implementation, []),
                        arrays.get(f"history_{layout}_fine", {}).get(implementation, []),
                        implementation + "_coarse_fine_joint_z_r",
                    )
        output["units"].append(item)
    output["reported_routes"] = sum(len(u["routes"]) for u in output["units"])
    coverage = {"native": {}, "source": {}, "refusals": 0, "missing_routes": 0}
    for unit in output["units"]:
        for route in unit["routes"].values():
            if route.get("coverage") == "refusal_not_a_history":
                coverage["refusals"] += 1
            if route.get("coverage") == "missing":
                coverage["missing_routes"] += 1
            if route["kind"] == "history":
                for implementation in ("native", "source"):
                    status = route.get(implementation, {}).get("coverage", "missing")
                    counts = coverage[implementation]
                    counts[status] = counts.get(status, 0) + 1
    output["coverage_counts"] = coverage
    output["issue_count"] = len(issues) + sum(
        len(r["issues"]) for u in output["units"] for r in u["routes"].values()
    )
    return _safe(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        raise FileExistsError("diagnostics require a new output path; input cannot be overwritten")
    raw = args.input.read_bytes()
    payload = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    report = json.loads(payload)
    if not isinstance(report, dict):
        raise ValueError("archived report must be a JSON object")
    result = diagnose(report)
    result["input_file"] = {
        "path": str(args.input),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
    }
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")
    return 0  # A diagnostic report is not a new scientific qualification decision.


if __name__ == "__main__":
    raise SystemExit(main())
