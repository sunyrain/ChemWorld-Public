"""Fail-closed structural assessment for the collector-SPM preparation runner.

This assessor intentionally does not claim an independent Green comparison.  It checks that
native/source traces retain the authored route, full source fields and finite histories.  A later
science adapter may add Green tab-voltage and current-integration comparisons without changing
this immutable trace schema.
"""

from __future__ import annotations

import json

import numpy as np
from battery_collector_spm_cases import (
    GRIDS,
    LAYOUTS,
    F,
    case_data,
    route_plan,
)


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple, np.ndarray)):
        return all(_finite(v) for v in value)
    if isinstance(value, (int, float, np.number)):
        return bool(np.isfinite(value))
    return True


def _array(value, shape=None):
    result = np.asarray(value, dtype=float)
    if shape is not None and result.shape != shape:
        raise ValueError(f"shape {result.shape} != {shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite array")
    return result


def _trace_check(route, data, grid, layout, kind):
    if route.get("implementation") != kind:
        raise ValueError("implementation identity mismatch")
    if route.get("unit") != data["id"]:
        raise ValueError("unit identity mismatch")
    if not _same(route.get("inputs"), data):
        raise ValueError("authored input identity mismatch")
    if not _same(route.get("grid"), grid) or not _same(route.get("layout"), layout):
        raise ValueError("grid/layout identity mismatch")
    if "independent_physical_tab_voltage_V" in route and route[
        "independent_physical_tab_voltage_V"
    ] is not None:
        raise ValueError("unlabelled Green physical-tab voltage")
    segments = route.get("segments")
    if not isinstance(segments, list) or len(segments) != len(data["waveform"]):
        raise ValueError("missing or shortened segment history")
    expected_start = 0.0
    for segment, (duration, _target, _unit) in zip(segments, data["waveform"], strict=True):
        times = _array(segment.get("times_s"))
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("invalid original time axis")
        if abs(times[0] - expected_start) > 1e-8:
            raise ValueError("segment time discontinuity")
        if abs(float(segment.get("end_time_s")) - times[-1]) > 1e-8:
            raise ValueError("endpoint/time mismatch")
        expected_times = expected_start + np.linspace(0.0, duration, data["samples_per_segment"])
        if len(times) != len(expected_times) or np.max(abs(times - expected_times)) > 1e-7:
            raise ValueError("missing/replaced authored samples")
        _array(segment.get("values"), (len(times), 4))
        if not _finite(segment):
            raise ValueError("nonfinite source/native segment")
        profiles = segment.get("raw_profiles")
        if set(profiles or {}) != {"negative", "positive"}:
            raise ValueError("missing negative/positive raw profiles")
        for name in ("negative", "positive"):
            profile = profiles[name]
            theta = _array(profile.get("theta"))
            if theta.shape[0] != len(times) or theta.ndim != 3:
                raise ValueError(f"{name} radial profile shape mismatch")
            for field in ("average_fraction",):
                values = _array(profile.get(field))
                if values.shape[0] != len(times):
                    raise ValueError(f"{name} {field} time mismatch")
        if kind == "native":
            amounts = segment.get("raw_amounts_mol")
            coordinates = segment.get("raw_coordinates")
            if len(amounts) != len(times) or len(coordinates) != len(times):
                raise ValueError("native inventory/QW history missing")
            if any(set(v) != {"discharged_charge_C", "delivered_work_J"} for v in coordinates):
                raise ValueError("native coordinate schema mismatch")
            if segment.get("source_boundary_voltage_V") is None:
                raise ValueError("native source-boundary diagnostic missing")
        else:
            fields = segment.get("source_fields")
            required = {
                "Negative current collector potential [V]",
                "Positive current collector potential [V]",
                "Current collector current density [A.m-2]",
                "Negative electrode current density [A.m-2]",
                "Positive electrode current density [A.m-2]",
                "Local voltage [V]",
                "source_boundary_voltage_V",
            }
            if not isinstance(fields, dict) or not required <= set(fields):
                raise ValueError("source local field schema missing")
            if len(segment.get("source_state_vectors", [])) != len(times):
                raise ValueError("source full state history missing")
            if not segment.get("source_event_values_at_endpoint"):
                raise ValueError("source endpoint event values missing")
        expected_start = times[-1]
    return True


def assess(unit):
    """Assess one authored unit and annotate route-level failures."""
    index = unit["inputs"]["id"]
    data = case_data(index)
    expected = {
        name: (kind, layout, GRIDS[grid])
        for name, kind, layout, grid in route_plan(index)
    }
    routes = unit.get("routes", {})
    checks, failures = {}, []
    if set(routes) != set(expected):
        failures.append("route denominator/identity mismatch")
    for name, (kind, layout_name, grid) in expected.items():
        route = routes.get(name)
        if route is None:
            failures.append(name + ":missing")
            continue
        try:
            if kind == "refusal":
                if route.get("implementation") != "refusal":
                    raise ValueError("refusal implementation mismatch")
                if route.get("unit") != data["id"]:
                    raise ValueError("refusal unit identity mismatch")
                if not _same(route.get("grid"), grid) or not _same(
                    route.get("layout"), LAYOUTS[layout_name]
                ):
                    raise ValueError("refusal grid/layout identity mismatch")
                refusals = route.get("refusals")
                if [r.get("name") for r in refusals or []] != list(data["refusal_cases"]):
                    raise ValueError("refusal reason/count mismatch")
                if not route.get("rejected") or not route.get("state_unchanged"):
                    raise ValueError("refusal did not reject atomically")
                if any(
                    not r.get("rejected")
                    or not r.get("state_unchanged")
                    or r.get("expected_reason") != r.get("name")
                    for r in refusals
                ):
                    raise ValueError("refusal subcase did not reject atomically")
            elif kind == "history":
                if route.get("implementation") != "history":
                    raise ValueError("history implementation mismatch")
                if route.get("unit") != data["id"]:
                    raise ValueError("history unit identity mismatch")
                if not _same(route.get("grid"), grid) or not _same(
                    route.get("layout"), LAYOUTS[layout_name]
                ):
                    raise ValueError("history grid/layout identity mismatch")
                for subkind in ("native", "source"):
                    subtrace = route.get(subkind)
                    if not isinstance(subtrace, dict):
                        raise ValueError(f"missing nested {subkind} trace")
                    _trace_check(subtrace, data, grid, LAYOUTS[layout_name], subkind)
            else:
                _trace_check(route, data, grid, LAYOUTS[layout_name], kind)
                # Independent Green/tab comparison is deliberately not claimed here.
                if route.get("independent_physical_tab_voltage_V") is not None:
                    raise ValueError("independent tab projection must remain absent")
            checks[name] = {"passed": True, "comparison": "source/native trace contract only"}
        except Exception as exc:
            checks[name] = {"passed": False, "error": str(exc)}
            failures.append(name + ":" + str(exc))
    unit["checks"] = checks
    unit["failures"] = failures
    unit["passed"] = not failures and len(routes) == len(expected)
    unit["assessment_scope"] = (
        "immutable source/native route extraction; Green physical-tab comparison deferred"
    )
    unit["faraday_C_mol"] = F
    return unit["passed"]
