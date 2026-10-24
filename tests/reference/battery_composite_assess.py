"""Fixed all-route and all-time qualification; no favorable-outcome selection."""

from copy import deepcopy

import numpy as np
from battery_composite_cases import RESPONSE_ATOL, RESPONSE_RTOL
from battery_composite_traces import history_metrics


def termination(segment):
    name = segment.get("termination")
    return None if name in (None, "final time") else str(name).removeprefix("event: ")


def complete(trace, data):
    if "exception" in trace or not trace.get("segments"):
        return False
    for segment in trace["segments"]:
        rows = np.asarray(segment["rows"])
        if (rows.ndim != 2 or rows.shape[1] != (14 if trace.get("single_material") else 19)
                or len(rows) < 2 or not np.isfinite(rows).all()):
            return False
    if data["expected"].startswith("battery:"):
        return termination(trace["segments"][-1]) == data["expected"]
    return (len(trace["segments"]) == len(data["waveform"])
        and all(termination(s) is None for s in trace["segments"])
        and abs(trace["segments"][-1]["end_time_s"] - sum(x[0] for x in data["waveform"])) < 1e-6)


def aligned(a, b):
    left, right = [], []
    for x, y in zip(a["segments"], b["segments"], strict=True):
        tx, ty = np.asarray(x["times_s"]), np.asarray(y["times_s"])
        for i, t in enumerate(tx):
            matches = np.flatnonzero(abs(ty-t) <= 1e-7)
            if matches.size:
                left.append(x["rows"][i])
                right.append(y["rows"][matches[0]])
    if len(left) < 2:
        raise ValueError("fewer than two exact common observation times")
    return np.asarray(left), np.asarray(right)


def compare(a, b, fraction=1.):
    left, right = aligned(a, b)
    indices = [*range(9), *range(14, 19)] if a.get("single_material") else list(range(19))
    tolerance = fraction * RESPONSE_ATOL[indices] + RESPONSE_RTOL[indices] * abs(right)
    error = abs(left-right)
    ratio = float(np.max(error/tolerance))
    profile = max(float(np.max(abs(np.asarray(x["profiles_theta"])
                                  - np.asarray(y["profiles_theta"]))))
                  for x, y in zip(a["segments"], b["segments"], strict=True))
    return {"common_time_count": len(left), "maximum_errors": np.max(error, axis=0).tolist(),
        "maximum_tolerance_ratio": ratio, "profile_error": profile,
        "passed": bool(ratio <= 1 and profile <= .003)}


def collapse_identical(trace):
    result = deepcopy(trace)
    result["single_material"] = True
    for segment in result["segments"]:
        old = np.asarray(segment["rows"])
        negative = np.column_stack((old[:, 4]+old[:, 9], (2*old[:, 5]+old[:, 10])/3,
            (2*old[:, 6]+old[:, 11])/3, old[:, 7]+old[:, 12], (2*old[:, 8]+old[:, 13])/3))
        segment["rows"] = np.column_stack((old[:, :4], negative, old[:, 14:])).tolist()
        profiles = np.asarray(segment["profiles_theta"])
        segment["profiles_theta"] = np.vstack(((2*profiles[0]+profiles[1])/3, profiles[2])).tolist()
    return result


def valid_refusal(route):
    return (route.get("rejected") is True and route.get("input_state_unchanged") is True
        and route.get("exception_type") in {"ValueError", "ResidualDomainError"}
        and bool(route.get("exception")))


def assess(item):
    data, routes = item["inputs"], item["routes"]
    if data["expected"] == "refusals":
        item["passed"] = all(valid_refusal(value) for value in routes.values())
        return
    checks = {key+":complete": complete(route, data) for key, route in routes.items()}
    item["checks"] = checks
    if not all(checks.values()):
        item["passed"] = False
        return
    item["comparisons"] = comparisons = {}
    pairs = (("coarse", "native_coarse", "reference_coarse", 1.),
             ("fine", "native_fine", "reference_fine", 1.),
             ("native_grid", "native_coarse", "native_fine", .5),
             ("reference_grid", "reference_coarse", "reference_fine", 1/3))
    for name, left, right, fraction in pairs:
        comparisons[name] = compare(routes[left], routes[right], fraction)
        checks[name] = comparisons[name]["passed"]
    checks["refinement_not_worse"] = (comparisons["fine"]["maximum_tolerance_ratio"]
        <= comparisons["coarse"]["maximum_tolerance_ratio"] + .05)
    item["history_metrics"] = metrics = {}
    limits = {"lithium_balance_mol": 1e-9, "faradaic_error_C": 1e-4,
        "phase_current_balance_A": 1e-7, "CC_charge_error_C": 1e-4, "rest_charge_change_C": 1e-7,
        "stage_inventory_jump_mol": 1e-9, "stage_average_jump": 1e-8, "stage_charge_jump_C": 1e-7}
    for name, route in routes.items():
        metrics[name] = history_metrics(route, data)
        checks[name+":balance"] = all(metrics[name][k] <= limit for k, limit in limits.items())
        rows = np.asarray([row for segment in route["segments"] for row in segment["rows"]])
        if data.get("same_material_limit") and not route.get("single_material"):
            checks[name+":identical_phases"] = bool(
                np.max(abs(rows[:, 5:7]-rows[:, 10:12])) <= 1e-7
                and np.max(abs(rows[:, 4]-2*rows[:, 9])) <= 1e-7)
        if data.get("internal_exchange"):
            checks[name+":internal_exchange"] = bool(min(abs(rows[0, 4]), abs(rows[0, 9])) >= 1e-4
                and rows[0, 4]*rows[0, 9] < 0
                and max(abs(rows[-1, [7, 12]]-rows[0, [7, 12]])) >= 1e-7)
        if data["name"] == "distinct-material-discharge":
            checks[name+":nonfixed_sharing"] = abs(rows[0, 4]/rows[0, 1] - 2/3) >= 1e-3
        if data.get("control_expectation") == "interior":
            checks[name+":CV_interior"] = abs(rows[0, 1]) < .99*data["maximum_current_A"]
        elif data.get("control_expectation") == "positive_limit":
            checks[name+":CV_limit"] = np.max(abs(rows[:, 1]-data["maximum_current_A"])) <= 1e-7
    if data.get("same_material_limit"):
        for label in ("coarse", "fine"):
            for implementation in ("native", "reference"):
                name = implementation+"_single_limit_"+label
                comparisons[name] = compare(collapse_identical(routes[implementation+"_"+label]),
                                            routes["single_"+label])
                checks[name] = comparisons[name]["passed"]
        comparisons["single_grid"] = compare(routes["single_coarse"], routes["single_fine"], 1/3)
        checks["single_grid"] = comparisons["single_grid"]["passed"]
    if data["expected"].startswith("battery:"):
        events = {}
        for name, route in routes.items():
            final = route["segments"][-1]
            row = final["rows"][-1]
            events[name] = {"time_s": final["end_time_s"], "voltage_V": row[0],
                            "surface_fractions": row[5::5]}
            if data["expected"] == "battery:minimum-voltage":
                checks[name+":event_face"] = abs(row[0]-data["minimum_voltage_V"]) <= 2e-5
            else:
                checks[name+":event_face"] = abs(row[5]-data["surface_margin"]) <= 2e-5
                checks[name+":individual_cutoff"] = row[10] > .02
        item["event_measurements"] = events
        for name, left, right, _fraction in pairs:
            error = abs(events[left]["time_s"]-events[right]["time_s"])
            limit = (.2+.005*abs(events[right]["time_s"])) * (.5 if name.endswith("grid") else 1.)
            checks[name+":event_time"] = error <= limit
    item["passed"] = bool(all(checks.values()))
