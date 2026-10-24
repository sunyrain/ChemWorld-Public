"""Fixed complete-route, common-time and event qualification, retaining failures."""

import numpy as np
from battery_half_cell_cases import RESPONSE_ATOL, F, initial_metal
from battery_half_cell_traces import history_metrics

EVENTS = {"metal_event": "half-cell:metal:depleted",
          "surface_event": "half-cell:working:saturated",
          "voltage_event": "half-cell:minimum-voltage"}


def termination(part):
    value = part.get("termination")
    return None if value in (None, "final time") else str(value).removeprefix("event: ")


def complete(trace, data):
    segments = trace.get("segments", [])
    if "exception" in trace or not segments:
        return False
    for segment in segments:
        rows = np.asarray(segment["rows"])
        if (rows.ndim != 2 or rows.shape[1] != 9 or len(rows) < 2
                or not np.isfinite(rows).all()):
            return False
    if data["expected"] in EVENTS:
        return termination(segments[-1]) == EVENTS[data["expected"]]
    return (len(segments) == len(data["waveform"])
        and all(termination(s) is None for s in segments)
        and abs(segments[-1]["end_time_s"] - sum(v[0] for v in data["waveform"])) < 1e-6)


def aligned(a, b):
    left, right = [], []
    for x, y in zip(a["segments"], b["segments"], strict=True):
        tx, ty = np.asarray(x["times_s"]), np.asarray(y["times_s"])
        for i, t in enumerate(tx):
            matching = np.flatnonzero(abs(ty - t) <= 1e-7)
            if matching.size:
                left.append(x["rows"][i])
                right.append(y["rows"][matching[0]])
    if len(left) < 2:
        raise ValueError("fewer than two exact shared scientific time points")
    return np.asarray(left), np.asarray(right)


def compare(a, b, fraction=1.):
    left, right = aligned(a, b)
    error = abs(left - right)
    tolerance = fraction * RESPONSE_ATOL + 1e-6 * abs(right)
    ratio = float(np.max(error / tolerance))
    profile = max(float(np.max(abs(np.asarray(x["profile_theta"])
                                  - np.asarray(y["profile_theta"]))))
                  for x, y in zip(a["segments"], b["segments"], strict=True))
    return {"common_time_count": len(left), "maximum_errors": np.max(error, axis=0).tolist(),
        "maximum_tolerance_ratio": ratio, "profile_error": profile,
        "passed": bool(ratio <= 1 and profile <= .003)}


def valid_refusal(route):
    return (route.get("rejected") is True and route.get("input_state_unchanged") is True
        and route.get("exception_type") in {"ValueError", "ResidualDomainError"}
        and bool(route.get("exception")))


def assess(item):
    data, routes = item["inputs"], item["routes"]
    if data["expected"] == "refusals":
        item["passed"] = all(valid_refusal(r) for r in routes.values())
        return
    checks = {key + ":complete": complete(route, data) for key, route in routes.items()}
    item["checks"] = checks
    if not all(checks.values()):
        item["passed"] = False
        return
    item["comparisons"] = comparisons = {}
    pairs = (("coarse", "native_coarse", "reference_coarse", 1.),
             ("fine", "native_fine", "reference_fine", 1.),
             ("native_grid", "native_coarse", "native_fine", .5),
             ("reference_grid", "reference_coarse", "reference_fine", 1 / 3))
    for name, left, right, fraction in pairs:
        comparisons[name] = compare(routes[left], routes[right], fraction)
        checks[name] = comparisons[name]["passed"]
    checks["refinement_not_worse"] = (comparisons["fine"]["maximum_tolerance_ratio"]
        <= comparisons["coarse"]["maximum_tolerance_ratio"] + .05)
    item["history_metrics"] = metrics = {}
    limits = {"lithium_balance_mol": 1e-10, "faradaic_error_C": 1e-5,
        "metal_mapping_error_mol": 1e-10, "CC_charge_error_C": 1e-5,
        "rest_metal_change_mol": 1e-10, "rest_charge_change_C": 1e-7,
        "stage_metal_jump_mol": 1e-10, "stage_average_jump": 1e-8, "stage_charge_jump_C": 1e-7}
    for name, route in routes.items():
        metrics[name] = history_metrics(route, data)
        checks[name + ":balance"] = all(metrics[name][k] <= limit for k, limit in limits.items())
        rows = np.asarray([row for s in route["segments"] for row in s["rows"]])
        if data.get("control_expectation") == "interior":
            checks[name + ":CV_interior"] = abs(rows[0, 1]) < .99 * data["maximum_current_A"]
        elif data.get("control_expectation") == "positive_limit":
            checks[name + ":CV_limit"] = np.max(abs(rows[:, 1] - data["maximum_current_A"])) <= 1e-7
    if data["expected"] in EVENTS:
        item["event_measurements"] = events = {}
        for name, route in routes.items():
            end = route["segments"][-1]
            row = end["rows"][-1]
            measurement = {"time_s": end["end_time_s"], "metal_mol": row[4],
                           "surface_fraction": row[2], "voltage_V": row[0]}
            if data["expected"] == "metal_event":
                expected_time = F * initial_metal(data) / data["waveform"][0][1]
                measurement["analytic_time_error_s"] = abs(end["end_time_s"] - expected_time)
                checks[name + ":event_face"] = (measurement["analytic_time_error_s"] <= .01
                    and abs(row[4]) <= 1e-10)
            elif data["expected"] == "surface_event":
                checks[name + ":event_face"] = abs(row[2] - (1 - data["surface_margin"])) <= 2e-5
            else:
                checks[name + ":event_face"] = abs(row[0] - data["minimum_voltage_V"]) <= 2e-5
            events[name] = measurement
        for name, left, right, _fraction in pairs:
            error = abs(events[left]["time_s"] - events[right]["time_s"])
            threshold = (.2 + .005 * abs(events[right]["time_s"])) * (
                .5 if name.endswith("grid") else 1.)
            checks[name + ":event_time"] = error <= threshold
    item["passed"] = bool(all(checks.values()))
