"""The fixed x-full qualification rules; every failed route remains visible."""

import numpy as np
from battery_spatial_thermal_cases import initial_temperatures, matrix_temperature
from battery_spatial_thermal_traces import history_metrics

LIMITS = {
    "voltage_V": (0.002, 1e-5),
    "current_A": (0.005, 0.005),
    "charge_C": (0.02, 1e-5),
    "work_J": (0.1, 1e-5),
    "negative_surface": (0.003, 0),
    "positive_surface": (0.003, 0),
    "negative_average": (0.003, 0),
    "positive_average": (0.003, 0),
    "negative_amount_mol": (2e-7, 1e-6),
    "positive_amount_mol": (2e-7, 1e-6),
    **dict.fromkeys(
        (
            "negative_collector_K",
            "negative_mean_K",
            "separator_mean_K",
            "positive_mean_K",
            "positive_collector_K",
            "C_mean_K",
            "volume_mean_K",
        ),
        (0.05, 0),
    ),
    "maximum_K": (0.15, 0),
    "minimum_K": (0.15, 0),
    "loss_W": (0.002, 0.01),
    "generated_W": (0.002, 0.01),
    "irreversible_W": (0.002, 0.01),
    "loss_J": (0.05, 0.005),
    "sensible_J": (0.05, 0.005),
}


def termination(segment):
    value = segment.get("termination")
    return None if value in (None, "final time") else str(value).removeprefix("event: ")


def complete(trace, data):
    if "exception" in trace or not trace.get("segments"):
        return False
    for segment in trace["segments"]:
        times = np.asarray(segment["times_s"])
        if times.size < 2 or not np.isfinite(times).all():
            return False
        for name in LIMITS:
            value = np.asarray(segment["values"][name])
            if value.shape != times.shape or not np.isfinite(value).all():
                return False
        if not np.isfinite(segment["temperatures_K"]).all():
            return False
    final = trace["segments"][-1]
    if data["expected"] == "maximum-temperature":
        return "maximum-temperature" in (termination(final) or "")
    return (
        len(trace["segments"]) == len(data["waveform"])
        and all(termination(s) is None for s in trace["segments"])
        and abs(final["end_time_s"] - sum(x[0] for x in data["waveform"])) < 1e-6
    )


def aligned(a, b):
    left, right, lp, rp = {key: [] for key in LIMITS}, {key: [] for key in LIMITS}, [], []
    for x, y in zip(a["segments"], b["segments"], strict=True):
        tx, ty = np.asarray(x["times_s"]), np.asarray(y["times_s"])
        for i, t in enumerate(tx):
            matches = np.flatnonzero(abs(ty - t) <= 1e-7)
            if matches.size:
                j = matches[0]
                for key in LIMITS:
                    left[key].append(x["values"][key][i])
                    right[key].append(y["values"][key][j])
                lp.append(x["sampled_profiles_K"][i])
                rp.append(y["sampled_profiles_K"][j])
    if len(lp) < 2:
        raise ValueError("fewer than two common observation times; no extrapolation allowed")
    return (
        {key: np.asarray(value) for key, value in left.items()},
        {key: np.asarray(value) for key, value in right.items()},
        np.asarray(lp),
        np.asarray(rp),
    )


def compare(a, b):
    left, right, lp, rp = aligned(a, b)
    result = {}
    for key, (absolute, relative) in LIMITS.items():
        error = abs(left[key] - right[key])
        result[key] = {
            "maximum_error": float(np.max(error)),
            "maximum_tolerance_ratio": float(
                np.max(error / (absolute + relative * abs(right[key])))
            ),
        }
    profile = float(np.max(abs(lp - rp)))
    return {
        "common_time_count": len(lp),
        "responses": result,
        "profile_maximum_error_K": profile,
        "passed": bool(
            profile <= 0.15 and all(v["maximum_tolerance_ratio"] <= 1 for v in result.values())
        ),
    }


def valid_refusal(route):
    return (
        route.get("rejected") is True
        and route.get("input_state_unchanged") is True
        and route.get("exception_type") in {"ValueError", "ResidualDomainError"}
        and bool(route.get("exception"))
    )


def assess(item):
    data, routes = item["inputs"], item["routes"]
    if data["expected"] == "refusals":
        item["checks"] = {name: valid_refusal(route) for name, route in routes.items()}
        item["passed"] = len(routes) == 8 and all(item["checks"].values())
        return
    item["checks"] = checks = {
        name + ":complete": complete(route, data) for name, route in routes.items()
    }
    item["comparisons"] = comparisons = {}
    pairs = (
        ("fine", "native_fine", "reference_fine"),
        ("native_grid", "native_coarse", "native_fine"),
        ("reference_grid", "reference_coarse", "reference_fine"),
    )
    for label, a, b in pairs:
        try:
            comparisons[label] = compare(routes[a], routes[b])
            checks[label] = comparisons[label]["passed"]
        except Exception as exc:
            comparisons[label] = {"exception_type": type(exc).__name__, "exception": str(exc)}
            checks[label] = False
    item["history_metrics"] = metrics = {}
    for name, route in routes.items():
        if not route.get("segments"):
            checks[name + ":balance"] = False
            continue
        metric = metrics[name] = history_metrics(route, data)
        checks[name + ":balance"] = (
            metric["lithium_balance_mol"] <= 2e-10
            and metric["negative_faraday_error_C"] <= 2e-5
            and metric["positive_faraday_error_C"] <= 2e-5
            and metric["energy_closure_maximum_tolerance_ratio"] <= 1
        )
        checks[name + ":zero_heat"] = metric["zero_heat_terms_maximum"] < 1e-10
        # These consistency values diagnose a mapping defect separately from
        # physical qualification; both original means and raw nodes are retained.
        values = {
            key: np.asarray([v for s in route["segments"] for v in s["values"][key]])
            for key in route["segments"][0]["values"]
        }
        if data.get("control_expectation") == "interior":
            checks[name + ":CV_interior"] = bool(
                np.max(abs(values["current_A"])) < data["maximum_current_A"]
                and np.max(abs(values["voltage_V"] - data["waveform"][0][1])) <= 2e-5
            )
        elif data.get("control_expectation") == "both-limits":
            checks[name + ":both_limits"] = bool(
                len(route["segments"]) == 2
                and all(
                    np.max(
                        abs(np.asarray(s["values"]["current_A"]) - sign * data["maximum_current_A"])
                    )
                    <= 1e-7
                    for s, sign in zip(route["segments"], (1, -1), strict=True)
                )
            )
        if data["name"] == "adiabatic-zero-current-gradient":
            first = route["segments"][0]
            raw = np.asarray(first["temperatures_K"])
            analytical = matrix_temperature(
                data, route["thermal_cells_per_layer"], first["times_s"]
            )
            metric["matrix_maximum_error_K"] = float(np.max(abs(raw - analytical)))
            cap = np.asarray(route["geometry"]["capacities_J_K"])
            initial = initial_temperatures(data, route["thermal_cells_per_layer"])
            metric["adiabatic_thermal_energy_error_J"] = float(np.max(abs((raw - initial) @ cap)))
            checks[name + ":matrix"] = metric["matrix_maximum_error_K"] <= 0.01
            checks[name + ":adiabatic_C_energy"] = (
                metric["adiabatic_thermal_energy_error_J"] <= 1e-5
            )
        if data["name"] == "lumped-thermal-limit":
            geometry = route["geometry"]
            tau = sum(geometry["capacities_J_K"]) / sum(geometry["loss_conductances_W_K"])
            times = np.asarray(route["segments"][0]["times_s"])
            exact = data["ambient_temperature_K"] + (
                data["initial_temperature_K"] - data["ambient_temperature_K"]
            ) * np.exp(-times / tau)
            metric["lumped_limit_maximum_error_K"] = float(np.max(abs(values["C_mean_K"] - exact)))
            checks[name + ":lumped_limit"] = metric["lumped_limit_maximum_error_K"] <= 0.02
    if data["expected"] == "maximum-temperature":
        item["events"] = events = {}
        for name, route in routes.items():
            if not route.get("segments"):
                checks[name + ":event_face"] = False
                continue
            final = route["segments"][-1]
            events[name] = {
                "time_s": final["end_time_s"],
                "reason": termination(final),
                "maximum_K": final["values"]["maximum_K"][-1],
                "C_mean_K": final["values"]["C_mean_K"][-1],
            }
            checks[name + ":event_face"] = (
                abs(events[name]["maximum_K"] - data["maximum_temperature_K"]) <= 2e-5
                and events[name]["time_s"] < sum(v[0] for v in data["waveform"])
                and data["maximum_temperature_K"] - events[name]["C_mean_K"] >= 0.01
            )
        for label, a, b in pairs:
            checks[label + ":event_time"] = (
                a in events
                and b in events
                and abs(events[a]["time_s"] - events[b]["time_s"])
                <= 0.5 + 0.01 * abs(events[b]["time_s"])
            )
    item["passed"] = bool(all(checks.values()))
