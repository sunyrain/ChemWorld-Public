"""All declared responses, original times and failures retained in fixed assessment."""

import numpy as np
from battery_msmr_cases import F


def termination(segment):
    value = segment.get("termination")
    if value in (None, "final time"):
        return None
    # Family prefixes are API labels, not different stopping physics. Raw route
    # termination is retained untouched; only the comparison kind is canonical.
    name = str(value).removeprefix("event: ")
    if name.startswith("msmr:"):
        name = "battery:" + name.removeprefix("msmr:")
    return name


def complete(route, data):
    if "exception" in route or not route.get("segments"):
        return False
    for s in route["segments"]:
        if len(s["times_s"]) < 2 or not all(np.isfinite(x).all() for x in s["values"].values()):
            return False
        if s.get("outcome", "success") not in {"success", "event"} and termination(s) is None:
            return False
    if data["expected"].startswith("battery:"):
        return termination(route["segments"][-1]) == data["expected"]
    return (
        len(route["segments"]) == len(data["waveform"])
        and all(termination(s) is None for s in route["segments"])
        and abs(route["segments"][-1]["end_time_s"] - sum(x[0] for x in data["waveform"])) < 1e-6
    )


def limits(key):
    if key == "voltage_V":
        return 0.002, 1e-5
    if key == "current_A" or ":current" in key:
        return 0.005, 0.005
    if key == "charge_C":
        return 0.02, 1e-5
    if key == "work_J":
        return 0.1, 1e-5
    if key.endswith(":surface_potential") or key.endswith(":eta"):
        return 0.001, 1e-5
    if key.endswith(":amount_mol"):
        return 2e-7, 1e-6
    return 0.003, 0.0


def aligned_indices(x, y):
    tx, ty = np.asarray(x["times_s"]), np.asarray(y["times_s"])
    pairs = []
    for i, value in enumerate(tx):
        found = np.flatnonzero(abs(ty - value) <= 1e-7)
        if found.size:
            pairs.append((i, int(found[0])))
    if len(pairs) < 2:
        raise ValueError("fewer than two exact common times; no interpolation of unmatched event")
    return np.asarray(pairs).T


def compare(a, b):
    errors, ratios, profile_errors, observations = {}, {}, {}, 0
    if len(a["segments"]) != len(b["segments"]):
        return {"passed": False, "reason": "segment count mismatch"}
    for x, y in zip(a["segments"], b["segments"], strict=True):
        ix, iy = aligned_indices(x, y)
        observations += len(ix)
        if x["values"].keys() != y["values"].keys():
            raise ValueError("response identity mismatch")
        for key in x["values"]:
            left, right = np.asarray(x["values"][key])[ix], np.asarray(y["values"][key])[iy]
            error = abs(left - right)
            atol, rtol = limits(key)
            errors[key] = max(errors.get(key, 0.0), float(np.max(error)))
            ratios[key] = max(
                ratios.get(key, 0.0), float(np.max(error / (atol + rtol * abs(right))))
            )
        for domain in ("negative", "positive"):
            for quantity, bound in (("sampled_theta", 0.003), ("sampled_potential_V", 0.001)):
                delta = abs(
                    np.asarray(x["raw_profiles"][domain][quantity])[ix]
                    - np.asarray(y["raw_profiles"][domain][quantity])[iy]
                )
                key = domain + ":" + quantity
                profile_errors[key] = max(profile_errors.get(key, 0.0), float(np.max(delta)))
                ratios[key] = max(ratios.get(key, 0.0), float(np.max(delta) / bound))
    maximum = max(ratios.values())
    return {
        "common_time_count": observations,
        "maximum_errors": errors,
        "profile_errors": profile_errors,
        "tolerance_ratios": ratios,
        "maximum_tolerance_ratio": maximum,
        "passed": maximum <= 1.0,
    }


def history_metrics(route, data):
    values = {
        k: np.concatenate([np.asarray(s["values"][k]) for s in route["segments"]])
        for k in route["segments"][0]["values"]
    }
    n = values["negative:amount_mol"] - values["negative:amount_mol"][0]
    p = values["positive:amount_mol"] - values["positive:amount_mol"][0]
    charge = values["charge_C"] - values["charge_C"][0]
    current_error = []
    for e, sign in zip(data["electrodes"], (1.0, -1.0), strict=True):
        summed = sum(values[e["name"] + f":current{j}_A"] for j in range(len(e["rows"])))
        current_error.append(float(np.max(abs(summed - sign * values["current_A"]))))
    return {
        "lithium_balance_mol": float(np.max(abs(n + p))),
        "negative_faraday_error_C": float(np.max(abs(F * n + charge))),
        "positive_faraday_error_C": float(np.max(abs(F * p - charge))),
        "branch_current_balance_A": max(current_error),
    }


def matrix_compare(history, linear):
    segment = history["segments"][0]
    ix, iy = aligned_indices(segment, linear)
    errors = {}
    for domain in ("negative", "positive"):
        left = np.asarray(segment["raw_profiles"][domain]["sampled_theta"])[ix]
        right = np.asarray(linear["profiles"][domain]["sampled_theta"])[iy]
        errors[domain] = float(np.max(abs(left - right)))
    return {
        "common_time_count": len(ix),
        "profile_errors": errors,
        "maximum_tolerance_ratio": max(errors.values()) / 2e-5,
        "passed": max(errors.values()) <= 2e-5,
    }


def assess(item):
    data, routes = item["inputs"], item["routes"]
    checks = item["checks"] = {}
    if data["expected"] == "static":
        value = routes["static"]
        checks["all_static"] = (
            value.get("point_denominator") == 486
            and value.get("passed_points") == 486
            and len(value.get("points", [])) == 486
        )
        item["passed"] = all(checks.values())
        return
    if data["expected"] == "refusals":
        for name, value in routes.items():
            checks[name] = (
                value.get("rejected") is True
                and value.get("input_state_unchanged") is True
                and value.get("exception_type") in {"ValueError", "ResidualDomainError"}
                and bool(value.get("exception"))
            )
        item["passed"] = len(checks) == 8 and all(checks.values())
        return
    history_names = [key for key in routes if not key.startswith("linear_")]
    item["history_metrics"], item["comparisons"] = {}, {}
    for name in history_names:
        route = routes[name]
        checks[name + ":complete"] = complete(route, data)
        if not checks[name + ":complete"]:
            continue
        metrics = history_metrics(route, data)
        item["history_metrics"][name] = metrics
        checks[name + ":balance"] = (
            metrics["lithium_balance_mol"] <= 2e-9
            and metrics["negative_faraday_error_C"] <= 2e-4
            and metrics["positive_faraday_error_C"] <= 2e-4
            and metrics["branch_current_balance_A"] <= 2e-5
        )
        if data["name"] == "nonuniform-potential-rest":
            for domain in ("negative", "positive"):
                profile = np.asarray(route["segments"][0]["raw_profiles"][domain]["potential_V"])
                checks[name + ":" + domain + ":rest_relaxes"] = np.ptp(profile[-1]) < np.ptp(
                    profile[0]
                )
        if data.get("control_expectation"):
            for index, (segment, (_, target, _)) in enumerate(
                zip(route["segments"], data["waveform"], strict=True)
            ):
                current = np.asarray(segment["values"]["current_A"])
                voltage = np.asarray(segment["values"]["voltage_V"])
                if data["control_expectation"] == "interior":
                    checks[name + f":CV{index}"] = bool(
                        np.max(abs(voltage - target)) <= 1e-7
                        and np.max(abs(current)) < data["maximum_current_A"]
                    )
                else:
                    sign = 1.0 if index == 0 else -1.0
                    checks[name + f":CV{index}"] = bool(
                        np.max(abs(current - sign * data["maximum_current_A"])) <= 1e-7
                    )
        if data["expected"].startswith("battery:"):
            v = route["segments"][-1]["values"]["voltage_V"][-1]
            checks[name + ":event_voltage"] = abs(v - data["minimum_voltage_V"]) <= 2e-5
    pairs = (
        ("coarse", "native_coarse", "reference_coarse"),
        ("fine", "native_fine", "reference_fine"),
        ("native_grid", "native_coarse", "native_fine"),
        ("reference_grid", "reference_coarse", "reference_fine"),
    )
    for label, a, b in pairs:
        if not (checks[a + ":complete"] and checks[b + ":complete"]):
            checks[label] = False
            continue
        item["comparisons"][label] = value = compare(routes[a], routes[b])
        checks[label] = value["passed"]
        if data["expected"].startswith("battery:"):
            left, right = (routes[k]["segments"][-1]["end_time_s"] for k in (a, b))
            checks[label + ":event_time"] = abs(left - right) <= 0.5 + 0.005 * abs(right)
    if "fick_multiplier" in data:
        for label in ("coarse", "fine"):
            linear = routes["linear_" + label]
            checks["linear_" + label] = (
                "exception" not in linear
                and linear["lithium_balance_mol"] <= 2e-9
                and all(v["faraday_error_C"] <= 2e-4 for v in linear["profiles"].values())
            )
            if not checks["linear_" + label]:
                continue
            for implementation in ("native", "reference"):
                key = implementation + "_" + label
                if checks[key + ":complete"]:
                    result = matrix_compare(routes[key], linear)
                    item["comparisons"][key + ":matrix"] = result
                    checks[key + ":matrix"] = result["passed"]
    item["passed"] = bool(all(checks.values()))
