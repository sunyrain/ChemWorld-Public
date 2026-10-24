"""Fixed completeness, balances and source comparisons; no selective qualification."""

import numpy as np
from battery_plating_cases import PROFILE_RADII, RESPONSE_ATOL, RESPONSE_RTOL, RESPONSES
from battery_plating_traces import history_metrics


def termination(segment):
    raw = segment.get("termination")
    return None if raw in (None, "final time") else str(raw).removeprefix("event: ")


def integrity(trace, data):
    checks = {"no_exception": "exception" not in trace, "has_segments": bool(trace.get("segments"))}
    segments = trace.get("segments", [])
    checks["segment_count"] = len(segments) == len(data["waveform"])
    if not segments:
        return checks
    first = np.asarray(segments[0]["rows"])
    authored = [
        e["active_volume_m3"] * e["maximum_concentration_mol_m3"] * e["initial_fraction"]
        for e in data["electrodes"]
    ]
    if first.ndim != 2 or first.shape[1] != len(RESPONSES):
        return {**checks, "first_shape": False}
    checks["authored_initial_inventory"] = bool(
        np.max(
            abs(
                first[0, [5, 6, 12, 18]]
                - [data["initial_plated_mol"], data["initial_dead_mol"], *authored]
            )
        )
        <= 1e-12
    )
    checks["initial_Q_W"] = bool(np.max(abs(first[0, 2:4])) <= 1e-12)
    cumulative = 0.0
    for index, s in enumerate(segments):
        prefix = str(index) + ":"
        t, rows = np.asarray(s["times_s"]), np.asarray(s["rows"])
        if index >= len(data["waveform"]):
            checks[prefix + "unexpected_segment"] = False
            continue
        duration = data["waveform"][index][0]
        checks[prefix + "shape"] = bool(
            t.ndim == 1 and rows.shape == (len(t), len(RESPONSES)) and len(t) >= 2
        )
        checks[prefix + "finite"] = bool(np.isfinite(t).all() and np.isfinite(rows).all())
        checks[prefix + "ordered"] = bool(np.all(np.diff(t) > 0))
        checks[prefix + "start"] = bool(
            abs(t[0] - cumulative) <= 1e-7 and abs(s["start_time_s"] - t[0]) <= 1e-7
        )
        checks[prefix + "reported_end"] = bool(abs(t[-1] - s["end_time_s"]) <= 1e-7)
        expected_event = data["expected"] if data["expected"].startswith("battery:") else None
        actual_event = termination(s)
        checks[prefix + "event"] = actual_event == expected_event
        if actual_event is None:
            checks[prefix + "complete_horizon"] = bool(abs(t[-1] - cumulative - duration) <= 1e-7)
            checks[prefix + "all_requested_times"] = bool(
                all(
                    np.any(abs(t - (cumulative + x)) <= 1e-7)
                    for x in np.linspace(0.0, duration, data["samples_per_segment"])
                )
            )
        else:
            checks[prefix + "real_event_horizon"] = bool(
                cumulative < t[-1] <= cumulative + duration + 1e-7
            )
            checks[prefix + "all_pre_event_times"] = bool(
                all(
                    np.any(abs(t - (cumulative + x)) <= 1e-7)
                    for x in np.linspace(0.0, duration, data["samples_per_segment"])
                    if cumulative + x < t[-1] - 1e-7
                )
            )
        if "outcome" in s:
            checks[prefix + "native_outcome"] = s["outcome"] == (
                "success" if actual_event is None else "stopped"
            )
        raw, nodes = np.asarray(s["raw_profiles_theta"]), np.asarray(s["raw_r_over_R"])
        profiles = np.asarray(s["profiles_theta"])
        checks[prefix + "raw_profile_shape"] = raw.shape == (2, len(t), trace["radial_cells"])
        checks[prefix + "raw_node_shape"] = nodes.shape == (2, trace["radial_cells"])
        checks[prefix + "profile_shape"] = profiles.shape == (2, len(t), len(PROFILE_RADII))
        checks[prefix + "raw_profiles_finite"] = bool(
            np.isfinite(raw).all() and np.isfinite(nodes).all() and np.isfinite(profiles).all()
        )
        if "raw_amounts_mol" in s:
            checks[prefix + "raw_amounts_finite"] = bool(
                len(s["raw_amounts_mol"]) == len(t)
                and all(all(np.isfinite(v) for v in r.values()) for r in s["raw_amounts_mol"])
            )
        if index == 0:
            checks[prefix + "original_uniform_profile"] = bool(
                np.max(
                    abs(
                        raw[:, 0, :]
                        - np.array([e["initial_fraction"] for e in data["electrodes"]])[:, None]
                    )
                )
                <= 1e-12
            )
        cumulative += duration
    return checks


def compare(left, right, *, old_limit=False):
    if len(left.get("segments", [])) != len(right.get("segments", [])):
        return {"passed": False, "reason": "segment count differs"}
    left_rows, right_rows, profile_errors, matched_counts = [], [], [], []
    for a, b in zip(left["segments"], right["segments"], strict=True):
        ta, tb = np.asarray(a["times_s"]), np.asarray(b["times_s"])
        count = 0
        for i, t in enumerate(ta):
            matches = np.flatnonzero(abs(tb - t) <= 1e-7)
            if matches.size:
                j = matches[0]
                left_rows.append(a["rows"][i])
                right_rows.append(b["rows"][j])
                profile_errors.append(
                    float(
                        np.max(
                            abs(
                                np.asarray(a["profiles_theta"])[:, i]
                                - np.asarray(b["profiles_theta"])[:, j]
                            )
                        )
                    )
                )
                count += 1
        matched_counts.append(count)
        # All shared regular observations are mandatory; only differing real event
        # endpoints may be outside the intersection, and their boundaries are checked.
        required = sum(t <= min(ta[-1], tb[-1]) + 1e-7 for t in ta[:-1])
        if count < max(1, required):
            return {
                "passed": False,
                "reason": "missing common requested observations",
                "matched_counts": matched_counts,
            }
    if not left_rows:
        return {"passed": False, "reason": "no common observations"}
    a, b = np.asarray(left_rows), np.asarray(right_rows)
    atol, rtol = RESPONSE_ATOL, RESPONSE_RTOL
    profile_limit = 0.003
    if old_limit:
        atol = np.array(
            [
                1e-7,
                1e-7,
                1e-5,
                1e-4,
                1e-7,
                1e-10,
                1e-10,
                1e-7,
                1e-4,
                *([1e-7, 1e-8, 1e-8, 1e-10, 1e-7, 1e-7] * 2),
            ]
        )
        rtol, profile_limit = 1e-7, 1e-8
    errors = abs(a - b)
    ratios = errors / (atol + rtol * abs(b))
    return {
        "matched_counts": matched_counts,
        "total_common_observations": len(a),
        "maximum_absolute_error_by_response": dict(
            zip(RESPONSES, np.max(errors, axis=0).tolist(), strict=True)
        ),
        "maximum_limit_ratio": float(np.max(ratios)),
        "maximum_profile_error": max(profile_errors),
        "profile_limit": profile_limit,
        "passed": bool(np.max(ratios) <= 1 and max(profile_errors) <= profile_limit),
    }


def assess(item):
    data, routes = item["inputs"], item["routes"]
    checks, metrics, comparisons = {}, {}, {}
    item.update(checks=checks, history_metrics=metrics, comparisons=comparisons)
    if data["expected"] == "refusals":
        for name, route in routes.items():
            checks[name] = (
                route.get("rejected") is True
                and route.get("exception_type") == "ValueError"
                and route.get("input_state_unchanged") is True
                and bool(route.get("exception"))
            )
        checks["all_refusals_present"] = len(routes) == 12
    else:
        expected_routes = 8 if data["index"] == 1 else 4
        checks["route_count"] = len(routes) == expected_routes
        valid = {}
        for name, route in routes.items():
            local = integrity(route, data)
            checks.update({name + ":" + k: bool(v) for k, v in local.items()})
            valid[name] = all(local.values())
            if not valid[name]:
                continue
            values = history_metrics(route, data)
            metrics[name] = values
            for key in (
                "total_Li_error_mol",
                "negative_Faraday_error_mol",
                "positive_Faraday_error_mol",
                "stage_inventory_jump_mol",
            ):
                checks[name + ":" + key] = values[key] <= 1e-9
            checks[name + ":algebraic_current"] = values["algebraic_current_error_A"] <= 1e-7
            checks[name + ":kinetic_law"] = values["independent_kinetic_limit_ratio"] <= 1
            checks[name + ":concentration_mapping"] = (
                values["concentration_amount_error_mol"] <= 1e-12
            )
            checks[name + ":nonnegative_metal"] = values["minimum_metal_mol"] >= -1e-12
            checks[name + ":work_quadrature"] = values["work_quadrature_limit_ratio"] <= 1
            checks[name + ":CC_charge"] = values["CC_charge_error_C"] <= 1e-5
            rows = np.asarray([r for s in route["segments"] for r in s["rows"]])
            if data["index"] == 1:
                checks[name + ":zero_side"] = bool(np.max(abs(rows[:, 4:7])) <= 1e-12)
            if data["index"] == 2:
                checks[name + ":deposition"] = bool(rows[0, 4] < 0 and rows[-1, 5] > rows[0, 5])
            if data["index"] in (3, 4):
                checks[name + ":stripping"] = bool(rows[0, 4] > 0 and rows[-1, 5] < rows[0, 5])
            if data["index"] == 4:
                checks[name + ":internal_rest"] = bool(
                    abs(rows[0, 4]) > 1e-4
                    and rows[0, 9] * rows[0, 4] < 0
                    and np.max(abs(rows[:, 1:4])) <= 1e-10
                )
            if data["mode"] == "irreversible":
                checks[name + ":irreversible_route"] = bool(
                    np.max(abs(rows[:, 5])) <= 1e-12
                    and np.min(np.diff(rows[:, 6])) >= -1e-12
                    and rows[-1, 6] > rows[0, 6]
                )
            if data["index"] in (9, 10):
                for i, (segment, (_, target, _)) in enumerate(
                    zip(route["segments"], data["waveform"], strict=True)
                ):
                    r = np.asarray(segment["rows"])
                    if data["index"] == 9:
                        checks[name + f":{i}:CV"] = bool(
                            np.max(abs(r[:, 0] - target)) <= 1e-7
                            and np.max(abs(r[:, 1])) < data["maximum_current_A"]
                        )
                    else:
                        required = (-1 if i == 0 else 1) * data["maximum_current_A"]
                        checks[name + f":{i}:current_limit"] = bool(
                            np.max(abs(r[:, 1] - required)) <= 1e-7
                        )
            if data["expected"].startswith("battery:"):
                face = (
                    abs(rows[-1, 10] - data["surface_margin"])
                    if data["index"] == 11
                    else abs(rows[-1, 0] - data["minimum_voltage_V"])
                )
                checks[name + ":event_boundary"] = bool(
                    face <= (1e-7 if data["index"] == 11 else 1e-6)
                )
        pairs = [
            ("paired_coarse", "native_coarse", "reference_coarse", False),
            ("paired_fine", "native_fine", "reference_fine", False),
            ("native_grid", "native_coarse", "native_fine", False),
            ("source_grid", "reference_coarse", "reference_fine", False),
        ]
        if data["index"] == 1:
            pairs.extend(
                (f"{impl}_old_{level}", f"{impl}_{level}", f"old_{impl}_{level}", True)
                for impl in ("native", "reference")
                for level in ("coarse", "fine")
            )
        for name, left, right, old in pairs:
            if not (valid.get(left) and valid.get(right)):
                comparisons[name] = {
                    "passed": False,
                    "reason": "required complete histories unavailable",
                }
            else:
                comparisons[name] = compare(routes[left], routes[right], old_limit=old)
                if data["expected"].startswith("battery:"):
                    a, b = (routes[key]["segments"][-1]["end_time_s"] for key in (left, right))
                    comparisons[name]["event_time_error_s"] = abs(a - b)
                    comparisons[name]["event_time_limit_s"] = 1.0 + 0.002 * abs(b)
                    checks[name + ":event_time"] = abs(a - b) <= 1.0 + 0.002 * abs(b)
            checks[name] = comparisons[name]["passed"]
    item["failures"] = [key for key, value in checks.items() if not value]
    item["passed"] = bool(checks and not item["failures"])
