"""Fixed completeness, balances and source comparisons; no selective qualification."""

import numpy as np
from battery_partial_plating_traces import history_metrics
from battery_plating_cases import PROFILE_RADII, RESPONSE_ATOL, RESPONSE_RTOL, RESPONSES
from scipy.integrate import simpson


def termination(segment):
    raw = segment["termination"]
    return None if raw in (None, "final time") else str(raw).removeprefix("event: ")


def integrity(trace, data):
    try:
        return _integrity(trace, data)
    except (KeyError, TypeError, ValueError, IndexError):
        return {"well_formed_required_history": False}


def _integrity(trace, data):
    checks = {"no_exception": "exception" not in trace, "has_segments": bool(trace.get("segments"))}
    implementations = {
        "native original reversible SPM plating": "native",
        "native SPM with constant metal deactivation": "native",
        "actual PyBaMM25.4.2 SPM partial constant decay": "reference",
        "actual PyBaMM25.4.2 original reversible SPM plating": "reference",
    }
    checks["implementation"] = trace.get("implementation") in implementations
    required = {
        "implementation",
        "segments",
        "radial_cells",
        "response_order",
        "profile_r_over_R",
        "F_C_mol",
        "R_J_mol_K",
    }
    checks["required_trace_fields"] = required <= trace.keys()
    if not checks["implementation"] or not checks["required_trace_fields"]:
        return checks
    is_native = implementations[trace["implementation"]] == "native"
    checks["declared_response_order"] = tuple(trace["response_order"]) == tuple(RESPONSES)
    expected_keys = {
        f"{name}:Li:{i}" for name in ("negative", "positive") for i in range(trace["radial_cells"])
    } | {"plating:Li", "dead:Li"}
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
        required_segment = {
            "times_s",
            "rows",
            "start_time_s",
            "end_time_s",
            "termination",
            "profiles_theta",
            "raw_r_over_R",
            "raw_profiles_theta",
            "deactivation_rate_mol_s",
        }
        if is_native:
            required_segment |= {
                "outcome",
                "raw_amounts_mol",
                "solver",
                "conservation_residuals",
                "roundoff_corrections_mol",
            }
        else:
            required_segment.add("sei_thickness_m")
        checks[prefix + "required_segment_fields"] = required_segment <= s.keys()
        if not checks[prefix + "required_segment_fields"]:
            continue
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
        if is_native or "outcome" in s:
            checks[prefix + "native_outcome"] = s["outcome"] == (
                "success" if actual_event is None else "stopped"
            )
        if not is_native:
            checks[prefix + "source_termination_label"] = s["termination"] == (
                "final time" if actual_event is None else "event: " + expected_event
            )
        raw, nodes = np.asarray(s["raw_profiles_theta"]), np.asarray(s["raw_r_over_R"])
        profiles = np.asarray(s["profiles_theta"])
        checks[prefix + "raw_profile_shape"] = raw.shape == (2, len(t), trace["radial_cells"])
        checks[prefix + "raw_node_shape"] = nodes.shape == (2, trace["radial_cells"])
        checks[prefix + "profile_shape"] = profiles.shape == (2, len(t), len(PROFILE_RADII))
        checks[prefix + "raw_profiles_finite"] = bool(
            np.isfinite(raw).all() and np.isfinite(nodes).all() and np.isfinite(profiles).all()
        )
        rates = np.asarray(s["deactivation_rate_mol_s"])
        checks[prefix + "deactivation_shape_finite"] = bool(
            rates.shape == t.shape and np.isfinite(rates).all()
        )
        if is_native:
            checks[prefix + "native_diagnostic_types"] = all(
                isinstance(s[k], dict)
                for k in ("solver", "conservation_residuals", "roundoff_corrections_mol")
            )
            checks[prefix + "raw_amounts_keys"] = len(s["raw_amounts_mol"]) == len(t) and all(
                set(r) == expected_keys for r in s["raw_amounts_mol"]
            )
            checks[prefix + "raw_amounts_finite"] = bool(
                len(s["raw_amounts_mol"]) == len(t)
                and all(all(np.isfinite(v) for v in r.values()) for r in s["raw_amounts_mol"])
            )
        else:
            sei = np.asarray(s["sei_thickness_m"])
            checks[prefix + "sei_shape_finite"] = bool(
                sei.shape == t.shape and np.isfinite(sei).all()
            )
        if index:
            checks[prefix + "stage_radial_mesh_unchanged"] = bool(
                np.array_equal(nodes, np.asarray(segments[index - 1]["raw_r_over_R"]))
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
                1e-5,
                *([1e-7, 1e-7, 1e-7, 1e-10, 1e-7, 1e-7] * 2),
            ]
        )
        rtol, profile_limit = 0.0, 1e-7
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
    item.update(checks=checks, metrics=metrics, comparisons=comparisons)
    if data["index"] == 12:
        from battery_partial_plating_cases import REFUSALS

        checks["route_names"] = set(routes) == set(REFUSALS)
        for name, route in routes.items():
            checks[name + ":rejected"] = route.get("rejected") is True
            checks[name + ":immutable"] = route.get("input_state_unchanged") is True
    else:
        expected = {
            f"{impl}_{level}" for impl in ("native", "reference") for level in ("coarse", "fine")
        }
        if data["index"] == 2:
            expected |= {"old_" + key for key in expected}
        checks["route_names"] = set(routes) == expected
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
                "stage_radial_inventory_jump_mol",
                "stage_raw_inventory_jump_mol",
            ):
                checks[name + ":" + key] = values[key] <= 1e-9
            for key, limit in (
                ("algebraic_current_error_A", 1e-7),
                ("independent_kinetic_limit_ratio", 1.0),
                ("concentration_amount_error_mol", 1e-12),
                ("work_quadrature_limit_ratio", 1.0),
                ("CC_charge_error_C", 1e-5),
                ("stage_charge_jump_C", 1e-5),
                ("stage_work_jump_J", 1e-4),
            ):
                checks[name + ":" + key] = values[key] <= limit
            checks[name + ":nonnegative_metal"] = values["minimum_metal_mol"] >= -1e-12
            rows = np.asarray([r for segment in route["segments"] for r in segment["rows"]])
            checks[name + ":dead_monotone"] = bool(np.min(np.diff(rows[:, 6])) >= -1e-12)
            gamma = data["gamma_s_inv"]
            rate_ratios, integral_ratios = [], []
            for i, segment in enumerate(route["segments"]):
                r, t = np.asarray(segment["rows"]), np.asarray(segment["times_s"])
                rate = np.asarray(segment.get("deactivation_rate_mol_s", []))
                good_rate = rate.shape == t.shape and np.isfinite(rate).all()
                checks[name + f":{i}:actual_deactivation_rate"] = bool(good_rate)
                if good_rate:
                    expected_rate = gamma * r[:, 5]
                    rate_ratios.append(
                        float(
                            np.max(abs(rate - expected_rate) / (1e-12 + 1e-8 * abs(expected_rate)))
                        )
                    )
                    delta = r[-1, 6] - r[0, 6]
                    integral_ratios.append(
                        float(abs(delta - simpson(rate, x=t)) / (2e-9 + 1e-3 * abs(delta)))
                    )
                if name.startswith("reference") or name.startswith("old_reference"):
                    sei = np.asarray(segment.get("sei_thickness_m", []))
                    checks[name + f":{i}:real_zero_SEI"] = bool(
                        sei.shape == t.shape and np.isfinite(sei).all() and np.all(sei == 0.0)
                    )
                if data["index"] == 1:
                    expected_plated = data["initial_plated_mol"] * np.exp(-gamma * t)
                    expected_dead = (
                        data["initial_dead_mol"] + data["initial_plated_mol"] - expected_plated
                    )
                    error = abs(r[:, 5:7] - np.column_stack((expected_plated, expected_dead)))
                    allowance = 2e-10 + 1e-7 * abs(
                        np.column_stack((expected_plated, expected_dead))
                    )
                    checks[name + ":pure_exp"] = bool(np.max(error / allowance) <= 1.0)
                    values["pure_exp_maximum_error_mol"] = float(np.max(error))
                    checks[name + ":no_electron_or_particle_effect"] = bool(
                        np.max(abs(r[:, 1:5])) <= 1e-10
                        and np.max(abs(r[:, [12, 18]] - r[0, [12, 18]])) <= 1e-10
                    )
                if data["index"] in (8, 9):
                    target = data["waveform"][i][1]
                    if data["index"] == 8:
                        checks[name + f":{i}:CV"] = bool(
                            np.max(abs(r[:, 0] - target)) <= 1e-7
                            and np.max(abs(r[:, 1])) < data["maximum_current_A"]
                        )
                    else:
                        required = (-1 if i == 0 else 1) * data["maximum_current_A"]
                        checks[name + f":{i}:current_limit"] = bool(
                            np.max(abs(r[:, 1] - required)) <= 1e-7
                        )
            values["deactivation_rate_limit_ratio"] = max(rate_ratios, default=float("inf"))
            values["dead_integral_limit_ratio"] = max(integral_ratios, default=float("inf"))
            checks[name + ":deactivation_law"] = values["deactivation_rate_limit_ratio"] <= 1.0
            checks[name + ":dead_integral"] = values["dead_integral_limit_ratio"] <= 1.0
            if gamma > 0:
                checks[name + ":dead_growth"] = bool(rows[-1, 6] > rows[0, 6])
            if data["index"] == 3:
                checks[name + ":zero_metal_deposition"] = bool(
                    rows[0, 4] < 0 and rows[-1, 5] > 0 and rows[-1, 6] > 0
                )
            if data["index"] == 4:
                checks[name + ":internal_rest"] = bool(
                    rows[0, 4] > 1e-4
                    and rows[0, 9] * rows[0, 4] < 0
                    and np.max(abs(rows[:, 1:4])) <= 1e-10
                )
            if data["index"] == 5:
                checks[name + ":stripping_with_deactivation"] = bool(
                    rows[0, 4] > 0 and rows[-1, 5] < rows[0, 5]
                )
            if data["expected"].startswith("battery:"):
                face = (
                    abs(rows[-1, 10] - data["surface_margin"])
                    if data["index"] == 10
                    else abs(rows[-1, 0] - data["minimum_voltage_V"])
                )
                values["event_boundary_error"] = float(face)
                checks[name + ":event_boundary"] = bool(
                    face <= (1e-7 if data["index"] == 10 else 1e-6)
                )
        pairs = [
            ("paired_coarse", "native_coarse", "reference_coarse", False),
            ("paired_fine", "native_fine", "reference_fine", False),
            ("native_grid", "native_coarse", "native_fine", False),
            ("source_grid", "reference_coarse", "reference_fine", False),
        ]
        if data["index"] == 2:
            pairs += [
                (f"{impl}_old_{level}", f"{impl}_{level}", f"old_{impl}_{level}", True)
                for impl in ("native", "reference")
                for level in ("coarse", "fine")
            ]
        for name, left, right, old in pairs:
            if not (valid.get(left) and valid.get(right)):
                comparisons[name] = {"passed": False, "reason": "required histories incomplete"}
            else:
                comparisons[name] = compare(routes[left], routes[right], old_limit=old)
                rate_ratios = []
                for a, b in zip(routes[left]["segments"], routes[right]["segments"], strict=True):
                    ta, tb = np.asarray(a["times_s"]), np.asarray(b["times_s"])
                    for i, t in enumerate(ta):
                        match = np.flatnonzero(abs(tb - t) <= 1e-7)
                        if match.size:
                            j = match[0]
                            allowance = (
                                data["gamma_s_inv"] * (2e-7 + 1e-5 * abs(b["rows"][j][5])) + 1e-12
                            )
                            rate_ratios.append(
                                abs(
                                    a["deactivation_rate_mol_s"][i]
                                    - b["deactivation_rate_mol_s"][j]
                                )
                                / allowance
                            )
                comparisons[name]["deactivation_limit_ratio"] = float(
                    max(rate_ratios, default=float("inf"))
                )
                checks[name + ":deactivation"] = (
                    comparisons[name]["deactivation_limit_ratio"] <= 1.0
                )
                if data["expected"].startswith("battery:"):
                    a, b = (routes[key]["segments"][-1]["end_time_s"] for key in (left, right))
                    comparisons[name].update(
                        event_time_error_s=abs(a - b), event_time_limit_s=1.0 + 0.002 * abs(b)
                    )
                    checks[name + ":event_time"] = abs(a - b) <= 1.0 + 0.002 * abs(b)
            checks[name] = comparisons[name]["passed"]
    item["failures"] = [key for key, value in checks.items() if not value]
    item["passed"] = bool(checks and not item["failures"])
