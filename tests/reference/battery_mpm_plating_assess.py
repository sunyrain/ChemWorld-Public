"""Fixed completeness, balances and source comparisons; no selective qualification."""

import numpy as np
from battery_mpm_plating_cases import RESPONSE_ATOL, RESPONSE_RTOL, RESPONSES
from battery_mpm_plating_traces import history_metrics, local_metrics


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
        for e in data["electrodes"]:
            name = e["name"]
            local = s["local"][name]
            bins = len(trace["mapping"][name]["radii_m"])
            raw = np.asarray(local["profiles_theta"])
            checks[prefix + name + ":raw_shape"] = raw.shape == (
                len(t),
                bins,
                trace["radial_cells"],
            )
            checks[prefix + name + ":raw_finite"] = bool(np.isfinite(raw).all())
            checks[prefix + name + ":radii_shape"] = len(local["radii_m"]) == bins
            for key in (
                "surface_fraction",
                "average_fraction",
                "current_A",
                "j_A_m2",
                "ocp_V",
                "overpotential_V",
            ):
                values = np.asarray(local[key])
                checks[prefix + name + ":" + key] = values.shape == (len(t), bins) and bool(
                    np.isfinite(values).all()
                )
            checks[prefix + name + ":comparison_shape"] = np.asarray(
                local["comparison_profiles_theta"]
            ).shape == (len(t), 5, 5)
            if index == 0:
                checks[prefix + name + ":original_uniform_profile"] = bool(
                    np.max(abs(raw[0] - e["initial_fraction"])) <= 1e-12
                )
        if "raw_amounts_mol" in s:
            checks[prefix + "raw_amounts"] = len(s["raw_amounts_mol"]) == len(t) and all(
                all(np.isfinite(v) for v in row.values()) for row in s["raw_amounts_mol"]
            )
        cumulative += duration
    return checks


def compare(left, right, *, old_limit=False):
    if len(left.get("segments", [])) != len(right.get("segments", [])):
        return {"passed": False, "reason": "segment count differs"}
    lrows, rrows, profile_errors, surface_errors, jratios, matched = [], [], [], [], [], []
    for a, b in zip(left["segments"], right["segments"], strict=True):
        ta, tb = np.asarray(a["times_s"]), np.asarray(b["times_s"])
        count = 0
        for i, t in enumerate(ta):
            found = np.flatnonzero(abs(tb - t) <= 1e-7)
            if not found.size:
                continue
            j = int(found[0])
            count += 1
            lrows.append(a["rows"][i])
            rrows.append(b["rows"][j])
            for name in ("negative", "positive"):
                x, y = a["local"][name], b["local"][name]
                profile_errors.append(
                    float(
                        np.max(
                            abs(
                                np.array(x["comparison_profiles_theta"])[i]
                                - np.array(y["comparison_profiles_theta"])[j]
                            )
                        )
                    )
                )
                surface_errors.append(
                    float(
                        np.max(
                            abs(
                                np.array(x["comparison_surface_fraction"])[i]
                                - np.array(y["comparison_surface_fraction"])[j]
                            )
                        )
                    )
                )
                jx, jy = np.array(x["comparison_j_A_m2"])[i], np.array(y["comparison_j_A_m2"])[j]
                jratios.append(
                    float(
                        np.max(
                            abs(jx - jy)
                            / (
                                (1e-7 if old_limit else 0.05)
                                + (1e-7 if old_limit else 0.01) * abs(jy)
                            )
                        )
                    )
                )
        matched.append(count)
        required = sum(t <= min(ta[-1], tb[-1]) + 1e-7 for t in ta[:-1])
        if count < max(1, required):
            return {
                "passed": False,
                "reason": "missing common observations",
                "matched_counts": matched,
            }
    if not lrows:
        return {"passed": False, "reason": "no common observations"}
    a, b = np.array(lrows), np.array(rrows)
    atol, rtol = RESPONSE_ATOL, RESPONSE_RTOL
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
        rtol = 1e-7
    ratio = float(np.max(abs(a - b) / (atol + rtol * abs(b))))
    area_ratio = max(
        abs(left["mapping"][n]["active_area_m2"] / right["mapping"][n]["active_area_m2"] - 1)
        for n in ("negative", "positive")
    )
    limit = 1e-8 if old_limit else 0.01
    return {
        "matched_counts": matched,
        "total_common_observations": len(a),
        "maximum_absolute_error_by_response": dict(
            zip(RESPONSES, np.max(abs(a - b), axis=0).tolist(), strict=True)
        ),
        "maximum_limit_ratio": ratio,
        "maximum_profile_error": max(profile_errors),
        "maximum_local_surface_error": max(surface_errors),
        "maximum_local_j_ratio": max(jratios),
        "maximum_area_relative_error": area_ratio,
        "profile_limit": limit,
        "passed": bool(
            ratio <= 1
            and max(profile_errors) <= limit
            and max(surface_errors) <= limit
            and max(jratios) <= 1
            and area_ratio <= 0.005
        ),
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
        checks["route_count"] = len(routes) == (
            6 if data["index"] == 1 else 8 if data["index"] == 2 else 4
        )
        valid = {}
        for name, route in routes.items():
            local = integrity(route, data)
            checks.update({name + ":" + k: bool(v) for k, v in local.items()})
            valid[name] = all(local.values())
            if not valid[name]:
                continue
            values = {**history_metrics(route, data), **local_metrics(route, data)}
            metrics[name] = values
            for key in (
                "total_Li_error_mol",
                "negative_Faraday_error_mol",
                "positive_Faraday_error_mol",
                "stage_inventory_jump_mol",
            ):
                checks[name + ":" + key] = values[key] <= 1e-9
            checks[name + ":current_closure"] = (
                max(values["algebraic_current_error_A"], values["bin_current_closure_A"]) <= 2e-7
            )
            checks[name + ":side_law"] = values["independent_kinetic_limit_ratio"] <= 1
            checks[name + ":local_BV"] = values["local_BV_limit_ratio"] <= 1
            checks[name + ":area_identity"] = values["area_identity_relative_error"] <= 1e-12
            checks[name + ":weight_normalization"] = values["weight_normalization_error"] <= 1e-12
            checks[name + ":cpl_mapping"] = values["concentration_amount_error_mol"] <= 1e-12
            checks[name + ":metal_nonnegative"] = values["minimum_metal_mol"] >= -1e-12
            checks[name + ":work_quadrature"] = values["work_quadrature_limit_ratio"] <= 1
            checks[name + ":CC_Q"] = values["CC_charge_error_C"] <= 1e-5
            rows = np.array([row for s in route["segments"] for row in s["rows"]])
            if data["index"] == 2:
                checks[name + ":zero_side"] = bool(np.max(abs(rows[:, 4:7])) <= 1e-12)
            if data["index"] == 3:
                checks[name + ":deposition"] = bool(rows[0, 4] < 0 and rows[-1, 5] > rows[0, 5])
            if data["index"] == 4:
                equilibrium = (
                    route["R_J_mol_K"]
                    * data["temperature_K"]
                    / route["F_C_mol"]
                    * np.log(data["ce_mol_m3"] / rows[0, 8])
                )
                checks[name + ":rest_direction"] = bool((rows[0, 7] - equilibrium) * rows[0, 4] > 0)
                checks[name + ":internal_rest"] = bool(
                    abs(rows[0, 4]) > 1e-4
                    and rows[0, 9] * rows[0, 4] < 0
                    and np.max(abs(rows[:, 1:4])) <= 1e-10
                )
            if data["index"] in (3, 7):
                checks[name + ":genuine_bin_allocation"] = (
                    values["allocation_departure_from_area_A"] > 1e-5
                )
            if data["mode"] == "irreversible":
                checks[name + ":dead_route"] = bool(
                    np.max(abs(rows[:, 5])) <= 1e-12
                    and np.min(np.diff(rows[:, 6])) >= -1e-12
                    and rows[-1, 6] > rows[0, 6]
                )
            else:
                checks[name + ":reversible_dead_fixed"] = bool(
                    np.max(abs(rows[:, 6] - data["initial_dead_mol"])) <= 1e-12
                )
            if data["index"] in (8, 9):
                for i, (s, (_, target, _)) in enumerate(
                    zip(route["segments"], data["waveform"], strict=True)
                ):
                    r = np.array(s["rows"])
                    checks[name + f":{i}:control"] = (
                        bool(
                            np.max(abs(r[:, 0] - target)) <= 1e-7
                            and np.max(abs(r[:, 1])) < data["maximum_current_A"]
                        )
                        if data["index"] == 8
                        else bool(
                            np.max(abs(r[:, 1] - (-1 if i == 0 else 1) * data["maximum_current_A"]))
                            <= 1e-7
                        )
                    )
            if data["expected"].startswith("battery:"):
                if data["index"] == 10:
                    end = np.array(
                        route["segments"][-1]["local"]["negative"]["surface_fraction"][-1]
                    )
                    error = abs(end.min() - data["surface_margin"])
                    limit = 1e-7
                else:
                    error = abs(rows[-1, 0] - data["minimum_voltage_V"])
                    limit = 1e-6
                checks[name + ":event_boundary"] = bool(error <= limit)
        pairs = [
            ("paired_coarse", "native_coarse", "reference_coarse", False),
            ("paired_fine", "native_fine", "reference_fine", False),
            ("native_grid", "native_coarse", "native_fine", False),
            ("source_grid", "reference_coarse", "reference_fine", False),
        ]
        if data["index"] == 1:
            pairs.extend(
                ("single_" + level, "native_" + level, "single_spm_" + level, True)
                for level in ("coarse", "fine")
            )
        if data["index"] == 2:
            pairs.extend(
                (impl + "_old_" + level, impl + "_" + level, "old_" + impl + "_" + level, True)
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
                    a, b = (routes[k]["segments"][-1]["end_time_s"] for k in (left, right))
                    comparisons[name].update(
                        event_time_error_s=abs(a - b), event_time_limit_s=2 + 0.005 * abs(b)
                    )
                    checks[name + ":event_time"] = abs(a - b) <= 2 + 0.005 * abs(b)
            checks[name] = comparisons[name]["passed"]
    item["failures"] = [k for k, v in checks.items() if not v]
    item["passed"] = bool(checks and not item["failures"])
