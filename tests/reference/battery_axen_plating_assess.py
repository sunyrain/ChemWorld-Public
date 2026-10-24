"""Fixed completeness, balances and source comparisons; no selective qualification."""

import numpy as np
from battery_axen_plating_cases import GRIDS, active_electrodes
from battery_axen_plating_traces import history_metrics
from battery_partial_plating_assess import compare
from battery_plating_cases import PROFILE_RADII, RESPONSES
from scipy.integrate import simpson


def termination(segment):
    raw = segment["termination"]
    return None if raw in (None, "final time") else str(raw).removeprefix("event: ")


def finite_record(value):
    """Diagnostic text/None is allowed, every stored numeric value must be finite."""
    if isinstance(value, dict):
        return all(finite_record(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_record(v) for v in value)
    if isinstance(value, (float, int, np.number)):
        return bool(np.isfinite(value))
    return value is None or isinstance(value, (str, bool))


def route_binding(name, trace, data):
    """Bind retained trace metadata to its authored route, not its own assertions."""
    baseline = name.startswith("old_")
    key = name.removeprefix("old_")
    variants = {
        f"{impl}_{level}": (impl, cells)
        for impl in ("native", "reference")
        for level, cells in zip(("coarse", "fine"), GRIDS[impl], strict=True)
    }
    if key not in variants or (baseline and data["index"] not in (1, 2)):
        return {"route_binding": False}
    impl, cells = variants[key]
    expected = (
        "native SPM Axen/plating" if impl == "native" else "actual PyBaMM25.4.2 SPM Axen/plating"
    )
    return {
        "route_binding": bool(
            trace.get("implementation") == expected
            and trace.get("radial_cells") == cells
            and trace.get("baseline") is baseline
            and trace.get("has_plating") is (not (baseline and data["index"] == 1))
            and tuple(trace.get("h_electrodes", ())) == active_electrodes(data, baseline=baseline)
        )
    }


def integrity(trace, data):
    try:
        return _integrity(trace, data)
    except (KeyError, TypeError, ValueError, IndexError):
        return {"well_formed_required_history": False}


def _integrity(trace, data):
    checks = {"no_exception": "exception" not in trace, "has_segments": bool(trace.get("segments"))}
    implementations = {
        "native SPM Axen/plating": "native",
        "actual PyBaMM25.4.2 SPM Axen/plating": "reference",
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
        "baseline",
        "h_electrodes",
        "has_plating",
    }
    checks["required_trace_fields"] = required <= trace.keys()
    if not checks["implementation"] or not checks["required_trace_fields"]:
        return checks
    is_native = implementations[trace["implementation"]] == "native"
    checks["declared_response_order"] = tuple(trace["response_order"]) == tuple(RESPONSES)
    expected_keys = {
        f"{name}:Li:{i}" for name in ("negative", "positive") for i in range(trace["radial_cells"])
    } | ({"plating:Li", "dead:Li"} if trace["has_plating"] else set())
    expected_h = active_electrodes(data, baseline=trace["baseline"])
    checks["h_electrode_set"] = tuple(trace["h_electrodes"]) == expected_h
    checks["plating_presence"] = trace["has_plating"] == (
        not (trace["baseline"] and data["index"] == 1)
    )
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
            "hysteresis",
        }
        if is_native:
            required_segment |= {
                "outcome",
                "raw_amounts_mol",
                "raw_coordinates",
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
        checks[prefix + "h_keys"] = set(s["hysteresis"]) == set(expected_h)
        for name in expected_h:
            saved = s["hysteresis"][name]
            required_h = {"h", "h-rate"} | (set() if is_native else {"raw-h", "raw-h-rate"})
            checks[prefix + name + ":required_h"] = required_h <= saved.keys()
            for field in required_h:
                v = np.asarray(saved[field])
                checks[prefix + name + ":" + field + ":shape_finite"] = bool(
                    v.shape == ((5, len(t)) if field.startswith("raw") else t.shape)
                    and np.isfinite(v).all()
                )
            if index == 0:
                checks[prefix + name + ":authored_initial_h"] = bool(
                    abs(saved["h"][0] - data[name + "_initial_h"]) <= 1e-12
                )
        if is_native:
            coordinate_keys = {"discharged_charge_C", "delivered_work_J"} | {
                f"battery:{e}:hysteresis" for e in expected_h
            }
            checks[prefix + "raw_coordinates_shape_finite"] = len(s["raw_coordinates"]) == len(
                t
            ) and all(
                set(v) == coordinate_keys and all(np.isfinite(x) for x in v.values())
                for v in s["raw_coordinates"]
            )
            raw_qw = np.array(
                [[v["discharged_charge_C"], v["delivered_work_J"]] for v in s["raw_coordinates"]]
            )
            checks[prefix + "raw_charge_identity"] = bool(
                np.max(abs(raw_qw[:, 0] - rows[:, 2])) <= 1e-5
            )
            checks[prefix + "raw_work_identity"] = bool(
                np.max(abs(raw_qw[:, 1] - rows[:, 3])) <= 1e-4
            )
            for name in expected_h:
                checks[prefix + name + ":raw_h_identity"] = bool(
                    np.max(
                        abs(
                            np.array(
                                [v[f"battery:{name}:hysteresis"] for v in s["raw_coordinates"]]
                            )
                            - np.asarray(s["hysteresis"][name]["h"])
                        )
                    )
                    <= 1e-12
                )
            checks[prefix + "native_diagnostic_types"] = all(
                isinstance(s[k], dict)
                for k in ("solver", "conservation_residuals", "roundoff_corrections_mol")
            )
            required_solver = {
                "solver_calls",
                "numerical_form",
                "residual",
                "rhs_boundary_extensions_mol",
                "inventory_scales_mol",
                "accepted_nonnegative_relative_tolerance",
            }
            checks[prefix + "native_solver_required"] = bool(
                isinstance(s["solver"], dict)
                and required_solver <= s["solver"].keys()
                and isinstance(s["solver"].get("residual"), dict)
                and bool(s["solver"].get("residual"))
            )
            checks[prefix + "native_diagnostics_finite"] = all(
                finite_record(s[k])
                for k in ("solver", "conservation_residuals", "roundoff_corrections_mol")
            )
            checks[prefix + "raw_amounts_keys"] = len(s["raw_amounts_mol"]) == len(t) and all(
                set(r) == expected_keys for r in s["raw_amounts_mol"]
            )
            checks[prefix + "raw_amounts_finite"] = bool(
                len(s["raw_amounts_mol"]) == len(t)
                and all(all(np.isfinite(v) for v in r.values()) for r in s["raw_amounts_mol"])
            )
            shell_weights = np.diff(np.linspace(0.0, 1.0, trace["radial_cells"] + 1) ** 3)
            for j, e in enumerate(data["electrodes"]):
                actual = np.array(
                    [
                        [v[f"{e['name']}:Li:{k}"] for k in range(trace["radial_cells"])]
                        for v in s["raw_amounts_mol"]
                    ]
                )
                expected_amount = (
                    raw[j]
                    * shell_weights[None, :]
                    * e["active_volume_m3"]
                    * e["maximum_concentration_mol_m3"]
                )
                checks[prefix + e["name"] + ":shell_inventory_identity"] = bool(
                    np.max(abs(actual - expected_amount)) <= 1e-9
                )
                checks[prefix + e["name"] + ":total_inventory_identity"] = bool(
                    np.max(abs(actual.sum(axis=1) - rows[:, 12 + 6 * j])) <= 1e-9
                )
            for k, column in (("plating:Li", 5), ("dead:Li", 6)):
                actual = np.array([v.get(k, 0.0) for v in s["raw_amounts_mol"]])
                checks[prefix + k + ":inventory_identity"] = bool(
                    np.max(abs(actual - rows[:, column])) <= 1e-9
                )
        else:
            scales = trace["source_state_scales"]
            expected_scales = {
                e.capitalize() + " electrode hysteresis state": 1.0 for e in expected_h
            }
            if trace["has_plating"]:
                expected_scales.update(
                    {
                        "X-averaged negative lithium plating concentration [mol.m-3]": 1000.0,
                        "X-averaged negative dead lithium concentration [mol.m-3]": 1000.0,
                    }
                )
            checks[prefix + "source_actual_scales"] = bool(scales == expected_scales)
            checks[prefix + "source_solver_metadata"] = trace.get("tolerances") == {
                "rtol": 1e-9,
                "atol": 1e-11,
                "root_tol": 1e-9,
            }
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


def axen_metrics(trace, data):
    maxima = dict.fromkeys(
        (
            "h_rate_limit_ratio",
            "h_integral_limit_ratio",
            "h_stage_jump",
            "raw_h_copy_error",
            "raw_rate_copy_error",
            "h_domain_excess",
            "OCP_mixture_error_V",
            "wrong_terminal_drive_rate_error_s_inv",
            "exact_zero_k_h_error",
        ),
        0.0,
    )
    for i, segment in enumerate(trace["segments"]):
        rows, t = np.asarray(segment["rows"]), np.asarray(segment["times_s"])
        for j, e in enumerate(data["electrodes"]):
            name, offset = e["name"], 9 + 6 * j
            theta = rows[:, offset + 1]
            mean = sum(c * theta**power for power, c in enumerate(e["polynomial"]))
            if name not in trace["h_electrodes"]:
                expected_U = mean
            else:
                saved = segment["hysteresis"][name]
                h, rate = np.asarray(saved["h"]), np.asarray(saved["h-rate"])
                q = rows[:, offset]
                capacity = (
                    trace["F_C_mol"] * e["maximum_concentration_mol_m3"] * e["active_volume_m3"]
                )
                decay = np.where(q >= 0, data["delithiation_decay"], data["lithiation_decay"])
                law = decay * np.abs(q) / capacity * (np.sign(q) - h)
                wrong_q = rows[:, 1] * (1 if j == 0 else -1)
                wrong = (
                    np.where(wrong_q >= 0, data["delithiation_decay"], data["lithiation_decay"])
                    * np.abs(wrong_q)
                    / capacity
                    * (np.sign(wrong_q) - h)
                )
                maxima["h_rate_limit_ratio"] = max(
                    maxima["h_rate_limit_ratio"],
                    float(np.max(abs(rate - law) / (1e-10 + 1e-7 * abs(law)))),
                )
                delta = h[-1] - h[0]
                maxima["h_integral_limit_ratio"] = max(
                    maxima["h_integral_limit_ratio"],
                    float(abs(delta - simpson(rate, x=t)) / (2e-4 + 2e-3 * abs(delta))),
                )
                maxima["wrong_terminal_drive_rate_error_s_inv"] = max(
                    maxima["wrong_terminal_drive_rate_error_s_inv"],
                    float(np.max(abs(rate - wrong))),
                )
                maxima["h_domain_excess"] = max(
                    maxima["h_domain_excess"], float(np.max(abs(h)) - 1)
                )
                if "raw-h" in saved:
                    maxima["raw_h_copy_error"] = max(
                        maxima["raw_h_copy_error"],
                        float(np.max(abs(np.asarray(saved["raw-h"]) - h))),
                    )
                    maxima["raw_rate_copy_error"] = max(
                        maxima["raw_rate_copy_error"],
                        float(np.max(abs(np.asarray(saved["raw-h-rate"]) - rate))),
                    )
                if i:
                    maxima["h_stage_jump"] = max(
                        maxima["h_stage_jump"],
                        abs(h[0] - trace["segments"][i - 1]["hysteresis"][name]["h"][-1]),
                    )
                if data["index"] == 1:
                    initial_h = (
                        data[name + "_initial_h"]
                        if i == 0
                        else trace["segments"][i - 1]["hysteresis"][name]["h"][-1]
                    )
                    scalar_q = data["waveform"][i][1] * (1 if j == 0 else -1)
                    k = data["delithiation_decay"] if scalar_q > 0 else data["lithiation_decay"]
                    exact = (
                        initial_h + np.zeros_like(t)
                        if scalar_q == 0
                        else np.sign(scalar_q)
                        + (initial_h - np.sign(scalar_q))
                        * np.exp(-k * abs(scalar_q) * (t - t[0]) / capacity)
                    )
                    maxima["exact_zero_k_h_error"] = max(
                        maxima["exact_zero_k_h_error"], float(np.max(abs(h - exact)))
                    )
                expected_U = mean + data[name + "_gap_V"] * h / 2
            maxima["OCP_mixture_error_V"] = max(
                maxima["OCP_mixture_error_V"], float(np.max(abs(rows[:, offset + 4] - expected_U)))
            )
    return maxima


def compare_h(left, right, *, old_limit=False):
    errors, ratios = [], []
    for a, b in zip(left["segments"], right["segments"], strict=True):
        ta, tb = np.asarray(a["times_s"]), np.asarray(b["times_s"])
        for name in set(left["h_electrodes"]) & set(right["h_electrodes"]):
            for i, t in enumerate(ta):
                match = np.flatnonzero(abs(tb - t) <= 1e-7)
                if match.size:
                    hv = b["hysteresis"][name]["h"][match[0]]
                    error = abs(a["hysteresis"][name]["h"][i] - hv)
                    errors.append(error)
                    ratios.append(error / (1e-7 if old_limit else 2e-4 + 1e-4 * abs(hv)))
    # A02 original single model has no h. The new h still passes independent laws.
    return {
        "compared_h_values": len(errors),
        "maximum_error": max(errors, default=0.0),
        "maximum_limit_ratio": max(ratios, default=0.0),
        "passed": bool(max(ratios, default=0.0) <= 1),
    }


def assess(item):
    data, routes = item["inputs"], item["routes"]
    checks, metrics, comparisons = {}, {}, {}
    item.update(checks=checks, metrics=metrics, comparisons=comparisons)
    if data["index"] == 13:
        from battery_axen_plating_cases import REFUSALS

        checks["route_names"] = set(routes) == set(REFUSALS)
        for name, route in routes.items():
            checks[name + ":rejected"] = route.get("rejected") is True
            checks[name + ":immutable"] = route.get("input_state_unchanged") is True
    else:
        expected = {
            f"{impl}_{level}" for impl in ("native", "reference") for level in ("coarse", "fine")
        }
        if data["index"] in (1, 2):
            expected |= {"old_" + key for key in expected}
        checks["route_names"] = set(routes) == expected
        valid = {}
        for name, route in routes.items():
            local = integrity(route, data)
            local.update(route_binding(name, route, data))
            checks.update({name + ":" + k: bool(v) for k, v in local.items()})
            valid[name] = all(local.values())
            if not valid[name]:
                continue
            values = history_metrics(route, data)
            values.update(axen_metrics(route, data))
            metrics[name] = values
            limits = {
                **dict.fromkeys(
                    (
                        "total_Li_error_mol",
                        "negative_Faraday_error_mol",
                        "positive_Faraday_error_mol",
                        "stage_inventory_jump_mol",
                        "stage_radial_inventory_jump_mol",
                        "stage_raw_inventory_jump_mol",
                    ),
                    1e-9,
                ),
                "algebraic_current_error_A": 1e-7,
                "independent_kinetic_limit_ratio": 1.0,
                "concentration_amount_error_mol": 1e-12,
                "work_quadrature_limit_ratio": 1.0,
                "CC_charge_error_C": 1e-5,
                "stage_charge_jump_C": 1e-5,
                "stage_work_jump_J": 1e-4,
                "h_rate_limit_ratio": 1.0,
                "h_integral_limit_ratio": 1.0,
                "h_stage_jump": 1e-10,
                "raw_h_copy_error": 1e-12,
                "raw_rate_copy_error": 1e-10,
                "h_domain_excess": 1e-10,
                "OCP_mixture_error_V": 1e-7,
            }
            if data["index"] == 1:
                limits["exact_zero_k_h_error"] = 2e-7
            checks.update({name + ":" + k: bool(values[k] <= limit) for k, limit in limits.items()})
            checks[name + ":nonnegative_metal"] = values["minimum_metal_mol"] >= -1e-12
            rows = np.asarray([r for s in route["segments"] for r in s["rows"]])
            checks[name + ":dead_constant"] = bool(
                np.max(abs(rows[:, 6] - data["initial_dead_mol"])) <= 1e-12
            )
            for i, segment in enumerate(route["segments"]):
                r = np.asarray(segment["rows"])
                if "sei_thickness_m" in segment:
                    checks[name + f":{i}:zero_SEI"] = bool(
                        np.all(np.asarray(segment["sei_thickness_m"]) == 0)
                    )
                if data["index"] == 9:
                    checks[name + f":{i}:CV"] = bool(
                        np.max(abs(r[:, 0] - data["waveform"][i][1])) <= 1e-7
                        and np.max(abs(r[:, 1])) < data["maximum_current_A"]
                    )
                if data["index"] == 10:
                    checks[name + f":{i}:limited_CV"] = bool(
                        np.max(abs(r[:, 1] - (-1 if i == 0 else 1) * data["maximum_current_A"]))
                        <= 1e-7
                    )
            if data["index"] == 3:
                checks[name + ":zero_metal_deposition"] = bool(rows[0, 4] < 0 and rows[-1, 5] > 0)
            if data["index"] == 6:
                h = route["segments"][0]["hysteresis"]["negative"]["h"]
                checks[name + ":internal_rest"] = bool(
                    rows[0, 4] > 1e-4
                    and rows[0, 9] * rows[0, 4] < 0
                    and np.max(abs(rows[:, 1:4])) <= 1e-10
                    and abs(h[-1] - h[0]) > 1e-5
                )
            if data["expected"].startswith("battery:"):
                error = (
                    abs(rows[-1, 10] - data["surface_margin"])
                    if data["index"] == 11
                    else abs(rows[-1, 0] - data["minimum_voltage_V"])
                )
                values["event_boundary_error"] = float(error)
                checks[name + ":event_boundary"] = bool(
                    error <= (1e-7 if data["index"] == 11 else 1e-6)
                )
        pairs = [
            ("paired_coarse", "native_coarse", "reference_coarse", False),
            ("paired_fine", "native_fine", "reference_fine", False),
            ("native_grid", "native_coarse", "native_fine", False),
            ("source_grid", "reference_coarse", "reference_fine", False),
        ]
        if data["index"] in (1, 2):
            pairs += [
                (f"{impl}_old_{level}", f"{impl}_{level}", f"old_{impl}_{level}", True)
                for impl in ("native", "reference")
                for level in ("coarse", "fine")
            ]
        for name, left, right, old in pairs:
            if not (valid.get(left) and valid.get(right)):
                comparisons[name] = {"passed": False, "reason": "required history incomplete"}
            else:
                result = compare(routes[left], routes[right], old_limit=old)
                result["h"] = compare_h(routes[left], routes[right], old_limit=old)
                end_left = routes[left]["segments"][-1]["end_time_s"]
                end_right = routes[right]["segments"][-1]["end_time_s"]
                result["event_time_error_s"] = abs(end_left - end_right)
                result["event_time_passed"] = abs(end_left - end_right) <= 1 + 0.002 * abs(
                    end_right
                )
                result["passed"] = bool(
                    result["passed"] and result["h"]["passed"] and result["event_time_passed"]
                )
                comparisons[name] = result
            checks["comparison:" + name] = comparisons[name]["passed"]
    item["failures"] = [k for k, v in checks.items() if not v]
    item["passed"] = bool(checks) and all(checks.values())
