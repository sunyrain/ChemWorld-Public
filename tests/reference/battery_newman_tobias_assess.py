"""Fixed response/field comparisons and independently evaluated physical balances."""

import numpy as np
from battery_newman_tobias_cases import (
    FIELD_TOLERANCES,
    REFUSALS,
    RESPONSE_ATOL,
    RESPONSE_RTOL,
    RESPONSES,
    SOURCE_F,
    F,
    R,
)


def event_kind(value):
    if value is None or value == "final time":
        return "complete"
    value = value.lower().replace("event: ", "").replace(" [v]", "")
    if "minimum-voltage" in value or "minimum voltage" in value:
        return "minimum-voltage"
    return value


def native_outcome_valid(outcome, termination):
    """CoupledSystem returns exactly success or stopped, with a real stop event."""
    return (outcome == "success" and termination is None) or (
        outcome == "stopped" and termination == "battery:minimum-voltage"
    )


def aligned(first, second, first_values, second_values):
    ta, tb = np.asarray(first["times_s"]), np.asarray(second["times_s"])
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    times = np.unique(np.r_[ta[(ta >= lo) & (ta <= hi)], tb[(tb >= lo) & (tb <= hi)]])
    if not len(times):
        raise ValueError("no shared time range")
    a, b = np.asarray(first_values), np.asarray(second_values)
    return (
        np.column_stack([np.interp(times, ta, a[:, j]) for j in range(a.shape[1])]),
        np.column_stack([np.interp(times, tb, b[:, j]) for j in range(b.shape[1])]),
    )


def ratio(a, b, atol, rtol):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(abs(a - b) / (atol + rtol * np.maximum(abs(a), abs(b)))))


def compare(first, second):
    result = {}
    if "exception" in first or "exception" in second:
        return {"route_completed": False}
    aa, bb = first.get("segments", []), second.get("segments", [])
    result["nonempty_histories"] = bool(aa) and bool(bb)
    result["segment_count"] = len(aa) == len(bb)
    for k, (a, b) in enumerate(zip(aa, bb, strict=False)):
        av, bv = aligned(a, b, a["values"], b["values"])
        for j, name in enumerate(RESPONSES):
            result[f"segment{k}:{name}"] = ratio(
                av[:, j], bv[:, j], RESPONSE_ATOL[j], RESPONSE_RTOL[j]
            )
        for region in ("negative", "separator", "positive"):
            for name in a["sampled_profiles"][region]:
                ap, bp = aligned(
                    a, b, a["sampled_profiles"][region][name], b["sampled_profiles"][region][name]
                )
                kind = (
                    "potential_V"
                    if "potential" in name
                    else "reaction_density_A"
                    if "reaction" in name
                    else "face_current_A"
                )
                result[f"segment{k}:{region}:{name}"] = ratio(ap, bp, *FIELD_TOLERANCES[kind])
        result[f"segment{k}:event_kind"] = event_kind(a["termination"]) == event_kind(
            b["termination"]
        )
        result[f"segment{k}:stop_time"] = abs(a["end_time_s"] - b["end_time_s"]) / (
            0.5 + 0.005 * max(a["end_time_s"], b["end_time_s"])
        )
    return result


def history_contract(data, trace):
    """Reject missing/truncated histories before response intersection comparison."""
    if "exception" in trace:
        return {"route_completed": False}
    segments = trace.get("segments", [])
    checks = {
        "nonempty_history": bool(segments),
        "expected_segment_count": len(segments) == len(data["waveform"]),
    }
    if not segments:
        return checks
    cumulative = 0.0
    for k, segment in enumerate(segments):
        if k >= len(data["waveform"]):
            checks[f"{k}:expected_segment"] = False
            continue
        duration = data["waveform"][k][0]
        times = np.asarray(segment.get("times_s", []), dtype=float)
        values = np.asarray(segment.get("values", []), dtype=float)
        shape_ok = (
            times.ndim == 1 and len(times) > 0 and values.shape == (len(times), len(RESPONSES))
        )
        checks[f"{k}:history_shape"] = shape_ok
        if not shape_ok:
            cumulative += duration
            continue
        checks[f"{k}:finite_times_and_values"] = bool(
            np.all(np.isfinite(times)) and np.all(np.isfinite(values))
        )
        checks[f"{k}:ordered_times"] = bool(np.all(np.diff(times) > 0))
        checks[f"{k}:requested_start"] = abs(times[0] - cumulative) <= 1e-7
        checks[f"{k}:reported_end_matches_history"] = (
            abs(segment.get("end_time_s", float("nan")) - times[-1]) <= 1e-7
        )
        cumulative += duration
        cutoff_last = data["expected"] != "complete" and k == len(data["waveform"]) - 1
        checks[f"{k}:requested_end"] = (
            times[0] <= times[-1] <= cumulative + 1e-7
            if cutoff_last
            else abs(times[-1] - cumulative) <= 1e-7
        )
        for kind in ("raw_profiles", "sampled_profiles"):
            profiles = segment.get(kind, {})
            checks[f"{k}:{kind}:all_regions"] = set(profiles) == {
                "negative",
                "separator",
                "positive",
            }
            for region in ("negative", "separator", "positive"):
                raw = profiles.get(region, {})
                required = {"electrolyte_potential_V", "electrolyte_face_current_A"}
                if kind == "raw_profiles":
                    required |= {"nodes_m", "edges_m"}
                if region != "separator":
                    required |= {"solid_potential_V", "solid_face_current_A"}
                    required |= (
                        {"reaction_current_A", "interfacial_current_density_A_m2"}
                        if kind == "raw_profiles"
                        else {"reaction_density_A"}
                    )
                checks[f"{k}:{kind}:{region}:required_fields"] = required <= set(raw)
                for name, values_array in raw.items():
                    array = np.asarray(values_array, dtype=float)
                    checks[f"{k}:{kind}:{region}:{name}:finite"] = bool(np.all(np.isfinite(array)))
                    checks[f"{k}:{kind}:{region}:{name}:shape"] = (
                        array.ndim == 1 and len(array) > 0
                        if name in {"nodes_m", "edges_m"}
                        else array.ndim == 2 and array.shape[0] == len(times) and array.shape[1] > 0
                    )
    first_times = segments[0].get("times_s", [])
    checks["initial_t0_retained"] = bool(first_times) and first_times[0] == 0.0
    if (
        np.asarray(segments[0].get("values", [])).shape == (len(first_times), len(RESPONSES))
        and first_times
    ):
        first = np.asarray(segments[0]["values"])[0]
        checks["initial_charge_zero"] = abs(first[2]) <= 1e-12
        checks["initial_work_zero"] = abs(first[3]) <= 1e-12
        for index, electrode in enumerate(data["electrodes"]):
            authored = (
                electrode["active_volume_m3"]
                * electrode["maximum_concentration_mol_m3"]
                * electrode["initial_fraction"]
            )
            checks[f"initial_{electrode['name']}_authored_Li"] = ratio(
                first[6 + index], authored, RESPONSE_ATOL[6 + index], RESPONSE_RTOL[6 + index]
            )
    return checks


def invariants(data, trace, implementation):
    checks = history_contract(data, trace)
    if not all(passes(value) for value in checks.values()):
        return checks
    first = np.asarray(trace["segments"][0]["values"])[0]
    f = F if implementation == "native" else SOURCE_F
    nonuniform_range = 0.0
    for k, segment in enumerate(trace["segments"]):
        v = np.asarray(segment["values"])
        checks[f"{k}:finite"] = bool(np.all(np.isfinite(v)))
        checks[f"{k}:positive_inventory"] = bool(np.all(v[:, 6:8] > 0))
        checks[f"{k}:Li"] = float(np.max(abs(v[:, 6] + v[:, 7] - first[6] - first[7]))) / 2e-9
        checks[f"{k}:negative_Faraday"] = (
            float(np.max(abs(f * (v[:, 6] - first[6]) + v[:, 2] - first[2]))) / 2e-4
        )
        checks[f"{k}:positive_Faraday"] = (
            float(np.max(abs(f * (v[:, 7] - first[7]) - v[:, 2] + first[2]))) / 2e-4
        )
        for name, sign in (("negative", 1), ("positive", -1)):
            raw = segment["raw_profiles"][name]
            q, solid, liquid = (
                np.asarray(raw[key])
                for key in (
                    "reaction_current_A",
                    "solid_face_current_A",
                    "electrolyte_face_current_A",
                )
            )
            checks[f"{k}:{name}:integrated_current"] = (
                float(np.max(abs(q.sum(axis=1) - sign * v[:, 1]))) / 2e-5
            )
            checks[f"{k}:{name}:liquid_divergence"] = (
                float(np.max(abs(np.diff(liquid, axis=1) - q))) / 2e-5
            )
            checks[f"{k}:{name}:solid_divergence"] = (
                float(np.max(abs(np.diff(solid, axis=1) + q))) / 2e-5
            )
            checks[f"{k}:{name}:total_face_current"] = (
                float(np.max(abs(solid + liquid - v[:, 1, None]))) / 2e-5
            )
            nonuniform_range = max(nonuniform_range, float(np.max(np.ptp(q * q.shape[1], axis=1))))
        sep = np.asarray(segment["raw_profiles"]["separator"]["electrolyte_face_current_A"])
        checks[f"{k}:separator_current"] = float(np.max(abs(sep - v[:, 1, None]))) / 2e-5
        if data.get("control_expectation") == "interior":
            checks[f"{k}:CV_voltage"] = float(np.max(abs(v[:, 0] - data["waveform"][k][1]))) / 1e-7
            checks[f"{k}:CV_interior"] = bool(
                np.all(abs(v[:, 1]) < data["maximum_current_A"] - 1e-6)
            )
        elif data.get("control_expectation") == "both_limits":
            target = data["maximum_current_A"] * (1 if k == 0 else -1)
            checks[f"{k}:CV_limit"] = float(np.max(abs(v[:, 1] - target))) / 1e-7
        if data["name"] == "rest-exact-limit":
            expected = sum(
                a * data["electrodes"][1]["initial_fraction"] ** j
                for j, a in enumerate(data["electrodes"][1]["polynomial"])
            ) - sum(
                a * data["electrodes"][0]["initial_fraction"] ** j
                for j, a in enumerate(data["electrodes"][0]["polynomial"])
            )
            checks[f"{k}:rest_voltage"] = float(np.max(abs(v[:, 0] - expected))) / 1e-7
            checks[f"{k}:rest_current"] = float(np.max(abs(v[:, 1]))) / 1e-7
        if data.get("analytic_BV_limit"):
            un, up, eta = [], [], []
            for index, e in enumerate(data["electrodes"]):
                theta = v[:, 4 + index]
                u = sum(a * theta**j for j, a in enumerate(e["polynomial"]))
                (un if index == 0 else up).extend(u.tolist())
                j0 = e["exchange_current_scale_A_m2"] * np.sqrt(
                    theta
                    * (1 - theta)
                    * data["electrolyte_concentration_mol_m3"]
                    / data["reference_electrolyte_concentration_mol_m3"]
                )
                active_area = 3 * e["active_volume_m3"] / e["radius_m"]
                eta.append(
                    2
                    * R
                    * data["temperature_K"]
                    / f
                    * np.arcsinh((1 if index == 0 else -1) * v[:, 1] / (2 * active_area * j0))
                )
            ideal = np.asarray(up) - np.asarray(un) + eta[1] - eta[0]
            checks[f"{k}:uniform_BV_limit"] = float(np.max(abs(v[:, 0] - ideal))) / 0.002
        if implementation == "native":
            checks[f"{k}:outcome"] = native_outcome_valid(
                segment.get("outcome"), segment["termination"]
            )
    if data.get("nonuniform_current"):
        checks["actual_nonuniform_current"] = nonuniform_range > 1e-4
    checks["expected_event"] = event_kind(trace["segments"][-1]["termination"]) == data["expected"]
    if data["expected"] == "complete":
        checks["whole_requested_duration"] = (
            abs(trace["segments"][-1]["end_time_s"] - sum(part[0] for part in data["waveform"]))
            <= 1e-7
        )
    if data["expected"] == "minimum-voltage":
        checks["cutoff_voltage"] = (
            abs(trace["segments"][-1]["values"][-1][0] - data["minimum_voltage_V"]) / 2e-5
        )
    return checks


def passes(value):
    return (
        bool(value)
        if isinstance(value, (bool, np.bool_))
        else bool(np.isfinite(value) and value <= 1.0)
    )


def assess(item):
    data, routes = item["inputs"], item["routes"]
    if data["expected"] == "refusals":
        checks = {key: routes.get(key, {}).get("rejected", False) for key in REFUSALS}
        item["checks"] = {"refusals": checks}
    else:
        expected = ("native_coarse", "native_fine", "reference_coarse", "reference_fine")
        present = {
            name: routes.get(name, {"exception": "required route missing"}) for name in expected
        }
        item["checks"] = {
            **{
                name: invariants(data, route, name.split("_")[0]) for name, route in present.items()
            },
            "paired_coarse": compare(present["native_coarse"], present["reference_coarse"]),
            "paired_fine": compare(present["native_fine"], present["reference_fine"]),
            "native_grid": compare(present["native_coarse"], present["native_fine"]),
            "reference_grid": compare(present["reference_coarse"], present["reference_fine"]),
        }
    item["failures"] = [
        {"group": group, "check": name, "value": value}
        for group, values in item["checks"].items()
        for name, value in values.items()
        if not passes(value)
    ]
    required = 8 if data["expected"] == "refusals" else 4
    if len(routes) != required:
        item["failures"].append(
            {
                "group": "denominator",
                "check": "all_routes_present",
                "value": len(routes),
                "expected": required,
            }
        )
    item["passed"] = not item["failures"]
