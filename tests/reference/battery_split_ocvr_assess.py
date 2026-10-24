"""Fixed complete-history, analytic, external-reference and analysis criteria."""

import numpy as np
from battery_split_ocvr_cases import ATOL, REFUSALS, RESPONSES
from battery_split_ocvr_oracle import cc_work, cv_at, eis, inventories, response, soh
from numpy.polynomial import Polynomial


def passes(value):
    return (
        bool(value)
        if isinstance(value, (bool, np.bool_))
        else bool(np.isfinite(value) and value <= 1)
    )


def ratio(a, b, atol, rtol=0.0):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(abs(a - b) / (atol + rtol * np.maximum(abs(a), abs(b)))))


def event_kind(value):
    mapping = {
        None: "complete",
        "final time": "complete",
        "battery:minimum-voltage": "minimum-voltage",
        "event: Minimum voltage [V]": "minimum-voltage",
        "battery:maximum-voltage": "maximum-voltage",
        "event: Maximum voltage [V]": "maximum-voltage",
    }
    for electrode in ("negative", "positive"):
        for bound in ("minimum", "maximum"):
            canonical = f"{electrode}-{bound}-fraction"
            mapping[f"split-ocvr:{electrode}:{bound}-fraction"] = canonical
            mapping[f"event: {bound.title()} {electrode.title()} Electrode stoichiometry"] = (
                canonical
            )
    return mapping.get(value, "unknown:" + str(value))


def history_contract(data, trace, implementation):
    if "exception" in trace:
        return {"route_completed": False}
    segments = trace.get("segments", [])
    checks = {
        "expected_segments": len(segments) == len(data["waveform"]),
        "nonempty": bool(segments),
    }
    start = 0.0
    for i, s in enumerate(segments):
        if i >= len(data["waveform"]):
            checks[f"{i}:authored_segment"] = False
            continue
        t, v = np.asarray(s.get("times_s", [])), np.asarray(s.get("values", []))
        shape = t.ndim == 1 and len(t) > 0 and v.shape == (len(t), len(RESPONSES))
        checks[f"{i}:shape"] = shape
        if not shape:
            continue
        checks[f"{i}:finite"] = bool(np.all(np.isfinite(t)) and np.all(np.isfinite(v)))
        checks[f"{i}:monotone"] = bool(np.all(np.diff(t) > 0))
        checks[f"{i}:start"] = abs(t[0] - start) <= 1e-7
        checks[f"{i}:reported_end"] = abs(t[-1] - s.get("end_time_s", np.nan)) <= 1e-7
        end = start + data["observation_horizons_s"][i]
        limit = 1e-7 if data["expected"] == "complete" else 1e-3 + 2e-7 * end
        checks[f"{i}:complete_horizon"] = abs(t[-1] - end) / limit
        samples = np.linspace(start, end, data["samples_per_segment"])
        checks[f"{i}:observations"] = all(
            np.min(abs(t - x)) <= 1e-7 for x in samples if x < t[-1] - limit
        )
        expected = data["expected"] if i == len(data["waveform"]) - 1 else "complete"
        checks[f"{i}:event"] = event_kind(s.get("termination")) == expected
        if implementation == "native":
            checks[f"{i}:outcome"] = (
                expected == "complete"
                and s.get("outcome") == "success"
                and s.get("termination") is None
            ) or (
                expected != "complete"
                and s.get("outcome") == "stopped"
                and event_kind(s.get("termination")) == expected
            )
            for key, expected_shape in (("amounts_mol", (len(t), 2)), ("work_J", (len(t),))):
                arr = np.asarray(s.get(key, []))
                checks[f"{i}:{key}_raw"] = arr.shape == expected_shape and bool(
                    np.all(np.isfinite(arr))
                )
        else:
            raw = np.asarray(s.get("original_source_state_coordinates", []))
            checks[f"{i}:source_raw"] = raw.shape == (3, len(t)) and bool(np.all(np.isfinite(raw)))
        start += data["waveform"][i][0]
    if (
        segments
        and np.asarray(segments[0].get("values", [])).ndim == 2
        and len(segments[0]["values"])
    ):
        first = segments[0]
        checks["original_t0"] = first["times_s"][0] == 0.0
        checks["initial_Q"] = abs(first["values"][0][2]) <= 1e-12
        checks["initial_fractions"] = ratio(
            first["values"][0][3:5],
            [data["initial_negative_fraction"], data["initial_positive_fraction"]],
            1e-12,
        )
        if implementation == "native" and len(first.get("work_J", [])):
            checks["initial_W"] = abs(first["work_J"][0]) <= 1e-12
            checks["initial_inventory"] = ratio(
                first["amounts_mol"][0], inventories(data, 0.0), 1e-12
            )
    return checks


def invariants(data, trace, implementation):
    checks = history_contract(data, trace, implementation)
    if not all(passes(v) for v in checks.values()):
        return checks
    q0 = w0 = 0.0
    for i, s in enumerate(trace["segments"]):
        t, v = np.asarray(s["times_s"]), np.asarray(s["values"])
        _duration, target, unit = data["waveform"][i]
        expected, works = [], []
        for elapsed in t - t[0]:
            if unit == "A":
                q, current = q0 + target * elapsed, target
                work = cc_work(data, q0, q, current)
            else:
                q, current, work, _ = cv_at(data, q0, target, elapsed)
            expected.append(response(data, q, current))
            works.append(w0 + work)
        expected = np.asarray(expected)
        for j, name in enumerate(RESPONSES):
            checks[f"{i}:independent_{name}"] = ratio(v[:, j], expected[:, j], ATOL[j], 1e-7)
        checks[f"{i}:source_capacity_Q_closure"] = max(
            ratio(
                data["negative_capacity_C"] * (data["initial_negative_fraction"] - v[:, 3]),
                v[:, 2],
                1e-6,
            ),
            ratio(
                data["positive_capacity_C"] * (v[:, 4] - data["initial_positive_fraction"]),
                v[:, 2],
                1e-6,
            ),
        )
        if implementation == "native":
            nn = np.asarray(s["amounts_mol"])
            checks[f"{i}:true_Li_Q_closure"] = ratio(
                nn, [inventories(data, q) for q in v[:, 2]], 1e-9
            )
            checks[f"{i}:total_Li"] = ratio(nn.sum(axis=1), sum(inventories(data, 0.0)), 1e-9)
            checks[f"{i}:independent_work_J"] = ratio(s["work_J"], works, 1e-3, 1e-7)
        if unit == "V":
            limited = abs(v[:, 1]) >= data["maximum_current_A"] - 1e-7
            if data["name"] == "interior-CV":
                checks[f"{i}:only_interior"] = not bool(np.any(limited))
            else:
                checks[f"{i}:both_regimes"] = bool(np.any(limited) and np.any(~limited))
            if np.any(~limited):
                checks[f"{i}:CV_voltage"] = ratio(v[~limited, 0], target, 1e-7)
            if np.any(limited):
                checks[f"{i}:limited_current"] = ratio(
                    abs(v[limited, 1]), data["maximum_current_A"], 1e-7
                )
        q0, w0 = float(expected[-1, 2]), works[-1]
    final = np.asarray(trace["segments"][-1]["values"])[-1]
    event = data["expected"]
    if event.endswith("voltage"):
        checks["cutoff_voltage"] = ratio(final[0], data[event.replace("-", "_") + "_V"], 1e-6)
    elif event.endswith("fraction"):
        electrode = event.split("-")[0]
        checks["cutoff_fraction"] = ratio(
            final[3 if electrode == "negative" else 4], data[event.replace("-", "_")], 1e-8
        )
    if data["name"] == "round-trip":
        checks["cycle_Q"] = abs(final[2]) / 1e-4
        if implementation == "native":
            checks["cycle_W"] = ratio(trace["segments"][-1]["work_J"][-1], -30.0, 1e-3)
    return checks


def compare(a, b):
    if "exception" in a or "exception" in b:
        return {"route_completed": False}
    aa, bb = a.get("segments", []), b.get("segments", [])
    checks = {"segment_count": len(aa) == len(bb), "nonempty": bool(aa) and bool(bb)}
    for i, (sa, sb) in enumerate(zip(aa, bb, strict=False)):
        ta, tb = np.asarray(sa["times_s"]), np.asarray(sb["times_s"])
        lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
        times = np.unique(np.r_[ta[(ta >= lo) & (ta <= hi)], tb[(tb >= lo) & (tb <= hi)]])
        if not len(times):
            checks[f"{i}:shared_interval"] = False
            continue
        va, vb = np.asarray(sa["values"]), np.asarray(sb["values"])
        for j, name in enumerate(RESPONSES):
            checks[f"{i}:{name}"] = ratio(
                np.interp(times, ta, va[:, j]), np.interp(times, tb, vb[:, j]), ATOL[j], 1e-7
            )
        checks[f"{i}:endpoint"] = abs(ta[-1] - tb[-1]) / (1e-3 + 2e-7 * max(ta[-1], tb[-1]))
        if "work_J" in sa and "work_J" in sb:
            checks[f"{i}:work_J"] = ratio(
                np.interp(times, ta, sa["work_J"]), np.interp(times, tb, sb["work_J"]), 1e-3, 1e-7
            )
    return checks


def analysis_checks(trace):
    if "exception" in trace:
        return {"route_completed": False}
    data = trace["inputs"]
    checks = {"read_only": trace["before"] == trace["after"]}
    if trace["kind"] == "EIS":
        freq = np.geomspace(1e-4, 1e2, 17)
        checks["complete_frequencies"] = np.array_equal(trace["frequencies_Hz"], freq)
        points = trace["result"]["points"]
        checks["complete_points"] = len(points) == 17
        checks["success"] = trace["result"]["success"] and all(p["success"] for p in points)
        if checks["complete_points"] and checks["success"]:
            z = np.asarray([complex(p["real_ohm"], p["imag_ohm"]) for p in points])
            checks["independent_complex_Z"] = ratio(
                z, eis(data, data["analysis_charge_C"], freq), 1e-8, 1e-7
            )
    else:
        expected = soh(data)
        checks["unique"] = trace["status"] == "unique"
        if not checks["unique"]:
            return checks
        for k, v in expected.items():
            checks["independent_" + k] = ratio(
                trace["branch"][k], v, 1e-7 if k[0] in "xy" else 1e-4, 0 if k[0] in "xy" else 1e-7
            )
        checks["soc_count"] = len(trace["soc_values"]) == 5
        for i, (s, actual) in enumerate(
            zip((0, 0.25, 0.5, 0.75, 1), trace["soc_values"], strict=False)
        ):
            x = expected["x_0"] + s * (expected["x_100"] - expected["x_0"])
            y = expected["y_0"] + s * (expected["y_100"] - expected["y_0"])
            voltage = Polynomial(data["positive_polynomial_V"])(y) - Polynomial(
                data["negative_polynomial_V"]
            )(x)
            checks[f"{i}:SOC"] = actual["soc"] == s
            checks[f"{i}:fractions"] = ratio([actual["x"], actual["y"]], [x, y], 1e-7)
            checks[f"{i}:voltage"] = ratio(actual["voltage_V"], voltage, 1e-6)
    return checks


def assess(item):
    data, routes = item["inputs"], item["routes"]
    index = int(item["id"][1:])
    if index == 15:
        checks = {
            "refusals": {name: routes.get(name, {}).get("rejected", False) for name in REFUSALS}
        }
    elif index in (16, 17):
        names = ("base", "zero-R", "nonlinear") if index == 16 else ("base", "shifted", "nonlinear")
        checks = {
            name: analysis_checks(routes.get(name, {"exception": "missing route"}))
            for name in names
        }
        if index == 17 and all(
            routes.get(n, {}).get("status") == "unique" for n in ("base", "shifted")
        ):
            checks["unchanged_total_Li"] = {
                k: ratio(routes["base"]["branch"][k], routes["shifted"]["branch"][k], 1e-7)
                for k in ("x_0", "x_100", "y_0", "y_100")
            }
    else:
        checks = {}
        implementations = ("native", "reference") if index <= 12 else ("native",)
        for implementation in implementations:
            for precision in ("coarse", "fine"):
                key = implementation + "_" + precision
                checks[key] = invariants(
                    data, routes.get(key, {"exception": "missing route"}), implementation
                )
            checks[implementation + "_precision"] = compare(
                routes.get(implementation + "_coarse", {"exception": "missing"}),
                routes.get(implementation + "_fine", {"exception": "missing"}),
            )
        if index <= 12:
            for precision in ("coarse", "fine"):
                checks["paired_" + precision] = compare(
                    routes.get("native_" + precision, {"exception": "missing"}),
                    routes.get("reference_" + precision, {"exception": "missing"}),
                )
    item["checks"] = checks
    item["failures"] = [
        {"group": group, "check": name, "value": value}
        for group, values in checks.items()
        for name, value in values.items()
        if not passes(value)
    ]
    item["passed"] = not item["failures"]
