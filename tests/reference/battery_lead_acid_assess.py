"""Fixed LOQS cross-implementation tests, history integrity and independent balances."""

import numpy as np
from battery_lead_acid_cases import (
    REFUSALS,
    RESPONSE_ATOL,
    RESPONSE_RTOL,
    RESPONSES,
    SOURCE_F,
    F,
)
from battery_lead_acid_oracle import amounts, cc_work, fields


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
    if value is None or value == "final time":
        return "complete"
    if value in ("battery:minimum-voltage", "event: Minimum voltage [V]"):
        return "minimum-voltage"
    if value == "lead-acid:minimum-acid":
        return "acid-threshold"
    return value.removeprefix("lead-acid:")


def observed_amounts(segment):
    if "amounts_mol" in segment:
        return np.asarray(segment["amounts_mol"])
    return np.asarray(segment["source_acid_water_amounts_mol"])


def history_contract(data, trace, implementation):
    if "exception" in trace:
        return {"route_completed": False}
    segments = trace.get("segments", [])
    checks = {
        "nonempty_history": bool(segments),
        "expected_segments": len(segments) == len(data["waveform"]),
    }
    if not segments:
        return checks
    cumulative = 0.0
    for index, segment in enumerate(segments):
        if index >= len(data["waveform"]):
            checks[f"{index}:authored_segment"] = False
            continue
        times = np.asarray(segment.get("times_s", []))
        values = np.asarray(segment.get("values", []))
        shape = times.ndim == 1 and len(times) > 0 and values.shape == (len(times), len(RESPONSES))
        checks[f"{index}:response_shape"] = shape
        if not shape:
            continue
        checks[f"{index}:finite"] = bool(np.all(np.isfinite(times)) and np.all(np.isfinite(values)))
        checks[f"{index}:monotone_times"] = bool(np.all(np.diff(times) > 0))
        checks[f"{index}:authored_start"] = abs(times[0] - cumulative) <= 1e-7
        checks[f"{index}:reported_end"] = (
            abs(times[-1] - segment.get("end_time_s", float("nan"))) <= 1e-7
        )
        expected_end = cumulative + data["observation_horizons_s"][index]
        stopped = data["expected"] != "complete"
        source_external = implementation == "reference" and data[
            "source_stop_semantics"
        ].startswith("author_imposed")
        limit = 1e-7 if not stopped or source_external else 1e-3 + 2e-7 * expected_end
        checks[f"{index}:complete_observation_horizon"] = abs(times[-1] - expected_end) / limit
        requested = np.linspace(cumulative, expected_end, data["samples_per_segment"])
        checks[f"{index}:all_interior_observations"] = all(
            np.min(abs(times - t)) <= 1e-7 for t in requested if t < times[-1] - limit
        )
        count = 6 if implementation == "native" else 2
        amounts_key = (
            "amounts_mol" if implementation == "native" else "source_acid_water_amounts_mol"
        )
        aa = np.asarray(segment.get(amounts_key, []))
        checks[f"{index}:amount_shape"] = aa.shape == (len(times), count)
        checks[f"{index}:finite_amounts"] = bool(np.all(np.isfinite(aa)))
        pore = np.asarray(segment.get("pore_volume_m3", []))
        checks[f"{index}:pore_shape"] = pore.shape == (len(times),)
        checks[f"{index}:finite_pore"] = bool(np.all(np.isfinite(pore)))
        if implementation == "reference":
            original = np.asarray(segment.get("original_source_state_coordinates", []))
            checks[f"{index}:original_source_states"] = original.ndim == 2 and original.shape[
                1
            ] == len(times)
            checks[f"{index}:finite_original_source_states"] = bool(np.all(np.isfinite(original)))
        cumulative += data["waveform"][index][0]
    first = segments[0]
    if np.asarray(first.get("values", [])).ndim == 2 and len(first["values"]):
        checks["raw_t0"] = first["times_s"][0] == 0.0
        checks["initial_Q_zero"] = abs(first["values"][0][2]) <= 1e-12
        checks["initial_W_zero"] = abs(first["values"][0][3]) <= 1e-12
        authored = np.asarray(data["initial_amounts_mol"])
        if implementation == "reference":
            authored = authored[4:]
        actual = observed_amounts(first)
        if actual.ndim == 2 and len(actual):
            checks["authored_initial_amounts"] = ratio(actual[0], authored, 1e-9, 2e-7)
    return checks


def invariants(data, trace, implementation):
    checks = history_contract(data, trace, implementation)
    if not all(passes(value) for value in checks.values()):
        return checks
    faraday = F if implementation == "native" else SOURCE_F
    initial = np.asarray(data["initial_amounts_mol"])
    element_matrix = np.array(
        [[1, 1, 1, 1, 0, 0], [0, 1, 0, 1, 1, 0], [0, 0, 0, 0, 2, 2], [0, 4, 2, 4, 4, 1]]
    )
    start_Q = start_W = 0.0
    for index, segment in enumerate(trace["segments"]):
        v = np.asarray(segment["values"])
        n = observed_amounts(segment)
        expected = np.asarray([amounts(data, q, faraday) for q in v[:, 2]])
        checks[f"{index}:nonnegative_inventory"] = bool(np.all(n >= -1e-10))
        checks[
            f"{index}:six_species_Q_closure"
            if implementation == "native"
            else f"{index}:source_acid_water_Q_closure"
        ] = ratio(n, expected if implementation == "native" else expected[:, 4:], 1e-9)
        if implementation == "native":
            checks[f"{index}:Pb_S_H_O_totals"] = ratio(
                n @ element_matrix.T, initial @ element_matrix.T, 2e-9
            )
        ve, vw = data["molar_volumes_m3_mol"][3:]
        liquid = n[:, 4:] if implementation == "native" else n
        checks[f"{index}:closed_liquid_volume"] = ratio(
            liquid[:, 0] * ve + liquid[:, 1] * vw, segment["pore_volume_m3"], 1e-12
        )
        analytical = [fields(data, q, current, faraday) for q, current in v[:, [2, 1]]]
        for column, key in (
            (0, "voltage_V"),
            (4, "acid_concentration_mol_m3"),
            (5, "water_concentration_mol_m3"),
            (6, "molality_mol_kg"),
            (10, "negative_ocp_V"),
            (11, "positive_ocp_V"),
            (12, "negative_overpotential_V"),
            (13, "positive_overpotential_V"),
            (14, "negative_j0_A_m2"),
            (15, "positive_j0_A_m2"),
        ):
            checks[f"{index}:independent_{key}"] = ratio(
                v[:, column],
                [r[key] for r in analytical],
                RESPONSE_ATOL[column],
                RESPONSE_RTOL[column],
            )
        checks[f"{index}:independent_porosity"] = ratio(
            v[:, 7:10], [r["porosities"] for r in analytical], 2e-7, 2e-6
        )
        _duration, target, unit = data["waveform"][index]
        times = np.asarray(segment["times_s"])
        if unit == "A":
            checks[f"{index}:CC_current"] = ratio(v[:, 1], target, 1e-7)
            expected_charge = start_Q + target * (times - times[0])
            checks[f"{index}:CC_charge"] = ratio(v[:, 2], expected_charge, 1e-4, 2e-7)
            work = [start_W + cc_work(data, start_Q, q, target, faraday) for q in v[:, 2]]
        elif data["control_expectation"] == "interior":
            checks[f"{index}:interior_CV_voltage"] = ratio(v[:, 0], target, 1e-7)
            checks[f"{index}:interior_CV_current"] = bool(
                np.all(abs(v[:, 1]) < data["maximum_current_A"] - 1e-6)
            )
            work = start_W + target * (v[:, 2] - start_Q)
        else:
            current = data["maximum_current_A"] * (1 if index == 0 else -1)
            checks[f"{index}:limited_CV_current"] = ratio(v[:, 1], current, 1e-7)
            work = [start_W + cc_work(data, start_Q, q, current, faraday) for q in v[:, 2]]
        checks[f"{index}:independent_terminal_work"] = ratio(v[:, 3], work, 3e-3, 2e-6)
        if index:
            checks[f"{index}:continuous_Q"] = abs(v[0, 2] - start_Q) / 1e-4
            checks[f"{index}:continuous_W"] = abs(v[0, 3] - start_W) / 3e-3
        start_Q, start_W = v[-1, 2:4]
        expected_event = data["expected"] if index == len(data["waveform"]) - 1 else "complete"
        if implementation == "reference" and data["source_stop_semantics"].startswith(
            "author_imposed"
        ):
            expected_event = "complete"
        checks[f"{index}:expected_event"] = event_kind(segment["termination"]) == expected_event
        if implementation == "native":
            checks[f"{index}:outcome"] = (
                segment.get("outcome") == "success"
                and expected_event == "complete"
                and segment["termination"] is None
            ) or (
                segment.get("outcome") == "stopped"
                and expected_event != "complete"
                and event_kind(segment["termination"]) == expected_event
            )
    final = np.asarray(trace["segments"][-1]["values"])[-1]
    if data["expected"] == "minimum-voltage":
        checks["cutoff_voltage"] = abs(final[0] - data["minimum_voltage_V"]) / 1e-6
    elif data["expected"] == "acid-threshold":
        checks["cutoff_acid"] = abs(final[4] - data["minimum_acid_concentration_mol_m3"]) / 1e-3
    elif data["expected"] == "negative:minimum-porosity":
        checks["cutoff_porosity"] = abs(final[7] - data["minimum_porosity"]) / 2e-7
    if data["name"] == "temperature-kinetics-sequence":
        checks["cycle_charge_zero"] = abs(final[2]) / 1e-4
        checks["cycle_dissipative_work"] = final[3] < 0
    return checks


def aligned(a, b, av, bv):
    ta, tb = np.asarray(a["times_s"]), np.asarray(b["times_s"])
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    times = np.unique(np.r_[ta[(ta >= lo) & (ta <= hi)], tb[(tb >= lo) & (tb <= hi)]])
    if not len(times):
        raise ValueError("no shared observation interval")
    return tuple(
        np.column_stack([np.interp(times, t, np.asarray(v)[:, j]) for j in range(np.shape(v)[1])])
        for t, v in ((ta, av), (tb, bv))
    )


def compare(a, b):
    if "exception" in a or "exception" in b:
        return {"route_completed": False}
    aa, bb = a.get("segments", []), b.get("segments", [])
    checks = {"nonempty_histories": bool(aa) and bool(bb), "segment_count": len(aa) == len(bb)}
    for index, (sa, sb) in enumerate(zip(aa, bb, strict=False)):
        av, bv = aligned(sa, sb, sa["values"], sb["values"])
        for j, name in enumerate(RESPONSES):
            checks[f"{index}:{name}"] = ratio(
                av[:, j], bv[:, j], RESPONSE_ATOL[j], RESPONSE_RTOL[j]
            )
        na, nb = observed_amounts(sa), observed_amounts(sb)
        if na.shape[1] != nb.shape[1]:
            na, nb = na[:, -2:], nb[:, -2:]
        na, nb = aligned(sa, sb, na, nb)
        checks[f"{index}:actual_shared_amounts"] = ratio(na, nb, 1e-9, 2e-7)
        checks[f"{index}:endpoint_time"] = abs(sa["end_time_s"] - sb["end_time_s"]) / (
            1e-3 + 2e-7 * max(sa["end_time_s"], sb["end_time_s"])
        )
    return checks


def assess(item):
    data, routes = item["inputs"], item["routes"]
    if data["expected"] == "refusals":
        item["checks"] = {
            "refusals": {name: routes.get(name, {}).get("rejected", False) for name in REFUSALS}
        }
    else:
        names = ("native_coarse", "native_fine", "reference_coarse", "reference_fine")
        present = {
            name: routes.get(name, {"exception": "required route missing"}) for name in names
        }
        item["checks"] = {
            name: invariants(data, route, name.split("_")[0]) for name, route in present.items()
        }
        for group, an, bn in (
            ("paired_coarse", "native_coarse", "reference_coarse"),
            ("paired_fine", "native_fine", "reference_fine"),
            ("native_precision", "native_coarse", "native_fine"),
            ("reference_precision", "reference_coarse", "reference_fine"),
        ):
            item["checks"][group] = compare(present[an], present[bn])
    item["failures"] = [
        {"group": group, "check": key, "value": value}
        for group, checks in item["checks"].items()
        for key, value in checks.items()
        if not passes(value)
    ]
    expected = len(REFUSALS) if data["expected"] == "refusals" else 4
    if len(routes) != expected:
        item["failures"].append(
            {
                "group": "denominator",
                "check": "all_routes_present",
                "value": len(routes),
                "expected": expected,
            }
        )
    item["passed"] = not item["failures"]
