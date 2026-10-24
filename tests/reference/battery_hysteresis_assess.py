"""Predeclared measurements; all negative results remain explicit."""

import numpy as np
from battery_hysteresis_cases import FREQUENCIES, response_atol, response_names, sweep_currents
from battery_hysteresis_traces import history_metrics
from scipy.interpolate import interp1d


def complete(trace, data):
    try:
        if trace["response_order"] != response_names(data):
            return False
        if len(trace["segments"]) != len(data["waveform"]):
            return False
        start = 0.
        for part, (duration, _, _) in zip(trace["segments"], data["waveform"], strict=True):
            times, rows = np.array(part["times_s"]), np.array(part["rows"])
            if (len(times) < 2 or not np.all(np.diff(times) > 0)
                    or rows.shape != (len(times), len(response_names(data)))
                    or not np.isfinite(rows).all()):
                return False
            if data["expected"] != "voltage_event":
                planned = np.linspace(start, start+duration, data["samples_per_segment"])
                if (times.shape != planned.shape or not np.allclose(times, planned, atol=1e-8,
                    rtol=0) or (part.get("outcome") != "success" if trace["implementation"] ==
                               "native" else part["termination"] != "final time")):
                    return False
            start += duration
        if data["expected"] == "voltage_event":
            return ("minimum-voltage" in str(part["termination"])
                    or "Minimum voltage" in str(part["termination"]))
        return True
    except (KeyError, TypeError, ValueError):
        return False


def aligned_rows(left, right, data):
    a, b = [], []
    for lp, rp in zip(left["segments"], right["segments"], strict=True):
        lt, rt = np.asarray(lp["times_s"]), np.asarray(rp["times_s"])
        count = min(len(lt), len(rt))
        # Only the scheduled common observations. Separately assess each own event.
        take = abs(lt[:count]-rt[:count]) <= 1e-8
        a.extend(np.asarray(lp["rows"])[:count][take])
        b.extend(np.asarray(rp["rows"])[:count][take])
    if len(a) < 2:
        raise ValueError("fewer than two genuine shared observation times")
    return np.asarray(a), np.asarray(b)


def comparison(left, right, data):
    a, b = aligned_rows(left, right, data)
    error = abs(a-b)
    bounds = response_atol(data)+1e-5*abs(b)
    result = {"maximum_absolute_error": np.max(error, axis=0).tolist(),
        "maximum_tolerance_ratio": np.max(error/bounds, axis=0).tolist(),
        "comparisons": int(error.size), "passed": bool(np.all(error <= bounds))}
    if data["size_distribution"] == "narrow":
        surface_error, ratios, outside, gap_ratio = [], [], 0, 0.
        for lp, rp in zip(left["segments"], right["segments"], strict=True):
            for name in ("negative", "positive"):
                local, reference = lp["local"][name], rp["local"][name]
                target, source = np.array(local["radii_m"]), np.array(reference["radii_m"])
                outside += int(np.sum((target < source[0]) | (target > source[-1])))
                gap_ratio = max(gap_ratio, max(0., source[0]-target[0])/(source[1]-source[0]),
                                max(0., target[-1]-source[-1])/(source[-1]-source[-2]))
                for key in ("surface", "j_A_m2"):
                    ref = interp1d(source, reference[key], axis=1, fill_value="extrapolate")(target)
                    delta = abs(np.array(local[key])-ref)
                    if key == "surface":
                        surface_error.append(float(np.max(delta)))
                    else:
                        ratios.append(float(np.max(delta/(.02+.02*abs(ref)))))
        result.update(local_surface_max_error=max(surface_error),
            local_current_max_tolerance_ratio=max(ratios), extrapolated_points=outside,
            maximum_extrapolation_grid_fraction=gap_ratio)
        result["passed"] &= (max(surface_error) <= 3e-3 and max(ratios) <= 1
                             and gap_ratio <= .5+1e-12)
    return result


def assess_time(item, data):
    checks, metrics, comparisons = {}, {}, {}
    paths = item["routes"]
    for label, trace in paths.items():
        ok = checks[label+"_complete"] = complete(trace, data)
        checks[label+"_invariants"] = False
        if ok:
            m = metrics[label] = history_metrics(trace, data)
            checks[label+"_invariants"] = (
                m["maximum_faradaic_residual_C"] <= 1e-5
                and m["maximum_total_lithium_error_mol"] <= 1e-10
                and m["maximum_analytic_h_error"] <= 1e-5
                and m["maximum_rest_h_change"] <= 1e-9
                and m["maximum_h_switch_jump"] <= 1e-9
                and m["maximum_h_magnitude"] <= 1+1e-8)
            if trace["implementation"] == "native":
                checks[label+"_native_bv_and_allocation"] = all(
                    p["maximum_bv_residual_V"] <= 1e-8
                    and p["maximum_current_sum_error_A"] <= 1e-9 for p in trace["segments"])
            if data["expected"] == "voltage_event":
                checks[label+"_voltage_event_surface"] = abs(
                    trace["segments"][-1]["rows"][-1][0]-data["minimum_voltage_V"]) <= 2e-5
    pairs = []
    for grid in ("coarse", "fine"):
        for accuracy in ("standard", "tight"):
            pairs.append((f"native_{grid}_{accuracy}", f"reference_{grid}_{accuracy}"))
        for implementation in ("native", "reference"):
            pairs.append((f"{implementation}_{grid}_standard", f"{implementation}_{grid}_tight"))
    for left, right in pairs:
        label = left+"_vs_"+right
        checks[label] = False
        if checks[left+"_complete"] and checks[right+"_complete"]:
            comparisons[label] = comparison(paths[left], paths[right], data)
            checks[label] = comparisons[label]["passed"]
            if data["expected"] == "voltage_event":
                le, re = (paths[k]["segments"][-1]["end_time_s"] for k in (left, right))
                comparisons[label]["event_time_error_s"] = abs(le-re)
                checks[label] &= abs(le-re) <= .1+.002*abs(re)
    mesh = {}
    for accuracy in ("standard", "tight"):
        for impl, divisor in (("native", 2), ("reference", 3)):
            lo, hi = f"{impl}_coarse_{accuracy}", f"{impl}_fine_{accuracy}"
            label = f"{impl}_{accuracy}_mesh"
            checks[label] = False
            if checks[lo+"_complete"] and checks[hi+"_complete"]:
                a, b = aligned_rows(paths[lo], paths[hi], data)
                err = np.max(abs(a-b), axis=0)
                mesh[label] = {"maximum_difference": err.tolist()}
                checks[label] = bool(np.all(err <= response_atol(data)/divisor))
                if data["expected"] == "voltage_event":
                    le, re = (paths[k]["segments"][-1]["end_time_s"] for k in (lo, hi))
                    mesh[label]["event_time_difference_s"] = abs(le-re)
                    checks[label] &= abs(le-re) <= (.1+.002*abs(re))/2
        lo = f"native_coarse_{accuracy}_vs_reference_coarse_{accuracy}"
        hi = f"native_fine_{accuracy}_vs_reference_fine_{accuracy}"
        checks[accuracy+"_error_trend"] = bool(lo in comparisons and hi in comparisons and np.all(
            np.array(comparisons[hi]["maximum_absolute_error"])
            <= np.array(comparisons[lo]["maximum_absolute_error"])+1e-7))
    item.update(checks=checks, comparisons=comparisons, metrics=metrics, mesh=mesh,
                passed=bool(all(checks.values())))


def assess_static(item, data):
    checks, metrics = {}, {}
    for name, route in item["routes"].items():
        try:
            rows = np.array([p["row"] for p in route["points"]])
            currents = np.array([p["current_A"] for p in route["points"]])
            checks[name+"_complete"] = (rows.shape == (9, len(response_names(data)))
                and np.array_equal(currents, sweep_currents()) and np.isfinite(rows).all())
            errors = []
            for electrode, column, sign in (("negative", 2, -1), ("positive", 3, 1)):
                a, b = data[electrode+"_equilibrium_polynomial_V"]
                expected = (a+b*rows[:, column]-.5*data[electrode+"_gap_V"]*np.tanh(
                    sign*data["current_density_scale_m2_A"]*currents/
                    data["effective_collector_area_m2"]))
                errors.append(float(np.max(abs(rows[:, -2+(column-2)]-expected))))
            metrics[name+"_net_mixing_error_V"] = errors
            checks[name+"_mixing"] = max(errors) <= 1e-8
        except (KeyError, ValueError, TypeError):
            checks[name+"_complete"] = False
    checks["same_static_values"] = False
    if all(checks.get(n+"_complete") for n in ("native", "reference")):
        a, b = [np.array([p["row"] for p in item["routes"][n]["points"]])
                for n in ("native", "reference")]
        metrics["voltage_ocp_max_error_V"] = float(np.max(abs(a[:, [0, -2, -1]]-b[:, [0, -2, -1]])))
        checks["same_static_values"] = metrics["voltage_ocp_max_error_V"] <= 1e-5
    item.update(checks=checks, metrics=metrics, passed=bool(all(checks.values())))


def frequency_values(trace):
    return np.array([complex(p["response_real"][0], p["response_imag"][0])
                     for p in trace["points"]])


def assess_frequency(item):
    checks, metrics, paths = {}, {}, item["routes"]
    for label, trace in paths.items():
        try:
            z = frequency_values(trace)
            checks[label+"_complete"] = (trace["success"] and z.shape == (13,)
                and np.isfinite(z).all()
                and [p["frequency_Hz"] for p in trace["points"]] == FREQUENCIES.tolist()
                and all(p["success"] and p["relative_linear_residual"] <= 1e-10
                        for p in trace["points"]))
            if label.startswith("native"):
                checks[label+"_readonly"] = trace.get("input_state_unchanged") is True
        except (KeyError, TypeError, ValueError):
            checks[label+"_complete"] = False
    for implementation in ("native", "reference"):
        for grid in ("coarse", "fine"):
            labels = [f"{implementation}_{grid}_{mode}" for mode in ("single", "sigmoid")]
            key = implementation+"_"+grid+"_analytic_increment"
            checks[key] = False
            if all(checks[label+"_complete"] for label in labels):
                error = max(abs(frequency_values(paths[labels[1]])-
                                frequency_values(paths[labels[0]])-500.))
                metrics[key+"_error_ohm"] = float(error)
                checks[key] = error <= 1e-5
    for mode in ("single", "sigmoid"):
        for grid in ("coarse", "fine"):
            labels = [f"{impl}_{grid}_{mode}" for impl in ("native", "reference")]
            key = grid+"_"+mode+"_agreement"
            checks[key] = False
            if all(checks[label+"_complete"] for label in labels):
                a, b = (frequency_values(paths[label]) for label in labels)
                metrics[key+"_max_error_ohm"] = float(max(abs(a-b)))
                checks[key] = bool(np.all(abs(a-b) <= 1e-3+.01*abs(b)))
        for implementation, divisor in (("native", 2), ("reference", 3)):
            labels = [f"{implementation}_{grid}_{mode}" for grid in ("coarse", "fine")]
            key = implementation+"_"+mode+"_mesh"
            checks[key] = False
            if all(checks[label+"_complete"] for label in labels):
                a, b = (frequency_values(paths[label]) for label in labels)
                metrics[key+"_max_error_ohm"] = float(max(abs(a-b)))
                checks[key] = bool(np.all(abs(a-b) <= (1e-3+.01*abs(b))/divisor))
    item.update(checks=checks, metrics=metrics, passed=bool(all(checks.values())))


def valid_refusal(trace):
    return (trace.get("rejected") is True and trace.get("input_state_unchanged") is True
            and trace.get("exception_type") in {"ValueError", "ResidualDomainError"}
            and any(word in trace.get("exception", "").lower() for word in
                    ("hysteresis", "ocp", "positive", "single", "isothermal", "side reaction")))
