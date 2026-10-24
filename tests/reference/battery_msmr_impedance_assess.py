"""Frozen numerical criteria and complete actual-state identity checks."""

import numpy as np
from battery_composite_impedance_assess import (
    canonical_native_variables,
    check,
    complete_snapshot,
    finite_tree,
    values,
)
from battery_composite_impedance_traces import snapshot
from battery_half_cell_impedance_assess import dense
from battery_msmr_cases import initial_profile, native_initial_state, native_spec
from battery_msmr_impedance_cases import (
    FREQUENCIES,
    GRIDS,
    RTOL,
    F,
    R,
    binding,
    output_schema,
    refusal_input,
    response_order,
    response_units,
    route_plan,
)
from battery_msmr_impedance_traces import (
    ABS_EXCLUSION,
    ABS_NAMES,
    COUNTERS,
    DEFINITION,
    native_observations,
    source_output_names,
)
from scipy.special import expit


def identity_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.all(np.isfinite(actual))
        or not np.all(np.isfinite(expected))
    ):
        return float("inf")
    return float(np.max(abs(actual - expected) / (1e-12 + 1e-10 * abs(expected)), initial=0))


def material_checks(trace, data, cells):
    checks = []
    f, r, t, ce = (
        trace["faraday_C_mol"],
        trace["gas_constant_J_mol_K"],
        data["temperature_K"],
        data["electrolyte_concentration_mol_m3"],
    )
    checks.append(
        check(
            "physical-constants",
            np.isfinite(f) and np.isfinite(r) and abs(f - F) < 1e-3 and abs(r - R) < 1e-8,
        )
    )
    for e in data["electrodes"]:
        m = trace["material_static"][e["name"]]
        rows = np.array(e["rows"])
        weight, u0, width, exchange, alpha = rows.T
        u, factor = m["potential_V"], f / (r * t)
        x = weight * expit(-factor * (u - u0) / width)
        dx = -factor / width * x * (1 - x / weight)
        j0 = (
            exchange
            * x**width
            * np.exp(factor * (1 - alpha) * (u - u0))
            * (ce / 1000) ** (1 - alpha)
        )
        dchem = -e["diffusivity_m2_s"] * factor * sum(x) * (1 - sum(x)) / sum(dx)
        errors = [
            identity_error(m[k], v)
            for k, v in (("x", x), ("dx", dx), ("j0", j0), ("chemical_diffusivity_m2_s", dchem))
        ]
        if cells:
            uu = np.array(m["radial_initial_U_V"])
            xx = weight[:, None] * expit(-factor * (uu - u0[:, None]) / width[:, None])
            slopes = np.sum(-factor / width[:, None] * xx * (1 - xx / weight[:, None]), axis=0)
            errors += [
                identity_error(m["radial_initial_x"], xx.sum(axis=0)),
                identity_error(m["radial_xU_V_inv"], slopes),
            ]
            errors += [identity_error(uu, initial_profile(e, cells, t)[1])]
        checks.append(
            check(
                "material-identity:" + e["name"],
                max(errors) <= 1,
                maximum_allowance_ratio=max(errors),
            )
        )
    if trace["implementation"] == "reference":
        expected = []
        for e in data["electrodes"]:
            for j, row in enumerate(e["rows"]):
                factor = (ce / 1000.0) ** (1 - row[4])
                expected.append(
                    {
                        "electrode": e["name"],
                        "row_index": j,
                        "authored_row": list(row),
                        "authored_reference_ce_mol_m3": 1000.0,
                        "source_reference_ce_mol_m3": ce,
                        "factor": factor,
                        "source_parameter": f"j0_ref_{e['name'][0]}_{j}",
                        "source_jref_A_m2": row[3] * factor,
                    }
                )
        checks.append(
            check(
                "source-jref-reference-conversion",
                trace["source_exchange_reference_mapping"] == expected,
            )
        )
    return checks


def source_operator_checks(trace, data, cells):
    linear = trace["source_linearization"]
    n = len(trace["initial_state"])
    m = len(response_order(data, cells))
    j, mass, c = (dense(linear[k]) for k in ("jacobian", "mass", "output_state"))
    b, d = np.array(linear["forcing"]), np.array(linear["output_input"])
    layout = {v["name"]: v for v in trace["source_state_layout"]}
    observer = [layout[k]["indices"][0] for k in ABS_NAMES]
    physical = [i for i in range(n) if i not in observer]
    feedback_rhs = float(np.max(abs(j[physical][:, observer]), initial=0))
    feedback_output = float(np.max(abs(c[:, observer]), initial=0))
    return [
        check(
            "source-full-operator",
            j.shape == mass.shape == (n, n)
            and c.shape == (m, n)
            and b.shape == (n,)
            and d.shape == (m,)
            and finite_tree(linear),
        ),
        check(
            "source-abs-observers-no-feedback",
            linear["abs_observer_indices"] == observer
            and linear["physical_row_indices"] == physical
            and linear["maximum_observer_feedback_rhs"] == feedback_rhs == 0.0
            and linear["maximum_observer_feedback_output"] == feedback_output == 0.0
            and linear["abs_input_derivative_convention"] == b[observer].tolist()
            and linear["abs_response_qualification"] == ABS_EXCLUSION,
        ),
    ]


def initial_checks(trace, data, kind, cells, gradient=False):
    checks = []
    try:
        initial, residual = (
            np.asarray(trace["initial_state"], float),
            np.asarray(trace["stationary_residual"], float),
        )
        checks.append(
            check(
                "initial-raw-complete",
                initial.ndim == 1
                and len(initial) > 0
                and residual.shape == initial.shape
                and np.all(np.isfinite(initial))
                and np.all(np.isfinite(residual))
                and trace["definition"] == DEFINITION,
            )
        )
        spec = native_spec(data, cells)
        authored = native_initial_state(spec, data, cells)
        checks.extend(material_checks(trace, data, cells))
        if kind == "native":
            from chemworld.physchem.battery_impedance import battery_impedance_system

            system, canonical_initial = battery_impedance_system(spec, authored)
            variables = trace["variables"]
            scales = np.asarray(trace["residual_scales"])
            checks.append(
                check(
                    "native-canonical-state",
                    variables == canonical_native_variables(spec)
                    and initial.shape == canonical_initial.shape
                    and np.array_equal(initial, canonical_initial)
                    and complete_snapshot(trace["state_before"])
                    and trace["state_before"] == snapshot(authored)
                    and trace["state_after"] == trace["state_before"]
                    and trace["state_unchanged"] is True,
                )
            )
            checks.append(
                check("native-residual-scales", np.array_equal(scales, system.residual_scales))
            )
            error = float(np.max(abs(residual / scales)))
            checks.append(
                check(
                    "native-nonstationarity" if gradient else "native-stationarity",
                    error > 1e-9 if gradient else error <= 1e-9,
                    maximum=error,
                    limit=1e-9,
                )
            )
        else:
            layout = trace["source_state_layout"]
            indexed = {v["name"]: v for v in layout}
            names = set(COUNTERS)
            for e in data["electrodes"]:
                names.update(
                    (
                        f"X-averaged {e['name']} particle potential [V]",
                        f"X-averaged {e['name']} electrode surface potential difference [V]",
                    )
                )
            checks.append(
                check(
                    "source-canonical-layout",
                    set(indexed) == names
                    and len(layout) == 8
                    and sorted(i for row in layout for i in row["indices"])
                    == list(range(len(initial)))
                    and len(initial) == 2 * cells + 6
                    and trace["state_count"] == len(initial)
                    and trace["differential_count"] == 2 * cells + 4
                    and all(
                        v["scale"] == 1.0
                        and np.isfinite(v["reference"])
                        and isinstance(v["domains"], dict)
                        for v in layout
                    ),
                )
            )
            for name in COUNTERS:
                entry = indexed[name]
                checks.append(
                    check(
                        "source-initial:" + name,
                        len(entry["indices"]) == 1
                        and entry["reference"] == 0.0
                        and np.all(initial[entry["indices"]] == 0.0),
                    )
                )
            outputs = np.array(trace["initial_outputs"])
            order = response_order(data, cells)
            checks.append(
                check(
                    "source-initial-outputs",
                    outputs.shape == (len(order),)
                    and np.all(np.isfinite(outputs))
                    and trace["rest_voltage_V"] == outputs[0]
                    and np.all(outputs[1:4] == 0),
                )
            )
            for e in data["electrodes"]:
                name = e["name"]
                entry = indexed[f"X-averaged {name} particle potential [V]"]
                u = initial[entry["indices"]] * entry["scale"] + entry["reference"]
                m = trace["material_static"][name]
                actual_u = [outputs[order.index(f"{name}:profile_U:{i}")] for i in range(cells)]
                actual_x = [outputs[order.index(f"{name}:profile_x:{i}")] for i in range(cells)]
                algebraic = indexed[f"X-averaged {name} electrode surface potential difference [V]"]
                phi = initial[algebraic["indices"]][0] + algebraic["reference"]
                checks.append(
                    check(
                        "source-authored:" + name,
                        len(entry["indices"]) == cells
                        and entry["reference"] == 0.0
                        and identity_error(u, m["radial_initial_U_V"]) <= 1
                        and identity_error(actual_u, u) <= 1
                        and identity_error(actual_x, m["radial_initial_x"]) <= 1
                        and len(algebraic["indices"]) == 1
                        and abs(
                            phi
                            - outputs[order.index(name + ":U")]
                            - outputs[order.index(name + ":eta")]
                        )
                        <= 1e-10,
                    )
                )
            original, retained = trace["source_original_model"], trace["source_retained_model"]
            checks.append(
                check(
                    "source-original-model-retained",
                    original["rhs"] == retained["rhs"]
                    and original["algebraic"] == retained["algebraic"]
                    and original["initial_conditions"] == retained["initial_conditions"]
                    and set(original["variables"]) <= set(retained["variables"])
                    and set(original["rhs"]) | set(original["algebraic"]) == names
                    and set(original["initial_conditions"]) == names
                    and trace["output_names"] == source_output_names(data, cells),
                )
            )
            checks.extend(source_operator_checks(trace, data, cells))
            error = float(np.max(abs(residual)))
            checks.append(
                check(
                    "source-nonstationarity" if gradient else "source-stationarity",
                    (error > 1e-6 if gradient else error <= 1e-6)
                    and trace["maximum_stationary_rhs"]
                    == float(np.max(abs(residual[: 2 * cells + 4])))
                    and trace["maximum_stationary_algebraic"]
                    == float(np.max(abs(residual[2 * cells + 4 :])))
                    and trace["stationary_tolerance"] == 1e-6
                    and trace["linear_tolerance"] == 1e-10
                    and trace["pybamm"] == "25.4.2"
                    and trace["casadi"] == "3.6.7",
                    maximum=error,
                    limit=1e-6,
                )
            )
    except Exception as exc:
        checks.append(
            check("missing-or-malformed-initial", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return checks


def structural(trace, data, kind, cells, frequencies=FREQUENCIES):
    checks = []
    try:
        count = 0 if kind == "analytic" else cells
        order = response_order(data, count)
        checks.append(
            check(
                "route-metadata",
                trace["implementation"] == kind
                and trace["grid"] == cells
                and trace["binding"] == binding(data, kind, cells)
                and trace["response_order"] == order
                and trace["response_units"] == response_units(data, count),
            )
        )
        points = trace["points"]
        array = values(trace)
        checks.append(
            check(
                "complete-finite-frequencies",
                len(points) == len(frequencies)
                and np.array_equal(trace["frequencies_Hz"], frequencies)
                and np.array_equal([p["frequency_Hz"] for p in points], frequencies)
                and array.shape == (len(frequencies), len(order))
                and np.all(np.isfinite(array))
                and trace["success"] is True
                and all(p["success"] is True for p in points),
            )
        )
        if kind == "analytic":
            checks.extend(material_checks(trace, data, 0))
            return checks
        checks.extend(initial_checks(trace, data, kind, cells))
        raw = np.array([p["state_response_real"] for p in points]) + 1j * np.array(
            [p["state_response_imag"] for p in points]
        )
        linear = np.array([p["relative_linear_residual"] for p in points])
        checks.append(
            check(
                "raw-responses-linear-residual",
                raw.shape == (len(frequencies), len(trace["initial_state"]))
                and np.all(np.isfinite(raw))
                and np.all(np.isfinite(linear))
                and np.all((linear >= 0) & (linear <= 1e-10)),
                maximum=float(np.max(linear)),
            )
        )
        weights = np.diff(np.linspace(0, 1, cells + 1) ** 3)
        for e in data["electrodes"]:
            name = e["name"]
            profile = array[:, [order.index(f"{name}:profile_x:{i}") for i in range(cells)]]
            mean = profile @ weights
            error = identity_error(mean, array[:, order.index(name + ":average")])
            checks.append(
                check("radial-mean-identity:" + name, error <= 1, maximum_allowance_ratio=error)
            )
        if kind == "native":
            diag = trace["diagnostics"]
            checks.append(
                check(
                    "native-diagnostics",
                    finite_tree(diag)
                    and diag["maximum_scaled_stationary_residual"] <= 1e-9
                    and diag["stationary_tolerance"] == 1e-9
                    and diag["linear_tolerance"] == 1e-10
                    and diag["physical_time_advanced_s"] == 0.0
                    and diag["completed"] == diag["passed"] == diag["total"] == len(frequencies),
                )
            )
            spec = native_spec(data, cells)
            projected, _ = native_observations(
                spec, native_initial_state(spec, data, cells), trace["variables"], raw, data
            )
            error = identity_error(array, projected)
            checks.append(
                check("native-actual-state-projection", error <= 1, maximum_allowance_ratio=error)
            )
        else:
            layout = {v["name"]: v for v in trace["source_state_layout"]}
            errors = []
            for e in data["electrodes"]:
                name = e["name"]
                entry = layout[f"X-averaged {name} particle potential [V]"]
                du = raw[:, entry["indices"]] * entry["scale"]
                m = trace["material_static"][name]
                dx = du * np.asarray(m["radial_xU_V_inv"])
                amount = (dx @ weights) * e["active_volume_m3"] * e["maximum_concentration_mol_m3"]
                errors += [
                    identity_error(
                        du, array[:, [order.index(f"{name}:profile_U:{i}") for i in range(cells)]]
                    ),
                    identity_error(
                        dx, array[:, [order.index(f"{name}:profile_x:{i}") for i in range(cells)]]
                    ),
                    identity_error(amount, array[:, order.index(name + ":amount")]),
                ]
            for rawkey, outkey in (
                ("Discharge capacity [A.h]", "charge_C_A"),
                ("Discharge energy [W.h]", "work_J_A"),
            ):
                entry = layout[rawkey]
                errors.append(
                    identity_error(
                        raw[:, entry["indices"][0]] * entry["scale"] * 3600,
                        array[:, order.index(outkey)],
                    )
                )
            checks.append(
                check(
                    "source-U-to-Li-and-QW-chain-map",
                    max(errors) <= 1,
                    maximum_allowance_ratio=max(errors),
                )
            )
            lin = trace["source_linearization"]
            c, d = dense(lin["output_state"]), np.array(lin["output_input"])
            projected = raw @ c.T + d
            projected[:, 0] *= -1
            error = identity_error(array, projected)
            checks.append(
                check("source-actual-output-operator", error <= 1, maximum_allowance_ratio=error)
            )
            j, mass, b = dense(lin["jacobian"]), dense(lin["mass"]), np.array(lin["forcing"])
            errors = []
            for f, y in zip(frequencies, raw, strict=True):
                operator = 2j * np.pi * f * mass - j
                errors.append(
                    np.max(abs(operator @ y - b))
                    / (np.linalg.norm(operator, np.inf) * np.max(abs(y)) + np.max(abs(b)))
                )
            checks.append(
                check(
                    "source-retained-full-linear-system",
                    max(errors) <= 1e-10,
                    maximum=float(max(errors)),
                )
            )
    except Exception as exc:
        checks.append(
            check("missing-or-malformed-record", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return checks


def invariants(trace, data):
    checks = []
    try:
        a = values(trace)
        order = trace["response_order"]
        v = {k: a[:, i] for i, k in enumerate(order)}
        omega = 2 * np.pi * np.asarray(trace["frequencies_Hz"])
        f = trace["faraday_C_mol"]
        errors = {
            "unit-current": np.max(abs(v["current_A_A"] - 1)),
            "signed-charge": np.max(abs(1j * omega * v["charge_C_A"] - 1)),
            "signed-work": np.max(abs(1j * omega * v["work_J_A"] - trace["rest_voltage_V"]))
            / max(1, abs(trace["rest_voltage_V"])),
        }
        for e, sign in zip(data["electrodes"], (1, -1), strict=True):
            name = e["name"]
            errors[name + ":Faraday"] = np.max(abs(1j * omega * f * v[name + ":amount"] + sign))
            errors[name + ":mean-inventory"] = np.max(
                abs(
                    v[name + ":amount"]
                    / (e["active_volume_m3"] * e["maximum_concentration_mol_m3"])
                    - v[name + ":average"]
                )
                / np.maximum(1, abs(v[name + ":average"]))
            )
            for prefix, total in (
                ("kinetic_x", v[name + ":surface"]),
                ("source_x", v[name + ":source_surface"]),
                ("current", sign * v["current_A_A"]),
            ):
                summed = sum(v[f"{name}:{prefix}{j}"] for j in range(len(e["rows"])))
                errors[name + ":" + prefix + "-sum"] = np.max(
                    abs(summed - total) / np.maximum(1, abs(total))
                )
        n, p = v["negative:amount"], v["positive:amount"]
        errors["total-Li"] = np.max(abs(n + p) / np.maximum(1 / (f * omega), abs(n) + abs(p)))
        for name, error in errors.items():
            checks.append(
                check(name, np.isfinite(error) and error <= 1e-8, maximum=float(error), limit=1e-8)
            )
    except Exception as exc:
        checks.append(check("invariant-input-failure", False, exception=str(exc)))
    return checks


def compare(left, right, data, *, exact=False, resistance_shift=0.0):
    schema = output_schema(data)
    count = len(schema)
    a, b = values(left)[:, :count].copy(), values(right)[:, :count]
    a[:, 0] -= resistance_shift
    bounds = 1e-8 + 1e-8 * abs(b) if exact else np.array([r[2] for r in schema]) + RTOL * abs(b)
    errors = abs(a - b)
    row = check(
        "response-comparison",
        np.all(np.isfinite(errors)) and np.all(errors <= bounds),
        maximum_allowance_ratio=float(np.max(errors / bounds)),
        response_order=[r[0] for r in schema],
        maximum_error_by_output=np.max(errors, axis=0).tolist(),
        errors=errors.tolist(),
        bounds=bounds.tolist(),
    )
    if left["grid"] == right["grid"] and left["grid"] is not None:
        pa, pb = values(left)[:, count:], values(right)[:, count:]
        tolerances = np.array([r[2] for r in output_schema(data, left["grid"])[count:]])
        allowance = 1e-8 + 1e-8 * abs(pb) if exact else tolerances + RTOL * abs(pb)
        error = float(np.max(abs(pa - pb) / allowance))
        row["profile_maximum_allowance_ratio"] = error
        row["passed"] = row["passed"] and error <= 1
    return row


def nonstationary_checks(trace, data, kind, cells, frequencies=FREQUENCIES):
    rows = initial_checks(trace, data, kind, cells, gradient=True)
    try:
        good = (
            trace["rejected"] is True
            and trace["success"] is False
            and trace["points"] == []
            and trace["implementation"] == kind
            and trace["grid"] == cells
            and trace["binding"] == binding(data, kind, cells, "radial-gradient")
            and np.array_equal(trace["frequencies_Hz"], frequencies)
            and trace["response_order"] == response_order(data, cells)
            and trace["response_units"] == response_units(data, cells)
        )
        good = good and (
            (
                trace["exception_type"] == "ResidualDomainError"
                and "nonstationary" in trace["exception"]
            )
            if kind == "native"
            else trace["failure"] == "reference_not_stationary"
        )
        rows.append(check("nonstationarity-rejection", good))
    except Exception as exc:
        rows.append(check("nonstationarity-rejection", False, exception=str(exc)))
    return rows


def domain_check(trace, data, name, cells):
    tokens = {
        "missing-Li": "inventory and coordinates",
        "missing-QW": "inventory and coordinates",
        "Faraday-inconsistent": "Faraday",
        "zero-local-Li": "strictly inside",
        "surface-below-margin": "stopping boundary",
        "voltage-face": "stopping boundary",
        "zero-frequency": "frequencies",
        "nan-frequency": "frequencies",
    }
    try:
        spec, state, freq = refusal_input(data, name, cells)
        good = (
            trace["implementation"] == "refusal"
            and trace["name"] == name
            and trace["grid"] == cells
            and trace["binding"] == binding(data, "refusal", cells, name)
            and trace["rejected"] is True
            and trace["exception_type"] in {"ValueError", "ResidualDomainError"}
            and tokens[name] in trace["exception"]
            and trace["state_unchanged"] is True
            and complete_snapshot(trace["state_before"])
            and trace["state_before"] == snapshot(state)
            and trace["state_after"] == trace["state_before"]
            and trace["specification"] == spec.to_dict()
            and np.array_equal(trace["frequencies_Hz"], freq, equal_nan=True)
        )
        return check("domain-refusal", good, route=name)
    except Exception as exc:
        return check("domain-refusal", False, route=name, exception=str(exc))


def assess_unit(index, data, routes):
    plan = route_plan(index)
    checks = [check("complete-route-denominator", set(routes) == {r[0] for r in plan})]
    if index <= 8:
        eligible = set()
        for name, kind, grid in plan:
            trace = routes.get(name, {})
            rows = structural(trace, data, kind, GRIDS[grid] if grid else None)
            checks.extend(dict(r, route=name) for r in rows)
            if all(r["passed"] for r in rows):
                eligible.add(name)
                checks.extend(dict(r, route=name) for r in invariants(trace, data))
        pairs = [(f"native_{g}", f"reference_{g}") for g in GRIDS]
        pairs += [(f"{k}_coarse", f"{k}_fine") for k in ("native", "reference")]
        pairs += [(f"{k}_{g}", "analytic") for k in ("native", "reference") for g in GRIDS]
        for left, right in pairs:
            try:
                row = (
                    compare(routes[left], routes[right], data)
                    if {left, right} <= eligible
                    else check("response-comparison", False, reason="incomplete-route")
                )
            except Exception as exc:
                row = check("response-comparison", False, exception=str(exc))
            checks.append(dict(row, left=left, right=right))
    elif index == 9:
        for name, kind, grid in plan:
            checks.extend(
                dict(r, route=name)
                for r in nonstationary_checks(routes.get(name, {}), data, kind, GRIDS[grid])
            )
    else:
        checks.extend(
            domain_check(routes.get(name, {}), data, name, GRIDS[grid]) for name, _, grid in plan
        )
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}
