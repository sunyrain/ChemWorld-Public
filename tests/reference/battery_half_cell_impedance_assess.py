"""Complete, fail-closed finite-metal EIS qualification; no result-dependent bounds."""

import numpy as np
from battery_composite_impedance_assess import (
    canonical_native_variables,
    check,
    complete_snapshot,
    finite_tree,
    values,
)
from battery_composite_impedance_traces import snapshot
from battery_half_cell_cases import initial_metal, native_spec
from battery_half_cell_impedance_cases import (
    ATOL,
    FREQUENCIES,
    GRIDS,
    RTOL,
    F,
    binding,
    gradient_profile,
    native_state,
    ocv,
    refusal_input,
    response_order,
    response_units,
    route_plan,
)
from battery_half_cell_impedance_traces import DEFINITION, native_observations, source_output_names


def dense(record):
    from scipy import sparse

    return sparse.coo_matrix(
        (record["data"], (record["row"], record["col"])), shape=record["shape"]
    ).toarray()


def initial_checks(trace, data, kind, cells, gradient=False):
    """Both successful and rejected routes must retain a genuine complete initial state."""
    checks = []
    try:
        initial = np.asarray(trace["initial_state"], float)
        residual = np.asarray(trace["stationary_residual"], float)
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
        expected_profile = (
            gradient_profile(cells, data["initial_positive_fraction"])
            if gradient
            else np.full(cells, data["initial_positive_fraction"])
        )
        if kind == "native":
            from chemworld.physchem.battery_impedance import battery_impedance_system

            spec = native_spec(data, cells)
            authored = native_state(spec, gradient=gradient)
            canonical_system, _ = battery_impedance_system(spec, authored)
            variables = trace["variables"]
            names = {v["name"]: i for i, v in enumerate(variables)}
            scales = np.asarray(trace["residual_scales"], float)
            checks.append(
                check(
                    "native-canonical-state",
                    variables == canonical_native_variables(spec)
                    and len(variables) == len(initial)
                    and complete_snapshot(trace["state_before"])
                    and trace["state_before"] == snapshot(authored)
                    and trace["state_after"] == trace["state_before"]
                    and trace["state_unchanged"] is True
                    and all(initial[names[k]] == v for k, v in authored.amounts_mol.items())
                    and all(initial[names[k]] == v for k, v in authored.coordinates.items()),
                )
            )
            checks.append(
                check(
                    "native-initial-algebraic",
                    initial[names["battery:current_A"]] == 0.0
                    and abs(initial[names["battery:voltage_V"]] - trace["rest_voltage_V"]) <= 1e-12,
                )
            )
            checks.append(
                check(
                    "native-residual-scales",
                    scales.shape == initial.shape
                    and np.all(np.isfinite(scales))
                    and np.all(scales > 0)
                    and np.array_equal(scales, canonical_system.residual_scales),
                )
            )
            error = float(np.max(abs(residual / scales)))
            checks.append(
                check(
                    "native-stationarity" if not gradient else "native-nonstationarity",
                    error > 1e-9 if gradient else error <= 1e-9,
                    maximum=error,
                    limit=1e-9,
                )
            )
        else:
            layout = trace["source_state_layout"]
            indexed = {entry["name"]: entry for entry in layout}
            # Actual positive-working SPM keeps both signed and absolute charge counters.
            expected_names = {
                "Discharge capacity [A.h]",
                "Throughput capacity [A.h]",
                "X-averaged positive particle concentration [mol.m-3]",
            }
            checks.append(
                check(
                    "source-canonical-layout",
                    set(indexed) == expected_names
                    and len(layout) == len(expected_names)
                    and sorted(i for row in layout for i in row["indices"])
                    == list(range(len(initial)))
                    and trace["state_count"] == trace["differential_count"] == len(initial)
                    and all(
                        np.isfinite(v["scale"])
                        and v["scale"] > 0
                        and np.isfinite(v["reference"])
                        and isinstance(v["domains"], dict)
                        for v in layout
                    ),
                )
            )
            for name in ("Discharge capacity [A.h]", "Throughput capacity [A.h]"):
                entry = indexed[name]
                checks.append(
                    check(
                        "source-initial-" + name,
                        len(entry["indices"]) == 1
                        and np.all(
                            initial[entry["indices"]] * entry["scale"] + entry["reference"] == 0.0
                        ),
                    )
                )
            particle = indexed["X-averaged positive particle concentration [mol.m-3]"]
            profile = (
                initial[particle["indices"]] * particle["scale"] + particle["reference"]
            ) / data["maximum_concentration_mol_m3"]
            outputs = np.asarray(trace["initial_outputs"], float)
            checks.append(
                check(
                    "source-authored-initial",
                    profile.shape == (cells,)
                    and np.allclose(profile, expected_profile, rtol=0, atol=1e-10)
                    and outputs.shape == (10 + cells,)
                    and np.all(np.isfinite(outputs))
                    and np.allclose(outputs[10:], profile, rtol=0, atol=1e-10)
                    and abs(outputs[3] - data["initial_positive_fraction"]) <= 1e-10
                    and abs(
                        outputs[4]
                        - data["initial_positive_fraction"]
                        * data["active_volume_m3"]
                        * data["maximum_concentration_mol_m3"]
                    )
                    <= 1e-12
                    and abs(outputs[5] - initial_metal(data)) <= 1e-12
                    and outputs[6] == 0.0
                    and abs(outputs[0] - trace["rest_voltage_V"]) <= 1e-12,
                )
            )
            checks.append(
                check(
                    "source-exact-observation-metadata",
                    trace["output_names"] == source_output_names(cells)
                    and trace["source_energy"]
                    == {
                        "dynamic": False,
                        "original_energy_Wh": 0.0,
                        "qualification": (
                            "excluded: original source calculate discharge energy is false"
                        ),
                    }
                    and trace["source_metal_definition"]
                    == (
                        "derived n0 - 3600*Discharge capacity [A.h]/F; "
                        "not an upstream independent metal state"
                    ),
                )
            )
            original, retained = trace["source_original_model"], trace["source_retained_model"]
            checks.append(
                check(
                    "source-model-not-deleted",
                    original["rhs"] == retained["rhs"]
                    and original["initial_conditions"] == retained["initial_conditions"]
                    and set(original["variables"]) <= set(retained["variables"])
                    and set(original["rhs"]) == expected_names
                    and set(original["initial_conditions"]) == expected_names,
                )
            )
            checks.extend(source_operator_checks(trace, cells))
            error = float(np.max(abs(residual)))
            checks.append(
                check(
                    "source-stationarity" if not gradient else "source-nonstationarity",
                    (error > 1e-6 if gradient else error <= 1e-6)
                    and trace["maximum_stationary_rhs"] == error
                    and trace["maximum_stationary_algebraic"] == 0.0
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


def source_operator_checks(trace, cells):
    linear = trace["source_linearization"]
    n = len(trace["initial_state"])
    j, mass, c = (dense(linear[k]) for k in ("jacobian", "mass", "output_state"))
    b, d = np.asarray(linear["forcing"]), np.asarray(linear["output_input"])
    layout = {v["name"]: v for v in trace["source_state_layout"]}
    observer = layout["Throughput capacity [A.h]"]["indices"]
    physical = [i for i in range(n) if i not in observer]
    feedback_rhs = float(np.max(abs(j[physical][:, observer]), initial=0))
    feedback_output = float(np.max(abs(c[:, observer]), initial=0))
    return [
        check(
            "source-full-operator",
            j.shape == mass.shape == (n, n)
            and c.shape == (10 + cells, n)
            and b.shape == (n,)
            and d.shape == (10 + cells,)
            and finite_tree(linear)
            and np.all(np.isfinite(j))
            and np.all(np.isfinite(c))
            and np.all(np.isfinite(mass)),
        ),
        check(
            "source-abs-observer-no-feedback",
            linear["abs_observer_indices"] == observer
            and linear["physical_row_indices"] == physical
            and linear["maximum_observer_feedback_rhs"] == feedback_rhs == 0.0
            and linear["maximum_observer_feedback_output"] == feedback_output == 0.0
            and linear["abs_input_derivative_convention"] == b[observer].tolist()
            and linear["abs_response_qualification"]
            == (
                "excluded: abs(I) has no two-sided derivative at I=0; "
                "actual CasADi convention retained"
            ),
        ),
    ]


def structural(trace, data, kind, cells, frequencies=FREQUENCIES):
    checks = []
    try:
        count = 0 if kind == "analytic" else cells
        checks.append(
            check(
                "route-metadata",
                trace["implementation"] == kind
                and trace["grid"] == cells
                and trace["binding"] == binding(data, kind, cells)
                and trace["response_order"] == response_order(count)
                and trace["response_units"] == response_units(count),
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
                and array.shape == (len(frequencies), 10 + count)
                and np.all(np.isfinite(array))
                and trace["success"] is True
                and all(p["success"] is True for p in points),
            )
        )
        if kind == "analytic":
            checks.append(
                check(
                    "analytic-finite-work",
                    all(
                        np.isfinite(p["work_response_real"])
                        and np.isfinite(p["work_response_imag"])
                        for p in points
                    ),
                )
            )
            return checks
        checks.extend(initial_checks(trace, data, kind, cells))
        initial = np.asarray(trace["initial_state"])
        raw = np.array([p["state_response_real"] for p in points]) + 1j * np.array(
            [p["state_response_imag"] for p in points]
        )
        linear = np.array([p["relative_linear_residual"] for p in points])
        checks.append(
            check(
                "raw-responses-linear-residual",
                raw.shape == (len(frequencies), len(initial))
                and np.all(np.isfinite(raw))
                and np.all(np.isfinite(linear))
                and np.all((linear >= 0) & (linear <= 1e-10)),
                maximum=float(np.max(linear)),
            )
        )
        edges = np.linspace(0.0, 1.0, cells + 1)
        mean = array[:, 10:] @ np.diff(edges**3)
        error = float(np.max(abs(mean - array[:, 3]) / (1e-12 + 1e-10 * abs(array[:, 3]))))
        checks.append(check("radial-mean-identity", error <= 1.0, maximum_allowance_ratio=error))
        if kind == "native":
            diagnostics = trace["diagnostics"]
            checks.append(
                check(
                    "native-diagnostics",
                    finite_tree(diagnostics)
                    and diagnostics["maximum_scaled_stationary_residual"] <= 1e-9
                    and diagnostics["stationary_tolerance"] == 1e-9
                    and diagnostics["linear_tolerance"] == 1e-10
                    and diagnostics["physical_time_advanced_s"] == 0.0
                    and diagnostics["completed"]
                    == diagnostics["passed"]
                    == diagnostics["total"]
                    == len(frequencies),
                )
            )
            inferred, work = native_observations(native_spec(data, cells), trace["variables"], raw)
            reported_work = np.array([p["work_response_real"] for p in points]) + 1j * np.array(
                [p["work_response_imag"] for p in points]
            )
            identity = float(max(np.max(abs(inferred - array)), np.max(abs(work - reported_work))))
            checks.append(
                check("native-complete-observation-identity", identity <= 1e-12, maximum=identity)
            )
        else:
            layout = {v["name"]: v for v in trace["source_state_layout"]}
            particle, charge = (
                layout[k]
                for k in (
                    "X-averaged positive particle concentration [mol.m-3]",
                    "Discharge capacity [A.h]",
                )
            )
            profile = (
                raw[:, particle["indices"]]
                * particle["scale"]
                / data["maximum_concentration_mol_m3"]
            )
            q = raw[:, charge["indices"][0]] * charge["scale"] * 3600
            identity = float(
                max(
                    np.max(abs(profile - array[:, 10:])),
                    np.max(abs(q - array[:, 6])),
                    np.max(abs(-q / F - array[:, 5])),
                )
            )
            checks.append(
                check("source-scaled-observation-identity", identity <= 1e-12, maximum=identity)
            )
            lin = trace["source_linearization"]
            c, d = dense(lin["output_state"]), np.asarray(lin["output_input"])
            expected = raw @ c.T + d
            expected[:, 0] *= -1
            identity = float(np.max(abs(expected - array)))
            checks.append(
                check(
                    "source-entire-observation-identity",
                    identity <= 1e-12 + 1e-10 * np.max(abs(array)),
                    maximum=identity,
                )
            )
            j, mass, b = dense(lin["jacobian"]), dense(lin["mass"]), np.asarray(lin["forcing"])
            back = []
            for f, y in zip(frequencies, raw, strict=True):
                operator = 2j * np.pi * f * mass - j
                back.append(
                    np.max(abs(operator @ y - b))
                    / (np.linalg.norm(operator, np.inf) * np.max(abs(y)) + np.max(abs(b)))
                )
            checks.append(
                check(
                    "source-retained-full-linear-system",
                    max(back) <= 1e-10,
                    maximum=float(max(back)),
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
        omega = 2 * np.pi * np.asarray(trace["frequencies_Hz"])
        f, fm = trace["faraday_C_mol"], trace["metal_mapping_faraday_C_mol"]
        checks.append(check("declared-Faraday", np.isfinite(f) and abs(f - F) < 1e-3 and fm == F))
        errors = {
            "unit-current": np.max(abs(a[:, 1] - 1)),
            "positive-Faraday": np.max(abs(1j * omega * f * a[:, 4] - 1)),
            "metal-Faraday": np.max(abs(1j * omega * fm * a[:, 5] + 1)),
            "signed-charge": np.max(abs(1j * omega * a[:, 6] - 1)),
            "total-Li": np.max(
                abs(a[:, 4] + a[:, 5]) / np.maximum(1 / (F * omega), abs(a[:, 4]) + abs(a[:, 5]))
            ),
            "mean-inventory": np.max(
                abs(
                    a[:, 4] / (data["active_volume_m3"] * data["maximum_concentration_mol_m3"])
                    - a[:, 3]
                )
                / np.maximum(1.0, abs(a[:, 3]))
            ),
        }
        if trace["implementation"] != "reference":
            work = np.array([p["work_response_real"] for p in trace["points"]]) + 1j * np.array(
                [p["work_response_imag"] for p in trace["points"]]
            )
            errors["signed-work"] = np.max(abs(1j * omega * work - trace["rest_voltage_V"])) / max(
                1.0, abs(trace["rest_voltage_V"])
            )
        for name, error in errors.items():
            checks.append(
                check(name, np.isfinite(error) and error <= 1e-8, maximum=float(error), limit=1e-8)
            )
        checks.append(
            check(
                "authored-rest-voltage",
                abs(trace["rest_voltage_V"] - ocv(data["initial_positive_fraction"])) <= 1e-8,
            )
        )
    except Exception as exc:
        checks.append(
            check("invariant-input-failure", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return checks


def compare(left, right, *, exact=False, resistance_shift=0.0):
    a, b = values(left)[:, :10].copy(), values(right)[:, :10]
    a[:, 0] -= resistance_shift
    a[:, 9] -= resistance_shift
    bounds = 1e-8 + 1e-8 * abs(b) if exact else ATOL + RTOL * abs(b)
    errors = abs(a - b)
    row = check(
        "response-comparison",
        np.all(np.isfinite(errors)) and np.all(errors <= bounds),
        maximum_allowance_ratio=float(np.max(errors / bounds)),
        maximum_error_by_output=np.max(errors, axis=0).tolist(),
        errors=errors.tolist(),
        bounds=bounds.tolist(),
    )
    if left["grid"] == right["grid"] and left["grid"] is not None:
        pa, pb = values(left)[:, 10:], values(right)[:, 10:]
        pbounds = (1e-8 + 1e-8 * abs(pb)) if exact else 0.0005 + RTOL * abs(pb)
        error = float(np.max(abs(pa - pb) / pbounds))
        row["profile_maximum_allowance_ratio"] = error
        row["passed"] = row["passed"] and error <= 1.0
    return row


def nonstationary_checks(trace, data, kind, cells, frequencies=FREQUENCIES):
    rows = initial_checks(trace, data, kind, cells, gradient=True)
    try:
        correct = (
            trace["rejected"] is True
            and trace["success"] is False
            and trace["points"] == []
            and trace["implementation"] == kind
            and trace["grid"] == cells
            and trace["binding"] == binding(data, kind, cells, "radial-gradient")
            and np.array_equal(trace["frequencies_Hz"], frequencies)
            and trace["response_order"] == response_order(cells)
            and trace["response_units"] == response_units(cells)
        )
        if kind == "native":
            correct = (
                correct
                and trace["exception_type"] == "ResidualDomainError"
                and "nonstationary" in trace["exception"]
            )
        else:
            correct = correct and trace["failure"] == "reference_not_stationary"
        rows.append(check("nonstationarity-rejection", correct))
    except Exception as exc:
        rows.append(check("nonstationarity-rejection", False, exception=str(exc)))
    return rows


def assess_unit(index, data, routes):
    expected = route_plan(index)
    checks = [check("complete-route-denominator", set(routes) == {r[0] for r in expected})]
    if index <= 8:
        eligible = set()
        for name, kind, grid in expected:
            trace = routes.get(name, {})
            rows = structural(trace, data, kind, GRIDS[grid] if grid else None)
            checks.extend(dict(row, route=name) for row in rows)
            if all(row["passed"] for row in rows):
                eligible.add(name)
                checks.extend(dict(row, route=name) for row in invariants(trace, data))
        pairs = [(f"native_{g}", f"reference_{g}") for g in GRIDS]
        pairs += [(f"{k}_coarse", f"{k}_fine") for k in ("native", "reference")]
        pairs += [(f"{k}_{g}", "analytic") for k in ("native", "reference") for g in GRIDS]
        for left, right in pairs:
            try:
                row = (
                    compare(routes[left], routes[right])
                    if {left, right} <= eligible
                    else check("response-comparison", False, reason="incomplete-route")
                )
            except Exception as exc:
                row = check("response-comparison", False, exception=str(exc))
            checks.append(dict(row, left=left, right=right))
    elif index == 9:
        for name, kind, grid in expected:
            trace = routes.get(name, {})
            rows = nonstationary_checks(trace, data, kind, GRIDS[grid])
            checks.extend(dict(row, route=name) for row in rows)
    else:
        tokens = {
            "empty-metal": "positive metal",
            "working-lower-face": "stopping boundary",
            "working-upper-face": "stopping boundary",
            "voltage-face": "stopping boundary",
            "missing-metal": "inventory",
            "missing-QW": "coordinates",
            "Faraday-inconsistent": "Faraday",
            "zero-frequency": "frequencies",
        }
        for name, _, grid in expected:
            trace = routes.get(name, {})
            try:
                spec, state, frequencies = refusal_input(data, name, GRIDS[grid])
                correct = (
                    trace["implementation"] == "refusal"
                    and trace["name"] == name
                    and trace["grid"] == GRIDS[grid]
                    and trace["binding"] == binding(data, "refusal", GRIDS[grid], name)
                    and trace["rejected"] is True
                    and trace["exception_type"] in {"ValueError", "ResidualDomainError"}
                    and tokens[name] in trace["exception"]
                    and trace["state_unchanged"] is True
                    and complete_snapshot(trace["state_before"])
                    and trace["state_before"] == snapshot(state)
                    and trace["state_after"] == trace["state_before"]
                    and trace["specification"] == spec.to_dict()
                    and trace["frequencies_Hz"] == frequencies
                )
            except Exception:
                correct = False
            checks.append(check("domain-refusal", correct, route=name))
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}
