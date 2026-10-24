"""Fail-closed NT EIS qualification against the prospective finite contract."""

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
from battery_msmr_impedance_assess import identity_error
from battery_msmr_impedance_traces import ABS_EXCLUSION, ABS_NAMES, COUNTERS, DEFINITION
from battery_newman_tobias_cases import native_spec
from battery_newman_tobias_impedance_cases import (
    FREQUENCIES,
    GRIDS,
    F,
    R,
    binding,
    ocv,
    refusal_input,
    response_order,
    response_units,
    route_plan,
    schema,
)
from battery_newman_tobias_impedance_traces import native_observations, source_output_names
from scipy import sparse


def source_operator_checks(trace, cells):
    linear = trace["source_linearization"]
    n = len(trace["initial_state"])
    size = len(response_order(cells))
    j, m, c = (dense(linear[k]) for k in ("jacobian", "mass", "output_state"))
    b, d = np.array(linear["forcing"]), np.array(linear["output_input"])
    layout = {r["name"]: r for r in trace["source_state_layout"]}
    observers = [layout[name]["indices"][0] for name in ABS_NAMES]
    physical = [i for i in range(n) if i not in observers]
    rhs = float(np.max(abs(j[physical][:, observers]), initial=0))
    out = float(np.max(abs(c[:, observers]), initial=0))
    return [
        check(
            "source-full-operator",
            j.shape == m.shape == (n, n)
            and c.shape == (size, n)
            and b.shape == (n,)
            and d.shape == (size,)
            and finite_tree(linear),
        ),
        check(
            "source-canonical-mass-matrix",
            np.array_equal(m, np.diag([1.0] * 6 + [0.0] * (n - 6))),
        ),
        check(
            "source-abs-observers-no-feedback",
            linear["abs_observer_indices"] == observers
            and linear["physical_row_indices"] == physical
            and linear["maximum_observer_feedback_rhs"] == rhs == 0.0
            and linear["maximum_observer_feedback_output"] == out == 0.0
            and linear["abs_input_derivative_convention"] == b[observers].tolist()
            and linear["abs_response_qualification"] == ABS_EXCLUSION,
        ),
    ]


def initial_checks(trace, data, kind, cells):
    rows = []
    try:
        initial = np.asarray(trace["initial_state"], float)
        residual = np.asarray(trace["stationary_residual"], float)
        rows.append(
            check(
                "initial-complete",
                initial.ndim == 1
                and len(initial) > 0
                and residual.shape == initial.shape
                and np.all(np.isfinite(initial))
                and np.all(np.isfinite(residual))
                and trace["definition"] == DEFINITION,
            )
        )
        if kind == "native":
            from chemworld.physchem.battery_impedance import battery_impedance_system

            spec = native_spec(data, cells)
            state = spec.initial_state()
            system, y = battery_impedance_system(spec, state)
            rows.append(
                check(
                    "native-canonical-state",
                    trace["variables"] == canonical_native_variables(spec)
                    and np.array_equal(initial, y)
                    and complete_snapshot(trace["state_before"])
                    and trace["state_before"] == snapshot(state)
                    and trace["state_after"] == trace["state_before"]
                    and trace["state_unchanged"] is True
                    and np.array_equal(trace["residual_scales"], system.residual_scales),
                )
            )
            error = float(np.max(abs(residual / system.residual_scales)))
            rows.append(check("native-stationary", error <= 1e-9, maximum=error, limit=1e-9))
        else:
            layout = trace["source_state_layout"]
            by_name = {r["name"]: r for r in layout}
            expected = set(COUNTERS)
            expected.update(
                f"Average {n} particle concentration [mol.m-3]" for n in ("negative", "positive")
            )
            expected.update(
                f"{n.capitalize()} electrode potential [V]" for n in ("negative", "positive")
            )
            expected.update(
                f"{n.capitalize()} electrolyte potential [V]"
                for n in ("negative", "separator", "positive")
            )
            nn, ns, np_ = cells
            nstate = 6 + 2 * nn + ns + 2 * np_
            rows.append(
                check(
                    "source-canonical-layout",
                    set(by_name) == expected
                    and len(layout) == 11
                    and sorted(i for r in layout for i in r["indices"]) == list(range(nstate))
                    and len(initial) == trace["state_count"] == nstate
                    and trace["differential_count"] == 6
                    and all(
                        np.isfinite(r["scale"])
                        and r["scale"] > 0
                        and np.isfinite(r["reference"])
                        and isinstance(r["domains"], dict)
                        for r in layout
                    ),
                )
            )
            for name in COUNTERS:
                row = by_name[name]
                rows.append(
                    check(
                        "source-counter:" + name,
                        len(row["indices"]) == 1
                        and row["scale"] == 1.0
                        and row["reference"] == 0.0
                        and initial[row["indices"][0]] == 0.0,
                    )
                )
            un = ocv(data["electrodes"][0])
            up = ocv(data["electrodes"][1])
            for e in data["electrodes"]:
                n = e["name"]
                row = by_name[f"Average {n} particle concentration [mol.m-3]"]
                concentration = initial[row["indices"]] * row["scale"] + row["reference"]
                rows.append(
                    check(
                        "source-authored-amount:" + n,
                        len(row["indices"]) == 1
                        and row["scale"] == e["maximum_concentration_mol_m3"]
                        and row["reference"] == 0.0
                        and identity_error(
                            concentration,
                            [e["initial_fraction"] * e["maximum_concentration_mol_m3"]],
                        )
                        <= 1,
                    )
                )
            for n, count in zip(("negative", "separator", "positive"), cells, strict=True):
                row = by_name[f"{n.capitalize()} electrolyte potential [V]"]
                rows.append(
                    check(
                        "source-electrolyte-gauge:" + n,
                        len(row["indices"]) == count
                        and row["scale"] == 1.0
                        and abs(row["reference"] + un) <= 1e-12
                        and np.max(abs(initial[row["indices"]] + row["reference"] + un)) <= 1e-10,
                    )
                )
                if n != "separator":
                    row = by_name[f"{n.capitalize()} electrode potential [V]"]
                    target = 0.0 if n == "negative" else up - un
                    rows.append(
                        check(
                            "source-solid-gauge:" + n,
                            len(row["indices"]) == count
                            and row["scale"] == 1.0
                            and abs(row["reference"] - target) <= 1e-12
                            and np.max(abs(initial[row["indices"]] + row["reference"] - target))
                            <= 1e-10,
                        )
                    )
            observed = np.array(trace["initial_outputs"])
            order = response_order(cells)
            initial_expected = {"Z_ohm": up - un, "current": 0.0, "charge": 0.0, "work": 0.0}
            for e in data["electrodes"]:
                n = e["name"]
                initial_expected.update(
                    {
                        n + ":amount": e["active_volume_m3"]
                        * e["maximum_concentration_mol_m3"]
                        * e["initial_fraction"],
                        n + ":theta": e["initial_fraction"],
                        n + ":U": ocv(e),
                    }
                )
            for n, count in zip(("negative", "separator", "positive"), cells, strict=True):
                for i in range(count):
                    initial_expected[f"{n}:phi_e:{i}"] = -un
                for i in range(count + 1):
                    initial_expected[f"{n}:ie:{i}"] = 0.0
                if n != "separator":
                    for key in ("phi_s", "eta", "j", "q"):
                        for i in range(count):
                            initial_expected[f"{n}:{key}:{i}"] = (
                                up - un if key == "phi_s" and n == "positive" else 0.0
                            )
                    for i in range(count + 1):
                        initial_expected[f"{n}:is:{i}"] = 0.0
            # Consistent initialization need not make derived high-conductivity
            # face currents bitwise zero. Retain raw values; apply the prospective
            # current-balance bound, not an extra 1e-12 ideal-zero requirement.
            expected = np.array([initial_expected[k] for k in order])
            zero_currents = np.array(
                [
                    k == "current" or any(tag in k for tag in (":ie:", ":is:", ":q:", ":j:"))
                    for k in order
                ]
            )
            identity = identity_error(observed[~zero_currents], expected[~zero_currents])
            zero_error = float(np.max(abs(observed[zero_currents]), initial=0))
            rows.append(
                check(
                    "source-initial-output-identity",
                    identity <= 1,
                    maximum_allowance_ratio=identity,
                )
            )
            rows.append(
                check(
                    "source-initial-zero-current",
                    np.isfinite(zero_error) and zero_error <= 1e-8,
                    maximum=zero_error,
                    limit=1e-8,
                )
            )
            before, after = trace["source_original_model"], trace["source_retained_model"]
            expected_rhs = set(COUNTERS) | {
                f"Average {n} particle concentration [mol.m-3]" for n in ("negative", "positive")
            }
            expected_alg = {
                "Negative electrode potential [V]",
                "Positive electrode potential [V]",
                "Electrolyte potential [V]",
            }
            rows.append(
                check(
                    "source-original-model-retained",
                    before["rhs"] == after["rhs"]
                    and before["algebraic"] == after["algebraic"]
                    and before["initial_conditions"] == after["initial_conditions"]
                    and set(before["variables"]) <= set(after["variables"])
                    and set(before["rhs"]) == expected_rhs
                    and set(before["algebraic"]) == expected_alg
                    and set(before["initial_conditions"]) == expected_rhs | expected_alg
                    and trace["output_names"] == source_output_names(cells),
                )
            )
            expected_mapping = {
                "net_OCV_endpoint_barrier": "explicitly compensated by original NT builder",
                "ce_reference_mol_m3": data["reference_electrolyte_concentration_mol_m3"],
                "solid_bulk_conductivity_S_m": [
                    sigma / (1 - data["porosities"][r]) ** bs
                    for sigma, r, bs in zip(
                        data["solid_conductivity_S_m"], (0, 2), data["solid_bruggeman"], strict=True
                    )
                ],
            }
            rows.append(
                check(
                    "source-material-parameter-mapping",
                    trace["source_material_mapping"] == expected_mapping,
                )
            )
            rows.extend(source_operator_checks(trace, cells))
            error = float(np.max(abs(residual)))
            rows.append(
                check(
                    "source-stationary",
                    error <= 1e-6
                    and trace["maximum_stationary_rhs"] == float(np.max(abs(residual[:6])))
                    and trace["maximum_stationary_algebraic"] == float(np.max(abs(residual[6:])))
                    and trace["stationary_tolerance"] == 1e-6
                    and trace["linear_tolerance"] == 1e-10
                    and trace["pybamm"] == "25.4.2"
                    and trace["casadi"] == "3.6.7",
                    maximum=error,
                    limit=1e-6,
                )
            )
    except Exception as exc:
        rows.append(
            check("missing-or-malformed-initial", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return rows


def structural(trace, data, kind, cells, frequencies=FREQUENCIES, grids=GRIDS):
    rows = []
    try:
        order = response_order(cells)
        rows.append(
            check(
                "route-metadata",
                trace["implementation"] == kind
                and trace["grid"] == (list(cells) if cells else None)
                and trace["binding"] == binding(data, kind, cells)
                and trace["response_order"] == order
                and trace["response_units"] == response_units(cells),
            )
        )
        array = values(trace)
        points = trace["points"]
        rows.append(
            check(
                "complete-finite-spectrum",
                trace["success"] is True
                and len(points) == len(frequencies)
                and np.array_equal(trace["frequencies_Hz"], frequencies)
                and np.array_equal([p["frequency_Hz"] for p in points], frequencies)
                and all(p["success"] is True for p in points)
                and array.shape == (len(frequencies), len(order))
                and np.all(np.isfinite(array)),
            )
        )
        if kind == "analytic":
            variants = trace["grid_responses"]
            rows.append(check("continuous-all-grids", set(variants) == set(grids)))
            for label, shape in grids.items():
                r = variants[label]
                a = np.array(r["response_real"]) + 1j * np.array(r["response_imag"])
                rows.append(
                    check(
                        "continuous-grid:" + label,
                        r["cells"] == list(shape)
                        and r["response_order"] == response_order(shape)
                        and r["response_units"] == response_units(shape)
                        and a.shape == (len(frequencies), len(response_order(shape)))
                        and np.all(np.isfinite(a))
                        and identity_error(a[:, :10], array) <= 1,
                    )
                )
            net = trace["continuous_network"]
            rows.append(
                check(
                    "continuous-complete-network",
                    finite_tree(net)
                    and len(net["electrodes"]) == 2
                    and all(
                        np.array(p["matrix"]).shape == (4, 4)
                        and np.array(p["initial"]).shape == np.array(p["end"]).shape == (4,)
                        and p["g"] > 0
                        and p["a"] > 0
                        and p["j0"] > 0
                        and p["length"] > 0
                        for p in net["electrodes"]
                    )
                    and np.array(net["effective_ionic_conductivity_S_m"]).shape == (3,),
                )
            )
            return rows
        rows.extend(initial_checks(trace, data, kind, cells))
        raw = np.array([p["state_response_real"] for p in points]) + 1j * np.array(
            [p["state_response_imag"] for p in points]
        )
        residual = np.array([p["relative_linear_residual"] for p in points])
        rows.append(
            check(
                "raw-responses-linear-residual",
                raw.shape == (len(frequencies), len(trace["initial_state"]))
                and np.all(np.isfinite(raw))
                and np.all(np.isfinite(residual))
                and np.all((residual >= 0) & (residual <= 1e-10)),
                maximum=float(np.max(residual)),
            )
        )
        if kind == "native":
            diag = trace["diagnostics"]
            rows.append(
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
            projected = native_observations(native_spec(data, cells), data, trace["variables"], raw)
            error = identity_error(array, projected)
            rows.append(
                check("native-original-projection", error <= 1, maximum_allowance_ratio=error)
            )
        else:
            layout = {r["name"]: r for r in trace["source_state_layout"]}
            errors = []
            for e in data["electrodes"]:
                n = e["name"]
                entry = layout[f"Average {n} particle concentration [mol.m-3]"]
                delta = raw[:, entry["indices"][0]] * entry["scale"]
                errors += [
                    identity_error(
                        delta * e["active_volume_m3"], array[:, order.index(n + ":amount")]
                    ),
                    identity_error(
                        delta / e["maximum_concentration_mol_m3"],
                        array[:, order.index(n + ":theta")],
                    ),
                ]
            for statekey, key in (
                ("Discharge capacity [A.h]", "charge"),
                ("Discharge energy [W.h]", "work"),
            ):
                entry = layout[statekey]
                errors.append(
                    identity_error(
                        raw[:, entry["indices"][0]] * entry["scale"] * 3600,
                        array[:, order.index(key)],
                    )
                )
            for n, count in zip(("negative", "separator", "positive"), cells, strict=True):
                entry = layout[n.capitalize() + " electrolyte potential [V]"]
                errors.append(
                    identity_error(
                        raw[:, entry["indices"]] * entry["scale"],
                        array[:, [order.index(f"{n}:phi_e:{i}") for i in range(count)]],
                    )
                )
                if n != "separator":
                    entry = layout[n.capitalize() + " electrode potential [V]"]
                    errors.append(
                        identity_error(
                            raw[:, entry["indices"]] * entry["scale"],
                            array[:, [order.index(f"{n}:phi_s:{i}") for i in range(count)]],
                        )
                    )
            rows.append(
                check(
                    "source-state-physical-projection",
                    max(errors) <= 1,
                    maximum_allowance_ratio=max(errors),
                )
            )
            lin = trace["source_linearization"]
            j, m = (dense(lin[k]) for k in ("jacobian", "mass"))
            b, d = np.array(lin["forcing"]), np.array(lin["output_input"])
            output_record = lin["output_state"]
            original_csc = sparse.csc_matrix(
                (
                    output_record["data"],
                    (output_record["row"], output_record["col"]),
                ),
                shape=output_record["shape"],
            )
            # Match the released helper's sparse summation order. Dense batched
            # products lose small gradients when low-frequency potentials cancel.
            projected = np.array([original_csc @ y + d for y in raw])
            projected[:, 0] *= -1
            error = identity_error(array, projected)
            rows.append(
                check("source-full-output-operator", error <= 1, maximum_allowance_ratio=error)
            )
            errors = []
            for f, y in zip(frequencies, raw, strict=True):
                operator = 2j * np.pi * f * m - j
                errors.append(
                    np.max(abs(operator @ y - b))
                    / (np.linalg.norm(operator, np.inf) * np.max(abs(y)) + np.max(abs(b)))
                )
            rows.append(
                check(
                    "source-original-linear-system",
                    max(errors) <= 1e-10,
                    maximum=float(max(errors)),
                )
            )
    except Exception as exc:
        rows.append(
            check("missing-or-malformed-record", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return rows


def scalar_invariants(trace, data):
    rows = []
    try:
        v = {k: values(trace)[:, i] for i, k in enumerate(trace["response_order"])}
        omega = 2 * np.pi * np.asarray(trace["frequencies_Hz"])
        f, r = trace["faraday_C_mol"], trace["gas_constant_J_mol_K"]
        rows.append(
            check(
                "physical-constants",
                np.isfinite(f) and np.isfinite(r) and abs(f - F) < 1e-3 and abs(r - R) < 1e-8,
            )
        )
        expected_v = ocv(data["electrodes"][1]) - ocv(data["electrodes"][0])
        rows.append(
            check("authored-rest-voltage", abs(trace["rest_voltage_V"] - expected_v) <= 1e-10)
        )
        errors = {
            "unit-current": np.max(abs(v["current"] - 1)),
            "signed-charge": np.max(abs(1j * omega * v["charge"] - 1)),
            "signed-work": np.max(abs(1j * omega * v["work"] - expected_v))
            / max(1, abs(expected_v)),
        }
        for e, sign in zip(data["electrodes"], (1, -1), strict=True):
            n = e["name"]
            errors[n + ":Faraday"] = np.max(abs(1j * omega * f * v[n + ":amount"] + sign))
            errors[n + ":mean-inventory"] = np.max(
                abs(
                    v[n + ":theta"]
                    - v[n + ":amount"] / (e["active_volume_m3"] * e["maximum_concentration_mol_m3"])
                )
                / np.maximum(1, abs(v[n + ":theta"]))
            )
            errors[n + ":OCV-projection"] = np.max(
                abs(v[n + ":U"] - ocv(e, True) * v[n + ":theta"]) / np.maximum(1, abs(v[n + ":U"]))
            )
        n, p = v["negative:amount"], v["positive:amount"]
        errors["total-Li"] = np.max(abs(n + p) / np.maximum(1 / (f * omega), abs(n) + abs(p)))
        for name, error in errors.items():
            rows.append(
                check(name, np.isfinite(error) and error <= 1e-8, maximum=float(error), limit=1e-8)
            )
    except Exception as exc:
        rows.append(check("invariant-input-failure", False, exception=str(exc)))
    return rows


def profile_invariants(array, order, data, cells, *, continuous=False):
    rows = []
    v = {k: array[:, i] for i, k in enumerate(order)}
    area = data["area_m2"]
    nf = len(array)
    power = np.zeros(nf)
    errors = {}
    previous_end = None
    for region, (n, count) in enumerate(
        zip(("negative", "separator", "positive"), cells, strict=True)
    ):
        dx = data["lengths_m"][region] / count
        k = data["conductivity_S_m"] * data["porosities"][region] ** data["bruggeman"][region]
        ie = np.column_stack([v[f"{n}:ie:{i}"] for i in range(count + 1)])
        pe = np.column_stack([v[f"{n}:phi_e:{i}"] for i in range(count)])
        errors[n + ":face-frequency-independence"] = np.max(abs(ie - ie[0]))
        if previous_end is not None:
            errors[n + ":interface-continuity"] = np.max(abs(ie[:, 0] - previous_end))
        previous_end = ie[:, -1]
        # Exact quadratic integral of the piecewise-linear face current response.
        power += np.sum(
            (abs(ie[:, :-1]) ** 2 + np.real(ie[:, :-1] * ie[:, 1:].conj()) + abs(ie[:, 1:]) ** 2)
            * dx
            / (3 * k * area),
            axis=1,
        )
        if region == 1:
            errors["separator:constant-current"] = np.max(abs(ie - v["current"][:, None]))
            continue
        sign = 1 if region == 0 else -1
        e = data["electrodes"][region // 2]
        active_area = 3 * e["active_volume_m3"] / e["radius_m"]
        q = np.column_stack([v[f"{n}:q:{i}"] for i in range(count)])
        j = np.column_stack([v[f"{n}:j:{i}"] for i in range(count)])
        iss = np.column_stack([v[f"{n}:is:{i}"] for i in range(count + 1)])
        eta = np.column_stack([v[f"{n}:eta:{i}"] for i in range(count)])
        ps = np.column_stack([v[f"{n}:phi_s:{i}"] for i in range(count)])
        errors[n + ":total-reaction"] = np.max(abs(q.sum(axis=1) - sign * v["current"]))
        errors[n + ":face-reaction-balance"] = np.max(abs(np.diff(ie, axis=1) - q))
        if not continuous:
            errors[n + ":node-j-cell-q"] = np.max(abs(j * active_area / count - q))
        errors[n + ":parallel-current"] = np.max(abs(iss + ie - v["current"][:, None]))
        errors[n + ":eta-gauge"] = np.max(abs(ps - pe - v[n + ":U"][:, None] - eta))
        if region == 0:
            errors["negative:collector-no-ionic"] = np.max(abs(ie[:, 0]))
        else:
            errors["positive:collector-no-ionic"] = np.max(abs(ie[:, -1]))
        sigma = data["solid_conductivity_S_m"][region // 2]
        power += np.sum(
            (
                abs(iss[:, :-1]) ** 2
                + np.real(iss[:, :-1] * iss[:, 1:].conj())
                + abs(iss[:, 1:]) ** 2
            )
            * dx
            / (3 * sigma * area),
            axis=1,
        )
        power += np.sum(np.real(q * eta.conj()), axis=1)
    for name, error in errors.items():
        rows.append(
            check(name, np.isfinite(error) and error <= 1e-8, maximum=float(error), limit=1e-8)
        )
    rn = v["Z_ohm"] - v["negative:U"] + v["positive:U"]
    allowance = 1e-4 + 0.01 * abs(rn.real)
    error = abs(power - rn.real)
    rows.append(
        check(
            "electrical-dissipation",
            np.all(np.isfinite(power)) and np.all(power >= -1e-12) and np.all(error <= allowance),
            maximum_allowance_ratio=float(np.max(error / allowance)),
            dissipation_W_A2=power.tolist(),
            port_network_ohm=rn.real.tolist(),
            maximum_absolute_error=float(np.max(error)),
        )
    )
    return rows


def invariants(trace, data, grids=GRIDS):
    rows = scalar_invariants(trace, data)
    try:
        if trace["implementation"] == "analytic":
            net = trace["continuous_network"]
            error = abs(net["dissipation_W_A2"] - net["resistance_ohm"])
            rows.append(
                check(
                    "continuous-electrical-dissipation",
                    net["dissipation_W_A2"] >= -1e-12
                    and error <= 1e-4 + 0.01 * abs(net["resistance_ohm"]),
                    maximum=error,
                )
            )
            for label, cells in grids.items():
                r = trace["grid_responses"][label]
                array = np.array(r["response_real"]) + 1j * np.array(r["response_imag"])
                rows.extend(
                    dict(c, grid=label)
                    for c in profile_invariants(
                        array, r["response_order"], data, cells, continuous=True
                    )
                )
        else:
            rows.extend(
                profile_invariants(values(trace), trace["response_order"], data, trace["grid"])
            )
    except Exception as exc:
        rows.append(check("profile-invariant-input-failure", False, exception=str(exc)))
    return rows


def compare(left, right, *, grid_label=None):
    cells = left["grid"]
    if right["implementation"] == "analytic":
        r = right["grid_responses"][grid_label]
        if r["cells"] != cells:
            raise ValueError("continuous grid identity mismatch")
        a = values(left)
        b = np.array(r["response_real"]) + 1j * np.array(r["response_imag"])
        declared = schema(cells)
    elif cells == right["grid"]:
        a, b = values(left), values(right)
        declared = schema(cells)
    else:
        a, b = values(left)[:, :10], values(right)[:, :10]
        declared = schema()
    if a.shape != b.shape:
        raise ValueError("comparison shape mismatch")
    bounds = np.array([r[2] for r in declared]) + np.array([r[3] for r in declared]) * abs(b)
    errors = abs(a - b)
    return check(
        "response-comparison",
        np.all(np.isfinite(errors)) and np.all(errors <= bounds),
        maximum_allowance_ratio=float(np.max(errors / bounds)),
        response_order=[r[0] for r in declared],
        maximum_error_by_output=np.max(errors, axis=0).tolist(),
        errors=errors.tolist(),
        bounds=bounds.tolist(),
    )


def domain_check(trace, data, name, cells):
    tokens = {
        n: n.split("-")[0] + ":Li"
        for n in ("negative-lower", "negative-upper", "positive-lower", "positive-upper")
    }
    tokens.update(
        {
            "missing-Li": "inventory and coordinates",
            "missing-QW": "inventory and coordinates",
            "Faraday-inconsistent": "Faraday",
            "voltage-face": "stopping boundary",
            "zero-frequency": "frequencies",
            "nan-frequency": "frequencies",
        }
    )
    try:
        spec, state, freq = refusal_input(data, name, cells)
        good = (
            trace["implementation"] == "refusal"
            and trace["name"] == name
            and trace["grid"] == list(cells)
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
    if index < 8:
        eligible = set()
        for name, kind, grid in plan:
            trace = routes.get(name, {})
            rows = structural(trace, data, kind, GRIDS[grid] if grid else None)
            checks.extend(dict(r, route=name) for r in rows)
            if all(r["passed"] for r in rows):
                eligible.add(name)
                checks.extend(dict(r, route=name) for r in invariants(trace, data))
        pairs = [(f"native_{g}", f"reference_{g}", g) for g in GRIDS]
        pairs += [(f"{k}_coarse", f"{k}_fine", None) for k in ("native", "reference")]
        pairs += [(f"{k}_{g}", "analytic", g) for k in ("native", "reference") for g in GRIDS]
        for left, right, grid in pairs:
            try:
                row = (
                    compare(routes[left], routes[right], grid_label=grid)
                    if {left, right} <= eligible
                    else check("response-comparison", False, reason="incomplete-route")
                )
            except Exception as exc:
                row = check("response-comparison", False, exception=str(exc))
            checks.append(dict(row, left=left, right=right))
    else:
        checks.extend(
            domain_check(routes.get(name, {}), data, name, GRIDS[grid]) for name, _, grid in plan
        )
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}
