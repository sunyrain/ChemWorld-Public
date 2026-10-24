"""Fail-closed completeness, conservation and fixed finite-grid comparisons."""

import numpy as np
from battery_composite_impedance_cases import (
    ATOL,
    FREQUENCIES,
    GRIDS,
    PHASES,
    RTOL,
    F,
    binding,
    gradient_profile,
    native_spec,
    phase_values,
    response_order,
    route_plan,
)

DEFINITION = "Z=-delta_V/delta_I; positive discharge; exp(+i*2*pi*f*t)"


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, (float, int)) or bool(np.isfinite(value))


def complete_snapshot(value):
    return (
        isinstance(value, dict)
        and set(value)
        == {"amounts_mol", "coordinates", "algebraic_values", "time_s", "temperature_K", "volume_L"}
        and all(
            isinstance(value[k], dict) for k in ("amounts_mol", "coordinates", "algebraic_values")
        )
        and finite_tree(value)
        and value["time_s"] == 0.0
        and value["temperature_K"] > 0
        and value["volume_L"] > 0
    )


def canonical_native_variables(spec):
    # Read the original declaration, without a second frequency or time solve.
    from chemworld.physchem.battery_models import battery_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    coupled = battery_system(spec, ScalarControlProgram(((0.0, 0.0),), "A"))
    return (
        [
            {"name": k, "unit": "mol", "scale": coupled.inventory_scales_mol[k]}
            for k in coupled.thermodynamics.inventory_keys
        ]
        + [{"name": v.name, "unit": v.unit, "scale": v.scale} for v in coupled.coordinates]
        + [{"name": v.name, "unit": v.unit, "scale": v.scale} for v in coupled.algebraic.variables]
    )


def check(name, passed, **details):
    return {"name": name, "passed": bool(passed), **details}


def values(trace):
    return np.asarray([p["response_real"] for p in trace["points"]], float) + 1j * np.asarray(
        [p["response_imag"] for p in trace["points"]], float
    )


def structural(trace, data, kind, cells, single=False, frequencies=FREQUENCIES):
    checks = []
    try:
        order = response_order(data, 0 if kind == "analytic" else cells, single=single)
        checks.append(
            check(
                "route-metadata",
                trace["implementation"] == kind
                and trace["single_material"] is single
                and trace["grid"] == cells
                and trace["response_order"] == order
                and trace["binding"] == binding(data, kind, cells, single),
            )
        )
        actual = np.asarray(trace["frequencies_Hz"])
        points = trace["points"]
        checks.append(
            check(
                "complete-frequencies",
                len(points) == len(frequencies)
                and actual.shape == np.shape(frequencies)
                and np.array_equal(actual, frequencies)
                and np.array_equal([p["frequency_Hz"] for p in points], frequencies),
            )
        )
        array = values(trace)
        checks.append(
            check(
                "finite-complete-outputs",
                array.shape == (len(frequencies), len(order))
                and np.all(np.isfinite(array))
                and trace["success"] is True
                and all(p["success"] is True for p in points),
            )
        )
        if kind == "analytic":
            return checks
        checks.append(check("definition", trace["definition"] == DEFINITION))
        initial = np.asarray(trace["initial_state"], float)
        residual = np.asarray(trace["stationary_residual"], float)
        raw = np.array([p["state_response_real"] for p in points]) + 1j * np.array(
            [p["state_response_imag"] for p in points]
        )
        checks.append(
            check(
                "raw-state-complete",
                initial.ndim == 1
                and len(initial) > 0
                and residual.shape == initial.shape
                and raw.shape == (len(frequencies), len(initial))
                and np.all(np.isfinite(initial))
                and np.all(np.isfinite(residual))
                and np.all(np.isfinite(raw)),
            )
        )
        linear = np.array([p["relative_linear_residual"] for p in points], float)
        checks.append(
            check(
                "linear-residual",
                np.all(np.isfinite(linear)) and np.all((linear >= 0) & (linear <= 1e-10)),
                maximum=float(max(linear)),
            )
        )
        phases = phase_values(data, single)
        phase_ids = [k for k in PHASES if k in phases]
        capacities = np.array(
            [
                phases[k]["maximum_concentration_mol_m3"] * phases[k]["active_volume_m3"]
                for k in phase_ids
            ]
        )
        edges = np.linspace(0.0, 1.0, cells + 1)
        weights = np.diff(edges**3)
        profiles = array[:, 1 + 4 * len(phase_ids) :].reshape(
            len(frequencies), len(phase_ids), cells
        )
        measured = array[:, 1 : 1 + 4 * len(phase_ids)].reshape(len(frequencies), len(phase_ids), 4)
        mean_error = np.max(np.abs(np.sum(profiles * weights, axis=2) - measured[:, :, 2]))
        checks.append(
            check(
                "radial-average-identity",
                mean_error <= 1e-12 + 1e-10 * max(1.0, np.max(np.abs(measured[:, :, 2]))),
                maximum=float(mean_error),
            )
        )
        if kind == "native":
            variables = trace["variables"]
            names = [v["name"] for v in variables]
            scales = np.array([v["scale"] for v in variables])
            diagnostics = trace["diagnostics"]
            checks.append(
                check(
                    "native-state-and-diagnostics",
                    len(variables) == len(initial)
                    and len(set(names)) == len(names)
                    and np.all(np.isfinite(scales))
                    and np.all(scales > 0)
                    and trace["state_unchanged"] is True
                    and trace["state_before"] == trace["state_after"]
                    and diagnostics["maximum_scaled_stationary_residual"] <= 1e-9
                    and diagnostics["stationary_tolerance"] == 1e-9
                    and diagnostics["linear_tolerance"] == 1e-10
                    and diagnostics["physical_time_advanced_s"] == 0.0
                    and diagnostics["completed"]
                    == diagnostics["passed"]
                    == diagnostics["total"]
                    == len(frequencies)
                    and all(
                        np.isfinite(diagnostics[k])
                        for k in (
                            "maximum_scaled_stationary_residual",
                            "stationary_tolerance",
                            "linear_tolerance",
                        )
                    ),
                )
            )
            spec = native_spec(data, cells, single=single)
            from battery_composite_impedance_traces import snapshot

            checks.append(
                check(
                    "native-canonical-declaration",
                    variables == canonical_native_variables(spec)
                    and complete_snapshot(trace["state_before"])
                    and trace["state_before"] == snapshot(spec.initial_state())
                    and finite_tree(diagnostics),
                )
            )
            residual_scales = np.asarray(trace["residual_scales"], float)
            checks.append(
                check(
                    "native-raw-stationarity",
                    residual_scales.shape == residual.shape
                    and np.all(np.isfinite(residual_scales))
                    and np.all(residual_scales > 0)
                    and np.max(abs(residual / residual_scales)) <= 1e-9,
                )
            )
            bins = (
                [
                    ("host_a", spec.negative, spec.negative.keys("negative")),
                    ("cathode", spec.positive, spec.positive.keys("positive")),
                ]
                if single
                else [
                    (p.phase_id, particle, keys)
                    for p, (_, _, particle, keys, _) in zip(
                        spec.phases, spec.particle_bins, strict=True
                    )
                ]
            )
            index = {name: i for i, name in enumerate(names)}
            checks.append(
                check(
                    "native-initial-QW", all(initial[index[k]] == 0.0 for k in spec.coordinate_keys)
                )
            )
            from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC

            q = (
                np.column_stack(
                    (raw[:, index[BATTERY_ALGEBRAIC[0]]], -raw[:, index[BATTERY_ALGEBRAIC[0]]])
                )
                if single
                else raw[:, [index[k] for k in spec.algebraic_names]] @ spec.reaction_map.T
            )
            observation_error = 0.0
            for ordinal, (identity, particle, keys) in enumerate(bins):
                column = phase_ids.index(identity)
                shell = raw[:, [index[k] for k in keys]]
                inferred = shell / (
                    particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3
                )
                observation_error = max(
                    observation_error,
                    float(np.max(abs(inferred - profiles[:, column]))),
                    float(np.max(abs(q[:, ordinal] - measured[:, column, 0]))),
                )
                expected_initial = (
                    particle.mesh.cell_volumes_m3
                    * particle.maximum_concentration_mol_m3
                    * phases[identity]["initial_fraction"]
                )
                checks.append(
                    check(
                        "native-authored-initial-" + identity,
                        np.allclose(
                            initial[[index[k] for k in keys]],
                            expected_initial,
                            rtol=1e-12,
                            atol=1e-18,
                        ),
                    )
                )
            checks.append(
                check(
                    "native-raw-observation-identity",
                    observation_error <= 1e-10,
                    maximum=observation_error,
                )
            )
        else:
            n_diff = trace["differential_count"]
            layout = trace["source_state_layout"]
            all_indices = [i for v in layout for i in v["indices"]]
            checks.append(
                check(
                    "source-complete-state-layout",
                    trace["state_count"] == len(initial)
                    and sorted(all_indices) == list(range(len(initial)))
                    and all(
                        v["name"]
                        and v["domains"] is not None
                        and np.isfinite(v["scale"])
                        and v["scale"] > 0
                        and np.isfinite(v["reference"])
                        for v in layout
                    ),
                )
            )
            checks.append(
                check(
                    "source-stationarity",
                    trace["pybamm"] == "25.4.2"
                    and trace["casadi"] == "3.6.7"
                    and trace["stationary_tolerance"] == 1e-6
                    and trace["linear_tolerance"] == 1e-10
                    and 0 < n_diff <= len(initial)
                    and max(abs(residual)) <= 1e-6
                    and trace["maximum_stationary_rhs"] == max(abs(residual[:n_diff]))
                    and trace["maximum_stationary_algebraic"]
                    == max(abs(residual[n_diff:]), default=0.0),
                )
            )
            expected_names = [
                "Voltage [V]",
                *(
                    f"qualification:{k}:{m}"
                    for k in phase_ids
                    for m in ("q", "surface", "average", "amount")
                ),
                *(f"eis:{k}:profile:{i}" for k in phase_ids for i in range(cells)),
            ]
            baseline = np.asarray(trace["initial_outputs"])
            checks.append(
                check(
                    "source-output-map",
                    trace["output_names"] == expected_names
                    and baseline.shape == (len(order),)
                    and np.all(np.isfinite(baseline)),
                )
            )
            observations = baseline[1 : 1 + 4 * len(phase_ids)].reshape(len(phase_ids), 4)
            initial_theta = np.array([phases[k]["initial_fraction"] for k in phase_ids])
            checks.append(
                check(
                    "source-authored-initial",
                    np.max(abs(observations[:, 0])) <= 1e-8
                    and np.allclose(observations[:, 1:3], initial_theta[:, None], rtol=0, atol=1e-9)
                    and np.allclose(
                        observations[:, 3], initial_theta * capacities, rtol=1e-10, atol=1e-12
                    ),
                )
            )
            # Scaled source concentration states must reproduce every retained radial response.
            for identity in phase_ids:
                source_position = list(phases).index(identity)
                domain = phases[identity]["electrode"]
                label = (
                    ""
                    if single or domain == "positive"
                    else ("primary " if source_position == 0 else "secondary ")
                )
                found = [
                    v
                    for v in layout
                    if f"{domain} {label}particle concentration" in v["name"].lower()
                ]
                if len(found) != 1:
                    checks.append(check("source-particle-state-" + identity, False))
                    continue
                entry = found[0]
                inferred = (
                    raw[:, entry["indices"]]
                    * entry["scale"]
                    / phases[identity]["maximum_concentration_mol_m3"]
                )
                expected = profiles[:, phase_ids.index(identity)]
                original = (
                    initial[entry["indices"]] * entry["scale"] + entry["reference"]
                ) / phases[identity]["maximum_concentration_mol_m3"]
                checks.append(
                    check(
                        "source-raw-observation-" + identity,
                        inferred.shape == expected.shape
                        and np.allclose(inferred, expected, rtol=1e-10, atol=1e-12)
                        and np.allclose(
                            original, phases[identity]["initial_fraction"], rtol=0, atol=1e-10
                        ),
                    )
                )
    except Exception as exc:
        checks.append(
            check("malformed-or-missing-record", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return checks


def invariants(trace, data):
    checks = []
    try:
        array, single = values(trace), trace["single_material"]
        phases = phase_values(data, single)
        identities = [k for k in PHASES if k in phases]
        measured = array[:, 1 : 1 + 4 * len(identities)].reshape(len(array), len(identities), 4)
        q, mean, amount = measured[:, :, 0], measured[:, :, 2], measured[:, :, 3]
        omega = 2 * np.pi * np.asarray(trace["frequencies_Hz"])
        faraday = trace["faraday_C_mol"]
        checks.append(check("Faraday-constant", np.isfinite(faraday) and abs(faraday - F) < 1e-3))
        errors = {
            "negative-current": float(np.max(abs(q[:, :-1].sum(axis=1) - 1))),
            "positive-current": float(np.max(abs(q[:, -1] + 1))),
            "phase-Faraday": float(
                np.max(abs(1j * omega[:, None] * faraday * amount + q) / np.maximum(1.0, abs(q)))
            ),
            "total-Li": float(
                np.max(
                    abs(amount.sum(axis=1)) / np.maximum(1 / (F * omega), abs(amount).sum(axis=1))
                )
            ),
        }
        checks.extend(
            check(name, error <= 1e-8, maximum=error, limit=1e-8) for name, error in errors.items()
        )
        capacities = np.array(
            [
                phases[k]["active_volume_m3"] * phases[k]["maximum_concentration_mol_m3"]
                for k in identities
            ]
        )
        error = float(np.max(abs(amount / capacities - mean) / np.maximum(1.0, abs(mean))))
        checks.append(check("mean-amount-identity", error <= 1.01e-10, maximum=error))
    except Exception as exc:
        checks.append(
            check("invariant-input-failure", False, exception=f"{type(exc).__name__}: {exc}")
        )
    return checks


def compare(left, right, *, single=False, exact=False, resistance_shift=0.0):
    a, b = values(left), values(right)
    if single:
        # Only E01 identical material limits: sum negative q/n; common theta.
        a = np.column_stack(
            (a[:, 0], a[:, 1] + a[:, 5], a[:, 2], a[:, 3], a[:, 4] + a[:, 8], a[:, 9:13])
        )
        b = b[:, :9]
        atol = ATOL[:9]
    else:
        a, b, atol = a[:, :13], b[:, :13], ATOL
    a = a.copy()
    a[:, 0] -= resistance_shift
    bounds = (1e-8 + 1e-8 * abs(b)) if exact else atol + RTOL * abs(b)
    errors = abs(a - b)
    return check(
        "response-comparison",
        np.all(np.isfinite(errors)) and np.all(errors <= bounds),
        maximum_allowance_ratio=float(np.max(errors / bounds)),
        maximum_error_by_output=np.max(errors, axis=0).tolist(),
        errors=errors.tolist(),
        bounds=bounds.tolist(),
    )


def assess_unit(index, data, routes):
    checks = []
    expected = route_plan(index)
    checks.append(check("complete-route-denominator", set(routes) == {v[0] for v in expected}))
    if index <= 10:
        eligible = set()
        for name, kind, grid, single in expected:
            if name not in routes:
                continue
            trace = routes[name]
            rows = structural(trace, data, kind, GRIDS[grid] if grid else None, single)
            checks.extend({**row, "route": name} for row in rows)
            if all(row["passed"] for row in rows):
                eligible.add(name)
                checks.extend({**row, "route": name} for row in invariants(trace, data))
        comparisons = [(f"native_{g}", f"reference_{g}", False) for g in GRIDS]
        comparisons += [(f"{k}_coarse", f"{k}_fine", False) for k in ("native", "reference")]
        comparisons += [
            (f"{k}_{g}", "analytic", False) for k in ("native", "reference") for g in GRIDS
        ]
        if index == 1:
            comparisons += [
                (f"{k}_{g}", f"old_{k}_{g}", True) for k in ("native", "reference") for g in GRIDS
            ]
        for left, right, single in comparisons:
            try:
                result = (
                    compare(routes[left], routes[right], single=single)
                    if {left, right} <= eligible
                    else check("response-comparison", False, reason="incomplete-route")
                )
            except Exception as exc:
                result = check("response-comparison", False, exception=str(exc))
            checks.append({**result, "left": left, "right": right})
        if index == 8 and "analytic" in eligible:
            error = float(
                np.max(abs(values(routes["analytic"])[:, 0] - values(routes["analytic"])[0, 0]))
            )
            checks.append(check("flat-OCP-constant-spectrum", error <= 1e-12, maximum=error))
    elif index == 11:
        for name, kind, grid, reason in expected:
            trace = routes.get(name, {})
            try:
                correct = (
                    trace["rejected"] is True
                    and trace["success"] is False
                    and trace["points"] == []
                    and trace["implementation"] == kind
                    and trace["grid"] == GRIDS[grid]
                    and trace["single_material"] is False
                    and trace["binding"] == binding(data, kind, GRIDS[grid], False, reason)
                    and trace["definition"] == DEFINITION
                )
                residual = np.asarray(trace["stationary_residual"], float)
                initial = np.asarray(trace["initial_state"], float)
                correct = (
                    correct
                    and initial.ndim == 1
                    and len(initial) > 0
                    and residual.shape == initial.shape
                    and np.all(np.isfinite(initial))
                    and np.all(np.isfinite(residual))
                )
                if kind == "native":
                    before = trace["state_before"]
                    scales = np.asarray(trace["residual_scales"], float)
                    correct = (
                        correct
                        and trace["exception_type"] == "ResidualDomainError"
                        and "nonstationary" in trace["exception"]
                        and trace["state_unchanged"] is True
                        and complete_snapshot(before)
                        and before == trace["state_after"]
                        and scales.shape == residual.shape
                        and np.all(np.isfinite(scales))
                        and np.all(scales > 0)
                        and np.max(abs(residual / scales)) > 1e-9
                    )
                    authored = {**data, "phases": [dict(v) for v in data["phases"]]}
                    if reason == "internal-current":
                        authored["phases"][1]["initial_fraction"] = 0.6
                    spec = native_spec(authored, GRIDS[grid])
                    expected_amounts = dict(spec.initial_state().amounts_mol)
                    if reason == "radial-gradient":
                        _, _, particle, keys, _ = spec.particle_bins[0]
                        expected_amounts.update(
                            zip(
                                keys,
                                gradient_profile(GRIDS[grid], 0.6)
                                * particle.mesh.cell_volumes_m3
                                * particle.maximum_concentration_mol_m3,
                                strict=True,
                            )
                        )
                    correct = (
                        correct
                        and trace["variables"] == canonical_native_variables(spec)
                        and before["amounts_mol"] == expected_amounts
                        and before["coordinates"] == spec.initial_coordinates()
                    )
                    vindex = {v["name"]: i for i, v in enumerate(trace["variables"])}
                    correct = correct and all(
                        initial[vindex[k]] == v for k, v in expected_amounts.items()
                    )
                else:
                    baseline = np.asarray(trace["initial_outputs"], float)
                    required_order = response_order(data, GRIDS[grid])
                    correct = (
                        correct
                        and trace["failure"] == "reference_not_stationary"
                        and trace["response_order"] == required_order
                        and baseline.shape == (len(required_order),)
                        and np.all(np.isfinite(baseline))
                        and max(abs(residual)) > 1e-6
                        and finite_tree(trace["source_state_layout"])
                        and bool(trace["source_state_layout"])
                    )
                    profiles = baseline[13:].reshape(3, GRIDS[grid])
                    target = (
                        gradient_profile(GRIDS[grid], 0.6)
                        if reason == "radial-gradient"
                        else np.full(GRIDS[grid], 0.6)
                    )
                    correct = correct and np.allclose(profiles[0], target, atol=1e-10, rtol=0)
                    if reason == "internal-current":
                        correct = (
                            correct
                            and np.allclose(profiles[1], 0.6, atol=1e-10, rtol=0)
                            and abs(baseline[1]) > 1e-6
                        )
                    else:
                        correct = correct and np.ptp(profiles[0]) > 0.01
                    layout = trace["source_state_layout"]
                    correct = (
                        correct
                        and trace["state_count"] == len(initial)
                        and sorted(i for entry in layout for i in entry["indices"])
                        == list(range(len(initial)))
                    )
                    for phase_position, phase in enumerate(data["phases"]):
                        label = (
                            ("primary " if phase_position == 0 else "secondary ")
                            if phase_position < 2
                            else ""
                        )
                        found = [
                            entry
                            for entry in layout
                            if f"{phase['electrode']} {label}particle concentration"
                            in entry["name"].lower()
                        ]
                        if len(found) != 1:
                            correct = False
                            continue
                        entry = found[0]
                        original = (
                            initial[entry["indices"]] * entry["scale"] + entry["reference"]
                        ) / phase["maximum_concentration_mol_m3"]
                        correct = (
                            correct
                            and entry["scale"] > 0
                            and np.allclose(original, profiles[phase_position], rtol=0, atol=1e-10)
                        )
            except Exception:
                correct = False
            checks.append(check("reject-nonstationary", correct, route=name))
    else:
        tokens = {
            "empty-frequency": "frequencies",
            "zero-frequency": "frequencies",
            "negative-frequency": "frequencies",
            "nan-frequency": "frequencies",
            "missing-inventory": "inventory",
            "missing-coordinates": "coordinates",
            "negative-inventory": "negative",
            "voltage-boundary": "stopping boundary",
        }
        for name, _, _, _ in expected:
            trace = routes.get(name, {})
            from battery_composite_impedance_traces import snapshot

            expected_before = snapshot(native_spec(data, 80).initial_state())
            if name == "missing-inventory":
                del expected_before["amounts_mol"][next(iter(expected_before["amounts_mol"]))]
            elif name == "missing-coordinates":
                expected_before["coordinates"] = {}
            checks.append(
                check(
                    "domain-refusal",
                    trace.get("implementation") == "refusal"
                    and trace.get("name") == name
                    and trace.get("rejected") is True
                    and trace.get("binding") == binding(data, "refusal", 80)
                    and trace.get("exception_type") in {"ValueError", "ResidualDomainError"}
                    and tokens[name] in trace.get("exception", "")
                    and trace.get("state_unchanged") is True
                    and complete_snapshot(trace.get("state_before"))
                    and trace.get("state_before") == expected_before
                    and trace.get("rejection_stage")
                    == ("input-construction" if name == "negative-inventory" else "analysis")
                    and trace.get("state_before") == trace.get("state_after"),
                    route=name,
                )
            )
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks}
