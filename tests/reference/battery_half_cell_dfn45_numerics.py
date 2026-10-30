"""Explicit source-only numerical policies; no qualification gate is changed."""

from copy import deepcopy

import numpy as np

NUMERICAL_POLICIES = ("legacy", "certified-rest-seed", "fv-current-rows", "combined")


def validate_numerical_policy(policy):
    if not isinstance(policy, str) or policy not in NUMERICAL_POLICIES:
        raise ValueError("unknown half-cell source numerical policy")
    return policy


def _read(expression, time_s, state, inputs):
    # PyBaMM StateVector boolean indexing expects NumPy, while the actual
    # Python consistency chain supplies a CasADi DM. Do not change that chain;
    # convert only this read-only expression evaluation argument.
    numeric = np.asarray(state, dtype=float).reshape(-1, 1)
    return np.asarray(expression.evaluate(t=time_s, y=numeric, inputs=inputs), dtype=float)


def _flat(value):
    return np.asarray(value, dtype=float).reshape(-1)


def install_row_policy(pybamm, model, mesh, parameters, declarations, policy, maximum_current_A):
    """Map actual discretized charge rows to dimensionless FV cell currents.

    The original Lratio-conditioned concatenation is retained, including its
    arithmetic. CasADi differentiates the mapped expressions, so the analytic
    Jacobian receives exactly the same positive left map.
    """
    validate_numerical_policy(policy)
    use_rows = policy in ("fv-current-rows", "combined")
    area = float(parameters["Electrode height [m]"] * parameters["Electrode width [m]"])
    dx_s = np.diff(np.asarray(mesh["separator"].edges))
    dx_p = np.diff(np.asarray(mesh["positive electrode"].edges))
    length = float(np.sum(dx_s) + np.sum(dx_p))
    if (
        not np.isfinite([area, length, maximum_current_A]).all()
        or min(area, length, maximum_current_A) <= 0
        or not np.all(np.r_[dx_s, dx_p] > 0)
    ):
        raise ValueError("FV current rows require positive actual geometry and current scale")
    legacy = model.concatenated_algebraic
    weights = np.ones(model.len_alg)
    legacy_units = [None] * model.len_alg
    solver_units = [None] * model.len_alg
    blocks = []
    by_name = {variable.name: variable for variable in model.algebraic}
    for declaration in declarations:
        variable = by_name[declaration["name"]]
        slices = model.y_slices[variable]
        indices = np.concatenate([np.arange(s.start, s.stop) for s in slices]) - model.len_rhs
        if declaration["charge_equation"]:
            if variable.name == "Positive electrode potential [V]":
                widths = dx_p
            elif variable.name == "Electrolyte potential [V]":
                widths = np.r_[dx_s, dx_p]
            else:
                raise ValueError("unrecognized charge block for FV current rows")
            if len(widths) != len(indices):
                raise ValueError("actual mesh widths do not match the source charge layout")
            physical_weights = area * widths / (length**2 * maximum_current_A)
            actual = physical_weights if use_rows else np.ones(len(indices))
            old_unit, new_unit = "A.m-1", "1" if use_rows else "A.m-1"
        else:
            widths = np.empty(0)
            physical_weights = np.ones(len(indices))
            actual = physical_weights
            old_unit = new_unit = "1"
        weights[indices] = actual
        for index in indices:
            legacy_units[index], solver_units[index] = old_unit, new_unit
        blocks.append(
            {
                "name": variable.name,
                "algebraic_indices": indices.tolist(),
                "actual_cell_widths_m": widths.tolist(),
                "legacy_units": old_unit,
                "solver_units": new_unit,
                "physical_fv_current_weights": physical_weights.tolist(),
                "solver_row_weights": actual.tolist(),
            }
        )
        if use_rows:
            expression = model.algebraic[variable]
            model.algebraic[variable] = (
                pybamm.Vector(actual, domains=expression.domains) * expression
            )
    if None in legacy_units or not np.isfinite(weights).all() or not np.all(weights > 0):
        raise ValueError("incomplete or nonpositive source numerical row policy")
    if use_rows:
        model.concatenated_algebraic = pybamm.Vector(weights) * legacy
    return legacy, {
        "name": policy,
        "certified_rest_seed": policy in ("certified-rest-seed", "combined"),
        "fv_current_rows": use_rows,
        "area_m2": area,
        "salt_length_m": length,
        "current_scale_A": float(maximum_current_A),
        "row_formula": "g_solver=A*dx/(Lsalt^2*Imax)*g_legacy for charge; control unchanged",
        "legacy_residual_gate": (
            "source_algebraic retains the original Lratio^2 residual; 1e-8 unchanged"
        ),
        "solver_root_tol_applies_to": (
            "solver_algebraic only; the legacy source_algebraic 1e-8 audit is independent"
        ),
        "solver_row_weights": weights.tolist(),
        "source_algebraic_units": legacy_units,
        "solver_algebraic_units": solver_units,
        "blocks": blocks,
        "jacobian_policy": "CasADi differentiates the same positively mapped algebraic expressions",
        "cpp_internal_initialization": "unavailable in the pinned public IDAKLU interface",
    }


def certified_rest_guess(pybamm, built, time_s, state, inputs, names):
    """Return an algebraic guess only after an exact uniform CC0 certificate."""
    raw = _flat(state)
    model, parameters = built["model"], built["parameter_values"]
    mode, current = names
    if inputs.get(current) != 0.0 or (not built["basic"] and inputs.get(mode) != 0.0):
        return raw.copy(), {"certified": False, "reason": "control is not exact prescribed CC0"}
    ce = _flat(_read(model.variables[built["alias_prefix"] + "ce"], time_s, state, inputs))
    cs = _flat(_read(model.variables[built["alias_prefix"] + "cs"], time_s, state, inputs))
    cmax = float(parameters["Maximum concentration in positive electrode [mol.m-3]"])
    if (
        not len(ce)
        or not len(cs)
        or not np.isfinite(np.r_[raw, ce, cs]).all()
        or not np.all(ce == ce[0])
        or not np.all(cs == cs[0])
        or ce[0] <= 0
        or not 0 < cs[0] < cmax
    ):
        return raw.copy(), {
            "certified": False, "reason": "actual ce/cs are not uniform interior stocks"
        }
    ocp = _flat(
        _read(model.variables[built["alias_prefix"] + "positive_ocp_V"], time_s, state, inputs)
    )
    if not len(ocp) or not np.isfinite(ocp).all() or not np.all(ocp == ocp[0]):
        return raw.copy(), {
            "certified": False, "reason": "complete source OCP is not uniform finite"
        }
    guess, covered = raw.copy(), set()
    supported = {
        "Current variable [A]": 0.0,
        "Positive electrode potential [V]": float(ocp[0]),
        "Separator electrolyte potential [V]": 0.0,
        "Positive electrolyte potential [V]": 0.0,
    }
    for variable, slices in model.y_slices.items():
        if (
            isinstance(variable, pybamm.Concatenation)
            or not slices
            or slices[0].start < model.len_rhs
        ):
            continue
        if variable.name not in supported:
            return raw.copy(), {
                "certified": False, "reason": "unsupported additional algebraic state"
            }
        scale = float(parameters.evaluate(variable.scale))
        reference = float(parameters.evaluate(variable.reference))
        if not np.isfinite([scale, reference]).all() or scale <= 0:
            raise ValueError("source algebraic state has invalid physical scale")
        for part in slices:
            guess[part] = (supported[variable.name] - reference) / scale
            covered.update(range(part.start, part.stop))
    if covered != set(range(model.len_rhs, len(raw))):
        return raw.copy(), {
            "certified": False, "reason": "incomplete algebraic equilibrium certificate"
        }
    if not np.array_equal(guess[: model.len_rhs], raw[: model.len_rhs]):
        raise ValueError("rest seed attempted to modify a differential stock")
    return guess, {
        "certified": True,
        "reason": (
            "actual uniform ce/cs and exact CC0; source symmetric BV and zero-flux equilibrium"
        ),
        "ce_mol_m3": float(ce[0]),
        "cs_mol_m3": float(cs[0]),
        "stoichiometry": float(cs[0] / cmax),
        "complete_source_ocp_V": float(ocp[0]),
        "ocp_scope": (
            "actual released expression including clip, endpoint barrier and temperature term"
        ),
        "differential_entries_unchanged": True,
        "algebraic_indices": sorted(covered),
        "accepted_equilibrium_forced": False,
    }


def make_source_solver(pybamm, policy, **kwargs):
    """Keep legacy construction intact; observe normal IC for explicit policies."""
    validate_numerical_policy(policy)
    if policy == "legacy":
        return pybamm.IDAKLUSolver(**kwargs)

    class ObservedIDAKLUSolver(pybamm.IDAKLUSolver):
        def __init__(self):
            super().__init__(**kwargs)
            self.source_ic_observations = []
            self.source_numerical_context = None

        def _set_consistent_initialization(self, model, time, inputs_dict):
            built, names = self.source_numerical_context
            before = _flat(model.y0).copy()
            attempt = {
                "time_s": float(time),
                "inputs": deepcopy(inputs_dict),
                "upstream_default_initial_y": _flat(
                    model.concatenated_initial_conditions.evaluate(inputs=inputs_dict)
                ).tolist(),
                "python_pre_consistency_y": before.tolist(),
                "cpp_post_consistency_y": "unavailable in pinned public IDAKLU interface",
                "cpp_post_consistency_ydot": "unavailable in pinned public IDAKLU interface",
            }
            self.source_ic_observations.append(attempt)
            if policy in ("certified-rest-seed", "combined"):
                guess, certificate = certified_rest_guess(
                    pybamm, built, time, model.y0, inputs_dict, names
                )
                attempt["rest_seed_certificate"] = certificate
                if certificate["certified"]:
                    model.y0 = guess.reshape(-1, 1)
            else:
                attempt["rest_seed_certificate"] = {
                    "certified": False, "reason": "seed policy disabled"
                }
            attempt["actual_python_root_guess_y"] = _flat(model.y0).tolist()
            try:
                super()._set_consistent_initialization(model, time, inputs_dict)
            except Exception as exc:
                attempt["python_consistency_exception"] = {
                    "type": type(exc).__name__, "message": str(exc)
                }
                raise
            attempt["python_post_consistency_y"] = _flat(model.y0).tolist()
            attempt["python_ida_initial_y"] = _flat(model.y0full).tolist()
            attempt["python_ida_initial_ydot"] = _flat(model.ydot0full).tolist()
            attempt["differential_entries_unchanged"] = bool(
                np.array_equal(_flat(model.y0)[: model.len_rhs], before[: model.len_rhs])
            )

    return ObservedIDAKLUSolver()
