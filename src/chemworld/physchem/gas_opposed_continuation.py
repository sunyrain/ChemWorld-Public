"""Finite original-BVP pseudo-arclength queries; no world or stability claim.

This query retains all opposed-flow coordinates and appends one
numerical predictor hyperplane. State derivatives remain explicitly numerical;
the fixed-ratio feed parameter column and numerical hyperplane are analytic.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from math import isfinite
from time import perf_counter, process_time
from traceback import format_exc
from typing import ClassVar

import numpy as np
from scipy import sparse

from chemworld.physchem.gas_opposed_flow import (
    GasOpposedFlowSpec,
    _opposed_fresh_checks,
    _OpposedResidual,
)
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.residual_system import (
    DEFAULT_ROOT_POLICY,
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEvaluator,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    bounds,
    linear_solve,
    solve_root,
)

MODEL = "opposed-gas-flow-pseudo-arclength-v1"
_FRESH_NAMES = (
    "all_species_including_omitted_balance",
    "both_incoming_all_species",
    "EOS",
    "both_trace_normalizations",
    "energy",
    "radial_momentum",
    "continuity",
    "both_G_boundary_values_and_signs",
    "chemical_mass",
    "global_mass",
    "global_mass_pde",
    "global_chemical_mass",
    "global_elements",
    "global_elements_pde",
    "chemical_elements",
    "global_chemical_elements",
    "global_enthalpy",
    "axial_centerline_low_Mach",
)
_QUERY_DEFAULTS = {
    "derivative_step": 6e-6,
    "maximum_unknowns": 512,
    "species_tolerance_mol_s": 1e-9,
    "energy_tolerance_W": 1e-6,
    "pressure_tolerance_Pa": 1e-4,
    "mass_tolerance_kg_s": 1e-10,
    "momentum_tolerance_kg_s2": 1e-8,
    "element_tolerance_mol_s": 1e-9,
    "jacobian_strategy": "sparse-colored",
    "thermo_memo_max_entries": None,
    "residual_only_trials": False,
}


def _number(value, name, *, positive=False):
    if type(value) not in (int, float) or not isfinite(value) or (positive and value <= 0):
        raise ValueError(
            f"{name} must be an explicit finite {'positive ' if positive else ''}number"
        )
    return float(value)


def _integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _options(options):
    if options is None:
        return dict(_QUERY_DEFAULTS)
    if not isinstance(options, Mapping) or set(options) - set(_QUERY_DEFAULTS):
        raise ValueError("unknown opposed query option")
    result = {**_QUERY_DEFAULTS, **options}
    for name in (
        "derivative_step",
        "species_tolerance_mol_s",
        "energy_tolerance_W",
        "pressure_tolerance_Pa",
        "mass_tolerance_kg_s",
        "momentum_tolerance_kg_s2",
        "element_tolerance_mol_s",
    ):
        _number(result[name], name, positive=True)
    _integer(result["maximum_unknowns"], "maximum_unknowns")
    if result["jacobian_strategy"] not in ("dense", "sparse-colored"):
        raise ValueError("declare dense or sparse-colored numerical state derivatives")
    if type(result["residual_only_trials"]) is not bool:
        raise ValueError("residual_only_trials must be boolean")
    if result["thermo_memo_max_entries"] is not None:
        _integer(result["thermo_memo_max_entries"], "thermo_memo_max_entries")
    return result


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _exception(error):
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error),
        "traceback": format_exc(),
    }


def _feed_parameter_column(assembly, base_spec):
    """Exact F_alpha at fixed physical y, including both signed G boundaries."""
    expected = {
        f"feed:{side}:{species}": sign
        * base_spec.mesh.area_m2
        * mdot
        * assembly.feed_y[index][species]
        / assembly.masses[species]
        for index, (side, sign, mdot) in enumerate(
            (
                ("left", -1, base_spec.left_mass_flux_kg_m2_s),
                ("right", 1, base_spec.right_mass_flux_kg_m2_s),
            )
        )
        for species in assembly.active
    }
    expected.update(
        {
            "G-boundary:left": -base_spec.mesh.area_m2 * base_spec.left_mass_flux_kg_m2_s,
            "G-boundary:right": base_spec.mesh.area_m2 * base_spec.right_mass_flux_kg_m2_s,
        }
    )
    names = [equation.name for equation in assembly.equations]
    if len(set(names)) != len(names) or not set(expected) <= set(names):
        raise ValueError("original opposed feed equation layout is incomplete")
    return np.asarray([expected.get(name, 0.0) for name in names])


def _scaled_matrix(jacobian, parameter_column, state_scales, row_scales, alpha_scale):
    state = sparse.diags(1 / row_scales) @ sparse.csc_matrix(jacobian) @ sparse.diags(state_scales)
    parameter = sparse.csc_matrix((parameter_column * alpha_scale / row_scales)[:, None])
    return sparse.hstack((state, parameter), format="csc")


def _tangent(matrix, previous, direction, tolerance):
    """A sparse square solve; never dense SVD, pseudo-inverse or least squares."""
    size = matrix.shape[1]
    if matrix.shape[0] + 1 != size:
        raise ValueError("a branch tangent requires an m by (m+1) matrix")
    if previous is None:
        state = linear_solve(matrix[:, :-1], -matrix[:, -1].toarray().ravel())
        tangent = np.r_[state, 1.0]
        tangent *= direction
    else:
        bordered = sparse.vstack((matrix, sparse.csc_matrix(previous[None, :])), format="csc")
        tangent = linear_solve(bordered, np.r_[np.zeros(size - 1), 1.0])
    norm = float(np.linalg.norm(tangent))
    if not isfinite(norm) or norm == 0:
        raise ResidualNumericalError("unresolved_tangent_norm")
    tangent /= norm
    if previous is None and abs(tangent[-1]) <= 64 * size * np.finfo(float).eps:
        raise ResidualNumericalError("unresolved_initial_parameter_direction")
    alignment = 1.0 if previous is None else float(tangent @ previous)
    if alignment < 0:
        tangent *= -1
        alignment *= -1
    defect = float(np.max(np.abs(matrix @ tangent), initial=0.0))
    if not isfinite(defect) or defect > tolerance:
        raise ResidualNumericalError("unresolved_tangent_residual")
    return tangent, alignment, defect


class _OpposedBranchProblem:
    """Query-owned full network and a bounded exact-alpha assembly LRU.

    Rebuilding an assembly never resets the outer residual or wall budget.
    Constructor/preflight work is recorded separately from F evaluations.
    """

    scope = "original-opposed-BVP-with-original-18-fresh-checks"
    derivative_modes: ClassVar = {
        "state": "finite_difference",
        "feed_scale": "analytic",
        "arclength_row": "analytic",
    }

    def __init__(
        self,
        spec,
        network,
        initial,
        seed_alpha,
        options,
        maximum_evaluations,
        maximum_wall_time_s,
        cache_entries,
        progress,
    ):
        self.spec = GasOpposedFlowSpec.from_dict(deepcopy(spec.to_dict()))
        self.network = ReactionNetworkSpec.from_dict(deepcopy(network.to_dict()))
        self.initial_profile = deepcopy(initial)
        self.options = options
        self.maximum_evaluations = maximum_evaluations
        self.maximum_wall_time_s = maximum_wall_time_s
        self.cache_entries = cache_entries
        self.progress = progress
        self.started, self.cpu_started = perf_counter(), process_time()
        self.calls, self.assembly_builds, self.evictions = 0, 0, 0
        self.jacobian_calls, self.parameter_calls = 0, 0
        self.derivative_records = []
        self.cache, self.build_records = OrderedDict(), []
        self.last_trial = None
        first = self._assembly(seed_alpha)
        self.variables, self.equations = tuple(first.variables), tuple(first.equations)
        self.initial = first.initial.copy()
        if len(self.initial) > options["maximum_unknowns"]:
            raise ValueError("opposed continuation exceeds explicit maximum_unknowns")
        self.parameter = _feed_parameter_column(first, self.spec)
        self.layout = self._layout(first)
        widths = np.diff(np.asarray(self.spec.mesh.faces_m, dtype=float))
        length = float(self.spec.mesh.faces_m[-1] - self.spec.mesh.faces_m[0])
        cells = widths / length
        dual = np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2] / length
        self.metric_weights = np.r_[
            np.repeat(cells, first.size), cells, cells, dual, 1.0, np.full(2 * first.size, 0.5)
        ]
        if (
            self.metric_weights.shape != first.initial.shape
            or not np.all(np.isfinite(self.metric_weights))
            or np.any(self.metric_weights <= 0)
        ):
            raise ValueError(
                "fixed geometric continuation weights must be finite positive and complete"
            )
        self.default_state_scales = np.asarray([v.scale for v in first.variables]) / np.sqrt(
            self.metric_weights
        )

    @staticmethod
    def _layout(assembly):
        return (
            tuple((v.name, v.unit, v.role, v.lower, v.upper) for v in assembly.variables),
            tuple((e.name, e.unit) for e in assembly.equations),
        )

    def check_budget(self):
        if perf_counter() - self.started > self.maximum_wall_time_s:
            raise ResidualDomainError("continuation_cooperative_wall_budget")

    def _assembly(self, alpha):
        self.check_budget()
        _number(alpha, "actual feed scale", positive=True)
        if alpha in self.cache:
            self.cache.move_to_end(alpha)
            return self.cache[alpha]
        actual = replace(
            self.spec,
            left_mass_flux_kg_m2_s=alpha * self.spec.left_mass_flux_kg_m2_s,
            right_mass_flux_kg_m2_s=alpha * self.spec.right_mass_flux_kg_m2_s,
        )
        started = perf_counter()
        result = _OpposedResidual(
            actual,
            self.network,
            self.initial_profile,
            self.maximum_evaluations,
            None,
            thermo_memo_max_entries=self.options["thermo_memo_max_entries"],
        )
        self.assembly_builds += 1
        if hasattr(self, "layout") and self._layout(result) != self.layout:
            raise ValueError("actual-alpha assembly changed original physical layout or bounds")
        self.build_records.append(
            {
                "index": self.assembly_builds,
                "alpha": alpha,
                "wall_time_s": perf_counter() - started,
                "preflight_role": "original-constructor-not-an-F-call",
            }
        )
        self.cache[alpha] = result
        if len(self.cache) > self.cache_entries:
            self.cache.popitem(last=False)
            self.evictions += 1
        self.check_budget()
        return result

    def _begin(self, y, alpha, kind):
        self.check_budget()
        if self.calls >= self.maximum_evaluations:
            raise ResidualDomainError("continuation_total_F_budget")
        self.calls += 1
        self.last_trial = {
            "kind": kind,
            "alpha": alpha,
            "value": np.asarray(y).tolist(),
            "F": None,
            "role": "last-evaluated-trial-not-root-result",
        }
        if self.progress is not None:
            self.progress(
                {
                    "event": "opposed-continuation-F",
                    "evaluations": self.calls,
                    "maximum_evaluations": self.maximum_evaluations,
                    "assembly_builds": self.assembly_builds,
                }
            )

    def value(self, y, alpha):
        self._begin(y, alpha, "residual")
        result = self._assembly(alpha).evaluate(
            y, retain_details=not self.options["residual_only_trials"]
        )[0]
        self.last_trial["F"] = result.tolist()
        self.check_budget()
        return result

    def jacobian(self, y, alpha, fixed_scales):
        self.jacobian_calls += 1
        assembly = self._assembly(alpha)
        variables = tuple(
            replace(v, scale=float(d))
            for v, d in zip(assembly.variables, fixed_scales, strict=True)
        )
        system = ResidualSystem(
            MODEL,
            variables,
            tuple(assembly.equations),
            lambda t, point, rate, p: self.value(point, float(p[0])),
            ResidualDerivatives(
                "hybrid",
                parameter=lambda t, point, rate, p: self.parameter[:, None],
                relative_step=self.options["derivative_step"],
                state_sparsity=(
                    assembly.state_sparsity()
                    if self.options["jacobian_strategy"] == "sparse-colored"
                    else None
                ),
            ),
            parameters=(ResidualVariable("feed-scale", "1", 1, role="parameter", lower=0),),
            parameter_values=(alpha,),
        )
        evaluator = ResidualEvaluator(system)
        try:
            return evaluator.derivative("state", 0, y, np.zeros(len(y)))
        finally:
            self.derivative_records.append({"alpha": alpha, **evaluator.diagnostics()})

    def parameter_column(self, y, alpha):
        self.parameter_calls += 1
        return self.parameter.copy()

    def fresh(self, y, alpha, root_policy):
        self._begin(y, alpha, "complete-fresh-candidate")
        assembly = self._assembly(alpha)
        residual, profile = assembly.evaluate(y)
        self.last_trial["F"] = residual.tolist()
        checks = _opposed_fresh_checks(
            assembly,
            profile,
            root_policy=root_policy,
            **{
                name: self.options[name]
                for name in (
                    "species_tolerance_mol_s",
                    "energy_tolerance_W",
                    "pressure_tolerance_Pa",
                    "mass_tolerance_kg_s",
                    "momentum_tolerance_kg_s2",
                    "element_tolerance_mol_s",
                )
            },
        )
        if tuple(checks) != _FRESH_NAMES or any(
            type(v) not in (bool, np.bool_) for v in checks.values()
        ):
            raise ValueError("original complete eighteen fresh checks are required")
        self.check_budget()
        return residual, profile, checks

    def readout(self, profile, alpha):
        assembly = self._assembly(alpha)
        left = assembly.response(
            assembly.spec.left_inlet_temperature_K, profile["left_inlet_trace_mole_fractions"]
        )
        right = assembly.response(
            assembly.spec.right_inlet_temperature_K, profile["right_inlet_trace_mole_fractions"]
        )
        g = np.asarray(profile["face_mass_fluxes_kg_m2_s"])
        cell_u = (g[:-1] + g[1:]) / (2 * np.asarray(profile["cell_density_kg_m3"]))
        coordinates = np.r_[
            self.spec.mesh.faces_m[0], self.spec.mesh.centers_m, self.spec.mesh.faces_m[-1]
        ]
        velocity = np.r_[g[0] / left.density_kg_m3, cell_u, g[-1] / right.density_kg_m3]
        slopes = np.diff(velocity) / np.diff(coordinates)
        self.check_budget()
        return {
            "axial_velocity_coordinates_m": coordinates.tolist(),
            "axial_velocity_m_s": velocity.tolist(),
            "piecewise_linear_du_dz_per_s": slopes.tolist(),
            "maximum_abs_piecewise_linear_du_dz_per_s": float(np.max(np.abs(slopes))),
            "endpoint_mean_abs_du_dz_per_s": float(
                abs(velocity[-1] - velocity[0]) / (coordinates[-1] - coordinates[0])
            ),
            "feed_scale_is_not_strain_rate": True,
            "original_upwind_G_signs": np.sign(g).astype(int).tolist(),
            "local_differentiability_certified": False,
        }

    def spec_record(self, alpha):
        return replace(
            self.spec,
            left_mass_flux_kg_m2_s=alpha * self.spec.left_mass_flux_kg_m2_s,
            right_mass_flux_kg_m2_s=alpha * self.spec.right_mass_flux_kg_m2_s,
        ).to_dict()

    def statistics(self):
        return {
            "total_F_evaluations_including_fresh": self.calls,
            "maximum_evaluations": self.maximum_evaluations,
            "assembly_builds": self.assembly_builds,
            "assembly_evictions": self.evictions,
            "assembly_cache_entries": len(self.cache),
            "assembly_cache_capacity": self.cache_entries,
            "assembly_build_records": self.build_records,
            "state_jacobian_calls": self.jacobian_calls,
            "analytic_parameter_column_calls": self.parameter_calls,
            "derivative_records": self.derivative_records,
            "wall_time_s": perf_counter() - self.started,
            "process_cpu_time_s": process_time() - self.cpu_started,
            "cooperative_wall_budget_s": self.maximum_wall_time_s,
            "hard_process_budget_enforced_here": False,
            "peak_memory_bytes": None,
            "resource_limit": "cooperative checkpoints cannot interrupt an active kernel",
            "live_thermo_memo_statistics": {
                str(alpha): a.thermo_memo.statistics()
                for alpha, a in self.cache.items()
                if a.thermo_memo is not None
            },
        }


def _trace_problem(
    problem,
    *,
    seed_alpha,
    feed_scale_bounds,
    feed_scale_metric,
    state_scales,
    direction,
    step_size,
    minimum_step,
    maximum_points,
    maximum_retries,
    root_policy,
    tangent_tolerance,
    maximum_correction_ratio,
    minimum_tangent_alignment,
    progress=None,
):
    """Shared-root continuation core; authored test adapters are not gas physics."""
    variables = tuple(problem.variables)
    count = len(variables)
    default_scales = np.asarray(
        getattr(problem, "default_state_scales", [v.scale for v in variables])
    )
    if state_scales is None:
        d = default_scales.copy()
    elif isinstance(state_scales, Mapping):
        if set(state_scales) != {v.name for v in variables}:
            raise ValueError("state scale mapping must cover the complete physical layout")
        d = np.asarray(
            [_number(state_scales[v.name], "state scale", positive=True) for v in variables]
        )
    else:
        if not isinstance(state_scales, tuple | list | np.ndarray):
            raise ValueError("state scales require a complete numeric sequence")
        d = np.asarray(
            [
                _number(v.item() if isinstance(v, np.generic) else v, "state scale", positive=True)
                for v in state_scales
            ]
        )
    e = np.asarray([equation.scale for equation in problem.equations])
    if d.shape != (count,) or not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("state scales must be a positive finite complete vector")
    if e.shape != (count,) or not np.all(np.isfinite(e)) or np.any(e <= 0):
        raise ValueError("seed equation scales must be positive finite and complete")
    low, high = bounds(variables)
    attempts, points, folds = [], [], []
    result = {
        "schema": MODEL,
        "scope": problem.scope,
        "success": False,
        "status": "partial",
        "termination_reason": "",
        "points": points,
        "attempts": attempts,
        "fold_brackets": folds,
        "initial_profile": deepcopy(getattr(problem, "initial_profile", None)),
        "initial_value": problem.initial.tolist(),
        "seed_alpha": seed_alpha,
        "fixed_scales": {
            "state": d.tolist(),
            "equations": e.tolist(),
            "feed_scale_metric": feed_scale_metric,
            "source": "fixed-seed-declaration-no-pointwise-rescaling",
            "metric_weights": (
                None
                if state_scales is not None
                else _plain(getattr(problem, "metric_weights", np.ones(count)))
            ),
            "metric_definition": (
                "caller-supplied-complete-D"
                if state_scales is not None
                else "seed-scales-with-fixed-cell-and-face-dual-weights"
            ),
            "state_variables": [v.name for v in variables],
            "equations_layout": [r.name for r in problem.equations],
        },
        "policy": {
            "feed_scale_bounds": list(feed_scale_bounds),
            "direction": direction,
            "step_size": step_size,
            "minimum_step": minimum_step,
            "maximum_points": maximum_points,
            "maximum_retries": maximum_retries,
            "maximum_correction_ratio": maximum_correction_ratio,
            "minimum_tangent_alignment": minimum_tangent_alignment,
            "tangent_tolerance": tangent_tolerance,
        },
        "numerical_coordinate_row": "previous_tangent_dot(u-predictor)=0",
        "derivative_modes": deepcopy(
            getattr(
                problem,
                "derivative_modes",
                {
                    "state": "problem-supplied-see-scope",
                    "feed_scale": "problem-supplied-see-scope",
                    "arclength_row": "analytic",
                },
            )
        ),
        "physical_phase_added": False,
        "stability": "not-assessed",
        "extinction": "not-established",
        "experimental_elapsed_s": 0.0,
        "live_operations_executed": 0,
        "accepted_states_projected": False,
    }
    origin = problem.initial.copy()

    def coordinates(u):
        return origin + d * np.asarray(u[:-1]), seed_alpha + feed_scale_metric * float(u[-1])

    def jacobian(y, alpha):
        return _scaled_matrix(
            problem.jacobian(y, alpha, d),
            problem.parameter_column(y, alpha),
            d,
            e,
            feed_scale_metric,
        )

    def root_call(initial, prior_tangent, predictor, attempt):
        def residual(u):
            y, alpha = coordinates(u) if prior_tangent is not None else (np.asarray(u), seed_alpha)
            attempt["last_evaluated_trial"] = {
                "value": y.tolist(),
                "alpha": alpha,
                "F": None,
                "role": "last-evaluated-trial-not-root-result",
                "spec": problem.spec_record(alpha),
            }
            raw = problem.value(y, alpha)
            attempt["last_evaluated_trial"]["F"] = raw.tolist()
            return (
                np.r_[raw / e, prior_tangent @ (u - predictor)]
                if prior_tangent is not None
                else raw / e
            )

        def derivative(u):
            if prior_tangent is None:
                return sparse.diags(1 / e) @ sparse.csc_matrix(problem.jacobian(u, seed_alpha, d))
            y, alpha = coordinates(u)
            return sparse.vstack(
                (jacobian(y, alpha), sparse.csc_matrix(prior_tangent[None, :])), format="csc"
            )

        def root_progress(event):
            attempt["root_progress"].append(_plain(event))
            if progress is not None:
                progress({**event, "continuation_attempt": len(attempts) - 1})

        if prior_tangent is None:
            root_variables = tuple(
                replace(v, scale=float(scale)) for v, scale in zip(variables, d, strict=True)
            )
        else:
            root_variables = (
                *tuple(
                    ResidualVariable(
                        v.name,
                        "1",
                        1,
                        role="algebraic",
                        lower=None if not isfinite(lo) else float((lo - x) / scale),
                        upper=None if not isfinite(hi) else float((hi - x) / scale),
                    )
                    for v, lo, hi, x, scale in zip(variables, low, high, origin, d, strict=True)
                ),
                ResidualVariable(
                    "feed-scale-coordinate",
                    "1",
                    1,
                    role="algebraic",
                    lower=(feed_scale_bounds[0] - seed_alpha) / feed_scale_metric,
                    upper=(feed_scale_bounds[1] - seed_alpha) / feed_scale_metric,
                ),
            )
        root_wall, root_cpu = perf_counter(), process_time()
        try:
            solved = solve_root(
                residual,
                initial,
                jacobian=derivative,
                variables=root_variables,
                residual_scales=np.ones(len(root_variables)),
                policy=root_policy,
                progress=root_progress,
            )
        finally:
            attempt["root_resources"] = {
                "wall_time_s": perf_counter() - root_wall,
                "process_cpu_time_s": process_time() - root_cpu,
                "peak_memory_bytes": None,
            }
            attempt["last_original_F_evaluation"] = deepcopy(getattr(problem, "last_trial", None))
        attempt["root"] = {
            "success": solved.success,
            "failure_reason": solved.failure_reason,
            "diagnostics": deepcopy(solved.diagnostics),
            "value": None if solved.value is None else solved.value.tolist(),
        }
        return solved

    def fresh_candidate(y, alpha, attempt):
        raw, profile, checks = problem.fresh(y, alpha, root_policy)
        candidate = {
            "alpha": alpha,
            "value": y.tolist(),
            "spec": problem.spec_record(alpha),
            "F": raw.tolist(),
            "scaled_residual_norm": float(np.max(np.abs(raw / e))),
            "profile": profile,
            "physical_checks": checks,
            "readout": (problem.readout(profile, alpha) if hasattr(problem, "readout") else {}),
            "branch_classification": "unclassified-no-stability-inference",
        }
        attempt["complete_candidate"] = candidate
        if any(type(v) not in (bool, np.bool_) for v in checks.values()) or not all(
            checks.values()
        ):
            raise ResidualDomainError("complete_fresh_physical_balance_failed")
        if candidate["scaled_residual_norm"] > root_policy.tolerance:
            raise ResidualDomainError("fresh_scaled_original_residual_failed")
        return candidate

    try:
        seed = {
            "kind": "seed",
            "alpha": seed_alpha,
            "initial": problem.initial.tolist(),
            "accepted": False,
            "root": None,
            "root_progress": [],
        }
        attempts.append(seed)
        solved = root_call(problem.initial, None, None, seed)
        if not solved.success:
            result["termination_reason"] = "seed_root_failed:" + solved.failure_reason
            return _finish(result, problem)
        y = np.asarray(solved.value)
        candidate = fresh_candidate(y, seed_alpha, seed)
        origin = y.copy()
        u = np.zeros(count + 1)
        tangent, _, defect = _tangent(jacobian(y, seed_alpha), None, direction, tangent_tolerance)
        candidate.update(
            index=0,
            scaled_coordinates=u.tolist(),
            tangent=tangent.tolist(),
            tangent_residual=defect,
            cumulative_chord_length=0.0,
            predictor_arclength=0.0,
        )
        points.append(candidate)
        seed["accepted"] = True
        arc, chord, last_sign_index = 0.0, 0.0, 0
        while len(points) < maximum_points:
            accepted = False
            step = step_size
            for retry in range(maximum_retries + 1):
                if step < minimum_step:
                    result["termination_reason"] = "minimum_step"
                    break
                predictor = u + step * tangent
                attempt = {
                    "kind": "corrector",
                    "from_index": len(points) - 1,
                    "retry": retry,
                    "step": step,
                    "predictor": predictor.tolist(),
                    "predictor_physical": _plain(coordinates(predictor)),
                    "accepted": False,
                    "root": None,
                    "root_progress": [],
                }
                attempts.append(attempt)
                try:
                    solved = root_call(predictor, tangent, predictor, attempt)
                    if not solved.success:
                        raise ResidualDomainError(solved.failure_reason)
                    candidate_u = np.asarray(solved.value)
                    next_y, next_alpha = coordinates(candidate_u)
                    correction = float(np.linalg.norm(candidate_u - predictor))
                    attempt["correction_norm"] = correction
                    if correction > maximum_correction_ratio * step:
                        raise ResidualDomainError("corrector_outside_local_branch_neighborhood")
                    next_tangent, alignment, defect = _tangent(
                        jacobian(next_y, next_alpha), tangent, direction, tangent_tolerance
                    )
                    attempt["tangent_alignment"] = alignment
                    if alignment < minimum_tangent_alignment:
                        raise ResidualDomainError(
                            "tangent_rotation_outside_local_branch_neighborhood"
                        )
                    candidate = fresh_candidate(next_y, next_alpha, attempt)
                    next_arc = arc + step
                    next_chord = chord + float(np.linalg.norm(candidate_u - u))
                    candidate.update(
                        index=len(points),
                        scaled_coordinates=candidate_u.tolist(),
                        tangent=next_tangent.tolist(),
                        tangent_residual=defect,
                        cumulative_chord_length=next_chord,
                        predictor_arclength=next_arc,
                    )
                    points.append(candidate)
                    threshold = 64 * (count + 1) * np.finfo(float).eps
                    if abs(next_tangent[-1]) > threshold:
                        prior = points[last_sign_index]
                        if prior["tangent"][-1] * next_tangent[-1] < 0:
                            folds.append(
                                {
                                    "indices": [last_sign_index, len(points) - 1],
                                    "alpha_values": [prior["alpha"], next_alpha],
                                    "classification": (
                                        "finite_tangent_alpha_sign_change_bracket_only"
                                    ),
                                    "extinction_established": False,
                                    "alpha_interval_is_an_extremum_enclosure": False,
                                }
                            )
                        last_sign_index = len(points) - 1
                    u, tangent, arc, chord = candidate_u, next_tangent, next_arc, next_chord
                    attempt["accepted"] = True
                    accepted = True
                    break
                except (ResidualDomainError, ResidualNumericalError) as error:
                    attempt["reason"] = str(error)
                    attempt["exception"] = _exception(error)
                    step *= 0.5
            if not accepted:
                result["termination_reason"] = (
                    result["termination_reason"] or "corrector_retries_exhausted"
                )
                break
        else:
            result.update(
                success=True,
                status="finite-point-budget-completed",
                termination_reason="maximum_points",
            )
    except Exception as error:
        result["termination_reason"] = "continuation_exception"
        result["exception"] = _exception(error)
        if attempts:
            attempts[-1].setdefault("exception", deepcopy(result["exception"]))
    return _finish(result, problem)


def _finish(result, problem):
    result["statistics"] = problem.statistics()
    result["last_evaluated_trial"] = deepcopy(getattr(problem, "last_trial", None))
    result["accepted_points"] = len(result["points"])
    result["rejected_attempts"] = sum(not a["accepted"] for a in result["attempts"])
    return _plain(result)


def trace_opposed_gas_branch(
    spec,
    network,
    initial_profile,
    *,
    seed_alpha=1.0,
    feed_scale_bounds,
    feed_scale_metric,
    state_scales=None,
    direction=1,
    step_size=0.05,
    minimum_step=0.001,
    maximum_points=32,
    maximum_retries=4,
    maximum_evaluations=100000,
    maximum_wall_time_s=900.0,
    root_policy=DEFAULT_ROOT_POLICY,
    query_options=None,
    assembly_cache_entries=2,
    tangent_tolerance=1e-7,
    maximum_correction_ratio=0.5,
    minimum_tangent_alignment=0.85,
    progress=None,
):
    """Trace a finite fixed-ratio-feed branch, retaining failed and fresh candidates.

    A completed point budget is a numerical trace completion, not evidence for
    global branch coverage, extinction, dynamic stability or mesh convergence.
    This query never installs profiles into a world or advances its clock.
    """
    if not isinstance(spec, GasOpposedFlowSpec) or not isinstance(network, ReactionNetworkSpec):
        raise ValueError("declare original opposed-flow spec and reaction network")
    if not isinstance(root_policy, RootPolicy):
        raise ValueError("declare the shared RootPolicy")
    if not isinstance(feed_scale_bounds, tuple | list) or len(feed_scale_bounds) != 2:
        raise ValueError("feed_scale_bounds require two endpoints")
    alpha_bounds = tuple(_number(v, "feed scale bound", positive=True) for v in feed_scale_bounds)
    seed_alpha = _number(seed_alpha, "seed_alpha", positive=True)
    if (
        not alpha_bounds[0] < alpha_bounds[1]
        or not alpha_bounds[0] <= seed_alpha <= alpha_bounds[1]
    ):
        raise ValueError("seed_alpha must lie in ordered positive feed bounds")
    for name, value in (
        ("feed_scale_metric", feed_scale_metric),
        ("step_size", step_size),
        ("minimum_step", minimum_step),
        ("maximum_wall_time_s", maximum_wall_time_s),
        ("tangent_tolerance", tangent_tolerance),
        ("maximum_correction_ratio", maximum_correction_ratio),
    ):
        _number(value, name, positive=True)
    if minimum_step > step_size:
        raise ValueError("minimum_step exceeds step_size")
    if type(direction) is not int or direction not in (-1, 1):
        raise ValueError("direction must be exactly -1 or +1")
    if not 0 < _number(minimum_tangent_alignment, "minimum_tangent_alignment") <= 1:
        raise ValueError("minimum_tangent_alignment must lie in (0,1]")
    _integer(maximum_points, "maximum_points")
    _integer(maximum_retries, "maximum_retries", minimum=0)
    _integer(maximum_evaluations, "maximum_evaluations")
    _integer(assembly_cache_entries, "assembly_cache_entries")
    if assembly_cache_entries > 4:
        raise ValueError("assembly cache capacity must be bounded by four")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable")
    options = _options(query_options)
    problem = _OpposedBranchProblem(
        spec,
        network,
        initial_profile,
        seed_alpha,
        options,
        maximum_evaluations,
        maximum_wall_time_s,
        assembly_cache_entries,
        progress,
    )
    return _trace_problem(
        problem,
        seed_alpha=seed_alpha,
        feed_scale_bounds=alpha_bounds,
        feed_scale_metric=feed_scale_metric,
        state_scales=state_scales,
        direction=direction,
        step_size=step_size,
        minimum_step=minimum_step,
        maximum_points=maximum_points,
        maximum_retries=maximum_retries,
        root_policy=root_policy,
        tangent_tolerance=tangent_tolerance,
        maximum_correction_ratio=maximum_correction_ratio,
        minimum_tangent_alignment=minimum_tangent_alignment,
        progress=progress,
    )


__all__ = ["trace_opposed_gas_branch"]
