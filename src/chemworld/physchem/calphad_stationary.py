"""Constrained CEF stationarity and potentials on the common nonlinear root."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from math import isfinite
from types import MappingProxyType

import numpy as np
from scipy.linalg import null_space, qr

from chemworld.physchem.calphad_response import (
    branch_diagnostics,
    derivatives,
    site_labels,
    site_response,
    site_values,
    unpack_sites,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualNumericalError,
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_root,
)
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R


@dataclass(frozen=True)
class CefStationaryPolicy:
    rank_tolerance: float = 1e-10
    absolute_stationarity_J_mol: float = 1e-6
    relative_stationarity: float = 1e-9
    site_floor: float = 1e-10
    maximum_saddle_escapes: int = 2
    root: RootPolicy = field(default_factory=lambda: RootPolicy(maximum_iterations=50))

    def __post_init__(self):
        if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in (
            self.rank_tolerance, self.absolute_stationarity_J_mol,
            self.relative_stationarity, self.site_floor,
        )) or self.rank_tolerance >= 1 or self.site_floor >= .5:
            raise ValueError("CEF stationarity policy needs positive finite tolerances and bounds")
        if not isinstance(self.root, RootPolicy):
            raise ValueError("CEF stationarity requires the common RootPolicy")
        if type(self.maximum_saddle_escapes) is not int or self.maximum_saddle_escapes < 0:
            raise ValueError("CEF saddle escape count must be a nonnegative integer")


DEFAULT_CEF_STATIONARY_POLICY = CefStationaryPolicy()


def independent_constraints(jacobian, tolerance):
    scales = np.maximum(np.max(np.abs(jacobian), axis=1), 1.0)
    normalized = jacobian / scales[:, None]
    _, r, pivots = qr(normalized.T, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank = int(np.sum(diagonal > tolerance * max(float(diagonal.max(initial=0)), 1.0)))
    indices = np.sort(pivots[:rank])
    return indices, scales[indices]


def _interior(model, fractions):
    fractions = model._fractions(fractions)
    if any(y <= 0 for subl in fractions for y in subl.values()):
        raise ValueError("CEF constrained derivatives require an interior site constitution")
    return fractions


def _stability(hessian, jacobian, policy):
    scales = np.maximum(np.max(np.abs(jacobian), axis=1), 1.0)
    tangent = null_space(jacobian / scales[:, None], rcond=policy.rank_tolerance)
    if tangent.shape[1] == 0:
        return None, True, None
    projected = tangent.T @ hessian @ tangent
    eigenvalues, eigenvectors = np.linalg.eigh((projected+projected.T)/2)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        raise ValueError("CEF internal stability spectrum is nonfinite")
    minimum = float(eigenvalues[0])
    tolerance = policy.absolute_stationarity_J_mol + policy.relative_stationarity * max(
        1.0, float(np.max(np.abs(hessian))),
    )
    direction = tangent @ eigenvectors[:, 0]
    significant = np.flatnonzero(np.abs(direction) > 1e-12)
    if len(significant) and direction[significant[0]] < 0:
        direction = -direction
    return minimum, minimum >= -tolerance, direction


def _curvature(hessian, jacobian, policy):
    minimum, stable, _ = _stability(hessian, jacobian, policy)
    return minimum, stable


def _escape_saddle(model, sites, direction, elements, target, temperature, pressure, policy):
    baseline = float(site_values(model, tuple(sites), temperature, pressure, elements)[0])
    best = None
    energy_tolerance = max(1e-8, 1e-12*abs(baseline))
    for sign in (-1, 1):
        proposed = sign*direction
        room = min(
            ((1-x)/d if d > 0 else (x-policy.site_floor)/(-d))
            for x, d in zip(sites, proposed, strict=True) if abs(d) > 1e-14
        )
        if room <= 0:
            continue
        for numerator in range(1, 8):
            fraction = numerator/8
            trial = sites+fraction*room*proposed
            try:
                output = np.asarray(site_values(
                    model, tuple(float(x) for x in trial), temperature, pressure, elements,
                ), dtype=float)
                residuals = np.r_[output[len(elements)+1:], output[1:len(elements)+1]-target]
                if np.max(np.abs(residuals)) > 10*policy.root.tolerance:
                    continue
                if output[0] < baseline-energy_tolerance and (
                    best is None or output[0] < best[1]["trial_energy_J_mol"]
                ):
                    best = trial, {"starting_energy_J_mol": baseline,
                                   "trial_energy_J_mol": float(output[0]),
                                   "bound_fraction": fraction, "direction_sign": sign,
                                   "maximum_constraint_residual": float(np.max(np.abs(residuals)))}
            except (ValueError, ArithmeticError):
                continue
    return best


@dataclass(frozen=True)
class CefChemicalPotentials:
    status: str
    values_J_mol: object | None
    matrix_rank: int
    unknowns: int
    condition_number: float | None
    maximum_residual_J_mol: float | None
    minimum_internal_curvature_J_mol: float | None
    internal_minimum: bool | None
    diagnostics: tuple[str, ...]


def chemical_potentials(model, fractions, temperature_K, pressure_Pa=101325.0, *,
                        require_local_minimum=True, allow_one_sided=False,
                        policy=DEFAULT_CEF_STATIONARY_POLICY):
    if (type(require_local_minimum) is not bool or type(allow_one_sided) is not bool
            or not isinstance(policy, CefStationaryPolicy)):
        raise ValueError("CEF stationarity requires boolean choices and a CefStationaryPolicy")
    fractions = _interior(model, fractions)
    temperature, pressure = model._potentials(temperature_K, pressure_Pa)
    branches = branch_diagnostics(model, fractions, temperature, pressure)
    if any("parameter sign boundary" in d for d in branches):
        raise ValueError(
            "CEF chemical potentials are undefined at a magnetic parameter sign boundary",
        )
    if branches and not allow_one_sided:
        raise ValueError("CEF constrained Hessian needs explicit one-sided branch acceptance")
    sites = [fractions[i][s] for i, s in site_labels(model)]
    elements, values, j, h = site_response(model, sites, temperature, pressure)
    ne = len(elements)
    indices, scales = independent_constraints(j[ne+1:], policy.rank_tolerance)
    cj, ch = j[ne+1:][indices]/scales[:, None], h[ne+1:][indices]/scales[:, None, None]
    matrix = np.block([[j[1:ne+1].T, cj.T],
                       [values[None, 1:ne+1], np.zeros((1, len(indices)))]])
    rhs = np.r_[j[0], values[0]]
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > policy.rank_tolerance * singular[0]))
    unknowns = matrix.shape[1]
    diagnostics = (*branches, "single-phase internal stationarity is not multiphase stability")
    if rank < unknowns:
        return CefChemicalPotentials(
            "underdetermined", None, rank, unknowns, None, None, None, None,
            (*diagnostics,
             "phase composition constraints do not determine individual elemental mu"),
        )
    solution = np.linalg.lstsq(matrix, rhs, rcond=policy.rank_tolerance)[0]
    residual = float(np.max(np.abs(matrix @ solution-rhs)))
    condition = float(singular[0]/singular[-1])
    tolerance = policy.absolute_stationarity_J_mol + policy.relative_stationarity * max(
        1.0, float(np.max(np.abs(rhs))),
    )
    if residual > tolerance:
        return CefChemicalPotentials("nonstationary", None, rank, unknowns, condition,
                                     residual, None, None, diagnostics)
    lagrangian = h[0]-np.einsum("i,ijk->jk", solution[:ne], h[1:ne+1])
    lagrangian -= np.einsum("i,ijk->jk", solution[ne:], ch)
    minimum, stable = _curvature(lagrangian, np.vstack((j[1:ne+1], cj)), policy)
    if not stable and require_local_minimum:
        return CefChemicalPotentials("unstable", None, rank, unknowns, condition,
                                     residual, minimum, False, diagnostics)
    return CefChemicalPotentials(
        "determined", MappingProxyType({e: float(v) for e, v
                                         in zip(elements, solution[:ne], strict=True)}),
        rank, unknowns, condition, residual, minimum, stable,
        (*diagnostics, *(("stationary saddle explicitly permitted",) if not stable else ())),
    )


@dataclass(frozen=True)
class CefConstitutionResult:
    success: bool
    failure_reason: str
    site_fractions: tuple | None
    properties: object | None
    chemical_potentials: CefChemicalPotentials | None
    relaxed_heat_capacity_J_mol_K: float | None
    site_temperature_derivatives_per_K: tuple | None
    diagnostics: dict


def _constraints(values, jacobian, hessian, target):
    ne = len(target)
    c = np.r_[values[ne+1:], values[1:ne+1]-target]
    j = np.concatenate((jacobian[ne+1:], jacobian[1:ne+1]))
    h = np.concatenate((hessian[ne+1:], hessian[1:ne+1]))
    return c, j, h


def stationary_constitution(model, element_fractions, initial_sites, temperature_K,
                            pressure_Pa=101325.0, *, require_local_minimum=True,
                            allow_one_sided=False, policy=DEFAULT_CEF_STATIONARY_POLICY):
    if (type(require_local_minimum) is not bool or type(allow_one_sided) is not bool
            or not isinstance(policy, CefStationaryPolicy)):
        raise ValueError("CEF stationarity requires boolean choices and a CefStationaryPolicy")
    temperature, pressure = model._potentials(temperature_K, pressure_Pa)
    initial = model._fractions(initial_sites, check_constraints=False)
    if any(y < policy.site_floor for subl in initial for y in subl.values()):
        raise ValueError("CEF initial sites are below the declared positive floor")
    labels = site_labels(model)
    sites = np.array([initial[i][s] for i, s in labels])
    elements, values, j, h = site_response(model, sites, temperature, pressure)
    if (not isinstance(element_fractions, Mapping) or set(element_fractions) != set(elements)
            or any(isinstance(v, bool) for v in element_fractions.values())):
        raise ValueError("CEF target composition must include every active element")
    target = np.array([element_fractions[e] for e in elements], dtype=float)
    if (not np.all(np.isfinite(target)) or np.any(target <= 0)
            or abs(float(target.sum())-1) > 1e-10):
        raise ValueError(
            "CEF target fractions must be positive and sum to one; filter zero elements",
        )
    _, cj, _ = _constraints(values, j, h, target)
    indices, scales = independent_constraints(cj, policy.rank_tolerance)
    n = len(sites)
    energy_scale = max(1.0, R*temperature, abs(float(values[0])))
    cached = {}

    def assemble(z):
        key = tuple(z)
        if key not in cached:
            try:
                _, value, jac, hes = site_response(model, z[:n], temperature, pressure)
                c, jc, hc = _constraints(value, jac, hes, target)
                selected_j = jc[indices]/scales[:, None]
                selected_h = hc[indices]/scales[:, None, None]
                stationarity = jac[0]/energy_scale + selected_j.T @ z[n:]
                residual = np.r_[stationarity, c[indices]/scales]
                curvature = hes[0]/energy_scale + np.einsum("i,ijk->jk", z[n:], selected_h)
                derivative = np.block([[curvature, selected_j.T],
                                       [selected_j, np.zeros((len(indices), len(indices)))]])
                cached.clear()
                cached[key] = residual, derivative, value, jac, hes, c, jc, hc
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc
        return cached[key]

    variables = [ResidualVariable(
        f"site:{i}:{s}", "dimensionless", lower=policy.site_floor, upper=1,
    ) for i, s in labels]
    variables.extend(ResidualVariable(f"constraint_multiplier:{i}", "dimensionless")
                     for i in indices)
    attempts, escapes, total_iterations = [], [], 0
    initial_root = np.r_[sites, np.zeros(len(indices))]
    for escape in range(policy.maximum_saddle_escapes+1):
        remaining = max(1, policy.root.maximum_iterations-total_iterations)
        root = solve_root(lambda z: assemble(z)[0], initial_root,
                          jacobian=lambda z: assemble(z)[1], variables=variables,
                          residual_scales=np.ones(n+len(indices)),
                          policy=replace(policy.root, maximum_iterations=remaining))
        attempts.append(root.diagnostics)
        total_iterations += root.diagnostics["iterations"]
        if not root.success or not require_local_minimum:
            break
        _, _, _, _, root_h, root_c, root_j, root_ch = assemble(root.value)
        if np.max(np.abs(root_c)) > 10*policy.root.tolerance:
            break
        root_lagrangian = root_h[0] + np.einsum(
            "i,ijk->jk", root.value[n:]*energy_scale, root_ch[indices]/scales[:, None, None],
        )
        minimum, stable, direction = _stability(root_lagrangian, root_j, policy)
        if stable or escape == policy.maximum_saddle_escapes:
            break
        if total_iterations >= policy.root.maximum_iterations:
            break
        probe = _escape_saddle(model, root.value[:n], direction, elements, target,
                               temperature, pressure, policy)
        if probe is None:
            break
        trial, receipt = probe
        receipt["minimum_internal_curvature_J_mol"] = minimum
        escapes.append(receipt)
        initial_root = np.r_[trial, root.value[n:]]
    diagnostics = {"root": root.diagnostics, "root_attempts": attempts,
                   "total_root_iterations": total_iterations, "saddle_escapes": escapes,
                   "constraint_indices": indices.tolist(),
                   "constraint_row_scales": scales.tolist(), "energy_scale_J_mol": energy_scale,
                   "site_floor": policy.site_floor,
                   "branch_scope": "local to the supplied initial sites"}

    def failure(reason):
        return CefConstitutionResult(False, reason, None, None, None, None, None, diagnostics)

    if not root.success:
        return failure(root.failure_reason)
    _, _, _, _, final_h, all_c, all_j, all_h = assemble(root.value)
    diagnostics["all_constraint_residuals"] = all_c.tolist()
    if float(np.max(np.abs(all_c))) > 10*policy.root.tolerance:
        return failure("full_constraints_not_satisfied")
    fractions = unpack_sites(model, root.value[:n])
    # Common root tolerances and the public state's legal-site tolerance must
    # both pass; no normalization repairs a solver result before acceptance.
    try:
        fractions = model._fractions(fractions)
    except ValueError as exc:
        return failure(str(exc))
    physical_multipliers = root.value[n:]*energy_scale
    selected_h = all_h[indices]/scales[:, None, None]
    lagrangian = final_h[0] + np.einsum("i,ijk->jk", physical_multipliers, selected_h)
    minimum, stable = _curvature(lagrangian, all_j, policy)
    diagnostics.update(minimum_internal_curvature_J_mol=minimum, internal_minimum=stable)
    if not stable and require_local_minimum:
        return failure("internal_saddle")
    branches = branch_diagnostics(model, fractions, temperature, pressure)
    diagnostics["branch_diagnostics"] = list(branches)
    if any("parameter sign boundary" in d for d in branches):
        return failure("magnetic_parameter_sign_boundary")
    if branches and not allow_one_sided:
        return failure("one_sided_branch_requires_acceptance")
    properties = model.evaluate(fractions, temperature, pressure, pressure_derivative=False)
    mu = chemical_potentials(model, fractions, temperature, pressure,
                             require_local_minimum=require_local_minimum,
                             allow_one_sided=allow_one_sided, policy=policy)
    cp, dy_dt = None, None
    try:
        response = derivatives(model, fractions, temperature, pressure, include_pressure=False,
                               allow_one_sided=allow_one_sided)
        _, c_j, c_h = _constraints(response.values, response.jacobian, response.hessian, target)
        constraint_sites = c_j[indices, 1:]/scales[:, None]
        cross = response.hessian[0, 1:, 0] + np.einsum(
            "i,ij->j", physical_multipliers, c_h[indices, 1:, 0]/scales[:, None],
        )
        matrix = np.block([[lagrangian, constraint_sites.T],
                           [constraint_sites, np.zeros((len(indices), len(indices)))]])
        rhs = -np.r_[cross, c_j[indices, 0]/scales]
        dy = linear_solve(matrix, rhs)[:n]
        cp = float(response.jacobian[0, 1:] @ dy - temperature * (
            response.hessian[0, 0, 0] + response.hessian[0, 0, 1:] @ dy
        ))
        if not isfinite(cp):
            raise ResidualNumericalError("nonfinite relaxed heat capacity")
        dy_dt = tuple((i, s, float(v)) for (i, s), v in zip(labels, dy, strict=True))
        diagnostics["fixed_constitution_heat_capacity_J_mol_K"] = float(
            -temperature*response.hessian[0, 0, 0],
        )
    except (ValueError, ResidualNumericalError) as exc:
        diagnostics["temperature_response_failure"] = str(exc)
    return CefConstitutionResult(True, "", tuple(MappingProxyType(dict(y)) for y in fractions),
                                  properties, mu, cp, dy_dt, diagnostics)
