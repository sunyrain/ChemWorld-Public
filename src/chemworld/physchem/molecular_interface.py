"""Planar square-gradient interfaces on the common molecular free energy.

Bulk endpoints must coexist. Interior profiles may cross spinodals: their free
energy is evaluated directly, without homogeneous stability filtering. Constant
positive-definite influence matrices are explicit; singular mixing rules require
a different constrained theory and are not silently regularized.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite, log, sqrt
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.linalg import eig_banded
from scipy.optimize import minimize
from scipy.sparse import diags, lil_matrix

from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.finite_volume import DiffusionMesh
from chemworld.physchem.molecular_eos import (
    MolecularEOSState,
    ResidualHelmholtzModel,
    molecular_residual_chemical_potentials,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SquareGradientInfluence:
    component_ids: tuple[str, ...]
    coefficients_J_m5_mol2: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", tuple(self.component_ids))
        object.__setattr__(self, "coefficients_J_m5_mol2",
                           tuple(tuple(row) for row in self.coefficients_J_m5_mol2))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        c = np.asarray(self.coefficients_J_m5_mol2)
        n = len(self.component_ids)
        if (not n or len(set(self.component_ids)) != n or not self.source
                or c.shape != (n, n) or not np.all(np.isfinite(c))
                or not np.allclose(c, c.T, rtol=1e-13, atol=0)
                or np.min(np.linalg.eigvalsh(c)) <= 0):
            raise ValueError("influence matrix must be finite, symmetric and positive definite")
        if (len(self.temperature_range_K) != 2
                or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
                or not all(isfinite(v) for v in self.temperature_range_K)):
            raise ValueError("influence parameters require a positive temperature domain")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "constant-square-gradient-influence-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SquareGradientInfluence:
        values = dict(data)
        if values.pop("model", None) != "constant-square-gradient-influence-v1":
            raise ValueError("unknown influence declaration")
        return cls(**values)


@dataclass(frozen=True)
class MolecularInterfaceBulk:
    helmholtz_density_J_m3: float
    chemical_potentials_J_mol: tuple[float | None, ...]
    pressure_Pa: float


@dataclass(frozen=True)
class MolecularInterfaceResult:
    success: bool
    surface_tension_N_m: float | None = None
    positions_m: tuple[float, ...] = ()
    component_densities_mol_m3: tuple[tuple[float, ...], ...] = ()
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "scope": "planar_constant_positive_definite_square_gradient",
                "profile_orientation": "rows=positions; columns=model_component_order"}


def _density(model: ResidualHelmholtzModel, temperature: float,
             concentrations: Sequence[float]) -> FloatArray:
    c = np.asarray(concentrations, dtype=float)
    if (c.shape != (len(model.component_ids),) or not np.all(np.isfinite(c)) or np.any(c < 0)
            or c.sum() <= 0 or not isfinite(temperature) or temperature <= 0):
        raise ValueError("molecular interface needs positive T and nonnegative nonempty densities")
    return c


def _free_energy(model: ResidualHelmholtzModel, temperature: float, c: FloatArray) -> float:
    rho = float(c.sum())
    ideal = fsum(float(v)*(log(float(v))-1) for v in c if v > 0)
    residual = rho*float(model.alphar(temperature, rho, tuple(float(v/rho) for v in c)))
    return model.gas_constant_J_mol_K*temperature*fsum((ideal, residual))


def molecular_interface_bulk(model: ResidualHelmholtzModel, temperature_K: float,
                             component_densities_mol_m3: Sequence[float]) -> MolecularInterfaceBulk:
    """Free energy in the c_ref=1 mol/m3 ideal gauge; absent ideal mu is undefined.

    Species-linear reference terms cancel in the coexistence grand potential.
    No absolute caloric reference or local mechanical stability is required.
    """
    c = _density(model, temperature_K, component_densities_mol_m3)
    rho, rt = float(c.sum()), model.gas_constant_J_mol_K*temperature_K
    residual_mu = molecular_residual_chemical_potentials(model, temperature_K, rho,
        tuple(float(v/rho) for v in c))
    mu = tuple(None if v == 0 else rt*log(float(v))+residual_mu[i] for i, v in enumerate(c))
    f = _free_energy(model, temperature_K, c)
    p = fsum(float(v)*float(m) for v, m in zip(c, mu, strict=True) if m is not None)-f
    return MolecularInterfaceBulk(f, mu, p)


def _coexistence(model: ResidualHelmholtzModel, left: MolecularEOSState,
                 right: MolecularEOSState, influence: SquareGradientInfluence) -> dict[str, Any]:
    if influence.component_ids != model.component_ids:
        raise ValueError("influence and EOS component identities must agree")
    t = left.temperature_K
    if (abs(t-right.temperature_K) > 1e-9
            or not influence.temperature_range_K[0] <= t <= influence.temperature_range_K[1]):
        raise ValueError("interface endpoints require one temperature inside the influence domain")
    c1 = np.array(left.mole_fractions)*left.molar_density_mol_m3
    c2 = np.array(right.mole_fractions)*right.molar_density_mol_m3
    if np.any((c1 > 0) != (c2 > 0)):
        raise ValueError("interface endpoints require the same exact component support")
    active = np.flatnonzero(c1 > 0)
    if not 1 <= len(active) <= 2:
        raise ValueError("this interface slice supports one or two active components")
    if np.max(np.abs(c1[active]-c2[active])/np.maximum(c1[active], c2[active])) < 1e-7:
        raise ValueError("interface endpoints must be distinct phases")
    a, b = (molecular_interface_bulk(model, t, tuple(c)) for c in (c1, c2))
    rt = model.gas_constant_J_mol_K*t
    mu = np.array([float(a.chemical_potentials_J_mol[i]) for i in active])
    other_mu = np.array([float(b.chemical_potentials_J_mol[i]) for i in active])
    mu_error = float(np.max(np.abs(mu-other_mu))/rt)
    pressure_error = abs(a.pressure_Pa-b.pressure_Pa)/max(abs(a.pressure_Pa), 1.)
    if (not left.mechanically_stable or not right.mechanically_stable
            or a.pressure_Pa <= 0 or b.pressure_Pa <= 0 or mu_error > 1e-6
            or pressure_error > 1e-7 or any(abs(state.pressure_Pa-bulk.pressure_Pa)
                > max(1e-4, 1e-7*abs(bulk.pressure_Pa))
                for state, bulk in ((left, a), (right, b)))):
        raise ValueError("interface endpoints do not coexist on the declared free energy")
    return {"temperature": t, "left": c1, "right": c2, "active": active, "mu": mu,
            "reference_f": a.helmholtz_density_J_m3, "RT": rt,
            "C": np.asarray(influence.coefficients_J_m5_mol2)[np.ix_(active, active)],
            "diagnostics": {"endpoint_mu_error_RT": mu_error,
                            "endpoint_pressure_relative_error": pressure_error,
                            "coexistence_pressure_Pa": a.pressure_Pa}}


def _omega(model: ResidualHelmholtzModel, pair: dict[str, Any], c: FloatArray) -> float:
    return fsum((_free_energy(model, pair["temperature"], c), -pair["reference_f"],
                 -float(pair["mu"]@(c-pair["left"])[pair["active"]])))


def _quadrature(function: Any, lo: float, hi: float, tolerance: float,
                 absolute: float) -> tuple[float, float, int]:
    value, error, info, *warning = quad(function, lo, hi, epsrel=tolerance,
                                       epsabs=absolute, limit=150, full_output=True)
    if warning or not isfinite(value) or error > max(absolute, tolerance*abs(value)):
        raise ValueError("interface quadrature failed: "+str(warning or error))
    return float(value), float(error), int(info["neval"])


def pure_square_gradient_interface(
    model: ResidualHelmholtzModel, left_state: MolecularEOSState, right_state: MolecularEOSState,
    influence: SquareGradientInfluence, *, interior_density_grid: Sequence[float],
    quadrature_tolerance: float = 1e-9,
) -> MolecularInterfaceResult:
    """Exact scalar first integral, full tension and explicitly truncated profile."""
    pair = _coexistence(model, left_state, right_state, influence)
    if len(pair["active"]) != 1:
        raise ValueError("pure interface integral requires exactly one active component")
    if not isfinite(quadrature_tolerance) or quadrature_tolerance <= 0:
        raise ValueError("interface quadrature tolerance must be positive")
    index = pair["active"][0]
    lo, hi = sorted((pair["left"][index], pair["right"][index]))
    requested = tuple(float(v) for v in interior_density_grid)
    if not requested or any(not isfinite(v) or not lo < v < hi for v in requested):
        raise ValueError("pure profile densities must be strictly inside coexistence endpoints")
    coefficient = float(pair["C"][0, 0])

    def omega(rho: float) -> float:
        c = np.zeros(len(model.component_ids))
        c[index] = rho
        value = _omega(model, pair, c)
        if value <= 0:
            raise ValueError("nonpositive interior grand potential; no single scalar interface")
        return value

    try:
        tension, error, evaluations = _quadrature(lambda rho: sqrt(2*coefficient*omega(rho)),
            lo, hi, quadrature_tolerance, 1e-14)
        positions, errors = [], []
        for rho in requested:
            position, bound, calls = _quadrature(lambda r: sqrt(coefficient/(2*omega(r))),
                (lo+hi)/2, rho, quadrature_tolerance, 1e-16)
            positions.append(position)
            errors.append(bound)
            evaluations += calls
        density_rows = []
        for rho in requested:
            row = np.zeros(len(model.component_ids))
            row[index] = rho
            density_rows.append(tuple(float(v) for v in row))
        return MolecularInterfaceResult(True, tension, tuple(positions), tuple(density_rows),
            diagnostics={**pair["diagnostics"], "method": "scalar_first_integral_quadrature",
                         "tension_error_estimate_N_m": error,
                         "position_error_estimates_m": errors, "evaluations": evaluations,
                         "profile_tails": "infinite_asymptotes_excluded_from_requested_grid"})
    except (ValueError, ArithmeticError) as exc:
        return MolecularInterfaceResult(False, failure_reason=str(exc),
                                          diagnostics=pair["diagnostics"])


def _hessian(model: ResidualHelmholtzModel, temperature: float, c: FloatArray,
              active: Any) -> FloatArray:
    values = {str(i): float(c[i]) for i in active}

    def extensive(v: dict[str, Any]) -> Any:
        concentrations = tuple(v.get(str(i), 0.) for i in range(len(c)))
        rho = sum(concentrations)
        return rho*model.alphar(temperature, rho, tuple(x/rho for x in concentrations))

    rt = model.gas_constant_J_mol_K*temperature
    return np.array([[rt*(partial_derivative(extensive, values, (str(i), str(j)))
                         +(1/c[i] if i == j else 0)) for j in active] for i in active])


def molecular_planar_interface(
    model: ResidualHelmholtzModel, left_state: MolecularEOSState, right_state: MolecularEOSState,
    influence: SquareGradientInfluence, *, mesh: DiffusionMesh,
    component_density_bounds: Sequence[tuple[float, float]],
    initial_profile: Sequence[Sequence[float]] | None = None, max_iterations: int = 500,
    residual_tolerance: float = 1e-7, first_integral_tolerance: float = .05,
) -> MolecularInterfaceResult:
    """Free component densities minimize the finite-domain planar functional.

    A line is only the default initial guess. Each interior component is an
    independent variable; adsorption/overshoot are allowed within explicit
    bounds. The result is a finite-domain local stationary minimum, not a global
    profile or bulk-stability proof. Mesh/length convergence is the caller's task.
    """
    pair = _coexistence(model, left_state, right_state, influence)
    if mesh.geometry != "planar" or mesh.cells < 4:
        raise ValueError("interface requires a planar mesh with at least four intervals")
    if (type(max_iterations) is not int or max_iterations < 1
            or any(not isfinite(v) or v <= 0 for v in
                   (residual_tolerance, first_integral_tolerance))):
        raise ValueError("interface solver requires positive iteration budget and tolerance")
    active, cmat = pair["active"], pair["C"]
    n, count = len(active), mesh.cells-1
    nodes = np.asarray(mesh.faces_m)
    dz = np.diff(nodes)
    weights = (dz[:-1]+dz[1:])/2
    scales = np.maximum(pair["left"][active], pair["right"][active])
    bounds = np.asarray(component_density_bounds, dtype=float)
    if (bounds.shape != (len(model.component_ids), 2) or not np.all(np.isfinite(bounds))
            or np.any(bounds[active, 0] <= 0) or np.any(bounds[active, 1] <= bounds[active, 0])
            or np.any(pair["left"] < bounds[:, 0]) or np.any(pair["left"] > bounds[:, 1])
            or np.any(pair["right"] < bounds[:, 0]) or np.any(pair["right"] > bounds[:, 1])):
        raise ValueError("declared density bounds must contain both endpoints and positive support")
    fraction = nodes/nodes[-1]
    # Break exact grid-translation symmetry in the initial guess, which can
    # otherwise pin a coarse-grid saddle at a node. No path is constrained.
    fraction += .01*np.sin(np.pi*fraction)
    initial = (pair["left"]+(pair["right"]-pair["left"])*fraction[:, None]
               if initial_profile is None else np.asarray(initial_profile, dtype=float))
    if (initial.shape != (len(nodes), len(model.component_ids))
            or not np.all(np.isfinite(initial)) or np.any(initial < bounds[:, 0])
            or np.any(initial > bounds[:, 1])
            or not np.allclose(initial[0], pair["left"], rtol=1e-12, atol=0)
            or not np.allclose(initial[-1], pair["right"], rtol=1e-12, atol=0)):
        raise ValueError("interface initial profile must match endpoints and declared bounds")
    inactive = [i for i in range(len(model.component_ids)) if i not in active]
    if inactive and np.any(initial[:, inactive] != 0):
        raise ValueError("interface cannot invent absent components")
    energy_scale = max(pair["RT"]*float(scales.sum())*nodes[-1]*.01,
                       float(scales@cmat@scales)/nodes[-1])
    lower = np.tile(bounds[active, 0]/scales, count)
    upper = np.tile(bounds[active, 1]/scales, count)

    def unpack(q: FloatArray) -> FloatArray:
        profile = np.zeros_like(initial)
        profile[0], profile[-1] = pair["left"], pair["right"]
        profile[1:-1, active] = q.reshape(count, n)*scales
        return profile

    def evaluate(q: FloatArray) -> tuple[float, FloatArray, FloatArray]:
        profile = unpack(q)
        omegas, mus = [], []
        for c in profile[1:-1]:
            try:
                bulk = molecular_interface_bulk(model, pair["temperature"], tuple(c))
                omegas.append(_omega(model, pair, c))
                mus.append([float(bulk.chemical_potentials_J_mol[i]) for i in active])
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc
        differences = np.diff(profile[:, active], axis=0)
        gradient_energy = .5*float(np.sum((differences@cmat)*differences/dz[:, None]))
        force = (differences[:-1]/dz[:-1, None]-differences[1:]/dz[1:, None])@cmat
        gradient = weights[:, None]*(np.asarray(mus)-pair["mu"])+force
        energy = float(weights@omegas)+gradient_energy
        return energy, gradient, np.array(omegas)

    def objective(q: FloatArray) -> tuple[float, FloatArray]:
        energy, gradient, _ = evaluate(q)
        return energy/energy_scale, (gradient*scales/energy_scale).ravel()

    def residual(q: FloatArray) -> FloatArray:
        return (evaluate(q)[1]/weights[:, None]/pair["RT"]).ravel()

    def jacobian(q: FloatArray) -> Any:
        profile = unpack(q)
        matrix = lil_matrix((count*n, count*n))
        for k, c in enumerate(profile[1:-1]):
            try:
                diagonal = _hessian(model, pair["temperature"], c, active)
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc
            diagonal += cmat*(1/dz[k]+1/dz[k+1])/weights[k]
            row = slice(k*n, (k+1)*n)
            matrix[row, row] = diagonal*scales/pair["RT"]
            if k > 0:
                matrix[row, slice((k-1)*n, k*n)] = -cmat*scales/(dz[k]*weights[k]*pair["RT"])
            if k < count-1:
                matrix[row, slice((k+1)*n, (k+2)*n)] = -cmat*scales/(
                    dz[k+1]*weights[k]*pair["RT"])
        return matrix.tocsr()

    diagnostics = {**pair["diagnostics"], "method": "free_density_variational_and_shared_root",
                   "nodes": len(nodes), "length_m": float(nodes[-1]),
                   "profile_scope": "local_finite_domain; grid_and_length_convergence_external",
                   "path_constraint": "none; all active component densities independently free"}
    try:
        optimum = minimize(objective, (initial[1:-1, active]/scales).ravel(), jac=True,
            method="L-BFGS-B", bounds=list(zip(lower, upper, strict=True)),
            options={"maxiter": max_iterations, "gtol": 1e-10, "ftol": 1e-15, "maxcor": 30})
        diagnostics["minimization"] = {"success": bool(optimum.success),
            "iterations": int(optimum.nit), "evaluations": int(optimum.nfev),
            "message": str(optimum.message)}
        root = solve_root(residual, optimum.x, jacobian=jacobian,
            variables=tuple(ResidualVariable(f"density_{i}", "scaled_mol_m3",
                lower=float(lo), upper=float(hi)) for i, (lo, hi) in enumerate(
                    zip(lower, upper, strict=True))), residual_scales=np.ones(count*n),
            policy=RootPolicy(tolerance=residual_tolerance, maximum_iterations=max_iterations))
        diagnostics["root"] = {"success": root.success, "failure_reason": root.failure_reason,
                               **root.diagnostics}
        if not root.success:
            return MolecularInterfaceResult(False, failure_reason="interface_root_failed",
                                              diagnostics=diagnostics)
        q = root.value
        profile = unpack(q)
        energy, _, omegas = evaluate(q)
        hessian = diags(np.repeat(weights, n)*np.tile(scales, count)*pair["RT"]/energy_scale
                        )@jacobian(q)
        hessian = (hessian+hessian.T)*.5
        bandwidth = 2*n-1
        banded = np.zeros((bandwidth+1, count*n))
        for offset in range(bandwidth+1):
            banded[bandwidth-offset, offset:] = hessian.diagonal(offset)
        curvature = float(eig_banded(banded, lower=False, eigvals_only=True,
                                     select="i", select_range=(0, 0))[0])
        negative_tolerance = 1e-8*pair["RT"]*float(scales.sum())
        diagnostics.update(scaled_minimum_curvature=curvature,
                           minimum_grand_potential_J_m3=float(omegas.min()),
                           negative_grand_potential_tolerance_J_m3=negative_tolerance)
        if curvature < -1e-7 or energy <= 0 or float(omegas.min()) < -negative_tolerance:
            return MolecularInterfaceResult(False,
                failure_reason="nonminimum_or_unstable_interface", diagnostics=diagnostics)
        slopes = np.diff(profile[:, active], axis=0)/dz[:, None]
        gradient_density = .5*np.sum((slopes@cmat)*slopes, axis=1)
        middle = .5*(profile[1:]+profile[:-1])
        middle_omega = np.array([_omega(model, pair, c) for c in middle])
        integral = gradient_density-middle_omega
        barrier = max(float(middle_omega.max()), 1e-30)
        diagnostics.update(first_integral_variation_relative=float(np.ptp(integral))/barrier,
            first_integral_boundary_J_m3=(float(integral[0]), float(integral[-1])),
            gradient_tension_N_m=float(2*gradient_density@dz),
            maximum_grand_potential_J_m3=barrier,
            stationarity_residual_RT=float(np.max(np.abs(residual(q)))))
        resolved = diagnostics["first_integral_variation_relative"] <= first_integral_tolerance
        diagnostics["first_integral_tolerance"] = first_integral_tolerance
        return MolecularInterfaceResult(resolved, energy, tuple(float(v) for v in nodes),
            tuple(tuple(float(v) for v in row) for row in profile),
            "" if resolved else "profile_underresolved_first_integral", diagnostics=diagnostics)
    except (ValueError, ArithmeticError, ResidualDomainError) as exc:
        return MolecularInterfaceResult(False, failure_reason=str(exc), diagnostics=diagnostics)
