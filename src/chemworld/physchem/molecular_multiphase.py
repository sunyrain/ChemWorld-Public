"""Declared N-phase molecular equilibrium and simultaneous extensive constraints.

The original Helmholtz/caloric model and phase records remain authoritative.
Finite basin/start searches are reported, never promoted to global certificates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations, product
from math import fsum, isfinite, log
from typing import Any, cast

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.special import softmax

from chemworld.physchem.caloric_math import SecondOrderPair
from chemworld.physchem.molecular_eos import molecular_residual_chemical_potentials
from chemworld.physchem.molecular_flash import (
    MolecularDensityBasin,
    MolecularFlashPhase,
    MolecularTPDResult,
)
from chemworld.physchem.molecular_mixture import MolecularCaloricModel, MolecularCaloricState
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)


@dataclass(frozen=True)
class MolecularMultiphaseSpec:
    model: MolecularCaloricModel
    basins: tuple[MolecularDensityBasin, ...]
    temperature_bounds_K: tuple[float, float]
    pressure_bounds_Pa: tuple[float, float]
    max_phases: int = 3
    phase_kind_by_basin: Mapping[str, str] = field(default_factory=dict)
    root_policy: RootPolicy = field(default_factory=lambda: RootPolicy(2e-9, 150))
    stability_max_iterations: int = 200
    tpd_tolerance: float = 1e-8
    phase_fraction_tolerance: float = 1e-9
    maximum_phase_updates: int = 20
    molar_masses_kg_mol: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        for key in ("basins", "temperature_bounds_K", "pressure_bounds_Pa"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        object.__setattr__(self, "phase_kind_by_basin", dict(self.phase_kind_by_basin))
        names = {b.name for b in self.basins}
        if not names or len(names) != len(self.basins):
            raise ValueError("multiphase specification needs unique declared density basins")
        for bounds in (self.temperature_bounds_K, self.pressure_bounds_Pa):
            if (
                len(bounds) != 2
                or any(isinstance(v, bool) or not isfinite(v) for v in bounds)
                or not 0 < bounds[0] < bounds[1]
            ):
                raise ValueError("multiphase T/P bounds must be ordered, finite and positive")
        if set(self.phase_kind_by_basin) - names or any(
            v not in {"gas", "liquid", "fluid"} for v in self.phase_kind_by_basin.values()
        ):
            raise ValueError("phase kinds must explicitly label declared gas/liquid/fluid basins")
        if any(
            type(v) is not int or v < 1
            for v in (self.max_phases, self.stability_max_iterations, self.maximum_phase_updates)
        ):
            raise ValueError("multiphase phase counts and search budgets must be positive integers")
        if (
            not isfinite(self.tpd_tolerance)
            or self.tpd_tolerance <= 0
            or not 0 < self.phase_fraction_tolerance < 0.01
        ):
            raise ValueError("multiphase stability and phase-fraction tolerances are invalid")
        masses = tuple(ref.molar_mass_kg_mol for ref in self.model.ideal_references)
        if self.molar_masses_kg_mol is not None:
            explicit = tuple(self.molar_masses_kg_mol)
            if (
                len(explicit) != len(masses)
                or any(not isfinite(v) or v <= 0 for v in explicit)
                or any(abs(a / b - 1) > 1e-12 for a, b in zip(explicit, masses, strict=True))
            ):
                raise ValueError("explicit masses must agree with the declared ideal references")
        object.__setattr__(self, "molar_masses_kg_mol", masses)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "molecular_multiphase_v1",
            "model": self.model.to_dict(),
            "basins": [asdict(b) for b in self.basins],
            **{
                k: getattr(self, k)
                for k in (
                    "temperature_bounds_K",
                    "pressure_bounds_Pa",
                    "max_phases",
                    "phase_kind_by_basin",
                    "stability_max_iterations",
                    "tpd_tolerance",
                    "phase_fraction_tolerance",
                    "maximum_phase_updates",
                    "molar_masses_kg_mol",
                )
            },
            "root_policy": asdict(self.root_policy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MolecularMultiphaseSpec:
        if data.get("model_type") != "molecular_multiphase_v1":
            raise ValueError("unknown molecular multiphase specification")
        values = {k: v for k, v in data.items() if k != "model_type"}
        values.update(
            model=MolecularCaloricModel.from_dict(data["model"]),
            basins=tuple(MolecularDensityBasin(**b) for b in data["basins"]),
            root_policy=RootPolicy(**data["root_policy"]),
        )
        return cls(**values)


@dataclass(frozen=True)
class MolecularMultiphaseSeed:
    temperature_K: float
    pressure_Pa: float
    phases: tuple[MolecularFlashPhase, ...] = ()


@dataclass(frozen=True)
class MolecularMultiphaseResult:
    success: bool
    phases: tuple[MolecularFlashPhase, ...] = ()
    temperature_K: float | None = None
    pressure_Pa: float | None = None
    enthalpy_J: float | None = None
    entropy_J_K: float | None = None
    internal_energy_J: float | None = None
    volume_m3: float | None = None
    mass_kg: float | None = None
    vapor_fraction: float | None = None
    material_balance_error_mol: float | None = None
    chemical_potential_error_RT: float | None = None
    pressure_relative_error: float | None = None
    constraint_errors: Mapping[str, float] = field(default_factory=dict)
    stability: tuple[MolecularTPDResult, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""
    stability_checked: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "phases": [p.to_dict() for p in self.phases],
            "stability": [s.to_dict() for s in self.stability],
            "scope": (
                "finite_declared_basins_and_starts; no_global_certificate"
                if self.stability_checked
                else "unqualified_numerical_trial"
            ),
        }


def _mu(state: Any, active: Any) -> Any:
    if state.log_fugacity_coefficients is None or state.pressure_Pa <= 0:
        raise ValueError("phase chemical potential requires positive pressure")
    return (
        np.log(np.asarray(state.mole_fractions)[active])
        + np.asarray(state.log_fugacity_coefficients)[active]
        + log(state.pressure_Pa)
    )


def _joint_stability(
    spec: MolecularMultiphaseSpec,
    temperature: float,
    pressure: float,
    reference: MolecularFlashPhase,
    composition_starts: Sequence[Sequence[float]] | None,
) -> MolecularTPDResult:
    """Minimize A+P V against a tangent plane in density and composition.

    A phase-specific density basin need not contain a pressure root at every
    composition. Constitutively impossible initial points and nonnegative
    constrained boundary minima are recorded as excluded points, not fabricated
    pressure roots. Failed feasible searches or negative boundary minima remain
    incomplete. This is finite multistart local evidence only.
    """
    model = spec.model.residual_model
    z = np.asarray(reference.state.mole_fractions)
    active0 = np.flatnonzero(z > 0)
    anchor = int(np.argmax(z))
    active = np.array([i for i in active0 if i != anchor] + [anchor])
    rt = model.gas_constant_J_mol_K * temperature
    d = _mu(reference.state, active) - log(rt)
    if composition_starts is None:
        seeds = [z.copy(), np.where(z > 0, 1 / len(active), 0)]
        if len(active) > 1:
            for i in active:
                x = np.zeros(len(z))
                x[active] = 0.01 / (len(active) - 1)
                x[i] = 0.99
                seeds.append(x)
    else:
        seeds = [np.asarray(x, dtype=float) for x in composition_starts]
    if not seeds or any(
        x.shape != z.shape
        or not np.all(np.isfinite(x))
        or np.any(x[active] <= 0)
        or np.any(x[z == 0] != 0)
        or abs(float(x.sum()) - 1) > 1e-10
        for x in seeds
    ):
        raise ValueError("TPD starts must preserve exact positive feed support")
    records: list[dict[str, Any]] = []
    complete, minimum = True, 0.0
    for basin in spec.basins:
        lower = np.r_[log(basin.density_bounds_mol_m3[0]), np.full(len(active) - 1, -70.0)]
        upper = np.r_[log(basin.density_bounds_mol_m3[1]), np.full(len(active) - 1, 70.0)]
        feasible_starts = 0
        for index, start_composition in enumerate(seeds):
            record: dict[str, Any] = {
                "basin": basin.name,
                "start": index,
                "initial_composition": tuple(map(float, start_composition)),
                "invalid_trial_count": 0,
            }
            initial = np.r_[
                log(basin.initial_density_mol_m3),
                np.log(start_composition[active][:-1] / start_composition[active][-1]),
            ]

            def evaluate(q: Any) -> tuple[float, Any, tuple[float, ...], float]:
                rho = float(np.exp(q[0]))
                w = softmax(np.r_[q[1:], 0.0])
                full = np.zeros(len(z))
                full[active] = w
                fractions = tuple(map(float, full))
                ar = SecondOrderPair(0) + model.alphar(
                    temperature, SecondOrderPair(rho, dx=1), fractions
                )
                mus = (
                    np.asarray(
                        molecular_residual_chemical_potentials(model, temperature, rho, fractions)
                    )[active]
                    / rt
                    + log(rho)
                    + np.log(w)
                )
                value = float(
                    ar.value + log(rho) - 1 + w @ np.log(w) + pressure / (rho * rt) - w @ d
                )
                delta = mus - d
                gradient = np.r_[
                    1 + rho * ar.dx - pressure / (rho * rt), (w * (delta - w @ delta))[:-1]
                ]
                if not isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise ValueError("nonfinite common Gibbs tangent response")
                return value, gradient, fractions, rho

            def objective(q: Any, record: dict[str, Any] = record) -> tuple[float, Any]:
                try:
                    value, derivative, _, _ = evaluate(q)
                    return value, derivative
                except (ValueError, ArithmeticError):
                    # Optimizer trial extension only. Every accepted point is
                    # re-evaluated on the unchanged constitutive model below.
                    record["invalid_trial_count"] += 1
                    return 1e100, np.zeros(len(q))

            try:
                evaluate(initial)
            except (ValueError, ArithmeticError) as exc:
                record.update(
                    success=False,
                    excluded="initial_outside_constitutive_domain",
                    failure_reason=str(exc),
                )
                records.append(record)
                continue
            feasible_starts += 1
            try:
                optimum = minimize(
                    objective,
                    initial,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=list(zip(lower, upper, strict=True)),
                    options={
                        "gtol": 1e-10,
                        "ftol": 1e-15,
                        "maxiter": spec.stability_max_iterations,
                        "maxls": 40,
                    },
                )
                point = optimum.x
                value, gradient, fractions, rho = evaluate(point)
                on_bound = (point - lower < 1e-7) | (upper - point < 1e-7)
                projected = gradient.copy()
                projected[(point - lower < 1e-7) & (gradient > 0)] = 0
                projected[(upper - point < 1e-7) & (gradient < 0)] = 0
                if (
                    np.max(np.abs(projected)) > 1e-5
                    or (np.any(on_bound) and np.max(np.abs(projected)) > spec.tpd_tolerance)
                ) and optimum.nit < spec.stability_max_iterations:
                    # A domain-rejected line-search trial can stop L-BFGS
                    # before stationarity. Complete that same
                    # start with the remaining optimization budget, retaining
                    # both attempts and all rejected constitutive probes.
                    first_iterations = int(optimum.nit)
                    record["bounded_optimization_first_attempt"] = {
                        "iterations": first_iterations,
                        "message": str(optimum.message),
                        "projected_gradient": float(np.max(np.abs(projected))),
                    }
                    optimum = minimize(
                        objective,
                        point,
                        jac=True,
                        method="SLSQP",
                        bounds=list(zip(lower, upper, strict=True)),
                        options={
                            "ftol": 1e-13,
                            "maxiter": spec.stability_max_iterations - first_iterations,
                        },
                    )
                    record["bounded_optimization_total_iterations"] = first_iterations + int(
                        optimum.nit
                    )
                    point = optimum.x
                    value, gradient, fractions, rho = evaluate(point)
                    on_bound = (point - lower < 1e-7) | (upper - point < 1e-7)
                    projected = gradient.copy()
                    projected[(point - lower < 1e-7) & (gradient > 0)] = 0
                    projected[(upper - point < 1e-7) & (gradient < 0)] = 0
                # Polish the pressure equation and compositional stationarity;
                # scaling the pressure row avoids accepting a liquid-pressure
                # error merely because P/(rho RT) is small.
                if not np.any(on_bound):

                    def stationary(q: Any) -> Any:
                        try:
                            _, grad, _, density = evaluate(q)
                            grad[0] *= density * rt / pressure
                            return grad
                        except (ValueError, ArithmeticError) as exc:
                            raise ResidualDomainError(str(exc)) from exc

                    polish = solve_root(
                        stationary,
                        point,
                        jacobian=lambda q, lower=lower, upper=upper: finite_jacobian(
                            stationary, q, np.ones(len(q)), lower=lower, upper=upper
                        ),
                        variables=tuple(
                            ResidualVariable(
                                f"tpd_{j}", "dimensionless", lower=float(lo), upper=float(hi)
                            )
                            for j, (lo, hi) in enumerate(zip(lower, upper, strict=True))
                        ),
                        residual_scales=np.ones(len(point)),
                        policy=RootPolicy(
                            min(spec.tpd_tolerance, 2e-10), spec.stability_max_iterations
                        ),
                    )
                    record["stationarity_refinement"] = {
                        "success": polish.success,
                        **polish.diagnostics,
                    }
                    if polish.success:
                        point = polish.value
                        value, gradient, fractions, rho = evaluate(point)
                        projected = gradient
                elif np.max(np.abs(projected)) > spec.tpd_tolerance:
                    free_indices = np.flatnonzero(~on_bound)
                    fixed_point = point.copy()
                    if len(free_indices):

                        def boundary_stationary(
                            q: Any, fixed_point: Any = fixed_point, free_indices: Any = free_indices
                        ) -> Any:
                            trial = fixed_point.copy()
                            trial[free_indices] = q
                            try:
                                return evaluate(trial)[1][free_indices]
                            except (ValueError, ArithmeticError) as exc:
                                raise ResidualDomainError(str(exc)) from exc

                        polish = solve_root(
                            boundary_stationary,
                            point[free_indices],
                            jacobian=lambda q, lo=lower, hi=upper, free_indices=free_indices: (
                                finite_jacobian(
                                    boundary_stationary,
                                    q,
                                    np.ones(len(q)),
                                    lower=lo[free_indices],
                                    upper=hi[free_indices],
                                )
                            ),
                            variables=tuple(
                                ResidualVariable(
                                    f"boundary_tpd_{j}",
                                    "dimensionless",
                                    lower=float(lower[j]),
                                    upper=float(upper[j]),
                                )
                                for j in free_indices
                            ),
                            residual_scales=np.ones(len(free_indices)),
                            policy=RootPolicy(
                                min(spec.tpd_tolerance, 2e-10), spec.stability_max_iterations
                            ),
                        )
                        record["boundary_stationarity_refinement"] = {
                            "success": polish.success,
                            **polish.diagnostics,
                        }
                        if polish.success:
                            point[free_indices] = polish.value
                            value, gradient, fractions, rho = evaluate(point)
                            projected = gradient.copy()
                            projected[(point - lower < 1e-7) & (gradient > 0)] = 0
                            projected[(upper - point < 1e-7) & (gradient < 0)] = 0
                record.update(
                    iterations=int(optimum.nit),
                    optimizer_success=bool(optimum.success),
                    message=str(optimum.message),
                    minimum_tpd_RT=value,
                    composition=fractions,
                    density_mol_m3=rho,
                    stationarity_error=float(np.max(np.abs(projected))),
                )
                if np.any(on_bound):
                    record.update(
                        success=bool(np.max(np.abs(projected)) <= spec.tpd_tolerance),
                        excluded="constrained_density_or_composition_boundary",
                    )
                    if value < -spec.tpd_tolerance or not record["success"]:
                        complete = False
                    # An off-pressure constrained minimum is not an accepted
                    # physical phase or a seed for inventory allocation.
                    record["candidate_phase"] = False
                else:
                    state = spec.model.state(temperature, rho, fractions)
                    pressure_error = abs(state.pressure_Pa / pressure - 1)
                    record.update(
                        success=bool(
                            np.max(np.abs(projected)) <= spec.tpd_tolerance
                            and pressure_error <= max(spec.tpd_tolerance, 2e-9)
                        ),
                        pressure_relative_error=pressure_error,
                        candidate_phase=True,
                    )
                    if not record["success"]:
                        complete = False
                    minimum = min(minimum, value)
            except (ValueError, ArithmeticError) as exc:
                record.update(success=False, failure_reason=str(exc))
                complete = False
            records.append(record)
        if not feasible_starts:
            complete = False
    return MolecularTPDResult(
        reference.basin,
        reference.state,
        minimum,
        complete,
        minimum < -spec.tpd_tolerance,
        ({"basin": reference.basin, "success": True, "gibbs_RT": float(z[active] @ d)},),
        tuple(records),
        "" if complete else "unresolved_joint_density_composition_tangent_search",
    )


def _totals(
    spec: MolecularMultiphaseSpec,
    phases: Sequence[MolecularFlashPhase],
    temperature: float,
    pressure: float,
    support: float,
    reference: float,
) -> dict[str, float]:
    total = fsum(p.amount_mol for p in phases)
    volume = fsum(p.volume_m3 for p in phases)
    mass = fsum(
        p.component_amounts_mol[k] * ref.molar_mass_kg_mol
        for p in phases
        for k, ref in zip(spec.model.component_ids, spec.model.ideal_references, strict=True)
    )
    h = fsum(
        p.amount_mol * cast(MolecularCaloricState, p.state).enthalpy_J_mol for p in phases
    ) + support * (temperature - reference)
    u = fsum(
        p.amount_mol * cast(MolecularCaloricState, p.state).internal_energy_J_mol for p in phases
    ) + support * (temperature - reference)
    s = fsum(
        p.amount_mol * cast(MolecularCaloricState, p.state).entropy_J_mol_K for p in phases
    ) + support * log(temperature / reference)
    gas = fsum(p.amount_mol for p in phases if spec.phase_kind_by_basin.get(p.basin) == "gas")
    return {
        "T": temperature,
        "P": pressure,
        "H": h,
        "S": s,
        "U": u,
        "V": volume,
        "rho": total / volume,
        "mass_density": mass / volume,
        "h_mass": h / mass,
        "s_mass": s / mass,
        "u_mass": u / mass,
        "Q": gas / total,
        "mass": mass,
    }


class _Closure:
    def __init__(
        self,
        spec: MolecularMultiphaseSpec,
        feed: Any,
        inputs: Mapping[str, float],
        support: float,
        reference: float,
    ) -> None:
        self.spec, self.feed, self.inputs = spec, feed, dict(inputs)
        self.support, self.reference = support, reference
        self.ids = spec.model.component_ids
        self.active = np.flatnonzero(feed > 0)
        self.total = float(feed.sum())
        self.basins = {b.name: b for b in spec.basins}
        self.free = tuple(k for k in ("T", "P") if k not in inputs)
        self.constraints = tuple(k for k in inputs if k not in {"T", "P"})
        self.calls = 0

    def normalized(
        self, phases: Sequence[MolecularFlashPhase], temperature: float, pressure: float
    ) -> tuple[MolecularFlashPhase, ...]:
        matrix = np.array([[p.component_amounts_mol.get(k, 0.0) for k in self.ids] for p in phases])
        if (
            not phases
            or any(p.basin not in self.basins for p in phases)
            or np.any(~np.isfinite(matrix))
            or np.any(matrix < 0)
            or any(set(p.component_amounts_mol) - set(self.ids) for p in phases)
            or np.any(matrix[:, self.feed == 0] != 0)
            or np.any(matrix[:, self.active] <= 0)
        ):
            raise ValueError("phase seed must preserve exact absent and positive component support")
        matrix[:, self.active] *= self.feed[self.active] / matrix[:, self.active].sum(axis=0)
        output = []
        for p, amounts in zip(phases, matrix, strict=True):
            basin = self.basins[p.basin]
            rho = p.state.molar_density_mol_m3
            if not basin.density_bounds_mol_m3[0] < rho < basin.density_bounds_mol_m3[1]:
                raise ValueError("phase seed density is outside its declared basin")
            state = self.spec.model.state(
                temperature, rho, tuple(map(float, amounts / amounts.sum()))
            )
            output.append(
                MolecularFlashPhase(
                    p.basin, dict(zip(self.ids, map(float, amounts), strict=True)), state
                )
            )
        return tuple(output)

    def homogeneous(
        self, temperature: float, pressure: float, basin: str
    ) -> tuple[MolecularFlashPhase, ...]:
        b = self.basins[basin]
        state = self.spec.model.state_tp(
            temperature,
            pressure,
            tuple(map(float, self.feed / self.total)),
            initial_density_mol_m3=b.initial_density_mol_m3,
            density_bounds_mol_m3=b.density_bounds_mol_m3,
        )
        return (
            MolecularFlashPhase(
                basin, dict(zip(self.ids, map(float, self.feed), strict=True)), state
            ),
        )

    def totals(self, phases: Sequence[MolecularFlashPhase], t: float, p: float) -> dict[str, float]:
        return _totals(self.spec, phases, t, p, self.support, self.reference)

    def prototype_seed(
        self,
        t: float,
        p: float,
        composition_starts: Sequence[Sequence[float]] | None,
        minimum_phases: int = 1,
    ) -> tuple[tuple[MolecularFlashPhase, ...], dict[str, Any]]:
        """Build a material-exact initial assemblage from declared phase starts.

        This only initializes the shared root. No source solution, phase amount
        floor or successful-equilibrium assertion enters the initialization.
        """
        if len(self.basins) < 2 and minimum_phases == 1:
            return (), {}
        z = self.feed / self.total
        if composition_starts is None:
            seeds = [z.copy(), np.where(z > 0, 1 / len(self.active), 0.0)]
            if len(self.active) > 1:
                for i in self.active:
                    x = np.zeros(len(z))
                    x[self.active] = 0.01 / (len(self.active) - 1)
                    x[i] = 0.99
                    seeds.append(x)
        else:
            seeds = [np.asarray(x, dtype=float) for x in composition_starts]
        attempts: list[dict[str, Any]] = []
        groups: list[list[tuple[str, Any]]] = []
        for basin in self.basins.values():
            group = []
            for index, x in enumerate(seeds):
                try:
                    state = self.spec.model.state_tp(
                        t,
                        p,
                        tuple(map(float, x)),
                        initial_density_mol_m3=basin.initial_density_mol_m3,
                        density_bounds_mol_m3=basin.density_bounds_mol_m3,
                    )
                    group.append((basin.name, state))
                    attempts.append({"basin": basin.name, "start": index, "success": True})
                except (ValueError, ArithmeticError) as exc:
                    attempts.append(
                        {
                            "basin": basin.name,
                            "start": index,
                            "success": False,
                            "failure_reason": str(exc),
                        }
                    )
            if group:
                groups.append(group)
        if not groups or len(groups) > self.spec.max_phases:
            return (), {
                "attempts": attempts,
                "failure_reason": "no_admissible_prototype_assemblage",
            }
        if np.prod([len(group) for group in groups]) > 10000:
            return (), {"attempts": attempts, "failure_reason": "prototype_combination_budget"}
        best: tuple[MolecularFlashPhase, ...] = ()
        best_g = float("inf")
        combinations_tested = 0
        choices = (
            product(*groups)
            if len(groups) >= minimum_phases
            else combinations([point for group in groups for point in group], minimum_phases)
        )
        for combination in choices:
            combinations_tested += 1
            xs = np.array([s.mole_fractions for _, s in combination])
            weights, _ = nnls(xs.T, z)
            present = np.flatnonzero(weights > self.spec.phase_fraction_tolerance)
            if len(present) < minimum_phases:
                # A deliberately nonvanishing initial guess for a constrained
                # branch, not a phase allocation accepted as equilibrium.
                weights = np.full(len(combination), 1 / len(combination))
                present = np.arange(len(combination))
            if not len(present):
                continue
            allocations = weights[present, None] * xs[present]
            if np.any(allocations[:, self.active].sum(axis=0) <= 0):
                continue
            allocations[:, self.active] *= self.feed[self.active] / allocations[:, self.active].sum(
                axis=0
            )
            candidate = []
            try:
                for index, amounts in zip(present, allocations, strict=True):
                    name, initial = combination[index]
                    basin = self.basins[name]
                    state = self.spec.model.state_tp(
                        t,
                        p,
                        tuple(map(float, amounts / amounts.sum())),
                        initial_density_mol_m3=initial.molar_density_mol_m3,
                        density_bounds_mol_m3=basin.density_bounds_mol_m3,
                    )
                    candidate.append(
                        MolecularFlashPhase(
                            name, dict(zip(self.ids, map(float, amounts), strict=True)), state
                        )
                    )
                gibbs = (
                    fsum(
                        ph.amount_mol
                        * float(
                            np.asarray(ph.state.mole_fractions)[self.active]
                            @ _mu(ph.state, self.active)
                        )
                        for ph in candidate
                    )
                    / self.total
                )
                if gibbs < best_g:
                    best, best_g = tuple(candidate), gibbs
            except (ValueError, ArithmeticError):
                continue
        return best, {
            "attempts": attempts,
            "combinations_tested": combinations_tested,
            "accepted_seed_phases": len(best),
        }

    def split(
        self, phases: Sequence[MolecularFlashPhase], start: Mapping[str, Any], t: float, p: float
    ) -> tuple[MolecularFlashPhase, ...]:
        x = np.asarray(start["composition"])
        # Add a small real inventory along the negative tangent direction;
        # no trace component is invented or removed.
        new_total = min(
            0.05 * self.total, 0.1 * float(np.min(self.feed[self.active] / x[self.active]))
        )
        added = new_total * x
        b = self.basins[start["basin"]]
        density = float(start.get("density_mol_m3", b.initial_density_mol_m3))
        state = self.spec.model.state(t, density, tuple(map(float, x)))
        if abs(state.pressure_Pa / p - 1) > max(
            self.spec.tpd_tolerance, self.spec.root_policy.tolerance
        ):
            state = self.spec.model.state_tp(
                t,
                p,
                tuple(map(float, x)),
                initial_density_mol_m3=density,
                density_bounds_mol_m3=b.density_bounds_mol_m3,
            )
        out = []
        for phase in phases:
            amounts = np.array([phase.component_amounts_mol[k] for k in self.ids])
            amounts[self.active] *= 1 - added[self.active] / self.feed[self.active]
            old_state = self.spec.model.state(
                t, phase.state.molar_density_mol_m3, tuple(map(float, amounts / amounts.sum()))
            )
            out.append(
                MolecularFlashPhase(
                    phase.basin, dict(zip(self.ids, map(float, amounts), strict=True)), old_state
                )
            )
        out.append(
            MolecularFlashPhase(b.name, dict(zip(self.ids, map(float, added), strict=True)), state)
        )
        return tuple(out)

    def reduce(self, phases: Sequence[MolecularFlashPhase]) -> tuple[MolecularFlashPhase, ...]:
        kept: list[MolecularFlashPhase] = []
        for phase in phases:
            if phase.amount_mol / self.total < self.spec.phase_fraction_tolerance:
                continue
            duplicate = next(
                (
                    j
                    for j, other in enumerate(kept)
                    if phase.basin == other.basin
                    and abs(
                        log(phase.state.molar_density_mol_m3 / other.state.molar_density_mol_m3)
                    )
                    < 1e-6
                    and max(
                        abs(a - b)
                        for a, b in zip(
                            phase.state.mole_fractions, other.state.mole_fractions, strict=True
                        )
                    )
                    < 1e-6
                ),
                None,
            )
            if duplicate is None:
                kept.append(phase)
            else:
                other = kept[duplicate]
                amounts = {
                    k: other.component_amounts_mol[k] + phase.component_amounts_mol[k]
                    for k in self.ids
                }
                kept[duplicate] = MolecularFlashPhase(other.basin, amounts, other.state)
        if not kept:
            raise ValueError("all candidate phases vanished")
        return tuple(kept)

    def tp_prepare(
        self, phases: tuple[MolecularFlashPhase, ...], t: float, p: float
    ) -> tuple[tuple[MolecularFlashPhase, ...], dict[str, Any]]:
        """Common Gibbs allocation prepares an N-phase simultaneous residual solve."""
        if len(phases) == 1:
            return phases, {}
        count, n = len(phases), len(self.active)
        matrix = np.array(
            [[ph.component_amounts_mol[self.ids[i]] for i in self.active] for ph in phases]
        )
        initial = np.r_[
            [log(ph.state.molar_density_mol_m3) for ph in phases],
            np.log(matrix[:-1] / matrix[-1]).ravel(),
        ]
        rt = self.spec.model.gas_constant_J_mol_K * t
        model = self.spec.model.residual_model
        bounds = [
            (
                log(self.basins[ph.basin].density_bounds_mol_m3[0]),
                log(self.basins[ph.basin].density_bounds_mol_m3[1]),
            )
            for ph in phases
        ]
        bounds += [(-70.0, 70.0)] * ((count - 1) * n)
        invalid_trials = 0

        def allocations(q: Any) -> tuple[Any, Any]:
            weights = softmax(
                np.vstack((np.reshape(q[count:], (count - 1, n)), np.zeros(n))), axis=0
            )
            amounts = np.zeros((count, len(self.ids)))
            amounts[:, self.active] = weights * self.feed[self.active]
            return weights, amounts

        def states(q: Any) -> tuple[MolecularFlashPhase, ...]:
            _, amounts = allocations(q)
            out = []
            for j, (phase, ns) in enumerate(zip(phases, amounts, strict=True)):
                s = self.spec.model.state(t, float(np.exp(q[j])), tuple(map(float, ns / ns.sum())))
                out.append(
                    MolecularFlashPhase(
                        phase.basin, dict(zip(self.ids, map(float, ns), strict=True)), s
                    )
                )
            return tuple(out)

        def objective(q: Any) -> tuple[float, Any]:
            nonlocal invalid_trials
            try:
                weights, amounts = allocations(q)
                mus, density_gradient = [], []
                for j, ns in enumerate(amounts):
                    rho = float(np.exp(q[j]))
                    x = tuple(map(float, ns / ns.sum()))
                    ar = SecondOrderPair(0) + model.alphar(t, SecondOrderPair(rho, dx=1), x)
                    mismatch = 1 + rho * ar.dx - p / (rho * rt)
                    mus.append(
                        np.asarray(molecular_residual_chemical_potentials(model, t, rho, x))[
                            self.active
                        ]
                        / rt
                        + log(rho)
                        + np.log(np.asarray(x)[self.active])
                        - mismatch
                    )
                    density_gradient.append(float(ns.sum()) / self.total * mismatch)
                mu_matrix = np.asarray(mus)
                average = np.sum(weights * mu_matrix, axis=0)
                value = float(self.feed[self.active] @ average / self.total)
                gradient = np.r_[
                    density_gradient,
                    (self.feed[self.active] / self.total * weights * (mu_matrix - average))[
                        :-1
                    ].ravel(),
                ]
                if not isfinite(value) or not np.all(np.isfinite(gradient)):
                    raise ValueError("nonfinite Gibbs allocation trial")
                return value, gradient
            except (ValueError, ArithmeticError):
                invalid_trials += 1
                return 1e100, np.zeros(len(q))

        try:
            optimum = minimize(
                objective,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=bounds,
                options={
                    "gtol": 1e-10,
                    "ftol": 1e-15,
                    "maxiter": self.spec.root_policy.maximum_iterations,
                    "maxls": 40,
                },
            )
            prepared = states(optimum.x)
            return prepared, {
                "success": bool(optimum.success),
                "iterations": int(optimum.nit),
                "message": str(optimum.message),
                "invalid_trial_count": invalid_trials,
            }
        except (ValueError, ArithmeticError) as exc:
            return phases, {"success": False, "failure_reason": str(exc)}

    def joint(
        self, phases: tuple[MolecularFlashPhase, ...], t: float, p: float
    ) -> tuple[Any, tuple[MolecularFlashPhase, ...], float, float, dict[str, Any]]:
        count, n = len(phases), len(self.active)
        values = {"T": t, "P": p}
        matrix = np.array(
            [[ph.component_amounts_mol[self.ids[i]] for i in self.active] for ph in phases]
        )
        initial = np.r_[
            [log(values[k]) for k in self.free],
            [log(ph.state.molar_density_mol_m3) for ph in phases],
            np.log(matrix[:-1] / matrix[-1]).ravel(),
        ]
        bounds = {"T": self.spec.temperature_bounds_K, "P": self.spec.pressure_bounds_Pa}
        ranges = [bounds[k] for k in self.free] + [
            self.basins[ph.basin].density_bounds_mol_m3 for ph in phases
        ]
        lower = np.r_[[log(b[0]) for b in ranges], np.full((count - 1) * n, -70.0)]
        upper = np.r_[[log(b[1]) for b in ranges], np.full((count - 1) * n, 70.0)]
        initial = np.minimum(upper, np.maximum(lower, initial))
        rt_total = self.total * self.spec.model.gas_constant_J_mol_K
        scales = {
            "H": rt_total * t,
            "U": rt_total * t,
            "S": rt_total,
            "V": self.total * self.spec.model.gas_constant_J_mol_K * t / p,
            "rho": p / (self.spec.model.gas_constant_J_mol_K * t),
            "Q": 1.0,
        }
        original = self.totals(phases, t, p)
        for k in self.constraints:
            scales[k] = max(abs(self.inputs[k]), abs(original[k]), scales.get(k, 1e-12), 1e-20)

        def unpack(q: Any) -> tuple[tuple[MolecularFlashPhase, ...], float, float]:
            coordinates = {
                **self.inputs,
                **{k: float(np.exp(q[i])) for i, k in enumerate(self.free)},
            }
            temperature, pressure = coordinates["T"], coordinates["P"]
            offset = len(self.free)
            weights = softmax(
                np.vstack((np.reshape(q[offset + count :], (count - 1, n)), np.zeros(n))), axis=0
            )
            amounts = np.zeros((count, len(self.ids)))
            amounts[:, self.active] = weights * self.feed[self.active]
            result = []
            for j, (phase, ns) in enumerate(zip(phases, amounts, strict=True)):
                state = self.spec.model.state(
                    temperature, float(np.exp(q[offset + j])), tuple(map(float, ns / ns.sum()))
                )
                result.append(
                    MolecularFlashPhase(
                        phase.basin, dict(zip(self.ids, map(float, ns), strict=True)), state
                    )
                )
            return tuple(result), temperature, pressure

        def residual(q: Any) -> Any:
            self.calls += 1
            try:
                out, temperature, pressure = unpack(q)
                mus = np.array([_mu(ph.state, self.active) for ph in out])
                totals = self.totals(out, temperature, pressure)
                return np.r_[
                    [log(ph.state.pressure_Pa / pressure) for ph in out],
                    (mus[:-1] - mus[-1]).ravel(),
                    [(totals[k] - self.inputs[k]) / scales[k] for k in self.constraints],
                ]
            except (ValueError, ArithmeticError) as exc:
                raise ResidualDomainError(str(exc)) from exc

        root = solve_root(
            residual,
            initial,
            jacobian=lambda q: finite_jacobian(
                residual, q, np.ones(len(q)), lower=lower, upper=upper
            ),
            variables=tuple(
                ResidualVariable(
                    f"equilibrium_{i}", "dimensionless", lower=float(lo), upper=float(hi)
                )
                for i, (lo, hi) in enumerate(zip(lower, upper, strict=True))
            ),
            residual_scales=np.ones(len(initial)),
            policy=self.spec.root_policy,
        )
        try:
            if root.value is None:
                raise ValueError("joint root has no accepted candidate")
            out, temperature, pressure = unpack(root.value)
        except (ValueError, ArithmeticError):
            out, temperature, pressure = phases, t, p
        return (
            root,
            out,
            temperature,
            pressure,
            {
                "success": root.success,
                "failure_reason": root.failure_reason,
                **root.diagnostics,
                "phase_basins": [ph.basin for ph in phases],
                "free_coordinates": self.free,
            },
        )


def solve_molecular_multiphase(
    spec: MolecularMultiphaseSpec,
    feed_amounts_mol: Mapping[str, float],
    *,
    inputs: Mapping[str, float],
    seed: MolecularMultiphaseSeed | None = None,
    composition_starts: Sequence[Sequence[float]] | None = None,
    support_heat_capacity_J_K: float = 0.0,
    support_reference_temperature_K: float = 298.15,
    verify_stability: bool = True,
) -> MolecularMultiphaseResult:
    """Solve two independent T/P/H/S/U/V/rho/mass-density/quality coordinates.

    H, S, U and V are extensive totals. h_mass/s_mass/u_mass use declared kg;
    mass_density uses the explicit ideal-reference molar masses. Q is the molar
    fraction of explicitly gas-labelled phases. Exact Q endpoints belong to the
    incipient-phase boundary solver. No phase allocation is installed here.
    ``verify_stability=False`` is an unqualified numerical trial for an outer
    integrator. Its endpoint must be resolved with full stability before use.
    """
    supported = {
        "T",
        "P",
        "H",
        "S",
        "U",
        "V",
        "rho",
        "mass_density",
        "h_mass",
        "s_mass",
        "u_mass",
        "Q",
    }
    ids = spec.model.component_ids
    if type(verify_stability) is not bool:
        raise ValueError("verify_stability must be boolean")
    if (
        len(inputs) != 2
        or set(inputs) - supported
        or any(isinstance(v, bool) or not isfinite(v) for v in inputs.values())
        or any(inputs[k] <= 0 for k in set(inputs) & {"T", "P", "V", "rho", "mass_density"})
    ):
        raise ValueError("multiphase closure requires two finite independent state coordinates")
    for aliases in (
        {"V", "rho", "mass_density"},
        {"H", "h_mass"},
        {"S", "s_mass"},
        {"U", "u_mass"},
    ):
        if len(set(inputs) & aliases) > 1:
            raise ValueError("state coordinates are dependent at fixed material inventory")
    if "Q" in inputs:
        if not 0 < inputs["Q"] < 1:
            raise ValueError("quality endpoints require an incipient-phase boundary solve")
        if (
            set(spec.phase_kind_by_basin) != {b.name for b in spec.basins}
            or "gas" not in spec.phase_kind_by_basin.values()
            or "liquid" not in spec.phase_kind_by_basin.values()
        ):
            raise ValueError("quality requires explicit gas and liquid labels for all basins")
    if set(feed_amounts_mol) - set(ids) or any(
        isinstance(v, bool) or not isfinite(v) or v < 0 for v in feed_amounts_mol.values()
    ):
        raise ValueError("multiphase feed requires nonnegative finite declared component amounts")
    feed = np.array([feed_amounts_mol.get(k, 0.0) for k in ids])
    if not isfinite(float(feed.sum())) or feed.sum() <= 0:
        raise ValueError("multiphase feed must contain a positive finite amount")
    if (
        not isfinite(support_heat_capacity_J_K)
        or support_heat_capacity_J_K < 0
        or not isfinite(support_reference_temperature_K)
        or support_reference_temperature_K <= 0
    ):
        raise ValueError("apparatus heat capacity and reference temperature are invalid")
    if seed is None and set(inputs) != {"T", "P"}:
        raise ValueError("non-TP state pairs require an explicit temperature/pressure seed")
    t = float(inputs.get("T", seed.temperature_K if seed else 0))
    p = float(inputs.get("P", seed.pressure_Pa if seed else 0))
    if (
        not spec.temperature_bounds_K[0] <= t <= spec.temperature_bounds_K[1]
        or not spec.pressure_bounds_Pa[0] <= p <= spec.pressure_bounds_Pa[1]
    ):
        raise ValueError("initial state is outside declared temperature/pressure bounds")
    closure = _Closure(
        spec, feed, inputs, support_heat_capacity_J_K, support_reference_temperature_K
    )
    diagnostic: dict[str, Any] = {
        "inputs": dict(inputs),
        "attempts": [],
        "phase_updates": [],
        "search_scope": "finite_declared_basins_and_composition_starts",
        "stability_checked": verify_stability,
        "equations": "joint_T_P_phase_densities_component_allocations_and_extensive_constraints",
    }
    queue: list[tuple[tuple[MolecularFlashPhase, ...], float, float]] = []
    initialization_calls = 0
    if seed is not None and seed.phases:
        if len(seed.phases) > spec.max_phases:
            raise ValueError("seed phase count exceeds its declared maximum")
        queue.append((closure.normalized(seed.phases, t, p), t, p))
    elif set(inputs) != {"T", "P"}:
        initialized = solve_molecular_multiphase(
            spec,
            feed_amounts_mol,
            inputs={"T": t, "P": p},
            composition_starts=composition_starts,
            support_heat_capacity_J_K=support_heat_capacity_J_K,
            support_reference_temperature_K=support_reference_temperature_K,
        )
        diagnostic["initial_TP_assemblage"] = initialized.to_dict()
        initialization_calls = initialized.diagnostics.get("actual_condition_evaluations", 0)
        if initialized.phases:
            queue.append((closure.normalized(initialized.phases, t, p), t, p))
    for basin in spec.basins:
        try:
            queue.append((closure.homogeneous(t, p, basin.name), t, p))
        except (ValueError, ArithmeticError) as exc:
            diagnostic["attempts"].append({"initial_basin": basin.name, "failure_reason": str(exc)})
    if set(inputs) == {"T", "P"} and not (seed is not None and seed.phases):
        prototype, prototype_record = closure.prototype_seed(t, p, composition_starts)
        diagnostic["prototype_initialization"] = prototype_record
        if prototype:
            queue.insert(0, (prototype, t, p))
    last_phases: tuple[MolecularFlashPhase, ...] = ()
    last_stability: tuple[MolecularTPDResult, ...] = ()
    failure = "no_initial_pressure_root"
    speculative_added = False
    for iteration in range(spec.maximum_phase_updates):
        if not queue:
            break
        phases, t, p = queue.pop(0)
        if len(phases) > 1 and set(inputs) == {"T", "P"}:
            prepared, prepare_record = closure.tp_prepare(phases, t, p)
            diagnostic["attempts"].append({"gibbs_preparation": prepare_record})
            reduced = closure.reduce(prepared)
            if len(reduced) != len(phases):
                diagnostic["phase_updates"].append(
                    {"action": "remove_or_merge", "before": len(phases), "after": len(reduced)}
                )
            phases = closure.normalized(reduced, t, p)
        root, candidate, t, p, record = closure.joint(phases, t, p)
        diagnostic["attempts"].append(record)
        last_phases = candidate
        if not root.success:
            failure = root.failure_reason
            if len(phases) == 1 and set(inputs) != {"T", "P"} and not speculative_added:
                speculative_added = True
                proposal, proposal_record = closure.prototype_seed(
                    t, p, composition_starts, minimum_phases=2
                )
                diagnostic["constrained_phase_proposal"] = proposal_record
                if proposal and len(proposal) <= spec.max_phases:
                    queue.insert(0, (proposal, t, p))
            # Phase disappearance can put the old branch outside its positive
            # allocation chart. Try every one-phase removal with exact material
            # renormalization, retaining the failed branch diagnostics.
            if len(phases) > 1:
                for keep in combinations(range(len(phases)), len(phases) - 1):
                    try:
                        subset = closure.normalized(tuple(candidate[i] for i in keep), t, p)
                        queue.append((subset, t, p))
                    except (ValueError, ArithmeticError) as exc:
                        diagnostic["attempts"].append({"removal_failure": str(exc)})
            continue
        reduced = closure.reduce(candidate)
        if len(reduced) < len(candidate):
            diagnostic["phase_updates"].append(
                {"action": "remove_or_merge", "before": len(candidate), "after": len(reduced)}
            )
            queue.insert(0, (closure.normalized(reduced, t, p), t, p))
            continue
        last_stability = (
            tuple(_joint_stability(spec, t, p, ph, composition_starts) for ph in candidate)
            if verify_stability
            else ()
        )
        if len(closure.active) == 1 and set(inputs) == {"T", "P"} and last_stability:
            reference_g = float(_mu(candidate[0].state, closure.active)[0])
            for other in spec.basins:
                if other.name == candidate[0].basin:
                    continue
                try:
                    other_phase = closure.homogeneous(t, p, other.name)[0]
                except (ValueError, ArithmeticError):
                    continue
                if (
                    abs(float(_mu(other_phase.state, closure.active)[0]) - reference_g)
                    <= spec.tpd_tolerance
                    and abs(
                        log(
                            other_phase.state.molar_density_mol_m3
                            / candidate[0].state.molar_density_mol_m3
                        )
                    )
                    > 1e-6
                ):
                    return MolecularMultiphaseResult(
                        False,
                        candidate,
                        stability=last_stability,
                        diagnostics=diagnostic,
                        failure_reason="pure_coexistence_amount_underdetermined",
                    )
        negative = sorted(
            (
                s
                for st in last_stability
                for s in st.starts
                if s.get("minimum_tpd_RT", 0) < -spec.tpd_tolerance
                and s.get("composition") is not None
                and s.get("candidate_phase", True)
            ),
            key=lambda s: s["minimum_tpd_RT"],
        )
        if negative:
            failure = (
                "phase_limit_with_remaining_instability"
                if len(candidate) >= spec.max_phases
                else "remaining_instability"
            )
            if len(candidate) < spec.max_phases:
                for start in negative:
                    try:
                        new = closure.split(candidate, start, t, p)
                        queue.insert(0, (new, t, p))
                        diagnostic["phase_updates"].append(
                            {
                                "action": "add",
                                "before": len(candidate),
                                "after": len(new),
                                "negative_tpd_RT": start["minimum_tpd_RT"],
                                "basin": start["basin"],
                            }
                        )
                        break
                    except (ValueError, ArithmeticError) as exc:
                        diagnostic["attempts"].append({"addition_failure": str(exc)})
            continue
        if any(not st.complete for st in last_stability):
            failure = "unresolved_post_split_stability"
            continue
        phases = tuple(
            sorted(
                candidate,
                key=lambda ph: (ph.basin, ph.state.mole_fractions, ph.state.molar_density_mol_m3),
            )
        )
        totals = closure.totals(phases, t, p)
        mus = np.array([_mu(ph.state, closure.active) for ph in phases])
        diagnostic.update(
            actual_condition_evaluations=closure.calls + initialization_calls,
            phase_iterations=iteration + 1,
        )
        return MolecularMultiphaseResult(
            True,
            phases,
            t,
            p,
            totals["H"],
            totals["S"],
            totals["U"],
            totals["V"],
            totals["mass"],
            totals["Q"] if spec.phase_kind_by_basin else None,
            max(
                abs(feed[i] - fsum(ph.component_amounts_mol[k] for ph in phases))
                for i, k in enumerate(ids)
            ),
            float(np.max(np.abs(mus - mus[-1]))),
            max(abs(ph.state.pressure_Pa / p - 1) for ph in phases),
            {k: totals[k] - v for k, v in inputs.items()},
            last_stability,
            diagnostic,
            stability_checked=verify_stability,
        )
    diagnostic.update(
        actual_condition_evaluations=closure.calls + initialization_calls,
        pending_candidates=len(queue),
    )
    if queue:
        failure = "phase_update_limit: " + failure
    return MolecularMultiphaseResult(
        False,
        last_phases,
        stability=last_stability,
        diagnostics=diagnostic,
        failure_reason=failure,
        stability_checked=verify_stability,
    )
