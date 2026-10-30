"""Declared-basin molecular TP stability and two-phase inventory closure.

Only the supplied density basins and composition starts are searched. Successful
finite searches are not a proof of global stability. No ideal caloric reference
is implied; each phase carries the common residual EOS response.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite, log
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit, softmax

from chemworld.physchem.molecular_eos import (
    MolecularEOSState,
    ResidualHelmholtzModel,
    molecular_density_from_pressure,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.intp]


@dataclass(frozen=True)
class MolecularDensityBasin:
    """A named local pressure root; boundaries and seed are caller supplied."""

    name: str
    density_bounds_mol_m3: tuple[float, float]
    initial_density_mol_m3: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "density_bounds_mol_m3", tuple(self.density_bounds_mol_m3))
        if (not self.name or len(self.density_bounds_mol_m3) != 2 or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (*self.density_bounds_mol_m3, self.initial_density_mol_m3)
        ) or not self.density_bounds_mol_m3[0] < self.initial_density_mol_m3
                < self.density_bounds_mol_m3[1]):
            raise ValueError("density basin requires a name, positive bounds and interior seed")


@dataclass(frozen=True)
class MolecularTPDResult:
    reference_basin: str | None
    reference_state: MolecularEOSState | None
    minimum_tpd_RT: float | None
    complete: bool
    unstable: bool
    root_attempts: tuple[dict[str, Any], ...]
    starts: tuple[dict[str, Any], ...]
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "scope": "finite_declared_basins_and_starts"}


@dataclass(frozen=True)
class MolecularFlashPhase:
    basin: str
    component_amounts_mol: dict[str, float]
    state: MolecularEOSState

    @property
    def amount_mol(self) -> float:
        return sum(self.component_amounts_mol.values())

    @property
    def volume_m3(self) -> float:
        return self.amount_mol / self.state.molar_density_mol_m3

    def to_dict(self) -> dict[str, Any]:
        return {"basin": self.basin, "component_amounts_mol": self.component_amounts_mol,
                "amount_mol": self.amount_mol, "volume_m3": self.volume_m3,
                "state": self.state.to_dict()}


@dataclass(frozen=True)
class MolecularTPFlashResult:
    success: bool
    phases: tuple[MolecularFlashPhase, ...]
    stability: MolecularTPDResult
    post_split_stability: tuple[MolecularTPDResult, ...] = ()
    material_balance_error_mol: float | None = None
    chemical_potential_error_RT: float | None = None
    pressure_relative_error: float | None = None
    failure_reason: str = ""
    solver_attempts: tuple[dict[str, Any], ...] = ()

    @property
    def phase_status(self) -> str:
        return "unresolved" if not self.success else "two_phase" if len(self.phases) == 2 else (
            "single_phase")

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "phase_status": self.phase_status,
                "phases": [p.to_dict() for p in self.phases],
                "scope": "finite_declared_basins_and_starts",
                "allocation_logit_bounds": [-70., 70.],
                "jacobian": "bounded_finite_difference_of_common_residual_chemical_potentials"}


def _inputs(model: ResidualHelmholtzModel, temperature: float, pressure: float,
            fractions: Sequence[float], basins: Sequence[MolecularDensityBasin],
            max_iterations: int, tolerance: float) -> tuple[FloatArray, IndexArray]:
    x = np.asarray(fractions, dtype=float)
    if (not model.component_ids or len(set(model.component_ids)) != len(model.component_ids)
            or x.shape != (len(model.component_ids),) or not np.all(np.isfinite(x))
            or np.any(x < 0) or abs(float(x.sum())-1) > 1e-10):
        raise ValueError("molecular TP composition requires normalized nonnegative fractions")
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0
           for v in (temperature, pressure, tolerance)):
        raise ValueError("molecular TP requires positive finite T, P and tolerance")
    if (type(max_iterations) is not int or max_iterations < 1 or not basins
            or len({b.name for b in basins}) != len(basins)):
        raise ValueError("molecular TP requires unique basins and positive iteration limit")
    active = np.flatnonzero(x > 0)
    # The softmax reference must not be a vanishing trace: otherwise every
    # major-component coordinate drifts together along a nearly flat direction.
    anchor = int(np.argmax(x))
    active = np.array([i for i in active if i != anchor]+[anchor])
    return x/x.sum(), active


def _state(model: ResidualHelmholtzModel, temperature: float, pressure: float,
           x: Sequence[float] | FloatArray, basin: MolecularDensityBasin) -> MolecularEOSState:
    return molecular_density_from_pressure(
        model, temperature, pressure, tuple(float(v) for v in x),
        initial_density_mol_m3=basin.initial_density_mol_m3,
        density_bounds_mol_m3=basin.density_bounds_mol_m3)


def _mu(state: MolecularEOSState, active: IndexArray) -> FloatArray:
    if state.log_fugacity_coefficients is None:
        raise ValueError("positive-pressure fugacity coefficients are unavailable")
    fractions = np.asarray(state.mole_fractions)[active]
    if np.any(fractions <= 0):
        raise ValueError("positive-support chemical potential composition underflow")
    return np.log(fractions) + np.asarray(
        state.log_fugacity_coefficients)[active]


def _seeds(z: FloatArray, active: IndexArray,
           supplied: Sequence[Sequence[float]] | None) -> list[FloatArray]:
    if supplied is None:
        values = [z, np.where(z > 0, 1/len(active), 0)]
        if len(active) > 1:
            for i in active:
                w: FloatArray = np.zeros(len(z))
                w[active] = .05/(len(active)-1)
                w[i] = .95
                values.append(w)
    else:
        values = [np.asarray(w, dtype=float) for w in supplied]
        if not values:
            raise ValueError("composition starts cannot be empty")
    for w in values:
        if (w.shape != z.shape or not np.all(np.isfinite(w)) or np.any(w < 0)
                or abs(float(w.sum())-1) > 1e-10 or np.any(w[active] <= 0)
                or np.any(w[z == 0] != 0)):
            raise ValueError("TPD starts must preserve exact positive feed support")
    return values


def molecular_tp_stability(
    model: ResidualHelmholtzModel, temperature_K: float, pressure_Pa: float,
    mole_fractions: Sequence[float], *, basins: Sequence[MolecularDensityBasin],
    composition_starts: Sequence[Sequence[float]] | None = None,
    max_iterations: int = 200, tpd_tolerance: float = 1e-8,
    reference_basin: str | None = None,
) -> MolecularTPDResult:
    """Minimize normalized phi-phi TPD from every declared start and basin.

    Zero-feed components stay exactly absent; positive traces are never clipped.
    Failed density roots or optimizer starts make the finite search incomplete.
    A negative trial is evidence of instability even if another search failed.
    """
    basins = tuple(basins)
    z, active = _inputs(model, temperature_K, pressure_Pa, mole_fractions, basins,
                        max_iterations, tpd_tolerance)
    seeds = _seeds(z, active, composition_starts)
    if reference_basin is not None and reference_basin not in {b.name for b in basins}:
        raise ValueError("reference basin is not declared")
    roots, candidates = [], []
    for basin in basins:
        try:
            state = _state(model, temperature_K, pressure_Pa, z, basin)
            g = float(z[active] @ _mu(state, active))
            candidates.append((g, basin.name, state))
            roots.append({"basin": basin.name, "success": True, "gibbs_RT": g})
        except (ValueError, ArithmeticError) as exc:
            roots.append({"basin": basin.name, "success": False, "failure_reason": str(exc)})
    eligible = [c for c in candidates if reference_basin is None or c[1] == reference_basin]
    if not eligible:
        return MolecularTPDResult(None, None, None, False, False, tuple(roots), (),
                                   "no_reference_pressure_root")
    _, ref_name, ref_state = min(eligible, key=lambda c: c[0])
    d = _mu(ref_state, active)
    starts = []
    minimum = 0.0
    for basin in basins:
        for index, seed in enumerate(seeds):
            outcome: dict[str, Any] = {"basin": basin.name, "start": index,
                                       "initial_composition": tuple(float(v) for v in seed)}
            best: list[Any] = [float("inf"), None]

            def objective(q: FloatArray, basin: MolecularDensityBasin = basin,
                          best: list[Any] = best) -> tuple[float, FloatArray]:
                w = softmax(np.r_[q, 0.0])
                if np.any(w <= 0):
                    raise ValueError("TPD composition underflow")
                full = np.zeros(len(z))
                full[active] = w
                state = _state(model, temperature_K, pressure_Pa, full, basin)
                delta = _mu(state, active)-d
                value = float(w @ delta)
                if value < best[0]:
                    best[:] = [value, tuple(float(v) for v in full)]
                return value, (w*(delta-value))[:-1]

            try:
                if len(active) == 1:
                    _value, _ = objective(np.empty(0))
                    success, nit, message, gradient = True, 0, "pure support", 0.0
                else:
                    initial = np.log(seed[active][:-1])-log(seed[active][-1])
                    result = minimize(objective, initial, jac=True, method="BFGS", options={
                        "gtol": min(tpd_tolerance, 1e-8), "maxiter": max_iterations})
                    gradient = float(np.max(np.abs(result.jac)))
                    # Pressure-root roundoff can stop a Gibbs line search near a minimum.
                    # Refine its stationary equations with the existing residual engine.
                    if (not result.success and result.nit < max_iterations
                            and gradient < 1e-5):
                        dimension = len(active)-1

                        def stationarity(q: FloatArray) -> FloatArray:
                            try:
                                return objective(q)[1]
                            except (ValueError, ArithmeticError) as exc:
                                raise ResidualDomainError(str(exc)) from exc

                        def stationary_jacobian(
                            q: FloatArray, dimension: int = dimension,
                        ) -> FloatArray:
                            return finite_jacobian(stationarity, q, np.ones(dimension))

                        polish = solve_root(stationarity, result.x,
                            jacobian=stationary_jacobian, variables=tuple(
                                    ResidualVariable(f"log_ratio_{i}", "dimensionless")
                                    for i in range(dimension)),
                            residual_scales=np.ones(dimension), policy=RootPolicy(
                                tolerance=min(tpd_tolerance, 1e-8),
                                maximum_iterations=max_iterations))
                        outcome["stationarity_refinement"] = {
                            "success": polish.success, "failure_reason": polish.failure_reason,
                            **polish.diagnostics}
                        if polish.success:
                            _value, derivative = objective(polish.value)
                            gradient = float(np.max(np.abs(derivative)))
                    success = bool(result.success or gradient <= min(tpd_tolerance, 1e-8))
                    nit, message = int(result.nit), str(result.message)
                outcome.update(success=success, iterations=nit, message=message,
                               minimum_tpd_RT=float(best[0]), composition=best[1],
                               stationarity_error=gradient,
                               stationarity_tolerance=min(tpd_tolerance, 1e-8))
            except (ValueError, ArithmeticError) as exc:
                outcome.update(success=False, failure_reason=str(exc))
                if best[1] is not None:
                    outcome.update(minimum_tpd_RT=float(best[0]), composition=best[1])
            if best[1] is not None:
                minimum = min(minimum, float(best[0]))
            starts.append(outcome)
    complete = all(r["success"] for r in roots) and all(s["success"] for s in starts)
    return MolecularTPDResult(ref_name, ref_state, minimum, complete,
                               minimum < -tpd_tolerance, tuple(roots), tuple(starts),
                               "" if complete else "unresolved_density_or_tpd_starts")


def molecular_tp_flash(
    model: ResidualHelmholtzModel, temperature_K: float, pressure_Pa: float,
    feed_amounts_mol: Mapping[str, float], *, basins: Sequence[MolecularDensityBasin],
    composition_starts: Sequence[Sequence[float]] | None = None,
    max_iterations: int = 200, tpd_tolerance: float = 1e-8,
) -> MolecularTPFlashResult:
    """Return one or two actual phase inventories, or an explicit failed result.

    Gibbs minimization seeds the shared root solver for equal chemical
    potentials. Component allocations enforce material closure identically.
    Post-split finite TPD searches reject remaining or unresolved instability.
    Allocation logits are bounded to [-70, 70]; more extreme partitioning
    reports a failed root rather than clipping or dropping a feed component.
    Pure-fluid coexistence amounts are not determined by T,P alone and fail.
    """
    if (set(feed_amounts_mol)-set(model.component_ids) or any(
        isinstance(v, bool) or not isfinite(v) or v < 0 for v in feed_amounts_mol.values()
    )):
        raise ValueError("molecular flash requires finite nonnegative declared inventories")
    feed = {k: float(feed_amounts_mol.get(k, 0)) for k in model.component_ids}
    amounts = np.array(tuple(feed.values()))
    total = float(amounts.sum())
    if not isfinite(total) or total <= 0:
        raise ValueError("molecular flash requires a positive total inventory")
    z = amounts/total
    active = np.flatnonzero(amounts > 0)
    basins = tuple(basins)
    stability = molecular_tp_stability(model, temperature_K, pressure_Pa, tuple(z), basins=basins,
        composition_starts=composition_starts, max_iterations=max_iterations,
        tpd_tolerance=tpd_tolerance)
    if not stability.complete:
        return MolecularTPFlashResult(False, (), stability, failure_reason=stability.failure_reason)
    if stability.reference_state is None or stability.reference_basin is None:
        raise AssertionError(
            "invariant violated: stability.reference_state or stability.reference_basin is None"
        )
    if not stability.unstable:
        # Distinct equal-G pure roots need another extensive constraint for their amounts.
        ref_g = float(z[active] @ _mu(stability.reference_state, active))
        if len(active) == 1 and any(r["basin"] != stability.reference_basin
                and abs(r["gibbs_RT"]-ref_g) < tpd_tolerance
                and abs(_state(model, temperature_K, pressure_Pa, z, b).molar_density_mol_m3
                        /stability.reference_state.molar_density_mol_m3-1) > 1e-6
                for r in stability.root_attempts for b in basins if b.name == r["basin"]):
            return MolecularTPFlashResult(False, (), stability,
                                           failure_reason="pure_coexistence_amount_underdetermined")
        phase = MolecularFlashPhase(stability.reference_basin, feed, stability.reference_state)
        return MolecularTPFlashResult(True, (phase,), stability, material_balance_error_mol=0,
            chemical_potential_error_RT=0,
            pressure_relative_error=abs(phase.state.pressure_Pa/pressure_Pa-1))
    if len(active) == 1:
        return MolecularTPFlashResult(False, (), stability,
                                       failure_reason="pure_coexistence_amount_underdetermined")
    reference = next(b for b in basins if b.name == stability.reference_basin)
    initial_g = float(z[active] @ _mu(stability.reference_state, active))
    attempts: list[dict[str, Any]] = []
    solutions: list[tuple[float, tuple[MolecularFlashPhase, ...], float]] = []
    used: list[tuple[str, FloatArray]] = []
    for seed in stability.starts:
        if seed.get("minimum_tpd_RT", 0) >= -tpd_tolerance:
            continue
        w = np.asarray(seed["composition"])[active]
        if any(seed["basin"] == name and np.max(np.abs(w-previous)) < 1e-6
               for name, previous in used):
            continue
        used.append((seed["basin"], w))
        first_basin = next(b for b in basins if b.name == seed["basin"])
        # Seed a small emerging phase along the negative-TPD direction. Taking
        # nearly all of its limiting component can cross the Gibbs barrier and
        # put a local optimizer back in the homogeneous solution's basin.
        initial_beta = min(.1, .1*float(np.min(z[active]/w)))
        fractions = initial_beta*w/z[active]
        initial = np.log(fractions)-np.log1p(-fractions)
        record: dict[str, Any] = {"first_basin": first_basin.name,
                                  "second_basin": reference.name}

        def states(q: FloatArray, first_basin: MolecularDensityBasin = first_basin
                   ) -> tuple[FloatArray, FloatArray, list[MolecularEOSState]]:
            f = expit(q)
            first = z[active]*f
            second = z[active]*expit(-q)
            pair = []
            for ns, basin in ((first, first_basin), (second, reference)):
                full = np.zeros(len(z))
                full[active] = ns/ns.sum()
                try:
                    pair.append(_state(model, temperature_K, pressure_Pa, full, basin))
                except (ValueError, ArithmeticError) as exc:
                    raise ResidualDomainError(str(exc)) from exc
            return first, second, pair

        def residual(q: FloatArray) -> FloatArray:
            _, _, pair = states(q)
            return _mu(pair[0], active)-_mu(pair[1], active)

        def objective(q: FloatArray) -> tuple[float, FloatArray]:
            first, second, pair = states(q)
            mus = [_mu(s, active) for s in pair]
            return float(first@mus[0]+second@mus[1]), (
                z[active]*expit(q)*expit(-q)*(mus[0]-mus[1]))

        try:
            n = len(active)
            lower, upper = np.full(n, -70.), np.full(n, 70.)
            optimum = minimize(objective, initial, jac=True, method="L-BFGS-B",
                bounds=list(zip(lower, upper, strict=True)), options={
                    "gtol": 1e-10, "ftol": 1e-14, "maxiter": max_iterations})
            record["minimization"] = {"success": bool(optimum.success),
                "iterations": int(optimum.nit), "message": str(optimum.message)}
            def residual_jacobian(
                q: FloatArray, n: int = n, lower: FloatArray = lower, upper: FloatArray = upper,
            ) -> FloatArray:
                return finite_jacobian(residual, q, np.ones(n), lower=lower, upper=upper)

            root = solve_root(residual, optimum.x,
                jacobian=residual_jacobian,
                variables=tuple(ResidualVariable(f"allocation_{i}", "dimensionless",
                    lower=-70, upper=70) for i in active), residual_scales=np.ones(n),
                policy=RootPolicy(tolerance=2e-9, maximum_iterations=max_iterations))
            record["root"] = {"success": root.success, "failure_reason": root.failure_reason,
                              **root.diagnostics}
            if not root.success:
                attempts.append(record)
                continue
            first, second, pair = states(root.value)
            g = objective(root.value)[0]
            different = max(np.max(np.abs(np.array(pair[0].mole_fractions)
                                         -np.array(pair[1].mole_fractions))),
                            abs(log(pair[0].molar_density_mol_m3/pair[1].molar_density_mol_m3)))
            if different < 1e-6 or initial_g-g <= tpd_tolerance:
                record["failure_reason"] = "trivial_or_nonlowering_split"
            else:
                phase_candidates = []
                for ns, state, basin in zip((first, second), pair, (first_basin, reference),
                                             strict=True):
                    inventory = dict.fromkeys(model.component_ids, 0.)
                    inventory.update((model.component_ids[i], float(ns[j]*total))
                                     for j, i in enumerate(active))
                    phase_candidates.append(MolecularFlashPhase(basin.name, inventory, state))
                solutions.append((g, tuple(phase_candidates),
                                  float(np.max(np.abs(residual(root.value))))))
                record["gibbs_decrease_RT_mol"] = total*(initial_g-g)
        except (ValueError, ArithmeticError, ResidualDomainError) as exc:
            record["failure_reason"] = str(exc)
        attempts.append(record)
    if not solutions:
        return MolecularTPFlashResult(False, (), stability, failure_reason="two_phase_solve_failed",
                                       solver_attempts=tuple(attempts))
    _, phases, mu_error = min(solutions, key=lambda s: s[0])
    phases = tuple(sorted(phases, key=lambda p: p.state.mole_fractions))
    post = tuple(molecular_tp_stability(model, temperature_K, pressure_Pa, p.state.mole_fractions,
        basins=basins, composition_starts=composition_starts, max_iterations=max_iterations,
        tpd_tolerance=tpd_tolerance, reference_basin=p.basin) for p in phases)
    material = max(abs(feed[k]-sum(p.component_amounts_mol[k] for p in phases)) for k in feed)
    pressure = max(abs(p.state.pressure_Pa/pressure_Pa-1) for p in phases)
    success = all(s.complete and not s.unstable for s in post)
    return MolecularTPFlashResult(success, phases, stability, post, material, mu_error, pressure,
        "" if success else "remaining_or_unresolved_post_split_instability", tuple(attempts))
