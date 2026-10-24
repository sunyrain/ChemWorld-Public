"""Numerical search helpers over the existing atom-molar Gibbs surface.

No thermodynamic equation or state owner lives here. Finite sampling and local
minimization are explicit search limitations, not a global stability proof.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import minimize

from chemworld.physchem.calphad_response import analytic_response
from chemworld.physchem.residual_system import ResidualDomainError, ResidualVariable, solve_root


def _partitions(total: int, size: int) -> Any:
    if size == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in _partitions(total - first, size - 1):
                yield (first, *rest)


class PhaseSurface:
    def __init__(self, model: Any, elements: Any, temperature: float,
                 pressure: float, floor: float) -> None:
        self.model, self.elements = model, elements
        self.temperature, self.pressure, self.floor = temperature, pressure, floor
        self.sizes = tuple(len(s) for s in model.constituents)
        self.dimension = sum(n - 1 for n in self.sizes)
        self._cache: Any = None

    def sites(self, q: Any) -> Any:
        result, offset = [], 0
        for members in self.model.constituents:
            values = tuple(q[offset:offset + len(members) - 1])
            result.append(dict(zip(members, (*values, 1 - sum(values)), strict=True)))
            offset += len(members) - 1
        return tuple(result)

    def values(self, q: Any) -> Any:
        sites = self.sites(q)
        if any(float(y) < 0 or float(y) > 1 for subl in sites for y in subl.values()):
            raise ResidualDomainError("constitution outside its simplex")
        composition = self.model._composition(sites)[1]
        return np.array((float(sum(self.model._energies(
            sites, self.temperature, self.pressure).values())),
            *(float(composition.get(e, 0.)) for e in self.elements)))

    def response(self, q: Any) -> Any:
        key = tuple(float(v) for v in q)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        self.values(key)

        def evaluate(z: Any) -> Any:
            sites = self.sites(z)
            composition = self.model._composition(sites)[1]
            return (sum(self.model._energies(sites, self.temperature, self.pressure).values()),
                    *(composition.get(e, 0.) for e in self.elements))

        result = analytic_response(evaluate, key)
        self._cache = (key, result)
        return result

    def preparation_response(self, q: Any) -> Any:
        """A differentiable feasible extension for constrained optimizer trials.

        SLSQP may evaluate an infeasible simplex trial before its constraint.
        Its objective extension retracts each positive row radially; the
        original linear simplex constraints still apply. Accepted constitutions
        and the common root always use the unextended thermodynamic response.
        """
        projected = np.asarray(q, dtype=float).copy()
        transform = np.eye(self.dimension)
        offset = 0
        for size in self.sizes:
            n = size-1
            row = projected[offset:offset+n]
            if sum(row) > 1-self.floor:
                total = sum(row)
                projected[offset:offset+n] = row*(1-self.floor)/total
                transform[offset:offset+n, offset:offset+n] = (
                    (np.eye(n)-row[:, None]/total)*(1-self.floor)/total)
            offset += n
        v, j, h = self.response(projected)
        return v, j@transform, h

    def interior(self, q: Any) -> Any:
        rows: list[Any] = []
        for subl in self.sites(q):
            values = np.array(tuple(subl.values()))
            values = self.floor + (1 - len(values)*self.floor)*values
            rows.extend(values[:-1])
        return np.array(rows)

    def samples(self, intervals: int) -> Any:
        axes = [tuple(_partitions(intervals, n)) for n in self.sizes]
        if np.prod([len(a) for a in axes]) > 10000:
            raise ValueError("declared constitution grid exceeds 10000 points per phase")
        return [self.interior(tuple(v/intervals for row in p for v in row[:-1]))
                for p in product(*axes)]

    def constraints(self, q: Any) -> Any:
        rows, offset = [], 0
        for size in self.sizes:
            if size > 1:
                rows.append(1 - self.floor - sum(q[offset:offset + size - 1]))
            offset += size - 1
        return np.array(rows)

    def constraint_jacobian(self) -> Any:
        rows, offset = [], 0
        for size in self.sizes:
            if size > 1:
                row = np.zeros(self.dimension)
                row[offset:offset + size - 1] = -1
                rows.append(row)
            offset += size - 1
        return np.array(rows).reshape(-1, self.dimension) if self.dimension else np.zeros((0, 0))


def refine_assemblage(surfaces: Any, active: Any, fractions: Any, target: Any,
                      energy_scale: float, policy: Any) -> Any:
    """Constrained energy seed, then the common analytic KKT root."""
    count, ne = len(active), len(target)
    slices, offset = [], count
    for _index, q in active:
        slices.append(slice(offset, offset + len(q)))
        offset += len(q)
    q0 = np.r_[fractions, *(q for _, q in active)]

    def response(z: Any, preparation: bool = False) -> Any:
        return [(surfaces[i].preparation_response(z[s])
                 if preparation else surfaces[i].response(z[s]))
                for (i, _), s in zip(active, slices, strict=True)]

    def objective(z: Any) -> Any:
        parts = response(z, True)
        gradient = np.zeros(offset)
        value = 0.
        for k, ((v, j, _h), sl) in enumerate(zip(parts, slices, strict=True)):
            value += z[k]*v[0]/energy_scale
            gradient[k] = v[0]/energy_scale
            gradient[sl] = z[k]*j[0]/energy_scale
        return value, gradient

    def balance(z: Any) -> Any:
        return sum(z[k]*r[0][1:] for k, r in enumerate(response(z, True))) - target

    def balance_jacobian(z: Any) -> Any:
        jac = np.zeros((ne, offset))
        for k, ((v, j, _h), sl) in enumerate(zip(response(z, True), slices, strict=True)):
            jac[:, k], jac[:, sl] = v[1:], z[k]*j[1:]
        return jac

    def simplex(z: Any) -> Any:
        return np.concatenate([surfaces[i].constraints(z[s])
                               for (i, _), s in zip(active, slices, strict=True)])

    def simplex_jacobian(z: Any) -> Any:
        rows: list[Any] = []
        for (i, _), s in zip(active, slices, strict=True):
            local = surfaces[i].constraint_jacobian()
            full = np.zeros((len(local), offset))
            full[:, s] = local
            rows.extend(full)
        return np.array(rows).reshape(-1, offset)

    bounds = [(0., 1.)]*count + [(policy.site_floor, 1-policy.site_floor)]*(offset-count)
    constraints = [{"type": "eq", "fun": balance, "jac": balance_jacobian}]
    if offset > count:
        constraints.append({"type": "ineq", "fun": simplex, "jac": simplex_jacobian})
    optimum = minimize(objective, q0, jac=True, bounds=bounds, constraints=constraints,
        method="SLSQP", options={"ftol": 1e-12, "maxiter": policy.maximum_iterations})
    diagnostic: dict[str, Any] = {"minimization_success": bool(optimum.success),
                  "message": str(optimum.message),
                  "iterations": int(optimum.nit)}
    # Remove zero-amount candidates and coalesce the same model/constitution.
    retained: list[Any] = []
    weights: list[float] = []
    for k, ((index, _), sl) in enumerate(zip(active, slices, strict=True)):
        if optimum.x[k] <= policy.amount_tolerance:
            continue
        point = optimum.x[sl]
        same = next((j for j, (i, q) in enumerate(retained) if i == index
                     and np.max(np.abs(point-q), initial=0.) < 1e-5), None)
        if same is None:
            retained.append((index, point.copy()))
            weights.append(float(optimum.x[k]))
        else:
            old = weights[same]
            retained[same] = (index, (old*retained[same][1]+optimum.x[k]*point)/(old+optimum.x[k]))
            weights[same] += float(optimum.x[k])
    if len(retained) != len(active):
        if not retained:
            return None, None, None, {**diagnostic, "failure_reason": "no_positive_phase_amount"}
        points, weights2, mu2, detail = refine_assemblage(
            surfaces, retained, weights, target, energy_scale, policy)
        detail["coalesced_preparation"] = diagnostic
        return points, weights2, mu2, detail
    points = [(i, optimum.x[s]) for (i, _), s in zip(active, slices, strict=True)]
    parts = response(optimum.x)
    potential_matrix = np.vstack([row for v, j, _h in parts for row in (v[None, 1:], j[1:].T)])
    potential_rhs = np.concatenate([np.r_[v[0], j[0]]/energy_scale for v, j, _h in parts])
    mu, _, rank, _singular = np.linalg.lstsq(potential_matrix, potential_rhs, rcond=1e-11)
    diagnostic.update(potential_rank=int(rank), potential_unknowns=ne,
                      potential_residual=float(np.max(np.abs(potential_matrix@mu-potential_rhs))))
    if rank < ne:
        return None, None, None, {**diagnostic,
                                 "failure_reason": "underdetermined_element_potentials"}

    def equations(z: Any, want_jacobian: bool = False) -> Any:
        values = response(z[:offset])
        residual = np.zeros(offset+ne)
        jac = np.zeros((offset+ne, offset+ne))
        chemical = z[offset:]
        for k, ((v, j, h), sl) in enumerate(zip(values, slices, strict=True)):
            residual[k] = v[0]/energy_scale-chemical@v[1:]
            residual[sl] = j[0]/energy_scale-chemical@j[1:]
            jac[k, sl], jac[k, offset:] = residual[sl], -v[1:]
            jac[sl, sl] = h[0]/energy_scale-np.einsum("i,ijk->jk", chemical, h[1:])
            jac[sl, offset:] = -j[1:].T
            residual[offset:] += z[k]*v[1:]
            jac[offset:, k], jac[offset:, sl] = v[1:], z[k]*j[1:]
        residual[offset:] -= target
        return jac if want_jacobian else residual

    root = solve_root(equations, np.r_[optimum.x, mu],
        jacobian=lambda z: equations(z, True),
        variables=tuple([ResidualVariable(f"fraction:{k}", "1", lower=0., upper=1.)
                         for k in range(count)] +
            [ResidualVariable(f"site:{k}", "1", lower=policy.site_floor,
                              upper=1-policy.site_floor) for k in range(offset-count)] +
            [ResidualVariable(f"mu:{e}", "RT") for e in range(ne)]),
        residual_scales=np.ones(offset+ne), policy=policy.root)
    diagnostic["root"] = {"success": root.success, "failure_reason": root.failure_reason,
                          **root.diagnostics}
    if not root.success:
        return None, None, None, {**diagnostic, "failure_reason": "assemblage_root_failed"}
    z = root.value
    jacobian = equations(z, True)
    rank = np.linalg.matrix_rank(jacobian, tol=1e-11)
    diagnostic.update(assemblage_rank=int(rank), assemblage_unknowns=len(z))
    if rank < len(z):
        return None, None, None, {**diagnostic,
                                 "failure_reason": "underdetermined_assemblage"}
    points = [(i, z[s]) for (i, _), s in zip(active, slices, strict=True)]
    return points, z[:count], z[offset:]*energy_scale, diagnostic


def search_phase(surface: PhaseSurface, samples: Any, mu: Any, energy_scale: float,
                 policy: Any) -> Any:
    """Finite multistart search of g - mu.x; every attempted outcome is retained."""
    if not surface.dimension:
        value = surface.values(())
        return [(np.zeros(0), float(value[0]-mu@value[1:]))], []

    def objective(q: Any) -> Any:
        v, j, _h = surface.preparation_response(q)
        return float((v[0]-mu@v[1:])/energy_scale), (j[0]-mu@j[1:])/energy_scale

    scored = sorted(enumerate(samples), key=lambda p: (
        surface.values(p[1])[0]-mu@surface.values(p[1])[1:], p[0]))
    # Keep distinct regions instead of taking many adjacent starts in one well.
    seeds: list[Any] = []
    for index, point in scored:
        if not seeds or min(np.linalg.norm(point-other) for _, other in seeds) >= .15:
            seeds.append((index, point))
        if len(seeds) >= policy.maximum_search_starts:
            break
    minima, attempts = [], []
    for index, seed in seeds:
        fit = minimize(objective, seed, jac=True, method="SLSQP",
            bounds=[(policy.site_floor, 1-policy.site_floor)]*surface.dimension,
            constraints={"type": "ineq", "fun": surface.constraints,
                         "jac": lambda _: surface.constraint_jacobian()},
            options={"maxiter": policy.maximum_iterations, "ftol": 1e-12})
        value, gradient = objective(fit.x)
        success = bool(fit.success and np.max(np.abs(gradient)) <= 1e-5)
        attempts.append({"sample_index": index, "success": success,
            "iterations": int(fit.nit), "message": str(fit.message),
            "distance_J_mol": value*energy_scale,
            "stationarity_RT": float(np.max(np.abs(gradient)))})
        if success:
            minima.append((fit.x.copy(), value*energy_scale))
    return minima, attempts
