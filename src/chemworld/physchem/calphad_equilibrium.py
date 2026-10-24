"""Finite-search TP equilibrium of neutral, fixed-site-ratio CEF phases.

All phases share one native database and atom-molar Gibbs reference. The
result is qualified against the declared finite grid and local starts; it is
not a proof of global stability. No inventory or thermal history is owned here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from math import fsum, isfinite
from typing import Any, cast

import numpy as np
from scipy.optimize import linprog

from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_equilibrium_search import (
    PhaseSurface,
    refine_assemblage,
    search_phase,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R


@dataclass(frozen=True)
class CalphadEquilibriumPolicy:
    grid_intervals: int = 8
    maximum_search_starts: int = 12
    maximum_column_rounds: int = 8
    maximum_iterations: int = 250
    site_floor: float = 1e-12
    amount_tolerance: float = 1e-9
    tangent_tolerance_J_mol: float = 1e-4
    root: RootPolicy = field(default_factory=lambda: RootPolicy(tolerance=1e-11))

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (
            self.grid_intervals, self.maximum_search_starts, self.maximum_column_rounds,
            self.maximum_iterations,
        )):
            raise ValueError("equilibrium search budgets must be positive integers")
        if (not isfinite(self.site_floor) or not 0 < self.site_floor < 1e-4
                or not isfinite(self.amount_tolerance) or not 0 < self.amount_tolerance < 1e-4
                or not isfinite(self.tangent_tolerance_J_mol)
                or self.tangent_tolerance_J_mol <= 0 or not isinstance(self.root, RootPolicy)):
            raise ValueError("invalid declared equilibrium tolerances")


@dataclass(frozen=True)
class CalphadEquilibriumSpec:
    models: tuple[CefPhaseModel, ...]
    policy: CalphadEquilibriumPolicy = field(default_factory=CalphadEquilibriumPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "models", tuple(self.models))
        if not self.models or any(not isinstance(m, CefPhaseModel) for m in self.models):
            raise ValueError("equilibrium requires explicit native CEF models")
        if len({m.phase_name for m in self.models}) != len(self.models):
            raise ValueError("declare a model once; multiple constitutions are solver outputs")
        if any(m.database != self.models[0].database for m in self.models):
            raise ValueError("all equilibrium phases require the same database/reference")
        for model in self.models:
            phase = model._phase
            hints = cast(Mapping[str, Any], phase.model_hints)
            species = cast(Mapping[str, Any], model.database.species)
            if (phase.options & {"Y", "G"} or hints.get("auxiliary_balances")
                    or hints.get("phase_family") in {"SUBQ", "SUBG"}
                    or any(species[s].charge != 0
                           for subl in model.constituents for s in subl)):
                raise ValueError("this equilibrium closure supports neutral fixed-ratio CEF only")
            if any(species[s].number_of_atoms <= 0
                   for subl in model.constituents for s in subl):
                raise ValueError("vacancy/electron constitutions require a separate closure")
        if not isinstance(self.policy, CalphadEquilibriumPolicy):
            raise ValueError("equilibrium requires its explicit search policy")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-calphad-equilibrium-v1",
                "models": [m.to_dict() for m in self.models], "policy": asdict(self.policy)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadEquilibriumSpec:
        from chemworld.data.scientific_inputs import _keys

        _keys(data, ("model", "models", "policy"))
        if data["model"] != "chemworld-calphad-equilibrium-v1":
            raise ValueError("unknown CALPHAD equilibrium declaration")
        policy = dict(data["policy"])
        policy["root"] = RootPolicy(**policy["root"])
        return cls(tuple(CefPhaseModel.from_dict(m) for m in data["models"]),
                   CalphadEquilibriumPolicy(**policy))


@dataclass(frozen=True)
class CalphadEquilibriumPhase:
    phase_name: str
    site_fractions: tuple[dict[str, float], ...]
    element_fractions: dict[str, float]
    amount_mol_atoms: float
    fraction: float
    gibbs_J_mol_atoms: float
    enthalpy_J_mol_atoms: float

    @property
    def element_amounts_mol(self) -> dict[str, float]:
        return {e: self.amount_mol_atoms*x for e, x in self.element_fractions.items()}


@dataclass(frozen=True)
class CalphadEquilibriumResult:
    success: bool
    phases: tuple[CalphadEquilibriumPhase, ...] = ()
    element_potentials_J_mol: dict[str, float] | None = None
    total_gibbs_J: float | None = None
    total_enthalpy_J: float | None = None
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def calphad_tp_equilibrium(
    specification: CalphadEquilibriumSpec, temperature_K: float, pressure_Pa: float,
    element_amounts_mol: Mapping[str, float], *,
    initial_phases: Sequence[CalphadEquilibriumPhase] | None = None,
) -> CalphadEquilibriumResult:
    """Redistribute actual element amounts among the allowed phase models.

    Exactly absent elements are removed from phase constitutions, never seeded
    with small invented stock. Boundary constitutions of active elements below
    the numerical site floor are outside this interior-KKT search scope.
    """
    if (not isfinite(temperature_K) or temperature_K <= 0 or not isfinite(pressure_Pa)
            or pressure_Pa <= 0):
        raise ValueError("equilibrium requires finite positive temperature and pressure")
    database = specification.models[0].database
    if (not element_amounts_mol
            or set(element_amounts_mol)-set(cast(Mapping[str, Any], database.elements))
            or set(element_amounts_mol) & {"VA", "/-"}
            or any(isinstance(n, bool) or not isfinite(n) or n < 0
                   for n in element_amounts_mol.values())):
        raise ValueError("equilibrium feed requires declared nonnegative finite element stock")
    elements = tuple(sorted(e for e, n in element_amounts_mol.items() if n > 0))
    total = fsum(element_amounts_mol.values())
    if not elements or not isfinite(total) or total <= 0:
        raise ValueError("equilibrium requires positive total atom inventory")
    target = np.array([element_amounts_mol[e]/total for e in elements])
    policy = specification.policy
    diagnostic: dict[str, Any] = {
        "scope": "finite_grid_and_multistart_local_tangent_search; no_global_certificate",
        "molar_basis": "nonvacant_atoms", "temperature_K": temperature_K,
        "pressure_Pa": pressure_Pa, "policy": asdict(policy), "active_elements": elements,
        "allowed_phases": [m.phase_name for m in specification.models],
        "inactive_phases": [], "rounds": [],
        "phase_amount_uniqueness": "not_globally_certified; selected_KKT_rank_checked",
    }

    def failure(reason: str) -> CalphadEquilibriumResult:
        return CalphadEquilibriumResult(False, failure_reason=reason, diagnostics=diagnostic)

    surfaces = []
    for model in specification.models:
        # Recompile only when the caller's model actually contains absent atoms.
        support = {e for subl in model.constituents for s in subl
                   for e, n in cast(Mapping[str, Any], database.species)[s].constituents if n > 0}
        if support-set(elements):
            try:
                model = CefPhaseModel(database, model.phase_name, elements, model.extrapolation,
                                      model.unspecified_endmembers, model.activate_suppressed)
            except ValueError as exc:
                diagnostic["inactive_phases"].append(
                    {"phase": model.phase_name, "reason": str(exc)})
                continue
        surfaces.append(PhaseSurface(model, elements, temperature_K,
                                     pressure_Pa, policy.site_floor))
    if not surfaces:
        return failure("no_phase_supports_actual_element_stock")
    try:
        samples = [surface.samples(policy.grid_intervals) for surface in surfaces]
        if initial_phases:
            warm: list[dict[str, Any]] = []
            diagnostic["warm_seed_attempts"] = warm
            for seed in initial_phases:
                if not isinstance(seed, CalphadEquilibriumPhase):
                    raise ValueError("CALPHAD warm seeds require original phase results")
                index = next((i for i, s in enumerate(surfaces)
                              if s.model.phase_name == seed.phase_name), None)
                if index is None:
                    warm.append({"phase": seed.phase_name, "accepted": False,
                                 "reason": "inactive_or_undeclared_phase"})
                    continue
                try:
                    sites = surfaces[index].model._fractions(seed.site_fractions)
                    q = np.array([sites[j][s] for j, members in
                                  enumerate(surfaces[index].model.constituents)
                                  for s in members[:-1]])
                    surfaces[index].values(q)
                except (ValueError, KeyError) as exc:
                    warm.append({"phase": seed.phase_name, "accepted": False, "reason": str(exc)})
                    continue
                if not any(np.max(np.abs(q-old), initial=0.) < 1e-12 for old in samples[index]):
                    samples[index].append(q)
                warm.append({"phase": seed.phase_name, "accepted": True})
        columns = [(i, q) for i, points in enumerate(samples) for q in points]
        values = [surfaces[i].values(q) for i, q in columns]
    except (ValueError, ArithmeticError) as exc:
        diagnostic["evaluation_error"] = str(exc)
        return failure("declared_grid_evaluation_failed")
    diagnostic["sample_counts"] = {s.model.phase_name: len(q)
                                    for s, q in zip(surfaces, samples, strict=True)}
    scale = R*temperature_K
    for round_index in range(policy.maximum_column_rounds):
        matrix = np.array(values)
        lp = linprog(matrix[:, 0]/scale, A_eq=matrix[:, 1:].T, b_eq=target,
                     bounds=(0, None), method="highs")
        record: dict[str, Any] = {"round": round_index, "columns": len(columns),
                                  "lp_success": bool(lp.success), "lp_message": str(lp.message)}
        diagnostic["rounds"].append(record)
        if not lp.success:
            return failure("sampled_assemblage_infeasible")
        selected = np.flatnonzero(lp.x > policy.amount_tolerance)
        active = [columns[k] for k in selected]
        try:
            points, weights, mu, refinement = refine_assemblage(
                surfaces, active, lp.x[selected], target, scale, policy)
        except (ValueError, ArithmeticError) as exc:
            record["evaluation_error"] = str(exc)
            return failure("assemblage_evaluation_failed")
        record["refinement"] = refinement
        if points is None:
            return failure(refinement["failure_reason"])
        search, negative = [], []
        for i, surface in enumerate(surfaces):
            try:
                minima, attempts = search_phase(surface, samples[i], mu, scale, policy)
            except (ValueError, ArithmeticError) as exc:
                search.append({"phase": surface.model.phase_name, "evaluation_error": str(exc)})
                record["search"] = search
                return failure("tangent_search_evaluation_failed")
            search.append({"phase": surface.model.phase_name, "attempts": attempts,
                           "successful_minima": len(minima),
                           "minimum_distance_J_mol": min((d for _, d in minima), default=None)})
            if not minima or any(not a["success"] for a in attempts):
                record["search"] = search
                return failure("phase_tangent_search_exhausted")
            for q, distance in minima:
                if distance < -policy.tangent_tolerance_J_mol:
                    negative.append((i, q))
        record["search"] = search
        record["negative_minima"] = len(negative)
        if negative:
            additions = [(i, q) for i, q in negative if not any(i == j and
                np.max(np.abs(q-other), initial=0.) < 1e-8 for j, other in columns)]
            if not additions:
                return failure("negative_tangent_minimum_without_new_column")
            columns.extend(additions)
            values.extend(surfaces[i].values(q) for i, q in additions)
            continue
        # Local convexity and physical balances remain independent acceptance checks.
        curvature = []
        compositions = []
        for i, q in points:
            v, _j, h = surfaces[i].response(q)
            compositions.append(v[1:])
            if len(q):
                curvature.append(float(np.min(np.linalg.eigvalsh(
                    h[0]/scale-np.einsum("i,ijk->jk", mu/scale, h[1:])))))
        record["minimum_local_curvature_RT"] = min(curvature, default=None)
        error = float(np.max(np.abs(np.asarray(weights)@np.array(compositions)-target)))
        record["element_balance_error"] = error
        if (error > 1e-10 or any(w <= 0 for w in weights)
                or abs(sum(weights)-1) > 1e-10):
            return failure("physical_assemblage_balance_failed")
        if curvature and min(curvature) < -1e-7:
            return failure("selected_phase_is_tangent_saddle")
        phases = []
        for (i, q), fraction in zip(points, weights, strict=True):
            sites = surfaces[i].sites(q)
            properties = surfaces[i].model.evaluate(sites, temperature_K, pressure_Pa,
                                                    pressure_derivative=False)
            phases.append(CalphadEquilibriumPhase(surfaces[i].model.phase_name,
                tuple({s: float(y) for s, y in row.items()} for row in sites),
                {e: float(properties.element_fractions.get(e, 0.)) for e in elements},
                float(fraction*total), float(fraction), properties.gibbs_J_mol,
                properties.enthalpy_J_mol))
        phases.sort(key=lambda p: (p.phase_name, tuple(p.element_fractions.values()),
                                  tuple(y for row in p.site_fractions for y in row.values())))
        return CalphadEquilibriumResult(True, tuple(phases),
            dict(zip(elements, map(float, mu), strict=True)),
            fsum(p.amount_mol_atoms*p.gibbs_J_mol_atoms for p in phases),
            fsum(p.amount_mol_atoms*p.enthalpy_J_mol_atoms for p in phases),
            diagnostics=diagnostic)
    return failure("column_search_budget_exhausted")
