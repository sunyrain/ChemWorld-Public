"""Finite-assemblage PH KKT, including invariant latent-heat plateaus.

All energies and amounts use the original CEF nonvacant-atom normalization.
The temperature and phase amounts are simultaneous unknowns; a TP phase amount
singularity at an invariant transition is resolved by the extensive H row.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import fsum, isfinite
from typing import Any, cast

import numpy as np

from chemworld.physchem.calphad import CefPhaseModel
from chemworld.physchem.calphad_equilibrium import (
    CalphadEquilibriumPhase,
    CalphadEquilibriumSpec,
    calphad_tp_equilibrium,
)
from chemworld.physchem.calphad_equilibrium_search import PhaseSurface, search_phase
from chemworld.physchem.calphad_response import analytic_response, branch_diagnostics
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)
from chemworld.physchem.tdb_expressions import TDB_GAS_CONSTANT_J_MOL_K as R


@dataclass(frozen=True)
class CalphadCaloricSpec:
    equilibrium: CalphadEquilibriumSpec
    pressure_Pa: float
    temperature_range_K: tuple[float, float]
    search_temperatures_K: tuple[float, ...]
    maximum_phase_instances: int = 3
    root_policy: RootPolicy = field(default_factory=lambda: RootPolicy(1e-11, 120))

    def __post_init__(self) -> None:
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        object.__setattr__(self, "search_temperatures_K", tuple(self.search_temperatures_K))
        if (
            len(self.temperature_range_K) != 2
            or not 0 < self.temperature_range_K[0] < self.temperature_range_K[1]
            or not self.search_temperatures_K
            or any(
                not isfinite(t)
                or not self.temperature_range_K[0] <= t <= self.temperature_range_K[1]
                for t in self.search_temperatures_K
            )
            or not isfinite(self.pressure_Pa)
            or self.pressure_Pa <= 0
            or type(self.maximum_phase_instances) is not int
            or not 1 <= self.maximum_phase_instances <= 8
        ):
            raise ValueError("CALPHAD PH needs finite pressure/T/search bounds and phase budget")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    e
                    for m in self.equilibrium.models
                    for sub in m.constituents
                    for s in sub
                    for e, n in cast(Mapping[str, Any], m.database.species)[s].constituents
                    if n > 0
                }
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "calphad_caloric_equilibrium_v1",
            "equilibrium": self.equilibrium.to_dict(),
            "pressure_Pa": self.pressure_Pa,
            "temperature_range_K": list(self.temperature_range_K),
            "search_temperatures_K": list(self.search_temperatures_K),
            "maximum_phase_instances": self.maximum_phase_instances,
            "root_policy": asdict(self.root_policy),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CalphadCaloricSpec:
        from chemworld.data.scientific_inputs import _keys

        _keys(
            data,
            (
                "model_type",
                "equilibrium",
                "pressure_Pa",
                "temperature_range_K",
                "search_temperatures_K",
                "maximum_phase_instances",
                "root_policy",
            ),
        )
        if data["model_type"] != "calphad_caloric_equilibrium_v1":
            raise ValueError("unknown CALPHAD caloric model")
        return cls(
            CalphadEquilibriumSpec.from_dict(data["equilibrium"]),
            data["pressure_Pa"],
            tuple(data["temperature_range_K"]),
            tuple(data["search_temperatures_K"]),
            data["maximum_phase_instances"],
            RootPolicy(**data["root_policy"]),
        )


@dataclass(frozen=True)
class CalphadCaloricResult:
    success: bool
    temperature_K: float | None = None
    phases: tuple[CalphadEquilibriumPhase, ...] = ()
    enthalpy_J: float | None = None
    entropy_J_K: float | None = None
    gibbs_J: float | None = None
    element_potentials_J_mol: dict[str, float] | None = None
    element_residual_mol: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CalphadCaloricProblem:
    """Disposable numerical preparation; original phase inventories stay outside."""

    def __init__(self, spec: CalphadCaloricSpec, amounts_mol: Mapping[str, float]) -> None:
        self.spec = spec
        if (
            set(amounts_mol) != set(spec.component_ids)
            or any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in amounts_mol.values())
            or not isfinite(fsum(amounts_mol.values()))
            or fsum(amounts_mol.values()) <= 0
        ):
            raise ValueError("CALPHAD PH requires matching finite nonnegative atom inventory")
        self.feed = dict(amounts_mol)
        self.elements = tuple(k for k in spec.component_ids if amounts_mol[k] > 0)
        self.total = fsum(amounts_mol.values())
        self.target = np.array([amounts_mol[e] / self.total for e in self.elements])
        self.models: list[CefPhaseModel] = []
        self.preparation: list[dict[str, Any]] = []
        for model in spec.equilibrium.models:
            try:
                self.models.append(
                    CefPhaseModel(
                        model.database,
                        model.phase_name,
                        self.elements,
                        model.extrapolation,
                        model.unspecified_endmembers,
                        model.activate_suppressed,
                    )
                )
            except ValueError as exc:
                self.preparation.append({"inactive_phase": model.phase_name, "reason": str(exc)})
        if not self.models:
            raise ValueError("no CALPHAD phase supports actual elements")
        self.pool: list[tuple[int, np.ndarray, float, float]] = []
        for temperature in spec.search_temperatures_K:
            tp = calphad_tp_equilibrium(
                CalphadEquilibriumSpec(tuple(self.models), spec.equilibrium.policy),
                temperature,
                spec.pressure_Pa,
                amounts_mol,
            )
            self.preparation.append({"temperature_K": temperature, "result": tp.to_dict()})
            if not tp.success:
                continue
            for phase in tp.phases:
                index = next(
                    i for i, m in enumerate(self.models) if m.phase_name == phase.phase_name
                )
                q = np.array(
                    [
                        phase.site_fractions[j][s]
                        for j, members in enumerate(self.models[index].constituents)
                        for s in members[:-1]
                    ]
                )
                if not any(
                    i == index and np.max(np.abs(q - p), initial=0.0) < 1e-5
                    for i, p, _, _ in self.pool
                ):
                    self.pool.append((index, q, phase.fraction, temperature))
        # Every allowed model is a possible entering phase, even if absent from
        # the finite TP seeds. Its declared interior sample remains only a seed.
        for i, model in enumerate(self.models):
            if not any(j == i for j, *_ in self.pool):
                q = np.array([1 / len(row) for row in model.constituents for _ in row[:-1]])
                self.pool.append((i, q, 1.0, spec.search_temperatures_K[0]))
        if len(self.pool) > 16:
            raise ValueError("declared CALPHAD caloric seed pool exceeds 16 finite candidates")
        self._warm: tuple[tuple[int, ...], np.ndarray] | None = None

    def solve(
        self,
        enthalpy_J: float,
        *,
        support_heat_capacity_J_K: float = 0.0,
        support_reference_temperature_K: float = 600.0,
    ) -> CalphadCaloricResult:
        if (
            not isfinite(enthalpy_J)
            or isinstance(enthalpy_J, bool)
            or not isfinite(support_heat_capacity_J_K)
            or support_heat_capacity_J_K < 0
            or not isfinite(support_reference_temperature_K)
            or support_reference_temperature_K <= 0
        ):
            raise ValueError("CALPHAD PH needs finite H and explicit nonmaterial support")
        spec, total, target = self.spec, self.total, self.target
        ne = len(self.elements)
        scale = R * fsum(spec.search_temperatures_K) / len(spec.search_temperatures_K)
        records: list[dict[str, Any]] = []
        selections = [
            c
            for count in range(1, min(spec.maximum_phase_instances, ne + 1, len(self.pool)) + 1)
            for c in combinations(range(len(self.pool)), count)
        ]
        if self._warm is not None:
            selections.remove(self._warm[0])
            selections.insert(0, self._warm[0])
        for selected in selections:
            count = len(selected)
            active = [self.pool[i] for i in selected]
            slices, offset = [], count
            for _, q, _, _ in active:
                slices.append(slice(offset, offset + len(q)))
                offset += len(q)
            temperature0 = fsum(v[3] for v in active) / count
            initial = np.r_[
                [v[2] for v in active], *(v[1] for v in active), np.zeros(ne), temperature0
            ]
            initial[:count] /= initial[:count].sum()
            if self._warm is not None:
                initial[-1] = self._warm[1][-1]
                if selected == self._warm[0]:
                    initial = self._warm[1].copy()

            def parts(z: Any, active: Any = active, slices: Any = slices) -> Any:
                temperature = float(z[-1])
                out = []
                for (index, *_), sl in zip(active, slices, strict=True):
                    model = self.models[index]
                    surface = PhaseSurface(
                        model,
                        self.elements,
                        temperature,
                        spec.pressure_Pa,
                        spec.equilibrium.policy.site_floor,
                    )
                    q = z[sl]
                    surface.values(q)

                    def evaluate(coords: Any, surface: Any = surface, model: Any = model) -> Any:
                        sites = surface.sites(coords[1:])
                        composition = model._composition(sites)[1]
                        return (
                            sum(model._energies(sites, coords[0], spec.pressure_Pa).values()),
                            *(composition.get(e, 0.0) for e in self.elements),
                        )

                    try:
                        v, j, second = analytic_response(evaluate, np.r_[temperature, q])
                        if -temperature * second[0, 0, 0] < -1e-10:
                            raise ValueError(
                                "CEF phase has negative fixed-constitution heat capacity"
                            )
                    except (ValueError, ArithmeticError) as exc:
                        raise ResidualDomainError(str(exc)) from exc
                    out.append(
                        (v, j[:, 1:], v[0] - temperature * j[0, 0], -j[0, 0], surface.sites(q))
                    )
                return out

            def equations(
                z: Any,
                parts: Any = parts,
                offset: int = offset,
                slices: Any = slices,
                temperature0: float = temperature0,
            ) -> Any:
                values = parts(z)
                mu = z[offset : offset + ne]
                residual = np.zeros(len(z))
                h = support_heat_capacity_J_K * (z[-1] - support_reference_temperature_K)
                for k, ((v, j, hm, _, _), sl) in enumerate(zip(values, slices, strict=True)):
                    residual[k] = v[0] / scale - mu @ v[1:]
                    residual[sl] = j[0] / scale - mu @ j[1:]
                    residual[offset : offset + ne] += z[k] * v[1:]
                    h += total * z[k] * hm
                residual[offset : offset + ne] -= target
                residual[-1] = (h - enthalpy_J) / max(
                    total * scale, support_heat_capacity_J_K * temperature0
                )
                return residual

            lower = np.r_[
                np.zeros(count),
                np.full(offset - count, spec.equilibrium.policy.site_floor),
                np.full(ne, -np.inf),
                spec.temperature_range_K[0],
            ]
            upper = np.r_[
                np.ones(count),
                np.full(offset - count, 1 - spec.equilibrium.policy.site_floor),
                np.full(ne, np.inf),
                spec.temperature_range_K[1],
            ]
            scales = np.r_[np.ones(offset + ne), 100.0]
            record: dict[str, Any] = {
                "seed_indices": selected,
                "phases": [self.models[i].phase_name for i, *_ in active],
            }
            records.append(record)
            try:
                initial_parts = parts(initial)
                matrix = np.vstack(
                    [row for v, j, *_ in initial_parts for row in (v[None, 1:], j[1:].T)]
                )
                rhs = np.concatenate([np.r_[v[0], j[0]] / scale for v, j, *_ in initial_parts])
                initial[offset : offset + ne] = np.linalg.lstsq(matrix, rhs, rcond=1e-11)[0]

                def jacobian(
                    z: Any,
                    equations: Any = equations,
                    scales: Any = scales,
                    lower: Any = lower,
                    upper: Any = upper,
                ) -> Any:
                    return finite_jacobian(equations, z, scales, lower=lower, upper=upper)

                root = solve_root(
                    equations,
                    initial,
                    jacobian=jacobian,
                    variables=tuple(
                        ResidualVariable(
                            f"caloric:{i}",
                            "scaled",
                            scale=float(s),
                            lower=None if not isfinite(lo) else float(lo),
                            upper=None if not isfinite(hi) else float(hi),
                        )
                        for i, (s, lo, hi) in enumerate(zip(scales, lower, upper, strict=True))
                    ),
                    residual_scales=np.ones(len(initial)),
                    policy=spec.root_policy,
                )
                record["root"] = {
                    "success": root.success,
                    "failure_reason": root.failure_reason,
                    **root.diagnostics,
                }
                if not root.success:
                    continue
                z, values = root.value, parts(root.value)
                rank = int(np.linalg.matrix_rank(jacobian(z) * scales, tol=1e-10))
                record.update(rank=rank, unknowns=len(z))
                if rank != len(z):
                    record["failure_reason"] = "underdetermined_PH_assemblage"
                    continue
                temperature = float(z[-1])
                mu = z[offset : offset + ne] * scale
                branch_notes = [
                    message
                    for (index, *_), value in zip(active, values, strict=True)
                    for message in branch_diagnostics(
                        self.models[index],
                        value[4],
                        temperature,
                        spec.pressure_Pa,
                        include_sites=False,
                    )
                ]
                if branch_notes:
                    record["failure_reason"] = (
                        "caloric_derivative_requires_explicit_one_sided_branch"
                    )
                    record["branch_diagnostics"] = branch_notes
                    continue
                # The solver's scaled residual is not a physical stock ledger.
                # Never drop a converged small phase while retaining its H/S:
                # retry the already-declared reduced assemblages instead.
                if any(z[k] <= spec.equilibrium.policy.amount_tolerance for k in range(count)):
                    record["failure_reason"] = "vanishing_phase_requires_reduced_assemblage"
                    continue
                balance = np.asarray(z[:count]) @ np.array([v[1:] for v, *_ in values])
                error = float(np.max(np.abs(balance - target)))
                fraction_error = abs(float(np.sum(z[:count])) - 1.0)
                record.update(element_balance_error=error, phase_fraction_error=fraction_error)
                if error > 1e-10 or fraction_error > 1e-10:
                    record["failure_reason"] = "physical_assemblage_balance_failed"
                    continue
                curvature = []
                for (index, *_), sl in zip(active, slices, strict=True):
                    q = z[sl]
                    if len(q):
                        surface = PhaseSurface(
                            self.models[index],
                            self.elements,
                            temperature,
                            spec.pressure_Pa,
                            spec.equilibrium.policy.site_floor,
                        )
                        _, _, second = surface.response(q)
                        tangent_hessian = (second[0] - np.einsum("i,ijk->jk", mu, second[1:])) / (
                            R * temperature
                        )
                        curvature.append(float(np.min(np.linalg.eigvalsh(tangent_hessian))))
                record["minimum_local_curvature_RT"] = min(curvature, default=None)
                if curvature and min(curvature) < -1e-7:
                    record["failure_reason"] = "selected_phase_is_tangent_saddle"
                    continue
                searches, stable = [], True
                # Stability is necessary even for transient caloric trials:
                # it decides whether to switch the active phase assemblage.
                for model in self.models:
                    surface = PhaseSurface(
                        model,
                        self.elements,
                        temperature,
                        spec.pressure_Pa,
                        spec.equilibrium.policy.site_floor,
                    )
                    minima, attempts = search_phase(
                        surface,
                        surface.samples(spec.equilibrium.policy.grid_intervals),
                        mu,
                        R * temperature,
                        spec.equilibrium.policy,
                    )
                    minimum = min((v for _, v in minima), default=None)
                    complete = bool(minima) and all(a["success"] for a in attempts)
                    searches.append(
                        {
                            "phase": model.phase_name,
                            "complete": complete,
                            "minimum_distance_J_mol": minimum,
                            "starts": attempts,
                        }
                    )
                    stable &= (
                        complete
                        and minimum is not None
                        and minimum >= -spec.equilibrium.policy.tangent_tolerance_J_mol
                    )
                record["post_tangent"] = searches
                if not stable:
                    record["failure_reason"] = "negative_or_unresolved_post_tangent"
                    continue
                phases = []
                h = support_heat_capacity_J_K * (temperature - support_reference_temperature_K)
                entropy = support_heat_capacity_J_K * np.log(
                    temperature / support_reference_temperature_K
                )
                for k, ((index, *_), (v, _, hm, sm, sites)) in enumerate(
                    zip(active, values, strict=True)
                ):
                    h += total * z[k] * hm
                    entropy += total * z[k] * sm
                    phases.append(
                        CalphadEquilibriumPhase(
                            self.models[index].phase_name,
                            tuple({k: float(v) for k, v in row.items()} for row in sites),
                            dict(zip(self.elements, map(float, v[1:]), strict=True)),
                            total * float(z[k]),
                            float(z[k]),
                            float(v[0]),
                            float(hm),
                        )
                    )
                phases.sort(
                    key=lambda p: (
                        p.phase_name,
                        tuple(p.element_fractions.values()),
                        tuple(tuple(s.values()) for s in p.site_fractions),
                    )
                )
                self._warm = selected, z.copy()
                return CalphadCaloricResult(
                    True,
                    temperature,
                    tuple(phases),
                    float(h),
                    float(entropy),
                    fsum(p.amount_mol_atoms * p.gibbs_J_mol_atoms for p in phases)
                    + support_heat_capacity_J_K
                    * (
                        (temperature - support_reference_temperature_K)
                        - temperature * np.log(temperature / support_reference_temperature_K)
                    ),
                    dict(zip(self.elements, map(float, mu), strict=True)),
                    max(
                        abs(self.feed[e] - fsum(p.element_amounts_mol.get(e, 0.0) for p in phases))
                        for e in self.feed
                    ),
                    {
                        "preparation": self.preparation,
                        "attempts": records,
                        "stability_checked": True,
                        "scope": "finite_temperature_seeds_and_assemblages; no_global_certificate",
                        "caloric_equations": "joint_T_phase_amounts_sites_element_potentials_H",
                    },
                )
            except (ValueError, ArithmeticError) as exc:
                record["failure_reason"] = str(exc)
        return CalphadCaloricResult(
            False,
            diagnostics={"preparation": self.preparation, "attempts": records},
            failure_reason="finite_PH_assemblages_unresolved",
        )


def solve_calphad_ph(
    spec: CalphadCaloricSpec,
    amounts_mol: Mapping[str, float],
    enthalpy_J: float,
    *,
    support_heat_capacity_J_K: float = 0.0,
    support_reference_temperature_K: float = 600.0,
) -> CalphadCaloricResult:
    return CalphadCaloricProblem(spec, amounts_mol).solve(
        enthalpy_J,
        support_heat_capacity_J_K=support_heat_capacity_J_K,
        support_reference_temperature_K=support_reference_temperature_K,
    )
