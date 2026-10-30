"""Declared finite liquid/pure-mineral TP closure on the common residual solver.

For a dissolution reaction with mineral coefficient -1, SI = DeltaG/(RT).
Absent stable minerals have SI <= 0; a present mineral has SI = 0. Phase-set
enumeration does not prove global stability of a nonideal liquid.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import isfinite
from typing import Any, cast

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from chemworld.physchem.aqueous_database import AqueousDatabase
from chemworld.physchem.aqueous_phase_properties import (
    AqueousPhaseProperties,
    aqueous_phase_properties,
)
from chemworld.physchem.aqueous_reaction_equilibrium import aqueous_chemical_potential_jacobian
from chemworld.physchem.elements import element_matrix
from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    solve_steady_residual,
)


@dataclass(frozen=True)
class AqueousMineralEquilibriumSpec:
    database: AqueousDatabase
    reaction_ids: tuple[str, ...]
    root_policy: RootPolicy = field(
        default_factory=lambda: RootPolicy(tolerance=1e-11, maximum_iterations=150)
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "reaction_ids", tuple(self.reaction_ids))
        db = self.database
        if any(s.phase == "gas" for s in db.species) or not 1 <= len(self.mineral_ids) <= 4:
            raise ValueError("declare one liquid and one to four pure minerals, without gas")
        reactions = {r.reaction_id: r for r in db.reactions}
        if len(set(self.reaction_ids)) != len(self.reaction_ids) or set(self.reaction_ids) - set(
            reactions
        ):
            raise ValueError("selected reaction identities must be unique and declared")
        assigned = set()
        for key in self.reaction_ids:
            reaction = reactions[key]
            solids = set(reaction.stoichiometry) & set(self.mineral_ids)
            if solids:
                mineral = next(iter(solids))
                if len(solids) != 1 or reaction.stoichiometry[mineral] != -1:
                    raise ValueError("mineral reactions must dissolve one pure mineral with nu=-1")
                if mineral in assigned:
                    raise ValueError("declare one selected dissolution reaction per mineral")
                assigned.add(mineral)
        for species in db.species:
            if species.standard_state.pressure_model != "constant-volume":
                raise ValueError("TP inventory requires declared constant-volume standard states")
            if species.phase == "mineral" and (
                species.standard_state.molar_volume_m3_mol is None
                or species.standard_state.molar_volume_m3_mol <= 0
            ):
                raise ValueError("pure mineral molar volume must be positive")

    @property
    def mineral_ids(self) -> tuple[str, ...]:
        return tuple(s.species_id for s in self.database.species if s.phase == "mineral")

    @property
    def dissolution_reactions(self) -> dict[str, str]:
        return {
            mineral: reaction.reaction_id
            for reaction in self.database.reactions
            if reaction.reaction_id in self.reaction_ids
            for mineral in self.mineral_ids
            if mineral in reaction.stoichiometry
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "finite-aqueous-pure-mineral-TP-v1",
            "database": self.database.to_dict(),
            "reaction_ids": self.reaction_ids,
            "root_policy": asdict(self.root_policy),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> AqueousMineralEquilibriumSpec:
        if raw.get("model") != "finite-aqueous-pure-mineral-TP-v1":
            raise ValueError("unknown mineral equilibrium declaration")
        return cls(
            AqueousDatabase.from_dict(raw["database"]),
            tuple(raw["reaction_ids"]),
            RootPolicy(**raw["root_policy"]),
        )


@dataclass(frozen=True)
class AqueousMineralEquilibriumResult:
    success: bool
    amounts_mol: dict[str, float] | None
    saturation_indices: dict[str, float | None]
    properties: AqueousPhaseProperties | None
    diagnostics: dict[str, Any]
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "amounts_mol": self.amounts_mol,
            "saturation_indices": self.saturation_indices,
            "properties": self.properties.to_dict() if self.properties else None,
            "diagnostics": self.diagnostics,
            "failure_reason": self.failure_reason,
        }


def aqueous_mineral_properties(
    spec: AqueousMineralEquilibriumSpec,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> AqueousPhaseProperties:
    """Extensive H/V/G/U/S from the original liquid G and pure solid standard states."""
    liquid = {s.species_id for s in spec.database.species if s.phase != "mineral"}
    if set(amounts_mol) - {s.species_id for s in spec.database.species} or any(
        isinstance(n, bool) or not isfinite(n) or n < 0 for n in amounts_mol.values()
    ):
        raise ValueError("properties require declared finite nonnegative inventories")
    result = aqueous_phase_properties(
        spec.database,
        {k: n for k, n in amounts_mol.items() if k in liquid},
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
    )
    standards = spec.database.standard_states(temperature_K, pressure_Pa)
    h, v, g = result.enthalpy_J, result.volume_m3, result.gibbs_energy_J
    for species in spec.database.species:
        if species.phase == "mineral":
            n = amounts_mol.get(species.species_id, 0.0)
            h += n * standards[species.species_id].enthalpy_J_mol
            g += n * standards[species.species_id].gibbs_J_mol
            v += n * cast(float, species.standard_state.molar_volume_m3_mol)
    return AqueousPhaseProperties(
        temperature_K, pressure_Pa, v, h, h - pressure_Pa * v, g, (h - g) / temperature_K
    )


def _independent_rows(matrix: np.ndarray) -> list[int]:
    selected: list[int] = []
    rank = 0
    for i, row in enumerate(matrix):
        if np.linalg.norm(row) == 0:
            continue
        candidate = matrix[[*selected, i]]
        candidate = candidate / np.linalg.norm(candidate, axis=1)[:, None]
        size = int(np.linalg.matrix_rank(candidate, tol=1e-11))
        if size > rank:
            selected.append(i)
            rank = size
    return selected


def solve_aqueous_mineral_equilibrium(
    spec: AqueousMineralEquilibriumSpec,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
) -> AqueousMineralEquilibriumResult:
    """Return a candidate only; failed phase sets and root diagnostics stay readable."""
    db = spec.database
    ids = tuple(s.species_id for s in db.species)
    if set(amounts_mol) - set(ids) or any(
        isinstance(n, bool) or not isfinite(n) or n < 0 for n in amounts_mol.values()
    ):
        raise ValueError("mineral equilibrium requires declared finite nonnegative amounts")
    n0 = np.array([amounts_mol.get(k, 0.0) for k in ids], dtype=float)
    feed = dict(zip(ids, map(float, n0), strict=True))
    initial_properties = aqueous_mineral_properties(
        spec, feed, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    charges = np.array([s.charge for s in db.species], dtype=float)
    if abs(charges @ n0) > 1e-10 * max(float(abs(charges) @ n0), 1e-12):
        raise ValueError("feed must be electroneutral; counterion repair is not allowed")
    atom_rows, labels = element_matrix([dict(s.composition) for s in db.species])
    atoms = np.asarray(atom_rows).T
    totals = atoms @ n0
    reactions = {r.reaction_id: r for r in db.reactions}
    stoich = np.array(
        [[reactions[r].stoichiometry.get(k, 0.0) for r in spec.reaction_ids] for k in ids]
    ).reshape((len(ids), len(spec.reaction_ids)))
    null = null_space(stoich.T, rcond=1e-12).T
    rows = np.array([charges, *sorted(atoms, key=lambda row: float(abs(row) @ n0)), *null])
    conserved = rows[_independent_rows(rows)]
    conserved /= np.linalg.norm(conserved, axis=1)[:, None]
    targets = conserved @ n0
    absent_element = np.any((atoms > 0) & (totals[:, None] == 0), axis=0)
    structural_zero = absent_element.copy()
    structural_zero |= np.all(stoich == 0, axis=1) & (n0 == 0)
    # An absent conserved element blocks both directions of a balanced reaction.
    # Its affinity is undefined at that exact face, not a failed positive-domain
    # equilibrium. Keep the undefined value and its explicit conservation proof.
    blocked_reactions = [
        key for key in spec.reaction_ids
        if any(absent_element[ids.index(k)] for k in reactions[key].stoichiometry)
    ]
    upper = np.array(
        [
            min(
                (totals[j] / atoms[j, i] for j in range(len(labels)) if atoms[j, i] > 0),
                default=0.0,
            )
            for i in range(len(ids))
        ]
    )
    mineral_indices = {k: ids.index(k) for k in spec.mineral_ids}
    attempts: list[dict[str, Any]] = []
    accepted: list[tuple[dict[str, float], dict[str, float | None], AqueousPhaseProperties]] = []
    for size in range(len(spec.mineral_ids) + 1):
        for present in combinations(spec.mineral_ids, size):
            attempt: dict[str, Any] = {"present_minerals": list(present), "success": False}
            attempts.append(attempt)
            active = ~structural_zero.copy()
            for mineral, i in mineral_indices.items():
                active[i] = mineral in present and not structural_zero[i]
            if any(structural_zero[mineral_indices[k]] for k in present):
                attempt["failure"] = "structurally_absent_mineral"
                continue
            indices = np.flatnonzero(active)
            names = tuple(ids[i] for i in indices)
            scale = upper[indices]
            matrix = conserved[:, indices]
            row_scale = np.maximum(abs(conserved) @ n0, np.max(abs(matrix) * scale, axis=1))
            row_scale = np.maximum(row_scale, 1e-30)
            width = len(indices)
            # This LP chooses a numerical interior without changing original targets.
            feasible = linprog(
                np.r_[np.zeros(width), -1.0],
                A_ub=np.column_stack((-np.eye(width), np.ones(width))),
                b_ub=np.zeros(width),
                A_eq=np.column_stack((matrix * scale / row_scale[:, None], np.zeros(len(matrix)))),
                b_eq=targets / row_scale,
                bounds=[(0.0, 1.0)] * (width + 1),
                method="highs",
            )
            attempt["initialization"] = {
                "status": int(feasible.status),
                "common_fraction": float(feasible.x[-1]) if feasible.success else None,
            }
            if not feasible.success or feasible.x[-1] <= 1e-14:
                attempt["failure"] = "no_positive_feasible_phase_set"
                attempt["unresolved"] = feasible.status != 2
                continue
            selected = _independent_rows(matrix)
            constraint, target = matrix[selected], targets[selected]
            baseline = np.maximum(abs(conserved[selected]) @ n0, 1e-30)
            tangent = null_space(constraint, rcond=1e-12)
            liquid_indices = [i for i, k in enumerate(names) if k not in mineral_indices]
            liquid_ids = tuple(names[i] for i in liquid_indices)

            def decode(
                y: np.ndarray, names: tuple[str, ...] = names, scale: np.ndarray = scale
            ) -> dict[str, float]:
                return {
                    **dict.fromkeys(ids, 0.0),
                    **dict(zip(names, map(float, scale * np.exp(y)), strict=True)),
                }

            def hessian(
                amounts: Mapping[str, float],
                liquid_ids: tuple[str, ...] = liquid_ids,
                liquid_indices: list[int] = liquid_indices,
                width: int = width,
            ) -> np.ndarray:
                full = np.zeros((width, width))
                full[np.ix_(liquid_indices, liquid_indices)] = aqueous_chemical_potential_jacobian(
                    db,
                    amounts,
                    temperature_K=temperature_K,
                    pressure_Pa=pressure_Pa,
                    species_ids=liquid_ids,
                )
                return full

            def residual(
                t: float,
                y: np.ndarray,
                rate: np.ndarray,
                p: np.ndarray,
                decode: Any = decode,
                names: tuple[str, ...] = names,
                tangent: np.ndarray = tangent,
                constraint: np.ndarray = constraint,
                target: np.ndarray = target,
                baseline: np.ndarray = baseline,
            ) -> np.ndarray:
                try:
                    amounts = decode(y)
                    report = db.chemical_potentials(
                        amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
                    )
                except ValueError as exc:
                    raise ResidualDomainError(str(exc)) from exc
                n = np.array([amounts[k] for k in names])
                mu = np.array([report.chemical_potentials_J_mol[k] for k in names]) / (
                    R * temperature_K
                )
                denominator = baseline + abs(constraint) @ n
                return np.r_[tangent.T @ mu, (constraint @ n - target) / denominator]

            def jacobian(
                t: float,
                y: np.ndarray,
                rate: np.ndarray,
                p: np.ndarray,
                decode: Any = decode,
                names: tuple[str, ...] = names,
                hessian: Any = hessian,
                tangent: np.ndarray = tangent,
                constraint: np.ndarray = constraint,
                target: np.ndarray = target,
                baseline: np.ndarray = baseline,
            ) -> np.ndarray:
                amounts = decode(y)
                n = np.array([amounts[k] for k in names])
                denominator = baseline + abs(constraint) @ n
                balance = constraint / denominator[:, None] - (
                    (constraint @ n - target) / denominator**2
                )[:, None] * abs(constraint)
                try:
                    return np.vstack((tangent.T @ hessian(amounts), balance)) * n
                except ValueError as exc:
                    raise ResidualDomainError(str(exc)) from exc

            def rate_jacobian(
                t: float, y: np.ndarray, rate: np.ndarray, p: np.ndarray, width: int = width,
            ) -> np.ndarray:
                return np.zeros((width, width))

            system = ResidualSystem(
                db.database_id + ":mineral-TP",
                tuple(
                    ResidualVariable("log_n:" + k, "1", role="algebraic", lower=-700.0, upper=0.0)
                    for k in names
                ),
                tuple(ResidualEquation("affinity:" + str(i), "1") for i in range(tangent.shape[1]))
                + tuple(
                    ResidualEquation("invariant:" + str(i), "1") for i in range(len(constraint))
                ),
                residual,
                ResidualDerivatives(
                    "analytic",
                    state=jacobian,
                    rate=rate_jacobian,
                ),
            )
            seed_fraction = feasible.x[:-1].copy()
            # HiGHS can return 1 + roundoff for a variable fixed at its upper
            # bound. Correct only floating-point scale excess, never an actually
            # infeasible LP proposal, before taking the bounded logarithm.
            roundoff = 64 * np.finfo(float).eps
            overshoot = (seed_fraction > 1) & (seed_fraction <= 1 + roundoff)
            if np.any(overshoot):
                attempt["initialization"]["roundoff_upper_bound_corrections"] = int(
                    np.count_nonzero(overshoot)
                )
                seed_fraction[overshoot] = 1.0
            initial = np.log(seed_fraction)
            if np.all(seed_fraction > 0) and np.all(seed_fraction <= 1):
                try:
                    db.chemical_potentials(
                        decode(initial), temperature_K=temperature_K, pressure_Pa=pressure_Pa
                    )
                except ValueError as exc:
                    # A positive LP interior can dissolve an artificial amount
                    # of solid, outside the declared liquid activity domain.
                    # The actual feed is a fallback only on precisely this
                    # same phase support; no trace material or balance repair.
                    if not np.any(n0[~active]) and np.all(n0[indices] > 0):
                        feed_fraction = n0[indices] / scale
                        feed_roundoff = (feed_fraction > 1) & (feed_fraction <= 1 + roundoff)
                        feed_fraction[feed_roundoff] = 1.0
                        if np.all(feed_fraction >= np.exp(-700)) and np.all(feed_fraction <= 1):
                            feed_initial = np.log(feed_fraction)
                            feed_candidate = decode(feed_initial)
                            feed_vector = np.array([feed_candidate[k] for k in ids])
                            error = np.max(abs(conserved @ (feed_vector - n0)) / row_scale)
                            if error <= roundoff:
                                try:
                                    db.chemical_potentials(
                                        feed_candidate,
                                        temperature_K=temperature_K,
                                        pressure_Pa=pressure_Pa,
                                    )
                                except ValueError:
                                    pass
                                else:
                                    initial = feed_initial
                                    attempt["initialization"].update(
                                        strategy="actual_feed_after_lp_domain_failure",
                                        lp_domain_failure=str(exc),
                                        feed_scaled_invariant_error=float(error),
                                    )
            root = solve_steady_residual(system, initial, policy=spec.root_policy)
            attempt["root"] = root.root.diagnostics
            attempt["root_success"] = bool(root.root.success)
            if not root.root.success:
                attempt["failure"] = root.root.failure_reason
                attempt["unresolved"] = True
                continue
            amounts = decode(root.root.value)
            n = np.array([amounts[k] for k in ids])
            report = db.chemical_potentials(
                amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
            )
            affinities = db.reaction_affinities(report)
            saturation = {
                k: (
                    affinities[r]["reaction_gibbs_J_mol"] / (R * temperature_K)
                    if affinities[r]["reaction_gibbs_J_mol"] is not None
                    else None
                )
                for k, r in spec.dissolution_reactions.items()
            }
            attempt.update(
                amounts_mol=amounts,
                saturation_indices=saturation,
                invariant_residuals_mol=(conserved @ (n - n0)).tolist(),
            )
            liquid_affinities = {
                key: (
                    affinities[key]["reaction_gibbs_J_mol"] / (R * temperature_K)
                    if affinities[key]["reaction_gibbs_J_mol"] is not None
                    else None
                )
                for key in spec.reaction_ids
                if key not in spec.dissolution_reactions.values()
            }
            attempt["liquid_reaction_affinities_RT"] = liquid_affinities
            if any(
                (value is None and key not in blocked_reactions)
                or (value is not None and abs(value) > 2e-7)
                for key, value in liquid_affinities.items()
            ):
                attempt["failure"] = "selected_liquid_affinity_not_closed"
                continue
            violation = any(
                (value is None and not absent_element[mineral_indices[k]])
                or (value is not None and (abs(value) if k in present else value) > 2e-7)
                for k, value in saturation.items()
            )
            if violation:
                attempt["failure"] = "mineral_complementarity_not_closed"
                continue
            if np.max(abs(conserved @ (n - n0)) / row_scale, initial=0.0) > 2e-8:
                attempt["failure"] = "original_reaction_invariant_not_closed"
                continue
            try:
                projected = tangent.T @ hessian(amounts) @ tangent
                eigenvalues = np.linalg.eigvalsh(projected)
                attempt["constrained_curvature_per_mol"] = eigenvalues.tolist()
                cutoff = 1e-10 * max(1.0, float(np.max(abs(eigenvalues), initial=0.0)))
                if len(eigenvalues) and eigenvalues[0] <= cutoff:
                    if eigenvalues[0] < -cutoff:
                        attempt["failure"] = "negative_local_curvature"
                    else:
                        attempt["failure"] = "unresolved_marginal_local_curvature"
                        attempt["unresolved"] = True
                    continue
                properties = aqueous_mineral_properties(
                    spec, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa
                )
            except ValueError as exc:
                attempt["failure"] = str(exc)
                attempt["unresolved"] = True
                continue
            if properties.gibbs_energy_J > initial_properties.gibbs_energy_J + 1e-7:
                attempt["failure"] = "candidate_gibbs_exceeds_original_feed"
                continue
            attempt.update(success=True, gibbs_energy_J=properties.gibbs_energy_J)
            accepted.append((amounts, saturation, properties))
    unresolved = [
        {"present_minerals": attempt["present_minerals"], "reason": attempt["failure"]}
        for attempt in attempts
        if attempt.get("unresolved", False)
    ]
    diagnostics: dict[str, Any] = {
        "phase_sets": attempts,
        "phase_sets_attempted": len(attempts),
        "phase_sets_accepted": len(accepted),
        "phase_sets_enumerated": True,
        "phase_search_complete": not unresolved,
        "unresolved_phase_sets": unresolved,
        "candidate_guarantee": "all_selected_reaction_KKT_and_local_curvature_only",
        "global_stability_claimed": False,
        "selected_reactions": spec.reaction_ids,
        "scope": "finite_pure_mineral_sets; local_liquid_stability; closed_TP",
        "SI_convention": "dissolution_DeltaG/RT; absent<=0; present=0",
        "initial_properties": initial_properties.to_dict(),
    }
    if blocked_reactions:
        diagnostics.update(
            structurally_blocked_reactions=blocked_reactions,
            absent_conserved_elements=[labels[i] for i, value in enumerate(totals) if value == 0],
        )
    if not accepted:
        return AqueousMineralEquilibriumResult(
            False, None, {}, None, diagnostics, "no_qualified_mineral_phase_set"
        )
    amounts, saturation, properties = min(accepted, key=lambda item: item[2].gibbs_energy_J)
    change = np.array([amounts[k] for k in ids]) - n0
    diagnostics.update(
        element_residuals_mol=dict(zip(labels, map(float, atoms @ change), strict=True)),
        charge_residual_mol=float(charges @ change),
        invariant_residuals_mol=(conserved @ change).tolist(),
    )
    return AqueousMineralEquilibriumResult(True, amounts, saturation, properties, diagnostics)
