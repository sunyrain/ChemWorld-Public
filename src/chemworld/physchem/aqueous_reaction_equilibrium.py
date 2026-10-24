"""Finite single-liquid equilibrium in a declared aqueous reaction subspace.

This is a pure closure: caller-owned amounts are never mutated. The shared
residual solver owns Newton iteration; the 39A database owns thermochemistry.
Linear feasibility can initialize a positive numerical trial, not material.
Only a local constrained minimum is claimed, never global phase stability.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import fsum, isfinite
from typing import Any

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog

from chemworld.physchem.aqueous_database import AqueousChemicalPotentialReport, AqueousDatabase
from chemworld.physchem.caloric_math import partial_derivative
from chemworld.physchem.elements import element_matrix
from chemworld.physchem.property_reports import R_J_PER_MOL_K, StatePropertyReport
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
class AqueousEquilibriumSpec:
    database: AqueousDatabase
    reaction_ids: tuple[str, ...]
    root_policy: RootPolicy = field(
        default_factory=lambda: RootPolicy(tolerance=1e-11, maximum_iterations=100)
    )
    derivative_mode: str = "analytic"

    def __post_init__(self) -> None:
        object.__setattr__(self, "reaction_ids", tuple(self.reaction_ids))
        reactions = {r.reaction_id: r for r in self.database.reactions}
        liquid = {s.species_id for s in self.database.species if s.phase in ("aqueous", "solvent")}
        if len(set(self.reaction_ids)) != len(self.reaction_ids):
            raise ValueError("equilibrium reaction identities must be unique")
        if set(self.reaction_ids) - set(reactions):
            raise ValueError("equilibrium references an undeclared database reaction")
        if any(set(reactions[key].stoichiometry) - liquid for key in self.reaction_ids):
            raise ValueError("single-liquid equilibrium accepts only aqueous/solvent reactions")
        if self.derivative_mode not in ("analytic", "finite_difference"):
            raise ValueError("declare analytic or finite_difference equilibrium derivatives")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-aqueous-reaction-equilibrium-v1",
            "database": self.database.to_dict(),
            "reaction_ids": list(self.reaction_ids),
            "root_policy": {
                "tolerance": self.root_policy.tolerance,
                "maximum_iterations": self.root_policy.maximum_iterations,
                "maximum_backtracks": self.root_policy.maximum_backtracks,
            },
            "derivative_mode": self.derivative_mode,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AqueousEquilibriumSpec:
        if payload.get("model") != "chemworld-aqueous-reaction-equilibrium-v1":
            raise ValueError("unsupported aqueous equilibrium schema")
        return cls(
            AqueousDatabase.from_dict(payload["database"]),
            tuple(payload["reaction_ids"]),
            RootPolicy(**payload["root_policy"]),
            payload["derivative_mode"],
        )


@dataclass(frozen=True)
class AqueousEquilibriumResult:
    success: bool
    failure_reason: str
    amounts_mol: dict[str, float] | None
    property_report: StatePropertyReport | None
    chemical_potential_report: AqueousChemicalPotentialReport | None
    reaction_residuals: dict[str, float | None]
    element_residuals_mol: dict[str, float]
    charge_residual_mol: float | None
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "failure_reason": self.failure_reason,
            "amounts_mol": self.amounts_mol,
            "property_report": self.property_report.to_dict() if self.property_report else None,
            "chemical_potential_report": (
                self.chemical_potential_report.to_dict() if self.chemical_potential_report else None
            ),
            "reaction_residuals": self.reaction_residuals,
            "element_residuals_mol": self.element_residuals_mol,
            "charge_residual_mol": self.charge_residual_mol,
            "diagnostics": self.diagnostics,
        }


def aqueous_chemical_potential_jacobian(
    database: AqueousDatabase,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    species_ids: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Analytic d(mu/RT)/dn for positive selected liquid species, in 1/mol.

    Absent species may remain fixed at zero. The excess Hessian uses the same
    nested derivative arithmetic as the activity energy, including its explicit
    nonsmooth zero-ionic-strength boundary.
    """
    report = database.chemical_potentials(
        amounts_mol, temperature_K=temperature_K, pressure_Pa=pressure_Pa
    )
    water = database.solvent_species.species_id
    solutes = tuple(s.species_id for s in database.species if s.phase == "aqueous")
    selected = species_ids if species_ids is not None else (water, *solutes)
    if len(set(selected)) != len(selected) or set(selected) - {water, *solutes}:
        raise ValueError("Jacobian species must be unique liquid identities")
    if any(report.amounts_mol[k] <= 0 for k in selected):
        raise ValueError("chemical-potential Jacobian requires positive selected species")
    m = {k: report.amounts_mol[k] / report.solvent_mass_kg for k in solutes}
    active = tuple(k for k in solutes if report.amounts_mol[k] > 0)

    def energy(values: Mapping[str, Any]) -> Any:
        return database.activity_model.excess_gibbs_RT_mol_kg(values, temperature_K, pressure_Pa)

    excess = np.empty((len(active), len(active)))
    for i, left in enumerate(active):
        for j in range(i + 1):
            value = partial_derivative(energy, m, (left, active[j]))
            excess[i, j] = excess[j, i] = value
    molal = np.array([m[k] for k in active])
    nw = report.amounts_mol[water]
    mass = report.solvent_mass_kg / nw
    total = len(active) + 1
    answer = np.zeros((total, total))
    answer[1:, 1:] = (excess + np.diag(1 / molal)) / report.solvent_mass_kg
    answer[0, 1:] = answer[1:, 0] = -(1 + excess @ molal) / nw
    answer[0, 0] = mass / nw * (sum(molal) + molal @ excess @ molal)
    order = (water, *active)
    indices = [order.index(key) for key in selected]
    return answer[np.ix_(indices, indices)]


def _basis(rows: list[np.ndarray], width: int) -> np.ndarray:
    selected: list[np.ndarray] = []
    rank = 0
    for row in rows:
        norm = np.linalg.norm(row)
        if norm == 0:
            continue
        value = row / norm
        candidate = np.asarray([*selected, value])
        new_rank = int(np.linalg.matrix_rank(candidate, tol=1e-11))
        if new_rank > rank:
            selected.append(value)
            rank = new_rank
    return np.array(selected).reshape((-1, width))


@dataclass(frozen=True)
class AqueousEquilibriumProblem:
    """Ephemeral coordinate view of one shared algebraic residual problem."""

    spec: AqueousEquilibriumSpec
    system: ResidualSystem
    initial: np.ndarray
    species_ids: tuple[str, ...]
    initial_amounts_mol: dict[str, float]
    amount_scales_mol: np.ndarray
    conservation_matrix: np.ndarray
    conservation_scales_mol: np.ndarray
    reaction_tangent: np.ndarray
    temperature_K: float
    pressure_Pa: float
    diagnostics: dict[str, Any]

    def decode(self, coordinates: np.ndarray) -> dict[str, float]:
        values = self.amount_scales_mol * np.exp(coordinates)
        return {
            **dict.fromkeys(self.initial_amounts_mol, 0.0),
            **dict(zip(self.species_ids, map(float, values), strict=True)),
        }


def build_aqueous_equilibrium_system(
    spec: AqueousEquilibriumSpec,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    initial_guess_mol: Mapping[str, float] | None = None,
) -> AqueousEquilibriumProblem:
    """Build a square, scaled reaction/conservation system without committing stock.

    A supplied positive guess is numerical only; the conservation target always
    remains the actual feed. Structural zero-element and fixed absent species
    stay exactly zero. Other unsupported boundary faces are explicit failures.
    """
    db = spec.database
    liquid = tuple(s for s in db.species if s.phase in ("aqueous", "solvent"))
    ids = tuple(s.species_id for s in liquid)
    if set(amounts_mol) - set(ids) or any(
        isinstance(v, bool) or not isfinite(v) or v < 0 for v in amounts_mol.values()
    ):
        raise ValueError("single-liquid feed requires declared finite nonnegative amounts")
    feed = {key: float(amounts_mol.get(key, 0)) for key in ids}
    water = db.solvent_species.species_id
    if feed[water] <= 0:
        raise ValueError("equilibrium requires positive actual water inventory")
    charges = np.array([s.charge for s in liquid], dtype=float)
    n0 = np.array([feed[k] for k in ids])
    initial_charge = fsum(q * n for q, n in zip(charges, n0, strict=True))
    charge_scale = max(float(abs(charges) @ n0), feed[water] * 1e-14)
    if abs(initial_charge) > 1e-10 * charge_scale:
        raise ValueError("feed must be electroneutral; no counterion repair is allowed")
    # Validate declarations and initial model domain before a root is attempted.
    db.chemical_potentials(feed, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
    atom_rows, labels = element_matrix([dict(s.composition) for s in liquid])
    atoms = np.asarray(atom_rows).T
    totals = atoms @ n0
    reaction_map = {r.reaction_id: r for r in db.reactions}
    stoich = np.asarray(
        [[reaction_map[r].stoichiometry.get(k, 0) for r in spec.reaction_ids] for k in ids],
        dtype=float,
    ).reshape((len(ids), len(spec.reaction_ids)))
    structural_zero = np.any((atoms > 0) & (totals[:, None] == 0), axis=0)
    structural_zero |= (np.all(stoich == 0, axis=1)) & (n0 == 0)
    active_indices = np.flatnonzero(~structural_zero)
    active_ids = tuple(ids[i] for i in active_indices)
    # Element/charge rows first make tiny charge constraints independent of the
    # large water inventory; nullspace rows preserve every additional invariant.
    conserved = null_space(stoich.T, rcond=1e-12).T
    rows = [charges, *sorted(atoms, key=lambda row: float(abs(row) @ n0)), *conserved]
    constraints = _basis([row[active_indices] for row in rows], len(active_ids))
    tangent = null_space(constraints, rcond=1e-12)
    initial_active = n0[active_indices]
    row_scales = abs(constraints) @ initial_active
    upper = np.array(
        [
            min(totals[j] / atoms[j, i] for j in range(len(labels)) if atoms[j, i] > 0)
            for i in active_indices
        ]
    )
    # Upper bounds come from actual atom inventory, not a fitted concentration.
    scale = upper.copy()
    init_diagnostics: dict[str, Any] = {"method": "feed"}
    if initial_guess_mol is not None:
        if set(initial_guess_mol) - set(ids) or any(
            isinstance(v, bool) or not isfinite(v) or v < 0 for v in initial_guess_mol.values()
        ):
            raise ValueError("initial guess must contain finite nonnegative liquid amounts")
        if any(initial_guess_mol.get(ids[i], 0) != 0 for i in np.flatnonzero(structural_zero)):
            raise ValueError("initial guess cannot populate structurally absent species")
        guess = np.array([initial_guess_mol.get(k, feed[k]) for k in active_ids], dtype=float)
        if np.any(guess <= 0):
            raise ValueError("initial guess needs positive amounts for every active species")
        init_diagnostics = {"method": "caller_numerical_guess"}
    elif np.all(initial_active > 0):
        guess = initial_active.copy()
    else:
        width = len(active_ids)
        # Maximize a common fractional amount. This linear problem supplies only
        # a feasible starting point; the thermodynamic root remains in task 21.
        feasibility_scales = np.max(abs(constraints) * scale[None, :], axis=1)
        aeq = constraints * scale[None, :] / feasibility_scales[:, None]
        beq = constraints @ initial_active / feasibility_scales
        feasible = linprog(
            np.r_[np.zeros(width), -1.0],
            A_ub=np.column_stack((-np.eye(width), np.ones(width))),
            b_ub=np.zeros(width),
            A_eq=np.column_stack((aeq, np.zeros(len(constraints)))),
            b_eq=beq,
            bounds=[(0, 1)] * (width + 1),
            method="highs",
        )
        init_diagnostics = {"method": "linear_feasible_interior", "status": int(feasible.status)}
        if not feasible.success or feasible.x[-1] <= 1e-14:
            raise ResidualDomainError("no_resolved_positive_feasible_state")
        interior = scale * feasible.x[:-1]
        # Stay near the feed so a numerical seed does not need a high-ionic
        # strength material model merely to initialize dilute equilibrium.
        guess = initial_active + 1e-8 * (interior - initial_active)
        init_diagnostics["maximum_common_fraction"] = float(feasible.x[-1])
    if np.any(guess > upper * (1 + 1e-10)):
        raise ValueError("initial guess exceeds an elemental inventory upper bound")
    lower_log = np.full(len(active_ids), -700.0)
    initial = np.log(guess / scale)
    if np.any(initial < lower_log):
        raise ResidualDomainError("positive_initial_amount_below_log_coordinate_resolution")

    def decode(values: np.ndarray) -> dict[str, float]:
        n = scale * np.exp(values)
        return {**dict.fromkeys(ids, 0.0), **dict(zip(active_ids, map(float, n), strict=True))}

    def chemical(values: np.ndarray) -> AqueousChemicalPotentialReport:
        try:
            return db.chemical_potentials(
                decode(values), temperature_K=temperature_K, pressure_Pa=pressure_Pa
            )
        except ValueError as exc:
            raise ResidualDomainError(str(exc)) from exc

    def residual(t: float, y: np.ndarray, rate: np.ndarray, p: np.ndarray) -> np.ndarray:
        report = chemical(y)
        mu = np.array([report.chemical_potentials_J_mol[k] for k in active_ids], dtype=float)
        n = np.array([report.amounts_mol[k] for k in active_ids])
        change = n - initial_active
        denominator = row_scales + abs(constraints) @ n
        return np.r_[
            tangent.T @ mu / (R_J_PER_MOL_K * temperature_K),
            constraints @ change / denominator,
        ]

    def jacobian(t: float, y: np.ndarray, rate: np.ndarray, p: np.ndarray) -> np.ndarray:
        amounts = decode(y)
        try:
            hessian = aqueous_chemical_potential_jacobian(
                db,
                amounts,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                species_ids=active_ids,
            )
        except ValueError as exc:
            raise ResidualDomainError(str(exc)) from exc
        n = np.array([amounts[k] for k in active_ids])
        denominator = row_scales + abs(constraints) @ n
        imbalance = constraints @ (n - initial_active)
        balance_jacobian = (
            constraints / denominator[:, None]
            - imbalance[:, None] * abs(constraints) / denominator[:, None] ** 2
        )
        return np.vstack((tangent.T @ hessian, balance_jacobian)) * n[None, :]

    def rate_jacobian(t: float, y: np.ndarray, rate: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.zeros((len(active_ids), len(active_ids)))

    derivatives = (
        ResidualDerivatives("analytic", state=jacobian, rate=rate_jacobian)
        if spec.derivative_mode == "analytic"
        else ResidualDerivatives("finite_difference")
    )
    system = ResidualSystem(
        db.database_id + ":single-liquid-equilibrium",
        tuple(
            ResidualVariable("log_amount:" + k, "1", role="algebraic", lower=-700, upper=0)
            for k in active_ids
        ),
        tuple(ResidualEquation("affinity:" + str(i), "1") for i in range(tangent.shape[1]))
        + tuple(ResidualEquation("conservation:" + str(i), "1") for i in range(len(constraints))),
        residual,
        derivatives,
    )
    return AqueousEquilibriumProblem(
        spec,
        system,
        initial,
        active_ids,
        feed,
        scale,
        constraints,
        row_scales,
        tangent,
        temperature_K,
        pressure_Pa,
        {
            "initialization": init_diagnostics,
            "structurally_absent_species": [ids[i] for i in np.flatnonzero(structural_zero)],
            "reaction_subspace_dimension": tangent.shape[1],
            "conserved_dimension": len(constraints),
            "selected_reaction_ids": list(spec.reaction_ids),
            "equilibrium_scope": "local_closed_single_liquid_TP",
        },
    )


def solve_aqueous_reaction_equilibrium(
    spec: AqueousEquilibriumSpec,
    amounts_mol: Mapping[str, float],
    *,
    temperature_K: float,
    pressure_Pa: float,
    initial_guess_mol: Mapping[str, float] | None = None,
) -> AqueousEquilibriumResult:
    """Resolve one closed liquid inventory; failed candidates are never commit-ready."""
    try:
        problem = build_aqueous_equilibrium_system(
            spec,
            amounts_mol,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            initial_guess_mol=initial_guess_mol,
        )
    except ResidualDomainError as exc:
        return AqueousEquilibriumResult(False, str(exc), None, None, None, {}, {}, None, {})
    root = solve_steady_residual(problem.system, problem.initial, policy=spec.root_policy)
    diagnostics = {**problem.diagnostics, **root.diagnostics, "root": root.root.diagnostics}
    if not root.root.success:
        return AqueousEquilibriumResult(
            False, root.root.failure_reason, None, None, None, {}, {}, None, diagnostics
        )
    amounts = problem.decode(root.root.value)
    db = spec.database
    chemical = db.chemical_potentials(amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa)
    reactions = db.reaction_affinities(chemical)
    affinities = {
        key: reactions[key]["reaction_gibbs_J_mol"] / (R_J_PER_MOL_K * temperature_K)
        if reactions[key]["reaction_gibbs_J_mol"] is not None
        else None
        for key in spec.reaction_ids
    }
    liquid = tuple(s for s in db.species if s.phase in ("aqueous", "solvent"))
    atom_rows, labels = element_matrix([dict(s.composition) for s in liquid])
    change = [amounts[s.species_id] - problem.initial_amounts_mol[s.species_id] for s in liquid]
    elements = {
        key: fsum(row[j] * dn for row, dn in zip(atom_rows, change, strict=True))
        for j, key in enumerate(labels)
    }
    charge = fsum(s.charge * dn for s, dn in zip(liquid, change, strict=True))
    final_active = np.array([amounts[k] for k in problem.species_ids])
    initial_active = np.array([problem.initial_amounts_mol[k] for k in problem.species_ids])
    conservation = problem.conservation_matrix @ (final_active - initial_active)
    diagnostics["conservation_residuals_mol"] = conservation.tolist()
    diagnostics["maximum_scaled_conservation_residual"] = float(
        np.max(
            abs(conservation)
            / (problem.conservation_scales_mol + abs(problem.conservation_matrix) @ final_active),
            initial=0,
        )
    )
    diagnostics["solvent_mass_kg"] = chemical.solvent_mass_kg
    failure = ""
    if problem.reaction_tangent.shape[1]:
        try:
            hessian = aqueous_chemical_potential_jacobian(
                db,
                amounts,
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                species_ids=problem.species_ids,
            )
            projected = problem.reaction_tangent.T @ hessian @ problem.reaction_tangent
            eig = np.linalg.eigvalsh(projected)
            diagnostics["constrained_curvature_per_mol"] = eig.tolist()
            cutoff = 1e-10 * max(float(np.max(abs(eig))), 1.0)
            stability = (
                "positive" if eig[0] > cutoff else "negative" if eig[0] < -cutoff else "marginal"
            )
            diagnostics["local_curvature"] = stability
            if stability != "positive":
                failure = "stationary_root_has_" + stability + "_curvature"
        except ValueError as exc:
            diagnostics["curvature_failure"] = str(exc)
            failure = "local_curvature_unavailable"
    else:
        diagnostics["local_curvature"] = "no_reaction_freedom"
        diagnostics["constrained_curvature_per_mol"] = []
    if max((abs(v) for v in affinities.values() if v is not None), default=0) > 2e-8:
        failure = "selected_reaction_affinity_not_closed"
    if failure:
        diagnostics["rejected_candidate_amounts_mol"] = amounts
        return AqueousEquilibriumResult(
            False, failure, None, None, None, affinities, elements, charge, diagnostics
        )
    return AqueousEquilibriumResult(
        True,
        "",
        amounts,
        db.property_report(amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa),
        chemical,
        affinities,
        elements,
        charge,
        diagnostics,
    )
