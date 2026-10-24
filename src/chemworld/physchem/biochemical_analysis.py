"""Steady state, conserved-space stability, MCA and qualified molecular LNA.

The original SBML evaluator remains the only source of kinetic equations. This
module owns numerical analysis candidates, never a live inventory or clock.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any, cast

import numpy as np
from scipy.linalg import qr, solve_continuous_lyapunov

from chemworld.data.sbml_parser import walk_records
from chemworld.foundation.discrete_inventory import AVOGADRO
from chemworld.physchem.biochemical_analysis_specs import (
    BiochemicalAnalysisPolicy,
    BiochemicalAnalysisResult,
)
from chemworld.physchem.caloric_math import FirstDerivative, scalar_value
from chemworld.physchem.residual_system import (
    ResidualDerivatives,
    ResidualDomainError,
    ResidualEquation,
    ResidualNumericalError,
    ResidualSystem,
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_steady_residual,
)
from chemworld.physchem.sbml_kinetics import SBMLKinetics


def _operations(node: Any) -> set[str]:
    return {node.operation} | {op for child in node.arguments for op in _operations(child)}


def _check_scope(frame: SBMLKinetics) -> None:
    root = frame.model.declaration
    if root.items("listOfEvents") or frame._rate_rules:
        raise ValueError(
            "steady analysis requires autonomous event-free reaction dynamics, no rate rules"
        )
    assigned = cast(Mapping[str, Any], frame._assigned)
    species = cast(Mapping[str, Any], frame._species)
    compartments = cast(Mapping[str, Any], frame._compartments)
    if assigned.keys() & (species.keys() | compartments.keys()):
        raise ValueError("steady analysis does not eliminate assigned species or compartments")
    nonsmooth = {
        "piecewise",
        "abs",
        "min",
        "max",
        "floor",
        "ceiling",
        "factorial",
        "eq",
        "neq",
        "lt",
        "leq",
        "gt",
        "geq",
        "and",
        "or",
        "xor",
        "not",
        "implies",
    }
    for record in walk_records(root):
        expression = record.mathematics
        if expression is None:
            continue
        if "$time" in expression.symbols:
            raise ValueError("steady analysis does not freeze an explicitly time-dependent model")
        if (
            record.kind in {"kineticLaw", "assignmentRule", "functionDefinition"}
            and _operations(expression._root) & nonsmooth
        ):
            raise ValueError("local analysis requires smooth declared kinetic mathematics")


def _stoichiometry(frame: SBMLKinetics, values: Mapping[str, Any], time_s: float) -> np.ndarray:
    context = frame.context(values, time_s)
    source = dict(frame._stoichiometries)
    default = frame.model.declaration.get("conversionFactor")
    rows = []
    amount_scales = cast(Mapping[str, float], frame._amount_scale)
    for species, declaration in cast(Mapping[str, Any], frame._species).items():
        conversion = declaration.get("conversionFactor", default)
        factor = context[conversion] if conversion else 1.0
        scale = amount_scales[species] / (frame._extent_scale or 1.0)
        rows.append([source[key].get(species, 0.0) * factor * scale for key in frame._reaction_ids])
    return np.asarray(rows, dtype=float).reshape(len(rows), len(frame._reaction_ids))


def _normalized_rows(matrix: np.ndarray, denominators: np.ndarray) -> list[list[float | None]]:
    return [
        [float(x / denominator) if denominator != 0 else None for x in row]
        for row, denominator in zip(matrix, denominators, strict=True)
    ]


def _noise_result(
    frame: SBMLKinetics,
    full_n: np.ndarray,
    n: np.ndarray,
    basis: np.ndarray,
    reduced_j: np.ndarray,
    rates: np.ndarray,
    stability: str,
    policy: BiochemicalAnalysisPolicy,
) -> dict[str, Any]:
    if stability != "asymptotically_stable":
        return {"status": "unavailable", "reason": "stationary_noise_requires_asymptotic_stability"}
    if any(r.get("reversible") for r in frame.model.declaration.items("listOfReactions")):
        return {
            "status": "unavailable",
            "reason": "reversible_net_rate_is_not_independent_noise_channels",
        }
    if np.any(rates < 0):
        return {"status": "unavailable", "reason": "negative_reaction_propensity"}
    if np.any(np.abs(full_n - np.rint(full_n)) > policy.integer_jump_tolerance):
        return {
            "status": "unavailable",
            "reason": "noninteger_molecular_jumps_after_unit_conversion",
        }
    if frame.model.items_per_mole is not None and frame.model.items_per_mole != AVOGADRO:
        return {"status": "unavailable", "reason": "item_policy_does_not_declare_molecular_moles"}
    diffusion = (n * rates[None, :]) @ n.T / AVOGADRO
    reduced_d = basis.T @ diffusion @ basis
    reduced_c = solve_continuous_lyapunov(reduced_j, -reduced_d)
    covariance = basis @ reduced_c @ basis.T
    residual = reduced_j @ reduced_c + reduced_c @ reduced_j.T + reduced_d
    scale = float(np.max(np.abs(reduced_d)))
    relative_residual = float(np.max(np.abs(residual))) / scale if scale else 0.0
    return {
        "status": "available",
        "basis": "formal_molecular_amounts_mol_squared",
        "avogadro_per_mol": AVOGADRO,
        "diffusion_mol2_s": diffusion.tolist(),
        "covariance_mol2": covariance.tolist(),
        "reduced_covariance_mol2": reduced_c.tolist(),
        "lyapunov_relative_residual": relative_residual,
        "symmetry_error_mol2": float(np.max(np.abs(covariance - covariance.T))),
        "minimum_covariance_eigenvalue_mol2": float(np.min(np.linalg.eigvalsh(covariance))),
        "interpretation": "linear-noise stationary approximation; not a stochastic trajectory",
    }


def analyze_biochemical_steady_state(
    frame: SBMLKinetics,
    values: Mapping[str, float],
    *,
    time_s: float = 0.0,
    policy: BiochemicalAnalysisPolicy | None = None,
    include_control: bool = True,
    include_lna: bool = False,
) -> BiochemicalAnalysisResult:
    """Analyze the supplied conserved-total class without modifying its source.

    Concentrations/fluxes retain original source semantics. Analytic derivatives
    are taken with respect to native mol coordinates; constant volumes/held
    pools are fixed. Unknown chemical composition and energy remain unknown.
    """
    if not isinstance(frame, SBMLKinetics):
        raise ValueError("analysis requires the original validated SBMLKinetics")
    if not isfinite(time_s) or type(include_control) is not bool or type(include_lna) is not bool:
        raise ValueError("analysis time and requested methods must be explicit")
    policy = policy or BiochemicalAnalysisPolicy()
    _check_scope(frame)
    fixed = {key: float(value) for key, value in values.items()}
    frame.validate_values(fixed, time_s)
    declared_species = cast(Mapping[str, Any], frame._species)
    names = tuple(
        s
        for s, declaration in declared_species.items()
        if not declaration.get("boundaryCondition") and not declaration.get("constant")
    )
    if not names:
        raise ValueError("analysis requires at least one dynamic species coordinate")
    coordinates = tuple("N:" + s for s in names)
    all_names = tuple(declared_species)
    full_n = _stoichiometry(frame, fixed, time_s)
    n = full_n[[all_names.index(s) for s in names], :]
    u, singular, _ = np.linalg.svd(n, full_matrices=True)
    rank = (
        int(np.sum(singular > policy.rank_relative_tolerance * singular[0])) if singular.size else 0
    )
    basis, conservation = u[:, :rank], u[:, rank:].T
    independent_rows = (
        qr(n.T, pivoting=True, mode="economic")[2][:rank] if rank else np.array([], int)
    )
    initial = np.asarray([fixed[k] for k in coordinates])
    totals = conservation @ initial
    scale = policy.amount_scale_mol or max((v for v in initial if v > 0), default=1.0)
    identity = np.eye(len(names))

    def evaluate(point: np.ndarray, *, derivative: bool = False) -> tuple[np.ndarray, ...]:
        trial: dict[str, Any] = dict(fixed)
        trial.update(
            {
                key: FirstDerivative(float(point[i]), identity[i])
                if derivative
                else float(point[i])
                for i, key in enumerate(coordinates)
            }
        )
        try:
            frame.validate_values(trial, time_s)
            result = frame.rates(trial, time_s)
        except (ValueError, ArithmeticError) as error:
            raise ResidualDomainError(str(error)) from error
        species_rates = [result.coordinate_rates[k] for k in coordinates]
        fluxes = [result.reaction_extents_mol_s[k] for k in frame._reaction_ids]
        f = np.asarray([scalar_value(x) for x in species_rates], dtype=float)
        v = np.asarray([scalar_value(x) for x in fluxes], dtype=float)
        if derivative:
            j = np.asarray(
                [
                    x.slope if isinstance(x, FirstDerivative) else np.zeros(len(names))
                    for x in species_rates
                ],
                dtype=float,
            )
            e = np.asarray(
                [
                    x.slope if isinstance(x, FirstDerivative) else np.zeros(len(names))
                    for x in fluxes
                ],
                dtype=float,
            ).reshape(len(fluxes), len(names))
            return f, v, j, e
        return f, v

    def residual(
        t: float, point: np.ndarray, rate: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        return np.concatenate((evaluate(point)[0][independent_rows], conservation @ point - totals))

    def jacobian(
        t: float, point: np.ndarray, rate: np.ndarray, parameters: np.ndarray
    ) -> np.ndarray:
        return np.vstack((evaluate(point, derivative=True)[2][independent_rows], conservation))

    system = ResidualSystem(
        "sbml-biochemical-steady-analysis",
        tuple(ResidualVariable(s, "mol", scale, "algebraic", lower=0) for s in names),
        tuple(
            ResidualEquation("rate:" + names[i], "mol/s", scale / policy.time_scale_s)
            for i in independent_rows
        )
        + tuple(ResidualEquation(f"conserved:{i}", "mol", scale) for i in range(len(totals))),
        residual,
        ResidualDerivatives(
            "analytic", state=jacobian, rate=lambda t, y, yd, p: np.zeros((len(names), len(names)))
        ),
    )
    solved = solve_steady_residual(
        system,
        initial,
        time_s=time_s,
        policy=RootPolicy(
            policy.root_tolerance, policy.maximum_iterations, policy.maximum_backtracks
        ),
    )
    payload: dict[str, Any] = {
        "model": "native-biochemical-steady-analysis-v1",
        "status": "steady_state_found" if solved.root.success else "steady_state_not_found",
        "failure_reason": solved.root.failure_reason,
        "model_id": frame.model.declaration.get("id", "anonymous"),
        "source": frame.model.source.to_dict(),
        "policy": policy.to_dict(),
        "time_s": time_s,
        "species_ids": list(names),
        "reaction_ids": list(frame._reaction_ids),
        "initial_amounts_mol": dict(zip(names, initial.tolist(), strict=True)),
        "fixed_coordinates": {k: v for k, v in fixed.items() if k not in coordinates},
        "stoichiometry": n.tolist(),
        "stoichiometric_rank": rank,
        "dynamic_basis": basis.tolist(),
        "conservation_rows": conservation.tolist(),
        "conserved_totals_mol": totals.tolist(),
        "removed_conservation_modes": len(totals),
        "root_diagnostic": solved.root.diagnostics,
        "derivative_diagnostic": solved.diagnostics,
        "chemical_composition_scope": "unknown_formal_entities",
        "energy_scope": "not_declared_formal_kinetics",
        "control": {"status": "not_requested" if not include_control else "unavailable"},
        "linear_noise": {"status": "not_requested" if not include_lna else "unavailable"},
    }
    if not solved.root.success:
        return BiochemicalAnalysisResult(payload)
    point = solved.root.value
    f, v, j, e = evaluate(point, derivative=True)
    reduced_j = basis.T @ j @ basis
    raw_eigenvalues = np.linalg.eigvals(reduced_j) if rank else np.array([], complex)
    eigenvalues = sorted((complex(z) for z in raw_eigenvalues), key=lambda z: (z.real, z.imag))
    if not rank:
        stability = "no_dynamic_modes"
    elif max(z.real for z in eigenvalues) > policy.stability_tolerance_s_inv:
        stability = "unstable"
    elif max(z.real for z in eigenvalues) < -policy.stability_tolerance_s_inv:
        stability = "asymptotically_stable"
    else:
        stability = "marginal_or_unresolved"
    payload.update(
        {
            "steady_amounts_mol": dict(zip(names, point.tolist(), strict=True)),
            "steady_reaction_extents_mol_s": dict(
                zip(frame._reaction_ids, v.tolist(), strict=True)
            ),
            "steady_rate_residual_mol_s": f.tolist(),
            "conservation_residual_mol": (conservation @ point - totals).tolist(),
            "stoichiometric_rhs_residual_mol_s": (n @ v - f).tolist(),
            "jacobian_s_inv": j.tolist(),
            "reduced_jacobian_s_inv": reduced_j.tolist(),
            "flux_elasticity_s_inv": e.tolist(),
            "scaled_flux_elasticity": _normalized_rows(e * point[None, :], v),
            "stability": stability,
            "reduced_eigenvalues_s_inv": [{"real": z.real, "imag": z.imag} for z in eigenvalues],
            "full_eigenvalues_s_inv": [
                {"real": float(z.real), "imag": float(z.imag)} for z in np.linalg.eigvals(j)
            ],
            "relaxation_times_s": [
                -1 / z.real if z.real < -policy.stability_tolerance_s_inv else None
                for z in eigenvalues
            ],
        }
    )
    if include_control:
        if not rank:
            payload["control"] = {"status": "unavailable", "reason": "no_dynamic_modes"}
        elif any(abs(z) <= policy.stability_tolerance_s_inv for z in eigenvalues):
            payload["control"] = {
                "status": "unavailable",
                "reason": "singular_or_unresolved_reduced_jacobian",
            }
        else:
            try:
                response = basis @ linear_solve(reduced_j, -(basis.T @ n) * v[None, :])
                flux_response = np.diag(v) + e @ response
                payload["control"] = {
                    "status": "available",
                    "perturbation": "independent_log_reaction_activity",
                    "fixed_conserved_totals": True,
                    "amount_response_mol": response.tolist(),
                    "flux_response_mol_s": flux_response.tolist(),
                    "scaled_amount_control": _normalized_rows(response, point),
                    "scaled_flux_control": _normalized_rows(flux_response, v),
                    "amount_summation_residual_mol": response.sum(axis=1).tolist(),
                    "flux_summation_residual_mol_s": (flux_response.sum(axis=1) - v).tolist(),
                    "conservation_response_residual_mol": (conservation @ response).tolist(),
                    "branch_interpretation": "local steady branch; stability reported separately",
                }
            except ResidualNumericalError as error:
                payload["control"] = {"status": "unavailable", "reason": str(error)}
    if include_lna:
        payload["linear_noise"] = _noise_result(
            frame, full_n, n, basis, reduced_j, v, stability, policy
        )
        if payload["linear_noise"]["status"] == "available":
            payload["linear_noise"]["conservation_covariance_residual_mol2"] = (
                conservation @ np.asarray(payload["linear_noise"]["covariance_mol2"])
            ).tolist()
    return BiochemicalAnalysisResult(payload)
