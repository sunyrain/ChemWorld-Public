"""Finite pseudo-arclength branches of the original smooth SBML equations.

Candidates belong to a read-only analysis. They never replace live inventories,
rerun initialization, execute pending controls, or create a second time owner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.biochemical_analysis import _stoichiometry
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)
from chemworld.physchem.sbml_sensitivity import check_sensitivity_scope
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.sbml_variational import sbml_variational_rates
from chemworld.physchem.stoichiometry import _rref


def sbml_steady_branch(
    model: SBMLSystemSpec,
    current_values: Mapping[str, float],
    *,
    parameter: str,
    start_time_s: float,
    amount_scales_mol: Mapping[str, float],
    parameter_scale: float,
    parameter_bounds: Sequence[float],
    initial_direction: int = 1,
    step_size: float = 0.05,
    minimum_step: float = 1e-5,
    maximum_points: int = 100,
    maximum_retries: int = 16,
    maximum_evaluations: int = 20000,
    maximum_directions: int = 32,
    max_integer_bits: int = 8192,
    time_scale_s: float = 1.0,
    stability_tolerance_s_inv: float = 1e-9,
    maximum_correction_ratio: float = 0.5,
    minimum_tangent_alignment: float = 0.5,
    root_policy: RootPolicy | None = None,
) -> dict[str, Any]:
    """Follow one finite branch, including unstable portions and simple folds.

    Source parameter units and the explicit mol metric define arclength. The
    initial root keeps the current control fixed and the current conservation
    class; subsequent correctors fix a local tangent hyperplane, not a parameter
    grid. ``complete`` means the requested number of points, not all branches.
    Invalid declarations raise; numerical termination returns partial records.
    """
    check_sensitivity_scope(model, (parameter,))
    frame = model.frame
    factors = {frame.model.declaration.get("conversionFactor")} | {
        s.get("conversionFactor") for s in frame._species.values()
    }
    if parameter in factors:
        raise ValueError("continuation control cannot be a stoichiometric conversionFactor")
    if isinstance(start_time_s, bool) or not isfinite(start_time_s) or start_time_s < 0:
        raise ValueError("analysis time must be finite nonnegative seconds")
    positive = (
        parameter_scale,
        step_size,
        minimum_step,
        time_scale_s,
        stability_tolerance_s_inv,
        maximum_correction_ratio,
    )
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in positive):
        raise ValueError("continuation scales, steps and tolerances must be finite positive")
    if minimum_step > step_size or maximum_correction_ratio > 1:
        raise ValueError("minimum step exceeds step or correction ratio exceeds one")
    if (
        isinstance(minimum_tangent_alignment, bool)
        or not isfinite(minimum_tangent_alignment)
        or not 0 < minimum_tangent_alignment < 1
    ):
        raise ValueError("minimum tangent alignment must be between zero and one")
    if type(initial_direction) is not int or initial_direction not in {-1, 1}:
        raise ValueError("initial parameter direction must be +1 or -1")
    if (
        any(
            type(v) is not int or v < 1
            for v in (
                maximum_points,
                maximum_evaluations,
                maximum_directions,
                max_integer_bits,
            )
        )
        or type(maximum_retries) is not int
        or maximum_retries < 0
    ):
        raise ValueError("continuation budgets must be explicit positive integers; retries >= 0")
    bounds = tuple(parameter_bounds)
    if (
        len(bounds) != 2
        or any(isinstance(v, bool) or not isfinite(v) for v in bounds)
        or bounds[0] >= bounds[1]
    ):
        raise ValueError("parameter bounds must be two increasing finite source-unit values")
    policy = RootPolicy() if root_policy is None else root_policy
    if not isinstance(policy, RootPolicy):
        raise ValueError("root policy must be the common RootPolicy")
    if set(current_values) != set(frame.initial_values):
        raise ValueError("continuation requires the complete original coordinate state")
    if any(isinstance(v, bool) or not isfinite(v) for v in current_values.values()):
        raise ValueError("current original coordinates must be finite scalars")
    fixed = {k: float(v) for k, v in current_values.items()}
    frame.validate_values(fixed, start_time_s)
    names = tuple(
        s
        for s, d in frame._species.items()
        if not d.get("boundaryCondition") and not d.get("constant")
    )
    if not names or set(amount_scales_mol) != set(names):
        raise ValueError("explicit mol metric must cover exactly all dynamic species")
    if any(isinstance(v, bool) or not isfinite(v) or v <= 0 for v in amount_scales_mol.values()):
        raise ValueError("amount scales must be finite positive mol")
    if any(fixed["N:" + s] <= 0 for s in names):
        raise ValueError("continuation requires strictly positive current dynamic amounts")
    p0 = fixed["P:" + parameter]
    if not bounds[0] <= p0 <= bounds[1]:
        raise ValueError("current parameter is outside declared source-unit bounds")
    keys = tuple(fixed)
    indices = [keys.index("N:" + s) for s in names]
    metric = np.asarray([amount_scales_mol[s] for s in names], dtype=float)
    all_names = tuple(frame._species)
    full_stoichiometry = _stoichiometry(frame, fixed, start_time_s)
    stoichiometry = full_stoichiometry[[all_names.index(s) for s in names], :]
    _, pivots = _rref(
        [[Fraction(float(v)) for v in row] for row in stoichiometry],
        width=stoichiometry.shape[1],
        max_integer_bits=max_integer_bits,
    )
    rank = len(pivots)
    if rank == 0 or rank > maximum_directions:
        raise ValueError("reachable dimension is zero or exceeds its explicit budget")
    weighted = stoichiometry / metric[:, None]
    if not np.all(np.isfinite(weighted)):
        raise ValueError("metric cannot represent the original stoichiometry")
    vectors, singular, _ = np.linalg.svd(weighted, full_matrices=True)
    rank_floor = 64 * max(weighted.shape) * np.finfo(float).eps * singular[0]
    if singular[rank - 1] <= rank_floor:
        raise ValueError("exact stoichiometric rank is unresolved in the declared metric")
    basis, conservation = vectors[:, :rank], vectors[:, rank:].T
    base = np.asarray([fixed["N:" + s] for s in names])
    totals = conservation @ (base / metric)
    slopes = np.zeros((len(keys), rank + 1))
    slopes[indices, :rank] = metric[:, None] * basis
    slopes[keys.index("P:" + parameter), -1] = parameter_scale
    counts = {
        "variational_rate_calls": 0,
        "record_rate_calls": 0,
        "root_residual_calls": 0,
        "root_jacobian_calls": 0,
        "invalid_trials": 0,
        "tangent_svd_calls": 0,
    }
    records: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "format": "native-sbml-steady-branch-v1",
        "status": "partial",
        "stop_reason": "",
        "parameter": parameter,
        "parameter_source_units": frame._parameters[parameter].get("units"),
        "parameter_scale": parameter_scale,
        "parameter_bounds": list(bounds),
        "start_time_s": start_time_s,
        "initial_coordinates": fixed,
        "species_ids": list(names),
        "amount_scales_mol": dict(amount_scales_mol),
        "basis_scaled_amounts": basis.tolist(),
        "reachable_dimension": rank,
        "rank_method": "exact binary-rational stoichiometry; floating metric resolved separately",
        "stoichiometric_singular_values": singular.tolist(),
        "conservation_rows_scaled_amounts": conservation.tolist(),
        "initial_conserved_totals": totals.tolist(),
        "records": records,
        "attempts": attempts,
        "fold_brackets": folds,
        "costs": counts,
        "policy": {
            "step_size": step_size,
            "minimum_step": minimum_step,
            "maximum_points": maximum_points,
            "maximum_retries": maximum_retries,
            "maximum_evaluations": maximum_evaluations,
            "maximum_directions": maximum_directions,
            "max_integer_bits": max_integer_bits,
            "time_scale_s": time_scale_s,
            "stability_tolerance_s_inv": stability_tolerance_s_inv,
            "maximum_correction_ratio": maximum_correction_ratio,
            "minimum_tangent_alignment": minimum_tangent_alignment,
            "initial_direction": initial_direction,
            "root_tolerance": policy.tolerance,
            "root_maximum_iterations": policy.maximum_iterations,
            "root_maximum_backtracks": policy.maximum_backtracks,
        },
        "interpretation": (
            "one finite connected branch; fold brackets, no all-root or bifurcation proof"
        ),
        "controls": "hold current other controls, volumes, boundary pools and conservation class",
        "energy_scope": "undeclared formal kinetics; no physical heat calculation",
        "world_advanced": False,
    }
    budget_exhausted = False

    def unpack(u: np.ndarray) -> dict[str, float]:
        values = dict(fixed)
        amounts = base + metric * (basis @ u[:rank])
        values.update(zip(("N:" + s for s in names), map(float, amounts), strict=True))
        values["P:" + parameter] = float(p0 + parameter_scale * u[-1])
        if (
            not np.all(np.isfinite(amounts))
            or np.any(amounts <= 0)
            or not bounds[0] <= values["P:" + parameter] <= bounds[1]
        ):
            raise ResidualDomainError("outside strictly positive amounts or parameter bounds")
        frame.validate_values(values, start_time_s)
        return values

    def evaluate(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        nonlocal budget_exhausted
        if counts["variational_rate_calls"] >= maximum_evaluations:
            budget_exhausted = True
            raise ResidualDomainError("variational_evaluation_budget")
        try:
            values = unpack(u)
            counts["variational_rate_calls"] += 1
            rates, derivatives = sbml_variational_rates(frame, values, keys, slopes, start_time_s)
            if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(derivatives)):
                raise ValueError("original rates or derivatives are not finite")
            f = time_scale_s * (basis.T @ (rates[indices] / metric))
            jac = time_scale_s * (basis.T @ (derivatives[indices] / metric[:, None]))
            return f, jac, rates[indices], values
        except (ValueError, ArithmeticError) as error:
            counts["invalid_trials"] += 1
            raise ResidualDomainError(str(error)) from error

    def tangent(jac: np.ndarray, previous: np.ndarray | None) -> tuple[np.ndarray, float]:
        counts["tangent_svd_calls"] += 1
        _, sv, right = np.linalg.svd(jac, full_matrices=True)
        cutoff = 64 * max(jac.shape) * np.finfo(float).eps * sv[0]
        if sv[0] == 0 or sv[-1] <= cutoff:
            raise ResidualDomainError("branch Jacobian has unresolved rank or a branch point")
        t = right[-1]
        if previous is None:
            if abs(t[-1]) <= 64 * max(jac.shape) * np.finfo(float).eps:
                raise ResidualDomainError("initial parameter direction unresolved at fold")
            t *= initial_direction * (1 if t[-1] > 0 else -1)
            return t, 1.0
        dot = float(t @ previous)
        t *= 1 if dot >= 0 else -1
        return t, abs(dot)

    def record(u: np.ndarray, t: np.ndarray, data: tuple, arc: float, chord: float) -> None:
        f, jac, rates, values = data
        if np.max(np.abs(rates / metric)) * time_scale_s > (
            10 * np.sqrt(len(names)) * policy.tolerance
        ):
            raise ResidualDomainError("full original RHS exceeds the accepted residual tolerance")
        spectrum = np.linalg.eigvals(jac[:, :rank] / time_scale_s)
        largest = float(np.max(spectrum.real))
        stability = (
            "asymptotically_stable"
            if largest < -stability_tolerance_s_inv
            else "unstable"
            if largest > stability_tolerance_s_inv
            else "marginal_or_unresolved"
        )
        counts["record_rate_calls"] += 1
        source_rates = frame.rates(values, start_time_s)
        conserved = conservation @ (np.asarray([values["N:" + s] for s in names]) / metric)
        records.append(
            {
                "index": len(records),
                "scaled_coordinates": u.tolist(),
                "coordinates": values,
                "parameter_value": values["P:" + parameter],
                "predictor_arclength": arc,
                "cumulative_chord_length": chord,
                "tangent": t.tolist(),
                "tangent_parameter_source_units": float(t[-1] * parameter_scale),
                "tangent_residual": float(np.max(np.abs(jac @ t))),
                "full_rhs_mol_s": rates.tolist(),
                "scaled_residual_norm": float(np.max(np.abs(f))),
                "full_scaled_rhs_norm": float(np.max(np.abs(rates / metric))) * time_scale_s,
                "conserved_totals": conserved.tolist(),
                "conservation_error": float(np.max(np.abs(conserved - totals), initial=0)),
                "conserved_zero_modes_excluded": len(names) - rank,
                "reduced_jacobian_s_inverse": (jac[:, :rank] / time_scale_s).tolist(),
                "eigenvalues_s_inverse": [[float(v.real), float(v.imag)] for v in spectrum],
                "stability": stability,
                "readout": dict(model.readout(values, start_time_s)),
                "reaction_fluxes_mol_s": dict(source_rates.reaction_extents_mol_s),
            }
        )

    variables = tuple(ResidualVariable(f"z{i}", "1", 1.0) for i in range(rank))
    pvariable = ResidualVariable(
        "q",
        "1",
        1.0,
        lower=(bounds[0] - p0) / parameter_scale,
        upper=(bounds[1] - p0) / parameter_scale,
    )

    def solve(initial: np.ndarray, previous: np.ndarray | None, target: np.ndarray) -> Any:
        def residual(v: np.ndarray) -> np.ndarray:
            counts["root_residual_calls"] += 1
            u = np.r_[v, 0.0] if previous is None else v
            f = evaluate(u)[0]
            return f if previous is None else np.r_[f, previous @ (u - target)]

        def jacobian(v: np.ndarray) -> np.ndarray:
            counts["root_jacobian_calls"] += 1
            u = np.r_[v, 0.0] if previous is None else v
            jac = evaluate(u)[1]
            return jac[:, :rank] if previous is None else np.vstack((jac, previous))

        return solve_root(
            residual,
            initial,
            jacobian=jacobian,
            variables=variables if previous is None else (*variables, pvariable),
            residual_scales=np.ones(rank if previous is None else rank + 1),
            policy=policy,
        )

    try:
        seeded = solve(np.zeros(rank), None, np.zeros(rank + 1))
        attempts.append(
            {
                "kind": "seed",
                "accepted": False,
                "reason": seeded.failure_reason,
                "root": seeded.diagnostics,
            }
        )
        if not seeded.success:
            raise ResidualDomainError("seed_root_failed: " + seeded.failure_reason)
        u = np.r_[seeded.value, 0.0]
        data = evaluate(u)
        t, _ = tangent(data[1], None)
        record(u, t, data, 0.0, 0.0)
        attempts[-1]["accepted"] = True
        arc = chord = 0.0
        last_resolved_direction_index = 0
        while len(records) < maximum_points:
            step = step_size
            accepted = False
            for retry in range(maximum_retries + 1):
                predictor = u + step * t
                attempt: dict[str, Any] = {
                    "kind": "corrector",
                    "from_index": len(records) - 1,
                    "retry": retry,
                    "step": step,
                    "predictor": predictor.tolist(),
                    "accepted": False,
                    "reason": "",
                    "root": None,
                }
                attempts.append(attempt)
                try:
                    solved = solve(predictor, t, predictor)
                    attempt["root"] = solved.diagnostics
                    if not solved.success:
                        raise ResidualDomainError(solved.failure_reason)
                    candidate = solved.value
                    data = evaluate(candidate)
                    next_t, alignment = tangent(data[1], t)
                    correction = float(np.linalg.norm(candidate - predictor))
                    attempt.update(correction_norm=correction, tangent_alignment=alignment)
                    if correction > maximum_correction_ratio * step:
                        raise ResidualDomainError("corrector exceeds local branch neighborhood")
                    if alignment < minimum_tangent_alignment:
                        raise ResidualDomainError(
                            "tangent rotation exceeds local branch neighborhood"
                        )
                    next_arc = arc + step
                    next_chord = chord + float(np.linalg.norm(candidate - u))
                    record(candidate, next_t, data, next_arc, next_chord)
                    # A point may land on the fold itself. Retain the last
                    # resolved sign across numerically zero parameter tangents.
                    if abs(next_t[-1]) > 64 * (rank + 1) * np.finfo(float).eps:
                        prior = records[last_resolved_direction_index]
                        if prior["tangent"][-1] * next_t[-1] < 0:
                            folds.append(
                                {
                                    "indices": [last_resolved_direction_index, len(records) - 1],
                                    "parameter_values": [
                                        prior["parameter_value"],
                                        records[-1]["parameter_value"],
                                    ],
                                    "classification": "tangent_parameter_sign_change_only",
                                }
                            )
                        last_resolved_direction_index = len(records) - 1
                    u, t, arc, chord = candidate, next_t, next_arc, next_chord
                    attempt["accepted"] = accepted = True
                    break
                except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
                    attempt["reason"] = str(error)
                    if budget_exhausted:
                        raise ResidualDomainError("variational_evaluation_budget") from error
                    if retry == maximum_retries:
                        raise ResidualDomainError(
                            "corrector_retry_budget: " + str(error)
                        ) from error
                    if step / 2 < minimum_step:
                        raise ResidualDomainError("minimum_step: " + str(error)) from error
                    step /= 2
            if not accepted:
                raise RuntimeError("unreachable corrector termination")
        report.update(status="complete", stop_reason="requested_point_count")
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as error:
        report["stop_reason"] = "variational_evaluation_budget" if budget_exhausted else str(error)
        if not records:
            report["status"] = "failed"
        if attempts and not attempts[-1]["accepted"] and not attempts[-1]["reason"]:
            attempts[-1]["reason"] = str(error)
    report.update(
        completed_points=len(records),
        requested_points=maximum_points,
        rejected_attempts=sum(not a["accepted"] for a in attempts),
    )
    counts["root_iterations"] = sum(a["root"]["iterations"] for a in attempts if a["root"])
    counts["root_backtracks"] = sum(a["root"]["backtracks"] for a in attempts if a["root"])
    return report
