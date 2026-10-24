"""Finite-window perturbation growth along original smooth SBML trajectories.

Perturbations remain in the stoichiometric compatibility class, with an explicit
fixed mol metric. Singular values of the tangent map are finite-time growth
factors, not instantaneous eigenvalues or asymptotic Lyapunov/chaos claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from math import isfinite
from typing import Any, cast

import numpy as np

from chemworld.physchem.biochemical_analysis import _check_scope, _stoichiometry
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.sbml_variational import sbml_variational_rates
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.stoichiometry import _rref


def sbml_finite_time_stability(
    model: SBMLSystemSpec,
    initial_values: Mapping[str, float],
    *,
    start_time_s: float,
    sample_offsets_s: Sequence[float],
    amount_scales_mol: Mapping[str, float],
    maximum_directions: int = 32,
    max_integer_bits: int = 8192,
) -> dict[str, Any]:
    """Read a hypothetical finite trajectory at held current external controls.

    Native amount coordinates and a dimensionless metric D^-1 delta_n define
    the result. Exact binary-rational rank determines the reachable dimension;
    an unresolved floating representation is rejected rather than dropping a
    conservation direction. Every growth factor is measured from the supplied
    initial state, not from the preceding output sample.
    """
    if not isinstance(model, SBMLSystemSpec) or model.numerical_form != "ode":
        raise ValueError("finite-time stability requires the original explicit SBML ODE model")
    _check_scope(model.frame)
    if isinstance(start_time_s, bool) or not isfinite(start_time_s) or start_time_s < 0:
        raise ValueError("start time must be finite nonnegative seconds")
    offsets = tuple(sample_offsets_s)
    if (
        not offsets
        or any(isinstance(t, bool) or not isfinite(t) or t <= 0 for t in offsets)
        or any(b <= a for a, b in pairwise(offsets))
        or not isfinite(start_time_s + offsets[-1])
    ):
        raise ValueError("sample offsets must be finite positive strictly increasing seconds")
    if type(maximum_directions) is not int or maximum_directions < 1:
        raise ValueError("maximum tangent directions must be a positive integer")
    if type(max_integer_bits) is not int or max_integer_bits < 1:
        raise ValueError("exact rank bit budget must be a positive integer")
    frame = model.frame
    if set(initial_values) != set(frame.initial_values):
        raise ValueError("analysis requires the complete original coordinate state")
    fixed = {key: float(value) for key, value in initial_values.items()}
    frame.validate_values(fixed, start_time_s)
    names = tuple(
        s
        for s, declaration in frame._species.items()
        if not declaration.get("boundaryCondition") and not declaration.get("constant")
    )
    if not names or set(amount_scales_mol) != set(names):
        raise ValueError("explicit mol metric must cover exactly all dynamic species")
    if any(isinstance(x, bool) or not isfinite(x) or x <= 0 for x in amount_scales_mol.values()):
        raise ValueError("perturbation amount scales must be finite positive mol")
    metric = np.asarray([amount_scales_mol[s] for s in names])
    keys = tuple(fixed)
    indices = [keys.index("N:" + s) for s in names]
    full = _stoichiometry(frame, fixed, start_time_s)
    all_names = tuple(frame._species)
    stoichiometry = full[[all_names.index(s) for s in names], :]
    _, pivots = _rref(
        [[Fraction(float(v)) for v in row] for row in stoichiometry],
        width=stoichiometry.shape[1],
        max_integer_bits=max_integer_bits,
    )
    rank = len(pivots)
    if rank == 0 or rank > maximum_directions:
        raise ValueError("reachable tangent dimension is zero or exceeds its explicit budget")
    weighted = stoichiometry / metric[:, None]
    if not np.all(np.isfinite(weighted)):
        raise ValueError("declared metric cannot represent the stoichiometric matrix")
    vectors, singular, _ = np.linalg.svd(weighted, full_matrices=True)
    rank_floor = 64 * max(weighted.shape) * np.finfo(float).eps * singular[0]
    if singular[rank - 1] <= rank_floor:
        raise ValueError("exact stoichiometric rank is unresolved in the declared floating metric")
    basis, conservation = vectors[:, :rank], vectors[:, rank:].T
    declared_scales = cast(Mapping[str, float], model.coordinate_scales)
    numerical_scales = np.asarray(
        [
            amount_scales_mol[key[2:]]
            if key in {"N:" + s for s in names}
            else declared_scales.get(key, max(abs(fixed[key]), 1e-6))
            for key in keys
        ]
    )
    n = len(keys)
    initial = np.asarray([fixed[key] for key in keys])
    initial_totals = conservation @ (initial[indices] / metric)
    directions = np.zeros((n, rank))
    directions[indices] = metric[:, None] * basis
    packed = np.r_[initial / numerical_scales, (directions / numerical_scales[:, None]).ravel()]
    policy = ODESolverPolicy(
        "chemworld_sbml_finite_time_v1",
        model.solver_method,
        rtol=model.rtol,
        atol=model.atol,
        max_step_s=model.maximum_step_s,
    )
    rhs_calls = 0

    def unpack(vector: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        return dict(zip(keys, vector[:n] * numerical_scales, strict=True)), (
            vector[n:].reshape(n, rank) * numerical_scales[:, None]
        )

    def validate(vector: np.ndarray, time_s: float) -> None:
        values, slopes = unpack(vector)
        frame.validate_values(values, time_s)
        if any(values["N:" + s] <= 0 for s in names):
            raise ValueError("finite-time two-sided perturbations require a strictly positive path")
        if not np.all(np.isfinite(slopes)):
            raise ValueError("tangent trajectory is not finite")

    validate(packed, start_time_s)

    def rhs(time_s: float, vector: np.ndarray) -> np.ndarray:
        nonlocal rhs_calls
        rhs_calls += 1
        values, slopes = unpack(vector)
        primal, tangent = sbml_variational_rates(frame, values, keys, slopes, time_s)
        return np.r_[primal / numerical_scales, (tangent / numerical_scales[:, None]).ravel()]

    records: list[dict[str, Any]] = []
    solvers: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "schema": "sbml-finite-time-stability-v1",
        "status": "complete",
        "failure": None,
        "start_time_s": start_time_s,
        "requested_offsets_s": list(offsets),
        "species_ids": list(names),
        "amount_scales_mol": dict(amount_scales_mol),
        "metric": "Euclidean norm of delta_n / declared fixed mol scales",
        "basis_scaled_amounts": basis.tolist(),
        "reachable_dimension": rank,
        "rank_method": "exact binary-rational stoichiometry; floating metric checked separately",
        "stoichiometric_singular_values": singular.tolist(),
        "conservation_rows_scaled_amounts": conservation.tolist(),
        "initial_conserved_totals": initial_totals.tolist(),
        "initial_coordinates": fixed,
        "records": records,
        "solvers": solvers,
        "interpretation": "singular growth from initial time; no asymptotic Lyapunov/chaos claim",
        "controls": (
            "held current parameters, volumes and boundary pools; future controls not predicted"
        ),
        "energy_scope": "undeclared formal kinetics; no physical heat calculation",
        "world_advanced": False,
    }
    previous = start_time_s
    try:
        for offset in offsets:
            target = start_time_s + offset
            solved = solve_ode(rhs, packed, time_span_s=(previous, target), policy=policy)
            solvers.append(solved.diagnostic.to_dict())
            if not solved.success:
                raise ValueError("finite-time tangent solve failed: " + solved.message)
            for j, node_time in enumerate(solved.t):
                validate(solved.y[:, j], float(node_time))
            packed = solved.y[:, -1]
            values, slopes = unpack(packed)
            scaled_slopes = slopes[indices] / metric[:, None]
            transfer = basis.T @ scaled_slopes
            growth = np.linalg.svd(transfer, compute_uv=False)
            # This is an explicit numerical resolution estimate, not a bound
            # on integration error or a physical/dynamic applicability rule.
            cutoff = max(
                64 * len(names) * np.finfo(float).eps * max(1.0, float(growth[0])),
                10 * np.sqrt(len(names) * rank) * model.atol,
            )
            resolved = growth > cutoff
            current = np.asarray([values["N:" + s] for s in names]) / metric
            records.append(
                {
                    "offset_s": offset,
                    "time_s": target,
                    "coordinates": values,
                    "reduced_tangent_map": transfer.tolist(),
                    "singular_growth_factors": growth.tolist(),
                    "finite_time_rates_s_inverse": [
                        float(np.log(v) / offset) if ok else None
                        for v, ok in zip(growth, resolved, strict=True)
                    ],
                    "resolved_singular_values": resolved.tolist(),
                    "singular_resolution_estimate": cutoff,
                    "maximum_amplification": float(growth[0]) if resolved[0] else None,
                    "conserved_totals": (conservation @ current).tolist(),
                    "conservation_error": float(
                        np.max(np.abs(conservation @ current - initial_totals), initial=0)
                    ),
                    "tangent_conservation_error": float(
                        np.max(np.abs(conservation @ scaled_slopes), initial=0)
                    ),
                    "conserved_zero_modes_excluded": len(names) - rank,
                }
            )
            previous = target
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as error:
        report.update(status="failed", failure=f"{type(error).__name__}: {error}")
    report.update(
        completed_windows=len(records),
        requested_windows=len(offsets),
        rhs_evaluations=rhs_calls,
        solver_reported_rhs_evaluations=sum(int(d["nfev"]) for d in solvers),
    )
    return report
