"""Instantaneous reachable modes along an original SBML trajectory."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import pairwise
from math import isfinite
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import scalar_value
from chemworld.physchem.sbml_analysis_geometry import sbml_analysis_geometry
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


def sbml_trajectory_timescales(
    model: SBMLSystemSpec,
    current_values: Mapping[str, float],
    *,
    start_time_s: float,
    sample_offsets_s: Sequence[float],
    amount_scales_mol: Mapping[str, float],
    separation_ratio: float = 10.0,
    zero_rate_tolerance_s_inverse: float = 1e-9,
    maximum_eigenvector_condition: float = 1e8,
    maximum_samples: int = 10000,
    maximum_rhs_evaluations: int = 1000000,
    maximum_directions: int = 32,
    max_integer_bits: int = 8192,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Classify decay, growth and oscillation; never assert ILDM/CSP reduction.

    Gaps describe local stable decay rates only. Eigenvector conditioning and
    nonnormality are exposed; separated eigenvalues do not bound transient gain.
    """
    offsets = tuple(sample_offsets_s)
    if (
        not offsets
        or any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in offsets)
        or any(b <= a for a, b in pairwise(offsets))
    ):
        raise ValueError("sample offsets must be finite nonnegative and strictly increasing")
    if (
        any(type(v) is not int or v < 1 for v in (maximum_samples, maximum_rhs_evaluations))
        or len(offsets) > maximum_samples
    ):
        raise ValueError("sample and RHS budgets must be positive and not exceeded")
    if any(
        isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (zero_rate_tolerance_s_inverse, maximum_eigenvector_condition)
    ) or (
        isinstance(separation_ratio, bool)
        or not isfinite(separation_ratio)
        or separation_ratio <= 1
    ):
        raise ValueError("modal tolerances must be positive; separation ratio must exceed one")
    g = sbml_analysis_geometry(
        model,
        current_values,
        start_time_s=start_time_s,
        amount_scales_mol=amount_scales_mol,
        maximum_directions=maximum_directions,
        max_integer_bits=max_integer_bits,
    )
    targets = [start_time_s + t for t in offsets]
    if (
        any(not isfinite(t) for t in targets)
        or any(b <= a for a, b in pairwise(targets))
        or (offsets[0] > 0 and targets[0] <= start_time_s)
    ):
        raise ValueError("sample times cannot be represented at the current time")
    packed = np.asarray([g.initial[k] for k in g.keys]) / g.numerical_scales
    policy = ODESolverPolicy(
        "chemworld_sbml_modal_timescales_v1",
        model.solver_method,
        rtol=model.rtol,
        atol=model.atol,
        max_step_s=model.maximum_step_s,
    )
    calls = 0
    directional_calls = 0

    def rhs(t: float, vector: np.ndarray) -> np.ndarray:
        nonlocal calls
        if calls >= maximum_rhs_evaluations:
            raise ValueError("timescale analysis RHS evaluation budget exhausted")
        calls += 1
        rates = model.frame.rates(g.values(vector), t).coordinate_rates
        return np.asarray([scalar_value(rates[k]) for k in g.keys]) / g.numerical_scales

    rows: list[dict[str, Any]] = []
    solvers: list[dict[str, Any]] = []
    report = {
        "schema": "sbml-trajectory-timescales-v1",
        **g.describe(),
        "start_time_s": start_time_s,
        "requested_offsets_s": list(offsets),
        "status": "partial",
        "failure": None,
        "records": rows,
        "solvers": solvers,
        "separation_ratio": separation_ratio,
        "zero_rate_tolerance_s_inverse": zero_rate_tolerance_s_inverse,
        "maximum_eigenvector_condition": maximum_eigenvector_condition,
        "interpretation": (
            "local linear decay/growth and oscillation along a trajectory; "
            "not ILDM/CSP or finite-time gain"
        ),
    }
    initial_totals = g.totals(g.initial)
    previous = start_time_s
    try:
        for target in targets:
            if target > previous:
                solved = solve_ode(rhs, packed, time_span_s=(previous, target), policy=policy)
                solvers.append(solved.diagnostic.to_dict())
                if not solved.success:
                    raise ValueError("original trajectory ODE failed: " + solved.message)
                for i, t in enumerate(solved.t):
                    g.validate(g.values(solved.y[:, i]), float(t))
                packed = solved.y[:, -1]
            values = g.values(packed)
            directional_calls += 1
            jacobian = g.reduced_jacobian(values, target)
            eigenvalues, vectors = np.linalg.eig(jacobian)
            if not np.all(np.isfinite(eigenvalues)):
                raise ValueError("nonfinite reachable spectrum")
            order = np.lexsort((eigenvalues.imag, eigenvalues.real))
            eigenvalues, vectors = eigenvalues[order], vectors[:, order]
            condition = float(np.linalg.cond(vectors))
            residual = float(
                np.linalg.norm(jacobian @ vectors - vectors * eigenvalues)
                / max(1.0, float(np.linalg.norm(jacobian)))
            )
            resolved = (
                isfinite(condition)
                and condition <= maximum_eigenvector_condition
                and residual <= 1e-7
            )
            modes: list[dict[str, Any]] = []
            for value in eigenvalues:
                real, imaginary = float(value.real), float(value.imag)
                status = (
                    "decay"
                    if real < -zero_rate_tolerance_s_inverse
                    else "growth"
                    if real > zero_rate_tolerance_s_inverse
                    else "neutral_or_unresolved"
                )
                modes.append(
                    {
                        "real_s_inverse": real,
                        "imaginary_rad_s": imaginary,
                        "status": status,
                        "decay_time_s": -1 / real if status == "decay" else None,
                        "growth_time_s": 1 / real if status == "growth" else None,
                        "oscillation_period_s": 2 * np.pi / abs(imaginary)
                        if abs(imaginary) > zero_rate_tolerance_s_inverse
                        else None,
                    }
                )
            decay = sorted(
                (-m["real_s_inverse"], i) for i, m in enumerate(modes) if m["status"] == "decay"
            )
            gaps = []
            for cut in range(1, len(decay)):
                ratio = decay[cut][0] / decay[cut - 1][0]
                if ratio >= separation_ratio:
                    gaps.append(
                        {
                            "ratio": ratio,
                            "slow_mode_indices": [i for _, i in decay[:cut]],
                            "fast_mode_indices": [i for _, i in decay[cut:]],
                            "resolved": resolved,
                        }
                    )
            rows.append(
                {
                    "time_s": target,
                    "offset_s": target - start_time_s,
                    "coordinates": values,
                    "reduced_jacobian_s_inverse": jacobian.tolist(),
                    "modes": modes,
                    "eigenvector_condition": condition if isfinite(condition) else None,
                    "eigenpair_relative_residual": residual,
                    "modal_basis_resolved": resolved,
                    "nonnormal_commutator_norm": float(
                        np.linalg.norm(jacobian.T @ jacobian - jacobian @ jacobian.T)
                    ),
                    "stable_decay_gaps": gaps,
                    "has_resolved_fast_slow_separation": bool(gaps) and resolved,
                    "conservation_error": float(
                        np.max(np.abs(g.totals(values) - initial_totals), initial=0)
                    ),
                }
            )
            previous = target
            if progress:
                progress(
                    {
                        "stage": "sbml_trajectory_timescales",
                        "completed": len(rows),
                        "total": len(targets),
                        "rhs_evaluations": calls,
                    }
                )
        report["status"] = "complete"
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    report.update(
        completed_samples=len(rows),
        requested_samples=len(offsets),
        rhs_evaluations=calls,
        directional_rate_evaluations=directional_calls,
        solver_reported_rhs_evaluations=sum(int(d["nfev"]) for d in solvers),
    )
    return report
