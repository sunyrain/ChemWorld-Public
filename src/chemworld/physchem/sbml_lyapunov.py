"""Periodic QR of original SBML variational dynamics, in a declared mol metric."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import ceil, isfinite
from typing import Any

import numpy as np

from chemworld.physchem.sbml_analysis_geometry import sbml_analysis_geometry
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.sbml_variational import sbml_variational_rates
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


def sbml_lyapunov_spectrum(
    model: SBMLSystemSpec,
    current_values: Mapping[str, float],
    *,
    start_time_s: float,
    qr_interval_s: float,
    measurement_duration_s: float,
    amount_scales_mol: Mapping[str, float],
    transient_duration_s: float = 0.0,
    maximum_intervals: int = 10000,
    maximum_rhs_evaluations: int = 1000000,
    maximum_directions: int = 32,
    max_integer_bits: int = 8192,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Estimate a Lyapunov spectrum with finite time, starts and numerical error.

    Warmup QR rotates the tangent frame without accumulating growth. Subsequent
    log diagonal growth is divided by actual measurement seconds. No base-state
    reset, singular-value clipping, trajectory replacement or chaos certificate.
    """
    if any(
        isinstance(v, bool) or not isfinite(v) or v <= 0
        for v in (qr_interval_s, measurement_duration_s)
    ) or (
        isinstance(transient_duration_s, bool)
        or not isfinite(transient_duration_s)
        or transient_duration_s < 0
    ):
        raise ValueError(
            "QR interval and measurement duration must be positive; transient nonnegative"
        )
    if any(type(v) is not int or v < 1 for v in (maximum_intervals, maximum_rhs_evaluations)):
        raise ValueError("QR and RHS budgets must be positive integers")
    g = sbml_analysis_geometry(
        model,
        current_values,
        start_time_s=start_time_s,
        amount_scales_mol=amount_scales_mol,
        maximum_directions=maximum_directions,
        max_integer_bits=max_integer_bits,
    )
    end_time = start_time_s + transient_duration_s + measurement_duration_s
    if not isfinite(end_time) or end_time <= start_time_s:
        raise ValueError("declared analysis duration cannot be represented at the current time")
    # Materialize no more than the declared budget, including both endpoints.
    requested = ceil(transient_duration_s / qr_interval_s) + ceil(
        measurement_duration_s / qr_interval_s
    )
    if requested > maximum_intervals:
        raise ValueError("declared QR interval count exceeds its explicit budget")
    schedule = []
    phase_start = start_time_s
    for phase, duration in (
        ("transient", transient_duration_s),
        ("measurement", measurement_duration_s),
    ):
        for i in range(ceil(duration / qr_interval_s)):
            schedule.append((phase, phase_start + min((i + 1) * qr_interval_s, duration)))
        phase_start += duration
    if any(
        b[1] <= a[1] for a, b in zip([("initial", start_time_s), *schedule], schedule, strict=False)
    ):
        raise ValueError("QR intervals cannot be represented at the declared absolute time")
    n, rank = len(g.keys), g.rank
    packed = np.r_[
        np.asarray([g.initial[k] for k in g.keys]) / g.numerical_scales,
        (g.directions() / g.numerical_scales[:, None]).ravel(),
    ]
    policy = ODESolverPolicy(
        "chemworld_sbml_qr_lyapunov_v1",
        model.solver_method,
        rtol=model.rtol,
        atol=model.atol,
        max_step_s=model.maximum_step_s,
    )
    calls = 0

    def unpack(vector: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        return g.values(vector[:n]), vector[n:].reshape(n, rank) * g.numerical_scales[:, None]

    def rhs(t: float, vector: np.ndarray) -> np.ndarray:
        nonlocal calls
        if calls >= maximum_rhs_evaluations:
            raise ValueError("QR analysis RHS evaluation budget exhausted")
        calls += 1
        values, tangent = unpack(vector)
        primal, derivative = sbml_variational_rates(model.frame, values, g.keys, tangent, t)
        return np.r_[
            primal / g.numerical_scales, (derivative / g.numerical_scales[:, None]).ravel()
        ]

    records: list[dict[str, Any]] = []
    solvers: list[dict[str, Any]] = []
    report = {
        "schema": "sbml-qr-lyapunov-v1",
        **g.describe(),
        "start_time_s": start_time_s,
        "transient_duration_s": transient_duration_s,
        "measurement_duration_s": measurement_duration_s,
        "qr_interval_s": qr_interval_s,
        "requested_intervals": requested,
        "status": "partial",
        "failure": None,
        "records": records,
        "solvers": solvers,
        "interpretation": (
            "finite-duration QR Lyapunov estimates; no infinite-time convergence or chaos proof"
        ),
    }
    logs = np.zeros(rank)
    elapsed = 0.0
    previous = start_time_s
    initial_totals = g.totals(g.initial)
    try:
        for phase, target in schedule:
            solved = solve_ode(rhs, packed, time_span_s=(previous, target), policy=policy)
            solvers.append(solved.diagnostic.to_dict())
            if not solved.success:
                raise ValueError("original variational ODE failed: " + solved.message)
            for i, t in enumerate(solved.t):
                values, tangent = unpack(solved.y[:, i])
                g.validate(values, float(t))
                if not np.all(np.isfinite(tangent)):
                    raise ValueError("nonfinite variational trajectory")
            packed = solved.y[:, -1].copy()
            values, tangent = unpack(packed)
            weighted = tangent[g.indices] / g.metric[:, None]
            transfer = g.basis.T @ weighted
            q, r = np.linalg.qr(transfer)
            diagonal = np.diag(r)
            resolution = max(
                64 * rank * np.finfo(float).eps * np.linalg.norm(transfer, 2),
                10 * np.sqrt(rank) * model.atol,
            )
            if np.any(np.abs(diagonal) <= resolution):
                raise ValueError(
                    "QR growth unresolved at declared interval/tolerance; no clipped logarithms"
                )
            signs = np.where(diagonal < 0, -1.0, 1.0)
            q, r = q * signs, signs[:, None] * r
            increments = np.log(np.diag(r))
            duration = target - previous
            if phase == "measurement":
                logs += increments
                elapsed += duration
            row = {
                "phase": phase,
                "time_s": target,
                "duration_s": duration,
                "coordinates": values,
                "log_growth": increments.tolist(),
                "interval_rates_s_inverse": (increments / duration).tolist(),
                "measurement_elapsed_s": elapsed,
                "running_exponents_s_inverse": (logs / elapsed).tolist() if elapsed else None,
                "qr_diagonal": np.diag(r).tolist(),
                "qr_resolution_estimate": float(resolution),
                "orthogonality_error": float(np.linalg.norm(q.T @ q - np.eye(rank))),
                "tangent_conservation_error": float(
                    np.max(np.abs(g.conservation @ weighted), initial=0)
                ),
                "conservation_error": float(
                    np.max(np.abs(g.totals(values) - initial_totals), initial=0)
                ),
            }
            records.append(row)
            restarted = np.zeros((n, rank))
            restarted[g.indices] = g.metric[:, None] * (g.basis @ q)
            packed[n:] = (restarted / g.numerical_scales[:, None]).ravel()
            previous = target
            if progress:
                progress(
                    {
                        "stage": "sbml_qr_lyapunov",
                        "completed": len(records),
                        "total": requested,
                        "rhs_evaluations": calls,
                    }
                )
        report["status"] = "complete"
    except (ValueError, ArithmeticError, RuntimeError, np.linalg.LinAlgError) as error:
        report["failure"] = f"{type(error).__name__}: {error}"
    measured = [row for row in records if row["phase"] == "measurement"]
    tail = measured[len(measured) // 2 :]
    tail_elapsed = sum(row["duration_s"] for row in tail)
    tail_exponents = (
        np.sum([row["log_growth"] for row in tail], axis=0) / tail_elapsed if tail_elapsed else None
    )
    report.update(
        completed_intervals=len(records),
        measurement_elapsed_s=elapsed,
        rhs_evaluations=calls,
        exponents_s_inverse=(logs / elapsed).tolist() if elapsed else None,
        last_half_exponents_s_inverse=tail_exponents.tolist()
        if tail_exponents is not None
        else None,
        last_half_vs_all_drift_s_inverse=(tail_exponents - logs / elapsed).tolist()
        if tail_exponents is not None
        else None,
        solver_reported_rhs_evaluations=sum(int(d["nfev"]) for d in solvers),
    )
    return report
