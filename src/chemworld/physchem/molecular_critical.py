"""Local criticality and full composition responses of the original Helmholtz model.

These read-only calculations neither allocate phases nor prove global stability.
Exact zero components select a composition face, not an insertion limit. Linear
ideal reference terms cancel from every density derivative used here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from math import exp, isfinite
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.caloric_math import derivative_values, logarithm, partial_derivative
from chemworld.physchem.molecular_eos import ResidualHelmholtzModel
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    finite_jacobian,
    solve_root,
)

CRITICAL_ROOT_POLICY = RootPolicy(2e-9, 60)
FloatArray = NDArray[np.float64]


def _inputs(
    model: ResidualHelmholtzModel, temperature: float, densities: Sequence[float] | FloatArray
) -> tuple[Any, ...]:
    c = np.asarray(densities, dtype=float)
    if (
        c.shape != (len(model.component_ids),)
        or not np.all(np.isfinite(c))
        or np.any(c < 0)
        or c.sum() <= 0
        or not isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("critical response needs positive T and nonnegative nonempty densities")
    rho = float(c.sum())
    active = np.flatnonzero(c > 0)
    x = c / rho
    model.alphar(temperature, rho, tuple(x))
    return c, rho, active, x


def _functional(
    model: ResidualHelmholtzModel, temperature: float, rho: float, active: Any
) -> Callable[[Sequence[Any]], Any]:
    # Density scale is held fixed while taking derivatives. Absent fractions
    # remain literal zero, avoiding artificial high-order insertion jets in GERG.
    def energy(values: Sequence[Any]) -> Any:
        total = sum(values)
        fractions = [0.0] * len(model.component_ids)
        for i, value in zip(active, values, strict=True):
            fractions[i] = value / total
        return sum(value * (logarithm(value) - 1) for value in values) + total * model.alphar(
            temperature, rho * total, tuple(fractions)
        )

    return energy


def _curvature(
    model: ResidualHelmholtzModel,
    temperature: float,
    densities: Sequence[float] | FloatArray,
    *,
    fourth: bool = False,
) -> tuple[Any, ...]:
    _, rho, active, x = _inputs(model, temperature, densities)
    energy = _functional(model, temperature, rho, active)
    values = {str(i): float(x[k]) for i, k in enumerate(active)}

    def mapping(v: dict[str, Any]) -> Any:
        return energy(tuple(v[str(i)] for i in range(len(active))))

    h = np.array(
        [
            [partial_derivative(mapping, values, (str(i), str(j))) for j in range(len(active))]
            for i in range(len(active))
        ]
    )
    eigenvalues, vectors = np.linalg.eigh(h)
    mode = vectors[:, 0]
    if mode[int(np.argmax(np.abs(mode)))] < 0:
        mode = -mode
    base = x[active]
    directional = derivative_values(
        lambda s: energy(tuple(v + s * d for v, d in zip(base, mode, strict=True))),
        0.0,
        order=4 if fourth else 3,
    )
    raw4, relaxed4 = None, None
    if fourth:
        raw4 = directional[4]
        if len(active) == 1:
            relaxed4 = raw4
        elif eigenvalues[1] > 1e-8:
            couplings = []
            for transverse in vectors[:, 1:].T:
                couplings.append(
                    partial_derivative(
                        lambda v, transverse=transverse: energy(
                            tuple(
                                a + v["s"] * b + v["y"] * d
                                for a, b, d in zip(base, mode, transverse, strict=True)
                            )
                        ),
                        {"s": 0.0, "y": 0.0},
                        ("s", "s", "y"),
                    )
                )
            relaxed4 = raw4 - 3 * float(np.sum(np.square(couplings) / eigenvalues[1:]))
    return rho, active, x, h, eigenvalues, mode, directional[3], raw4, relaxed4


def molecular_composition_response(
    model: ResidualHelmholtzModel,
    temperature_K: float,
    component_densities_mol_m3: Sequence[float] | FloatArray,
    *,
    singular_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Full d(mu)/d(component density), active-face T/rho and T/P responses.

    Simplex columns replace the last active component; returned chemical
    potentials are differences against it. Inverses require positive curvature.
    """
    if not isfinite(singular_tolerance) or singular_tolerance <= 0:
        raise ValueError("response singular tolerance must be finite and positive")
    rho, active, x, h, eigen, mode, third, raw4, relaxed4 = _curvature(
        model, temperature_K, component_densities_mol_m3, fourth=True
    )
    rt = model.gas_constant_J_mol_K * temperature_K
    pressure = (
        rho
        * rt
        * (
            1
            + derivative_values(
                lambda s: model.alphar(temperature_K, rho * (1 + s), tuple(x)), 0.0
            )[1]
        )
    )
    if not isfinite(pressure):
        raise ValueError("critical pressure is not finite")
    # The existing multifluid source pressure ceiling also applies at singular
    # critical states, where molecular_eos_state's finite-Cp contract cannot apply.
    validator = getattr(model, "validate_pressure", None)
    if validator is not None:
        validator(pressure)
    n = len(active)
    basis = np.vstack((np.eye(n - 1), -np.ones(n - 1))) if n > 1 else np.zeros((1, 0))
    xa = x[active]
    stiffness = float(xa @ h @ xa)
    fixed_rho = basis.T @ h @ basis
    fixed_p, density_response = None, None
    if abs(stiffness) > singular_tolerance:
        hx = h @ xa
        fixed_p = basis.T @ (h - np.outer(hx, hx) / stiffness) @ basis
        density_response = -rho * (xa @ h @ basis) / stiffness
    positive = bool(eigen[0] > singular_tolerance)
    return {
        "temperature_K": temperature_K,
        "molar_density_mol_m3": rho,
        "mole_fractions": x.tolist(),
        "pressure_Pa": pressure,
        "active_component_ids": [model.component_ids[i] for i in active],
        "dependent_component": model.component_ids[active[-1]],
        "density_hessian_J_m3_mol2": (h * rt / rho).tolist(),
        "scaled_density_hessian": h.tolist(),
        "scaled_eigenvalues": eigen.tolist(),
        "critical_mode": mode.tolist(),
        "third_mode_derivative_scaled": third,
        "fourth_mode_derivative_scaled": raw4,
        "relaxed_fourth_derivative_scaled": relaxed4,
        "dP_drho_T_Pa_m3_mol": rt * stiffness,
        "dP_dcomponent_density_Pa_m3_mol": (rt * h @ xa).tolist(),
        "simplex_fixed_Trho_mu_difference_J_mol": (rt * fixed_rho).tolist(),
        "simplex_fixed_TP_mu_difference_J_mol": None
        if fixed_p is None
        else (rt * fixed_p).tolist(),
        "fixed_TP_density_response_mol_m3": None
        if density_response is None
        else density_response.tolist(),
        "density_susceptibility_mol2_J_m3": (np.linalg.inv(h) * rho / rt).tolist()
        if positive
        else None,
        "locally_convex": positive,
        "singular_or_unstable": not positive,
        "scope": "local_active_face; no absent insertion or global phase-stability claim",
    }


@dataclass(frozen=True)
class MolecularCriticalResult:
    success: bool
    response: dict[str, Any] | None = None
    failure_reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "scope": "local_non_degenerate_active_face_critical_point"}


def _bounds(
    temperature_bounds: tuple[float, float], density_bounds: tuple[float, float]
) -> tuple[Any, ...]:
    pairs = (tuple(temperature_bounds), tuple(density_bounds))
    if any(len(p) != 2 or not all(isfinite(v) and v > 0 for v in p) or p[0] >= p[1] for p in pairs):
        raise ValueError("critical root needs finite positive ordered T/rho bounds")
    lower = np.log([p[0] for p in pairs])
    upper = np.log([p[1] for p in pairs])
    variables = tuple(
        ResidualVariable(k, "1", 1, lower=a, upper=b)
        for k, a, b in zip(("log_T", "log_rho"), lower, upper, strict=True)
    )
    return lower, upper, variables


def solve_molecular_critical(
    model: ResidualHelmholtzModel,
    mole_fractions: Sequence[float],
    *,
    initial_temperature_K: float,
    initial_density_mol_m3: float,
    temperature_bounds_K: tuple[float, float],
    density_bounds_mol_m3: tuple[float, float],
    policy: RootPolicy = CRITICAL_ROOT_POLICY,
    progress: Callable | None = None,
) -> MolecularCriticalResult:
    """Fixed-composition HK root; a successful root still requires stable fourth order."""
    x = np.asarray(mole_fractions, dtype=float)
    _inputs(model, initial_temperature_K, x * initial_density_mol_m3)
    if abs(float(x.sum()) - 1) > 1e-12:
        raise ValueError("critical composition must sum to one without normalization")
    low, high, variables = _bounds(temperature_bounds_K, density_bounds_mol_m3)
    calls = 0
    last: dict[str, Any] = {}

    def residual(y: FloatArray) -> FloatArray:
        nonlocal calls
        calls += 1
        try:
            values = _curvature(model, exp(y[0]), exp(y[1]) * x)
            result = np.array([values[4][0], values[6]])
            last.update(temperature_K=exp(y[0]), density_mol_m3=exp(y[1]), residual=result.tolist())
            if progress:
                progress(calls)
            return result
        except (ArithmeticError, ValueError) as error:
            raise ResidualDomainError(str(error)) from error

    # Preserve a productive supplied seed. Reserve part of the same total
    # iteration budget for one finite residual-only restart if it stalls.
    initial = np.log([initial_temperature_K, initial_density_mol_m3])
    initialization: list[dict[str, Any]] = []
    selected = initial.copy()

    def run(seed: FloatArray, limit: int) -> Any:
        return solve_root(
            residual,
            seed,
            jacobian=lambda y: finite_jacobian(residual, y, [1, 1], lower=low, upper=high),
            variables=variables,
            residual_scales=[1, 1],
            policy=RootPolicy(policy.tolerance, limit, policy.maximum_backtracks),
        )

    root = run(initial, min(20, policy.maximum_iterations))
    attempts = [
        {"success": root.success, "failure_reason": root.failure_reason, **root.diagnostics}
    ]
    remaining = policy.maximum_iterations - root.diagnostics["iterations"]
    if not root.success and root.failure_reason != "initial_outside_bounds" and remaining > 0:
        best = float("inf")
        for ft in (1.0, 0.9, 1.1, 1.2):
            for fr in (1.0, 0.8, 1.2, 1.4):
                candidate = initial + np.log([ft, fr])
                if np.any(candidate < low) or np.any(candidate > high):
                    initialization.append(
                        {"factors": [ft, fr], "failure": "outside_declared_bounds"}
                    )
                    continue
                try:
                    value = residual(candidate)
                    merit = float(value @ value)
                    initialization.append(
                        {"factors": [ft, fr], "residual": value.tolist(), "merit": merit}
                    )
                    if merit < best:
                        selected, best = candidate, merit
                except ResidualDomainError as error:
                    initialization.append({"factors": [ft, fr], "failure": str(error)})
                if best <= policy.tolerance**2:
                    break
            if best <= policy.tolerance**2:
                break
        root = run(selected, remaining)
        attempts.append(
            {"success": root.success, "failure_reason": root.failure_reason, **root.diagnostics}
        )
    diagnostics = {
        "root": root.diagnostics,
        "root_attempts": attempts,
        "root_iterations_total": sum(a["iterations"] for a in attempts),
        "root_iteration_budget": policy.maximum_iterations,
        "initialization": initialization,
        "selected_initial_temperature_K": exp(selected[0]),
        "selected_initial_density_mol_m3": exp(selected[1]),
        "actual_condition_evaluations": calls,
        "last_attempt": last,
        "derivatives": "analytic Hessian/third/fourth; numerical root Jacobian",
    }
    if not root.success:
        return MolecularCriticalResult(
            False, failure_reason=root.failure_reason, diagnostics=diagnostics
        )
    try:
        response = molecular_composition_response(model, exp(root.value[0]), exp(root.value[1]) * x)
    except (ArithmeticError, ValueError) as error:
        return MolecularCriticalResult(False, failure_reason=str(error), diagnostics=diagnostics)
    transverse = response["scaled_eigenvalues"][1:]
    reason = (
        "nonpositive_critical_pressure"
        if response["pressure_Pa"] <= 0
        else "degenerate_or_unstable_transverse_mode"
        if transverse and min(transverse) <= 1e-8
        else "nonpositive_relaxed_fourth_order"
        if response["relaxed_fourth_derivative_scaled"] is None
        or response["relaxed_fourth_derivative_scaled"] <= 1e-8
        else ""
    )
    return MolecularCriticalResult(not reason, response, reason, diagnostics)


def trace_molecular_critical(
    model: ResidualHelmholtzModel,
    compositions: Sequence[Sequence[float]],
    *,
    initial_temperature_K: float,
    initial_density_mol_m3: float,
    temperature_bounds_K: tuple[float, float],
    density_bounds_mol_m3: tuple[float, float],
    policy: RootPolicy = CRITICAL_ROOT_POLICY,
    progress: Callable | None = None,
) -> dict[str, Any]:
    """Declared composition-path continuation; failures break the local curve.

    A secant predicts log T/rho, then the original HK conditions correct each
    station. This is not arclength continuation through composition folds or a
    global envelope. A failed station is retained and never bridged.
    """
    rows: list[dict[str, Any]] = []
    for index, x in enumerate(compositions):
        seed = np.log([initial_temperature_K, initial_density_mol_m3])
        if rows:
            previous = rows[-1]["result"]["response"]
            seed = np.log([previous["temperature_K"], previous["molar_density_mol_m3"]])
        if len(rows) >= 2:
            before = rows[-2]["result"]["response"]
            dx = np.asarray(x) - np.asarray(rows[-1]["mole_fractions"])
            prior_dx = np.asarray(rows[-1]["mole_fractions"]) - np.asarray(
                rows[-2]["mole_fractions"]
            )
            denom = float(prior_dx @ prior_dx)
            if denom:
                seed += (
                    float(dx @ prior_dx)
                    / denom
                    * (seed - np.log([before["temperature_K"], before["molar_density_mol_m3"]]))
                )
        result = solve_molecular_critical(
            model,
            x,
            initial_temperature_K=exp(seed[0]),
            initial_density_mol_m3=exp(seed[1]),
            temperature_bounds_K=temperature_bounds_K,
            density_bounds_mol_m3=density_bounds_mol_m3,
            policy=policy,
            progress=progress,
        )
        rows.append({"index": index, "mole_fractions": list(x), "result": result.to_dict()})
        if not result.success:
            break
    return {
        "success": bool(rows)
        and len(rows) == len(compositions)
        and all(r["result"]["success"] for r in rows),
        "requested_stations": len(compositions),
        "completed_stations": sum(r["result"]["success"] for r in rows),
        "stations": rows,
        "scope": "local_declared_composition_path; not global or fold-complete",
    }
