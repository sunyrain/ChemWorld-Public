"""One slow finite-mineral coordinate with fast aqueous closure per candidate."""

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from chemworld.physchem import solver_backend
from chemworld.physchem.aqueous_mineral import (
    CarbonateEquilibriumResult,
    element_amounts,
    solve_carbonate_equilibrium,
)
from chemworld.physchem.solver_backend import ODESolverPolicy


@dataclass(frozen=True)
class MineralKineticsSpec:
    model_id: str = "synthetic-finite-calcite-kinetics-v1"
    rate_constant_mol_m2_s: float = 1e-6
    saturation_tolerance: float = 1e-9

    def __post_init__(self):
        if (
            not self.model_id
            or not isfinite(self.rate_constant_mol_m2_s)
            or self.rate_constant_mol_m2_s < 0
            or not isfinite(self.saturation_tolerance)
            or not 0 < self.saturation_tolerance <= 1e-9
        ):
            raise ValueError("finite mineral kinetics requires bounded nonnegative rate data")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


@dataclass(frozen=True)
class MineralGeometry:
    reference_area_m2: float = 0.01
    reference_amount_mol: float = 0.01

    def __post_init__(self):
        if (
            not isfinite(self.reference_area_m2)
            or self.reference_area_m2 < 0
            or not isfinite(self.reference_amount_mol)
            or self.reference_amount_mol < 0
            or (self.reference_area_m2 > 0 and self.reference_amount_mol == 0)
        ):
            raise ValueError("mineral reference geometry must be finite and material-owned")

    def area(self, amount_mol):
        if not isfinite(amount_mol) or amount_mol < 0:
            raise ValueError("mineral surface requires finite nonnegative actual solid")
        if amount_mol == 0 or self.reference_area_m2 == 0:
            return 0.0
        return self.reference_area_m2 * (amount_mol / self.reference_amount_mol) ** (2 / 3)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MineralKineticPoint:
    time_s: float
    aqueous: CarbonateEquilibriumResult
    area_m2: float

    def to_dict(self):
        return {"time_s": self.time_s, "aqueous": self.aqueous.to_dict(), "area_m2": self.area_m2}


@dataclass(frozen=True)
class MineralAdvanceReport:
    trajectory: tuple[MineralKineticPoint, ...]
    actual_duration_s: float
    chemical_stop_time_s: float | None
    chemical_stop_reason: str | None
    diagnostics: dict

    @property
    def final_aqueous(self):
        return self.trajectory[-1].aqueous

    def to_dict(self):
        return {
            "trajectory": [p.to_dict() for p in self.trajectory],
            "actual_duration_s": self.actual_duration_s,
            "chemical_stop_time_s": self.chemical_stop_time_s,
            "chemical_stop_reason": self.chemical_stop_reason,
            "diagnostics": self.diagnostics,
        }


def advance_mineral(
    model,
    aqueous_model,
    amounts,
    geometry,
    duration_s,
    *,
    temperature_K=298.15,
    pressure_Pa=101325,
    evaluation_times_s=(),
):
    """Advance one finite solid and the water state using one ODE call.

    Relative radius u=(n/n_start)^(1/3) removes the non-Lipschitz exhaustion
    endpoint. Every RHS and output solves the same fixed-solid aqueous problem
    using the original atoms. Boundary extension is only for rejected trial
    radii while finding an event, never an accepted material correction.
    """
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("mineral duration must be finite and nonnegative")
    if not aqueous_model.allow_calcite:
        raise ValueError("mineral kinetics requires its actual allowed calcite phase")
    times = tuple(sorted({0.0, float(duration_s), *(float(t) for t in evaluation_times_s)}))
    if any(not isfinite(t) or not 0 <= t <= duration_s for t in times):
        raise ValueError("mineral output times must lie inside the actual interval")
    initial_n = amounts.get("Calcite", 0.0)
    totals = element_amounts(amounts)
    limit = min(totals["C"], totals["Ca"])
    calls = 0

    def close(n):
        nonlocal calls
        calls += 1
        return solve_carbonate_equilibrium(
            aqueous_model,
            amounts,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            fixed_calcite_mol=n,
        )

    initial = close(initial_n)
    area = geometry.area(initial_n)
    reason = (
        "no_seed"
        if initial_n == 0
        else "inactive_surface"
        if area == 0
        else "zero_rate"
        if model.rate_constant_mol_m2_s == 0
        else "saturated"
        if abs(initial.saturation_ratio - 1) <= model.saturation_tolerance
        else None
    )
    stop_time = 0.0 if reason else None
    diagnostic = {"solver_calls": 0, "caloric_closure": "unavailable"}
    final_u, raw, stop_index = 1.0, None, None
    if duration_s > 0 and reason is None:
        maximum_u = (limit / initial_n) ** (1 / 3)
        coefficient = (
            geometry.reference_area_m2
            * model.rate_constant_mol_m2_s
            / (3 * geometry.reference_amount_mol ** (2 / 3) * initial_n ** (1 / 3))
        )
        sign = 1.0 if initial.saturation_ratio > 1 else -1.0

        def trial(u):
            bounded = min(max(float(u), 0.0), maximum_u)
            # Multiplicative packing represents n_start exactly at u=1.
            n = initial_n * bounded**3
            if bounded == maximum_u:
                n = limit
            return close(n)

        def rhs(t, y):
            return np.array([-coefficient * (1 - trial(y[0]).saturation_ratio)])

        def depleted(t, y):
            return y[0]

        def saturated(t, y):
            return sign * (trial(y[0]).saturation_ratio - 1) - model.saturation_tolerance

        depleted.terminal = saturated.terminal = True
        depleted.direction = saturated.direction = -1

        def jac(t, y):
            h = max(1e-8, min(1e-5, abs(y[0]) * 1e-5))
            return np.array([[(rhs(t, y)[0] - rhs(t, [y[0] - h])[0]) / h]])

        policy = ODESolverPolicy(
            "finite-calcite-relative-radius-bdf-v1",
            method="BDF",
            rtol=1e-10,
            atol=1e-13,
            dense_output=True,
            event_names=("mineral_exhausted", "mineral_saturated"),
        )
        report = solver_backend.solve_ode(
            rhs,
            (1.0,),
            time_span_s=(0.0, duration_s),
            policy=policy,
            events=(depleted, saturated),
            jac=jac,
        )
        report.raise_for_failure("finite mineral kinetics")
        raw = report.raw_result
        terminal = report.terminal_point
        if terminal is not None:
            stop_time, state, stop_index = terminal
            final_u = 0.0 if stop_index == 0 else float(state[0])
            reason = "depleted" if stop_index == 0 else "saturated"
        else:
            final_u = float(raw.y[0, -1])
        diagnostic.update(solver_calls=1, solver=report.diagnostic.to_dict())

    points = []
    for t in times:
        if raw is None:
            result = initial
        else:
            u = final_u if stop_time is not None and t >= stop_time else float(raw.sol(t)[0])
            n = (
                0.0
                if stop_index == 0 and stop_time is not None and t >= stop_time
                else initial_n * u**3
            )
            if not 0 <= n <= limit:
                raise RuntimeError("accepted mineral stock left its finite elemental domain")
            result = close(n)
        points.append(MineralKineticPoint(t, result, geometry.area(result.calcite_mol)))
    diagnostic.update(
        aqueous_closure_calls=calls,
        stationary_hold_s=max(0.0, duration_s - stop_time) if stop_time is not None else 0.0,
        geometry=geometry.to_dict(),
    )
    return MineralAdvanceReport(tuple(points), float(duration_s), stop_time, reason, diagnostic)
