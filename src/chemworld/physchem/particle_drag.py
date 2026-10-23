"""Named sphere drag laws, signed terminal motion and shared-ODE trajectories.

Uniform quiescent Newtonian carrier, rigid isolated particle, no added mass,
history force, wall correction or particle-particle interactions are implied.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import copysign, exp, expm1, isfinite, log, log10, pi, sqrt, tanh
from typing import Any

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.flow_meters import _engineering_root, _positive
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode
from chemworld.physchem.transport import FluidState

PARTICLE_DRAG_RANGES = {
    "Stokes": (0.0, 0.3),
    "Barati": (0.0, 2e5),
    "Barati_high": (0.0, 1e6),
    "Rouse": (0.0, 2e5),
    "Engelund_Hansen": (0.0, 2e5),
    "Clift_Gauvin": (0.0, 2e5),
    "Morsi_Alexander": (0.0, 2e5),
    "Graf": (0.0, 2e5),
    "Flemmer_Banks": (0.0, 2e5),
    "Khan_Richardson": (0.0, 2e5),
    "Swamee_Ojha": (0.0, 1.5e5),
    "Yen": (0.0, 2e5),
    "Haider_Levenspiel": (0.0, 2e5),
    "Cheng": (0.0, 2e5),
    "Terfous": (0.1, 5e4),
    "Mikhailov_Freire": (0.0, 118300),
    "Clift": (0.0, 1e6),
    "Ceylan": (0.1, 1e6),
    "Almedeij": (0.0, 1e6),
    "Morrison": (0.0, 1e6),
    "Song_Xu": (0.0, 1000),
}


def available_particle_drag_methods(
    reynolds: float, *, check_ranges: bool = True
) -> tuple[str, ...]:
    _positive(reynolds, "Reynolds number")
    return tuple(
        name
        for name, (low, high) in PARTICLE_DRAG_RANGES.items()
        if not check_ranges or low <= reynolds <= high
    )


def _log_sum(values: tuple[float, ...]) -> float:
    maximum = max(values)
    return maximum + log(sum(exp(value - maximum) for value in values))


def particle_drag_coefficient(
    reynolds: float,
    *,
    method: str = "auto",
    sphericity: float = 1.0,
    projected_area_ratio: float = 1.0,
    check_recommended_ranges: bool = True,
    cap_barati_high_at_million: bool = False,
) -> float:
    """Projected-area Cd. Only Song-Xu accepts nonspherical shape parameters.

    Auto uses the source's Stokes/Barati transition at Re.01-.1 and switches
    to high-Re Barati at212963.26847812787; that small overlap is explicit.
    Above1e6 the default rejects; legacy high-Re capping must be requested.
    """
    _positive(reynolds, "Reynolds number")
    if not isfinite(sphericity) or not 0 < sphericity <= 1:
        raise ValueError("sphericity must lie in (0,1]")
    _positive(projected_area_ratio, "projected area ratio")
    if method != "Song_Xu" and (sphericity != 1 or projected_area_ratio != 1):
        raise ValueError("only Song-Xu has declared nonspherical shape parameters")
    if method == "auto":
        if reynolds < 0.01:
            return 24 / reynolds
        if reynolds <= 0.1:
            blend = (reynolds - 0.01) / 0.09
            return (1 - blend) * 24 / reynolds + blend * particle_drag_coefficient(
                reynolds, method="Barati"
            )
        method = "Barati" if reynolds <= 212963.26847812787 else "Barati_high"
        if method == "Barati" and reynolds > 2e5:
            check_recommended_ranges = False
    if method not in PARTICLE_DRAG_RANGES:
        raise ValueError(f"unknown particle drag method {method!r}")
    if cap_barati_high_at_million:
        if method != "Barati_high":
            raise ValueError("Reynolds capping applies only to Barati_high")
        reynolds = min(reynolds, 1e6)
    low, high = PARTICLE_DRAG_RANGES[method]
    if check_recommended_ranges and not low <= reynolds <= high:
        raise ValueError(f"{method} Reynolds number outside recommended range")
    r = reynolds
    if method == "Stokes":
        result = 24 / r
    elif method == "Barati":
        result = (
            5.4856e9 * tanh(4.3774e-9 / r)
            + 0.0709 * tanh(700.6574 / r)
            + 0.3894 * tanh(74.1539 / r)
            - 0.1198 * tanh(7429.0843 / r)
            + 1.7174 * tanh(9.9851 / (r + 2.3384))
            + 0.4744
        )
    elif method == "Barati_high":
        result = (
            8e-6 * ((r / 6530) ** 2 + tanh(r) - 8 * log10(r))
            - 0.4119 * exp(-2.08e43 / (r + r * r) ** 4)
            - 2.1344 * exp(-(log10(r * r + 10.7563) ** 2 + 9.9867) / r)
            + 0.1357 * exp(-((r / 1620) ** 2 + 10370) / r)
            - 0.0085 / r * (2 * log10(tanh(tanh(r))) - 2825.7162)
            + 2.4795
        )
    elif method == "Rouse":
        result = 24 / r + 3 / sqrt(r) + 0.34
    elif method == "Engelund_Hansen":
        result = 24 / r + 1.5
    elif method == "Clift_Gauvin":
        result = 24 / r * (1 + 0.152 * r**0.677) + 0.417 / (1 + 5070 * r**-0.94)
    elif method == "Morsi_Alexander":
        rows = (
            (0.1, 24, 0, 0),
            (1, 22.73, 0.0903, 3.69),
            (10, 29.1667, -3.8889, 1.222),
            (100, 46.5, -116.67, 0.6167),
            (1000, 98.33, -2778, 0.3644),
            (5000, 148.62, -47500, 0.357),
            (10000, -490.546, 578700, 0.46),
            (float("inf"), -1662.5, 5416700, 0.5191),
        )
        _, a, b, c = next(row for row in rows if r < row[0])
        result = a / r + b / r**2 + c
    elif method == "Graf":
        result = 24 / r + 7.3 / (1 + sqrt(r)) + 0.25
    elif method == "Flemmer_Banks":
        power = 0.383 * r**0.356 - 0.207 * r**0.396 - 0.143 / (1 + log10(r) ** 2)
        result = 24 / r * 10**power
    elif method == "Khan_Richardson":
        result = (2.49 * r**-0.328 + 0.34 * r**0.067) ** 3.18
    elif method == "Swamee_Ojha":
        result = (
            0.5
            * (16 * ((24 / r) ** 1.6 + (130 / r) ** 0.72) ** 2.5 + ((40000 / r) ** 2 + 1) ** -0.25)
            ** 0.25
        )
    elif method == "Yen":
        result = 24 / r * (1 + 0.15 * sqrt(r) + 0.017 * r) - 0.208 / (1 + 10000 / sqrt(r))
    elif method == "Haider_Levenspiel":
        result = 24 / r * (1 + 0.1806 * r**0.6459) + 0.4251 / (1 + 6880.95 / r)
    elif method == "Cheng":
        result = 24 / r * (1 + 0.27 * r) ** 0.43 - 0.47 * expm1(-0.04 * r**0.38)
    elif method == "Terfous":
        result = 2.689 + 21.683 / r + 0.131 / r**2 - 10.616 / r**0.1 + 12.216 / r**0.2
    elif method == "Mikhailov_Freire":
        result = (
            3808
            / 681
            * (1617933 / 2030 + 178861 / 1063 * r + 1219 / 1084 * r * r)
            / (r * (77531 / 422 + 13529 / 976 * r - r * r / 71154))
        )
    elif method == "Clift":
        logarithm = log10(r)
        if r < 0.01:
            result = 24 / r + 3 / 16
        elif r < 20:
            result = 24 / r * (1 + 0.1315 * r ** (0.82 - 0.05 * logarithm))
        elif r < 260:
            result = 24 / r * (1 + 0.1935 * r**0.6305)
        elif r < 1500:
            result = 10 ** (1.6435 - 1.1242 * logarithm + 0.1558 * logarithm**2)
        elif r < 12000:
            result = 10 ** (
                -2.4571 + 2.5558 * logarithm - 0.9295 * logarithm**2 + 0.1049 * logarithm**3
            )
        elif r < 44000:
            result = 10 ** (-1.9181 + 0.637 * logarithm - 0.0636 * logarithm**2)
        elif r < 338000:
            result = 10 ** (-4.339 + 1.5809 * logarithm - 0.1546 * logarithm**2)
        elif r < 400000:
            result = 29.78 - 5.3 * logarithm
        else:
            result = 0.19 * logarithm - 0.49
    elif method == "Ceylan":
        result = (
            1
            - 0.5 * exp(0.182)
            + 10.11 * r ** (-2 / 3) * exp(0.952 / r**0.25)
            - 0.03859 * r ** (-4 / 3) * exp(1.3 / sqrt(r))
            + 3.7e-6 * r * exp(-1.25e-5 * r)
            - 1.16e-11 * r * r * exp(-4.44e-6 * r)
        )
    elif method == "Almedeij":
        lr = log(r)
        log_phi1 = _log_sum(
            tuple(10 * (log(a) + b * lr) for a, b in ((24, -1), (21, -0.67), (4, -0.33), (0.4, 0)))
        )
        log_phi2 = -_log_sum((-10 * (log(0.148) + 0.11 * lr), -10 * log(0.5)))
        log_phi3 = 10 * (log(1.57e8) - 1.625 * lr)
        log_phi4 = -_log_sum((-10 * (log(6e-17) + 2.63 * lr), -10 * log(0.2)))
        combined = -_log_sum((-_log_sum((log_phi1, log_phi2)), -log_phi3))
        result = exp(0.1 * _log_sum((combined, log_phi4)))
    elif method == "Morrison":
        scaled = r / 263000
        result = (
            24 / r
            + 2.6 * r / 5 / (1 + (r / 5) ** 1.52)
            + 0.411 * scaled**0.06 / (1 + scaled**8)
            + r**0.8 / 461000
        )
    else:
        result = 24 * (1 + 0.35 * r) ** 0.44 / (r * sphericity**0.65 * projected_area_ratio**0.3)
    _positive(result, "drag coefficient")
    return result


@dataclass(frozen=True)
class ParticleDragSpec:
    diameter_m: float
    particle_density_kg_m3: float
    fluid: FluidState
    method: str = "auto"
    gravity_m_s2: float = 9.80665
    sphericity: float = 1.0
    projected_area_ratio: float = 1.0
    check_recommended_ranges: bool = True

    def __post_init__(self) -> None:
        for name in ("diameter_m", "particle_density_kg_m3", "gravity_m_s2"):
            _positive(getattr(self, name), name)
        if self.method != "auto" and self.method not in PARTICLE_DRAG_RANGES:
            raise ValueError("unknown particle drag model")
        if not isfinite(self.sphericity) or not 0 < self.sphericity <= 1:
            raise ValueError("sphericity must lie in (0,1]")
        _positive(self.projected_area_ratio, "projected area ratio")
        if self.method != "Song_Xu" and (self.sphericity != 1 or self.projected_area_ratio != 1):
            raise ValueError("shape corrections require Song-Xu")

    @property
    def particle_mass_kg(self) -> float:
        return self.particle_density_kg_m3 * pi * self.diameter_m**3 / 6

    @property
    def driving_acceleration_m_s2(self) -> float:
        return self.gravity_m_s2 * (1 - self.fluid.density_kg_m3 / self.particle_density_kg_m3)

    @property
    def stokes_relaxation_time_s(self) -> float:
        return self.particle_density_kg_m3 * self.diameter_m**2 / (18 * self.fluid.viscosity_Pa_s)

    def reynolds(self, velocity_m_s: float) -> float:
        return (
            self.fluid.density_kg_m3
            * abs(velocity_m_s)
            * self.diameter_m
            / self.fluid.viscosity_Pa_s
        )

    def coefficient(self, reynolds: float) -> float:
        return particle_drag_coefficient(
            reynolds,
            method=self.method,
            sphericity=self.sphericity,
            projected_area_ratio=self.projected_area_ratio,
            check_recommended_ranges=self.check_recommended_ranges,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ParticleDragSpec:
        return cls(**{**value, "fluid": FluidState(**value["fluid"])})


@dataclass(frozen=True)
class ParticleTerminalResult:
    velocity_m_s: float
    reynolds: float
    drag_coefficient: float | None
    force_residual_N: float
    root_selection: str


def particle_terminal_velocity(
    spec: ParticleDragSpec,
    *,
    reynolds_bracket: tuple[float, float] | None = None,
) -> ParticleTerminalResult:
    """Signed settling/rising speed; optional bracket selects a drag-crisis branch.

    Without a bracket the first sign change on a129-point log grid is used.
    No claim of global uniqueness is made for nonmonotone empirical drag laws.
    """
    acceleration = spec.driving_acceleration_m_s2
    if acceleration == 0:
        return ParticleTerminalResult(0, 0, None, 0, "neutral buoyancy")
    stokes = acceleration * spec.stokes_relaxation_time_s
    if spec.method == "Stokes" or (spec.method == "auto" and spec.reynolds(stokes) < 0.01):
        coefficient = spec.coefficient(spec.reynolds(stokes))
        return ParticleTerminalResult(
            stokes, spec.reynolds(stokes), coefficient, 0, "analytic Stokes"
        )
    target = (
        4
        / 3
        * spec.gravity_m_s2
        * spec.diameter_m**3
        * spec.fluid.density_kg_m3
        * abs(spec.particle_density_kg_m3 - spec.fluid.density_kg_m3)
        / spec.fluid.viscosity_Pa_s**2
    )

    def residual(log_reynolds: float) -> float:
        reynolds = exp(log_reynolds)
        return log(spec.coefficient(reynolds)) + 2 * log_reynolds - log(target)

    if reynolds_bracket is None:
        low, high = (0, 1e6) if spec.method == "auto" else PARTICLE_DRAG_RANGES[spec.method]
        points = np.linspace(log(max(low, 1e-12)), log(high) * (1 - 1e-15), 129)
        bracket = None
        previous, previous_value = float(points[0]), residual(float(points[0]))
        for coordinate in points[1:]:
            current, value = float(coordinate), residual(float(coordinate))
            if previous_value <= 0 <= value:
                bracket = (previous, current)
                break
            previous, previous_value = current, value
        if bracket is None:
            raise ValueError("no terminal root bracket within declared drag Reynolds domain")
        selection = "first sign change on129-point logarithmic domain grid"
    else:
        lower, upper = reynolds_bracket
        _positive(lower, "lower Reynolds bound")
        _positive(upper, "upper Reynolds bound")
        if upper <= lower:
            raise ValueError("terminal Reynolds bounds must increase")
        bracket = (log(lower), log(upper))
        if residual(bracket[0]) * residual(bracket[1]) > 0:
            raise ValueError("declared terminal bounds do not bracket a root")
        selection = "caller Reynolds bracket"
    value = _engineering_root(
        residual,
        sum(bracket) / 2,
        bracket[0],
        bracket[1],
        name="log_terminal_reynolds",
        unit="dimensionless",
        scale=1,
    )
    reynolds = exp(value)
    velocity = copysign(
        reynolds * spec.fluid.viscosity_Pa_s / (spec.fluid.density_kg_m3 * spec.diameter_m),
        acceleration,
    )
    coefficient = spec.coefficient(reynolds)
    resistance = (
        3
        * spec.fluid.density_kg_m3
        * coefficient
        * velocity
        * abs(velocity)
        / (4 * spec.diameter_m * spec.particle_density_kg_m3)
    )
    return ParticleTerminalResult(
        velocity,
        reynolds,
        coefficient,
        spec.particle_mass_kg * (acceleration - resistance),
        selection,
    )


def time_to_stokes_terminal(
    spec: ParticleDragSpec,
    initial_velocity_m_s: float,
    *,
    relative_tolerance: float = 1e-9,
    absolute_velocity_tolerance_m_s: float = 0.0,
) -> float:
    """Exact exponential settling time; initial states already in tolerance return0."""
    if not isfinite(initial_velocity_m_s) or not 0 < relative_tolerance < 1:
        raise ValueError("finite initial velocity and relative tolerance in(0,1) required")
    if not isfinite(absolute_velocity_tolerance_m_s) or absolute_velocity_tolerance_m_s < 0:
        raise ValueError("absolute velocity tolerance must be finite and nonnegative")
    terminal = spec.driving_acceleration_m_s2 * spec.stokes_relaxation_time_s
    threshold = max(relative_tolerance * abs(terminal), absolute_velocity_tolerance_m_s)
    difference = abs(initial_velocity_m_s - terminal)
    if difference <= threshold:
        return 0.0
    if threshold == 0:
        raise ValueError("neutral-buoyancy relaxation requires an absolute velocity tolerance")
    return spec.stokes_relaxation_time_s * log(difference / threshold)


@dataclass(frozen=True)
class ParticleMotionState:
    time_s: float = 0.0
    position_m: float = 0.0
    velocity_m_s: float = 0.0

    def __post_init__(self) -> None:
        if (
            not all(isfinite(value) for value in (self.time_s, self.position_m, self.velocity_m_s))
            or self.time_s < 0
        ):
            raise ValueError("particle state must be finite with nonnegative time")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, float]) -> ParticleMotionState:
        return cls(**value)


@dataclass(frozen=True)
class ParticleMotionResult:
    state: ParticleMotionState
    displacement_m: float
    kinetic_energy_change_J: float
    buoyancy_gravity_work_J: float
    drag_dissipation_J: float
    energy_residual_J: float
    integration: str
    solver_diagnostic: dict[str, object] | None = None


PARTICLE_ODE_POLICY = ODESolverPolicy(
    "isolated_particle_drag_v1", method="LSODA", rtol=1e-9, atol=1e-12
)


def advance_particle_motion(
    spec: ParticleDragSpec,
    state: ParticleMotionState,
    duration_s: float,
    *,
    policy: ODESolverPolicy = PARTICLE_ODE_POLICY,
) -> ParticleMotionResult:
    """Advance signed velocity/displacement/dissipation in the existing ODE backend.

    Downward is positive. Signed drag opposes motion even for initial upward
    launch or rising buoyant particles. Fluid density/viscosity are constant
    over this step; caller owns any surrounding process-state coupling.
    """
    if not isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration must be finite and nonnegative")
    initial = state.velocity_m_s
    acceleration = spec.driving_acceleration_m_s2
    stokes_terminal = acceleration * spec.stokes_relaxation_time_s
    analytic = spec.method == "Stokes" or (
        spec.method == "auto" and max(spec.reynolds(initial), spec.reynolds(stokes_terminal)) < 0.01
    )
    diagnostic = None
    if duration_s == 0:
        velocity, displacement, dissipation_specific = initial, 0.0, 0.0
        integration = "zero-duration identity"
    elif analytic:
        if (
            spec.method == "Stokes"
            and spec.check_recommended_ranges
            and max(spec.reynolds(initial), spec.reynolds(stokes_terminal)) > 0.3
        ):
            raise ValueError("Stokes trajectory exceeds declared Reynolds domain")
        tau = spec.stokes_relaxation_time_s
        decay = exp(-duration_s / tau)
        velocity = stokes_terminal + (initial - stokes_terminal) * decay
        displacement = stokes_terminal * duration_s + (initial - stokes_terminal) * tau * (
            -expm1(-duration_s / tau)
        )
        dissipation_specific = acceleration * displacement - (velocity**2 - initial**2) / 2
        integration = "stable analytic Stokes exponential"
    else:

        def rhs(time_s: float, values: NDArray[np.float64]) -> NDArray[np.float64]:
            velocity = float(values[0])
            drag_acceleration = (
                0.0
                if velocity == 0
                else 3
                * spec.fluid.density_kg_m3
                * spec.coefficient(spec.reynolds(velocity))
                * velocity
                * abs(velocity)
                / (4 * spec.diameter_m * spec.particle_density_kg_m3)
            )
            return np.array(
                [acceleration - drag_acceleration, velocity, drag_acceleration * velocity]
            )

        report = solve_ode(
            rhs,
            [initial, 0, 0],
            time_span_s=(state.time_s, state.time_s + duration_s),
            evaluation_times_s=[state.time_s + duration_s],
            policy=policy,
        )
        report.raise_for_failure("isolated particle motion")
        velocity, displacement, dissipation_specific = map(float, report.y[:, -1])
        diagnostic = report.diagnostic.to_dict()
        integration = "shared solve_ode velocity/displacement/specific-dissipation"
    mass = spec.particle_mass_kg
    kinetic, work, dissipation = (
        mass * (velocity**2 - initial**2) / 2,
        mass * acceleration * displacement,
        mass * dissipation_specific,
    )
    return ParticleMotionResult(
        ParticleMotionState(state.time_s + duration_s, state.position_m + displacement, velocity),
        displacement,
        kinetic,
        work,
        dissipation,
        work - kinetic - dissipation,
        integration,
        diagnostic,
    )
