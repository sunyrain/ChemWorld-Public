# ruff: noqa: N802 -- property names preserve the shared SI quantity spelling
"""Region-aware IF97 pressure roots and equilibrium PH/PS using shared roots.

Region-3 pressure is a polynomial in reduced density. Its real polynomial
roots supply seeds only; the common scaled root solver refines each and the
original Helmholtz evaluator enforces local stability and the IF97 domain.
The release's saturation curve selects the physical branch below critical T.
"""

from dataclasses import dataclass, replace
from itertools import pairwise
from math import isclose, isfinite, sqrt, ulp
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.residual_system import (
    ResidualVariable,
    RootPolicy,
    solve_root,
)
from chemworld.physchem.water_if97 import (
    WATER_CRITICAL_TEMPERATURE_K,
    WATER_GAS_CONSTANT_J_KG_K,
    IF97State,
    _gibbs_state,
    _positive,
    _region3_energy,
    water_if97_gibbs_state,
    water_if97_region,
    water_if97_region3_state,
)
from chemworld.physchem.water_if97_coefficients import REGION3, REGION3_LOG_COEFFICIENT
from chemworld.physchem.water_saturation import (
    water_saturation_pressure,
    water_saturation_temperature,
)

_ROOT_POLICY = RootPolicy(tolerance=2e-12, maximum_iterations=80)


def _dense_roots(T: float, p: float) -> list[IF97State]:
    tau = WATER_CRITICAL_TEMPERATURE_K / T
    coefficients = np.zeros(13)
    coefficients[0] = -p / (322 * WATER_GAS_CONSTANT_J_KG_K * T)
    coefficients[1] = REGION3_LOG_COEFFICIENT
    for i, j, n in REGION3:
        coefficients[i + 1] += n * i * tau**j
    seeds = np.polynomial.polynomial.polyroots(coefficients)
    states: list[IF97State] = []
    rejected: list[str] = []

    def pressure_at(point: NDArray[np.float64]) -> float:
        rho = point[0]
        return rho * rho * _region3_energy(T, rho).dx

    def jacobian(point: NDArray[np.float64]) -> list[list[float]]:
        rho = point[0]
        f = _region3_energy(T, rho)
        return [[2 * rho * f.dx + rho * rho * f.dxx]]

    for root in seeds:
        if abs(root.imag) > 1e-7 or root.real <= 0:
            continue
        rho = float(root.real * 322)
        # Local bounds keep polishing on the seed's branch; this is not a
        # clipping/repair of an accepted physical density.
        variables = [ResidualVariable("rho", "kg/m3", 322, "algebraic",
                                      max(1e-9, rho * .98), rho * 1.02)]
        answer = solve_root(
            lambda point: [pressure_at(point) - p], [rho], jacobian=jacobian,
            variables=variables, residual_scales=[max(p, 1e6)], policy=_ROOT_POLICY,
        )
        if not answer.success:
            rejected.append(answer.failure_reason)
            continue
        try:
            state = water_if97_region3_state(T, float(answer.value[0]))
        except ValueError as exc:
            rejected.append(str(exc))
            continue
        if not isclose(state.pressure_Pa, p, rel_tol=5e-12, abs_tol=1e-5):
            rejected.append("pressure residual exceeds acceptance")
            continue
        if not any(isclose(state.mass_density_kg_m3, old.mass_density_kg_m3,
                           rel_tol=1e-8) for old in states):
            states.append(state)
    if not states:
        raise ValueError(f"no admissible IF97 region-3 pressure root: {rejected}")
    return sorted(states, key=lambda state: state.mass_density_kg_m3)


def water_if97_tp(
    temperature_K: float, pressure_Pa: float, *, phase: str | None = None,
) -> IF97State:
    """Stable fluid TP state; a saturation point needs liquid/vapor selection."""
    T, p = temperature_K, pressure_Pa
    region = water_if97_region(T, p)
    if phase not in (None, "liquid", "vapor"):
        raise ValueError("phase must be liquid, vapor or None")
    if region in (1, 2, 5) or (region == 4 and T <= 623.15):
        if region == 4 and phase is None:
            raise ValueError("IF97 saturation requires an explicit phase or vapor quality")
        expected = "liquid" if region == 1 else "vapor"
        if region != 4 and phase is not None and phase != expected:
            raise ValueError("requested phase is not the stable IF97 phase")
        selected = (1 if phase == "liquid" else 2) if region == 4 else None
        return replace(water_if97_gibbs_state(T, p, region=selected),
                       stability_scope="IF97_stable_region")
    return _dense_state(T, p, region, phase)


def _dense_state(T: float, p: float, region: int, phase: str | None = None) -> IF97State:
    if isclose(T, WATER_CRITICAL_TEMPERATURE_K, rel_tol=0, abs_tol=1e-10) and (
        isclose(p, 22.064e6, rel_tol=0, abs_tol=.01)
    ):
        raise ValueError("IF97 critical response is singular")
    if region == 4 and phase is None:
        raise ValueError("IF97 saturation requires an explicit phase or vapor quality")
    states = _dense_roots(T, p)
    if T < WATER_CRITICAL_TEMPERATURE_K:
        stable_phase = phase if region == 4 else (
            "liquid" if p > water_saturation_pressure(T) else "vapor")
        if phase is not None and phase != stable_phase:
            raise ValueError("requested phase is not the stable IF97 phase")
        state = states[-1] if stable_phase == "liquid" else states[0]
    else:
        if phase is not None:
            raise ValueError("supercritical IF97 fluid has no liquid/vapor phase selector")
        if len(states) != 1:
            raise ValueError("multiple locally stable dense roots; physical selection unresolved")
        state = states[0]
    return replace(state, stability_scope="IF97_stable_region")


@dataclass(frozen=True)
class IF97Equilibrium:
    """Mass-quality lever rule, with no invented two-phase Cp/Cv/sound speed."""

    homogeneous: IF97State | None = None
    liquid: IF97State | None = None
    vapor: IF97State | None = None
    vapor_mass_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.homogeneous is not None:
            if (self.liquid is not None or self.vapor is not None
                    or self.vapor_mass_fraction is not None):
                raise ValueError("homogeneous equilibrium cannot also own coexistence endpoints")
            return
        quality = self.vapor_mass_fraction
        if (self.liquid is None or self.vapor is None or isinstance(quality, bool)
                or quality is None or not isfinite(quality) or not 0 <= quality <= 1):
            raise ValueError("coexistence needs both endpoints and a quality within [0,1]")
        if (not isclose(self.liquid.temperature_K, self.vapor.temperature_K, abs_tol=1e-8)
                or not isclose(self.liquid.pressure_Pa, self.vapor.pressure_Pa, rel_tol=1e-9)):
            raise ValueError("coexistence endpoints must share T and P")

    def _mass_property(self, name: Literal[
        "enthalpy_J_kg", "internal_energy_J_kg", "entropy_J_kg_K", "specific_volume_m3_kg",
    ]) -> float:
        if self.homogeneous is not None:
            return getattr(self.homogeneous, name)
        quality = cast(float, self.vapor_mass_fraction)
        return (1 - quality) * getattr(self.liquid, name) + quality * getattr(self.vapor, name)

    @property
    def temperature_K(self) -> float:
        return cast(IF97State, self.homogeneous or self.liquid).temperature_K

    @property
    def pressure_Pa(self) -> float:
        return cast(IF97State, self.homogeneous or self.liquid).pressure_Pa

    @property
    def enthalpy_J_kg(self) -> float:
        return self._mass_property("enthalpy_J_kg")

    @property
    def internal_energy_J_kg(self) -> float:
        return self._mass_property("internal_energy_J_kg")

    @property
    def entropy_J_kg_K(self) -> float:
        return self._mass_property("entropy_J_kg_K")

    @property
    def specific_volume_m3_kg(self) -> float:
        return self._mass_property("specific_volume_m3_kg")

    @property
    def mass_density_kg_m3(self) -> float:
        return 1 / self.specific_volume_m3_kg

    def to_dict(self) -> dict[str, Any]:
        result = {name: getattr(self, name) for name in (
            "temperature_K", "pressure_Pa", "enthalpy_J_kg", "internal_energy_J_kg",
            "entropy_J_kg_K", "specific_volume_m3_kg", "mass_density_kg_m3",
            "vapor_mass_fraction",
        )}
        for name in ("homogeneous", "liquid", "vapor"):
            value = getattr(self, name)
            result[name] = value.to_dict() if value is not None else None
        result["response_scope"] = "single_phase_only; no two_phase_heat_capacity_or_sound_speed"
        return result


def water_if97_saturation(pressure_Pa: float, vapor_mass_fraction: float) -> IF97Equilibrium:
    """Pressure/quality equilibrium; the exact critical endpoint is singular."""
    _positive(pressure_Pa, "pressure_Pa")
    if pressure_Pa >= 22.064e6:
        raise ValueError("two-phase IF97 requires subcritical saturation pressure")
    temperature = water_saturation_temperature(pressure_Pa)
    # The shared analytic inverse recovers the 623.15 K regional join with
    # up to 13 ulps of roundoff. Canonicalize only that numerical endpoint;
    # otherwise its R1/R2 endpoints can incorrectly become R3 endpoints.
    if abs(temperature - 623.15) <= 16 * ulp(623.15):
        temperature = 623.15
    liquid = water_if97_tp(temperature, pressure_Pa, phase="liquid")
    vapor = water_if97_tp(temperature, pressure_Pa, phase="vapor")
    return IF97Equilibrium(liquid=liquid, vapor=vapor, vapor_mass_fraction=vapor_mass_fraction)


def water_if97_at_pressure(
    pressure_Pa: float, *, enthalpy_J_kg: float | None = None,
    entropy_J_kg_K: float | None = None, region: int | None = None,
) -> IF97Equilibrium:
    """Equilibrium PH or PS, solving each region separately without extrapolation.

    A homogeneous region hint resolves the small caloric overlaps permitted by
    IF97's piecewise equations. Without a hint, distinct accepted temperatures
    are reported as ambiguous, never selected by the order of iteration.
    """
    p = pressure_Pa
    _positive(p, "pressure_Pa")
    if p > 100e6:
        raise ValueError("IF97 pressure must not exceed 100 MPa")
    if region is not None and (type(region) is not int or region not in (1, 2, 3, 5)):
        raise ValueError("homogeneous region hint must be 1, 2, 3, 5 or None")
    if (enthalpy_J_kg is None) == (entropy_J_kg_K is None):
        raise ValueError("supply exactly one of enthalpy_J_kg and entropy_J_kg_K")
    name = "enthalpy_J_kg" if enthalpy_J_kg is not None else "entropy_J_kg_K"
    target = enthalpy_J_kg if enthalpy_J_kg is not None else cast(float, entropy_J_kg_K)
    if isinstance(target, bool) or not isfinite(target):
        raise ValueError("IF97 caloric target must be finite and non-boolean")
    splits = [273.15, 623.15, 1073.15]
    if p <= 50e6:
        splits.append(2273.15)
    saturation = None
    if water_saturation_pressure(273.15) <= p < 22.064e6:
        saturation = water_if97_saturation(p, 0)
        lo, hi = (getattr(s, name) for s in (saturation.liquid, saturation.vapor))
        if lo <= target <= hi:
            if region is not None:
                endpoint = saturation.liquid if target == lo else (
                    saturation.vapor if target == hi else None)
                if endpoint is None or endpoint.region != region:
                    raise ValueError("homogeneous region hint conflicts with coexistence")
                return IF97Equilibrium(homogeneous=endpoint)
            return replace(saturation, vapor_mass_fraction=(target - lo) / (hi - lo))
        splits.append(saturation.temperature_K)
    if p > water_saturation_pressure(623.15):
        boundary = .57254459862746e3 + sqrt(
            (p / 1e6 - .13918839778870e2) / .10192970039326e-2)
        splits.append(min(boundary, 863.15))
    splits = sorted(set(splits))
    failures: list[str] = []
    accepted: list[IF97State] = []
    for low, high in pairwise(splits):
        middle = (low + high) / 2
        segment_region = water_if97_region(middle, p)
        if region is not None and segment_region != region:
            continue
        phase: str | None = None
        if saturation is not None:
            phase = "liquid" if middle < saturation.temperature_K else "vapor"

        def state_at(
            temperature: float, phase: str | None = phase, segment_region: int = segment_region,
        ) -> IF97State:
            at_saturation = (saturation is not None and isclose(
                temperature, saturation.temperature_K, rel_tol=0, abs_tol=1e-9))
            # At a regional join evaluate this interval's own limiting energy,
            # preserving the documented IF97 jump rather than interpolating it.
            if not at_saturation and water_if97_region(temperature, p) != segment_region:
                if segment_region == 3:
                    return _dense_state(temperature, p, 3)
                return _gibbs_state(temperature, p, segment_region)
            return water_if97_tp(temperature, p, phase=phase if at_saturation else None)

        try:
            left, right = state_at(low), state_at(high)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        lower, upper = getattr(left, name), getattr(right, name)
        if not lower <= target <= upper:
            continue
        if target in (lower, upper):
            accepted.append(left if target == lower else right)
            continue
        initial = low + (high - low) * (target - lower) / (upper - lower)

        def residual(point: NDArray[np.float64]) -> list[float]:
            return [getattr(state_at(point[0]), name) - target]

        def jacobian(point: NDArray[np.float64]) -> list[list[float]]:
            state = state_at(point[0])
            return [[state.cp_J_kg_K / (point[0] if name == "entropy_J_kg_K" else 1)]]

        answer = solve_root(
            residual, [initial], jacobian=jacobian,
            variables=[ResidualVariable("T", "K", 500, "algebraic", low, high)],
            residual_scales=[max(abs(target), 1e3)], policy=_ROOT_POLICY,
        )
        if answer.success:
            accepted.append(state_at(float(answer.value[0])))
        else:
            failures.append(answer.failure_reason)
    if accepted:
        if any(abs(state.temperature_K - accepted[0].temperature_K) > 1e-6
               for state in accepted[1:]):
            raise ValueError("ambiguous IF97 caloric boundary overlap; specify a region hint")
        return IF97Equilibrium(homogeneous=accepted[0])
    raise ValueError(
        f"IF97 caloric target has no accepted state in the declared regions: {failures}"
    )
