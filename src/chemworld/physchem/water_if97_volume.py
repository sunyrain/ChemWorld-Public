"""IF97 equilibrium T/rho and internal-energy derivatives for fixed volumes."""

from dataclasses import replace
from math import isclose
from typing import cast

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.water_if97 import (
    WATER_CRITICAL_TEMPERATURE_K,
    WATER_GAS_CONSTANT_J_KG_K,
    IF97State,
    _gibbs_state,
    _positive,
    water_if97_b23_pressure,
    water_if97_region,
    water_if97_region3_state,
)
from chemworld.physchem.water_if97_solvers import IF97Equilibrium, water_if97_tp
from chemworld.physchem.water_saturation import water_saturation_pressure


def water_if97_trho(temperature_K: float, mass_density_kg_m3: float) -> IF97Equilibrium:
    """Stable fixed-volume fluid state, including mass-quality coexistence."""
    T, rho = temperature_K, mass_density_kg_m3
    _positive(T, "temperature_K")
    _positive(rho, "mass_density_kg_m3")
    if not 273.15 <= T <= 2273.15:
        raise ValueError("IF97 T/rho temperature is outside the fluid domain")
    saturation = None
    if T < WATER_CRITICAL_TEMPERATURE_K:
        pressure = water_saturation_pressure(T)
        liquid, vapor = (water_if97_tp(T, pressure, phase=phase) for phase in ("liquid", "vapor"))
        saturation = IF97Equilibrium(liquid=liquid, vapor=vapor, vapor_mass_fraction=0)
        if vapor.mass_density_kg_m3 <= rho <= liquid.mass_density_kg_m3:
            quality = ((1 / rho - liquid.specific_volume_m3_kg)
                       / (vapor.specific_volume_m3_kg - liquid.specific_volume_m3_kg))
            return replace(saturation, vapor_mass_fraction=quality)
    if (isclose(T, WATER_CRITICAL_TEMPERATURE_K, rel_tol=0, abs_tol=1e-10)
            and isclose(rho, 322, rel_tol=0, abs_tol=1e-8)):
        raise ValueError("IF97 critical response is singular")
    if 623.15 < T <= 863.15:
        try:
            dense = water_if97_region3_state(T, rho)
        except ValueError:
            dense = None  # Try the other declared region, never extrapolate region 3.
        # T/rho already resolves the side of coexistence above. A homogeneous
        # endpoint may fall inside the TP selector's saturation pressure tube;
        # do not discard its unique density or coerce it to a guessed quality.
        if dense is not None and water_if97_region(T, dense.pressure_Pa) in (3, 4):
            return IF97Equilibrium(homogeneous=replace(dense, stability_scope="IF97_stable_region"))
    region = 5 if T > 1073.15 else 2
    high = 50e6 if region == 5 else (water_if97_b23_pressure(T) if 623.15 < T <= 863.15 else 100e6)
    low = min(1.0, rho * WATER_GAS_CONSTANT_J_KG_K * T * .1)
    if saturation is not None:
        if rho > cast(IF97State, saturation.liquid).mass_density_kg_m3:
            if T > 623.15:
                raise ValueError("density exceeds the accepted IF97 dense-fluid pressure domain")
            region, low = 1, saturation.pressure_Pa
        else:
            high = min(high, saturation.pressure_Pa)
    # Select the interval's own limiting Gibbs equation at joins.
    def state_at(pressure: float) -> IF97State:
        return _gibbs_state(T, pressure, region)

    left, right = state_at(low), state_at(high)
    if not left.mass_density_kg_m3 <= rho <= right.mass_density_kg_m3:
        raise ValueError("density is outside the IF97 stable regional pressure bounds")
    fraction = ((rho - left.mass_density_kg_m3)
                / (right.mass_density_kg_m3 - left.mass_density_kg_m3))

    def jacobian(point: NDArray[np.float64]) -> list[list[float]]:
        state = state_at(point[0])
        return [[state.mass_density_kg_m3 * state.isothermal_compressibility_Pa_inv]]

    answer = solve_root(
        lambda point: [state_at(point[0]).mass_density_kg_m3 - rho],
        [low + (high - low) * fraction], jacobian=jacobian,
        variables=[ResidualVariable("P", "Pa", max(high, 1e6), "algebraic", low, high)],
        residual_scales=[rho], policy=RootPolicy(tolerance=2e-12, maximum_iterations=80))
    if not answer.success:
        raise ValueError(f"IF97 fixed-volume pressure did not converge: {answer.failure_reason}")
    state = state_at(float(answer.value[0]))
    return IF97Equilibrium(homogeneous=replace(state, stability_scope="IF97_stable_region"))


def water_if97_internal_energy_derivatives(state: IF97Equilibrium) -> tuple[float, float]:
    """Return (du/dT at rho, du/drho at T); includes equilibrium latent heat."""
    if state.homogeneous is not None:
        point = state.homogeneous
        dp_dT = point.thermal_expansion_K_inv / point.isothermal_compressibility_Pa_inv
        return point.cv_J_kg_K, ((point.pressure_Pa - point.temperature_K * dp_dT)
                                / point.mass_density_kg_m3**2)
    T, p = state.temperature_K, state.pressure_Pa
    dp_dT = cast(float, water_saturation_pressure(SecondOrderPair(T, dx=1)).dx)

    def along_saturation(point: IF97State) -> tuple[float, float]:
        volume = point.specific_volume_m3_kg
        vT = volume * point.thermal_expansion_K_inv
        vp = -volume * point.isothermal_compressibility_Pa_inv
        return (point.cp_J_kg_K - p * vT + (-T * vT - p * vp) * dp_dT,
                vT + vp * dp_dT)

    # IF97Equilibrium validates both endpoints and quality for coexistence.
    liquid, vapor = cast(IF97State, state.liquid), cast(IF97State, state.vapor)
    uL, vL = along_saturation(liquid)
    uV, vV = along_saturation(vapor)
    quality = cast(float, state.vapor_mass_fraction)
    volume_gap = vapor.specific_volume_m3_kg - liquid.specific_volume_m3_kg
    energy_gap = vapor.internal_energy_J_kg - liquid.internal_energy_J_kg
    quality_T = -((1 - quality) * vL + quality * vV) / volume_gap
    return ((1 - quality) * uL + quality * uV + energy_gap * quality_T,
            -energy_gap / (state.mass_density_kg_m3**2 * volume_gap))
