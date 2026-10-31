"""Conservative G8 enthalpy inversion and thermodynamic wet/ice-bulb closure.

Wet bulb means adiabatic isobaric saturation with condensate supplied at the
outlet temperature. The feed surface is explicit because ice/liquid have
different enthalpies. These functions own no material or process state.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
from math import exp, isfinite, log
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from chemworld.physchem.humid_air import (
    HumidAirState,
    _mixture_energy,
    _state_from_fractions,
    _validate,
    humid_air_from_water_fraction,
)
from chemworld.physchem.humid_air_equilibrium import (
    HumidAirSaturation,
    _gas_trial,
    _validate_pressure,
    condensed_water_state,
    humid_air_saturation,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    linear_solve,
    solve_root,
)
from chemworld.physchem.second_order_pair import SecondOrderPair
from chemworld.physchem.water_ice_boundaries import (
    ice_sublimation_pressure,
    ice_sublimation_temperature,
)
from chemworld.physchem.water_saturation import water_saturation_temperature


def humid_air_ph(
    water_mass_fraction: float, pressure_Pa: float, enthalpy_J_kg: float,
) -> HumidAirState:
    """Homogeneous gas root at fixed water fraction and mass-specific enthalpy.

    The gas TP seed at 473 K connects to the dilute branch. Like humid_air_tp,
    this does not perform a condensate flash; local stability is checked.
    """
    V, p, target = water_mass_fraction, pressure_Pa, enthalpy_J_kg
    _validate(V, 300.)
    _validate_pressure(p)
    if isinstance(target, bool) or not isfinite(target):
        raise ValueError("humid-air enthalpy must be finite and non-boolean")
    A = 1-V
    initial = humid_air_from_water_fraction(V, 473., p)

    def trial(point: NDArray[np.float64]) -> tuple[
        float, float, float, float, float, float, float, float,
    ]:
        T, rho = float(point[0]), exp(point[1])
        f = _mixture_energy(A, V, SecondOrderPair(T, dx=1), SecondOrderPair(rho, dy=1))
        dpdr = 2*rho*f.dy+rho*rho*f.dyy
        pressure = rho*rho*f.dy
        if pressure <= 0 or dpdr <= 0 or f.dxx >= 0:
            raise ResidualDomainError("humid-air PH trial is locally unstable")
        return (T, rho, pressure, f.value-T*f.dx+rho*f.dy,
                rho*rho*f.dxy, dpdr, -T*f.dxx+rho*f.dxy,
                2*f.dy-T*f.dxy+rho*f.dyy)

    def residual(point: NDArray[np.float64]) -> list[float]:
        _, _, pressure, h, *_ = trial(point)
        return [pressure-p, h-target]

    def jacobian(point: NDArray[np.float64]) -> list[list[float]]:
        _, rho, _, _, pt, pr, ht, hr = trial(point)
        return [[pt, pr*rho], [ht, hr*rho]]

    root = solve_root(residual, [473., log(initial.mass_density_kg_m3)], jacobian=jacobian,
        variables=[ResidualVariable("temperature", "K", 300., "algebraic", 193., 473.),
                   ResidualVariable("log_gas_density", "1", 1., "algebraic", -100., log(500.))],
        residual_scales=[p, 1e5], policy=RootPolicy(tolerance=1e-12, maximum_iterations=80))
    if not root.success:
        raise ValueError("humid-air PH root failed in supported domain: " + root.failure_reason)
    T, rho, *_ = trial(root.value)
    state = _state_from_fractions(A, V, T, rho)
    if abs(state.pressure_Pa-p) > 5e-11*p or abs(state.enthalpy_J_kg-target) > 1e-6:
        raise ValueError("humid-air PH residual exceeds acceptance")
    return state


@dataclass(frozen=True)
class HumidAirSaturationResponse:
    equilibrium: HumidAirSaturation
    dwater_mass_fraction_dT_K_inv: float
    dmass_density_dT_kg_m3_K: float
    dhumidity_ratio_dT_K_inv: float
    denthalpy_dT_J_kg_dry_air_K: float

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "equilibrium": self.equilibrium.to_dict()}


def humid_air_saturation_response(
    temperature_K: float, pressure_Pa: float, *, condensed_phase: str,
) -> HumidAirSaturationResponse:
    """Analytic constant-pressure saturation response on one explicit surface."""
    if condensed_phase not in ("liquid", "ice"):
        raise ValueError("saturation derivatives require an explicit liquid or ice surface")
    equilibrium = humid_air_saturation(temperature_K, pressure_Pa, condensed_phase=condensed_phase)
    gas, condensed = equilibrium.gas, equilibrium.condensate
    V, A = gas.water_mass_fraction, gas.dry_air_mass_fraction
    d = _gas_trial(V, temperature_K, gas.mass_density_kg_m3)
    v_t, rho_t = map(float, linear_solve(
        [[d["p_V"], d["p_rho"]], [d["mu_V"], d["mu_rho"]]],
        [-d["p_T"], -d["mu_T"]-condensed.entropy_J_kg_K]))
    h_t = (d["h_T"]+d["h_V"]*v_t+d["h_rho"]*rho_t)/A+d["h"]*v_t/(A*A)
    return HumidAirSaturationResponse(equilibrium, v_t, rho_t, v_t/(A*A), h_t)


@dataclass(frozen=True)
class HumidAirWetBulb:
    inlet: HumidAirState
    outlet: HumidAirSaturation
    water_added_kg_per_kg_dry_air: float
    energy_residual_J_kg_dry_air: float
    definition: str = "adiabatic_isobaric_saturation_condensate_feed_at_outlet_temperature"

    @property
    def temperature_K(self) -> float:  # noqa: N802 - shared SI quantity spelling
        return self.outlet.temperature_K

    def to_dict(self) -> dict[str, Any]:
        return {"temperature_K": self.temperature_K, "inlet": self.inlet.to_dict(),
                "outlet": self.outlet.to_dict(),
                "water_added_kg_per_kg_dry_air": self.water_added_kg_per_kg_dry_air,
                "energy_residual_J_kg_dry_air": self.energy_residual_J_kg_dry_air,
                "definition": self.definition}


def humid_air_wet_bulb(
    dry_bulb_temperature_K: float, pressure_Pa: float, water_mass_fraction: float,
    *, condensed_phase: str,
) -> HumidAirWetBulb:
    """Thermodynamic wet/ice bulb with explicit liquid/ice supply, no phase jump.

    No kinetic psychrometer correction is implied. An explicit supercooled
    liquid surface is permitted and labelled by the original condensate model.
    A saturated inlet requires zero added water; an initially supersaturated
    inlet requires a condensation flash instead and is rejected here.
    """
    T, p, V = dry_bulb_temperature_K, pressure_Pa, water_mass_fraction
    _validate(V, T)
    _validate_pressure(p)
    if V == 1:
        raise ValueError("wet bulb requires a nonzero dry-air amount")
    if condensed_phase not in ("liquid", "ice"):
        raise ValueError("wet bulb requires an explicit liquid or ice supply phase")
    lower, surface_upper = (270., 473.) if condensed_phase == "liquid" else (193., 273.16)
    upper = min(T, surface_upper)
    if upper < lower:
        raise ValueError("no wet-bulb temperature in the selected surface domain")
    inlet = humid_air_from_water_fraction(V, T, p)
    inlet_surface = condensed_water_state(T, p, phase=(
        condensed_phase if lower <= T <= surface_upper else "auto"))
    if (inlet.water_chemical_potential_J_kg is not None
            and inlet.water_chemical_potential_J_kg-inlet_surface.gibbs_energy_J_kg > 1e-5):
        raise ValueError("supersaturated wet-bulb inlet requires a condensation flash")
    # The dry-air endpoint is required above; both dry-mass quantities are finite.
    W_in = cast(float, inlet.humidity_ratio_kg_water_kg_dry_air)
    h_in = cast(float, inlet.enthalpy_J_kg_dry_air)

    @lru_cache(maxsize=16)
    def trial(Tw: float) -> tuple[HumidAirSaturation, float, float, float]:
        try:
            response = humid_air_saturation_response(Tw, p, condensed_phase=condensed_phase)
            saturated = response.equilibrium
            W_out = cast(float, saturated.gas.humidity_ratio_kg_water_kg_dry_air)
            delta = W_out-W_in
            hc = saturated.condensate.enthalpy_J_kg
            residual = h_in+delta*hc-cast(float, saturated.gas.enthalpy_J_kg_dry_air)
            slope = (response.dhumidity_ratio_dT_K_inv*hc
                     + delta*saturated.condensate.cp_J_kg_K
                     - response.denthalpy_dT_J_kg_dry_air_K)
            return saturated, delta, residual, slope
        except (ValueError, OverflowError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    def accepted(Tw: float) -> HumidAirWetBulb:
        saturated, delta, residual, _ = trial(Tw)
        if abs(residual) > 1e-6 or delta < -1e-12 or Tw > T+2e-8:
            raise ValueError("wet-bulb water/energy balance or operating direction failed")
        return HumidAirWetBulb(inlet, saturated, delta, -residual)

    # Handle the degenerate allowed interval explicitly, never manufacture
    # an endpoint for an unreachable root. Saturated states at Tmin are valid.
    if lower == upper:
        return accepted(lower)
    # Pure-water phase curves supply an initial guess only. Avoid starting
    # above boiling, where a saturated humid-gas composition is unreachable.
    boil = (ice_sublimation_temperature(max(p, ice_sublimation_pressure(50.)))
            if p < ice_sublimation_pressure(273.16) else water_saturation_temperature(p))
    initial = max(lower, min(upper, T-5., boil-2.))
    root = solve_root(lambda point: [trial(float(point[0]))[2]], [initial],
        jacobian=lambda point: [[trial(float(point[0]))[3]]],
        variables=[ResidualVariable("wet_bulb_temperature", "K", 300., "algebraic", lower, upper)],
        residual_scales=[1e5], policy=RootPolicy(tolerance=1e-12, maximum_iterations=80))
    if not root.success:
        raise ValueError("wet-bulb root failed in the selected surface domain: "
                         + root.failure_reason)
    return accepted(float(root.value[0]))
