"""G8-10 Eqs. 12/13: humid air in equilibrium with pure liquid water or ice.

Air dissolution in condensate is neglected, exactly as in G8 section 5.
All roots use the original native potentials and shared residual solver.
The selected condensed surface may be metastable; no inventory is created.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from math import exp, isfinite, log
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from chemworld.physchem.helmholtz import HelmholtzFluid, HelmholtzState

from chemworld.physchem.humid_air import (
    HumidAirState,
    _mixture_energy,
    _state_from_fractions,
    _validate,
    humid_air_from_water_fraction,
)
from chemworld.physchem.humid_air_parameters import (
    DRY_AIR_MOLAR_MASS_KG_MOL as MA,
)
from chemworld.physchem.humid_air_parameters import (
    WATER_MOLAR_MASS_KG_MOL as MW,
)
from chemworld.physchem.humid_air_parameters import iapws95_water
from chemworld.physchem.precise_scalar import PreciseScalar
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualVariable,
    RootPolicy,
    solve_root,
)
from chemworld.physchem.second_order_pair import PairScalar, SecondOrderPair
from chemworld.physchem.water_ice import ice_ih_state
from chemworld.physchem.water_ice_boundaries import (
    ice_sublimation_pressure,
    ice_sublimation_temperature,
)
from chemworld.physchem.water_saturation import (
    water_saturation_pressure,
    water_saturation_temperature,
)


def _validate_pressure(p: float) -> None:
    if isinstance(p, bool) or not isfinite(p) or not 0 < p <= 5e6:
        raise ValueError("humid-air equilibrium requires 0 < p <= 5 MPa")


@lru_cache(maxsize=1)
def _liquid_water_model() -> HelmholtzFluid:
    # Explicit narrow extension, needed to compare ice/liquid near the
    # pressure-depressed freezing point. The general factory is unchanged.
    return replace(iapws95_water(), minimum_temperature_K=270)


@dataclass(frozen=True)
class CondensedWaterState:
    temperature_K: float
    pressure_Pa: float
    phase: str
    mass_density_kg_m3: float
    gibbs_energy_J_kg: float
    enthalpy_J_kg: float
    entropy_J_kg_K: float
    cp_J_kg_K: float
    thermal_expansion_K_inv: float
    isothermal_compressibility_Pa_inv: float
    density_pressure_residual_Pa: float
    metastable_against_other_condensed_phase: bool
    selection_scope: str = "pure_condensed_water_air_dissolution_neglected"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _liquid(T: float, p: float) -> CondensedWaterState:
    fluid = _liquid_water_model()

    @lru_cache(maxsize=8)
    def trial(density: float) -> HelmholtzState[PairScalar]:
        try:
            state = fluid.state(temperature_K=PreciseScalar(T),
                                molar_density_mol_m3=PreciseScalar(density)/MW,
                                require_stable=False)
            if state.cv_J_mol_K <= 0 or state.dP_drho_T_Pa_m3_mol <= 0:
                raise ValueError("liquid trial is locally unstable")
            return state
        except (ValueError, OverflowError) as exc:
            raise ResidualDomainError(str(exc)) from exc

    root = solve_root(
        lambda point: [float(trial(float(point[0])).pressure_Pa-p)], [1000.],
        jacobian=lambda point: [[float(trial(float(point[0])).dP_drho_T_Pa_m3_mol/MW)]],
        variables=[ResidualVariable("rho_liquid", "kg/m3", 1000., "algebraic", 700., 1300.)],
        residual_scales=[1e5], policy=RootPolicy(tolerance=1e-10, maximum_iterations=80),
    )
    if not root.success:
        raise ValueError(f"condensed liquid pressure root failed: {root.failure_reason}")
    state = trial(float(root.value[0]))
    error = float(state.pressure_Pa-p)
    if abs(error) > 1e-5:
        raise ValueError("liquid pressure residual exceeds 1e-5 Pa")
    return CondensedWaterState(T, p, "liquid", float(state.mass_density_kg_m3),
                               float(state.gibbs_energy_J_mol/MW), float(state.enthalpy_J_mol/MW),
                               float(state.entropy_J_mol_K/MW), float(state.cp_J_mol_K/MW),
                               float(state.thermal_expansion_K_inv),
                               float(state.isothermal_compressibility_Pa_inv), error, False)


def _ice(T: float, p: float) -> CondensedWaterState:
    state = ice_ih_state(T, p)
    return CondensedWaterState(T, p, "ice", state.mass_density_kg_m3,
                               state.gibbs_energy_J_kg, state.enthalpy_J_kg,
                               state.entropy_J_kg_K, state.cp_J_kg_K,
                               state.thermal_expansion_K_inv,
                               state.isothermal_compressibility_Pa_inv, 0., False)


def condensed_water_state(
    temperature_K: float, pressure_Pa: float, *, phase: str = "auto",
) -> CondensedWaterState:
    """Pure condensate, choosing lower g in the 270--273.16 K overlap.

    Liquid is supported at 270--473 K; ice at 193--273.16 K, p<=5 MPa.
    An explicit surface can be metastable against the other condensed phase.
    This selection does not assert stability against the humid gas reservoir.
    """
    T, p = temperature_K, pressure_Pa
    _validate(.5, T)
    _validate_pressure(p)
    if phase not in ("auto", "liquid", "ice"):
        raise ValueError("condensed phase must be auto, liquid or ice")
    if phase == "liquid" and T < 270:
        raise ValueError("liquid condensate requires T >= 270 K")
    if phase == "ice" and T > 273.16:
        raise ValueError("ice condensate requires T <= 273.16 K")
    if T < 270:
        return _ice(T, p)
    if T > 273.16:
        return _liquid(T, p)
    liquid, ice = _liquid(T, p), _ice(T, p)
    difference = liquid.gibbs_energy_J_kg-ice.gibbs_energy_J_kg
    if phase == "auto" and abs(difference) <= 1e-6:
        raise ValueError("ice/liquid coexistence is ambiguous; specify a condensed phase")
    stable = "ice" if difference > 0 else "liquid"
    selected = stable if phase == "auto" else phase
    return replace(liquid if selected == "liquid" else ice,
                   metastable_against_other_condensed_phase=selected != stable)


def _gas_trial(V: float, T: float, rho: float) -> dict[str, float]:
    """Pressure, water chemical potential and analytic derivatives in V,T,rho."""
    A = 1-V
    vr = _mixture_energy(SecondOrderPair(A, dx=-1), SecondOrderPair(V, dx=1),
                         T, SecondOrderPair(rho, dy=1))
    tr = _mixture_energy(A, V, SecondOrderPair(T, dx=1), SecondOrderPair(rho, dy=1))
    vt = _mixture_energy(SecondOrderPair(A, dx=-1), SecondOrderPair(V, dx=1),
                         SecondOrderPair(T, dy=1), rho)
    p = rho*rho*vr.dy
    dpdr = 2*rho*vr.dy+rho*rho*vr.dyy
    stiffness = vr.dxx-rho*rho*vr.dxy**2/dpdr if dpdr > 0 else -1
    if p <= 0 or dpdr <= 0 or tr.dxx >= 0 or stiffness <= 0:
        raise ResidualDomainError("humid-air equilibrium trial is locally unstable")
    return {"p": p, "mu": vr.value+rho*vr.dy+A*vr.dx,
            "p_rho": dpdr, "p_V": rho*rho*vr.dxy, "p_T": rho*rho*tr.dxy,
            "mu_rho": dpdr/rho+A*vr.dxy,
            "mu_V": rho*vr.dxy+A*vr.dxx,
            "mu_T": tr.dx+rho*tr.dxy+A*vt.dxy,
            "h": vr.value-T*tr.dx+rho*vr.dy,
            "h_T": -T*tr.dxx+rho*tr.dxy,
            "h_V": vr.dx-T*vt.dxy+rho*vr.dxy,
            "h_rho": 2*vr.dy-T*tr.dxy+rho*vr.dyy,
            "u": vr.value-T*tr.dx, "u_T": -T*tr.dxx,
            "u_V": vr.dx-T*vt.dxy, "u_rho": vr.dy-T*tr.dxy}


@dataclass(frozen=True)
class HumidAirSaturation:
    pressure_Pa: float
    gas: HumidAirState
    condensate: CondensedWaterState
    chemical_potential_residual_J_kg: float

    @property
    def temperature_K(self) -> float:  # noqa: N802 - shared SI quantity spelling
        return self.gas.temperature_K

    def to_dict(self) -> dict[str, object]:
        return {"pressure_Pa": self.pressure_Pa, "temperature_K": self.temperature_K,
                "gas": self.gas.to_dict(), "condensate": self.condensate.to_dict(),
                "chemical_potential_residual_J_kg": self.chemical_potential_residual_J_kg}


def _accepted(
    V: float, T: float, rho: float, p: float, condensed: CondensedWaterState,
) -> HumidAirSaturation:
    gas = _state_from_fractions(1-V, V, T, rho)
    error = gas.water_chemical_potential_J_kg-condensed.gibbs_energy_J_kg
    if abs(error) > 1e-5 or abs(gas.pressure_Pa-p) > 5e-11*p:
        raise ValueError("humid-air equilibrium residual exceeds acceptance")
    return HumidAirSaturation(p, gas, condensed, error)


def _water_mass_fraction(mole_fraction: float) -> float:
    return mole_fraction*MW/(mole_fraction*MW+(1-mole_fraction)*MA)


def _water_mole_fraction(V: float) -> float:
    return (V/MW)/(V/MW+(1-V)/MA)


def humid_air_saturation(
    temperature_K: float, pressure_Pa: float, *, condensed_phase: str = "auto",
) -> HumidAirSaturation:
    """Solve equal pressure and water chemical potential; never a vapor-pressure fit."""
    T, p = temperature_K, pressure_Pa
    condensed = condensed_water_state(T, p, phase=condensed_phase)
    # Correlations only seed the coupled EOS solve. Clipping a seed does not
    # approve an unreachable saturation composition (e.g. above boiling).
    vapor_guess = (ice_sublimation_pressure(T) if condensed.phase == "ice"
                   else water_saturation_pressure(max(T, 273.15)))
    V0 = _water_mass_fraction(min(.95, vapor_guess/p))
    rho0 = p/(T*((1-V0)*8.31451/MA+V0*461.51805))

    def trial(point: NDArray[np.float64]) -> tuple[float, float, dict[str, float]]:
        V, rho = exp(point[0]), exp(point[1])
        return V, rho, _gas_trial(V, T, rho)

    def residual(point: NDArray[np.float64]) -> list[float]:
        _, _, gas = trial(point)
        return [gas["p"]-p, gas["mu"]-condensed.gibbs_energy_J_kg]

    def jacobian(point: NDArray[np.float64]) -> list[list[float]]:
        V, rho, gas = trial(point)
        return [[gas["p_V"]*V, gas["p_rho"]*rho],
                [gas["mu_V"]*V, gas["mu_rho"]*rho]]

    root = solve_root(residual, [log(V0), log(rho0)], jacobian=jacobian,
        variables=[ResidualVariable("log_water_fraction", "1", 1., "algebraic",
                                    -80., log(1-1e-12)),
                   ResidualVariable("log_gas_density", "1", 1., "algebraic", -100., log(500.))],
        residual_scales=[p, 1e5], policy=RootPolicy(tolerance=1e-12, maximum_iterations=80))
    if not root.success:
        raise ValueError("no converged humid-air saturation in supported domain: "
                         + root.failure_reason)
    V, rho, _ = trial(root.value)
    return _accepted(V, T, rho, p, condensed)


def humid_air_from_relative_humidity(
    temperature_K: float, pressure_Pa: float, relative_humidity: float,
    *, condensed_phase: str = "auto",
) -> HumidAirState:
    """WMO RH = x_water / x_water,sat, not a humidity-ratio or fugacity ratio."""
    rh = relative_humidity
    if isinstance(rh, bool) or not isfinite(rh) or not 0 <= rh <= 1:
        raise ValueError("relative humidity must be finite and in [0,1]")
    saturated = humid_air_saturation(temperature_K, pressure_Pa, condensed_phase=condensed_phase)
    if rh == 1:
        return saturated.gas
    V = _water_mass_fraction(rh*saturated.gas.water_mole_fraction)
    return humid_air_from_water_fraction(V, temperature_K, pressure_Pa)


def humid_air_relative_humidity(
    water_mass_fraction: float, temperature_K: float, pressure_Pa: float,
    *, condensed_phase: str = "auto",
) -> float:
    """Composition diagnostic; a supersaturated input can return RH > 1."""
    _validate(water_mass_fraction, temperature_K)
    saturated = humid_air_saturation(temperature_K, pressure_Pa, condensed_phase=condensed_phase)
    return _water_mole_fraction(water_mass_fraction)/saturated.gas.water_mole_fraction


def humid_air_condensation_point(
    water_mass_fraction: float, pressure_Pa: float, *, condensed_phase: str = "auto",
) -> HumidAirSaturation:
    """Fixed-composition/pressure dew or frost point, returned with its surface.

    Auto selects the thermodynamically favored condensed phase. Explicit
    liquid/ice selects dew/frost on that surface, with metastability labelled.
    No root in the supported temperature interval is an error, not an extrapolation.
    """
    V, p = water_mass_fraction, pressure_Pa
    _validate(V, 300.)
    _validate_pressure(p)
    if not 0 < V < 1:
        raise ValueError("condensation point requires both air and water; "
                         "dry air has no finite point")
    if condensed_phase not in ("auto", "liquid", "ice"):
        raise ValueError("condensed phase must be auto, liquid or ice")
    lower = 270. if condensed_phase == "liquid" else 193.
    upper = 273.16 if condensed_phase == "ice" else 473.
    partial = _water_mole_fraction(V)*p
    if partial < ice_sublimation_pressure(273.16):
        guess = ice_sublimation_temperature(max(partial, ice_sublimation_pressure(50)))
    else:
        guess = water_saturation_temperature(partial)
    T0 = min(upper, max(lower, guess))
    rho0 = p/(T0*((1-V)*8.31451/MA+V*461.51805))

    def trial(
        point: NDArray[np.float64],
    ) -> tuple[float, float, dict[str, float], CondensedWaterState]:
        T, rho = float(point[0]), exp(point[1])
        try:
            condensed = condensed_water_state(T, p, phase=condensed_phase)
            return T, rho, _gas_trial(V, T, rho), condensed
        except (ValueError, OverflowError) as exc:
            # A nested liquid root or local-stability failure rejects this
            # Newton trial. Let the shared line search shorten the step;
            # never substitute a correlation or an unconverged liquid state.
            raise ResidualDomainError(str(exc)) from exc

    def residual(point: NDArray[np.float64]) -> list[float]:
        _, _, gas, condensed = trial(point)
        return [gas["p"]-p, gas["mu"]-condensed.gibbs_energy_J_kg]

    def jacobian(point: NDArray[np.float64]) -> list[list[float]]:
        _, rho, gas, condensed = trial(point)
        return [[gas["p_T"], gas["p_rho"]*rho],
                [gas["mu_T"]+condensed.entropy_J_kg_K, gas["mu_rho"]*rho]]

    root = solve_root(residual, [T0, log(rho0)], jacobian=jacobian,
        variables=[ResidualVariable("temperature", "K", 300., "algebraic", lower, upper),
                   ResidualVariable("log_gas_density", "1", 1., "algebraic", -100., log(500.))],
        residual_scales=[p, 1e5], policy=RootPolicy(tolerance=1e-12, maximum_iterations=80))
    if not root.success:
        raise ValueError("no converged condensation point in supported domain: "
                         + root.failure_reason)
    T, rho, _, condensed = trial(root.value)
    return _accepted(V, T, rho, p, condensed)
