"""Constant-gamma gas engineering, with explicit molar/mass and work signs.

Positive work enters the gas; an expander returns negative actual work.
These constitutive relations do not create a second energy/inventory runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, expm1, isfinite, log, log1p, sqrt
from typing import Any, Literal

from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.property_reports import R_J_PER_MOL_K

GasProcess = Literal["compression", "expansion"]


def _gamma_exponent(heat_capacity_ratio: float) -> float:
    if not isfinite(heat_capacity_ratio) or heat_capacity_ratio <= 1:
        raise ValueError("heat capacity ratio must be finite and greater than one")
    return (heat_capacity_ratio - 1) / heat_capacity_ratio


def _efficiency(value: float) -> None:
    if not isfinite(value) or not 0 < value <= 1:
        raise ValueError("efficiency must lie in (0, 1]")


def _pressure_log(p1: float, p2: float, process: GasProcess | None = None) -> float:
    _positive(p1, "inlet absolute pressure")
    _positive(p2, "outlet absolute pressure")
    if process is not None:
        if process not in {"compression", "expansion"}:
            raise ValueError("process must be compression or expansion")
        if (process == "compression" and p2 < p1) or (process == "expansion" and p2 > p1):
            raise ValueError("pressure direction conflicts with the declared process")
    return log1p((p2 - p1) / p1) if p2 > 0.5 * p1 else log(p2 / p1)


def isothermal_gas_work_molar(
    *,
    inlet_pressure_Pa: float,
    outlet_pressure_Pa: float,
    temperature_K: float,
    compressibility_factor: float = 1.0,
) -> float:
    """Reversible constant-T work [J/mol]; it is not an isothermal enthalpy rise."""
    _positive(temperature_K, "temperature")
    _positive(compressibility_factor, "compressibility factor")
    return (
        compressibility_factor
        * R_J_PER_MOL_K
        * temperature_K
        * _pressure_log(inlet_pressure_Pa, outlet_pressure_Pa)
    )


def isentropic_gas_work_molar(
    *,
    inlet_pressure_Pa: float,
    outlet_pressure_Pa: float,
    inlet_temperature_K: float,
    heat_capacity_ratio: float,
    efficiency: float = 1.0,
    compressibility_factor: float = 1.0,
    process: GasProcess = "compression",
) -> float:
    """Actual constant-gamma shaft work [J/mol], positive for compression."""
    exponent = _gamma_exponent(heat_capacity_ratio)
    _positive(inlet_temperature_K, "inlet temperature")
    _positive(compressibility_factor, "compressibility factor")
    _efficiency(efficiency)
    reversible = compressibility_factor * R_J_PER_MOL_K * inlet_temperature_K / exponent
    reversible *= expm1(exponent * _pressure_log(inlet_pressure_Pa, outlet_pressure_Pa, process))
    return reversible / efficiency if process == "compression" else reversible * efficiency


def solve_isentropic_gas_work(
    *,
    inlet_temperature_K: float,
    heat_capacity_ratio: float,
    inlet_pressure_Pa: float | None = None,
    outlet_pressure_Pa: float | None = None,
    work_J_mol: float | None = None,
    efficiency: float | None = None,
    compressibility_factor: float = 1.0,
    process: GasProcess = "compression",
) -> float:
    """Return the single unknown among P1, P2, actual work and efficiency."""
    if sum(x is None for x in (inlet_pressure_Pa, outlet_pressure_Pa, work_J_mol, efficiency)) != 1:
        raise ValueError("declare exactly one unknown P1/P2/work/efficiency")
    exponent = _gamma_exponent(heat_capacity_ratio)
    _positive(inlet_temperature_K, "inlet temperature")
    _positive(compressibility_factor, "compressibility factor")
    if process not in {"compression", "expansion"}:
        raise ValueError("process must be compression or expansion")
    if work_J_mol is None:
        if inlet_pressure_Pa is None or outlet_pressure_Pa is None or efficiency is None:
            raise AssertionError(
                "invariant violated: inlet_pressure_Pa or outlet_pressure_Pa or efficiency is None"
            )
        return isentropic_gas_work_molar(
            inlet_pressure_Pa=inlet_pressure_Pa,
            outlet_pressure_Pa=outlet_pressure_Pa,
            inlet_temperature_K=inlet_temperature_K,
            heat_capacity_ratio=heat_capacity_ratio,
            efficiency=efficiency,
            compressibility_factor=compressibility_factor,
            process=process,
        )
    if (
        not isfinite(work_J_mol)
        or (process == "compression" and work_J_mol < 0)
        or (process == "expansion" and work_J_mol > 0)
    ):
        raise ValueError("work sign conflicts with the declared process")
    if efficiency is None:
        if inlet_pressure_Pa is None or outlet_pressure_Pa is None:
            raise AssertionError(
                "invariant violated: inlet_pressure_Pa or outlet_pressure_Pa is None"
            )
        reversible = isentropic_gas_work_molar(
            inlet_pressure_Pa=inlet_pressure_Pa,
            outlet_pressure_Pa=outlet_pressure_Pa,
            inlet_temperature_K=inlet_temperature_K,
            heat_capacity_ratio=heat_capacity_ratio,
            compressibility_factor=compressibility_factor,
            process=process,
        )
        if reversible == 0 or work_J_mol == 0:
            raise ValueError("efficiency is indeterminate at zero work")
        value = reversible / work_J_mol if process == "compression" else work_J_mol / reversible
        _efficiency(value)
        return value
    _efficiency(efficiency)
    reversible = work_J_mol * efficiency if process == "compression" else work_J_mol / efficiency
    dimensionless = (
        reversible * exponent / (compressibility_factor * R_J_PER_MOL_K * inlet_temperature_K)
    )
    if dimensionless <= -1:
        raise ValueError("work would require nonpositive outlet temperature")
    ratio = exp(log1p(dimensionless) / exponent)
    if inlet_pressure_Pa is None:
        if outlet_pressure_Pa is None:
            raise AssertionError("invariant violated: outlet_pressure_Pa is not None")
        _positive(outlet_pressure_Pa, "outlet absolute pressure")
        return outlet_pressure_Pa / ratio
    _positive(inlet_pressure_Pa, "inlet absolute pressure")
    return inlet_pressure_Pa * ratio


def isentropic_gas_outlet_temperature(
    *,
    inlet_temperature_K: float,
    inlet_pressure_Pa: float,
    outlet_pressure_Pa: float,
    heat_capacity_ratio: float,
    efficiency: float = 1.0,
    process: GasProcess = "compression",
) -> float:
    exponent = _gamma_exponent(heat_capacity_ratio)
    _positive(inlet_temperature_K, "inlet temperature")
    _efficiency(efficiency)
    rise = expm1(exponent * _pressure_log(inlet_pressure_Pa, outlet_pressure_Pa, process))
    rise = rise / efficiency if process == "compression" else rise * efficiency
    return inlet_temperature_K * (1 + rise)


def gas_efficiency_conversion(
    *,
    inlet_pressure_Pa: float,
    outlet_pressure_Pa: float,
    heat_capacity_ratio: float,
    isentropic_efficiency: float | None = None,
    polytropic_efficiency: float | None = None,
    process: GasProcess = "compression",
) -> float:
    """Return the missing isentropic or polytropic efficiency for this pressure ratio."""
    if (isentropic_efficiency is None) == (polytropic_efficiency is None):
        raise ValueError("declare exactly one efficiency")
    powered_log = _gamma_exponent(heat_capacity_ratio) * _pressure_log(
        inlet_pressure_Pa, outlet_pressure_Pa, process
    )
    given = isentropic_efficiency if isentropic_efficiency is not None else polytropic_efficiency
    if given is None:
        raise AssertionError("invariant violated: given is not None")
    _efficiency(given)
    if powered_log == 0:
        return given  # continuous zero-pressure-ratio limit
    ideal_change = expm1(powered_log)
    if isentropic_efficiency is None:
        if process == "compression":
            return ideal_change / expm1(powered_log / given)
        return expm1(powered_log * given) / ideal_change
    if process == "compression":
        return powered_log / log1p(ideal_change / given)
    return log1p(given * ideal_change) / powered_log


def gas_polytropic_exponent(
    *,
    heat_capacity_ratio: float,
    efficiency: float | None = None,
    exponent: float | None = None,
    process: GasProcess = "compression",
) -> float:
    """Convert polytropic efficiency to exponent n, or n to efficiency."""
    a = _gamma_exponent(heat_capacity_ratio)
    if process not in {"compression", "expansion"}:
        raise ValueError("process must be compression or expansion")
    if (efficiency is None) == (exponent is None):
        raise ValueError("declare exactly one polytropic efficiency or exponent")
    if efficiency is not None:
        _efficiency(efficiency)
        if process == "compression":
            if efficiency <= a:
                raise ValueError(
                    "compressor efficiency must exceed (gamma-1)/gamma for finite positive n"
                )
            return 1 / (1 - a / efficiency)
        return 1 / (1 - a * efficiency)
    if exponent is None:
        raise AssertionError("invariant violated: exponent is not None")
    if not isfinite(exponent) or exponent <= 1:
        raise ValueError("polytropic exponent must exceed one")
    value = a * exponent / (exponent - 1) if process == "compression" else (1 - 1 / exponent) / a
    if value > 1 and value < 1 + 1e-14:
        value = 1.0
    _efficiency(value)
    return value


def critical_gas_pressure(*, stagnation_pressure_Pa: float, heat_capacity_ratio: float) -> float:
    a = _gamma_exponent(heat_capacity_ratio)
    _positive(stagnation_pressure_Pa, "stagnation absolute pressure")
    return stagnation_pressure_Pa * (2 / (heat_capacity_ratio + 1)) ** (1 / a)


def critical_gas_temperature(
    *, stagnation_temperature_K: float, heat_capacity_ratio: float
) -> float:
    _gamma_exponent(heat_capacity_ratio)
    _positive(stagnation_temperature_K, "stagnation temperature")
    return stagnation_temperature_K * 2 / (heat_capacity_ratio + 1)


def is_critical_gas_flow(
    *,
    stagnation_pressure_Pa: float,
    downstream_pressure_Pa: float,
    heat_capacity_ratio: float,
) -> bool:
    if not isfinite(downstream_pressure_Pa) or downstream_pressure_Pa < 0:
        raise ValueError("downstream absolute pressure must be finite and nonnegative")
    return downstream_pressure_Pa <= critical_gas_pressure(
        stagnation_pressure_Pa=stagnation_pressure_Pa, heat_capacity_ratio=heat_capacity_ratio
    )


def stagnation_energy_J_kg(velocity_m_s: float) -> float:  # noqa: N802
    if not isfinite(velocity_m_s):
        raise ValueError("velocity must be finite")
    return velocity_m_s**2 / 2


def stagnation_temperature_from_velocity(
    *,
    static_temperature_K: float,
    velocity_m_s: float,
    heat_capacity_J_kg_K: float,
) -> float:
    _positive(static_temperature_K, "static temperature")
    _positive(heat_capacity_J_kg_K, "mass heat capacity")
    return static_temperature_K + stagnation_energy_J_kg(velocity_m_s) / heat_capacity_J_kg_K


def gas_stagnation_pressure(
    *,
    static_pressure_Pa: float,
    static_temperature_K: float,
    stagnation_temperature_K: float,
    heat_capacity_ratio: float,
) -> float:
    a = _gamma_exponent(heat_capacity_ratio)
    _positive(static_pressure_Pa, "static absolute pressure")
    _positive(static_temperature_K, "static temperature")
    _positive(stagnation_temperature_K, "stagnation temperature")
    if stagnation_temperature_K < static_temperature_K:
        raise ValueError("stagnation temperature cannot be below static temperature")
    return static_pressure_Pa * (stagnation_temperature_K / static_temperature_K) ** (1 / a)


def gas_stagnation_temperature(
    *,
    static_pressure_Pa: float,
    static_temperature_K: float,
    stagnation_pressure_Pa: float,
    heat_capacity_ratio: float,
) -> float:
    a = _gamma_exponent(heat_capacity_ratio)
    _positive(static_temperature_K, "static temperature")
    return static_temperature_K * exp(
        a * _pressure_log(static_pressure_Pa, stagnation_pressure_Pa, "compression")
    )


@dataclass(frozen=True)
class GasNozzleResult:
    mass_flow_kg_s: float
    area_m2: float
    mass_flux_kg_m2_s: float
    exit_pressure_Pa: float
    exit_temperature_K: float
    critical_pressure_Pa: float
    choked: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ideal_gas_nozzle(
    *,
    stagnation_pressure_Pa: float,
    stagnation_temperature_K: float,
    downstream_pressure_Pa: float,
    heat_capacity_ratio: float,
    specific_gas_constant_J_kg_K: float,
    discharge_coefficient: float = 1.0,
    area_m2: float | None = None,
    mass_flow_kg_s: float | None = None,
) -> GasNozzleResult:
    """Isentropic converging nozzle, one unknown area/flow, fixed Cd, no shock model."""
    if (area_m2 is None) == (mass_flow_kg_s is None):
        raise ValueError("declare exactly one area or mass flow")
    _positive(stagnation_temperature_K, "stagnation temperature")
    _positive(specific_gas_constant_J_kg_K, "specific gas constant")
    _efficiency(discharge_coefficient)
    critical = critical_gas_pressure(
        stagnation_pressure_Pa=stagnation_pressure_Pa, heat_capacity_ratio=heat_capacity_ratio
    )
    if (
        not isfinite(downstream_pressure_Pa)
        or not 0 <= downstream_pressure_Pa <= stagnation_pressure_Pa
    ):
        raise ValueError("nozzle downstream absolute pressure must be between zero and upstream")
    a = _gamma_exponent(heat_capacity_ratio)
    exit_pressure = max(critical, downstream_pressure_Pa)
    log_ratio = log(exit_pressure / stagnation_pressure_Pa)
    thermal_energy = specific_gas_constant_J_kg_K * stagnation_temperature_K
    flux = discharge_coefficient * stagnation_pressure_Pa * exp(log_ratio / heat_capacity_ratio)
    flux *= sqrt(-2 * expm1(a * log_ratio) / (a * thermal_energy))
    if area_m2 is None:
        if mass_flow_kg_s is None:
            raise AssertionError("invariant violated: mass_flow_kg_s is not None")
        _positive(mass_flow_kg_s, "specified mass flow")
        if flux == 0:
            raise ValueError("positive flow cannot be sized at zero pressure difference")
        area_m2 = mass_flow_kg_s / flux
    else:
        _positive(area_m2, "area")
        mass_flow_kg_s = area_m2 * flux
    return GasNozzleResult(
        mass_flow_kg_s,
        area_m2,
        flux,
        exit_pressure,
        stagnation_temperature_K * exp(a * log_ratio),
        critical,
        downstream_pressure_Pa <= critical,
    )
