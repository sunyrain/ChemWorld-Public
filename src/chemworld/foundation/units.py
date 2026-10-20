"""Lightweight canonical unit system for JSON-friendly benchmark state."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitSpec:
    """A supported physical unit and its canonical dimension."""

    unit: str
    dimension: str
    canonical_unit: str


@dataclass(frozen=True)
class Quantity:
    value: float
    unit: str

    def to(self, target_unit: str) -> Quantity:
        return Quantity(convert_value(self.value, self.unit, target_unit), target_unit)

    def to_dict(self) -> dict[str, float | str]:
        return {"value": self.value, "unit": self.unit}


_UNIT_TABLE: dict[str, tuple[str, str, float, float]] = {
    "K": ("temperature", "K", 1.0, 0.0),
    "degC": ("temperature", "K", 1.0, 273.15),
    "s": ("time", "s", 1.0, 0.0),
    "min": ("time", "s", 60.0, 0.0),
    "h": ("time", "s", 3600.0, 0.0),
    "L": ("volume", "L", 1.0, 0.0),
    "mL": ("volume", "L", 0.001, 0.0),
    "m^3/s": ("volume_flow", "m^3/s", 1.0, 0.0),
    "L/s": ("volume_flow", "m^3/s", 0.001, 0.0),
    "mL/min": ("volume_flow", "m^3/s", 1e-6 / 60.0, 0.0),
    "mol": ("amount", "mol", 1.0, 0.0),
    "mmol": ("amount", "mol", 0.001, 0.0),
    "mol/L": ("concentration", "mol/L", 1.0, 0.0),
    "mol/m^3": ("concentration", "mol/L", 0.001, 0.0),
    "m/s": ("velocity", "m/s", 1.0, 0.0),
    "kg": ("mass", "kg", 1.0, 0.0),
    "g": ("mass", "kg", 0.001, 0.0),
    "J": ("energy", "J", 1.0, 0.0),
    "kJ": ("energy", "J", 1000.0, 0.0),
    "W": ("power", "W", 1.0, 0.0),
    "kW": ("power", "W", 1000.0, 0.0),
    "J/mol": ("molar_enthalpy", "J/mol", 1.0, 0.0),
    "kJ/mol": ("molar_enthalpy", "J/mol", 1000.0, 0.0),
    "J/(mol*K)": ("molar_heat_capacity", "J/(mol*K)", 1.0, 0.0),
    "kJ/(mol*K)": ("molar_heat_capacity", "J/(mol*K)", 1000.0, 0.0),
    "g/mol": ("molecular_weight", "g/mol", 1.0, 0.0),
    "m^3/mol": ("molar_volume", "m^3/mol", 1.0, 0.0),
    "m^3/(mol*K)": ("molar_volume_per_temperature", "m^3/(mol*K)", 1.0, 0.0),
    "m^3/(mol*Pa)": ("molar_volume_per_pressure", "m^3/(mol*Pa)", 1.0, 0.0),
    "L/mol": ("molar_volume", "m^3/mol", 0.001, 0.0),
    "kg/m^3": ("mass_density", "kg/m^3", 1.0, 0.0),
    "g/mL": ("mass_density", "kg/m^3", 1000.0, 0.0),
    "Pa*s": ("dynamic_viscosity", "Pa*s", 1.0, 0.0),
    "mPa*s": ("dynamic_viscosity", "Pa*s", 0.001, 0.0),
    "W/(m*K)": ("thermal_conductivity", "W/(m*K)", 1.0, 0.0),
    "mW/(m*K)": ("thermal_conductivity", "W/(m*K)", 0.001, 0.0),
    "m^2/s": ("diffusivity", "m^2/s", 1.0, 0.0),
    "cm^2/s": ("diffusivity", "m^2/s", 0.0001, 0.0),
    "N/m": ("surface_tension", "N/m", 1.0, 0.0),
    "mN/m": ("surface_tension", "N/m", 0.001, 0.0),
    "W/(m^2*K)": ("heat_transfer_coefficient", "W/(m^2*K)", 1.0, 0.0),
    "Pa": ("pressure", "Pa", 1.0, 0.0),
    "bar": ("pressure", "Pa", 100000.0, 0.0),
    "mmHg": ("pressure", "Pa", 133.32236842105263, 0.0),
    "A": ("electrical_current", "A", 1.0, 0.0),
    "mA": ("electrical_current", "A", 0.001, 0.0),
    "uA": ("electrical_current", "A", 1e-6, 0.0),
    "V": ("electric_potential", "V", 1.0, 0.0),
    "mV": ("electric_potential", "V", 0.001, 0.0),
    "pH": ("acidity", "pH", 1.0, 0.0),
    "C": ("electric_charge", "C", 1.0, 0.0),
    "mC": ("electric_charge", "C", 0.001, 0.0),
    "Ohm": ("electrical_resistance", "Ohm", 1.0, 0.0),
    "kOhm": ("electrical_resistance", "Ohm", 1000.0, 0.0),
    "F": ("capacitance", "F", 1.0, 0.0),
    "uF": ("capacitance", "F", 1e-6, 0.0),
    "Hz": ("frequency", "Hz", 1.0, 0.0),
    "ppm": ("chemical_shift", "ppm", 1.0, 0.0),
    "nm": ("length", "nm", 1.0, 0.0),
    "cm^-1": ("wavenumber", "cm^-1", 1.0, 0.0),
    "cm-1": ("wavenumber", "cm^-1", 1.0, 0.0),
    "relative_signal": ("relative_signal", "relative_signal", 1.0, 0.0),
    "relative_detector_response": ("relative_signal", "relative_signal", 1.0, 0.0),
    "relative_intensity": ("relative_signal", "relative_signal", 1.0, 0.0),
    "a.u.": ("relative_signal", "relative_signal", 1.0, 0.0),
    "absorbance": ("absorbance", "AU", 1.0, 0.0),
    "transmittance_fraction": ("dimensionless", "dimensionless", 1.0, 0.0),
    "m/z": ("mass_to_charge", "m/z", 1.0, 0.0),
    "AU": ("absorbance", "AU", 1.0, 0.0),
    "counts": ("detector_response", "counts", 1.0, 0.0),
    "AU*s": ("integrated_detector_response", "AU*s", 1.0, 0.0),
    "currency": ("cost", "currency", 1.0, 0.0),
    "risk": ("risk", "risk", 1.0, 0.0),
    "dimensionless": ("dimensionless", "dimensionless", 1.0, 0.0),
}


def unit_spec(unit: str) -> UnitSpec:
    if unit not in _UNIT_TABLE:
        raise ValueError(f"Unsupported unit: {unit}")
    dimension, canonical_unit, _, _ = _UNIT_TABLE[unit]
    return UnitSpec(unit=unit, dimension=dimension, canonical_unit=canonical_unit)


def canonical_unit(unit: str) -> str:
    return unit_spec(unit).canonical_unit


def convert_value(value: float, source_unit: str, target_unit: str) -> float:
    """Convert between supported units with affine temperature handling."""

    if source_unit not in _UNIT_TABLE:
        raise ValueError(f"Unsupported source unit: {source_unit}")
    if target_unit not in _UNIT_TABLE:
        raise ValueError(f"Unsupported target unit: {target_unit}")

    source_dimension, _, source_scale, source_offset = _UNIT_TABLE[source_unit]
    target_dimension, _, target_scale, target_offset = _UNIT_TABLE[target_unit]
    if source_dimension != target_dimension:
        raise ValueError(f"Cannot convert {source_unit} to {target_unit}")

    canonical_value = value * source_scale + source_offset
    return (canonical_value - target_offset) / target_scale


def supported_units() -> dict[str, UnitSpec]:
    return {unit: unit_spec(unit) for unit in sorted(_UNIT_TABLE)}
