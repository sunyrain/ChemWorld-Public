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


# A finite, declared set of registered coherent SI readout units. The native
# canonical table also includes non-SI choices such as litres and grams/mole.
COHERENT_SI_UNITS = ("dimensionless", "s", "m", "m^3", "mol", "kg", "K", "A", "Pa", "J", "W",
    "V", "C", "Ohm", "F", "Hz", "mol/m^3", "mol/s", "J/mol", "J/(mol*K)", "kg/m^3",
    "Pa*s", "N/m", "m/s", "m^2/s", "V/K", "m^2", "m^2/A", "A/m^2", "S/m")


_UNIT_TABLE: dict[str, tuple[str, str, float, float]] = {
    "K": ("temperature", "K", 1.0, 0.0),
    "degC": ("temperature", "K", 1.0, 273.15),
    "degF": ("temperature", "K", 5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    "degR": ("temperature", "K", 5.0 / 9.0, 0.0),
    "s": ("time", "s", 1.0, 0.0),
    "min": ("time", "s", 60.0, 0.0),
    "h": ("time", "s", 3600.0, 0.0),
    "L": ("volume", "L", 1.0, 0.0),
    "mL": ("volume", "L", 0.001, 0.0),
    "m^3": ("volume", "L", 1000.0, 0.0),
    "m^3/s": ("volume_flow", "m^3/s", 1.0, 0.0),
    "L/s": ("volume_flow", "m^3/s", 0.001, 0.0),
    "mL/min": ("volume_flow", "m^3/s", 1e-6 / 60.0, 0.0),
    "mol": ("amount", "mol", 1.0, 0.0),
    "mol_eq": ("chemical_equivalent", "mol_eq", 1.0, 0.0),
    "mol_eq/m^3": ("equivalent_concentration", "mol_eq/m^3", 1.0, 0.0),
    "mol/s": ("amount_rate", "mol/s", 1.0, 0.0),
    "mmol": ("amount", "mol", 0.001, 0.0),
    "mol/L": ("concentration", "mol/L", 1.0, 0.0),
    "mol/m^3": ("concentration", "mol/L", 0.001, 0.0),
    "m/s": ("velocity", "m/s", 1.0, 0.0),
    "m^2": ("area", "m^2", 1.0, 0.0),
    "m^2/A": ("inverse_current_density", "m^2/A", 1.0, 0.0),
    "s/m^(1/3)": ("manning_roughness", "s/m^(1/3)", 1.0, 0.0),
    "m^0.5/s": ("chezy_coefficient", "m^0.5/s", 1.0, 0.0),
    "kg": ("mass", "kg", 1.0, 0.0),
    "kg/s": ("mass_flow", "kg/s", 1.0, 0.0),
    "kg/h": ("mass_flow", "kg/s", 1.0 / 3600.0, 0.0),
    "g/s": ("mass_flow", "kg/s", 0.001, 0.0),
    "g": ("mass", "kg", 0.001, 0.0),
    "J": ("energy", "J", 1.0, 0.0),
    "kJ": ("energy", "J", 1000.0, 0.0),
    "W": ("power", "W", 1.0, 0.0),
    "kW": ("power", "W", 1000.0, 0.0),
    "J/K": ("heat_capacity", "J/K", 1.0, 0.0),
    "kJ/K": ("heat_capacity", "J/K", 1000.0, 0.0),
    "W/K": ("thermal_conductance", "W/K", 1.0, 0.0),
    "mW/K": ("thermal_conductance", "W/K", 0.001, 0.0),
    "J/mol": ("molar_enthalpy", "J/mol", 1.0, 0.0),
    "kJ/mol": ("molar_enthalpy", "J/mol", 1000.0, 0.0),
    "J/(mol*K)": ("molar_heat_capacity", "J/(mol*K)", 1.0, 0.0),
    "J/(kg*K)": ("specific_heat_capacity", "J/(kg*K)", 1.0, 0.0),
    "kJ/(mol*K)": ("molar_heat_capacity", "J/(mol*K)", 1000.0, 0.0),
    "g/mol": ("molecular_weight", "g/mol", 1.0, 0.0),
    "kg/mol": ("molecular_weight", "g/mol", 1000.0, 0.0),
    "m^3/mol": ("molar_volume", "m^3/mol", 1.0, 0.0),
    "m^6/mol^2": ("third_virial_coefficient", "m^6/mol^2", 1.0, 0.0),
    "L^2/mol^2": ("third_virial_coefficient", "m^6/mol^2", 1e-6, 0.0),
    "m^6/kmol^2": ("third_virial_coefficient", "m^6/mol^2", 1e-6, 0.0),
    "m^3/(mol*K)": ("molar_volume_per_temperature", "m^3/(mol*K)", 1.0, 0.0),
    "(m^3/mol)/K": ("molar_volume_per_temperature", "m^3/(mol*K)", 1.0, 0.0),
    "L/(mol*K)": ("molar_volume_per_temperature", "m^3/(mol*K)", 0.001, 0.0),
    "(L/mol)/K": ("molar_volume_per_temperature", "m^3/(mol*K)", 0.001, 0.0),
    "m^3/(mol*Pa)": ("molar_volume_per_pressure", "m^3/(mol*Pa)", 1.0, 0.0),
    "L/mol": ("molar_volume", "m^3/mol", 0.001, 0.0),
    "kg/m^3": ("mass_density", "kg/m^3", 1.0, 0.0),
    "mg/m^3": ("mass_density", "kg/m^3", 1e-6, 0.0),
    "g/mL": ("mass_density", "kg/m^3", 1000.0, 0.0),
    "kg/(m^3*K)": ("mass_density_per_temperature", "kg/(m^3*K)", 1.0, 0.0),
    "(kg/m^3)/K": ("mass_density_per_temperature", "kg/(m^3*K)", 1.0, 0.0),
    "g/(mL*K)": ("mass_density_per_temperature", "kg/(m^3*K)", 1000.0, 0.0),
    "(g/mL)/K": ("mass_density_per_temperature", "kg/(m^3*K)", 1000.0, 0.0),
    "Pa*s": ("dynamic_viscosity", "Pa*s", 1.0, 0.0),
    "mPa*s": ("dynamic_viscosity", "Pa*s", 0.001, 0.0),
    "Pa*s/K": ("viscosity_per_temperature", "Pa*s/K", 1.0, 0.0),
    "(Pa*s)/K": ("viscosity_per_temperature", "Pa*s/K", 1.0, 0.0),
    "mPa*s/K": ("viscosity_per_temperature", "Pa*s/K", 0.001, 0.0),
    "(mPa*s)/K": ("viscosity_per_temperature", "Pa*s/K", 0.001, 0.0),
    "W/(m*K)": ("thermal_conductivity", "W/(m*K)", 1.0, 0.0),
    "mW/(m*K)": ("thermal_conductivity", "W/(m*K)", 0.001, 0.0),
    "W/(m*K^2)": ("conductivity_per_temperature", "W/(m*K^2)", 1.0, 0.0),
    "(W/(m*K))/K": ("conductivity_per_temperature", "W/(m*K^2)", 1.0, 0.0),
    "mW/(m*K^2)": ("conductivity_per_temperature", "W/(m*K^2)", 0.001, 0.0),
    "(mW/(m*K))/K": ("conductivity_per_temperature", "W/(m*K^2)", 0.001, 0.0),
    "m^2/s": ("diffusivity", "m^2/s", 1.0, 0.0),
    "cm^2/s": ("diffusivity", "m^2/s", 0.0001, 0.0),
    "N/m": ("surface_tension", "N/m", 1.0, 0.0),
    "mN/m": ("surface_tension", "N/m", 0.001, 0.0),
    "N/(m*K)": ("surface_tension_per_temperature", "N/(m*K)", 1.0, 0.0),
    "(N/m)/K": ("surface_tension_per_temperature", "N/(m*K)", 1.0, 0.0),
    "mN/(m*K)": ("surface_tension_per_temperature", "N/(m*K)", 0.001, 0.0),
    "(mN/m)/K": ("surface_tension_per_temperature", "N/(m*K)", 0.001, 0.0),
    "W/(m^2*K)": ("heat_transfer_coefficient", "W/(m^2*K)", 1.0, 0.0),
    "Pa": ("pressure", "Pa", 1.0, 0.0),
    "kPa": ("pressure", "Pa", 1000.0, 0.0),
    "MPa": ("pressure", "Pa", 1e6, 0.0),
    "bar": ("pressure", "Pa", 100000.0, 0.0),
    "mmHg": ("pressure", "Pa", 133.32236842105263, 0.0),
    "A": ("electrical_current", "A", 1.0, 0.0),
    "mA": ("electrical_current", "A", 0.001, 0.0),
    "uA": ("electrical_current", "A", 1e-6, 0.0),
    "A/m^2": ("electrical_current_density", "A/m^2", 1.0, 0.0),
    "A*m^(5/2)/mol^(3/2)": (
        "intercalation_rate_coefficient", "A*m^(5/2)/mol^(3/2)", 1.0, 0.0
    ),
    "mA/cm^2": ("electrical_current_density", "A/m^2", 10.0, 0.0),
    "S/m": ("electrical_conductivity", "S/m", 1.0, 0.0),
    "S/cm": ("electrical_conductivity", "S/m", 100.0, 0.0),
    "V": ("electric_potential", "V", 1.0, 0.0),
    "mV": ("electric_potential", "V", 0.001, 0.0),
    "V/K": ("electric_potential_per_temperature", "V/K", 1.0, 0.0),
    "mV/K": ("electric_potential_per_temperature", "V/K", 0.001, 0.0),
    "pH": ("acidity", "pH", 1.0, 0.0),
    "C": ("electric_charge", "C", 1.0, 0.0),
    "Ah": ("electric_charge", "C", 3600.0, 0.0),
    "mAh": ("electric_charge", "C", 3.6, 0.0),
    "Wh": ("energy", "J", 3600.0, 0.0),
    "mC": ("electric_charge", "C", 0.001, 0.0),
    "Ohm": ("electrical_resistance", "Ohm", 1.0, 0.0),
    "Ohm*m": ("electrical_resistivity", "Ohm*m", 1.0, 0.0),
    "kOhm": ("electrical_resistance", "Ohm", 1000.0, 0.0),
    "F": ("capacitance", "F", 1.0, 0.0),
    "F/m^2": ("surface_capacitance", "F/m^2", 1.0, 0.0),
    "uF": ("capacitance", "F", 1e-6, 0.0),
    "Hz": ("frequency", "Hz", 1.0, 0.0),
    # Inverse-time rates share T^-1; input tables may retain rate notation.
    "1/s": ("frequency", "Hz", 1.0, 0.0),
    "s^-1": ("frequency", "Hz", 1.0, 0.0),
    "ppm": ("chemical_shift", "ppm", 1.0, 0.0),
    "ppmv": ("gas_volume_fraction", "gas_volume_fraction", 1e-6, 0.0),
    "gas_volume_fraction": ("gas_volume_fraction", "gas_volume_fraction", 1.0, 0.0),
    "nm": ("length", "nm", 1.0, 0.0),
    "m": ("length", "nm", 1e9, 0.0),
    "m/K": ("length_per_temperature", "m/K", 1.0, 0.0),
    "mm": ("length", "nm", 1e6, 0.0),
    "inch": ("length", "nm", 25.4e6, 0.0),
    "um": ("length", "nm", 1e3, 0.0),
    "angstrom": ("length", "nm", 0.1, 0.0),
    "1/K": ("thermal_expansivity", "1/K", 1.0, 0.0),
    "(dimensionless)/K": ("thermal_expansivity", "1/K", 1.0, 0.0),
    "degBx": ("brix", "degBx", 1.0, 0.0),
    "C*m": ("electric_dipole_moment", "C*m", 1.0, 0.0),
    "Debye": ("electric_dipole_moment", "C*m", 1e-21 / 299792458.0, 0.0),
    "Pa^0.5": ("solubility_parameter", "Pa^0.5", 1.0, 0.0),
    "MPa^0.5": ("solubility_parameter", "Pa^0.5", 1000.0, 0.0),
    "kg/kmol": ("molecular_weight", "g/mol", 1.0, 0.0),
    "m^3/kmol": ("molar_volume", "m^3/mol", 0.001, 0.0),
    "J/kmol": ("molar_enthalpy", "J/mol", 0.001, 0.0),
    "J/(kmol*K)": ("molar_heat_capacity", "J/(mol*K)", 0.001, 0.0),
    "m^2/mol": ("molar_area", "m^2/mol", 1.0, 0.0),
    "m^2/kmol": ("molar_area", "m^2/mol", 0.001, 0.0),
    "kg^0.25*m^3/(s^0.5*mol)": ("parachor", "kg^0.25*m^3/(s^0.5*mol)", 1.0, 0.0),
    "kg^0.25*m^3/(s^0.5*kmol)": ("parachor", "kg^0.25*m^3/(s^0.5*mol)", 0.001, 0.0),
    "eV": ("energy", "J", 1.602176634e-19, 0.0),
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


# Temperature derivatives use intervals in kelvin (no absolute-temperature offset).
# Derive pressure scales from the same table used by the original scalar quantity.
_UNIT_TABLE.update({
    f"1/{pressure_unit}": ("compressibility", "1/Pa", 1.0 / scale, 0.0)
    for pressure_unit, (dimension, _, scale, _) in tuple(_UNIT_TABLE.items())
    if dimension == "pressure"
})

_UNIT_TABLE.update({
    name: (f"pressure_temperature_derivative_{order}", f"Pa/{suffix}", scale, 0.0)
    for pressure_unit, (dimension, _, scale, _) in tuple(_UNIT_TABLE.items())
    if dimension == "pressure"
    for order, suffix in ((1, "K"), (2, "K^2"), (3, "K^3"))
    for name in (f"{pressure_unit}/{suffix}", f"({pressure_unit})/{suffix}")
})


# Same source scales for B/C temperature derivatives, including existing dB/dT.
for _base, (_dimension, _canonical, _scale, _) in tuple(_UNIT_TABLE.items()):
    if _dimension in {"molar_volume", "third_virial_coefficient"}:
        for _order in (1, 2, 3):
            _suffix = "K" if _order == 1 else f"K^{_order}"
            _key = f"({_base})/{_suffix}"
            _unit = ("m^3/(mol*K)" if _dimension == "molar_volume" and _order == 1
                     else f"({_canonical})/{_suffix}")
            _dim = ("molar_volume_per_temperature" if _dimension == "molar_volume" and _order == 1
                    else f"{_dimension}_temperature_derivative_{_order}")
            _UNIT_TABLE[_key] = (_dim, _unit, _scale, 0.)


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
