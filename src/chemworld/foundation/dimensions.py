"""Canonical dimension contracts for physical and instrument quantities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from chemworld.foundation.units import supported_units, unit_spec


@dataclass(frozen=True)
class DimensionVector:
    """Finite exponents over ChemWorld's physical and semantic base axes.

    Integral exponents retain their original JSON representation. Fractional
    powers support quantities such as solubility parameters and parachor.
    """

    mass: int | float = 0
    length: int | float = 0
    time: int | float = 0
    temperature: int | float = 0
    amount: int | float = 0
    current: int | float = 0
    signal: int | float = 0
    currency: int | float = 0
    risk: int | float = 0

    def __post_init__(self) -> None:
        for name, value in self.to_dict().items():
            object.__setattr__(self, name, _exponent(value))

    def __mul__(self, other: DimensionVector) -> DimensionVector:
        return self._combine(other, 1)

    def __truediv__(self, other: DimensionVector) -> DimensionVector:
        return self._combine(other, -1)

    def __pow__(self, exponent: int | float) -> DimensionVector:
        exponent = _exponent(exponent)
        return DimensionVector(**{name: value * exponent for name, value in self.to_dict().items()})

    def _combine(self, other: DimensionVector, sign: int) -> DimensionVector:
        if not isinstance(other, DimensionVector):
            return NotImplemented
        left = self.to_dict()
        right = other.to_dict()
        return DimensionVector(**{name: left[name] + sign * right[name] for name in left})

    def to_dict(self) -> dict[str, int | float]:
        return {
            "mass": self.mass,
            "length": self.length,
            "time": self.time,
            "temperature": self.temperature,
            "amount": self.amount,
            "current": self.current,
            "signal": self.signal,
            "currency": self.currency,
            "risk": self.risk,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DimensionVector:
        values = {name: payload.get(name, 0) for name in cls().to_dict()}
        return cls(**{name: float(v) if isinstance(v, str) else v for name, v in values.items()})


def _exponent(value: int | float) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("dimension exponent must be a finite number")
    if not isfinite(value):
        raise ValueError("dimension exponent must be finite")
    return int(value) if value == int(value) else value


@dataclass(frozen=True)
class DimensionDefinition:
    dimension_id: str
    canonical_unit: str
    vector: DimensionVector
    category: str
    description: str

    def __post_init__(self) -> None:
        if not self.dimension_id or not self.canonical_unit:
            raise ValueError("dimension_id and canonical_unit cannot be empty")
        if self.category not in {"physical", "transport", "instrument", "benchmark"}:
            raise ValueError("unsupported dimension category")
        if not self.description:
            raise ValueError("dimension description cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension_id": self.dimension_id,
            "canonical_unit": self.canonical_unit,
            "vector": self.vector.to_dict(),
            "category": self.category,
            "description": self.description,
        }


@dataclass(frozen=True)
class DimensionCheckReport:
    field_id: str
    unit: str
    actual_dimension: str
    expected_dimension: str
    compatible: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "unit": self.unit,
            "actual_dimension": self.actual_dimension,
            "expected_dimension": self.expected_dimension,
            "compatible": self.compatible,
            "message": self.message,
        }


@dataclass(frozen=True)
class DimensionContract:
    field_id: str
    expected_dimension: str
    allowed_units: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.field_id:
            raise ValueError("field_id cannot be empty")
        dimension_definition(self.expected_dimension)
        if not self.allowed_units:
            raise ValueError("allowed_units cannot be empty")
        for unit in self.allowed_units:
            report = check_unit_dimension(
                unit,
                self.expected_dimension,
                field_id=self.field_id,
            )
            if not report.compatible:
                raise ValueError(report.message)

    def check(self, unit: str, *, strict: bool = False) -> DimensionCheckReport:
        report = check_unit_dimension(
            unit,
            self.expected_dimension,
            field_id=self.field_id,
        )
        if report.compatible and unit not in self.allowed_units:
            report = DimensionCheckReport(
                field_id=self.field_id,
                unit=unit,
                actual_dimension=report.actual_dimension,
                expected_dimension=report.expected_dimension,
                compatible=False,
                message=f"{unit!r} is not allowed by field contract {self.field_id!r}",
            )
        if strict and not report.compatible:
            raise ValueError(report.message)
        return report

    def to_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "expected_dimension": self.expected_dimension,
            "allowed_units": list(self.allowed_units),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> DimensionContract:
        allowed_units = payload["allowed_units"]
        if not isinstance(allowed_units, list | tuple):
            raise ValueError("allowed_units must be a list or tuple")
        return cls(
            field_id=str(payload["field_id"]),
            expected_dimension=str(payload["expected_dimension"]),
            allowed_units=tuple(str(unit) for unit in allowed_units),
            description=str(payload.get("description", "")),
        )


M = DimensionVector(mass=1)
L = DimensionVector(length=1)
T = DimensionVector(time=1)
THETA = DimensionVector(temperature=1)
N = DimensionVector(amount=1)
CURRENT = DimensionVector(current=1)
SIGNAL = DimensionVector(signal=1)


def _definition(
    dimension_id: str,
    canonical_unit: str,
    vector: DimensionVector,
    category: str,
    description: str,
) -> DimensionDefinition:
    return DimensionDefinition(
        dimension_id,
        canonical_unit,
        vector,
        category,
        description,
    )


_DIMENSIONS = (
    _definition("dimensionless", "dimensionless", DimensionVector(), "physical", "Pure ratio."),
    _definition("amount", "mol", N, "physical", "Amount of substance."),
    _definition("amount_rate", "mol/s", N / T, "physical", "Amount or formal extent per time."),
    _definition("mass", "kg", M, "physical", "Mass."),
    _definition("volume", "L", L**3, "physical", "Volume."),
    _definition("volume_flow", "m^3/s", L**3 / T, "transport", "Volumetric flow rate."),
    _definition("velocity", "m/s", L / T, "transport", "Linear flow velocity."),
    _definition("viscosity_per_temperature", "Pa*s/K", M / L / T / THETA, "transport",
                "Dynamic-viscosity derivative per kelvin temperature interval."),
    _definition("conductivity_per_temperature", "W/(m*K^2)", M * L / T**3 / THETA**2,
                "transport", "Temperature derivative of thermal conductivity."),
    _definition("temperature", "K", THETA, "physical", "Thermodynamic temperature."),
    _definition("thermal_expansivity", "1/K", THETA**-1, "physical",
                "Fractional volume change per temperature interval."),
    _definition("time", "s", T, "physical", "Elapsed time."),
    _definition("pressure", "Pa", M / L / T**2, "physical", "Mechanical pressure."),
    *(_definition(f"pressure_temperature_derivative_{order}", f"Pa/{suffix}",
                  M/L/T**2/THETA**order, "physical",
                  f"Order-{order} pressure derivative per kelvin temperature interval.")
      for order, suffix in ((1, "K"), (2, "K^2"), (3, "K^3"))),
    _definition("energy", "J", M * L**2 / T**2, "physical", "Energy or heat."),
    _definition("power", "W", M * L**2 / T**3, "physical", "Energy rate."),
    _definition("concentration", "mol/L", N / L**3, "physical", "Amount concentration."),
    _definition("molar_enthalpy", "J/mol", M * L**2 / T**2 / N, "physical", "Energy per amount."),
    _definition("heat_capacity", "J/K", M * L**2 / T**2 / THETA, "physical",
                "Energy per temperature interval."),
    _definition("thermal_conductance", "W/K", M * L**2 / T**3 / THETA, "transport",
                "Heat flow per temperature difference."),
    _definition(
        "molar_heat_capacity",
        "J/(mol*K)",
        M * L**2 / T**2 / N / THETA,
        "physical",
        "Molar heat capacity.",
    ),
    _definition("molecular_weight", "g/mol", M / N, "physical", "Mass per amount."),
    _definition("molar_volume", "m^3/mol", L**3 / N, "physical", "Volume per amount."),
    _definition("molar_area", "m^2/mol", L**2 / N, "physical", "Area per amount."),
    _definition("solubility_parameter", "Pa^0.5", (M / L / T**2)**0.5, "physical",
                "Square root of cohesive energy density."),
    _definition("parachor", "kg^0.25*m^3/(s^0.5*mol)", M**0.25 * L**3 / T**0.5 / N,
                "physical", "Molar volume times the fourth root of surface tension."),
    _definition("molar_volume_per_temperature", "m^3/(mol*K)", L**3 / N / THETA,
                "physical", "Temperature derivative of molar volume."),
    _definition("molar_volume_per_pressure", "m^3/(mol*Pa)", L**4 * T**2 / M / N,
                "physical", "Pressure derivative of molar volume."),
    _definition("mass_density", "kg/m^3", M / L**3, "physical", "Mass per volume."),
    _definition("mass_density_per_temperature", "kg/(m^3*K)", M / L**3 / THETA,
                "physical", "Temperature derivative of mass density on a declared path."),
    _definition("dynamic_viscosity", "Pa*s", M / L / T, "transport", "Dynamic viscosity."),
    _definition(
        "thermal_conductivity",
        "W/(m*K)",
        M * L / T**3 / THETA,
        "transport",
        "Thermal conductivity.",
    ),
    _definition("diffusivity", "m^2/s", L**2 / T, "transport", "Molecular or thermal diffusivity."),
    _definition("surface_tension", "N/m", M / T**2, "transport", "Surface tension."),
    _definition("surface_tension_per_temperature", "N/(m*K)", M / T**2 / THETA,
                "transport", "Temperature derivative of surface tension."),
    _definition(
        "heat_transfer_coefficient",
        "W/(m^2*K)",
        M / T**3 / THETA,
        "transport",
        "Area heat-transfer coefficient.",
    ),
    _definition("electrical_current", "A", CURRENT, "instrument", "Electrical current."),
    _definition(
        "electric_potential",
        "V",
        M * L**2 / T**3 / CURRENT,
        "instrument",
        "Electric potential.",
    ),
    _definition("electric_charge", "C", CURRENT * T, "instrument", "Electric charge."),
    _definition("electric_potential_per_temperature", "V/K",
                M * L**2 / T**3 / CURRENT / THETA, "physical",
                "Potential derivative per kelvin temperature interval."),
    _definition("electric_dipole_moment", "C*m", CURRENT * T * L, "physical",
                "Electric dipole moment."),
    _definition("brix", "degBx", L**0, "instrument",
                "Sucrose reference refractometer scale, not arbitrary-solution mass fraction."),
    _definition(
        "electrical_resistance",
        "Ohm",
        M * L**2 / T**3 / CURRENT**2,
        "instrument",
        "Electrical resistance.",
    ),
    _definition(
        "capacitance",
        "F",
        CURRENT**2 * T**4 / M / L**2,
        "instrument",
        "Electrical capacitance.",
    ),
    _definition("frequency", "Hz", T**-1, "instrument", "Frequency."),
    _definition("length", "nm", L, "instrument", "Spectral wavelength or molecular geometry."),
    _definition("wavenumber", "cm^-1", L**-1, "instrument", "Infrared coordinate."),
    _definition("relative_signal", "relative_signal", DimensionVector(), "instrument",
                "Relative spectral detector signal."),
    _definition("chemical_shift", "ppm", DimensionVector(), "instrument", "NMR chemical shift."),
    _definition("gas_volume_fraction", "gas_volume_fraction", DimensionVector(), "physical",
                "Gas volumetric concentration; distinct from NMR chemical shift."),
    _definition("acidity", "pH", DimensionVector(), "physical",
                "Negative logarithm of hydrogen-ion activity; a distinct semantic scale."),
    _definition(
        "mass_to_charge",
        "m/z",
        DimensionVector(),
        "instrument",
        "Mass-spectrometry mass-to-charge coordinate.",
    ),
    _definition(
        "absorbance", "AU", DimensionVector(), "instrument", "Logarithmic absorbance response."
    ),
    _definition(
        "detector_response",
        "counts",
        SIGNAL,
        "instrument",
        "Arbitrary detector response.",
    ),
    _definition(
        "integrated_detector_response",
        "AU*s",
        SIGNAL * T,
        "instrument",
        "Time-integrated detector response.",
    ),
    _definition("cost", "currency", DimensionVector(currency=1), "benchmark", "Benchmark cost."),
    _definition("risk", "risk", DimensionVector(risk=1), "benchmark", "Benchmark risk score."),
)

_DIMENSION_INDEX = {definition.dimension_id: definition for definition in _DIMENSIONS}


def canonical_dimensions() -> dict[str, DimensionDefinition]:
    """Return the complete canonical dimension catalog."""

    return dict(_DIMENSION_INDEX)


def dimension_definition(dimension_id: str) -> DimensionDefinition:
    try:
        return _DIMENSION_INDEX[dimension_id]
    except KeyError as exc:
        raise ValueError(f"Unsupported dimension: {dimension_id}") from exc


def check_unit_dimension(
    unit: str,
    expected_dimension: str,
    *,
    field_id: str = "value",
    strict: bool = False,
) -> DimensionCheckReport:
    """Check semantic dimension compatibility and optionally hard-fail."""

    actual_dimension = unit_spec(unit).dimension
    dimension_definition(actual_dimension)
    dimension_definition(expected_dimension)
    compatible = actual_dimension == expected_dimension
    message = (
        f"{unit!r} is compatible with {expected_dimension!r}"
        if compatible
        else (
            f"field {field_id!r} expects dimension {expected_dimension!r}, "
            f"but unit {unit!r} has dimension {actual_dimension!r}"
        )
    )
    report = DimensionCheckReport(
        field_id,
        unit,
        actual_dimension,
        expected_dimension,
        compatible,
        message,
    )
    if strict and not compatible:
        raise ValueError(message)
    return report


def unmapped_supported_unit_dimensions() -> tuple[str, ...]:
    """Return unit-table dimensions missing from the canonical catalog."""

    return tuple(
        sorted(
            {
                spec.dimension
                for spec in supported_units().values()
                if spec.dimension not in _DIMENSION_INDEX
            }
        )
    )


def core_dimension_contracts() -> tuple[DimensionContract, ...]:
    """Stable field examples spanning physics, transport, and instruments."""

    return (
        DimensionContract("amount_mol", "amount", ("mol", "mmol")),
        DimensionContract("mass_kg", "mass", ("kg", "g")),
        DimensionContract("volume_L", "volume", ("L", "mL")),
        DimensionContract("temperature_K", "temperature", ("K", "degC")),
        DimensionContract("pressure_Pa", "pressure", ("Pa", "bar", "mmHg")),
        DimensionContract("energy_J", "energy", ("J", "kJ")),
        DimensionContract("viscosity_Pa_s", "dynamic_viscosity", ("Pa*s", "mPa*s")),
        DimensionContract("diffusivity_m2_s", "diffusivity", ("m^2/s", "cm^2/s")),
        DimensionContract("current_A", "electrical_current", ("A", "mA", "uA")),
        DimensionContract("potential_V", "electric_potential", ("V", "mV")),
        DimensionContract("nmr_shift_ppm", "chemical_shift", ("ppm",)),
        DimensionContract("ms_mass_to_charge", "mass_to_charge", ("m/z",)),
        DimensionContract("detector_counts", "detector_response", ("counts",)),
    )


__all__ = [
    "DimensionCheckReport",
    "DimensionContract",
    "DimensionDefinition",
    "DimensionVector",
    "canonical_dimensions",
    "check_unit_dimension",
    "core_dimension_contracts",
    "dimension_definition",
    "unmapped_supported_unit_dimensions",
]
