"""Canonical dimension contracts for physical and instrument quantities."""

from __future__ import annotations

from dataclasses import dataclass

from chemworld.foundation.units import supported_units, unit_spec


@dataclass(frozen=True)
class DimensionVector:
    """Integer exponents over ChemWorld's physical and semantic base axes."""

    mass: int = 0
    length: int = 0
    time: int = 0
    temperature: int = 0
    amount: int = 0
    current: int = 0
    signal: int = 0
    currency: int = 0
    risk: int = 0

    def __mul__(self, other: DimensionVector) -> DimensionVector:
        return self._combine(other, 1)

    def __truediv__(self, other: DimensionVector) -> DimensionVector:
        return self._combine(other, -1)

    def __pow__(self, exponent: int) -> DimensionVector:
        if not isinstance(exponent, int):
            raise TypeError("dimension exponent must be an integer")
        return DimensionVector(**{name: value * exponent for name, value in self.to_dict().items()})

    def _combine(self, other: DimensionVector, sign: int) -> DimensionVector:
        if not isinstance(other, DimensionVector):
            return NotImplemented
        left = self.to_dict()
        right = other.to_dict()
        return DimensionVector(**{name: left[name] + sign * right[name] for name in left})

    def to_dict(self) -> dict[str, int]:
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
        return cls(**{name: int(str(payload.get(name, 0))) for name in cls().to_dict()})


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
    _definition("mass", "kg", M, "physical", "Mass."),
    _definition("volume", "L", L**3, "physical", "Volume."),
    _definition("temperature", "K", THETA, "physical", "Thermodynamic temperature."),
    _definition("time", "s", T, "physical", "Elapsed time."),
    _definition("pressure", "Pa", M / L / T**2, "physical", "Mechanical pressure."),
    _definition("energy", "J", M * L**2 / T**2, "physical", "Energy or heat."),
    _definition("power", "W", M * L**2 / T**3, "physical", "Energy rate."),
    _definition("concentration", "mol/L", N / L**3, "physical", "Amount concentration."),
    _definition("molar_enthalpy", "J/mol", M * L**2 / T**2 / N, "physical", "Energy per amount."),
    _definition(
        "molar_heat_capacity",
        "J/(mol*K)",
        M * L**2 / T**2 / N / THETA,
        "physical",
        "Molar heat capacity.",
    ),
    _definition("molecular_weight", "g/mol", M / N, "physical", "Mass per amount."),
    _definition("molar_volume", "m^3/mol", L**3 / N, "physical", "Volume per amount."),
    _definition("molar_volume_per_temperature", "m^3/(mol*K)", L**3 / N / THETA,
                "physical", "Temperature derivative of molar volume."),
    _definition("molar_volume_per_pressure", "m^3/(mol*Pa)", L**4 * T**2 / M / N,
                "physical", "Pressure derivative of molar volume."),
    _definition("mass_density", "kg/m^3", M / L**3, "physical", "Mass per volume."),
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
    _definition("length", "nm", L, "instrument", "Spectral wavelength."),
    _definition("wavenumber", "cm^-1", L**-1, "instrument", "Infrared coordinate."),
    _definition("relative_signal", "relative_signal", DimensionVector(), "instrument",
                "Relative spectral detector signal."),
    _definition("chemical_shift", "ppm", DimensionVector(), "instrument", "NMR chemical shift."),
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
