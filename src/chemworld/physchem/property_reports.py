"""Shared property report contracts for ChemWorld property models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Literal

from chemworld.foundation.units import Quantity, convert_value

R_J_PER_MOL_K = 8.31446261815324
STANDARD_PRESSURE_PA = 101325.0
ValidityPolicy = Literal["warn", "raise", "ignore"]
PhaseLabel = Literal["solid", "liquid", "gas"]
_PHASES = {"solid", "liquid", "gas"}


@dataclass(frozen=True)
class PropertyEvaluation:
    property_id: str
    correlation_id: str
    equation_id: str
    value: float
    unit: str
    inputs: dict[str, float]
    warnings: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.warnings

    def quantity(self) -> Quantity:
        return Quantity(self.value, self.unit)

    def to(self, target_unit: str) -> PropertyEvaluation:
        return PropertyEvaluation(
            property_id=self.property_id,
            correlation_id=self.correlation_id,
            equation_id=self.equation_id,
            value=convert_value(self.value, self.unit, target_unit),
            unit=target_unit,
            inputs=dict(self.inputs),
            warnings=self.warnings,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "property_id": self.property_id,
            "correlation_id": self.correlation_id,
            "equation_id": self.equation_id,
            "value": self.value,
            "unit": self.unit,
            "inputs": dict(self.inputs),
            "warnings": list(self.warnings),
            "valid": self.valid,
        }


@dataclass(frozen=True)
class CaloricReference:
    """An explicit caloric datum, not a formation-data substitute.

    Entropy is optional: without it the package cannot return a chemical
    potential. A sensible energy zero is a declared convention; reference
    offsets for different phases must be supplied on the same basis.
    """

    # None denotes a declared caloric inventory without a phase/EOS closure.
    # Its U/Cv or H/Cp pair is usable, but no PV conversion may be inferred.
    phase: PhaseLabel | None = None
    temperature_K: float = 298.15
    pressure_Pa: float = STANDARD_PRESSURE_PA
    energy_basis: Literal["enthalpy", "internal_energy"] = "enthalpy"
    energy_J_mol: float = 0.0
    entropy_J_mol_K: float | None = None
    source_note: str = "Explicit caloric reference; no formation-data claim"

    def __post_init__(self) -> None:
        if self.phase is not None:
            _validate_phase(self.phase)
        if (self.energy_basis not in {"enthalpy", "internal_energy"}
                or not self.source_note or not isfinite(self.energy_J_mol)
                or any(not isfinite(v) or v <= 0 for v in
                       (self.temperature_K, self.pressure_Pa))
                or (self.entropy_J_mol_K is not None and not isfinite(self.entropy_J_mol_K))):
            raise ValueError("caloric reference needs a phase, finite datum and provenance")

    def to_dict(self):
        return {"phase": self.phase, "temperature_K": self.temperature_K,
                "pressure_Pa": self.pressure_Pa, "energy_basis": self.energy_basis,
                "energy_J_mol": self.energy_J_mol, "entropy_J_mol_K": self.entropy_J_mol_K,
                "source_note": self.source_note}


@dataclass(frozen=True)
class StatePropertyReport:
    """Requested scalar/partial properties at one candidate state, with units.

    This is a transient evaluation of caller-owned inventory. It does not own
    evolving material, choose a phase split, or imply unrequested properties.
    """

    model_id: str
    phase: PhaseLabel
    temperature_K: float
    pressure_Pa: float
    component_amounts_mol: Mapping[str, float]
    properties: Mapping[str, PropertyEvaluation]
    reference_states: Mapping[str, CaloricReference] = field(default_factory=dict)
    component_properties: Mapping[str, Mapping[str, PropertyEvaluation]] = field(
        default_factory=dict)
    model_details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_phase(self.phase)
        amounts = {key: float(value) for key, value in self.component_amounts_mol.items()}
        if (not self.model_id or not amounts or not isfinite(sum(amounts.values()))
                or sum(amounts.values()) <= 0
                or any(not key or not isfinite(value) or value < 0
                       for key, value in amounts.items())
                or any(not isfinite(v) or v <= 0 for v in
                       (self.temperature_K, self.pressure_Pa))):
            raise ValueError("property state requires finite T/P and a nonempty inventory")
        evaluations = [*self.properties.values(),
                       *(value for group in self.component_properties.values()
                         for value in group.values())]
        if any(not isfinite(value.value) for value in evaluations):
            raise ValueError("state property evaluations must be finite")
        object.__setattr__(self, "component_amounts_mol", MappingProxyType(amounts))
        object.__setattr__(self, "properties", MappingProxyType(dict(self.properties)))
        object.__setattr__(self, "reference_states", MappingProxyType(dict(self.reference_states)))
        object.__setattr__(self, "component_properties", MappingProxyType({
            key: MappingProxyType(dict(values)) for key, values in self.component_properties.items()
        }))
        object.__setattr__(self, "model_details", MappingProxyType(dict(self.model_details)))

    @property
    def valid(self):
        return all(value.valid for value in self.properties.values()) and all(
            value.valid for values in self.component_properties.values()
            for value in values.values())

    def to_dict(self):
        return {"model_id": self.model_id, "phase": self.phase,
                "temperature_K": self.temperature_K, "pressure_Pa": self.pressure_Pa,
                "component_amounts_mol": dict(self.component_amounts_mol),
                "properties": {key: value.to_dict() for key, value in self.properties.items()},
                "component_properties": {
                    key: {component: value.to_dict() for component, value in values.items()}
                    for key, values in self.component_properties.items()},
                "reference_states": {key: value.to_dict()
                                     for key, value in self.reference_states.items()},
                "model_details": dict(self.model_details),
                "valid": self.valid}


def _validate_phase(phase: str) -> None:
    if phase not in _PHASES:
        raise ValueError(f"Unsupported phase {phase!r}; expected one of {sorted(_PHASES)}")


def _pressure_unit_to_pa_factor(unit: str) -> float:
    return convert_value(1.0, unit, "Pa")


def _validated_fraction_mapping(
    fractions: Mapping[str, float],
    fraction_name: str,
) -> dict[str, float]:
    if not fractions:
        raise ValueError(f"component_{fraction_name}_fractions cannot be empty")
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"component {fraction_name} fractions must sum to 1")
    normalized: dict[str, float] = {}
    for component_id, fraction in fractions.items():
        if not component_id:
            raise ValueError("component id cannot be empty")
        if fraction < 0 or not isfinite(fraction):
            raise ValueError(f"invalid {fraction_name} fraction for {component_id!r}")
        normalized[component_id] = float(fraction)
    return normalized


def _require_same_keys(
    reference: Mapping[str, float],
    values: Mapping[str, float],
    value_name: str,
) -> None:
    missing = sorted(set(reference) - set(values))
    extra = sorted(set(values) - set(reference))
    if missing or extra:
        raise ValueError(
            f"{value_name} keys must match components; missing={missing}, extra={extra}"
        )


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)

