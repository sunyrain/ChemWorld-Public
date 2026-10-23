"""JSON-friendly component, mixture, and property-correlation specs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, ClassVar

from chemworld.foundation.units import unit_spec
from chemworld.physchem.correlation_families import (
    CORRELATION_FAMILIES,
    validate_family_parameters,
)
from chemworld.physchem.elements import (
    explicit_formula_charge,
    hill_formula,
    mass_fractions_from_formula,
    molecular_weight,
    parse_formula,
)
from chemworld.physchem.identifiers import validate_cas_number

_VALID_PHASES = {"gas", "liquid", "solid", "aqueous", "organic", "unknown"}
_FRACTION_TOL = 1e-9
_MOLECULAR_WEIGHT_REL_TOL = 0.02

_PHASE_COMPATIBILITY: dict[str, set[str]] = {
    "gas": {"gas", "unknown"},
    "liquid": {"liquid", "aqueous", "organic", "unknown"},
    "aqueous": {"liquid", "aqueous", "unknown"},
    "organic": {"liquid", "organic", "unknown"},
    "solid": {"solid", "unknown"},
    "unknown": _VALID_PHASES,
}

_EQUATION_CONTRACTS: dict[str, dict[str, object]] = {
    "antoine": {
        "required_coefficients": {"A", "B", "C"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "pressure",
    },
    "wagner": {
        "required_coefficients": {"Tc", "Pc", "a", "b", "c", "d"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "pressure",
    },
    "dippr101_vapor_pressure": {
        "required_coefficients": {"A", "B", "C", "D", "E"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "pressure",
    },
    "cp_polynomial": {
        "required_coefficients": {"a"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "molar_heat_capacity",
    },
    "watson_hvap": {
        "required_coefficients": {"Tc", "T_ref", "Hvap_ref"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "molar_enthalpy",
    },
    "constant_phase_change_enthalpy": {
        "required_coefficients": {"delta_h"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "molar_enthalpy",
    },
    "linear_liquid_density": {
        "required_coefficients": {"rho_ref", "T_ref"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "mass_density",
    },
    "ideal_gas_density": {
        "required_coefficients": {"unused"},
        "input_dimensions": {"temperature": "temperature", "pressure": "pressure"},
        "output_dimension": "mass_density",
    },
    "rackett_liquid_molar_volume": {
        "required_coefficients": {"Tc", "Pc", "Zc"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "molar_volume",
    },
    "crc_second_virial": {
        "required_coefficients": {"a1"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "molar_volume",
    },
    "andrade_viscosity": {
        "required_coefficients": {"A", "B"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "dynamic_viscosity",
    },
    "sutherland_gas_viscosity": {
        "required_coefficients": {"mu_ref", "T_ref", "S"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "dynamic_viscosity",
    },
    "linear_thermal_conductivity": {
        "required_coefficients": {"k_ref", "T_ref"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "thermal_conductivity",
    },
    "thermal_conductivity_polynomial": {
        "required_coefficients": {"a"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "thermal_conductivity",
    },
    "surface_tension_power": {
        "required_coefficients": {"Tc", "T_ref", "sigma_ref"},
        "input_dimensions": {"temperature": "temperature"},
        "output_dimension": "surface_tension",
    },
}
_EQUATION_CONTRACTS.update({
    key: {"required_coefficients": set(family.required),
          "input_dimensions": family.input_dimensions,
          "output_dimension": "property_declared"}
    for key, family in CORRELATION_FAMILIES.items()
})


@dataclass(frozen=True)
class ComponentProvenance:
    """Structured source metadata for a curated component record."""

    source_id: str
    source_name: str
    source_table: str = ""
    source_key: str = ""
    source_path: str = ""
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("source_id", "source_name"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} cannot be empty")
        _validate_string_tuple(self.notes, "notes")
        object.__setattr__(self, "notes", tuple(self.notes))

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_table": self.source_table,
            "source_key": self.source_key,
            "source_path": self.source_path,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentProvenance:
        return cls(
            source_id=str(payload["source_id"]),
            source_name=str(payload["source_name"]),
            source_table=str(payload.get("source_table", "")),
            source_key=str(payload.get("source_key", "")),
            source_path=str(payload.get("source_path", "")),
            notes=tuple(str(value) for value in payload.get("notes", ())),
        )


@dataclass(frozen=True)
class ComponentUncertainty:
    """Uncertainty metadata for a component-level field or curated constant."""

    field_id: str
    unit: str = ""
    standard_uncertainty: float | None = None
    relative_uncertainty: float | None = None
    coverage: str = ""
    source_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if not self.field_id.strip():
            raise ValueError("field_id cannot be empty")
        if self.standard_uncertainty is not None and self.standard_uncertainty < 0:
            raise ValueError("standard_uncertainty must be nonnegative")
        if self.relative_uncertainty is not None and self.relative_uncertainty < 0:
            raise ValueError("relative_uncertainty must be nonnegative")

    def to_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "unit": self.unit,
            "standard_uncertainty": self.standard_uncertainty,
            "relative_uncertainty": self.relative_uncertainty,
            "coverage": self.coverage,
            "source_id": self.source_id,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentUncertainty:
        standard = payload.get("standard_uncertainty")
        relative = payload.get("relative_uncertainty")
        return cls(
            field_id=str(payload["field_id"]),
            unit=str(payload.get("unit", "")),
            standard_uncertainty=None if standard is None else float(standard),
            relative_uncertainty=None if relative is None else float(relative),
            coverage=str(payload.get("coverage", "")),
            source_id=str(payload.get("source_id", "")),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class ComponentFieldCandidate:
    """One source candidate for a component-level field."""

    field_id: str
    value: object
    source_id: str
    source_priority: int = 100
    uncertainty: ComponentUncertainty | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.field_id.strip():
            raise ValueError("field_id cannot be empty")
        if not self.source_id.strip():
            raise ValueError("source_id cannot be empty")
        if not isinstance(self.source_priority, int):
            raise ValueError("source_priority must be an integer")
        if self.uncertainty is not None:
            object.__setattr__(
                self,
                "uncertainty",
                _coerce_component_uncertainty(self.uncertainty),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "value": self.value,
            "source_id": self.source_id,
            "source_priority": self.source_priority,
            "uncertainty": None if self.uncertainty is None else self.uncertainty.to_dict(),
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentFieldCandidate:
        uncertainty = payload.get("uncertainty")
        return cls(
            field_id=str(payload["field_id"]),
            value=payload["value"],
            source_id=str(payload["source_id"]),
            source_priority=int(payload.get("source_priority", 100)),
            uncertainty=None
            if uncertainty is None
            else ComponentUncertainty.from_dict(dict(uncertainty)),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class ComponentConflictPolicy:
    """Deterministic policy for resolving component field disagreements."""

    mode: str = "raise"
    source_priority: tuple[str, ...] = ()
    default_rtol: float = 0.0
    default_atol: float = 0.0
    field_rtol: dict[str, float] = field(default_factory=dict)
    field_atol: dict[str, float] = field(default_factory=dict)
    required_uncertainty_fields: tuple[str, ...] = ()
    missing_uncertainty_mode: str = "warn"

    def __post_init__(self) -> None:
        if self.mode not in {"raise", "warn", "prefer_priority"}:
            raise ValueError("mode must be raise, warn, or prefer_priority")
        if self.missing_uncertainty_mode not in {"ignore", "warn", "raise"}:
            raise ValueError("missing_uncertainty_mode must be ignore, warn, or raise")
        _validate_string_tuple(self.source_priority, "source_priority")
        _validate_string_tuple(
            self.required_uncertainty_fields,
            "required_uncertainty_fields",
        )
        if self.default_rtol < 0 or self.default_atol < 0:
            raise ValueError("default tolerances must be nonnegative")
        for mapping_name, mapping in (
            ("field_rtol", self.field_rtol),
            ("field_atol", self.field_atol),
        ):
            for key, value in mapping.items():
                if not str(key).strip():
                    raise ValueError(f"{mapping_name} keys cannot be empty")
                if float(value) < 0:
                    raise ValueError(f"{mapping_name} values must be nonnegative")
        object.__setattr__(self, "source_priority", tuple(self.source_priority))
        object.__setattr__(
            self,
            "required_uncertainty_fields",
            tuple(self.required_uncertainty_fields),
        )
        object.__setattr__(
            self,
            "field_rtol",
            {str(key): float(value) for key, value in self.field_rtol.items()},
        )
        object.__setattr__(
            self,
            "field_atol",
            {str(key): float(value) for key, value in self.field_atol.items()},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "source_priority": list(self.source_priority),
            "default_rtol": self.default_rtol,
            "default_atol": self.default_atol,
            "field_rtol": dict(self.field_rtol),
            "field_atol": dict(self.field_atol),
            "required_uncertainty_fields": list(self.required_uncertainty_fields),
            "missing_uncertainty_mode": self.missing_uncertainty_mode,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentConflictPolicy:
        return cls(
            mode=str(payload.get("mode", "raise")),
            source_priority=tuple(str(value) for value in payload.get("source_priority", ())),
            default_rtol=float(payload.get("default_rtol", 0.0)),
            default_atol=float(payload.get("default_atol", 0.0)),
            field_rtol={
                str(key): float(value) for key, value in dict(payload.get("field_rtol", {})).items()
            },
            field_atol={
                str(key): float(value) for key, value in dict(payload.get("field_atol", {})).items()
            },
            required_uncertainty_fields=tuple(
                str(value) for value in payload.get("required_uncertainty_fields", ())
            ),
            missing_uncertainty_mode=str(payload.get("missing_uncertainty_mode", "warn")),
        )


@dataclass(frozen=True)
class ComponentConflictResolution:
    """JSON-friendly audit record for one resolved component field."""

    field_id: str
    resolved_value: object
    resolved_source_id: str
    status: str
    candidates: tuple[ComponentFieldCandidate, ...]
    message: str = ""

    def __post_init__(self) -> None:
        if self.status not in {"consistent", "conflict_warning", "preferred"}:
            raise ValueError("unknown conflict resolution status")
        object.__setattr__(
            self,
            "candidates",
            tuple(_coerce_component_field_candidate(item) for item in self.candidates),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "resolved_value": self.resolved_value,
            "resolved_source_id": self.resolved_source_id,
            "status": self.status,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentConflictResolution:
        return cls(
            field_id=str(payload["field_id"]),
            resolved_value=payload["resolved_value"],
            resolved_source_id=str(payload["resolved_source_id"]),
            status=str(payload["status"]),
            candidates=tuple(
                ComponentFieldCandidate.from_dict(dict(item))
                for item in payload.get("candidates", ())
            ),
            message=str(payload.get("message", "")),
        )


@dataclass(frozen=True)
class ComponentSpec:
    """A compact component identity and composition record.

    The spec intentionally stores only stable identity, composition, phase, and
    policy metadata. Property models live in `PropertyCorrelation` records.
    """

    identifier: str
    formula: str
    molecular_weight_g_mol: float | None = None
    charge: int | None = None
    default_phase: str = "liquid"
    safety_tags: tuple[str, ...] = ()
    allowed_property_correlations: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    provenance: tuple[ComponentProvenance, ...] = ()
    uncertainty: tuple[ComponentUncertainty, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)
    cas_number: str = ""
    inchi: str = ""
    inchi_key: str = ""
    molecular_weight_rel_tolerance: ClassVar[float] = _MOLECULAR_WEIGHT_REL_TOL

    def __post_init__(self) -> None:
        if not self.identifier:
            raise ValueError("Component identifier cannot be empty")
        if self.identifier.strip() != self.identifier or any(
            char.isspace() for char in self.identifier
        ):
            raise ValueError(
                "Component identifier cannot contain leading, trailing, or internal whitespace"
            )
        if self.cas_number:
            object.__setattr__(self, "cas_number", validate_cas_number(self.cas_number))
        if self.inchi and not self.inchi.startswith(("InChI=1/", "InChI=1S/")):
            raise ValueError("inchi must use an InChI=1/ or InChI=1S/ prefix")
        if self.inchi_key and re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", self.inchi_key) is None:
            raise ValueError("inchi_key must use the standard 14-10-1 uppercase shape")
        if self.default_phase not in _VALID_PHASES:
            raise ValueError(f"Unsupported default phase: {self.default_phase}")
        composition = parse_formula(self.formula)
        calculated_mw = molecular_weight(composition)
        if self.molecular_weight_g_mol is None:
            object.__setattr__(self, "molecular_weight_g_mol", calculated_mw)
        elif self.molecular_weight_g_mol <= 0:
            raise ValueError("molecular_weight_g_mol must be positive")
        elif (
            abs(self.molecular_weight_g_mol - calculated_mw) / calculated_mw
            > self.molecular_weight_rel_tolerance
        ):
            raise ValueError(
                "molecular_weight_g_mol is inconsistent with formula-derived "
                f"value {calculated_mw:g} g/mol"
            )
        declared_charge = explicit_formula_charge(self.formula)
        if self.charge is None:
            object.__setattr__(self, "charge", 0 if declared_charge is None else declared_charge)
        if isinstance(self.charge, bool) or not isinstance(self.charge, int):
            raise ValueError("charge must be an integer")
        if declared_charge is not None and self.charge != declared_charge:
            raise ValueError("explicit component charge conflicts with its formula charge")
        _validate_string_tuple(self.safety_tags, "safety_tags")
        _validate_string_tuple(
            self.allowed_property_correlations,
            "allowed_property_correlations",
        )
        _validate_string_tuple(self.aliases, "aliases")
        object.__setattr__(self, "safety_tags", tuple(self.safety_tags))
        object.__setattr__(
            self,
            "allowed_property_correlations",
            tuple(self.allowed_property_correlations),
        )
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(
            self,
            "provenance",
            tuple(_coerce_component_provenance(item) for item in self.provenance),
        )
        object.__setattr__(
            self,
            "uncertainty",
            tuple(_coerce_component_uncertainty(item) for item in self.uncertainty),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def composition(self) -> dict[str, float]:
        return parse_formula(self.formula)

    @property
    def hill_formula(self) -> str:
        return hill_formula(self.composition)

    def element_mass_fractions(self) -> dict[str, float]:
        return mass_fractions_from_formula(self.composition)

    def to_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "formula": self.formula,
            "cas_number": self.cas_number,
            "inchi": self.inchi,
            "inchi_key": self.inchi_key,
            "hill_formula": self.hill_formula,
            "composition": self.composition,
            "molecular_weight_g_mol": self.molecular_weight_g_mol,
            "charge": self.charge,
            "default_phase": self.default_phase,
            "safety_tags": list(self.safety_tags),
            "allowed_property_correlations": list(self.allowed_property_correlations),
            "aliases": list(self.aliases),
            "provenance": [item.to_dict() for item in self.provenance],
            "uncertainty": [item.to_dict() for item in self.uncertainty],
            "metadata": dict(self.metadata),
            "units": {"molecular_weight_g_mol": "g/mol"},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ComponentSpec:
        return cls(
            identifier=str(payload["identifier"]),
            formula=str(payload["formula"]),
            cas_number=str(payload.get("cas_number", payload.get("casrn", ""))),
            inchi=str(payload.get("inchi", "")),
            inchi_key=str(payload.get("inchi_key", "")),
            molecular_weight_g_mol=(
                None
                if payload.get("molecular_weight_g_mol") is None
                else float(payload["molecular_weight_g_mol"])
            ),
            charge=payload.get("charge"),
            default_phase=str(payload.get("default_phase", "liquid")),
            safety_tags=tuple(str(value) for value in payload.get("safety_tags", ())),
            allowed_property_correlations=tuple(
                str(value) for value in payload.get("allowed_property_correlations", ())
            ),
            aliases=tuple(str(value) for value in payload.get("aliases", ())),
            provenance=tuple(
                ComponentProvenance.from_dict(dict(value))
                for value in payload.get("provenance", ())
            ),
            uncertainty=tuple(
                ComponentUncertainty.from_dict(dict(value))
                for value in payload.get("uncertainty", ())
            ),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(frozen=True)
class MixtureSpec:
    """A phase-local mixture state with both mole and mass fractions."""

    component_ids: tuple[str, ...]
    molecular_weights_g_mol: tuple[float, ...]
    mole_fractions: tuple[float, ...]
    mass_fractions: tuple[float, ...]
    phase_label: str
    temperature_K: float
    pressure_Pa: float

    def __post_init__(self) -> None:
        if not self.component_ids:
            raise ValueError("Mixture must contain at least one component")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("Mixture component_ids must be unique")
        _validate_string_tuple(self.component_ids, "component_ids")
        if self.phase_label not in _VALID_PHASES:
            raise ValueError(f"Unsupported phase label: {self.phase_label}")
        n = len(self.component_ids)
        if (
            len(self.molecular_weights_g_mol) != n
            or len(self.mole_fractions) != n
            or len(self.mass_fractions) != n
        ):
            raise ValueError("Fraction vectors must match component_ids length")
        if any(value <= 0 for value in self.molecular_weights_g_mol):
            raise ValueError("molecular_weights_g_mol must be positive")
        _validate_fraction_vector(self.mole_fractions, "mole_fractions")
        _validate_fraction_vector(self.mass_fractions, "mass_fractions")
        if self.temperature_K <= 0:
            raise ValueError("temperature_K must be positive")
        if self.pressure_Pa <= 0:
            raise ValueError("pressure_Pa must be positive")

    @classmethod
    def from_mole_fractions(
        cls,
        components: Sequence[ComponentSpec],
        mole_fractions: Sequence[float],
        *,
        phase_label: str,
        temperature_K: float,
        pressure_Pa: float,
    ) -> MixtureSpec:
        _validate_component_phase_compatibility(components, phase_label)
        zs = _normalize_fractions(tuple(float(value) for value in mole_fractions))
        ws = mass_fractions_from_mole_fractions(components, zs)
        return cls(
            component_ids=tuple(component.identifier for component in components),
            molecular_weights_g_mol=tuple(_mw(component) for component in components),
            mole_fractions=zs,
            mass_fractions=ws,
            phase_label=phase_label,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )

    @classmethod
    def from_mass_fractions(
        cls,
        components: Sequence[ComponentSpec],
        mass_fractions: Sequence[float],
        *,
        phase_label: str,
        temperature_K: float,
        pressure_Pa: float,
    ) -> MixtureSpec:
        _validate_component_phase_compatibility(components, phase_label)
        ws = _normalize_fractions(tuple(float(value) for value in mass_fractions))
        zs = mole_fractions_from_mass_fractions(components, ws)
        return cls(
            component_ids=tuple(component.identifier for component in components),
            molecular_weights_g_mol=tuple(_mw(component) for component in components),
            mole_fractions=zs,
            mass_fractions=ws,
            phase_label=phase_label,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )

    @property
    def average_molecular_weight_g_mol(self) -> float:
        return sum(
            z * mw
            for z, mw in zip(
                self.mole_fractions,
                self.molecular_weights_g_mol,
                strict=True,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "component_ids": list(self.component_ids),
            "molecular_weights_g_mol": list(self.molecular_weights_g_mol),
            "mole_fractions": list(self.mole_fractions),
            "mass_fractions": list(self.mass_fractions),
            "phase_label": self.phase_label,
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "average_molecular_weight_g_mol": self.average_molecular_weight_g_mol,
            "units": {
                "temperature_K": "K",
                "pressure_Pa": "Pa",
                "average_molecular_weight_g_mol": "g/mol",
            },
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MixtureSpec:
        return cls(
            component_ids=tuple(str(value) for value in payload["component_ids"]),
            molecular_weights_g_mol=tuple(
                float(value) for value in payload["molecular_weights_g_mol"]
            ),
            mole_fractions=tuple(float(value) for value in payload["mole_fractions"]),
            mass_fractions=tuple(float(value) for value in payload["mass_fractions"]),
            phase_label=str(payload["phase_label"]),
            temperature_K=float(payload["temperature_K"]),
            pressure_Pa=float(payload["pressure_Pa"]),
        )


@dataclass(frozen=True)
class PropertyCorrelation:
    """A portable property-correlation declaration with explicit units."""

    correlation_id: str
    equation_id: str
    coefficients: dict[str, float]
    input_units: dict[str, str]
    output_unit: str
    property_id: str = ""
    validity_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    source_note: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.correlation_id:
            raise ValueError("correlation_id cannot be empty")
        if self.correlation_id.strip() != self.correlation_id:
            raise ValueError("correlation_id cannot contain leading or trailing whitespace")
        if self.property_id and not self.property_id.replace("_", "").isalnum():
            raise ValueError(f"Invalid property_id: {self.property_id}")
        if not self.equation_id:
            raise ValueError("equation_id cannot be empty")
        if self.equation_id not in _EQUATION_CONTRACTS:
            raise ValueError(f"Unsupported property equation_id: {self.equation_id}")
        if not self.coefficients and _required_coefficients(self.equation_id):
            raise ValueError("coefficients cannot be empty")
        _validate_coefficients(self.coefficients, self.equation_id)
        contract = _EQUATION_CONTRACTS[self.equation_id]
        output_spec = unit_spec(self.output_unit)
        expected_output_dimension = str(contract["output_dimension"])
        if self.equation_id in CORRELATION_FAMILIES:
            family = CORRELATION_FAMILIES[self.equation_id]
            validate_family_parameters(self.equation_id, self.coefficients)
            if self.property_id not in family.property_dimensions:
                raise ValueError(f"{self.equation_id} requires a supported explicit property_id")
            expected_output_dimension = family.property_dimensions[self.property_id]
            if family.fixed_output_unit and self.output_unit != family.fixed_output_unit:
                raise ValueError(
                    f"{self.equation_id} coefficients require {family.fixed_output_unit}")
            if self.input_units != family.input_units:
                raise ValueError("correlation families require absolute-Kelvin input units "
                                 "and Pa for declared pressure inputs")
        if output_spec.dimension != expected_output_dimension:
            raise ValueError(
                f"output_unit for {self.equation_id} must have dimension "
                f"{expected_output_dimension}, got {output_spec.dimension}"
            )
        input_dimensions = contract["input_dimensions"]
        if not isinstance(input_dimensions, dict):
            raise TypeError("Internal equation contract input_dimensions must be a mapping")
        for field_name, unit in self.input_units.items():
            if not field_name:
                raise ValueError("input unit field names cannot be empty")
            spec = unit_spec(unit)
            expected_dimension = input_dimensions.get(field_name)
            if expected_dimension is None:
                raise ValueError(
                    f"Unexpected input field {field_name!r} for equation {self.equation_id}"
                )
            if spec.dimension != expected_dimension:
                raise ValueError(
                    f"input_units[{field_name!r}] must have dimension "
                    f"{expected_dimension}, got {spec.dimension}"
                )
        for required_field in input_dimensions:
            if required_field not in self.input_units:
                raise ValueError(
                    f"Missing input unit for field {required_field!r} on {self.equation_id}"
                )
        for variable, bounds in self.validity_ranges.items():
            if variable not in input_dimensions:
                raise ValueError(
                    f"Validity range field {variable!r} is not an input of {self.equation_id}"
                )
            if len(bounds) != 2:
                raise ValueError(f"Validity range must have two values: {variable}")
            lower, upper = bounds
            is_source_point = (self.equation_id in {
                "phase_change_point", "interface_source_point",
                "dimensionless_source_point", "molar_volume_source_point"}
                               and variable == "temperature"
                               and lower == upper == self.coefficients["T_ref"])
            if (not all(isfinite(v) for v in (lower, upper))
                    or (lower >= upper and not is_source_point)):
                raise ValueError(f"Invalid validity range for {variable}: {bounds}")
        object.__setattr__(self, "coefficients", dict(self.coefficients))
        object.__setattr__(self, "input_units", dict(self.input_units))
        object.__setattr__(self, "validity_ranges", dict(self.validity_ranges))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "correlation_id": self.correlation_id,
            "property_id": self.property_id,
            "equation_id": self.equation_id,
            "coefficients": dict(self.coefficients),
            "input_units": dict(self.input_units),
            "output_unit": self.output_unit,
            "validity_ranges": {key: list(value) for key, value in self.validity_ranges.items()},
            "source_note": self.source_note,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PropertyCorrelation:
        return cls(
            correlation_id=str(payload["correlation_id"]),
            equation_id=str(payload["equation_id"]),
            coefficients={
                str(key): float(value) for key, value in dict(payload["coefficients"]).items()
            },
            input_units={
                str(key): str(value) for key, value in dict(payload["input_units"]).items()
            },
            output_unit=str(payload["output_unit"]),
            property_id=str(payload.get("property_id", "")),
            validity_ranges={
                str(key): (float(value[0]), float(value[1]))
                for key, value in dict(payload.get("validity_ranges", {})).items()
            },
            source_note=str(payload.get("source_note", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def model_card(self) -> dict[str, object]:
        output = unit_spec(self.output_unit)
        required = _required_coefficients(self.equation_id)
        return {
            "correlation_id": self.correlation_id,
            "property_id": self.property_id,
            "equation_id": self.equation_id,
            "required_coefficients": sorted(required),
            "input_units": dict(self.input_units),
            "output_unit": self.output_unit,
            "output_dimension": output.dimension,
            "validity_ranges": {key: list(value) for key, value in self.validity_ranges.items()},
            "source_note": self.source_note,
            "metadata": dict(self.metadata),
        }


def mass_fractions_from_mole_fractions(
    components: Sequence[ComponentSpec],
    mole_fractions: Sequence[float],
) -> tuple[float, ...]:
    _validate_component_fraction_lengths(components, mole_fractions)
    from chemworld.physchem.composition_math import fraction_basis

    return fraction_basis(mole_fractions, tuple(_mw(component) for component in components))


def mole_fractions_from_mass_fractions(
    components: Sequence[ComponentSpec],
    mass_fractions: Sequence[float],
) -> tuple[float, ...]:
    _validate_component_fraction_lengths(components, mass_fractions)
    from chemworld.physchem.composition_math import fraction_basis

    return fraction_basis(mass_fractions, tuple(_mw(component) for component in components),
                          inverse=True)


def normalize_component_token(value: str) -> str:
    """Normalize an identifier or alias for registry lookup."""

    token = "_".join(value.strip().lower().split())
    if not token:
        raise ValueError("component identifier or alias cannot be empty")
    return token


def component_alias_index(components: Sequence[ComponentSpec]) -> dict[str, str]:
    """Build a normalized alias-to-identifier map and reject conflicts."""

    index: dict[str, str] = {}
    for component in components:
        tokens = _component_identity_tokens(component)
        for token_value in tokens:
            token = normalize_component_token(token_value)
            existing = index.get(token)
            if existing is not None and existing != component.identifier:
                raise ValueError(
                    "component identifier/alias conflict: "
                    f"{token_value!r} maps to both {existing!r} and "
                    f"{component.identifier!r}"
                )
            index[token] = component.identifier
    return index


def resolve_component_identifier(
    components: Sequence[ComponentSpec],
    identifier_or_alias: str,
) -> str:
    """Resolve an identifier or alias against a component registry."""

    token = normalize_component_token(identifier_or_alias)
    index = component_alias_index(components)
    try:
        return index[token]
    except KeyError as exc:
        allowed = ", ".join(sorted({component.identifier for component in components}))
        raise KeyError(
            f"unknown component identifier or alias {identifier_or_alias!r}; allowed={allowed}"
        ) from exc


def resolve_component_field_conflict(
    field_id: str,
    candidates: Sequence[ComponentFieldCandidate | dict[str, object]],
    policy: ComponentConflictPolicy | dict[str, object] | None = None,
) -> ComponentConflictResolution:
    """Resolve one component field with an auditable source-priority policy."""

    if not field_id.strip():
        raise ValueError("field_id cannot be empty")
    normalized_policy = _coerce_component_conflict_policy(
        ComponentConflictPolicy() if policy is None else policy
    )
    normalized_candidates = tuple(
        _coerce_component_field_candidate(candidate) for candidate in candidates
    )
    if not normalized_candidates:
        raise ValueError("at least one field candidate is required")
    if any(candidate.field_id != field_id for candidate in normalized_candidates):
        raise ValueError("all candidates must match field_id")

    ordered = tuple(
        sorted(
            normalized_candidates,
            key=lambda candidate: (
                _source_rank(candidate.source_id, normalized_policy),
                candidate.source_priority,
                candidate.source_id,
            ),
        )
    )
    selected = ordered[0]
    conflicts = [
        candidate
        for candidate in ordered[1:]
        if not _values_within_tolerance(
            selected.value,
            candidate.value,
            field_id=field_id,
            policy=normalized_policy,
        )
    ]
    if not conflicts:
        return ComponentConflictResolution(
            field_id=field_id,
            resolved_value=selected.value,
            resolved_source_id=selected.source_id,
            status="consistent",
            candidates=ordered,
            message="all candidate values agree within policy tolerance",
        )

    conflict_sources = ", ".join(candidate.source_id for candidate in conflicts)
    message = (
        f"{field_id} candidates conflict; selected {selected.source_id!r} by "
        f"source priority over {conflict_sources}"
    )
    if normalized_policy.mode == "raise":
        raise ValueError(message)
    return ComponentConflictResolution(
        field_id=field_id,
        resolved_value=selected.value,
        resolved_source_id=selected.source_id,
        status=("conflict_warning" if normalized_policy.mode == "warn" else "preferred"),
        candidates=ordered,
        message=message,
    )


def _mw(component: ComponentSpec) -> float:
    if component.molecular_weight_g_mol is None:
        raise ValueError(f"component {component.identifier!r} has no molecular weight")
    return component.molecular_weight_g_mol


def _validate_component_fraction_lengths(
    components: Sequence[ComponentSpec],
    fractions: Sequence[float],
) -> None:
    if not components:
        raise ValueError("At least one component is required")
    if len(components) != len(fractions):
        raise ValueError("components and fractions must have the same length")


def _validate_fraction_vector(fractions: tuple[float, ...], name: str) -> None:
    if any(value < 0 for value in fractions):
        raise ValueError(f"{name} cannot contain negative values")
    if abs(sum(fractions) - 1.0) > _FRACTION_TOL:
        raise ValueError(f"{name} must sum to 1")


def _normalize_fractions(fractions: tuple[float, ...]) -> tuple[float, ...]:
    from chemworld.physchem.composition_math import normalized_fractions

    return normalized_fractions(fractions)


def _validate_string_tuple(values: Sequence[str], field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} values must be strings")
        if not value or value.strip() != value:
            raise ValueError(f"{field_name} values cannot be empty or padded")
        if value in seen:
            raise ValueError(f"{field_name} values must be unique")
        seen.add(value)


def _component_identity_tokens(component: ComponentSpec) -> tuple[str, ...]:
    tokens = [component.identifier, *component.aliases]
    if component.cas_number:
        tokens.extend((component.cas_number, component.cas_number.replace("-", "")))
    if component.inchi:
        tokens.append(component.inchi)
    if component.inchi_key:
        tokens.append(component.inchi_key)
    return tuple(tokens)


def _validate_component_phase_compatibility(
    components: Sequence[ComponentSpec],
    phase_label: str,
) -> None:
    allowed = _PHASE_COMPATIBILITY[phase_label]
    incompatible = [
        component.identifier for component in components if component.default_phase not in allowed
    ]
    if incompatible:
        raise ValueError(
            f"Components are not compatible with phase {phase_label!r}: {incompatible}"
        )


def _validate_coefficients(coefficients: dict[str, float], equation_id: str) -> None:
    required = _required_coefficients(equation_id)
    missing = sorted(required - set(coefficients))
    if missing:
        raise ValueError(f"Missing coefficients for {equation_id}: {missing}")
    for key, value in coefficients.items():
        if not isinstance(value, int | float):
            raise ValueError(f"Coefficient {key!r} must be numeric")
        if not isfinite(float(value)):
            raise ValueError(f"Coefficient {key!r} must be finite")


def _required_coefficients(equation_id: str) -> set[str]:
    required = _EQUATION_CONTRACTS[equation_id]["required_coefficients"]
    if not isinstance(required, set):
        raise TypeError("Internal equation contract required_coefficients must be a set")
    return {str(value) for value in required}


def _coerce_component_provenance(
    value: ComponentProvenance | dict[str, object],
) -> ComponentProvenance:
    if isinstance(value, ComponentProvenance):
        return value
    if isinstance(value, dict):
        return ComponentProvenance.from_dict(value)
    raise ValueError("provenance entries must be ComponentProvenance or dict")


def _coerce_component_uncertainty(
    value: ComponentUncertainty | dict[str, object],
) -> ComponentUncertainty:
    if isinstance(value, ComponentUncertainty):
        return value
    if isinstance(value, dict):
        return ComponentUncertainty.from_dict(value)
    raise ValueError("uncertainty entries must be ComponentUncertainty or dict")


def _coerce_component_field_candidate(
    value: ComponentFieldCandidate | dict[str, object],
) -> ComponentFieldCandidate:
    if isinstance(value, ComponentFieldCandidate):
        return value
    if isinstance(value, dict):
        return ComponentFieldCandidate.from_dict(value)
    raise ValueError("field candidates must be ComponentFieldCandidate or dict")


def _coerce_component_conflict_policy(
    value: ComponentConflictPolicy | dict[str, object],
) -> ComponentConflictPolicy:
    if isinstance(value, ComponentConflictPolicy):
        return value
    if isinstance(value, dict):
        return ComponentConflictPolicy.from_dict(value)
    raise ValueError("conflict policy must be ComponentConflictPolicy or dict")


def _source_rank(source_id: str, policy: ComponentConflictPolicy) -> int:
    try:
        return policy.source_priority.index(source_id)
    except ValueError:
        return len(policy.source_priority)


def _values_within_tolerance(
    left: object,
    right: object,
    *,
    field_id: str,
    policy: ComponentConflictPolicy,
) -> bool:
    if isinstance(left, int | float) and isinstance(right, int | float):
        left_float = float(left)
        right_float = float(right)
        rtol = policy.field_rtol.get(field_id, policy.default_rtol)
        atol = policy.field_atol.get(field_id, policy.default_atol)
        return abs(left_float - right_float) <= atol + rtol * abs(left_float)
    return left == right


def property_equation_contracts() -> dict[str, dict[str, object]]:
    """Return JSON-friendly contracts for supported property equations."""

    contracts: dict[str, dict[str, object]] = {}
    for equation_id, contract in _EQUATION_CONTRACTS.items():
        input_dimensions = contract["input_dimensions"]
        if not isinstance(input_dimensions, dict):
            raise TypeError("Internal equation contract input_dimensions must be a mapping")
        contracts[equation_id] = {
            "required_coefficients": sorted(_required_coefficients(equation_id)),
            "input_dimensions": {
                str(field): str(dimension) for field, dimension in input_dimensions.items()
            },
            "output_dimension": str(contract["output_dimension"]),
        }
        if equation_id in CORRELATION_FAMILIES:
            contracts[equation_id].update(CORRELATION_FAMILIES[equation_id].to_dict())
    return contracts


def supported_property_equations() -> tuple[str, ...]:
    """Return stable equation identifiers accepted by `PropertyCorrelation`."""

    return tuple(sorted(_EQUATION_CONTRACTS))


__all__ = [
    "ComponentConflictPolicy",
    "ComponentConflictResolution",
    "ComponentFieldCandidate",
    "ComponentProvenance",
    "ComponentSpec",
    "ComponentUncertainty",
    "MixtureSpec",
    "PropertyCorrelation",
    "component_alias_index",
    "mass_fractions_from_mole_fractions",
    "mole_fractions_from_mass_fractions",
    "normalize_component_token",
    "property_equation_contracts",
    "resolve_component_field_conflict",
    "resolve_component_identifier",
    "supported_property_equations",
]
