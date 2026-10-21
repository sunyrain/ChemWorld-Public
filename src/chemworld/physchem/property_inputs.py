"""Auditable import envelope for the existing component/property package."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType

from chemworld.data.scientific_inputs import (
    InputSource,
    _keys,
    _number,
    _sequence,
    _text,
    strict_json_loads,
)
from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, ComponentUncertainty, PropertyCorrelation

PROPERTY_INPUT_VERSION = "chemworld-property-input-1"


def _correlation_from_input(raw, *, segment=False):
    if "model" in raw:
        if segment:
            raise ValueError("piecewise input segments must be individual correlations")
        _keys(raw, ("model", "correlation_id", "source_note", "segments"))
        if raw["model"] != "piecewise-heat-capacity-v1":
            raise ValueError("unsupported imported correlation model")
        return PiecewiseHeatCapacity(raw["correlation_id"], tuple(
            _correlation_from_input(value, segment=True)
            for value in _sequence(raw["segments"], "segments")), raw["source_note"])
    _keys(raw, (
        "correlation_id", "property_id", "equation_id", "coefficients",
        "input_units", "output_unit", "validity_ranges", "source_note", "metadata",
    ))
    # Existing codecs remain authoritative; validate before their historical
    # float coercion, including every segment of a newly imported curve.
    if not isinstance(raw["coefficients"], Mapping):
        raise ValueError("correlation coefficients must be an object")
    if not isinstance(raw["validity_ranges"], Mapping):
        raise ValueError("correlation validity_ranges must be an object")
    for key, value in raw["coefficients"].items():
        _number(value, f"coefficient.{key}")
    for bounds in raw["validity_ranges"].values():
        if len(bounds) != 2:
            raise ValueError("imported validity range requires exactly two bounds")
        for value in bounds:
            _number(value, "validity range")
    return PropertyCorrelation.from_dict(dict(raw))


@dataclass(frozen=True)
class PropertyInputCard:
    package: ComponentPropertyPackage
    sources: tuple[InputSource, ...]
    uncertainties: Mapping[str, ComponentUncertainty]
    reference: CaloricReference | None
    reference_note: str

    def __post_init__(self):
        _text(self.reference_note, "reference_note")
        sources = {source.provenance.source_id for source in self.sources}
        if not sources or len(sources) != len(self.sources):
            raise ValueError("property input needs unique declared sources")
        correlations = {c.correlation_id: c for c in self.package.correlations}
        if not correlations or set(correlations) != set(self.uncertainties):
            raise ValueError("every imported correlation needs uncertainty/source metadata")
        for key, correlation in correlations.items():
            _text(correlation.source_note, "correlation.source_note")
            _text(correlation.property_id, "correlation.property_id")
            if set(correlation.validity_ranges) != set(correlation.input_units):
                raise ValueError("imported correlations need a domain for each input")
            for bounds in correlation.validity_ranges.values():
                if any(not isfinite(v) for v in bounds):
                    raise ValueError("imported correlation domains must be finite")
            uncertainty = self.uncertainties[key]
            if uncertainty.source_id not in sources or uncertainty.field_id != key:
                raise ValueError("uncertainty must identify its correlation and declared source")
            _text(uncertainty.note, "uncertainty.note")
            _text(uncertainty.coverage, "uncertainty.coverage")
            for value in (uncertainty.standard_uncertainty, uncertainty.relative_uncertainty):
                if value is not None and _number(value, "uncertainty") < 0:
                    raise ValueError("uncertainty must be nonnegative")
            if (uncertainty.standard_uncertainty is not None
                    and uncertainty.unit != correlation.output_unit):
                raise ValueError("correlation standard uncertainty must use its output unit")
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "uncertainties", MappingProxyType(dict(self.uncertainties)))

    def evaluate_state(self, **request):
        """Use the existing property evaluator and the explicitly imported datum."""
        if "reference" in request:
            raise ValueError("replace the input card explicitly to change its reference")
        if self.reference is None and set(request.get("quantities", ())) & {
            "enthalpy", "internal_energy", "entropy", "chemical_potential",
        }:
            raise ValueError(
                "imported energy/entropy requests require an explicit caloric reference"
            )
        return self.package.evaluate_state(reference=self.reference, **request)

    def to_dict(self):
        return {
            "schema_version": PROPERTY_INPUT_VERSION, "package": self.package.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "uncertainties": {key: value.to_dict() for key, value in self.uncertainties.items()},
            "reference": None if self.reference is None else self.reference.to_dict(),
            "reference_note": self.reference_note,
        }

    @classmethod
    def from_dict(cls, payload):
        _keys(payload, (
            "schema_version", "package", "sources", "uncertainties", "reference", "reference_note",
        ))
        if payload["schema_version"] != PROPERTY_INPUT_VERSION:
            raise ValueError("unsupported property input schema")
        package = payload["package"]
        _keys(package, ("component", "correlations"))
        correlations = tuple(_correlation_from_input(raw)
            for raw in _sequence(package["correlations"], "correlations"))
        reference = payload["reference"]
        if reference is not None:
            _keys(reference, (
                "phase", "temperature_K", "pressure_Pa", "energy_basis", "energy_J_mol",
                "entropy_J_mol_K", "source_note",
            ))
            for key in ("temperature_K", "pressure_Pa", "energy_J_mol", "entropy_J_mol_K"):
                if key != "entropy_J_mol_K" or reference[key] is not None:
                    _number(reference[key], f"reference.{key}")
            reference = CaloricReference(**reference)
        uncertainties = {}
        if not isinstance(payload["uncertainties"], Mapping):
            raise ValueError("correlation uncertainties must be an object")
        for key, raw in payload["uncertainties"].items():
            if not isinstance(raw, Mapping):
                raise ValueError("each uncertainty must be an object")
            for field in ("standard_uncertainty", "relative_uncertainty"):
                if raw.get(field) is not None:
                    _number(raw[field], field)
            uncertainties[key] = ComponentUncertainty.from_dict(dict(raw))
        return cls(
            ComponentPropertyPackage(ComponentSpec.from_dict(dict(package["component"])),
                                     tuple(correlations)),
            tuple(InputSource.from_dict(raw) for raw in _sequence(payload["sources"], "sources")),
            uncertainties, reference, payload["reference_note"],
        )

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(strict_json_loads(text))
