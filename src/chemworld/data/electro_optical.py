"""Electro-optical consumers of the original external constant sources.

Only the CRC dielectric table is a new source. RI and dipole records retain
the same database, identities, values, candidates and source provenance.
"""

from dataclasses import dataclass, fields
from math import isfinite
from typing import cast

from chemworld.data.constant_catalog import CONSTANT_SOURCES, ConstantField, ConstantSource
from chemworld.data.constants import ConstantDatabase, ConstantRecord
from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_packages import ComponentPropertyPackage
from chemworld.physchem.property_reports import PropertyEvaluation
from chemworld.physchem.specs import ComponentUncertainty, PropertyCorrelation

DIPOLE_SOURCES = ("CCCBDB", "MULLER", "POLING", "PSI4_DIPOLE")
REFRACTIVE_INDEX_SOURCES = ("CRC_RI", "WIKIDATA")
_CRC_PERMITTIVITY = ConstantSource(
    "CRC_PERMITTIVITY",
    "Electrolytes/Permittivity (Dielectric Constant) of Liquids.tsv",
    citation="CRC Handbook, 2014; chemicals 1.5.2 dielectric source table",
)
ELECTRO_OPTICAL_SOURCES = (
    _CRC_PERMITTIVITY,
    *tuple(
        s for s in CONSTANT_SOURCES if s.source_id in {*DIPOLE_SOURCES, *REFRACTIVE_INDEX_SOURCES}
    ),
)


def _metadata(record):
    return {
        "source": record.source.provenance.to_dict(),
        "source_record": {
            "record_id": record.record_id,
            "source_key": record.source_key,
            "fields": dict(zip(record.field_names, record.raw_fields, strict=True)),
        },
    }


def _point_curve(identity, prop, temperature, value, note, metadata):
    return PropertyCorrelation(
        identity + ":" + prop,
        "dimensionless_source_point",
        {"T_ref": temperature, "value": value},
        {"temperature": "K"},
        "dimensionless",
        prop,
        {"temperature": (temperature, temperature)},
        note,
        metadata,
    )


@dataclass(frozen=True)
class PermittivityRecord(ConstantRecord):
    @property
    def source(self):
        return _CRC_PERMITTIVITY

    def descriptor(self, field_id):
        self.raw(field_id)
        return ConstantField(
            field_id,
            "K"
            if field_id in {"T", "Tmin", "Tmax"}
            else "dimensionless"
            if field_id in {"Permittivity", "A"}
            else None,
            "CRC relative permittivity; raw coefficients use declared polynomial powers",
            "22F2",
        )

    def correlation(self, *, source_point=False, temperature_bounds_K=None):
        if type(source_point) is not bool:
            raise ValueError("source_point must be boolean")

        def number(key, required=False):
            v = self.value(key)
            if v.quality == "invalid":
                raise ValueError(f"invalid dielectric field {key}")
            if v.quality != "valid":
                if required:
                    raise ValueError(f"missing or invalid dielectric field {key}")
                return None
            return v.value

        metadata = _metadata(self)
        if source_point:
            if temperature_bounds_K is not None:
                raise ValueError("single source observation has no application interval")
            T, value = number("T", True), number("Permittivity", True)
            if T <= 0 or value <= 0:
                raise ValueError("dielectric source point needs positive T and value")
            return _point_curve(
                self.record_id,
                "relative_permittivity",
                T,
                value,
                self.source.provenance.citation,
                metadata,
            )
        p = {"A": number("A", True)}
        omitted = []
        for key in "BCD":
            value = number(key)
            if value is None:
                # Source convention: the highest supplied term is the degree.
                # An interior gap is not an omitted high-order coefficient.
                if any(number(k) is not None for k in "BCD"["BCD".index(key) + 1 :]):
                    raise ValueError(f"interior missing dielectric polynomial coefficient {key}")
                value = 0
                omitted.append(key)
            p[key] = value
        source_bounds = number("Tmin"), number("Tmax")
        bounds = tuple(temperature_bounds_K) if temperature_bounds_K is not None else source_bounds
        if (
            len(bounds) != 2
            or any(
                isinstance(t, bool) or not isinstance(t, int | float) or not isfinite(t)
                for t in bounds
            )
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError(
                "dielectric fit needs an explicit finite positive temperature interval"
            )
        if any(
            s is not None and (b < s if i == 0 else b > s)
            for i, (b, s) in enumerate(zip(bounds, source_bounds, strict=True))
        ):
            raise ValueError("application interval exceeds dielectric source bounds")
        return PropertyCorrelation(
            self.record_id + ":relative_permittivity",
            "permittivity_crc",
            p,
            {"temperature": "K"},
            "dimensionless",
            "relative_permittivity",
            {"temperature": bounds},
            self.source.provenance.citation,
            {
                **metadata,
                "source_bounds_K": list(source_bounds),
                "omitted_high_order_terms": omitted,
                "temperature_range_basis": "source"
                if temperature_bounds_K is None
                else "caller_application",
            },
        )


@dataclass(frozen=True)
class RefractiveIndexObservation:
    """A reported Na D-line index with its known or unknown temperature."""

    component_id: str
    index: InputValue
    temperature: InputValue
    source: DataSourceProvenance
    record_id: str

    def __post_init__(self) -> None:
        if not self.component_id or not self.record_id:
            raise ValueError("RI observation needs component and record identities")
        for cell, unit in ((self.index, "dimensionless"), (self.temperature, "K")):
            value = cell.to(unit).value
            if value is not None and value <= 0:
                raise ValueError("RI observation and temperature must be positive if supplied")

    @classmethod
    def from_record(cls, record):
        if record.source_id not in REFRACTIVE_INDEX_SOURCES:
            raise ValueError("select a CRC or Wikidata refractive-index record")
        return cls(
            record.cas_number or record.source_key,
            record.value("RI").input_value(),
            record.value("RIT").input_value("K"),
            record.source.provenance,
            record.record_id,
        )

    def evaluate(self):
        if self.index.quality != "valid" or self.index.value is None:
            raise ValueError("source refractive index is missing or invalid")
        if self.temperature.quality not in {"valid", "missing"}:
            raise ValueError("source refractive-index temperature is invalid")
        inputs, warnings = {}, ()
        if self.temperature.value is not None:
            inputs["temperature"] = cast(float, self.temperature.to("K").value)
        else:
            warnings = ("Source refractive-index measurement temperature unavailable",)
        return PropertyEvaluation(
            "refractive_index",
            self.record_id,
            "Na_D_source_observation",
            cast(float, self.index.to("dimensionless").value),
            "dimensionless",
            inputs,
            warnings,
        )

    def correlation(self):
        report = self.evaluate()
        if "temperature" not in report.inputs:
            raise ValueError(
                "unknown measurement temperature cannot define a temperature correlation"
            )
        return _point_curve(
            self.record_id,
            "refractive_index",
            report.inputs["temperature"],
            report.value,
            self.source.citation,
            {
                "source": self.source.to_dict(),
                "spectral_condition": "sodium D line; exact wavelength not supplied",
                "observation": self.to_dict(),
            },
        )

    def to_dict(self):
        return {
            "model": "refractive-index-observation-1",
            "component_id": self.component_id,
            "index": self.index.to_dict(),
            "temperature": self.temperature.to_dict(),
            "source": self.source.to_dict(),
            "record_id": self.record_id,
        }

    @classmethod
    def from_dict(cls, payload):
        if set(payload) != {
            "model",
            "component_id",
            "index",
            "temperature",
            "source",
            "record_id",
        } or (payload["model"] != "refractive-index-observation-1"):
            raise ValueError("unknown refractive-index observation schema")
        return cls(
            payload["component_id"],
            InputValue.from_dict(payload["index"]),
            InputValue.from_dict(payload["temperature"]),
            DataSourceProvenance.from_dict(payload["source"]),
            payload["record_id"],
        )


class ElectroOpticalDatabase(ConstantDatabase):
    @property
    def catalog(self):
        return ELECTRO_OPTICAL_SOURCES

    def source(self, source_id):
        return _CRC_PERMITTIVITY if source_id == "CRC_PERMITTIVITY" else super().source(source_id)

    def _read_records(self, source, path):
        records = super()._read_records(source, path)
        if source.source_id != "CRC_PERMITTIVITY":
            return records
        return tuple(
            PermittivityRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def refractive_index(self, identity, *, method):
        return RefractiveIndexObservation.from_record(self.record(identity, method=method))

    def property_input(
        self, identity, component, *, method, source_point=False, temperature_bounds_K=None
    ):
        if type(source_point) is not bool:
            raise ValueError("source_point must be boolean")
        record = self.record(identity, method=method)
        if (record.cas_number or record.source_key) not in {
            component.cas_number,
            component.identifier,
        }:
            raise ValueError("electro-optical source identity does not match component")
        if method == "CRC_PERMITTIVITY":
            curve = cast(PermittivityRecord, record).correlation(
                source_point=source_point, temperature_bounds_K=temperature_bounds_K
            )
        elif method in REFRACTIVE_INDEX_SOURCES:
            if not source_point or temperature_bounds_K is not None:
                raise ValueError("source RI needs explicit source_point=True without an interval")
            curve = RefractiveIndexObservation.from_record(record).correlation()
        else:
            raise ValueError(
                "no temperature correlation for this source; use its original InputValue"
            )
        source = record.source.provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)),
            (InputSource(source, "literature"),),
            {
                curve.correlation_id: ComponentUncertainty(
                    curve.correlation_id,
                    unit="dimensionless",
                    source_id=source.source_id,
                    coverage="unspecified",
                    note="Original source; uncertainty not quantified",
                )
            },
            None,
            "Electro-optical property only; no caloric reference or implicit state model",
        )
