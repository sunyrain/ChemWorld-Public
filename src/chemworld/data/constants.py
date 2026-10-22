"""Lazy external constant tables and saturation data with native unit semantics."""

from __future__ import annotations

import csv
import gzip
import json
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from types import MappingProxyType

from chemworld.data.constant_catalog import (
    CONSTANT_SOURCES,
    ELEMENT_TEXT_DEFAULTS,
    ELEMENT_TEXT_FIELDS,
    ELEMENT_TEXT_UNITS,
    TABULAR_FIELDS,
    VDI_UNITS,
    XML_FIELD_ALIASES,
    XML_UNIT_NAMES,
    ConstantField,
)
from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    ScientificRow,
    ScientificTable,
)
from chemworld.foundation.units import Quantity, canonical_unit
from chemworld.physchem.identifiers import check_cas_number, int_to_cas

_SOURCES = {source.source_id: source for source in CONSTANT_SOURCES}


class ConstantSourceUnavailableError(FileNotFoundError):
    pass


class ConstantSourceFormatError(ValueError):
    pass


class UnknownConstantMethodError(ValueError):
    pass


class ConstantIdentityNotFoundError(LookupError):
    pass


class AmbiguousConstantIdentityError(LookupError):
    pass


class ConstantFieldNotFoundError(LookupError):
    pass


class ConstantValueUnavailableError(ValueError):
    pass


class ConstantUnitUnavailableError(ValueError):
    pass


class NonScalarConstantError(ValueError):
    pass


class ConflictingConstantDataError(ValueError):
    def __init__(self, candidates):
        self.candidates = tuple(candidates)
        super().__init__("constant sources disagree; select a method or explicit source priority")


def _cas(value):
    text = str(value).strip()
    if check_cas_number(text):
        return text
    if text.isascii() and text.isdigit():
        try:
            return int_to_cas(int(text))
        except ValueError:
            pass
    return None


def _lookup_key(identity):
    if isinstance(identity, bool) or not isinstance(identity, (str, int)):
        raise ValueError("constant identity requires text or an integer source key")
    text = str(identity).strip()
    if not text:
        raise ValueError("constant identity cannot be empty")
    cas = _cas(text)
    return ("cas", cas) if cas else ("source_key", text)


def _field_names(headers):
    counts = {}
    result = []
    for index, header in enumerate(headers):
        base = header.strip() or f"unnamed_{index}"
        count = counts.get(base, 0) + 1
        counts[base] = count
        result.append(base if count == 1 else f"{base}__{count}")
    return tuple(result)


@dataclass(frozen=True)
class SourceCorrelation:
    """Unevaluated complete XML correlation; implementation belongs to 22C-G."""

    xml: str
    source_unit: str


@dataclass(frozen=True, slots=True)
class ConstantRecord:
    source_id: str
    row_number: int
    source_key: str
    cas_number: str | None
    headers: tuple[str, ...]
    field_names: tuple[str, ...]
    raw_fields: tuple[object, ...]
    source_units: Mapping[str, str] = field(default_factory=dict)
    source_attributes: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    document_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.field_names) != len(self.raw_fields) or len(set(self.field_names)) != len(
            self.field_names
        ):
            raise ValueError("constant fields must have one unique name per value")
        object.__setattr__(self, "source_units", MappingProxyType(dict(self.source_units)))
        object.__setattr__(self, "source_attributes", MappingProxyType({
            name: MappingProxyType(dict(attributes))
            for name, attributes in self.source_attributes.items()
        }))
        object.__setattr__(self, "document_metadata",
                           MappingProxyType(dict(self.document_metadata)))

    @property
    def record_id(self):
        return f"{self.source_id}:{self.row_number}"

    @property
    def source(self):
        return _SOURCES[self.source_id]

    def raw(self, field_id):
        field_id = self.source_field(field_id)
        try:
            return self.raw_fields[self.field_names.index(field_id)]
        except ValueError as exc:
            raise ConstantFieldNotFoundError(f"{self.record_id}: no field {field_id!r}") from exc

    def source_field(self, field_id):
        if self.source.kind == "xml":
            return XML_FIELD_ALIASES.get(field_id, field_id)
        return field_id

    def descriptor(self, field_id):
        name = self.source_field(field_id)
        self.raw(name)
        if self.source.kind == "xml":
            unit = XML_UNIT_NAMES.get(self.source_units.get(name))
            description = f"ChemSep {name}; original unit={self.source_units.get(name)!r}"
            if isinstance(self.raw(name), SourceCorrelation):
                return ConstantField(
                    name, None, "Raw XML correlation, not a scalar constant", "22C-G"
                )
            return ConstantField(name, unit, description)
        if self.source.kind == "vdi":
            return ConstantField(
                name, VDI_UNITS.get(name), "VDI saturation data; original nulls retained"
            )
        if self.source.kind == "elements":
            return ConstantField(
                name,
                ELEMENT_TEXT_UNITS.get(name),
                "Historical Open Babel data; documented defaults are not observations",
            )
        if self.source_id == "LEGACY_ELEMENTS":
            # The historical CSV has no per-column unit declaration. Preserve
            # it as raw evidence rather than assigning an inferred unit.
            return ConstantField(
                name, None, "Historical atomic field without an established unit contract"
            )
        return TABULAR_FIELDS.get(
            name, ConstantField(name, None, "Raw source metadata or undocumented field")
        )

    def value(self, field_id, *, point=None):
        name = self.source_field(field_id)
        raw = self.raw(name)
        descriptor = self.descriptor(name)
        if isinstance(raw, SourceCorrelation):
            raise NonScalarConstantError(f"{self.record_id}/{name}: unevaluated correlation")
        if isinstance(raw, tuple):
            if isinstance(point, bool) or not isinstance(point, int) or not 0 <= point < len(raw):
                raise NonScalarConstantError(
                    f"{self.record_id}/{name}: select a valid temperature point"
                )
            raw = raw[point]
        elif point is not None:
            raise NonScalarConstantError(
                f"{self.record_id}/{name}: scalar field has no point index"
            )
        quality = "valid"
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            number, quality = None, "missing"
        else:
            try:
                if name == "linear" and raw in {"True", "False", True, False}:
                    number = float(raw in {"True", True})
                else:
                    if isinstance(raw, bool):
                        raise ValueError("a boolean is not a physical constant")
                    number = float(raw)
            except (TypeError, ValueError):
                number, quality = None, "invalid"
            else:
                if not isfinite(number):
                    number, quality = None, "invalid"
        if (
            quality == "valid"
            and self.source.kind == "elements"
            and name in ELEMENT_TEXT_DEFAULTS
            and number == ELEMENT_TEXT_DEFAULTS[name]
        ):
            quality = "source_default_or_ambiguous"
        return ConstantValue(self, name, raw, number, descriptor, quality, point)


@dataclass(frozen=True)
class ConstantValue:
    record: ConstantRecord
    field_id: str
    raw_value: object
    value: float | None
    descriptor: ConstantField
    quality: str
    point: int | None = None

    @property
    def unit(self):
        return self.descriptor.unit

    def quantity(self, target_unit=None):
        if self.quality != "valid":
            raise ConstantValueUnavailableError(
                f"{self.record.record_id}/{self.field_id}: {self.quality}"
            )
        if self.unit is None:
            raise ConstantUnitUnavailableError(
                f"{self.record.record_id}/{self.field_id}: no established physical unit"
            )
        return Quantity(self.value, self.unit).to(target_unit or canonical_unit(self.unit))

    def input_value(self, target_unit=None):
        if self.unit is None:
            raise ConstantUnitUnavailableError(f"{self.field_id}: no established physical unit")
        quality = "missing" if self.quality == "source_default_or_ambiguous" else self.quality
        note = (
            f"{self.record.source.nature}; uncertainty coverage unspecified; "
            f"{self.descriptor.description}"
        )
        if self.quality == "source_default_or_ambiguous":
            note += "; literal matches a documented source default; retained in raw record"
        answer = InputValue(
            self.value if quality == "valid" else None, self.unit, None, note, quality
        )
        return answer.to(target_unit or canonical_unit(self.unit))

    def to_dict(self):
        return {
            "record_id": self.record.record_id,
            "source_key": self.record.source_key,
            "source": self.record.source.provenance.to_dict(),
            "field_id": self.field_id,
            "point": self.point,
            "raw_value": self.raw_value,
            "value": self.value,
            "unit": self.unit,
            "quality": self.quality,
            "description": self.descriptor.description,
            "continuation": self.descriptor.continuation,
        }


@dataclass(frozen=True)
class SaturationSeries:
    record: ConstantRecord
    field_id: str
    temperatures_K: tuple[float, ...]
    values: tuple[ConstantValue, ...]

    def valid_points(self, target_unit=None):
        selected = [
            (temperature, value.quantity(target_unit).value)
            for temperature, value in zip(self.temperatures_K, self.values, strict=True)
            if value.quality == "valid"
        ]
        return tuple(t for t, _ in selected), tuple(v for _, v in selected)


class ConstantDatabase:
    """Read selected source files under an explicit chemicals data directory."""

    def __init__(self, source_directory):
        self.source_directory = Path(source_directory).resolve()
        self._records = {}
        self._indices = {}

    @property
    def catalog(self):
        return CONSTANT_SOURCES

    @property
    def loaded_sources(self):
        return tuple(self._records)

    def source(self, source_id):
        try:
            return _SOURCES[source_id]
        except KeyError as exc:
            raise UnknownConstantMethodError(f"unknown constant method {source_id!r}") from exc

    def records(self, source_id):
        source = self.source(source_id)
        if source_id in self._records:
            return self._records[source_id]
        path = self.source_directory / source.relative_path
        if not path.is_file():
            raise ConstantSourceUnavailableError(f"{source_id}: source file unavailable: {path}")
        try:
            result = self._read_records(source, path)
        except (ValueError, TypeError, IndexError, StopIteration, csv.Error, ET.ParseError) as exc:
            raise ConstantSourceFormatError(f"{source_id}: {exc}") from exc
        index = {}
        for record in result:
            aliases = [record.source_key, record.record_id]
            if record.cas_number:
                aliases.append(record.cas_number)
            for name in (
                "CompoundID",
                "Name",
                "name",
                "Chemical",
                "Compound",
                "NAME/T(K) J/mol/K",
                "symbol",
                "element symbol",
                "atomic_number",
            ):
                if name in record.field_names:
                    aliases.append(str(record.raw(name)))
            for key in dict.fromkeys(_lookup_key(value) for value in aliases if value.strip()):
                index.setdefault(key, []).append(record)
        self._records[source_id] = tuple(result)
        self._indices[source_id] = index
        return self._records[source_id]

    def _read_records(self, source, path):
        """Format hook shared by source-specific, read-only property data adapters."""
        if source.kind == "xml":
            return self._xml_records(source, path)
        if source.kind == "vdi":
            return self._vdi_records(source, path)
        if source.kind == "mapping-json":
            return self._mapping_records(source, path)
        return self._text_records(source, path)

    @staticmethod
    def _text_records(source, path):
        opener = gzip.open if source.kind == "tsv-gzip" else open
        with opener(path, mode="rt", encoding="utf-8", newline="") as stream:
            if source.kind == "elements":
                headers = ELEMENT_TEXT_FIELDS
                rows = csv.reader(
                    (line for line in stream if line.strip() and not line.startswith("#")),
                    delimiter="\t",
                )
            else:
                rows = csv.reader(stream, delimiter="\t")
                headers = tuple(next(rows))
            names = _field_names(headers)
            result = []
            for number, row in enumerate(rows, 1):
                if len(row) != len(headers):
                    raise ValueError(
                        f"row {number}: expected {len(headers)} fields, got {len(row)}"
                    )
                if "CAS" in names:
                    key = row[names.index("CAS")]
                elif source.kind == "elements":
                    key = row[0]
                else:
                    key = row[0]
                result.append(
                    ConstantRecord(
                        source.source_id, number, key, _cas(key), tuple(headers), names, tuple(row)
                    )
                )
        return tuple(result)

    @staticmethod
    def _mapping_records(source, path):
        """Flat source-keyed records; nulls, booleans and text remain original."""
        with path.open(encoding="utf8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("source JSON must map identities to field mappings")
        result = []
        for number, (key, values) in enumerate(payload.items(), 1):
            if not isinstance(values, dict) or any(
                not isinstance(k, str) or isinstance(v, dict | list)
                for k, v in values.items()
            ):
                raise ValueError("source JSON record requires scalar, text or null fields")
            names = tuple(values)
            result.append(ConstantRecord(source.source_id, number, key, _cas(key),
                names, _field_names(names), tuple(values.values())))
        return tuple(result)

    @staticmethod
    def _xml_records(source, path):
        root = ET.parse(path).getroot()
        if root.tag != "compounds":
            raise ValueError("unexpected ChemSep document root")
        result = []
        for number, node in enumerate(root, 1):
            names, values, units, attributes = [], [], {}, {}
            for item in node:
                names.append(item.tag)
                attributes[item.tag] = dict(item.attrib)
                if "value" in item.attrib and not len(item):
                    values.append(item.attrib["value"])
                else:
                    values.append(
                        SourceCorrelation(
                            ET.tostring(item, encoding="unicode"), item.attrib.get("units", "")
                        )
                    )
                if "units" in item.attrib:
                    units[item.tag] = item.attrib["units"]
            fields = dict(zip(names, values, strict=True))
            key = str(fields.get("CAS") or fields.get("LibraryIndex") or number)
            result.append(
                ConstantRecord(
                    source.source_id,
                    number,
                    key,
                    _cas(fields.get("CAS", "")),
                    tuple(names),
                    _field_names(names),
                    tuple(values),
                    units,
                    attributes,
                    dict(root.attrib),
                )
            )
        return tuple(result)

    @staticmethod
    def _vdi_records(source, path):
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("VDI data requires a mapping of compound identities")
        result = []
        for number, (key, values) in enumerate(payload.items(), 1):
            if not isinstance(values, dict) or "T" not in values:
                raise ValueError("VDI compound needs a field mapping and temperature axis")
            if not isinstance(values["T"], list) or not values["T"]:
                raise ValueError("VDI temperature axis requires a nonempty array")
            length = len(values["T"])
            if any(isinstance(value, list) and len(value) != length for value in values.values()):
                raise ValueError("VDI fields must align with every temperature point")
            names = tuple(values)
            result.append(
                ConstantRecord(
                    source.source_id,
                    number,
                    key,
                    _cas(key),
                    names,
                    names,
                    tuple(tuple(v) if isinstance(v, list) else v for v in values.values()),
                )
            )
        return tuple(result)

    def virial_input(self, identity, component, *, method, constant_source,
                     temperature_range_K, parameters=None, polar=None):
        """Select one source explicitly; use existing records without another cache."""
        from chemworld.data.virial_inputs import sourced_virial_input

        return sourced_virial_input(self, identity, component, method=method,
            constant_source=constant_source, temperature_range_K=temperature_range_K,
            parameters=parameters, polar=polar)

    def records_for(self, identity, *, method):
        self.records(method)
        return tuple(self._indices[method].get(_lookup_key(identity), ()))

    def record(self, identity, *, method):
        matches = self.records_for(identity, method=method)
        if not matches:
            raise ConstantIdentityNotFoundError(f"{method}: no record for {identity!r}")
        if len(matches) != 1:
            raise AmbiguousConstantIdentityError(
                f"{method}: {identity!r} matches {[r.record_id for r in matches]}; "
                "select a record id"
            )
        return matches[0]

    def value(self, identity, field_id, *, method, point=None):
        return self.record(identity, method=method).value(field_id, point=point)

    def candidates(self, identity, field_id, *, methods=None):
        methods = (tuple(source.source_id for source in self.catalog)
                   if methods is None else tuple(methods))
        if len(set(methods)) != len(methods):
            raise ValueError("duplicate constant methods")
        result = []
        for method in methods:
            for record in self.records_for(identity, method=method):
                if record.source_field(field_id) in record.field_names:
                    if isinstance(record.raw(field_id), (tuple, list, Mapping, SourceCorrelation)):
                        continue  # Temperature series/correlations are not scalar candidates.
                    result.append(record.value(field_id))
        return tuple(result)

    def methods(self, identity, field_id, *, sources=None):
        return tuple(
            dict.fromkeys(
                value.record.source_id
                for value in self.candidates(identity, field_id, methods=sources)
                if value.quality == "valid" and value.unit is not None
            )
        )

    def resolve(self, identity, field_id, *, methods=None, source_priority=()):
        values = self.candidates(identity, field_id, methods=methods)
        valid = [value for value in values if value.quality == "valid" and value.unit is not None]
        if not valid:
            raise ConstantValueUnavailableError(
                f"no quantified {field_id} for {identity!r}; inspect candidates"
            )
        for method in source_priority:
            self.source(method)
        if source_priority:
            for method in source_priority:
                selected = [v for v in valid if v.record.source_id == method]
                if selected:
                    valid = selected
                    break
        quantities = {(value.quantity().value, value.quantity().unit) for value in valid}
        if len(quantities) != 1:
            raise ConflictingConstantDataError(valid)
        return valid[0]

    def saturation_series(self, identity, field_id):
        record = self.record(identity, method="VDI")
        temperatures = tuple(float(value) for value in record.raw("T"))
        if any(not isfinite(t) or t <= 0 for t in temperatures):
            raise ConstantValueUnavailableError("VDI temperatures must be positive finite Kelvin")
        if not isinstance(record.raw(field_id), tuple):
            raise NonScalarConstantError("saturation series requires a tabulated property")
        values = tuple(record.value(field_id, point=i) for i in range(len(temperatures)))
        return SaturationSeries(record, field_id, temperatures, values)

    def parameter_table(self, identities, *, method, columns, field_bindings, dataset_id, group_id):
        columns = tuple(columns)
        if set(field_bindings) != {column.quantity_id for column in columns}:
            raise ValueError("every parameter column needs one explicit source field binding")
        source = self.source(method)
        rows = []
        for identity in identities:
            record = self.record(identity, method=method)
            rows.append(
                ScientificRow(
                    record.record_id,
                    source.provenance.source_id,
                    group_id,
                    record.record_id,
                    "parameters",
                    {
                        column.quantity_id: record.value(
                            field_bindings[column.quantity_id]
                        ).input_value(column.unit)
                        for column in columns
                    },
                )
            )
        return ScientificTable(
            dataset_id,
            "parameters",
            columns,
            (InputSource(source.provenance, "literature"),),
            tuple(rows),
        )

    def saturation_table(self, identity, *, columns, field_bindings, dataset_id, group_id, split):
        record = self.record(identity, method="VDI")
        columns = tuple(columns)
        if set(field_bindings) != {column.quantity_id for column in columns}:
            raise ValueError("every saturation column needs one explicit source field binding")
        rows = []
        for point in range(len(record.raw("T"))):
            values = {
                column.quantity_id: record.value(
                    field_bindings[column.quantity_id], point=point
                ).input_value(column.unit)
                for column in columns
            }
            row_id = f"{record.record_id}:{point}"
            rows.append(
                ScientificRow(
                    row_id, record.source.provenance.source_id, group_id, row_id, split, values
                )
            )
        return ScientificTable(
            dataset_id,
            "measurements",
            columns,
            (InputSource(record.source.provenance, "literature"),),
            tuple(rows),
        )
