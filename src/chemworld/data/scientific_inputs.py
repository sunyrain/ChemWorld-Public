"""Finite, unit-aware parameter and measurement tables at the input boundary.

Tables are evidence/configuration, never live simulator state. Splits are
supplied by the experiment author and kept at the experiment-group level.
"""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Never, cast

from chemworld.foundation.units import convert_value, unit_spec
from chemworld.physchem.data_conflicts import DataSourceProvenance

SCIENTIFIC_TABLE_VERSION = "chemworld-scientific-table-1"
DATA_KINDS = {"synthetic", "measured", "literature"}
QUALITIES = {"valid", "missing", "invalid", "below_detection", "saturated", "out_of_domain"}
_CELL_FIELDS = ("value", "unit", "uncertainty", "uncertainty_note", "quality")
_ROW_FIELDS = ("row_id", "source_id", "group_id", "sample_id", "split")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} requires nonempty text without surrounding whitespace")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} requires a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} exceeds the numeric input domain") from exc
    if not isfinite(number):
        raise ValueError(f"{field} requires a finite number")
    if isinstance(value, int) and int(number) != value:
        raise ValueError(f"{field} integer cannot be represented exactly by this numeric contract")
    return number


def _keys(payload: object, required: Collection[str], *, optional: Collection[str] = ()) -> None:
    if not isinstance(payload, Mapping) or any(not isinstance(key, str) for key in payload):
        raise ValueError("scientific input must be an object")
    missing, extra = set(required) - payload.keys(), payload.keys() - set(required) - set(optional)
    if missing or extra:
        raise ValueError(
            f"scientific input fields: missing={sorted(missing)}, unknown={sorted(extra)}"
        )


def _sequence(value: object, field: str) -> Sequence[Any]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be a list")
    return value


def strict_json_loads(text: str) -> Any:
    """Decode raw JSON; domain constructors validate its shape and scalar fields."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> Never:
        raise ValueError(f"nonfinite JSON number: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


@dataclass(frozen=True)
class InputSource:
    provenance: DataSourceProvenance
    data_kind: str

    def __post_init__(self) -> None:
        for key in ("source_id", "citation", "license", "version"):
            _text(getattr(self.provenance, key), f"source.{key}")
        if self.data_kind not in DATA_KINDS:
            raise ValueError("data_kind must be synthetic, measured or literature")
        # An explicit 'unknown/not established' licence is a declaration, not
        # permission to redistribute. No importer infers licensing from code.

    def to_dict(self) -> dict[str, Any]:
        return {"provenance": self.provenance.to_dict(), "data_kind": self.data_kind}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InputSource:
        _keys(payload, ("provenance", "data_kind"))
        raw = payload["provenance"]
        _keys(
            raw, ("source_id", "priority", "citation", "version", "license"),
            optional=("checksum",),
        )
        for key in ("source_id", "citation", "version", "license"):
            _text(raw[key], f"source.{key}")
        if isinstance(raw["priority"], bool) or not isinstance(raw["priority"], int):
            raise ValueError("source priority must be an integer")
        return cls(DataSourceProvenance.from_dict(dict(raw)), payload["data_kind"])


@dataclass(frozen=True)
class InputValue:
    value: float | None
    unit: str
    uncertainty: float | None
    uncertainty_note: str
    quality: str = "valid"

    def __post_init__(self) -> None:
        unit_spec(self.unit)
        _text(self.uncertainty_note, "uncertainty_note")
        if self.quality not in QUALITIES:
            raise ValueError("unsupported input quality")
        if self.value is None:
            if self.quality not in {"missing", "invalid"}:
                raise ValueError("only missing/invalid observations can omit a value")
        else:
            _number(self.value, "value")
            if self.quality == "missing":
                raise ValueError("a missing observation cannot carry a numeric value")
        if self.uncertainty is not None and _number(self.uncertainty, "uncertainty") < 0:
            raise ValueError("standard uncertainty must be nonnegative")

    def to(self, unit: str) -> InputValue:
        # Standard uncertainty is a difference, so Celsius offsets cancel.
        scale = abs(convert_value(1.0, self.unit, unit) - convert_value(0.0, self.unit, unit))
        return replace(
            self, unit=unit,
            value=None if self.value is None else convert_value(self.value, self.unit, unit),
            uncertainty=None if self.uncertainty is None else self.uncertainty * scale,
        )

    def to_dict(self) -> dict[str, str | float | None]:
        return {key: getattr(self, key) for key in _CELL_FIELDS}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InputValue:
        _keys(payload, _CELL_FIELDS)
        return cls(**payload)


@dataclass(frozen=True)
class QuantityColumn:
    quantity_id: str
    unit: str
    role: str
    valid_range: tuple[float, float]
    reference_state: str

    def __post_init__(self) -> None:
        _text(self.quantity_id, "quantity_id")
        _text(self.reference_state, "reference_state")
        unit_spec(self.unit)
        if self.role not in {"parameter", "condition", "observation"}:
            raise ValueError("column role must be parameter, condition or observation")
        if len(self.valid_range) != 2:
            raise ValueError("valid_range requires two finite bounds")
        lower, upper = (_number(v, "valid_range") for v in self.valid_range)
        if lower >= upper:
            raise ValueError("valid_range must be increasing")
        object.__setattr__(self, "valid_range", (lower, upper))

    def normalize(self, value: InputValue) -> InputValue:
        normalized = value.to(self.unit)
        if self.role != "observation" and normalized.quality != "valid":
            raise ValueError("parameters and conditions must be valid numeric values")
        if normalized.quality == "valid" and not (
            self.valid_range[0] <= cast(float, normalized.value) <= self.valid_range[1]
        ):
            raise ValueError(f"{self.quantity_id} is outside its declared valid range")
        return normalized

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantity_id": self.quantity_id, "unit": self.unit, "role": self.role,
            "valid_range": list(self.valid_range), "reference_state": self.reference_state,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> QuantityColumn:
        _keys(payload, ("quantity_id", "unit", "role", "valid_range", "reference_state"))
        return cls(**payload)


@dataclass(frozen=True)
class ScientificRow:
    row_id: str
    source_id: str
    group_id: str
    sample_id: str
    split: str
    values: Mapping[str, InputValue]

    def __post_init__(self) -> None:
        for key in _ROW_FIELDS:
            _text(getattr(self, key), key)
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: getattr(self, key) for key in _ROW_FIELDS},
            "values": {key: value.to_dict() for key, value in self.values.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScientificRow:
        _keys(payload, (*_ROW_FIELDS, "values"))
        if not isinstance(payload["values"], Mapping):
            raise ValueError("row values must be an object")
        return cls(
            **{key: payload[key] for key in _ROW_FIELDS},
            values={key: InputValue.from_dict(value) for key, value in payload["values"].items()},
        )


@dataclass(frozen=True)
class ScientificTable:
    dataset_id: str
    kind: str
    columns: tuple[QuantityColumn, ...]
    sources: tuple[InputSource, ...]
    rows: tuple[ScientificRow, ...]

    def __post_init__(self) -> None:
        _text(self.dataset_id, "dataset_id")
        if self.kind not in {"parameters", "measurements"}:
            raise ValueError("table kind must be parameters or measurements")
        for name in ("columns", "sources", "rows"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
            if not getattr(self, name):
                raise ValueError(f"{name} cannot be empty")
        columns = {column.quantity_id: column for column in self.columns}
        sources = {source.provenance.source_id for source in self.sources}
        if len(columns) != len(self.columns) or len(sources) != len(self.sources):
            raise ValueError("duplicate column or source")
        roles = {column.role for column in self.columns}
        if (self.kind == "parameters" and roles != {"parameter"}) or (
            self.kind == "measurements" and ("parameter" in roles or "observation" not in roles)
        ):
            raise ValueError("column roles disagree with table kind")
        splits = {"parameters"} if self.kind == "parameters" else {
            "train", "validation", "test", "unused",
        }
        seen_rows: set[str] = set()
        seen_samples: set[tuple[str, str]] = set()
        group_splits: dict[str, str] = {}
        normalized: list[ScientificRow] = []
        for row in self.rows:
            if row.source_id not in sources:
                raise ValueError(f"undefined source: {row.source_id}")
            if row.row_id in seen_rows or (row.source_id, row.sample_id) in seen_samples:
                raise ValueError("duplicate row or source/acquisition sample identity")
            seen_rows.add(row.row_id)
            seen_samples.add((row.source_id, row.sample_id))
            if row.split not in splits:
                raise ValueError("split is incompatible with table kind")
            if row.group_id in group_splits and group_splits[row.group_id] != row.split:
                raise ValueError("one experiment group cannot cross data splits")
            group_splits[row.group_id] = row.split
            if set(row.values) != set(columns):
                raise ValueError("row must supply every declared column exactly once")
            normalized.append(replace(row, values={
                key: column.normalize(row.values[key]) for key, column in columns.items()
            }))
        object.__setattr__(self, "rows", tuple(normalized))

    def rows_for(self, split: str) -> tuple[ScientificRow, ...]:
        allowed = {"parameters"} if self.kind == "parameters" else {
            "train", "validation", "test", "unused",
        }
        if split not in allowed:
            raise ValueError("unknown data split")
        return tuple(row for row in self.rows if row.split == split)

    def summary(self) -> dict[str, str | int | dict[str, int]]:
        kinds = {source.provenance.source_id: source.data_kind for source in self.sources}
        return {
            "dataset_id": self.dataset_id, "kind": self.kind,
            "rows": len(self.rows), "groups": len({row.group_id for row in self.rows}),
            "splits": dict(Counter(row.split for row in self.rows)),
            "data_kinds": dict(Counter(kinds[row.source_id] for row in self.rows)),
            "observation_quality": dict(Counter(
                row.values[col.quantity_id].quality for row in self.rows
                for col in self.columns if col.role == "observation"
            )),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCIENTIFIC_TABLE_VERSION, "dataset_id": self.dataset_id,
            "kind": self.kind, "columns": [col.to_dict() for col in self.columns],
            "sources": [source.to_dict() for source in self.sources],
            "rows": [row.to_dict() for row in self.rows],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScientificTable:
        _keys(payload, ("schema_version", "dataset_id", "kind", "columns", "sources", "rows"))
        if payload["schema_version"] != SCIENTIFIC_TABLE_VERSION:
            raise ValueError("unsupported scientific table schema")
        return cls(
            payload["dataset_id"], payload["kind"],
            tuple(QuantityColumn.from_dict(v) for v in _sequence(payload["columns"], "columns")),
            tuple(InputSource.from_dict(v) for v in _sequence(payload["sources"], "sources")),
            tuple(ScientificRow.from_dict(v) for v in _sequence(payload["rows"], "rows")),
        )

    @classmethod
    def from_json(cls, text: str) -> ScientificTable:
        return cls.from_dict(strict_json_loads(text))

    def to_csv(self) -> str:
        stream = io.StringIO(newline="")
        names = [*_ROW_FIELDS, *(
            f"{col.quantity_id}.{field}" for col in self.columns for field in _CELL_FIELDS
        )]
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in self.rows:
            writer.writerow({
                **{key: getattr(row, key) for key in _ROW_FIELDS},
                **{f"{name}.{key}": value for name, cell in row.values.items()
                   for key, value in cell.to_dict().items()},
            })
        return stream.getvalue()

    @classmethod
    def from_csv(cls, text: str, metadata: Mapping[str, Any]) -> ScientificTable:
        """Load wide CSV with an explicit table metadata object (without rows).

        sample_id identifies one acquisition, including all its measured columns;
        repeated readings of the same specimen use separate acquisition IDs.
        """
        _keys(metadata, ("schema_version", "dataset_id", "kind", "columns", "sources"))
        reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
        columns = [col["quantity_id"] for col in metadata["columns"]]
        required = {*_ROW_FIELDS, *(f"{col}.{key}" for col in columns for key in _CELL_FIELDS)}
        if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
                or set(reader.fieldnames) != required):
            raise ValueError("CSV headers must match the declared columns and metadata fields")
        rows = []
        for raw in reader:
            if set(raw) != required or any(value is None for value in raw.values()):
                raise ValueError(f"CSV row {reader.line_num} has missing or extra cells")
            # Exact headers and row width above exclude DictReader's rest key/list.
            fields = cast(dict[str, str], raw)
            values = {}
            for column in columns:
                cell: dict[str, str | float | None] = {
                    key: fields[f"{column}.{key}"] for key in _CELL_FIELDS
                }
                for key in ("value", "uncertainty"):
                    text_value = fields[f"{column}.{key}"]
                    cell[key] = None if text_value == "" else float(text_value)
                values[column] = cell
            rows.append({**{key: raw[key] for key in _ROW_FIELDS}, "values": values})
        return cls.from_dict({**metadata, "rows": rows})
