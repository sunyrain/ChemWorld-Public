"""Read-only environmental/safety semantics on the original constant database.

Categories and source intervals are not floating-point constants. Missing
listings are explicit; all numeric observations use the original InputValue.
No regulatory currency, exposure duration or material state is inferred.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any, SupportsInt, cast

from chemworld.data.assessment_catalog import (
    ASSESSMENT_METHODS,
    ASSESSMENT_SOURCE_MAP,
    ASSESSMENT_SOURCES,
    IARC_CLASSIFICATIONS,
    NTP_CLASSIFICATIONS,
    AssessmentMethod,
)
from chemworld.data.constant_catalog import TABULAR_FIELDS, ConstantField, ConstantSource
from chemworld.data.constants import (
    ConstantDatabase,
    ConstantIdentityNotFoundError,
    ConstantRecord,
    UnknownConstantMethodError,
)
from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_reports import PropertyEvaluation


@dataclass(frozen=True)
class AssessmentValue:
    property_id: str
    method: str
    component_id: str
    record_id: str
    value: float | str | bool | None
    unit: str | None
    quality: str
    source: DataSourceProvenance
    raw_value: object
    conditions: Mapping[str, object]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all((self.property_id, self.method, self.component_id, self.record_id)):
            raise ValueError("assessment requires property, method, component and record")
        if self.quality not in {"valid", "missing", "invalid", "not_listed"}:
            raise ValueError("unknown assessment quality")
        if self.quality != "valid" and self.value is not None:
            raise ValueError("an unavailable assessment cannot carry a valid value")
        if self.quality == "valid" and self.value is None:
            raise ValueError("a valid assessment requires a value")
        if isinstance(self.value, int | float) and not isinstance(self.value, bool):
            if not isfinite(self.value) or self.unit is None:
                raise ValueError("numeric assessment needs a finite value and an explicit unit")
            self.input_value()
        elif self.unit is not None and self.value is not None:
            raise ValueError("categorical/text assessments have no numeric unit")
        object.__setattr__(self, "conditions", MappingProxyType(dict(self.conditions)))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def input_value(self) -> InputValue:
        if self.unit is None or isinstance(self.value, str | bool):
            raise ValueError("text/classification is not a numeric observation")
        return InputValue(
            self.value,
            self.unit,
            None,
            "Source uncertainty not quantified",
            "missing" if self.quality == "not_listed" else self.quality,
        )

    def evaluate(self) -> PropertyEvaluation:
        if self.quality != "valid":
            raise ValueError(f"assessment is {self.quality}")
        cell = self.input_value()
        return PropertyEvaluation(
            self.property_id,
            self.record_id,
            self.method,
            cast(float, cell.value),
            cell.unit,
            {k: cast(float, v) for k, v in self.conditions.items() if type(v) in (int, float)},
            self.warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "source-assessment-1",
            **{
                f.name: dict(self.conditions)
                if f.name == "conditions"
                else list(self.warnings)
                if f.name == "warnings"
                else self.source.to_dict()
                if f.name == "source"
                else getattr(self, f.name)
                for f in fields(self)
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AssessmentValue:
        if (
            set(payload) != {"model", *(f.name for f in fields(cls))}
            or payload["model"] != "source-assessment-1"
        ):
            raise ValueError("unknown source assessment schema")
        data = {k: v for k, v in payload.items() if k != "model"}
        data["source"] = DataSourceProvenance.from_dict(data["source"])
        return cls(**data)


@dataclass(frozen=True)
class AssessmentRecord(ConstantRecord):
    @property
    def source(self) -> ConstantSource:
        return ASSESSMENT_SOURCE_MAP[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        unit = None
        if (
            self.source_id.startswith("IPCC_")
            and ("yr GWP" in field_id or "yr GTP" in field_id or field_id == "SAR 100yr")
        ) or (self.source_id == "ODP" and field_id.startswith("ODP") and " " in field_id):
            unit = "dimensionless"
        elif self.source_id == "ONTARIO":
            if field_id.endswith(" (ppm)"):
                unit = "ppmv"
            elif field_id.endswith(" (mg/m^3)"):
                unit = "mg/m^3"
            elif field_id == "MW":
                unit = "g/mol"
        if unit is not None:
            return ConstantField(
                field_id, unit, "Versioned source value; original basis retained", "22F3"
            )
        return TABULAR_FIELDS.get(
            field_id,
            ConstantField(
                field_id,
                None,
                "Original text, classification, auxiliary quantity or undocumented unit",
                "22F3",
            ),
        )


class AssessmentDatabase(ConstantDatabase):
    @property
    def catalog(self) -> tuple[ConstantSource, ...]:
        return ASSESSMENT_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        return ASSESSMENT_SOURCE_MAP.get(source_id) or super().source(source_id)

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[ConstantRecord, ...]:
        original = super()._read_records(source, path)
        if source.source_id == "WIKIDATA" or source.source_id not in ASSESSMENT_SOURCE_MAP:
            return original
        return tuple(
            AssessmentRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in original
        )

    @staticmethod
    def method_specs(property_id: str) -> tuple[AssessmentMethod, ...]:
        result = tuple(m for m in ASSESSMENT_METHODS if m.property_id == property_id)
        if not result:
            raise UnknownConstantMethodError(f"unknown assessment property {property_id!r}")
        return result

    def assessment(
        self, identity: object, property_id: str, *, method: str, unit: str | None = None,
    ) -> AssessmentValue:
        specs = [m for m in self.method_specs(property_id) if m.method == method]
        if not specs:
            raise UnknownConstantMethodError(f"unknown {property_id} method {method!r}")
        spec = specs[0]
        if unit is not None and (
            property_id not in {"TWA", "STEL", "Ceiling"} or unit not in {"ppmv", "mg/m^3"}
        ):
            raise ValueError("unit selection is only for the two original exposure source units")
        try:
            record = self.record(identity, method=spec.source_id)
        except ConstantIdentityNotFoundError:
            return AssessmentValue(
                property_id,
                method,
                str(identity),
                f"{spec.source_id}:unlisted:{identity}",
                None,
                unit or spec.unit,
                "not_listed",
                self.source(spec.source_id).provenance,
                None,
                dict(spec.conditions),
                ("Not listed in this source snapshot; no inference of absence of hazard",),
            )
        conditions = dict(spec.conditions)
        if spec.source_id == "ONTARIO":
            conditions["numeric_view"] = (
                "Source JSON includes derived unit conversions; "
                "raw source annotations remain separate"
            )
        value: str | float | bool | None = None
        quality = "missing"
        raw: object = None
        selected_unit = spec.unit
        if property_id in {"TWA", "STEL", "Ceiling"}:
            source_units = {"ppmv": f"{property_id} (ppm)", "mg/m^3": f"{property_id} (mg/m^3)"}
            available = {u: record.raw(f) for u, f in source_units.items()}
            conditions["source_values"] = available
            selected_unit = unit or ("ppmv" if available["ppmv"] is not None else "mg/m^3")
            cell = record.value(source_units[selected_unit])
            value, quality, raw = cell.value, cell.quality, cell.raw_value
        elif property_id in {"Skin", "Carcinogen"}:
            raw = record.raw(spec.field_id)
            if raw is not None and raw != "":
                if property_id == "Skin":
                    if type(raw) is bool:
                        value, quality = raw, "valid"
                    else:
                        quality = "invalid"
                else:
                    codes = (
                        IARC_CLASSIFICATIONS if spec.source_id == "IARC" else NTP_CLASSIFICATIONS
                    )
                    try:
                        value = codes[int(cast(str | SupportsInt, raw))]
                    except (ValueError, TypeError, KeyError):
                        quality = "invalid"
                    else:
                        quality = "valid"
                    for field_id in ("year", "volumes", "description", "Chemical"):
                        if field_id in record.field_names:
                            conditions[field_id] = record.raw(field_id)
        elif selected_unit is None:
            raw = record.raw(spec.field_id)
            if raw is not None and raw != "":
                value, quality = str(raw), "valid"
        else:
            cell = record.value(spec.field_id)
            value, quality, raw = cell.value, cell.quality, cell.raw_value
        if property_id == "ODP":
            version = conditions["version"]
            conditions["source_interval"] = {
                key: record.raw(f"ODP{version}{suffix}")
                for key, suffix in (("text", ""), ("minimum", " Min"), ("maximum", " Max"))
            }
        warnings: tuple[str, ...] = (
            "Historical source snapshot; no assertion of current regulatory limits",
        )
        if (property_id in {"LFL", "UFL"} and value is not None
                and not 0 < cast(float, value) <= 1):
            warnings += (
                "Reported flammability limit is outside 0 < fraction <= 1; unusable in mixing",
            )
        return AssessmentValue(
            property_id,
            method,
            record.cas_number or record.source_key,
            record.record_id,
            value,
            selected_unit,
            quality,
            record.source.provenance,
            raw,
            conditions,
            warnings,
        )

    def assessment_methods(self, identity: object, property_id: str) -> tuple[str, ...]:
        specs = self.method_specs(property_id)
        if property_id == "Carcinogen":
            return tuple(m.method for m in specs)
        return tuple(
            m.method
            for m in specs
            if any(
                self.assessment(r.record_id, property_id, method=m.method).quality == "valid"
                for r in self.records_for(identity, method=m.source_id)
            )
        )

    def exposure_source_records(self, identity: object) -> tuple[ConstantRecord, ...]:
        """All exact source-label matches; never merge several CAS/agent rows."""
        record = self.record(identity, method="ONTARIO")
        name = record.raw("Name")
        return tuple(r for r in self.records("ONTARIO_RAW") if r.raw("Agent [CAS No.]") == name)

    def flammability_methods(
        self, identity: object, *, limit: str, heat_of_combustion_J_mol: float | None = None,
        atoms: Mapping[str, float] | None = None,
    ) -> tuple[str, ...]:
        """Source methods plus estimates whose required inputs are supplied.

        Availability is not validation of an estimate's physical applicability.
        Choosing a method remains explicit and validates its full input domain.
        """
        if limit not in {"LFL", "UFL"}:
            raise ValueError("select LFL or UFL")
        from chemworld.physchem.safety_environment import ESTIMATION_METHODS

        methods = self.assessment_methods(identity, limit)
        if heat_of_combustion_J_mol is not None:
            methods += (ESTIMATION_METHODS[0],)
        if atoms is not None:
            methods += (ESTIMATION_METHODS[1],)
        return methods

    def flammability_value(
        self, identity: object, *, limit: str, method: str,
        heat_of_combustion_J_mol: float | None = None, atoms: Mapping[str, float] | None = None,
    ) -> PropertyEvaluation:
        """Selected data or estimate -> the original property report, no fallback."""
        if limit not in {"LFL", "UFL"}:
            raise ValueError("select LFL or UFL")
        from chemworld.physchem.safety_environment import ESTIMATION_METHODS, flammability_estimate

        if method not in ESTIMATION_METHODS:
            return self.assessment(identity, limit, method=method).evaluate()
        value = flammability_estimate(
            limit=limit,
            method=method,
            heat_of_combustion_J_mol=heat_of_combustion_J_mol,
            atoms=atoms,
        )
        inputs = (
            {"combustion_enthalpy_J_mol": cast(float, heat_of_combustion_J_mol)}
            if method == ESTIMATION_METHODS[0]
            else {"atoms:" + k: v for k, v in cast(Mapping[str, float], atoms).items()}
        )
        return PropertyEvaluation(
            limit,
            str(identity),
            method,
            value,
            "dimensionless",
            inputs,
            ("Empirical estimate from caller inputs; not a measured source limit",),
        )
