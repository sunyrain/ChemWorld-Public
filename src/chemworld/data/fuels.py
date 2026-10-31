"""Nine complete external fuel tables on the existing lazy constant database."""

from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import cast

from chemworld.data.assessments import AssessmentValue
from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import (
    ConstantDatabase,
    ConstantIdentityNotFoundError,
    ConstantRecord,
    UnknownConstantMethodError,
)
from chemworld.foundation.units import Quantity
from chemworld.physchem.combustion import ignition_delay_quality, octane_index
from chemworld.physchem.property_reports import PropertyEvaluation

FUEL_SOURCES = tuple(
    ConstantSource(identity, "Misc/" + name)
    for identity, name in (
        ("DAHMEN_MARQUARDT", "Dahmen_Marquardt_ignition_delay_IQT.tsv"),
        ("FLORIAN_CETANE", "Florian_Liming_CETANE_experimental.tsv"),
        ("FLORIAN_MON", "Florian_Liming_MON_experimental.tsv"),
        ("FLORIAN_RON", "Florian_Liming_RON_experimental.tsv"),
        ("FLORIAN_LIMING_ANN", "Florian_Liming_RON_MON_ANN.tsv"),
        ("COMBUSTDB_CETANE", "Travis_Kessler_Combustdb_CETANE.tsv"),
        ("COMBUSTDB_MON", "Travis_Kessler_Combustdb_MON.tsv"),
        ("COMBUSTDB_PREDICTIONS", "Travis_Kessler_Combustdb_predictions.tsv"),
        ("COMBUSTDB_RON", "Travis_Kessler_Combustdb_RON.tsv"),
    )
)
_SOURCES = MappingProxyType({s.source_id: s for s in FUEL_SOURCES})
FUEL_METHODS = MappingProxyType(
    {
        prop: MappingProxyType(methods)
        for prop, methods in {
            "RON": {
                "FLORIAN_LIMING": "FLORIAN_RON",
                "COMBUSTDB": "COMBUSTDB_RON",
                "FLORIAN_LIMING_ANN": "FLORIAN_LIMING_ANN",
                "COMBUSTDB_PREDICTIONS": "COMBUSTDB_PREDICTIONS",
            },
            "MON": {
                "FLORIAN_LIMING": "FLORIAN_MON",
                "COMBUSTDB": "COMBUSTDB_MON",
                "FLORIAN_LIMING_ANN": "FLORIAN_LIMING_ANN",
                "COMBUSTDB_PREDICTIONS": "COMBUSTDB_PREDICTIONS",
            },
            "CETANE": {
                "FLORIAN_LIMING": "FLORIAN_CETANE",
                "COMBUSTDB": "COMBUSTDB_CETANE",
                "COMBUSTDB_PREDICTIONS": "COMBUSTDB_PREDICTIONS",
            },
            "ignition_delay": {"DAHMEN_MARQUARDT": "DAHMEN_MARQUARDT"},
            "POUR_POINT": {"COMBUSTDB_PREDICTIONS": "COMBUSTDB_PREDICTIONS"},
            "CLOUD_POINT": {"COMBUSTDB_PREDICTIONS": "COMBUSTDB_PREDICTIONS"},
        }.items()
    }
)
_UNITS = {
    "RON": "dimensionless",
    "MON": "dimensionless",
    "CETANE": "dimensionless",
    "IGNITION_DELAY": "s",
    "POUR_POINT": "degC",
    "CLOUD_POINT": "degC",
}


@dataclass(frozen=True)
class FuelRecord(ConstantRecord):
    @property
    def source(self) -> ConstantSource:
        return _SOURCES[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        return ConstantField(
            field_id,
            _UNITS.get(field_id),
            "Original selected fuel source; unverified units remain unavailable",
            "22F4",
        )


class FuelDatabase(ConstantDatabase):
    @property
    def catalog(self) -> tuple[ConstantSource, ...]:
        return FUEL_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        return _SOURCES.get(source_id) or super().source(source_id)

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[ConstantRecord, ...]:
        records = super()._read_records(source, path)
        if source.source_id not in _SOURCES:
            return records
        return tuple(
            FuelRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def fuel_value(self, identity: object, property_id: str, *, method: str) -> AssessmentValue:
        try:
            source_id = FUEL_METHODS[property_id][method]
        except KeyError as exc:
            raise UnknownConstantMethodError(
                f"unknown fuel property/method: {property_id}/{method}"
            ) from exc
        field_id = "IGNITION_DELAY" if property_id == "ignition_delay" else property_id
        unit = _UNITS[field_id]
        prediction = source_id in {"COMBUSTDB_PREDICTIONS", "FLORIAN_LIMING_ANN"}
        conditions = {
            "data_kind": "prediction" if prediction else "compiled experimental",
            "edition": "chemicals 1.5.2 external source snapshot",
        }
        if property_id == "ignition_delay":
            conditions["measurement"] = "IQT device; different instruments can differ"
        if property_id in {"POUR_POINT", "CLOUD_POINT"}:
            conditions["unit_reference"] = (
                "ECNet bundled property dataset cards: cloud/pour point targets in degC; "
                "https://github.com/ecrl/ecnet/blob/master/docs/source/data.rst"
            )
        try:
            record = self.record(identity, method=source_id)
        except ConstantIdentityNotFoundError:
            return AssessmentValue(
                property_id,
                method,
                str(identity),
                f"{source_id}:unlisted:{identity}",
                None,
                unit,
                "not_listed",
                self.source(source_id).provenance,
                None,
                conditions,
                ("Not listed in this source",),
            )
        cell = record.value(field_id)
        value, quality, raw = cell.value, cell.quality, cell.raw_value
        return AssessmentValue(
            property_id,
            method,
            record.cas_number or record.source_key,
            record.record_id,
            value,
            unit,
            quality,
            record.source.provenance,
            raw,
            conditions,
            (
                "Predicted source property; not an experimental observation"
                if prediction
                else "Compiled source value; test conditions and uncertainty not fully quantified",
            ),
        )

    def fuel_methods(self, identity: object, property_id: str) -> tuple[str, ...]:
        if property_id not in FUEL_METHODS:
            raise UnknownConstantMethodError(f"unknown fuel property {property_id}")
        return tuple(
            method
            for method, source in FUEL_METHODS[property_id].items()
            if any(
                self.fuel_value(r.record_id, property_id, method=method).quality == "valid"
                for r in self.records_for(identity, method=source)
            )
        )


def source_octane_index(
    ron: AssessmentValue, mon: AssessmentValue, *, quantity: str = "AKI",
) -> PropertyEvaluation:
    if (
        ron.property_id != "RON"
        or mon.property_id != "MON"
        or ron.component_id != mon.component_id
        or ron.quality != "valid"
        or mon.quality != "valid"
        or ron.method != mon.method
    ):
        raise ValueError(
            "RON/MON must be valid values of the same identity and selected source method"
        )
    value = octane_index(ron.input_value().value, mon.input_value().value, quantity=quantity)
    return PropertyEvaluation(
        quantity,
        ron.record_id + "/" + mon.record_id,
        ron.method,
        value,
        "dimensionless",
        {"RON": cast(float, ron.value), "MON": cast(float, mon.value)},
        tuple(dict.fromkeys((*ron.warnings, *mon.warnings))),
    )


def source_ignition_quality(delay: AssessmentValue, *, quantity: str) -> PropertyEvaluation:
    if delay.property_id != "ignition_delay" or delay.quality != "valid":
        raise ValueError("a valid source ignition delay is required")
    seconds = cast(float, delay.input_value().to("s").value)
    value = ignition_delay_quality(seconds, quantity=quantity)
    warnings = (*delay.warnings, "Empirical derived index; not measured octane/cetane number")
    if quantity in {"RON", "MON"} and not 0.0036 <= seconds <= 0.067:
        warnings += ("Outside the approximate Perez-Boehman fitted delay range",)
    return PropertyEvaluation(
        quantity,
        delay.record_id,
        "ignition_delay_correlation",
        value,
        "dimensionless",
        {"ignition_delay_s": seconds},
        warnings,
    )


def fuel_quality_comparison(
    value: AssessmentValue, criterion: Quantity, *, relation: str,
) -> dict[str, object]:
    """One source property versus an explicit engineering threshold, no auto-approval.

    Returned ratio is dimensionless for quality indices; temperatures use a
    Boolean decision only so absolute and difference temperature units cannot
    be confused. The original value/threshold and both units remain in output.
    """
    if value.quality != "valid":
        raise ValueError("fuel comparison requires a valid numeric source")
    if relation not in {"minimum", "maximum"}:
        raise ValueError("select minimum or maximum criterion")
    cell = value.input_value()
    threshold = criterion.to(cell.unit).value
    numeric_value = cast(float, cell.value)
    passes = numeric_value >= threshold if relation == "minimum" else numeric_value <= threshold
    return {
        "property_id": value.property_id,
        "source": value.to_dict(),
        "criterion": {"value": criterion.value, "unit": criterion.unit, "relation": relation},
        "source_unit_threshold": threshold,
        "meets_criterion": passes,
        "interpretation": "Caller-selected criterion; not an engine or blending prediction",
    }
