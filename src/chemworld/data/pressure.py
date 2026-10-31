"""Lazy pressure data on the existing external source and identity contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConstantDatabase,
    ConstantIdentityNotFoundError,
    ConstantRecord,
)
from chemworld.data.pressure_catalog import PRESSURE_SOURCE_BY_ID, PRESSURE_SOURCES

if TYPE_CHECKING:
    from chemworld.data.pressure_curves import PressureData
    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


@dataclass(frozen=True)
class PressureRecord(ConstantRecord):
    @property
    def source(self) -> ConstantSource:
        return PRESSURE_SOURCE_BY_ID[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        unit = (
            "K" if field_id in {"Tc", "Tm", "Tmin", "Tmax"} else "Pa" if field_id == "Pc" else None
        )
        return ConstantField(
            field_id,
            unit,
            "Pressure source field; coefficients require the complete formula contract",
            "22C3",
        )


class PressureDatabase(ConstantDatabase):
    """Ten pressure tables plus explicitly selected B3 sources, with one cache.

    No implicit installation discovery, method ranking, identity merging or
    runtime reference-library import. Missing intervals remain source data.
    """

    @property
    def catalog(self) -> tuple[ConstantSource, ...]:
        return PRESSURE_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        if source_id in PRESSURE_SOURCE_BY_ID:
            return PRESSURE_SOURCE_BY_ID[source_id]
        return super().source(source_id)

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[ConstantRecord, ...]:
        records = super()._read_records(source, path)
        if source.source_id not in PRESSURE_SOURCE_BY_ID:
            return records
        return tuple(
            PressureRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def pressures(self, identity: object, *, method: str) -> tuple[PressureData, ...]:
        from chemworld.data.pressure_curves import source_pressures

        return tuple(
            curve
            for record in self.records_for(identity, method=method)
            for curve in source_pressures(record)
        )

    def pressure(
        self,
        identity: object,
        *,
        method: str,
        curve_id: str | None = None,
        **build_options: Any,
    ) -> PropertyCorrelation:
        """Build one explicit source curve; multiple ranges require curve identity."""
        candidates = self.pressures(identity, method=method)
        if curve_id is not None:
            candidates = tuple(c for c in candidates if c.curve_id == curve_id)
        if not candidates:
            raise ConstantIdentityNotFoundError(f"no pressure curve for {method}/{identity}")
        if len(candidates) != 1:
            raise AmbiguousConstantIdentityError(
                "select one pressure curve_id; no implicit ranking"
            )
        return candidates[0].build(**build_options)

    def property_input(
        self,
        identity: object,
        component: ComponentSpec,
        *,
        method: str,
        curve_id: str | None = None,
        **build_options: Any,
    ) -> PropertyInputCard:
        """Import a selected pressure curve using the established input envelope."""
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        curve = self.pressure(identity, method=method, curve_id=curve_id, **build_options)
        source = self.source(method).provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)),
            (InputSource(source, "literature"),),
            {
                curve.correlation_id: ComponentUncertainty(
                    curve.correlation_id,
                    unit="Pa",
                    source_id=source.source_id,
                    coverage="unspecified",
                    note="Source pressure fit; uncertainty not quantified",
                )
            },
            None,
            "Pressure correlation only; no caloric reference is inferred",
        )
