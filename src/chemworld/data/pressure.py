"""Lazy pressure data on the existing external source and identity contracts."""

from dataclasses import dataclass, fields

from chemworld.data.constant_catalog import ConstantField
from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConstantDatabase,
    ConstantIdentityNotFoundError,
    ConstantRecord,
)
from chemworld.data.pressure_catalog import PRESSURE_SOURCE_BY_ID, PRESSURE_SOURCES


@dataclass(frozen=True)
class PressureRecord(ConstantRecord):
    @property
    def source(self):
        return PRESSURE_SOURCE_BY_ID[self.source_id]

    def descriptor(self, field_id):
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
    def catalog(self):
        return PRESSURE_SOURCES

    def source(self, source_id):
        if source_id in PRESSURE_SOURCE_BY_ID:
            return PRESSURE_SOURCE_BY_ID[source_id]
        return super().source(source_id)

    def _read_records(self, source, path):
        records = super()._read_records(source, path)
        if source.source_id not in PRESSURE_SOURCE_BY_ID:
            return records
        return tuple(
            PressureRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def pressures(self, identity, *, method):
        from chemworld.data.pressure_curves import source_pressures

        return tuple(
            curve
            for record in self.records_for(identity, method=method)
            for curve in source_pressures(record)
        )

    def pressure(self, identity, *, method, curve_id=None, **build_options):
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

    def property_input(self, identity, component, *, method, curve_id=None, **build_options):
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
