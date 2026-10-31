"""Lazy external latent-heat observations, curves and transition temperatures."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chemworld.data.constant_catalog import TABULAR_FIELDS, ConstantField, ConstantSource
from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConstantDatabase,
    ConstantIdentityNotFoundError,
    ConstantRecord,
)
from chemworld.data.phase_change_catalog import PHASE_CHANGE_SOURCE_BY_ID, PHASE_CHANGE_SOURCES

if TYPE_CHECKING:
    from chemworld.data.phase_change_curves import LatentHeatData
    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


@dataclass(frozen=True)
class PhaseChangeRecord(ConstantRecord):
    @property
    def source(self) -> ConstantSource:
        return PHASE_CHANGE_SOURCE_BY_ID[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        if field_id in TABULAR_FIELDS:
            return TABULAR_FIELDS[field_id]
        unit = (
            "J/mol"
            if field_id in {"Hvap298", "HvapTb", "error"}
            else "K"
            if field_id in {"Tmin", "Tmax"}
            else "dimensionless"
            if self.source_id == "BELL_TWU" and field_id in {"c0", "c1", "c2"}
            else "J/(mol*K)"
            if self.source_id == "ALIBAKHSHI" and field_id == "C"
            else None
        )
        return ConstantField(
            field_id,
            unit,
            "Source phase-change field; error is reported magnitude, not assumed one sigma",
            "22C4",
        )


class PhaseChangeDatabase(ConstantDatabase):
    """One inherited source cache/index; no implicit upstream package or ranking.

    Bell's table is Twu EOS alpha parameters (DOI 10.1021/acs.jced.7b00967),
    not latent heat. Its full source rows and dimensionless parameters remain
    queryable by InChIKey; the EOS consumer belongs to task 23.
    """

    @property
    def catalog(self) -> tuple[ConstantSource, ...]:
        return PHASE_CHANGE_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        if source_id in PHASE_CHANGE_SOURCE_BY_ID:
            return PHASE_CHANGE_SOURCE_BY_ID[source_id]
        return super().source(source_id)

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[ConstantRecord, ...]:
        records = super()._read_records(source, path)
        if source.source_id not in PHASE_CHANGE_SOURCE_BY_ID:
            return records
        result = []
        for record in records:
            values = {f.name: getattr(record, f.name) for f in fields(ConstantRecord)}
            if source.source_id == "BELL_TWU":
                # The first column is Tc, not an identity or a CAS integer.
                values.update(
                    source_key=str(record.raw_fields[record.field_names.index("inchikey")]),
                    cas_number=None,
                )
            result.append(PhaseChangeRecord(**values))
        return tuple(result)

    def latent_heats(self, identity: object, *, method: str) -> tuple[LatentHeatData, ...]:
        from chemworld.data.phase_change_curves import source_latent_heats

        return tuple(
            curve
            for record in self.records_for(identity, method=method)
            for curve in source_latent_heats(record)
        )

    def latent_heat(
        self,
        identity: object,
        *,
        method: str,
        curve_id: str | None = None,
        **build_options: Any,
    ) -> PropertyCorrelation:
        candidates = self.latent_heats(identity, method=method)
        if curve_id is not None:
            candidates = tuple(c for c in candidates if c.curve_id == curve_id)
        if not candidates:
            raise ConstantIdentityNotFoundError(f"no latent heat for {method}/{identity}")
        if len(candidates) != 1:
            raise AmbiguousConstantIdentityError(
                "select one latent heat curve_id; no implicit ranking"
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
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        curve = self.latent_heat(identity, method=method, curve_id=curve_id, **build_options)
        source = self.source(method).provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)),
            (InputSource(source, "literature"),),
            {
                curve.correlation_id: ComponentUncertainty(
                    curve.correlation_id,
                    unit="J/mol",
                    source_id=source.source_id,
                    coverage="unspecified",
                    note="Source transition heat; uncertainty not quantified",
                )
            },
            None,
            "Latent heat only; no absolute H, U or S datum is inferred",
        )
