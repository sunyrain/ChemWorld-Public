"""Lazy caloric data reading on the existing source/record/query contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from chemworld.data.caloric_catalog import (
    CALORIC_FIELDS,
    CALORIC_SOURCE_BY_ID,
    CALORIC_SOURCES,
    CaloricSource,
)
from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import (
    ConstantDatabase,
    ConstantRecord,
    ConstantValue,
    NonScalarConstantError,
    _cas,
)

if TYPE_CHECKING:
    from chemworld.data.caloric_curves import HeatCapacityData
    from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity
    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.property_reports import CaloricReference, PhaseLabel
    from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation


@dataclass(frozen=True)
class CaloricRecord(ConstantRecord):
    @property
    def source(self) -> CaloricSource:
        return CALORIC_SOURCE_BY_ID[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        if field_id in CALORIC_FIELDS:
            return CALORIC_FIELDS[field_id]
        if self.source_id in {"ATCT_GAS", "ATCT_LIQUID"} and field_id == "uncertainty":
            return ConstantField(
                field_id,
                "J/mol",
                "ATcT source uncertainty magnitude; statistical coverage not inferred",
                "22C2",
            )
        if self.source_id == "CRC_SOLID_CP" and field_id.isdigit():
            return ConstantField(
                field_id,
                "J/(mol*K)",
                f"Solid isobaric capacity at {field_id} K; blank values remain missing",
                "22C2",
            )
        return ConstantField(
            field_id,
            None,
            "Original caloric source metadata or formula parameter; select its full curve contract",
            "22C2",
        )

    def value(self, field_id: str, *, point: int | None = None) -> ConstantValue:
        if isinstance(self.raw(field_id), (list, dict)):
            raise NonScalarConstantError("nested caloric data require a curve or table selection")
        return super().value(field_id, point=point)


class CaloricDatabase(ConstantDatabase):
    """All 16 Heat Capacity files and five Reactions tables, explicitly selected.

    B3 constant data keep their existing catalog and provenance. This adapter
    shares parsing, identity indexing, conflicts and quantity/input conversions;
    it never imports chemicals or discovers an implicit installation directory.
    """

    def __init__(self, source_directory: str | PathLike[str]) -> None:
        super().__init__(source_directory)
        self.constants = ConstantDatabase(source_directory)

    @property
    def catalog(self) -> tuple[CaloricSource, ...]:
        return CALORIC_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        if source_id in CALORIC_SOURCE_BY_ID:
            return CALORIC_SOURCE_BY_ID[source_id]
        return self.constants.source(source_id)

    @property
    def loaded_sources(self) -> tuple[str, ...]:
        return (*super().loaded_sources, *self.constants.loaded_sources)

    def records(self, source_id: str) -> tuple[ConstantRecord, ...]:
        if source_id not in CALORIC_SOURCE_BY_ID:
            return self.constants.records(source_id)
        return super().records(source_id)

    def records_for(self, identity: object, *, method: str) -> tuple[ConstantRecord, ...]:
        if method not in CALORIC_SOURCE_BY_ID:
            return self.constants.records_for(identity, method=method)
        return super().records_for(identity, method=method)

    def heat_capacities(self, identity: object, *, method: str) -> tuple[HeatCapacityData, ...]:
        """All ranges for an identity, including incomplete source specifications."""
        from chemworld.data.caloric_curves import source_heat_capacities

        return tuple(
            curve
            for record in self.records_for(identity, method=method)
            for curve in source_heat_capacities(record)
        )

    def heat_capacity(
        self,
        identity: object,
        *,
        method: str,
        phase: str,
        equation_id: str | None = None,
        subphase: str | None = None,
        condition: str | None = None,
        parameters: Mapping[str, float] | None = None,
        temperature_bounds_K: Sequence[float] | None = None,
        interpolation: str | None = None,
    ) -> PropertyCorrelation | PiecewiseHeatCapacity:
        """Build one selected same-phase domain without implicit ranking/overlap repair."""
        from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity

        candidates = tuple(
            curve
            for curve in self.heat_capacities(identity, method=method)
            if curve.phase == phase
            and (equation_id is None or curve.equation_id == equation_id)
            and (subphase is None or curve.subphase == subphase)
            and (condition is None or curve.condition == condition)
        )
        if not candidates:
            raise ValueError("no heat-capacity range for the selected identity/phase/form")
        if len({(c.subphase, c.condition) for c in candidates}) != 1:
            raise ValueError("select a source condition and subphase; no implicit phase transition")
        curves: list[PropertyCorrelation] = []
        for candidate in candidates:
            bounds = temperature_bounds_K
            if bounds is not None and candidate.source_bounds_K is not None:
                low, high = candidate.source_bounds_K
                bounds = (
                    max(bounds[0], low) if low is not None else bounds[0],
                    min(bounds[1], high) if high is not None else bounds[1],
                )
                if bounds[0] >= bounds[1]:
                    continue
            curve = candidate.build(
                parameters=parameters, temperature_bounds_K=bounds, interpolation=interpolation
            )
            curves.extend(curve.segments if isinstance(curve, PiecewiseHeatCapacity) else (curve,))
        if not curves:
            raise ValueError("selected source has no range within the requested temperatures")
        curves.sort(key=lambda c: c.validity_ranges["temperature"][0])
        actual = (
            curves[0].validity_ranges["temperature"][0],
            curves[-1].validity_ranges["temperature"][1],
        )
        if temperature_bounds_K is not None and tuple(temperature_bounds_K) != actual:
            raise ValueError("requested temperatures exceed the selected source coverage")
        return (
            curves[0]
            if len(curves) == 1
            else PiecewiseHeatCapacity(
                f"{method}:{identity}:{phase}",
                tuple(curves),
                self.source(method).provenance.citation,
            )
        )

    def formation_reference(
        self, identity: object, *, phase: str, enthalpy_method: str, pressure_Pa: float,
        entropy_method: str | None = None,
    ) -> CaloricReference:
        """298.15 K Hf and optional ABSOLUTE entropy, with explicit source and pressure.

        Zero-K fields remain queryable data, but cannot masquerade as a positive-T
        caloric datum. Sources lacking the selected phase are never silently reused.
        """
        from chemworld.physchem.property_reports import CaloricReference

        suffix = {"gas": "g", "liquid": "l", "solid": "s"}.get(phase)
        if suffix is None:
            raise ValueError("formation reference requires gas, liquid or solid phase")
        enthalpy = self.value(identity, "Hf" + suffix, method=enthalpy_method)
        entropy = (
            None
            if entropy_method is None
            else self.value(identity, "S0" + suffix, method=entropy_method)
        )
        sources = [enthalpy.record.source.provenance.citation + "/" + enthalpy.field_id]
        if entropy is not None:
            sources.append(entropy.record.source.provenance.citation + "/" + entropy.field_id)
        return CaloricReference(
            phase=cast("PhaseLabel", phase),
            temperature_K=298.15,
            pressure_Pa=pressure_Pa,
            energy_J_mol=enthalpy.quantity("J/mol").value,
            entropy_J_mol_K=None if entropy is None else entropy.quantity("J/(mol*K)").value,
            source_note="; ".join(sources) + "; reference pressure explicitly selected by caller",
        )

    def property_input(
        self, identity: object, component: ComponentSpec, *, method: str, phase: str,
        reference: CaloricReference | None, reference_note: str, **curve_options: Any,
    ) -> PropertyInputCard:
        """Carry the selected source through the existing scientific input envelope."""
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        curve = self.heat_capacity(identity, method=method, phase=phase, **curve_options)
        source = self.source(method)
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)),
            (InputSource(source.provenance, "literature"),),
            {
                curve.correlation_id: ComponentUncertainty(
                    field_id=curve.correlation_id,
                    unit=curve.output_unit,
                    coverage="not_quantified",
                    source_id=source.provenance.source_id,
                    note=source.nature
                    + "; fit/estimate does not establish measurement uncertainty",
                )
            },
            reference,
            reference_note,
        )

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[CaloricRecord, ...]:
        if source.kind != "caloric-json":
            return tuple(
                CaloricRecord(**{f.name: getattr(record, f.name) for f in fields(ConstantRecord)})
                for record in self._text_records(source, path)
            )
        with path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("caloric JSON requires a mapping of original identities")
        return tuple(
            CaloricRecord(
                source.source_id,
                number,
                identity,
                _cas(identity),
                ("payload",),
                ("payload",),
                (payload,),
            )
            for number, (identity, payload) in enumerate(data.items(), 1)
        )
