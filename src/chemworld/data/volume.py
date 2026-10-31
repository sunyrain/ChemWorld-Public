"""External density tables on the existing lazy identity/source database.

No automatic data discovery, reference-library import, or implicit ranking.
Caller supplied critical inputs are explicit; source coefficients cannot be
overridden. CRC constant observations require a stated application temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import (
    AmbiguousConstantIdentityError,
    ConstantDatabase,
    ConstantIdentityNotFoundError,
    ConstantRecord,
)

if TYPE_CHECKING:
    from chemworld.data.volume_curves import VolumeData
    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation

VOLUME_SOURCES = tuple(
    ConstantSource(key, "Density/" + filename, citation=citation + "; chemicals 1.5.2")
    for key, filename, citation in (
        ("COSTALD_PARAMETERS", "COSTALD Parameters.tsv", "Hankinson and Thomson, COSTALD"),
        (
            "CRC_MOLTEN",
            "CRC Inorganics densties of molten compounds and salts.tsv",
            "CRC molten inorganic densities",
        ),
        (
            "CRC_LIQUID_VOLUME",
            "CRC Liquid Inorganic Constant Densities.tsv",
            "CRC liquid inorganic constant volumes",
        ),
        (
            "CRC_SOLID_VOLUME",
            "CRC Solid Inorganic Constant Densities.tsv",
            "CRC solid inorganic constant volumes",
        ),
        ("CRC_VIRIAL", "CRC Virial polynomials.tsv", "CRC second virial polynomials"),
        ("MCHAWEH_SN0", "Mchaweh SN0 deltas.tsv", "Mchaweh et al., modified SNM0"),
        ("PERRY_VOLUME", "Perry Parameters 105.tsv", "Perry liquid molar density, equation 105"),
        (
            "VDI_PPDS_VOLUME",
            "VDI PPDS Density of Saturated Liquids.tsv",
            "VDI PPDS saturated liquid densities",
        ),
    )
)
VOLUME_SOURCE_BY_ID = {s.source_id: s for s in VOLUME_SOURCES}


@dataclass(frozen=True)
class VolumeRecord(ConstantRecord):
    @property
    def source(self) -> ConstantSource:
        return VOLUME_SOURCE_BY_ID[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        units = {
            "Tc": "K",
            "Tm": "K",
            "Tmin": "K",
            "Tmax": "K",
            "Vm": "m^3/mol",
            "Vchar": "m^3/mol",
            "rho": "kg/m^3",
            "rhoc": "kg/m^3",
            "MW": "g/mol",
            "omega_SRK": "dimensionless",
            "Z_RA": "dimensionless",
            "delta_SRK": "dimensionless",
        }
        return ConstantField(
            field_id,
            units.get(field_id),
            "Source field; fit coefficients require the complete equation",
            "22D2C",
        )


class VolumeDatabase(ConstantDatabase):
    @property
    def catalog(self) -> tuple[ConstantSource, ...]:
        return VOLUME_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        if source_id in VOLUME_SOURCE_BY_ID:
            return VOLUME_SOURCE_BY_ID[source_id]
        return super().source(source_id)

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[ConstantRecord, ...]:
        records = super()._read_records(source, path)
        if source.source_id not in VOLUME_SOURCE_BY_ID:
            return records
        return tuple(
            VolumeRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def volumes(self, identity: object, *, method: str) -> tuple[VolumeData, ...]:
        from chemworld.data.volume_curves import source_volumes

        return tuple(
            c for r in self.records_for(identity, method=method) for c in source_volumes(r)
        )

    def _select(self, identity: object, method: str, curve_id: str | None) -> VolumeData:
        candidates = self.volumes(identity, method=method)
        if curve_id is not None:
            candidates = tuple(c for c in candidates if c.curve_id == curve_id)
        if not candidates:
            raise ConstantIdentityNotFoundError(f"no volume curve for {method}/{identity}")
        if len(candidates) != 1:
            raise AmbiguousConstantIdentityError("select one volume curve_id; no implicit ranking")
        return candidates[0]

    def volume(
        self, identity: object, *, method: str, curve_id: str | None = None, **build_options: Any,
    ) -> PropertyCorrelation:
        return self._select(identity, method, curve_id).build(**build_options)

    def property_input(
        self, identity: object, component: ComponentSpec, *, method: str,
        curve_id: str | None = None, **build_options: Any,
    ) -> PropertyInputCard:
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        selected = self._select(identity, method, curve_id)
        record = selected.record
        if (record.cas_number or record.source_key) not in {
            component.cas_number,
            component.identifier,
        }:
            raise ValueError("volume source identity does not match ComponentSpec")
        if "MW" in record.field_names and record.value("MW").quality == "valid":
            mw = cast(float, record.value("MW").value)
            # Source tables often publish rounded MW. Never silently replace it.
            if abs(mw - cast(float, component.molecular_weight_g_mol)) > 1e-9 * mw:
                raise ValueError("source MW must match the component molecular weight")
        # A sourced fit may require explicitly selected critical estimates.
        # Keep them as parameter provenance, without replacing table coefficients.
        from chemworld.data.scientific_inputs import InputValue
        from chemworld.physchem.critical_properties import CriticalEstimate, _input

        from .constants import ConstantValue

        supplemental = build_options.get("parameters", {})
        origins, converted = {}, {}
        for key, value in supplemental.items():
            if isinstance(value, (CriticalEstimate, ConstantValue, InputValue)):
                converted[key] = _input(value, key, component.identifier, component.cas_number)
                origins[key] = value.to_dict()
            else:
                converted[key] = value
        if supplemental:
            build_options = {**build_options, "parameters": converted}
        if selected.equation_id == "tabulated":
            curves = selected.build_segments(**build_options)
        else:
            curves = (selected.build(**build_options),)
        if origins:
            curves = tuple(replace(c, metadata={**c.metadata, "parameter_sources": origins})
                           for c in curves)
        source = record.source.provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, curves),
            (InputSource(source, "literature"),),
            {
                c.correlation_id: ComponentUncertainty(
                    c.correlation_id,
                    unit=c.output_unit,
                    source_id=source.source_id,
                    coverage="unspecified",
                    note="Source fit/observation; uncertainty not quantified",
                )
                for c in curves
            },
            None,
            "Source volume relation only; no caloric datum or independent pressure EOS inferred",
        )
