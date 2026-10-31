"""Five external surface-tension tables using the existing source/input API."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from math import isfinite, log
from pathlib import Path
from typing import TYPE_CHECKING, cast

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import ConstantDatabase, ConstantRecord
from chemworld.physchem.specs import PropertyCorrelation

if TYPE_CHECKING:
    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.specs import ComponentSpec

INTERFACE_SOURCES = tuple(
    ConstantSource(source, "Interface/" + filename, citation=citation + "; chemicals 1.5.2")
    for source, filename, citation in (
        ("JASPER_LANGE", "Jasper-Lange.tsv", "Jasper, 1972; Lange handbook; mN/m linear fit"),
        (
            "MULERO_CACHADINA",
            "MuleroCachadinaParameters.tsv",
            "Mulero et al., 2012; REFPROP sum of powers",
        ),
        ("SOMAYAJULU", "Somayajulu.tsv", "Somayajulu, 1988; original coefficients"),
        ("SOMAYAJULU_REVISED", "SomayajuluRevised.tsv", "Somayajulu revised coefficients"),
        (
            "VDI_SURFACE_TENSION",
            "VDI PPDS surface tensions.tsv",
            "VDI Heat Atlas, 2010; DIPPR106 form",
        ),
    )
)
_SOURCES = {s.source_id: s for s in INTERFACE_SOURCES}


@dataclass(frozen=True)
class InterfaceRecord(ConstantRecord):
    @property
    def source(self) -> ConstantSource:
        return _SOURCES[self.source_id]

    def descriptor(self, field_id: str) -> ConstantField:
        self.raw(field_id)
        return ConstantField(
            field_id,
            "K" if field_id in {"Tmin", "Tmax", "Tt", "Tm", "Tc"} else None,
            "Raw interface fit field; units follow the complete equation",
            "22E4",
        )

    def correlation(
        self, *, temperature_bounds_K: Sequence[float] | None = None, source_point: bool = False,
    ) -> PropertyCorrelation:
        def number(key: str) -> float:
            result = self.value(key)
            if result.quality != "valid" or result.value is None or not isfinite(result.value):
                raise ValueError(f"missing or invalid interface field {key}")
            return result.value

        source = self.source_id

        def bound(key: str) -> float | None:
            result = self.value(key)
            return (result.value if result.quality == "valid" and result.value is not None
                    and isfinite(result.value) else None)

        if source == "JASPER_LANGE":
            method, p = "Jasper", {k: number(k) for k in ("a", "b")}
            source_bounds = (bound("Tmin"), bound("Tmax"))
        elif source == "MULERO_CACHADINA":
            method = "REFPROP_sigma"
            p = {k: number(k) for k in ("Tc", "sigma0", "n0", "sigma1", "n1", "sigma2", "n2")}
            source_bounds = (bound("Tmin"), bound("Tmax"))
        elif source.startswith("SOMAYAJULU"):
            method, p = "Somayajulu", {k: number(k) for k in ("Tc", "A", "B", "C")}
            source_bounds = (number("Tt"), p["Tc"])
        else:
            # A*tau**(B+C*Tr+D*Tr²+E*Tr³) is the Watson log-amplitude form.
            a = number("A")
            if a <= 0:
                raise ValueError("VDI amplitude must be positive for the logarithmic fit")
            method = "Watson_sigma"
            p = {
                "Tc": number("Tc"),
                "a1": log(a),
                **{f"a{i}": number(k) for i, k in enumerate("BCDE", 2)},
            }
            source_bounds = (number("Tm"), p["Tc"])
        bounds = source_bounds if temperature_bounds_K is None else tuple(temperature_bounds_K)
        if source_point:
            if (
                temperature_bounds_K is not None
                or source_bounds[0] is None
                or source_bounds[0] != source_bounds[1]
                or source_bounds[0] <= 0
            ):
                raise ValueError("source_point requires one positive source temperature")
            from chemworld.physchem.interface_kernels import interface_value

            p = {
                "T_ref": source_bounds[0],
                "sigma_ref": interface_value(method, source_bounds[0], **p),
            }
            method = "source_point"
        if (
            len(bounds) != 2
            or any(
                isinstance(t, bool) or not isinstance(t, int | float) or not isfinite(t)
                for t in bounds
            )
            or not (0 < cast(float, bounds[0]) < cast(float, bounds[1]) or source_point)
        ):
            raise ValueError("select an explicit finite positive temperature interval")
        bounds = cast(tuple[float, float], bounds)
        # A zero lower bound in a source means no positive lower bound is supplied.
        # Preserve it in metadata, but require an explicit application interval.
        if (source_bounds[0] is not None and bounds[0] < source_bounds[0]) or (
            source_bounds[1] is not None and bounds[1] > source_bounds[1]
        ):
            raise ValueError("application interval exceeds source temperature bounds")
        return PropertyCorrelation(
            f"{self.record_id}:surface_tension",
            "interface_" + method.lower(),
            p,
            {"temperature": "K"},
            "N/m",
            "surface_tension",
            {"temperature": bounds},
            self.source.provenance.citation,
            {
                "source": self.source.provenance.to_dict(),
                "phase": "liquid",
                "interface": "liquid-gas",
                "source_bounds_K": list(source_bounds),
                "temperature_range_basis": "source_point"
                if source_point
                else "source"
                if temperature_bounds_K is None
                else "caller_application",
                "source_record": {
                    "record_id": self.record_id,
                    "source_key": self.source_key,
                    "fields": dict(zip(self.field_names, self.raw_fields, strict=True)),
                },
            },
        )


class InterfaceDatabase(ConstantDatabase):
    @property
    def catalog(self) -> tuple[ConstantSource, ...]:
        return INTERFACE_SOURCES

    def source(self, source_id: str) -> ConstantSource:
        return _SOURCES[source_id] if source_id in _SOURCES else super().source(source_id)

    def _read_records(self, source: ConstantSource, path: Path) -> tuple[ConstantRecord, ...]:
        records = super()._read_records(source, path)
        if source.source_id not in _SOURCES:
            return records
        return tuple(
            InterfaceRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def surface_tension(
        self, identity: object, *, method: str,
        temperature_bounds_K: Sequence[float] | None = None, source_point: bool = False,
    ) -> PropertyCorrelation:
        record = self.record(identity, method=method)
        if not isinstance(record, InterfaceRecord):
            raise ValueError("select a interface correlation source")
        return record.correlation(
            temperature_bounds_K=temperature_bounds_K, source_point=source_point
        )

    def property_input(
        self, identity: object, component: ComponentSpec, *, method: str,
        temperature_bounds_K: Sequence[float] | None = None, source_point: bool = False,
    ) -> PropertyInputCard:
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        record = self.record(identity, method=method)
        if (record.cas_number or record.source_key) not in {
            component.cas_number,
            component.identifier,
        }:
            raise ValueError("interface source identity does not match ComponentSpec")
        if not isinstance(record, InterfaceRecord):
            raise ValueError("select a interface correlation source")
        curve = record.correlation(
            temperature_bounds_K=temperature_bounds_K, source_point=source_point
        )
        source = record.source.provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)),
            (InputSource(source, "literature"),),
            {
                curve.correlation_id: ComponentUncertainty(
                    curve.correlation_id,
                    unit="N/m",
                    source_id=source.source_id,
                    coverage="unspecified",
                    note="Source fit; uncertainty not quantified",
                )
            },
            None,
            "Liquid-gas interface only; no liquid-liquid or caloric model inferred",
        )
