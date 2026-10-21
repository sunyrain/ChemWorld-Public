"""Four complete external conductivity tables on the original lazy source API."""

from dataclasses import dataclass, fields
from math import isfinite

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import ConstantDatabase, ConstantRecord
from chemworld.physchem.specs import PropertyCorrelation

CONDUCTIVITY_SOURCES = tuple(
    ConstantSource(
        source, "Thermal Conductivity/" + filename, citation=citation + "; chemicals 1.5.2"
    )
    for source, filename, citation in (
        (
            "PERRY_GAS_CONDUCTIVITY",
            "Table 2-314 Vapor Thermal Conductivity of Inorganic and Organic Substances.tsv",
            "Perry, 8th ed., Table 2-314; DIPPR102, W/(m K)",
        ),
        (
            "PERRY_LIQUID_CONDUCTIVITY",
            "Table 2-315 Thermal Conductivity of Inorganic and Organic Liquids.tsv",
            "Perry, 8th ed., Table 2-315; DIPPR100, W/(m K)",
        ),
        (
            "VDI_GAS_CONDUCTIVITY",
            "VDI PPDS Thermal conductivity of gases.tsv",
            "VDI Heat Atlas, 2010; fourth-order T polynomial, W/(m K)",
        ),
        (
            "VDI_LIQUID_CONDUCTIVITY",
            "VDI PPDS Thermal conductivity of saturated liquids.tsv",
            "VDI Heat Atlas, 2010; fourth-order T polynomial, W/(m K)",
        ),
    )
)
_SOURCES = {s.source_id: s for s in CONDUCTIVITY_SOURCES}


@dataclass(frozen=True)
class ConductivityRecord(ConstantRecord):
    @property
    def source(self):
        return _SOURCES[self.source_id]

    def descriptor(self, field_id):
        self.raw(field_id)
        return ConstantField(
            field_id,
            "K" if field_id in {"Tmin", "Tmax"} else None,
            "Raw conductivity fit field; units follow the complete equation",
            "22E3",
        )

    def correlation(self, *, temperature_bounds_K=None):
        def number(key):
            value = self.value(key)
            if value.quality != "valid" or not isfinite(value.value):
                raise ValueError(f"missing or invalid conductivity field {key}")
            return value.value

        source_bounds = (number("Tmin"), number("Tmax")) if "Tmin" in self.field_names else None
        bounds = source_bounds if temperature_bounds_K is None else tuple(temperature_bounds_K)
        if (
            bounds is None
            or len(bounds) != 2
            or any(
                isinstance(t, bool) or not isinstance(t, int | float) or not isfinite(t)
                for t in bounds
            )
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError("select an explicit finite positive temperature interval")
        if source_bounds and (bounds[0] < source_bounds[0] or bounds[1] > source_bounds[1]):
            raise ValueError("application interval exceeds source temperature bounds")
        gas = self.source_id in {"PERRY_GAS_CONDUCTIVITY", "VDI_GAS_CONDUCTIVITY"}
        if self.source_id.startswith("PERRY"):
            names = "ABCD" if gas else "ABCDE"
            coefficients = {key: number(f"C{i}") for i, key in enumerate(names, 1)}
        else:
            coefficients = {key: number(key) for key in "ABCDE"}
        equation = "dippr102" if self.source_id == "PERRY_GAS_CONDUCTIVITY" else "dippr100"
        return PropertyCorrelation(
            f"{self.record_id}:conductivity",
            equation,
            coefficients,
            {"temperature": "K"},
            "W/(m*K)",
            "thermal_conductivity",
            {"temperature": bounds},
            self.source.provenance.citation,
            {
                "source": self.source.provenance.to_dict(),
                "source_record": {
                    "record_id": self.record_id,
                    "source_key": self.source_key,
                    "fields": dict(zip(self.field_names, self.raw_fields, strict=True)),
                },
                "phase": "gas" if gas else "liquid",
                "source_bounds_K": None if source_bounds is None else list(source_bounds),
                "temperature_range_basis": "caller_application"
                if source_bounds is None
                else "source",
            },
        )


class ConductivityDatabase(ConstantDatabase):
    @property
    def catalog(self):
        return CONDUCTIVITY_SOURCES

    def source(self, source_id):
        return _SOURCES[source_id] if source_id in _SOURCES else super().source(source_id)

    def _read_records(self, source, path):
        records = super()._read_records(source, path)
        if source.source_id not in _SOURCES:
            return records
        return tuple(
            ConductivityRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
            for r in records
        )

    def thermal_conductivity(self, identity, *, method, temperature_bounds_K=None):
        return self.record(identity, method=method).correlation(
            temperature_bounds_K=temperature_bounds_K
        )

    def property_input(self, identity, component, *, method, temperature_bounds_K=None):
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        record = self.record(identity, method=method)
        if (record.cas_number or record.source_key) not in {
            component.cas_number,
            component.identifier,
        }:
            raise ValueError("conductivity source identity does not match ComponentSpec")
        curve = record.correlation(temperature_bounds_K=temperature_bounds_K)
        source = record.source.provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)),
            (InputSource(source, "literature"),),
            {
                curve.correlation_id: ComponentUncertainty(
                    curve.correlation_id,
                    unit="W/(m*K)",
                    source_id=source.source_id,
                    coverage="unspecified",
                    note="Source fit; uncertainty not quantified",
                )
            },
            None,
            "Conductivity only; no caloric reference inferred",
        )
