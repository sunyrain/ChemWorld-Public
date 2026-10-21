"""Eight complete viscosity sources through the existing lazy database contract.

The external source files are not copied into the distribution. Source fields
remain unchanged; only construction of the native correlation converts units.
"""

from dataclasses import dataclass, fields
from math import isfinite, log

from chemworld.data.constant_catalog import ConstantField, ConstantSource
from chemworld.data.constants import ConstantDatabase, ConstantRecord
from chemworld.physchem.specs import PropertyCorrelation

VISCOSITY_SOURCES = tuple(ConstantSource(source, "Viscosity/" + filename,
                                        citation=citation + "; chemicals 1.5.2")
                          for source, filename, citation in (
    ("DUTT_PRASAD_VISCOSITY", "Dutt Prasad 3 term.tsv", "Dutt and Prasad, liquid viscosity"),
    ("PERRY_GAS_VISCOSITY", "Table 2-312 Vapor Viscosity of Inorganic and Organic Substances.tsv",
     "Perry, 8th ed., Table 2-312; DIPPR102, Pa s"),
    ("PERRY_LIQUID_VISCOSITY", "Table 2-313 Viscosity of Inorganic and Organic Liquids.tsv",
     "Perry, 8th ed., Table 2-313; DIPPR101, Pa s"),
    ("VDI_GAS_VISCOSITY", "VDI PPDS Dynamic viscosity of gases polynomials.tsv",
     "VDI Heat Atlas, 2010; polynomial, Pa s"),
    ("VDI_LIQUID_VISCOSITY", "VDI PPDS Dynamic viscosity of saturated liquids polynomials.tsv",
     "VDI Heat Atlas, 2010; PPDS9, Pa s"),
    ("VN2E_VISCOSITY", "Viswanath Natarajan Dynamic 2 term Exponential.tsv",
     "Viswanath and Natarajan, 1989; power law, Pa s"),
    ("VN2_VISCOSITY", "Viswanath Natarajan Dynamic 2 term.tsv",
     "Viswanath and Natarajan, 1989; exponential, source scale 0.01 Pa s"),
    ("VN3_VISCOSITY", "Viswanath Natarajan Dynamic 3 term.tsv",
     "Viswanath and Natarajan, 1989; base-10, cP"),
))
_SOURCES = {source.source_id: source for source in VISCOSITY_SOURCES}


@dataclass(frozen=True)
class ViscosityRecord(ConstantRecord):
    @property
    def source(self):
        return _SOURCES[self.source_id]

    def descriptor(self, field_id):
        self.raw(field_id)
        return ConstantField(field_id, "K" if field_id in {"Tmin", "Tmax"} else None,
                             "Raw viscosity fit field; units follow the complete equation",
                             "22E2")

    def correlation(self, *, temperature_bounds_K=None):
        """Explicit native curve; VDI tables require a caller-selected interval.

        No source temperature interval is invented. A supplied application
        interval may narrow a published range, but cannot widen it. VN2's
        scale follows its public numerical example (A-ln(100)); the docs'
        conflicting prose saying 1e-3 Pa s is not used as a conversion.
        """
        def number(key):
            value = self.value(key)
            if value.quality != "valid" or not isfinite(value.value):
                raise ValueError(f"missing or invalid viscosity source field {key}")
            return value.value

        source_bounds = (number("Tmin"), number("Tmax")) if "Tmin" in self.field_names else None
        bounds = source_bounds if temperature_bounds_K is None else tuple(temperature_bounds_K)
        if (bounds is None or len(bounds) != 2
                or any(isinstance(t, bool) or not isinstance(t, int | float)
                       or not isfinite(t) for t in bounds) or not 0 < bounds[0] < bounds[1]):
            raise ValueError("select an explicit finite positive temperature interval")
        if source_bounds and (bounds[0] < source_bounds[0] or bounds[1] > source_bounds[1]):
            raise ValueError("application interval exceeds source temperature bounds")
        source = self.source_id
        phase = "gas" if source in {"PERRY_GAS_VISCOSITY", "VDI_GAS_VISCOSITY"} else "liquid"
        if source.startswith("PERRY_"):
            names = "ABCD" if phase == "gas" else "ABCDE"
            coefficients = {key: number(f"C{i}") for i, key in enumerate(names, 1)}
            equation = "dippr102" if phase == "gas" else "dippr101"
        elif source == "VDI_GAS_VISCOSITY":
            coefficients = {key: number(key) for key in "ABCDE"}
            equation = "dippr100"
        elif source == "VDI_LIQUID_VISCOSITY":
            coefficients = {key: number(key) for key in "ABCDE"}
            equation = "viscosity_ppds9"
        elif source == "VN2E_VISCOSITY":
            coefficients = {key: number(key) for key in "CD"}
            equation = "viscosity_viswanath_natarajan_2_exponential"
        elif source == "VN2_VISCOSITY":
            coefficients = {"A": number("A") - log(100), "B": number("B")}
            equation = "viscosity_viswanath_natarajan_2"
        else:
            coefficients = {"A": number("A") - 3, "B": number("B"), "C": number("C")}
            equation = "viscosity_viswanath_natarajan_3"
        return PropertyCorrelation(
            f"{self.record_id}:viscosity", equation, coefficients, {"temperature": "K"},
            "Pa*s", phase + "_viscosity", {"temperature": bounds},
            self.source.provenance.citation,
            {"source": self.source.provenance.to_dict(),
             "source_record": {"record_id": self.record_id, "source_key": self.source_key,
                               "fields": dict(zip(self.field_names, self.raw_fields, strict=True))},
             "source_bounds_K": list(source_bounds) if source_bounds else None,
             "temperature_range_basis": "source" if source_bounds else "caller_application",
             "coefficient_conversion": ("A-ln(100)" if source == "VN2_VISCOSITY" else
                 "A-3; exponent A+B/(C-T)" if source in {"VN3_VISCOSITY", "DUTT_PRASAD_VISCOSITY"}
                 else "source already yields Pa*s")},
        )


class ViscosityDatabase(ConstantDatabase):
    @property
    def catalog(self):
        return VISCOSITY_SOURCES

    def source(self, source_id):
        return _SOURCES[source_id] if source_id in _SOURCES else super().source(source_id)

    def _read_records(self, source, path):
        records = super()._read_records(source, path)
        if source.source_id not in _SOURCES:
            return records
        return tuple(ViscosityRecord(**{f.name: getattr(r, f.name) for f in fields(ConstantRecord)})
                     for r in records)

    def viscosity(self, identity, *, method, temperature_bounds_K=None):
        return self.record(identity, method=method).correlation(
            temperature_bounds_K=temperature_bounds_K)

    def property_input(self, identity, component, *, method, temperature_bounds_K=None):
        from chemworld.data.scientific_inputs import InputSource
        from chemworld.physchem.property_inputs import PropertyInputCard
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.specs import ComponentUncertainty

        record = self.record(identity, method=method)
        if (record.cas_number or record.source_key) not in {
                component.cas_number, component.identifier}:
            raise ValueError("viscosity source identity does not match ComponentSpec")
        curve = record.correlation(temperature_bounds_K=temperature_bounds_K)
        source = record.source.provenance
        return PropertyInputCard(
            ComponentPropertyPackage(component, (curve,)), (InputSource(source, "literature"),),
            {curve.correlation_id: ComponentUncertainty(
                curve.correlation_id, unit="Pa*s", source_id=source.source_id,
                coverage="unspecified", note="Source fit; uncertainty not quantified",
            )}, None, "Viscosity only; no caloric reference inferred",
        )
