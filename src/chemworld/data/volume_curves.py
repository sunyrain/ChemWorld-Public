"""Lossless source candidates mapped onto native volume and DIPPR equations."""

from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite
from types import MappingProxyType
from xml.etree import ElementTree

from chemworld.data.caloric_curves import number
from chemworld.data.constants import ConstantRecord, SourceCorrelation
from chemworld.physchem.specs import PropertyCorrelation


@dataclass(frozen=True)
class VolumeData:
    record: ConstantRecord
    curve_key: str
    equation_id: str
    coefficients: dict = field(default_factory=dict)
    source_bounds_K: tuple = (None, None)
    property_id: str = "liquid_molar_volume"
    output_unit: str = "m^3/mol"
    required_inputs: tuple = ()
    temperatures_K: tuple = ()
    values: tuple = ()
    notes: tuple = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", MappingProxyType(dict(self.coefficients)))

    @property
    def curve_id(self):
        return f"{self.record.record_id}:{self.curve_key}"

    def _bounds(self, selected):
        bounds = self.source_bounds_K if selected is None else tuple(selected)
        if (
            len(bounds) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v)
                for v in bounds
            )
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError("select an explicit finite positive temperature interval")
        low, high = self.source_bounds_K
        if (low is not None and bounds[0] < low) or (high is not None and bounds[1] > high):
            raise ValueError("application interval exceeds source temperature bounds")
        return tuple(bounds)

    def _curve(self, equation, coefficients, bounds, *, key=None, basis="source"):
        phase = self.property_id.split("_")[0]
        return PropertyCorrelation(
            self.curve_id if key is None else self.curve_id + ":" + str(key),
            equation,
            coefficients,
            {"temperature": "K"},
            self.output_unit,
            self.property_id,
            {"temperature": bounds},
            self.record.source.provenance.citation,
            {
                "source": self.record.source.provenance.to_dict(),
                "source_record": self.record.record_id,
                "source_curve": self.curve_key,
                "source_bounds_K": list(self.source_bounds_K),
                "source_notes": list(self.notes),
                "temperature_range_basis": basis,
                "phase": phase if phase in {"liquid", "solid", "gas"} else "gas",
                "physical_relation": "source_temperature_path; no independent pressure model",
            },
        )

    def build(
        self,
        *,
        temperature_bounds_K=None,
        parameters=None,
        application_temperature_K=None,
        application_note=None,
        interpolation=None,
    ):
        parameters = {} if parameters is None else dict(parameters)
        if set(parameters) != set(self.required_inputs):
            raise ValueError(f"supply exactly the explicit inputs {self.required_inputs}")
        if any(v is None or not isfinite(v) for v in self.coefficients.values()):
            raise ValueError("missing or invalid source volume coefficients")
        if self.equation_id == "unlocated_point":
            if (
                temperature_bounds_K is not None
                or interpolation is not None
                or isinstance(application_temperature_K, bool)
                or not isinstance(application_temperature_K, int | float)
                or not isfinite(application_temperature_K)
                or application_temperature_K <= 0
                or not isinstance(application_note, str)
                or not application_note.strip()
            ):
                raise ValueError(
                    "unlocated observation requires an explicit application temperature and note"
                )
            curve = self._curve(
                "molar_volume_source_point",
                {"value": self.coefficients["Vm"], "T_ref": application_temperature_K},
                (application_temperature_K, application_temperature_K),
                basis="caller_point_assumption",
            )
            curve.metadata["application_note"] = application_note
            return curve
        if application_temperature_K is not None or application_note is not None:
            raise ValueError("point assumptions apply only to unlocated constant observations")
        bounds = self._bounds(temperature_bounds_K)
        if self.equation_id == "tabulated":
            segments = self.build_segments(temperature_bounds_K=bounds, interpolation=interpolation)
            if len(segments) != 1:
                raise ValueError("select one table interval or build_segments for the full table")
            return segments[0]
        if interpolation is not None:
            raise ValueError("analytic source curves do not require interpolation")
        if self.equation_id.startswith("unresolved_"):
            raise ValueError("unresolved source volume equation: " + self.equation_id)
        coeffs = {**self.coefficients, **parameters}
        if self.equation_id.startswith("volume:"):
            from chemworld.physchem.volume_correlations import volume_correlation

            curve = volume_correlation(
                self.equation_id.split(":")[1],
                parameters=coeffs,
                temperature_range_K=bounds,
                source_note=self.record.source.provenance.citation,
                output="density" if self.output_unit == "kg/m^3" else "molar_volume",
                correlation_id=self.curve_id,
            )
            curve.metadata.update(self._curve(curve.equation_id, coeffs, bounds).metadata)
        else:
            curve = self._curve(self.equation_id, coeffs, bounds)
        curve.metadata["temperature_range_basis"] = (
            "source" if temperature_bounds_K is None else "caller_application"
        )
        curve.metadata["explicit_supplemental_parameters"] = parameters
        return curve

    def build_segments(self, *, interpolation, temperature_bounds_K=None):
        if self.equation_id != "tabulated" or interpolation != "linear":
            raise ValueError("volume table requires explicit linear interpolation")
        bounds = self._bounds(temperature_bounds_K)
        ts, values = self.temperatures_K, self.values
        if (
            len(ts) != len(values)
            or len(ts) < 2
            or any(v is None for v in ts)
            or any(b <= a for a, b in pairwise(ts))
        ):
            raise ValueError(
                "table requires aligned increasing temperature points; no sorting/filling"
            )
        if bounds[0] < ts[0] or bounds[1] > ts[-1]:
            raise ValueError("cannot extrapolate a volume table")
        result = []
        for i, (low, high) in enumerate(pairwise(ts)):
            interval = max(low, bounds[0]), min(high, bounds[1])
            if interval[0] >= interval[1]:
                continue
            v0, v1 = values[i : i + 2]
            if any(v is None or not isfinite(v) or v <= 0 for v in (v0, v1)):
                raise ValueError(
                    "selected interval crosses a missing/nonpositive volume observation"
                )
            slope = (v1 - v0) / (high - low)
            curve = self._curve(
                "dippr100",
                {"A": v0 - slope * low, "B": slope},
                interval,
                key=i,
                basis="explicit_table_interpolation",
            )
            curve.metadata.update(
                interpolation=interpolation,
                source_point_indices=[i, i + 1],
                derivative_domain="segment interior; no derivative across a knot",
            )
            result.append(curve)
        return tuple(result)


def source_volumes(record):
    def raw(key):
        return record.raw(key) if key in record.field_names else None

    def n(key):
        return number(raw(key))

    def curve(key, equation, coefficients, bounds=(None, None), **kwargs):
        return VolumeData(record, key, equation, coefficients, bounds, **kwargs)

    source = record.source_id
    if source == "COSTALD_PARAMETERS":
        return (
            curve(
                "COSTALD",
                "volume:COSTALD",
                {"omega": n("omega_SRK"), "Vc": n("Vchar")},
                required_inputs=("Tc",),
            ),
            curve("Rackett", "volume:Rackett", {"Zc": n("Z_RA")}, required_inputs=("Tc", "Pc")),
        )
    if source == "MCHAWEH_SN0":
        return (
            curve(
                "SNM0",
                "volume:SNM0",
                {"delta_SRK": n("delta_SRK")},
                required_inputs=("Tc", "Vc", "omega"),
            ),
        )
    if source == "CRC_MOLTEN":
        return (
            curve(
                "molten",
                "volume:CRC_inorganic",
                {"rho0": n("rho"), "k": n("k"), "Tm": n("Tm")},
                (n("Tm"), n("Tmax")),
                property_id="liquid_density",
                output_unit="kg/m^3",
            ),
        )
    if source in {"CRC_LIQUID_VOLUME", "CRC_SOLID_VOLUME"}:
        phase = "solid" if source == "CRC_SOLID_VOLUME" else "liquid"
        return (
            curve(
                "constant",
                "unlocated_point",
                {"Vm": n("Vm")},
                property_id=phase + "_molar_volume",
                notes=(
                    "Source supplies no observation temperature; caller point is an assumption.",
                ),
            ),
        )
    if source == "CRC_VIRIAL":
        return (
            curve(
                "B",
                "crc_second_virial",
                {f"a{i}": n(f"a{i}") for i in range(1, 6)},
                property_id="second_virial_coefficient",
            ),
        )
    if source == "PERRY_VOLUME":
        return (
            curve(
                "density",
                "dippr105_reciprocal",
                {k: n(f"C{i}") for i, k in enumerate("ABCD", 1)},
                (n("Tmin"), n("Tmax")),
            ),
        )
    if source == "VDI_PPDS_VOLUME":
        return (
            curve(
                "density",
                "volume:volume_VDI_PPDS",
                {"Tc": n("Tc"), "rhoc": n("rhoc"), **{k.lower(): n(k) for k in "ABCD"}},
                (None, n("Tc")),
                property_id="liquid_density",
                output_unit="kg/m^3",
            ),
        )
    if source == "CHEMSEP_XML":
        result = []
        for key, phase in (
            ("LiquidDensity", "liquid"),
            ("SolidDensity", "solid"),
            ("SecondVirialCoefficient", "gas"),
        ):
            item = raw(key)
            if item is None:
                continue
            expected_unit = "m3/kmol" if phase == "gas" else "kmol/m3"
            if not isinstance(item, SourceCorrelation) or item.source_unit != expected_unit:
                raise ValueError("ChemSep source quantity must have explicit molar units")
            node = ElementTree.fromstring(item.xml)
            values = {c.tag: c.attrib.get("value") for c in node}
            if any(c.attrib.get("units") != "K" for c in node if c.tag in {"Tmin", "Tmax"}):
                raise ValueError("ChemSep source temperatures require K")
            eq = values.get("eqno")
            keys = {"2": "AB", "100": "ABCDE", "104": "ABCDE", "105": "ABCD", "106": "ABCDE"}.get(
                eq, "ABCDE"
            )
            params = {k: number(values.get(k)) for k in keys}
            if phase == "gas":
                equation = "dippr104" if eq == "104" else "unresolved_chemsep_" + str(eq)
                params = {k: None if v is None else v / 1000 for k, v in params.items()}
                prop = "second_virial_coefficient"
            else:
                equation = {
                    "2": "dippr100_reciprocal",
                    "100": "dippr100_reciprocal",
                    "105": "dippr105_reciprocal",
                    "106": "dippr106_reciprocal",
                }.get(eq, "unresolved_chemsep_" + str(eq))
                # Density coefficients in kmol/m3 -> mol/m3 before taking reciprocal.
                for k in keys if eq in {"2", "100"} else "A":
                    coefficient = params[k]
                    if coefficient is not None:
                        params[k] = coefficient * 1000
                if eq == "106":
                    params["Tc"] = n("CriticalTemperature")
                prop = phase + "_molar_volume"
            result.append(
                curve(
                    key,
                    equation,
                    params,
                    tuple(number(values.get(k)) for k in ("Tmin", "Tmax")),
                    property_id=prop,
                    notes=(
                        f"ChemSep equation {eq}; PCDman Help 2024 Table 1; original XML retained",
                    ),
                )
            )
        return tuple(result)
    if source == "VDI":
        ts = tuple(number(v) for v in raw("T"))
        return tuple(
            curve(
                key,
                "tabulated",
                {},
                (ts[0], ts[-1]),
                property_id=phase + "_" + quantity,
                output_unit=unit,
                temperatures_K=ts,
                values=tuple(number(v) for v in raw(key)),
            )
            for phase, letter in (("liquid", "l"), ("gas", "g"))
            for quantity, title, unit in (
                ("molar_volume", "Volume", "m^3/mol"),
                ("density", "Density", "kg/m^3"),
            )
            for key in (f"{title} ({letter})",)
        )
    return ()
