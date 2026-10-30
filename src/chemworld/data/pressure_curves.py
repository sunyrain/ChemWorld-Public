"""Source pressure coefficients and explicit table intervals become native curves."""

from dataclasses import dataclass, field
from itertools import pairwise
from math import e, isfinite, log
from types import MappingProxyType
from xml.etree import ElementTree

from chemworld.data.caloric_curves import number
from chemworld.data.constants import ConstantRecord, SourceCorrelation
from chemworld.physchem.specs import PropertyCorrelation


@dataclass(frozen=True)
class PressureData:
    record: ConstantRecord
    curve_key: str
    equation_id: str
    coefficients: dict[str, float | None] = field(default_factory=dict)
    source_bounds_K: tuple[float | None, float | None] | None = None
    property_id: str = "vapor_pressure"
    temperatures_K: tuple[float | None, ...] = ()
    pressures_Pa: tuple[float | None, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.property_id not in {"vapor_pressure", "sublimation_pressure"}:
            raise ValueError("pressure source requires vapor or sublimation identity")
        object.__setattr__(self, "coefficients", MappingProxyType(dict(self.coefficients)))

    @property
    def curve_id(self):
        return f"{self.record.record_id}:{self.curve_key}"

    def _bounds(self, selected):
        bounds = self.source_bounds_K if selected is None else tuple(selected)
        if (
            bounds is None
            or len(bounds) != 2
            or any(not isinstance(v, int | float) or number(v) is None for v in bounds)
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError("select an explicit finite positive source temperature interval")
        if self.source_bounds_K is not None:
            low, high = self.source_bounds_K
            if (low is not None and bounds[0] < low) or (high is not None and bounds[1] > high):
                raise ValueError("requested range extrapolates beyond source bounds")
        return bounds

    def build(self, *, temperature_bounds_K=None, interpolation=None):
        """Build analytic data, or one explicitly selected table interval.

        For a full table use build_segments; retaining individual source
        intervals makes derivative jumps and missing observations explicit.
        """
        bounds = self._bounds(temperature_bounds_K)
        if self.equation_id == "tabulated":
            segments = self.build_segments(temperature_bounds_K=bounds, interpolation=interpolation)
            if len(segments) != 1:
                raise ValueError(
                    "select one table interval or use build_segments for all intervals"
                )
            return segments[0]
        if interpolation is not None:
            raise ValueError("analytic pressure data do not require interpolation")
        if self.equation_id.startswith("unresolved_"):
            raise ValueError(f"source equation definition is unresolved: {self.equation_id}")
        if any(v is None for v in self.coefficients.values()):
            raise ValueError("missing or invalid pressure source coefficients")
        return self._correlation(self.curve_key, self.equation_id, self.coefficients, bounds)

    def _correlation(self, key, equation, coefficients, bounds):
        return PropertyCorrelation(
            f"{self.record.record_id}:{key}",
            equation,
            dict(coefficients),
            {"temperature": "K"},
            "Pa",
            self.property_id,
            {"temperature": tuple(bounds)},
            self.record.source.provenance.citation + f"; record={self.record.record_id}",
            {
                "source": self.record.source.provenance.to_dict(),
                "source_record": self.record.record_id,
                "source_curve": self.curve_key,
                "source_bounds_K": self.source_bounds_K,
                "source_notes": list(self.notes),
                "temperature_range_basis": "explicit-source-interval",
                "derivative_domain": (
                    "formula-specific; fractional critical derivatives may not exist"
                ),
            },
        )

    def build_segments(self, *, interpolation, temperature_bounds_K=None):
        """All source intervals in a range, linear in ln(P) versus inverse T.

        Every segment is the existing native Arrhenius formula. No smoothing,
        filling, ranking, extrapolation or implicit derivative across a knot.
        """
        if self.equation_id != "tabulated" or interpolation != "log_pressure_inverse_temperature":
            raise ValueError("select tabulated log_pressure_inverse_temperature interpolation")
        bounds = self._bounds(temperature_bounds_K)
        ts, ps = self.temperatures_K, self.pressures_Pa
        if len(ts) != len(ps) or len(ts) < 2 or any(t is None for t in ts):
            raise ValueError("pressure table needs aligned finite temperature points")
        if any(b <= a for a, b in pairwise(ts)):
            raise ValueError("table temperatures must increase; no sorting or merging")
        if bounds[0] < ts[0] or bounds[1] > ts[-1]:
            raise ValueError("cannot extrapolate a pressure table")
        segments = []
        for i, (low, high) in enumerate(pairwise(ts)):
            selected = max(bounds[0], low), min(bounds[1], high)
            if selected[0] >= selected[1]:
                continue
            p0, p1 = ps[i : i + 2]
            if any(p is None or not isfinite(p) or p <= 0 for p in (p0, p1)):
                raise ValueError("selected interval crosses missing or nonpositive pressure")
            slope = (log(p1) - log(p0)) / (1 / high - 1 / low)
            curve = self._correlation(
                f"{self.curve_key}:{i}",
                "arrhenius_pressure",
                {"T_ref": low, "P_ref": p0, "slope": slope},
                selected,
            )
            curve.metadata.update(interpolation=interpolation, source_point_indices=[i, i + 1])
            segments.append(curve)
        return tuple(segments)


def source_pressures(record):
    """Preserve every pressure range, including missing bounds and unknown XML forms."""
    source = record.source_id

    def raw(key):
        return record.raw(key) if key in record.field_names else None

    def n(key):
        return number(raw(key))

    def curve(equation, coefficients, bounds=None, key="pressure", **kwargs):
        return PressureData(
            record,
            key,
            equation,
            coefficients,
            (n("Tmin"), n("Tmax")) if bounds is None else bounds,
            **kwargs,
        )

    prop = "sublimation_pressure" if "SUBLIMATION" in source else "vapor_pressure"
    if source in {"ANTOINE_POLING", "LANDOLT_PRESSURE", "LANDOLT_SUBLIMATION"}:
        return (
            curve(
                "antoine",
                {**{k: n(k) for k in "ABC"}, "base": 10.0 if source == "ANTOINE_POLING" else e},
                property_id=prop,
            ),
        )
    if source in {"ALCOCK_PRESSURE", "ALCOCK_SUBLIMATION", "PERRY_PRESSURE"}:
        values = {
            key: n(f"C{i}" if source == "PERRY_PRESSURE" else key)
            for i, key in enumerate("ABCDE", 1)
        }
        return (curve("dippr101", values, property_id=prop),)
    if source == "TRC_ANTOINE_POLING":
        return (
            curve(
                "trc_antoine_extended",
                {k: n(k) for k in ("A", "B", "C", "Tc", "to", "n", "E", "F")},
            ),
        )
    if source in {"WAGNER_POLING", "WAGNER_MCGARRY", "VDI_PPDS_PRESSURE"}:
        values = {**{k.lower(): n(k) for k in "ABCD"}, "Tc": n("Tc"), "Pc": n("Pc")}
        bounds = (
            (n("Tm"), n("Tc"))
            if source == "VDI_PPDS_PRESSURE"
            else (n("Tmin"), n("Tc"))
            if source == "WAGNER_MCGARRY"
            else (n("Tmin"), n("Tmax"))
        )
        equation = "wagner" if source == "WAGNER_MCGARRY" else "wagner25"
        return (
            curve(
                equation,
                values,
                bounds,
                notes=("Critical endpoint retained; legacy Wagner 3,6 evaluation requires T<Tc.",),
            ),
        )
    if source == "WEBBOOK":
        fields = tuple("Antoine" + k for k in ("A", "B", "C", "Tmin", "Tmax"))
        if not any(raw(k) not in (None, "") for k in fields):
            return ()
        return (
            curve(
                "antoine",
                {**{k: n("Antoine" + k) for k in "ABC"}, "base": e},
                (n("AntoineTmin"), n("AntoineTmax")),
                key="Antoine",
            ),
        )
    if source == "CHEMSEP_XML":
        result = []
        for key in ("VaporPressure", "AntoineVaporPressure"):
            item = raw(key)
            if item is None:
                continue
            if not isinstance(item, SourceCorrelation) or item.source_unit != "Pa":
                raise ValueError("ChemSep pressure requires an explicit Pa correlation")
            node = ElementTree.fromstring(item.xml)
            values = {child.tag: child.attrib.get("value") for child in node}
            if any(
                child.attrib.get("units") != "K" for child in node if child.tag in {"Tmin", "Tmax"}
            ):
                raise ValueError("ChemSep pressure temperatures require K")
            eq = values.get("eqno")
            coefficients = {k: number(values.get(k)) for k in ("ABC" if eq == "10" else "ABCDE")}
            equation = {"10": "antoine", "101": "dippr101", "200": "wagner25"}.get(
                eq, "unresolved_chemsep_" + str(eq)
            )
            notes = [f"ChemSep equation {eq}; original XML retained in source record"]
            if eq == "10":
                coefficients["base"] = e
            if eq == "200":
                # PCDmanager's official tutorial calls this the Ambrose-Walton
                # four-coefficient form with Pc as the fifth parameter. Its
                # reduced-temperature basis is the same as Wagner 2.5,5.
                # Use E from this fit, not a rounded or cross-source Pc value.
                Tc = record.value("CriticalTemperature")
                coefficients = {
                    **{k.lower(): number(values.get(k)) for k in "ABCD"},
                    "Tc": Tc.quantity("K").value if Tc.quality == "valid" else None,
                    "Pc": number(values.get("E")),
                }
                notes += [
                    "ChemSep Lite 8.50 PCDmanager help, p.5: equation 200, Ambrose-Walton",
                    "Interpretation: tau powers 1,1.5,2.5,5; Tc=CriticalTemperature; Pc=E",
                    "http://www.chemsep.org/downloads/data/lite.zip (help/pcdman-help.pdf)",
                ]
            result.append(
                curve(
                    equation,
                    coefficients,
                    tuple(number(values.get(k)) for k in ("Tmin", "Tmax")),
                    key=key,
                    notes=tuple(notes),
                )
            )
        return tuple(result)
    if source == "VDI":
        ts, ps = tuple(number(v) for v in raw("T")), tuple(number(v) for v in raw("P"))
        return (curve("tabulated", {}, (ts[0], ts[-1]), temperatures_K=ts, pressures_Pa=ps),)
    return ()
