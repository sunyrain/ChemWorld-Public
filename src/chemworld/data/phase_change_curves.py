"""Preserve phase-change source observations and explicitly construct native models."""

from dataclasses import dataclass, field
from itertools import pairwise
from types import MappingProxyType
from xml.etree import ElementTree

from chemworld.data.caloric_curves import number
from chemworld.data.constant_catalog import XML_UNIT_NAMES
from chemworld.data.constants import ConstantRecord, SourceCorrelation
from chemworld.foundation.units import convert_value
from chemworld.physchem.specs import PropertyCorrelation


@dataclass(frozen=True)
class LatentHeatData:
    record: ConstantRecord
    curve_key: str
    property_id: str
    equation_id: str
    coefficients: dict[str, float | None] = field(default_factory=dict)
    source_bounds_K: tuple[float | None, float | None] | None = None
    required_parameters: tuple[str, ...] = ()
    temperatures_K: tuple[float | None, ...] = ()
    enthalpies_J_mol: tuple[float | None, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.property_id not in {
            "heat_of_vaporization",
            "heat_of_fusion",
            "heat_of_sublimation",
        }:
            raise ValueError("latent-heat data require an explicit transition identity")
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
            raise ValueError("select an explicit finite positive temperature interval")
        if self.source_bounds_K is not None:
            low, high = self.source_bounds_K
            if (low is not None and bounds[0] < low) or (high is not None and bounds[1] > high):
                raise ValueError("requested range extrapolates beyond source bounds")
        return bounds

    def build(self, *, parameters=None, temperature_bounds_K=None, interpolation=None):
        parameters = {} if parameters is None else dict(parameters)
        if set(parameters) != set(self.required_parameters):
            raise ValueError(
                f"supply exactly the missing physical inputs {self.required_parameters}"
            )
        if self.equation_id == "tabulated":
            segments = self.build_segments(
                temperature_bounds_K=temperature_bounds_K, interpolation=interpolation
            )
            if len(segments) != 1:
                raise ValueError("select one table interval, or request build_segments explicitly")
            return segments[0]
        if interpolation is not None:
            raise ValueError("analytic or point data do not accept interpolation")
        coefficients = {**self.coefficients, **parameters}
        if any(number(v) is None for v in coefficients.values()):
            raise ValueError("missing or invalid source coefficient or observation")
        if self.equation_id == "phase_change_point":
            if temperature_bounds_K is not None:
                raise ValueError("a single observation does not define a temperature interval")
            if coefficients["T_ref"] <= 0 or coefficients["delta_h"] <= 0:
                raise ValueError("point latent heat and its temperature must be positive")
            bounds = (coefficients["T_ref"], coefficients["T_ref"])
        else:
            bounds = self._bounds(temperature_bounds_K)
        curve = self._correlation(self.curve_key, self.equation_id, coefficients, bounds)
        curve.metadata.update(caller_parameters=parameters)
        return curve

    def _correlation(self, key, equation, coefficients, bounds):
        return PropertyCorrelation(
            f"{self.record.record_id}:{key}",
            equation,
            dict(coefficients),
            {"temperature": "K"},
            "J/mol",
            self.property_id,
            {} if bounds is None else {"temperature": tuple(bounds)},
            self.record.source.provenance.citation + f"; record={self.record.record_id}",
            {
                "source": self.record.source.provenance.to_dict(),
                "source_record": self.record.record_id,
                "source_curve": self.curve_key,
                "source_bounds_K": None
                if self.source_bounds_K is None
                else list(self.source_bounds_K),
                "source_notes": list(self.notes),
                "temperature_range_basis": "single_source_point"
                if equation == "phase_change_point"
                else "explicit_source_or_caller_interval",
            },
        )

    def build_segments(self, *, interpolation, temperature_bounds_K=None):
        """Piecewise linear H(T), with original knots, nulls and derivative jumps."""
        if self.equation_id != "tabulated" or interpolation != "linear":
            raise ValueError("tabulated latent heat requires explicit linear interpolation")
        bounds = self._bounds(temperature_bounds_K)
        ts, hs = self.temperatures_K, self.enthalpies_J_mol
        if (
            len(ts) != len(hs)
            or len(ts) < 2
            or any(t is None for t in ts)
            or any(b <= a for a, b in pairwise(ts))
        ):
            raise ValueError(
                "latent table needs aligned increasing temperatures; no sorting or merging"
            )
        if bounds[0] < ts[0] or bounds[1] > ts[-1]:
            raise ValueError("cannot extrapolate a latent-heat table")
        segments = []
        for i, (low, high) in enumerate(pairwise(ts)):
            selected = max(bounds[0], low), min(bounds[1], high)
            if selected[0] >= selected[1]:
                continue
            h0, h1 = hs[i : i + 2]
            if any(number(h) is None or h < 0 for h in (h0, h1)):
                raise ValueError("selected interval crosses missing or negative latent heat")
            slope = (h1 - h0) / (high - low)
            curve = self._correlation(
                f"{self.curve_key}:{i}", "dippr100", {"A": h0 - slope * low, "B": slope}, selected
            )
            curve.metadata.update(interpolation=interpolation, source_point_indices=[i, i + 1])
            segments.append(curve)
        return tuple(segments)


def source_latent_heats(record):
    """All source curves, points and nulls; no method ranking or invented datum."""
    source = record.source_id

    def n(key):
        return number(record.raw(key)) if key in record.field_names else None

    def curve(key, equation, coefficients, bounds=None, **kwargs):
        return LatentHeatData(
            record, key, "heat_of_vaporization", equation, coefficients, bounds, **kwargs
        )

    def point(key, prop, temperature, *, notes=()):
        return LatentHeatData(
            record,
            key,
            prop,
            "phase_change_point",
            {"delta_h": n(key), "T_ref": temperature},
            required_parameters=("T_ref",) if temperature is None else (),
            notes=notes,
        )

    if source == "PERRY_VAPORIZATION":
        return (
            curve(
                "Hvap",
                "dippr106",
                {"Tc": n("Tc"), **{k: n(f"C{i}") for i, k in enumerate("ABCD", 1)}},
                (n("Tmin"), n("Tmax")),
                notes=("Original DIPPR106 domain is strictly T < Tc",),
            ),
        )
    if source == "VDI_PPDS_VAPORIZATION":
        return (curve("Hvap", "ppds12_hvap", {k: n(k) for k in ("Tc", *"ABCDE")}, (None, n("Tc"))),)
    if source == "ALIBAKHSHI":
        return (
            curve(
                "Hvap",
                "alibakhshi_hvap",
                {"C": n("C")},
                required_parameters=("Tc",),
                notes=("Reported fitting interval Tb-50 to Tc-100 K; explicit range required",),
            ),
        )
    if source == "CRC_FUSION":
        return (
            point(
                "Hfus",
                "heat_of_fusion",
                None,
                notes=("At normal melting point; temperature not supplied in this table",),
            ),
        )
    if source == "CRC_VAPORIZATION":
        return (
            point("HvapTb", "heat_of_vaporization", n("Tb")),
            point("Hvap298", "heat_of_vaporization", 298.15),
        )
    if source == "GHAZARATI_SUBLIMATION":
        return (
            point(
                "Hsub",
                "heat_of_sublimation",
                298.15,
                notes=("Reported error magnitude remains separate; not assumed one sigma",),
            ),
        )
    if source == "GHAZARATI_VAPORIZATION":
        return (point("Hvap298", "heat_of_vaporization", 298.15),)
    if source == "CHEMSEP_XML":
        results = []
        if "HeatOfVaporization" in record.field_names:
            data = record.raw("HeatOfVaporization")
            if not isinstance(data, SourceCorrelation):
                raise ValueError("ChemSep latent heat requires a correlation record")
            node = ElementTree.fromstring(data.xml)
            values = {child.tag: child.attrib.get("value") for child in node}
            if values.get("eqno") != "106":
                raise ValueError("unsupported ChemSep latent-heat equation")
            unit = XML_UNIT_NAMES.get(data.source_unit)
            if unit is None:
                raise ValueError("unknown ChemSep latent-heat unit")
            coefficients = {k: number(values.get(k)) for k in "ABCDE"}
            if coefficients["A"] is not None:
                coefficients["A"] = convert_value(coefficients["A"], unit, "J/mol")
            coefficients["Tc"] = n("CriticalTemperature")
            results.append(
                curve(
                    "HeatOfVaporization",
                    "dippr106",
                    coefficients,
                    (number(values.get("Tmin")), number(values.get("Tmax"))),
                    notes=("Scale coefficient A converted from source units; T < Tc",),
                )
            )
        if "HeatOfFusionAtMeltingPoint" in record.field_names:
            value = record.value("HeatOfFusionAtMeltingPoint")
            h = value.quantity("J/mol").value if value.quality == "valid" else None
            results.append(
                LatentHeatData(
                    record,
                    "Hfus",
                    "heat_of_fusion",
                    "phase_change_point",
                    {"delta_h": h, "T_ref": n("NormalMeltingPointTemperature")},
                    required_parameters=("T_ref",)
                    if n("NormalMeltingPointTemperature") is None
                    else (),
                    notes=(
                        "Source zero values are retained; not accepted as measured fusion heat",
                    ),
                )
            )
        return tuple(results)
    if source == "VDI":
        temperatures = tuple(number(v) for v in record.raw("T"))
        enthalpies = tuple(number(v) for v in record.raw("Hvap"))
        return (
            curve(
                "Hvap",
                "tabulated",
                {},
                (temperatures[0], temperatures[-1]),
                temperatures_K=temperatures,
                enthalpies_J_mol=enthalpies,
            ),
        )
    if source in {"WEBBOOK", "JOBACK", "WIKIDATA"}:
        # These aggregated fields do not carry per-observation temperatures.
        # Even where Tb/Tm is present, do not silently assert it is the datum
        # for every reported latent heat. Caller must supply the source condition.
        return tuple(
            point(
                key,
                prop,
                None,
                notes=(
                    "Aggregated source lacks observation temperature; "
                    "supply explicit source condition",
                    "Joback is predicted, not measured"
                    if source == "JOBACK"
                    else "Reported source value",
                ),
            )
            for key, prop in (
                ("Hfus", "heat_of_fusion"),
                ("Hvap", "heat_of_vaporization"),
                ("Hsub", "heat_of_sublimation"),
            )
            if key in record.field_names
        )
    # The other complete tables contain Tb/Tm and Twu alpha parameters, not
    # latent-heat observations. They remain native ConstantRecord consumers.
    return ()
