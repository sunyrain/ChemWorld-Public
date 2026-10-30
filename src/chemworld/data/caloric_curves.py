"""Source coefficients and explicit interpolation become existing native curves."""

from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite
from types import MappingProxyType
from xml.etree import ElementTree

from chemworld.data.constant_catalog import XML_UNIT_NAMES
from chemworld.data.constants import ConstantRecord, SourceCorrelation
from chemworld.physchem.heat_capacity import PiecewiseHeatCapacity
from chemworld.physchem.specs import PropertyCorrelation

_PHASE_PROPERTY = {
    "gas": "ideal_gas_heat_capacity",
    "liquid": "liquid_heat_capacity",
    "solid": "solid_heat_capacity",
}


def number(value):
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


@dataclass(frozen=True)
class HeatCapacityData:
    """One source range/shape, including unavailable parameters and original data.

    Inspecting a source never fills absent coefficients or silently extrapolates.
    Curves are built only when their missing physical inputs are explicit.
    """

    record: ConstantRecord
    curve_key: str
    phase: str
    equation_id: str
    coefficients: dict[str, float | None] = field(default_factory=dict)
    source_bounds_K: tuple[float | None, float | None] | None = None
    required_parameters: tuple[str, ...] = ()
    condition: str = "isobaric"
    temperatures_K: tuple[float | None, ...] = ()
    capacities_J_mol_K: tuple[float | None, ...] = ()
    output_unit: str = "J/(mol*K)"
    subphase: str = ""
    notes: tuple[str, ...] = ()
    capacity_qualities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in _PHASE_PROPERTY:
            raise ValueError("source curve requires an explicit physical phase")
        object.__setattr__(self, "coefficients", MappingProxyType(dict(self.coefficients)))

    @property
    def curve_id(self):
        return f"{self.record.record_id}:{self.curve_key}"

    @property
    def metadata(self):
        return {
            "phase": self.phase,
            "subphase": self.subphase,
            "source_id": self.record.source.provenance.source_id,
            "source_record": self.record.record_id,
            "source_curve": self.curve_key,
            "source_condition": self.condition,
            "source_notes": list(self.notes),
            "source_version": self.record.source.provenance.version,
        }

    def build(self, *, parameters=None, temperature_bounds_K=None, interpolation=None):
        supplied = {} if parameters is None else dict(parameters)
        if set(supplied) != set(self.required_parameters):
            raise ValueError(f"{self.curve_id}: supply exactly {self.required_parameters}")
        bounds = self.source_bounds_K if temperature_bounds_K is None else temperature_bounds_K
        if (
            bounds is None
            or len(bounds) != 2
            or any(not isinstance(v, int | float) or number(v) is None for v in bounds)
            or not 0 < bounds[0] < bounds[1]
        ):
            raise ValueError("select an explicit finite positive temperature range for this source")
        if self.source_bounds_K is not None:
            low, high = self.source_bounds_K
            if (low is not None and bounds[0] < low) or (high is not None and bounds[1] > high):
                raise ValueError("requested range extrapolates beyond source bounds")
        if self.equation_id == "tabulated":
            if interpolation != "linear":
                raise ValueError("tabulated capacity requires explicit interpolation='linear'")
            return self._linear_curve(bounds)
        if interpolation is not None:
            raise ValueError("analytic coefficients do not require interpolation")
        missing = [key for key, value in self.coefficients.items() if value is None]
        if missing:
            raise ValueError(f"{self.curve_id}: missing or invalid source coefficients {missing}")
        correlation = self._correlation(
            self.curve_key, self.equation_id, {**self.coefficients, **supplied}, bounds
        )
        correlation.metadata.update(
            caller_parameters=supplied,
            temperature_range_basis=(
                "caller-supplied"
                if self.source_bounds_K is None or any(v is None for v in self.source_bounds_K)
                else "source-subrange"
            ),
        )
        return correlation

    def _correlation(self, key, equation, coefficients, bounds):
        return PropertyCorrelation(
            f"{self.record.record_id}:{key}",
            equation,
            coefficients,
            {"temperature": "K"},
            self.output_unit,
            _PHASE_PROPERTY[self.phase],
            {"temperature": tuple(bounds)},
            self.record.source.provenance.citation + f"; record={self.record.record_id}",
            self.metadata,
        )

    def _linear_curve(self, bounds):
        temperatures, values = self.temperatures_K, self.capacities_J_mol_K
        if len(temperatures) != len(values) or len(temperatures) < 2:
            raise ValueError("tabulated capacity needs aligned temperature and value arrays")
        if any(T is None for T in temperatures) or any(b < a for a, b in pairwise(temperatures)):
            raise ValueError("tabulated temperatures cannot decrease; no sorting/merging")
        if bounds[0] < temperatures[0] or bounds[1] > temperatures[-1]:
            raise ValueError("tabulated range cannot extrapolate")
        # Repeated source temperatures can be distinct phase/subphase limits.
        # Keep both observations. A selected interval may end at the left limit
        # or start at the right limit, but cannot silently integrate across it.
        if any(a == b and bounds[0] < a < bounds[1] for a, b in pairwise(temperatures)):
            raise ValueError(
                "select one side of repeated source temperatures; "
                "crossing requires explicit phase/transition information"
            )
        segments = []
        for index in range(len(temperatures) - 1):
            low, high = temperatures[index : index + 2]
            selected = (max(low, bounds[0]), min(high, bounds[1]))
            if selected[0] >= selected[1]:
                continue
            first, second = values[index : index + 2]
            if first is None or second is None:
                raise ValueError("selected interval crosses a missing capacity; no filling gaps")
            slope = (second - first) / (high - low)
            segments.append(
                self._correlation(
                    f"{self.curve_key}:{index}",
                    "cp_polynomial",
                    {"a": first - slope * low, "b": slope},
                    selected,
                )
            )
        return PiecewiseHeatCapacity(
            self.curve_id,
            tuple(segments),
            self.record.source.provenance.citation + "; linear interpolation",
        )

    def scientific_table(self, *, temperature_column, capacity_column, dataset_id, group_id, split):
        """Every original table point, including zero temperature and missing Cp."""
        from chemworld.data.scientific_inputs import (
            InputSource,
            InputValue,
            ScientificRow,
            ScientificTable,
        )

        if (
            self.equation_id != "tabulated"
            or temperature_column.role != "condition"
            or capacity_column.role != "observation"
        ):
            raise ValueError("a table needs temperature-condition and capacity-observation columns")
        if len(self.temperatures_K) != len(self.capacities_J_mol_K):
            raise ValueError("temperature and capacity arrays must align")
        source = InputSource(self.record.source.provenance, "literature")
        note = self.record.source.nature + "; source uncertainty not quantified"
        qualities = self.capacity_qualities or tuple(
            "missing" if cp is None else "valid" for cp in self.capacities_J_mol_K
        )
        if len(qualities) != len(self.capacities_J_mol_K):
            raise ValueError("one quality value is required per capacity observation")
        rows = tuple(
            ScientificRow(
                f"{self.curve_id}:{i}",
                source.provenance.source_id,
                group_id,
                f"{self.curve_id}:{i}",
                split,
                {
                    temperature_column.quantity_id: InputValue(T, "K", None, note),
                    capacity_column.quantity_id: InputValue(cp, "J/(mol*K)", None, note, quality),
                },
            )
            for i, (T, cp, quality) in enumerate(
                zip(self.temperatures_K, self.capacities_J_mol_K, qualities, strict=True)
            )
        )
        return ScientificTable(
            dataset_id, "measurements", (temperature_column, capacity_column), (source,), rows
        )


def _bounds(raw, low="Tmin", high="Tmax"):
    return number(raw.get(low)), number(raw.get(high))


def _qualities(values):
    return tuple(
        "valid"
        if number(value) is not None
        else "missing"
        if value is None or value == ""
        else "invalid"
        for value in values
    )


def _perry_curve(record, key, raw):
    phase = {"c": "solid", "gls": "solid", "l": "liquid", "g": "gas"}.get(raw["Phase"])
    if phase is None:
        raise ValueError(f"{record.record_id}: unsupported original Perry phase {raw['Phase']!r}")
    values = [
        number(raw.get("Const")),
        *(
            0.0 if raw.get(name) in (None, "") else number(raw[name])
            for name in ("Lin", "Quadinv", "Quad")
        ),
    ]
    # The source omits terms which are absent from the regression. The constant
    # itself is required; an absent constant is not silently replaced by zero.
    a, b, c, d = values
    if phase == "solid":
        equation, coefficients = "perry151_cp", {"a": a, "b": b, "c": c, "d": d}
    else:
        # Same published polynomial; the existing Shomate shape accepts these
        # phases. Convert original calories exactly, without a new equation.
        equation = "shomate_cp"
        coefficients = {
            key: None if value is None else 4.184 * value
            for key, value in zip("ABCDE", (a, b, d, 0.0, c), strict=True)
        }
    return HeatCapacityData(
        record,
        key,
        phase,
        equation,
        coefficients,
        _bounds(raw),
        subphase=str(raw.get("Subphase") or ("glass" if raw["Phase"] == "gls" else "")),
    )


def source_heat_capacities(record):
    """Return every declared range in this record; no method ranking or deduplication."""
    source = record.source_id
    raw = dict(zip(record.field_names, record.raw_fields, strict=True))
    if source == "CHEMSEP_XML":
        return _chemsep_curves(record)
    if source == "JOBACK":
        return (
            HeatCapacityData(
                record,
                "gas-prediction",
                "gas",
                "dippr100",
                {key: number(raw[f"Cpg{i}"]) for i, key in enumerate("ABCD")},
                condition="Joback group-contribution prediction; range supplied by caller",
            ),
        )
    if source == "VDI":
        return tuple(
            HeatCapacityData(
                record,
                key,
                phase,
                "tabulated",
                source_bounds_K=(number(raw["T"][0]), number(raw["T"][-1])),
                temperatures_K=tuple(map(number, raw["T"])),
                capacities_J_mol_K=tuple(map(number, raw[key])),
                condition="saturation",
                capacity_qualities=_qualities(raw[key]),
            )
            for key, phase in (("Cp (l)", "liquid"), ("Cp (g)", "gas"))
        )
    if source == "POLING_CP":
        coefficients = {key: number(raw[f"a{i}"]) for i, key in enumerate("abcde")}
        return (HeatCapacityData(record, "gas", "gas", "poling_cp", coefficients, _bounds(raw)),)
    if source == "TRC_GAS":
        coefficients = {f"a{i}": number(raw[f"a{i}"]) for i in range(8)}
        return (HeatCapacityData(record, "gas", "gas", "trc_cp", coefficients, _bounds(raw)),)
    if source in {"PERRY_LIQUID_100", "PERRY_LIQUID_114"}:
        special = source.endswith("114")
        coefficients = {key: number(raw[key]) for key in ("ABCD" if special else "ABCDE")}
        return (
            HeatCapacityData(
                record,
                "liquid",
                "liquid",
                "dippr114" if special else "dippr100",
                coefficients,
                _bounds(raw),
                ("Tc",) if special else (),
                output_unit="J/(kmol*K)",
            ),
        )
    if source in {"PERRY_151_RAW", "PERRY_151_CLEAN"}:
        converted = {
            "Phase": raw["main phase"],
            "Subphase": raw["subphase"],
            "Const": raw["Constant Term"],
            "Lin": raw["+*T term"],
            "Quadinv": raw["/T^2 term"],
            "Quad": raw["*T^2 term"],
            "Tmin": raw["Tmin"],
            "Tmax": raw["Tmax"],
        }
        return (_perry_curve(record, "polynomial", converted),)
    if source == "PERRY_151_JSON":
        return tuple(_perry_curve(record, key, value) for key, value in raw["payload"].items())
    if source == "WEBBOOK_SHOMATE":
        result = []
        payload = raw["payload"]
        if len(payload) != 3:
            raise ValueError("WebBook Shomate record needs solid/liquid/gas groups")
        for phase, ranges in zip(("solid", "liquid", "gas"), payload, strict=True):
            for i, segment in enumerate(ranges or ()):
                if len(segment) != 7:
                    raise ValueError("WebBook Shomate range needs Tmin/Tmax and five coefficients")
                result.append(
                    HeatCapacityData(
                        record,
                        f"{phase}:{i}",
                        phase,
                        "shomate_cp",
                        dict(zip("ABCDE", map(number, segment[2:]), strict=True)),
                        tuple(map(number, segment[:2])),
                    )
                )
        return tuple(result)
    if source == "ZABRANSKY":
        result = []
        for kind, total in (("spline", 4), ("quasi", 6)):
            coefficients = {f"a{i}": number(raw[f"A{i}-{kind}"]) for i in range(1, total + 1)}
            if all(value == 0 for value in coefficients.values()):
                continue  # Source zero-filled columns explicitly denote an absent form.
            if kind == "quasi":
                coefficients["Tc"] = number(raw["Tc"])
            result.append(
                HeatCapacityData(
                    record,
                    kind,
                    "liquid",
                    "zabransky_cubic_cp" if kind == "spline" else "zabransky_quasi_cp",
                    coefficients,
                    _bounds(raw),
                    condition="Zabransky:" + raw["Data Type"],
                )
            )
        return tuple(result)
    if source.startswith("JANAF_") and source.endswith("_CP"):
        temperatures, capacities = raw["payload"]
        phase = {"JANAF_GAS_CP": "gas", "JANAF_LIQUID_CP": "liquid", "JANAF_SOLID_CP": "solid"}[
            source
        ]
        bounds = (number(temperatures[0]), number(temperatures[-1])) if temperatures else None
        return (
            HeatCapacityData(
                record,
                "table",
                phase,
                "tabulated",
                source_bounds_K=bounds,
                temperatures_K=tuple(map(number, temperatures)),
                capacities_J_mol_K=tuple(map(number, capacities)),
                capacity_qualities=_qualities(capacities),
            ),
        )
    if source == "CRC_SOLID_CP":
        keys = [key for key in record.field_names if key.isdigit()]
        temperatures = tuple(map(float, keys))
        return (
            HeatCapacityData(
                record,
                "table",
                "solid",
                "tabulated",
                source_bounds_K=(temperatures[0], temperatures[-1]),
                temperatures_K=temperatures,
                capacities_J_mol_K=tuple(number(raw[key]) for key in keys),
                capacity_qualities=_qualities(raw[key] for key in keys),
            ),
        )
    if source in {"PSI4_ADJUSTED_THETA", "PSI4_UNADJUSTED_THETA"}:
        return (
            HeatCapacityData(
                record,
                "vibration",
                "gas",
                "statistical_cpg",
                {f"theta_{i}": number(value) for i, value in enumerate(raw["payload"])},
                required_parameters=("linear",),
                condition=record.source.nature,
            ),
        )
    return ()  # Thermodynamic scalar tables are not invented temperature curves.


def _chemsep_curves(record):
    result = []
    for field_id, phase in (
        ("LiquidHeatCapacityCp", "liquid"),
        ("SolidHeatCapacityCp", "solid"),
        ("IdealGasHeatCapacityCp", "gas"),
        ("RPPHeatCapacityCp", "gas"),
    ):
        if field_id not in record.field_names:
            continue
        value = record.raw(field_id)
        if not isinstance(value, SourceCorrelation):
            raise ValueError(f"{record.record_id}/{field_id}: expected complete XML correlation")
        node = ElementTree.fromstring(value.xml)
        data = {child.tag: child.get("value") for child in node}
        code = data["eqno"]
        if code in {"1", "3", "4", "5", "100"}:
            keys = "ABCDE"[: 5 if code == "100" else int(code)]
            equation = "dippr100"
        elif code == "102":
            keys, equation = "ABCD", "dippr102"
        elif code == "16":
            keys, equation = "ABCDE", "chemsep16"
        else:
            raise ValueError(f"unsupported ChemSep heat-capacity equation {code!r}")
        unit = XML_UNIT_NAMES.get(value.source_unit)
        if unit is None:
            raise ValueError(f"unknown ChemSep heat-capacity unit {value.source_unit!r}")
        bounds = _bounds(data)
        invalid = tuple(
            key for key, v in zip(("Tmin", "Tmax"), bounds, strict=True) if v is None or v <= 0
        )
        bounds = tuple(v if v is not None and v > 0 else None for v in bounds)
        notes = (
            (f"Source has unavailable positive bounds {invalid}; explicitly supply a range",)
            if invalid
            else ()
        )
        result.append(
            HeatCapacityData(
                record,
                field_id,
                phase,
                equation,
                {key: number(data.get(key)) for key in keys},
                bounds,
                output_unit=unit,
                condition="ChemSep:" + field_id,
                notes=notes,
            )
        )
    return tuple(result)
