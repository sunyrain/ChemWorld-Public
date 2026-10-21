"""Independent public-formula references from raw 1.5.2 distribution files.

No production adapter, model or curve coefficients are used to form a reference.
The published formula API is a test-only dependency; no upstream code is copied.
"""

import csv
import json
import math
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from chemicals import dippr, heat_capacity


def numeric(value):
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


def read_source(path):
    if path.suffix == ".json":
        return list(json.loads(path.read_text(encoding="utf-8")).items())
    if path.suffix == ".xml":
        return list(ET.parse(path).getroot())
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


@dataclass
class ReferenceCurve:
    key: str
    phase: str
    function: object = None
    bounds: tuple = (None, None)
    parameters: dict = field(default_factory=dict)
    missing: bool = False
    temperatures: tuple = ()
    values: tuple = ()
    condition: str | None = None
    subphase: str | None = None


def interval(bounds):
    low, high = bounds
    if low is not None and high is not None and 0 < low < high:
        return low, high
    # Only unavailable/nonpositive endpoints get the prespecified supplemental range.
    low = max(200, low) if low is not None and low > 0 else 200
    high = min(600, high) if high is not None and high > 0 else 600
    return low, high


def analytic(key, phase, function, args, bounds=(None, None), scale=1, **options):
    values = [numeric(value) for value in args]

    def evaluate(T):
        return scale * function(T, *values)

    return ReferenceCurve(key, phase, evaluate, bounds, missing=None in values, **options)


def table(key, phase, temperatures, values, condition=None):
    return ReferenceCurve(
        key, phase, temperatures=tuple(temperatures), values=tuple(values), condition=condition
    )


def perry(key, raw):
    phase = {"c": "solid", "gls": "solid", "l": "liquid", "g": "gas"}[raw["Phase"]]
    terms = [
        raw["Const"],
        *(0 if raw.get(k) in (None, "") else raw[k] for k in ("Lin", "Quadinv", "Quad")),
    ]
    return analytic(
        key,
        phase,
        heat_capacity.Perry_151,
        terms,
        (numeric(raw.get("Tmin")), numeric(raw.get("Tmax"))),
        subphase=str(raw.get("Subphase") or ("glass" if raw["Phase"] == "gls" else "")),
    )


def source_curves(source, raw):
    bounds = (numeric(raw.get("Tmin")), numeric(raw.get("Tmax"))) if isinstance(raw, dict) else ()
    if source == "JOBACK":
        return [analytic("gas-prediction", "gas", dippr.EQ100, [raw[f"Cpg{i}"] for i in range(4)])]
    if source == "VDI":
        data = raw[1]
        return [
            table(key, phase, data["T"], data[key], "saturation")
            for key, phase in (("Cp (l)", "liquid"), ("Cp (g)", "gas"))
        ]
    if source == "POLING_CP":
        return [
            analytic("gas", "gas", heat_capacity.Poling, [raw[f"a{i}"] for i in range(5)], bounds)
        ]
    if source == "TRC_GAS":
        return [
            analytic("gas", "gas", heat_capacity.TRCCp, [raw[f"a{i}"] for i in range(8)], bounds)
        ]
    if source == "PERRY_LIQUID_100":
        return [analytic("liquid", "liquid", dippr.EQ100, [raw[k] for k in "ABCDE"], bounds, 0.001)]
    if source == "PERRY_LIQUID_114":
        high = interval(bounds)[1]
        Tc = max(1.1 * high, high + 1)
        return [
            ReferenceCurve(
                "liquid",
                "liquid",
                lambda T: 0.001 * dippr.EQ114(T, Tc, *(float(raw[k]) for k in "ABCD")),
                bounds,
                {"Tc": Tc},
                any(numeric(raw[k]) is None for k in "ABCD"),
            )
        ]
    if source in {"PERRY_151_RAW", "PERRY_151_CLEAN"}:
        return [
            perry(
                "polynomial",
                {
                    "Phase": raw["main phase"],
                    "Subphase": raw["subphase"],
                    "Const": raw["Constant Term"],
                    "Lin": raw["+*T term"],
                    "Quadinv": raw["/T^2 term"],
                    "Quad": raw["*T^2 term"],
                    "Tmin": raw["Tmin"],
                    "Tmax": raw["Tmax"],
                },
            )
        ]
    if source == "PERRY_151_JSON":
        return [perry(key, values) for key, values in raw[1].items()]
    if source == "WEBBOOK_SHOMATE":
        return [
            analytic(f"{phase}:{i}", phase, heat_capacity.Shomate, row[2:], tuple(row[:2]))
            for phase, rows in zip(("solid", "liquid", "gas"), raw[1], strict=True)
            for i, row in enumerate(rows or ())
        ]
    if source == "ZABRANSKY":
        result = []
        for kind, count, function in (
            ("spline", 4, heat_capacity.Zabransky_cubic),
            ("quasi", 6, heat_capacity.Zabransky_quasi_polynomial),
        ):
            args = [numeric(raw[f"A{i}-{kind}"]) for i in range(1, count + 1)]
            if all(v == 0 for v in args):
                continue
            if kind == "quasi":
                args.insert(0, numeric(raw["Tc"]))
            result.append(
                analytic(
                    kind,
                    "liquid",
                    function,
                    args,
                    bounds,
                    condition="Zabransky:" + raw["Data Type"],
                )
            )
        return result
    if source.startswith("JANAF_") and source.endswith("_CP"):
        phase = {"JANAF_GAS_CP": "gas", "JANAF_LIQUID_CP": "liquid", "JANAF_SOLID_CP": "solid"}[
            source
        ]
        return [table("table", phase, *raw[1])]
    if source == "CRC_SOLID_CP":
        columns = [k for k in raw if k.isdigit()]
        return [table("table", "solid", list(map(float, columns)), [raw[k] for k in columns])]
    if source.startswith("PSI4_"):
        return [
            ReferenceCurve(
                "vibration",
                "gas",
                lambda T: heat_capacity.Cpg_statistical_mechanics(T, raw[1], linear=False),
                parameters={"linear": 0},
                missing=any(numeric(v) is None for v in raw[1]),
            )
        ]
    if source == "CHEMSEP_XML":
        return chemsep_curves(raw)
    return []


def chemsep_curves(raw):
    curves = []
    for key, phase in (
        ("LiquidHeatCapacityCp", "liquid"),
        ("SolidHeatCapacityCp", "solid"),
        ("IdealGasHeatCapacityCp", "gas"),
        ("RPPHeatCapacityCp", "gas"),
    ):
        node = raw.find(key)
        if node is None:
            continue
        data = {item.tag: item.attrib["value"] for item in node}
        code = int(data["eqno"])
        scale = {"J/kmol/K": 0.001, "J/mol/K": 1}[node.attrib["units"]]
        if code == 16:

            def function(T, a, b, c, d, e):
                return a + math.exp(b / T + c + d * T + e * T * T)

            count = 5
        elif code == 102:
            function, count = dippr.EQ102, 4
        elif code in {1, 3, 4, 5, 100}:
            function, count = dippr.EQ100, 5 if code == 100 else code
        else:
            raise ValueError(f"unmapped source equation {code}")
        curves.append(
            analytic(
                key,
                phase,
                function,
                [data[k] for k in "ABCDE"[:count]],
                (numeric(data.get("Tmin")), numeric(data.get("Tmax"))),
                scale,
                condition="ChemSep:" + key,
            )
        )
    return curves
