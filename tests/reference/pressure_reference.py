"""Independent raw-data and public mathematical API pressure references.

Only this test module imports chemicals. No production source adapter supplies
reference parameters, source intervals, identities, values or derivatives.
"""

from dataclasses import dataclass, field
from math import e, isfinite
from xml.etree import ElementTree as ET

from caloric_source_reference import numeric, read_source  # noqa: F401 -- shared raw reader
from chemicals import dippr
from chemicals import vapor_pressure as vp


@dataclass
class ReferencePressure:
    key: str
    method: str
    coefficients: dict = field(default_factory=dict)
    bounds: tuple = (None, None)
    property_id: str = "vapor_pressure"
    temperatures: tuple = ()
    values: tuple = ()

    def evaluate(self, T, order=0):
        if self.method.startswith("EQ"):
            return getattr(dippr, self.method)(T, **self.coefficients, order=order)
        name = (
            self.method
            if order == 0
            else "d" + self.method + "_dT"
            if order == 1
            else f"d{order}" + self.method + f"_dT{order}"
        )
        return getattr(vp, name)(T, **self.coefficients)


def source_curves(source, raw):
    if source == "VDI":
        data = raw[1]
        return [
            ReferencePressure(
                "pressure",
                "tabulated",
                bounds=(data["T"][0], data["T"][-1]),
                temperatures=tuple(data["T"]),
                values=tuple(data["P"]),
            )
        ]
    if source == "CHEMSEP_XML":
        curves = []
        for key in ("VaporPressure", "AntoineVaporPressure"):
            node = raw.find(key)
            if node is None:
                continue
            assert node.attrib["units"] == "Pa"
            d = {c.tag: c.attrib.get("value") for c in node}
            eq = d["eqno"]
            params = {k: numeric(d.get(k)) for k in ("ABC" if eq == "10" else "ABCDE")}
            if eq == "10":
                params["base"] = e
            if eq == "200":
                critical = raw.find("CriticalTemperature")
                assert critical.attrib["units"] == "K"
                params = {
                    **{k.lower(): numeric(d[k]) for k in "ABCD"},
                    "Tc": numeric(critical.attrib["value"]),
                    "Pc": numeric(d["E"]),
                }
            curves.append(
                ReferencePressure(
                    key,
                    {"10": "Antoine", "101": "EQ101", "200": "Wagner"}.get(eq, "unresolved_" + eq),
                    params,
                    (numeric(d.get("Tmin")), numeric(d.get("Tmax"))),
                )
            )
        return curves
    if source == "WEBBOOK":
        names = ["Antoine" + k for k in ("A", "B", "C", "Tmin", "Tmax")]
        if not any(raw.get(k) not in (None, "") for k in names):
            return []
        return [
            ReferencePressure(
                "Antoine",
                "Antoine",
                {**{k: numeric(raw.get("Antoine" + k)) for k in "ABC"}, "base": e},
                (numeric(raw.get("AntoineTmin")), numeric(raw.get("AntoineTmax"))),
            )
        ]
    bounds = tuple(numeric(raw.get(k)) for k in ("Tmin", "Tmax"))
    prop = "sublimation_pressure" if "SUBLIMATION" in source else "vapor_pressure"
    if source in {"ANTOINE_POLING", "LANDOLT_PRESSURE", "LANDOLT_SUBLIMATION"}:
        return [
            ReferencePressure(
                "pressure",
                "Antoine",
                {
                    **{k: numeric(raw[k]) for k in "ABC"},
                    "base": 10 if source == "ANTOINE_POLING" else e,
                },
                bounds,
                prop,
            )
        ]
    if source in {"ALCOCK_PRESSURE", "ALCOCK_SUBLIMATION", "PERRY_PRESSURE"}:
        return [
            ReferencePressure(
                "pressure",
                "EQ101",
                {
                    k: numeric(raw[f"C{i}" if source == "PERRY_PRESSURE" else k])
                    for i, k in enumerate("ABCDE", 1)
                },
                bounds,
                prop,
            )
        ]
    if source == "TRC_ANTOINE_POLING":
        return [
            ReferencePressure(
                "pressure",
                "TRC_Antoine_extended",
                {k: numeric(raw[k]) for k in ("Tc", "to", "A", "B", "C", "n", "E", "F")},
                bounds,
            )
        ]
    if source in {"WAGNER_POLING", "WAGNER_MCGARRY", "VDI_PPDS_PRESSURE"}:
        params = {
            **{k.lower(): numeric(raw[k]) for k in "ABCD"},
            "Tc": numeric(raw["Tc"]),
            "Pc": numeric(raw["Pc"]),
        }
        if source == "VDI_PPDS_PRESSURE":
            bounds = numeric(raw["Tm"]), numeric(raw["Tc"])
        if source == "WAGNER_MCGARRY":
            bounds = numeric(raw["Tmin"]), numeric(raw["Tc"])
        return [
            ReferencePressure(
                "pressure",
                "Wagner_original" if source == "WAGNER_MCGARRY" else "Wagner",
                params,
                bounds,
            )
        ]
    raise AssertionError(source)


def check_raw(record, raw):
    if isinstance(raw, dict):
        assert record.headers == tuple(raw)
        assert record.raw_fields == tuple(raw.values())
        assert record.source_key == raw.get("CAS", next(iter(raw.values())))
    elif isinstance(raw, tuple):
        assert record.source_key == raw[0]
        assert record.field_names == tuple(raw[1])
        for name, expected in raw[1].items():
            assert record.raw(name) == (tuple(expected) if isinstance(expected, list) else expected)
    else:
        assert record.source_key == raw.find("CAS").attrib["value"]
        assert record.field_names == tuple(child.tag for child in raw)
        for child in raw:
            actual = record.raw(child.tag)
            if not len(child) and "value" in child.attrib:
                assert actual == child.attrib["value"]
            else:
                assert actual.xml == ET.tostring(child, encoding="unicode")


def finite_difference(function, T, order):
    h = 1e-3 * T
    values = []
    for step in (h, h / 2):
        fm2, fm, f0, fp, fp2 = (function(T + i * step) for i in (-2, -1, 0, 1, 2))
        values.append(
            (fm2 - 8 * fm + 8 * fp - fp2) / (12 * step)
            if order == 1
            else (-fp2 + 16 * fp - 30 * f0 + 16 * fm - fm2) / (12 * step**2)
        )
    return values


def interval(bounds):
    low, high = bounds
    if low is None and high is None:
        return 200.0, 600.0
    if low is None:
        low = high / 2
    if high is None:
        high = 2 * low
    return low, high


def valid_pressure(value):
    return isfinite(value) and value > 0
