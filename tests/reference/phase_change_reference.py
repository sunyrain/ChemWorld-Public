"""Independent raw source interpretation; no native parameters form references."""

from dataclasses import dataclass, field

from caloric_source_reference import numeric, read_source  # noqa: F401
from chemicals import dippr, phase_change


@dataclass
class ReferenceLatent:
    key: str
    property_id: str
    method: str
    parameters: dict = field(default_factory=dict)
    bounds: tuple | None = None
    supplied: dict = field(default_factory=dict)
    temperatures: tuple = ()
    values: tuple = ()

    def evaluate(self, temperature):
        if self.method == "point":
            assert temperature == self.parameters["T_ref"]
            return self.parameters["delta_h"]
        if self.method == "EQ106":
            return dippr.EQ106(temperature, **self.parameters)
        return getattr(phase_change, self.method)(temperature, **self.parameters)


def source_curves(source, raw):
    def n(key):
        return numeric(raw.get(key))

    def point(key, prop, T, *, enthalpy=None):
        return ReferenceLatent(
            key,
            "heat_of_" + prop,
            "point",
            {"delta_h": n(key) if enthalpy is None else enthalpy, "T_ref": 300 if T is None else T},
            supplied={"T_ref": 300} if T is None else {},
        )

    if source == "PERRY_VAPORIZATION":
        return [
            ReferenceLatent(
                "Hvap",
                "heat_of_vaporization",
                "EQ106",
                {"Tc": n("Tc"), **{key: n(f"C{i}") for i, key in enumerate("ABCD", 1)}},
                (n("Tmin"), n("Tmax")),
            )
        ]
    if source == "VDI_PPDS_VAPORIZATION":
        tc = n("Tc")
        return [
            ReferenceLatent(
                "Hvap",
                "heat_of_vaporization",
                "PPDS12",
                {k: n(k) for k in ("Tc", *"ABCDE")},
                (0.5 * tc, 0.9 * tc),
            )
        ]
    if source == "ALIBAKHSHI":
        return [
            ReferenceLatent(
                "Hvap",
                "heat_of_vaporization",
                "Alibakhshi",
                {"C": n("C"), "Tc": 600},
                (250, 400),
                supplied={"Tc": 600},
            )
        ]
    if source == "CRC_FUSION":
        return [point("Hfus", "fusion", None)]
    if source == "CRC_VAPORIZATION":
        return [point("HvapTb", "vaporization", n("Tb")), point("Hvap298", "vaporization", 298.15)]
    if source == "GHAZARATI_SUBLIMATION":
        return [point("Hsub", "sublimation", 298.15)]
    if source == "GHAZARATI_VAPORIZATION":
        return [point("Hvap298", "vaporization", 298.15)]
    if source == "CHEMSEP_XML":
        result = []
        node = raw.find("HeatOfVaporization")
        if node is not None:
            assert node.attrib["units"] == "J/kmol"
            values = {child.tag: child.attrib.get("value") for child in node}
            assert values["eqno"] == "106"
            params = {key: numeric(values.get(key)) for key in "ABCDE"}
            if params["A"] is not None:
                params["A"] /= 1000
            params["Tc"] = numeric(raw.find("CriticalTemperature").attrib["value"])
            result.append(
                ReferenceLatent(
                    "HeatOfVaporization",
                    "heat_of_vaporization",
                    "EQ106",
                    params,
                    (numeric(values.get("Tmin")), numeric(values.get("Tmax"))),
                )
            )
        node = raw.find("HeatOfFusionAtMeltingPoint")
        if node is not None:
            assert node.attrib["units"] == "J/kmol"
            h = numeric(node.attrib["value"])
            tnode = raw.find("NormalMeltingPointTemperature")
            temperature = None if tnode is None else numeric(tnode.attrib["value"])
            result.append(
                ReferenceLatent(
                    "Hfus",
                    "heat_of_fusion",
                    "point",
                    {
                        "delta_h": None if h is None else h / 1000,
                        "T_ref": 300 if temperature is None else temperature,
                    },
                    supplied={"T_ref": 300} if temperature is None else {},
                )
            )
        return result
    if source == "VDI":
        values = raw[1]
        return [
            ReferenceLatent(
                "Hvap",
                "heat_of_vaporization",
                "table",
                bounds=(values["T"][0], values["T"][-1]),
                temperatures=tuple(values["T"]),
                values=tuple(values["Hvap"]),
            )
        ]
    if source in {"WEBBOOK", "JOBACK", "WIKIDATA"}:
        return [
            point(key, prop, None)
            for key, prop in (("Hfus", "fusion"), ("Hvap", "vaporization"), ("Hsub", "sublimation"))
            if key in raw
        ]
    return []
