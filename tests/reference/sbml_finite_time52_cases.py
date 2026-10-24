"""Authored physical-unit-equivalent formal models for finite-time stability."""

from chemworld.data.sbml import SBMLModel
from chemworld.examples.sbml_calibration import pool_model, source


def model(kind="nonnormal", *, scaled=False):
    if kind == "closed":
        return pool_model(kind="reversible", initial_b=0.2, scaled=scaled)
    amount, seconds = (0.001, 60.0) if scaled else (1.0, 1.0)
    exponent = -3 if scaled else 0
    if kind == "nonnormal":
        names, initial = ("X", "Y"), (0.6, 4.0)
        parameters = (
            ("j", 0.3 * seconds / amount, "incoming"),
            ("a", 0.5 * seconds, "inverse_time"),
            ("c", 8 * seconds, "inverse_time"),
            ("d", 1.2 * seconds, "inverse_time"),
        )
        channels = (
            ("birth", {}, {"X": 1}, "<ci>j</ci>"),
            ("loss_x", {"X": 1}, {}, "<apply><times/><ci>a</ci><ci>X</ci></apply>"),
            ("make_y", {"X": 1}, {"X": 1, "Y": 1}, "<apply><times/><ci>c</ci><ci>X</ci></apply>"),
            ("loss_y", {"Y": 1}, {}, "<apply><times/><ci>d</ci><ci>Y</ci></apply>"),
        )
    elif kind == "nonlinear":
        names, initial = ("A", "B"), (0.8, 0.1)
        parameters = (
            ("a", 0.7 * seconds * amount, "second_order"),
            ("b", 0.3 * seconds, "inverse_time"),
        )
        channels = (
            (
                "forward",
                {"A": 2},
                {"B": 1},
                "<apply><times/><ci>a</ci><ci>A</ci><ci>A</ci></apply>",
            ),
            ("backward", {"B": 1}, {"A": 2}, "<apply><times/><ci>b</ci><ci>B</ci></apply>"),
        )
    elif kind == "growth":
        names, initial = ("X",), (0.5,)
        parameters = (("a", 0.4 * seconds, "inverse_time"),)
        channels = (("grow", {"X": 1}, {"X": 2}, "<apply><times/><ci>a</ci><ci>X</ci></apply>"),)
    elif kind == "birth":
        names, initial = ("X",), (1.0,)
        parameters = (
            ("j", 0.5 * seconds / amount, "incoming"),
            ("a", 0.5 * seconds, "inverse_time"),
        )
        channels = (
            ("birth", {}, {"X": 1}, "<ci>j</ci>"),
            ("loss", {"X": 1}, {}, "<apply><times/><ci>a</ci><ci>X</ci></apply>"),
        )
    else:
        raise ValueError(kind)

    def unit(name, terms):
        return (
            f'<unitDefinition id="{name}"><listOfUnits>'
            + "".join(
                f'<unit kind="{base}" exponent="{power}" scale="{scale}" multiplier="{multiple}"/>'
                for base, power, scale, multiple in terms
            )
            + "</listOfUnits></unitDefinition>"
        )

    units = "".join(
        (
            unit("amount", (("mole", 1, exponent, 1),)),
            unit("clock", (("second", 1, 0, seconds),)),
            unit("inverse_time", (("second", -1, 0, seconds),)),
            unit("incoming", (("mole", 1, exponent, 1), ("second", -1, 0, seconds))),
            unit("second_order", (("mole", -1, exponent, 1), ("second", -1, 0, seconds))),
        )
    )
    species = "".join(
        f'<species id="{s}" compartment="V" initialAmount="{value / amount}" '
        'hasOnlySubstanceUnits="true" boundaryCondition="false" constant="false"/>'
        for s, value in zip(names, initial, strict=True)
    )
    params = "".join(
        f'<parameter id="{p}" value="{v}" units="{u}" constant="true"/>' for p, v, u in parameters
    )

    def side(tag, entries):
        if not entries:
            return ""
        return (
            f"<listOf{tag}>"
            + "".join(
                f'<speciesReference species="{s}" stoichiometry="{n}" constant="true"/>'
                for s, n in entries.items()
            )
            + f"</listOf{tag}>"
        )

    reactions = "".join(
        f'<reaction id="{identity}" reversible="false">{side("Reactants", left)}'
        f'{side("Products", right)}<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"{law}</math></kineticLaw></reaction>"
        for identity, left, right, law in channels
    )
    xml = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        f'<model id="finite_time_{kind}" timeUnits="clock" substanceUnits="amount" '
        'extentUnits="amount" '
        f'volumeUnits="litre"><listOfUnitDefinitions>{units}</listOfUnitDefinitions>'
        '<listOfCompartments><compartment id="V" size="1" units="litre" '
        'spatialDimensions="3" constant="true"/></listOfCompartments>'
        f"<listOfSpecies>{species}</listOfSpecies><listOfParameters>{params}</listOfParameters>"
        f"<listOfReactions>{reactions}</listOfReactions></model></sbml>"
    )
    return SBMLModel(xml, source())


def cases():
    for kind in ("closed", "nonnormal", "growth", "nonlinear"):
        for scaled in (False, True):
            for metric in ("equal", "unequal"):
                names = (
                    ("X", "Y")
                    if kind == "nonnormal"
                    else ("X",)
                    if kind == "growth"
                    else ("A", "B")
                )
                values = (1.0, 1.0) if metric == "equal" else (0.2, 2.0)
                scales = dict(zip(names, values[: len(names)], strict=True))
                yield f"{kind}-{int(scaled)}-{metric}", kind, scaled, scales
