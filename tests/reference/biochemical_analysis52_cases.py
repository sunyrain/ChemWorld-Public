"""Authored SBML formal pools and independent closed-form analysis references."""

from __future__ import annotations

from itertools import product

import numpy as np

from chemworld.data.sbml import SBMLModel
from chemworld.data.scientific_inputs import InputSource
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.sbml_kinetics import SBMLKinetics

AVOGADRO = 6.02214076e23


def expression(op, *args):
    return f"<apply><{op}/>" + "".join(args) + "</apply>"


def symbol(name):
    return f"<ci>{name}</ci>"


def build_model(
    kind="birth", *, j=0.2, k1=0.5, k2=1.5, volume=1.0, total=0.8, scaled=False, zero=False
):
    amount_scale, volume_scale, time_scale = (0.001, 0.001, 60) if scaled else (1, 1, 1)
    if kind == "closed":
        names, initial = ("A", "B"), (0.7 * total, 0.3 * total)
        reactions = (
            (
                "forward",
                {"A": 1},
                {"B": 1},
                expression("times", symbol("V"), symbol("k1"), symbol("A")),
            ),
            (
                "backward",
                {"B": 1},
                {"A": 1},
                expression("times", symbol("V"), symbol("k2"), symbol("B")),
            ),
        )
    elif kind == "unstable":
        names, initial = ("X",), (k1 * volume / 2,)
        reactions = (
            (
                "autocatalysis",
                {"X": 2},
                {"X": 3},
                expression(
                    "times",
                    symbol("V"),
                    symbol("a"),
                    expression("power", symbol("X"), "<cn>2</cn>"),
                ),
            ),
            ("loss", {"X": 1}, {}, expression("times", symbol("V"), symbol("k1"), symbol("X"))),
        )
    else:
        names = ("X", "Y") if kind == "cascade" else ("X",)
        initial = (
            tuple(0.3 * j * volume / k for k in (k1, k2)[: len(names)]) if not zero else (0.2,)
        )
        reactions = [
            ("inflow", {}, {"X": 1}, expression("times", symbol("V"), symbol("j"))),
            (
                "transfer" if kind == "cascade" else "loss",
                {"X": 1},
                {"Y": 1} if kind == "cascade" else {},
                expression("times", symbol("V"), symbol("k1"), symbol("X")),
            ),
        ]
        if kind == "cascade":
            reactions.append(
                ("loss", {"Y": 1}, {}, expression("times", symbol("V"), symbol("k2"), symbol("Y")))
            )
    base_units = (
        ("amount_unit", "mole", -3 if scaled else 0, 1),
        ("volume_unit", "litre", -3 if scaled else 0, 1),
        ("time_unit", "second", 0, time_scale),
    )
    units = "".join(
        f'<unitDefinition id="{name}"><listOfUnits><unit kind="{base}" '
        f'exponent="1" scale="{scale}" multiplier="{mult}"/></listOfUnits></unitDefinition>'
        for name, base, scale, mult in base_units
    )
    units += (
        f'<unitDefinition id="inverse_time"><listOfUnits><unit kind="second" exponent="-1" '
        f'scale="0" multiplier="{time_scale}"/></listOfUnits></unitDefinition>'
    )
    units += (
        f'<unitDefinition id="concentration_rate"><listOfUnits>'
        f'<unit kind="mole" exponent="1" scale="{-3 if scaled else 0}" multiplier="1"/>'
        f'<unit kind="litre" exponent="-1" scale="{-3 if scaled else 0}" multiplier="1"/>'
        f'<unit kind="second" exponent="-1" scale="0" multiplier="{time_scale}"/>'
        f"</listOfUnits></unitDefinition>"
    )
    units += (
        f'<unitDefinition id="second_order"><listOfUnits>'
        f'<unit kind="mole" exponent="-1" scale="{-3 if scaled else 0}" multiplier="1"/>'
        f'<unit kind="litre" exponent="1" scale="{-3 if scaled else 0}" multiplier="1"/>'
        f'<unit kind="second" exponent="-1" scale="0" multiplier="{time_scale}"/>'
        f"</listOfUnits></unitDefinition>"
    )
    species = "".join(
        f'<species id="{s}" compartment="V" initialAmount="{n / amount_scale}" '
        'hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false"/>'
        for s, n in zip(names, initial, strict=True)
    )
    parameters = "".join(
        f'<parameter id="{name}" value="{value}" units="{unit}" constant="true"/>'
        for name, value, unit in (
            ("j", j * time_scale * volume_scale / amount_scale, "concentration_rate"),
            ("k1", k1 * time_scale, "inverse_time"),
            ("k2", k2 * time_scale, "inverse_time"),
            ("a", 2 * time_scale * amount_scale / volume_scale, "second_order"),
        )
    )

    def side(tag, items):
        return (
            f"<listOf{tag}>"
            + "".join(
                f'<speciesReference species="{s}" stoichiometry="{n}" constant="true"/>'
                for s, n in items.items()
            )
            + f"</listOf{tag}>"
            if items
            else ""
        )

    chemistry = "".join(
        f'<reaction id="{name}" reversible="false">{side("Reactants", left)}'
        f'{side("Products", right)}<kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">'
        f"{law}</math></kineticLaw></reaction>"
        for name, left, right, law in reactions
    )
    text = (
        '<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">'
        f'<model id="{kind}" timeUnits="time_unit" substanceUnits="amount_unit" '
        f'extentUnits="amount_unit" volumeUnits="volume_unit"><listOfUnitDefinitions>{units}'
        f'</listOfUnitDefinitions><listOfCompartments><compartment id="V" spatialDimensions="3" '
        f'size="{volume / volume_scale}" units="volume_unit" constant="true"/></listOfCompartments>'
        f"<listOfSpecies>{species}</listOfSpecies><listOfParameters>{parameters}</listOfParameters>"
        f"<listOfReactions>{chemistry}</listOfReactions></model></sbml>"
    )
    source = InputSource(
        DataSourceProvenance(
            "synthetic-analysis52",
            0,
            "Authored formal pools; chemical composition and caloric data unknown",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )
    return SBMLModel(text, source)


def analytic_case(kind, *, j=0.2, k1=0.5, k2=1.5, volume=1.0, total=0.8):
    incoming = j * volume
    if kind == "birth":
        amounts = np.array([incoming / k1])
        return {
            "amounts": amounts,
            "eigenvalues": [-k1],
            "jacobian": [[-k1]],
            "scaled_amount_control": [[1, -1]],
            "scaled_flux_control": [[1, 0], [1, 0]],
            "covariance": np.diag(amounts / AVOGADRO),
        }
    if kind == "cascade":
        amounts = np.array([incoming / k1, incoming / k2])
        return {
            "amounts": amounts,
            "eigenvalues": sorted([-k1, -k2]),
            "jacobian": [[-k1, 0], [k1, -k2]],
            "scaled_amount_control": [[1, -1, 0], [1, 0, -1]],
            "scaled_flux_control": [[1, 0, 0]] * 3,
            "covariance": np.diag(amounts / AVOGADRO),
        }
    pa, pb = k2 / (k1 + k2), k1 / (k1 + k2)
    return {
        "amounts": np.array([total * pa, total * pb]),
        "eigenvalues": [-k1 - k2],
        "jacobian": [[-k1, k2], [k1, -k2]],
        "scaled_amount_control": [[-pb, pb], [pa, -pa]],
        "scaled_flux_control": [[pa, pb]] * 2,
        "covariance": total * pa * pb / AVOGADRO * np.array([[1, -1], [-1, 1]]),
    }


def fixed_cases():
    for j, k1, volume in product((0.2, 0.8), (0.5, 2.0), (0.5, 2.0)):
        yield "birth", {"j": j, "k1": k1, "volume": volume}
    for k1, k2, volume in product((0.5, 2.0), (0.25, 1.5), (0.5, 2.0)):
        yield "cascade", {"j": 0.3, "k1": k1, "k2": k2, "volume": volume}
    for k1, k2, total in product((0.5, 2.0), (0.25, 1.5), (0.2, 0.8)):
        yield "closed", {"k1": k1, "k2": k2, "total": total}


def frame_case(kind="birth", **kwargs):
    frame = SBMLKinetics(build_model(kind, **kwargs))
    return frame, dict(frame.initial_values)
