"""Explicit inputs shared as data; independent particle-coordinate SBML writer."""

from math import prod
from xml.etree.ElementTree import Element, SubElement, tostring

AVOGADRO = 6.02214076e23
VOLUME_L = 1e-15
STEP_S = 0.125


def adapter_cases():
    one = [{"id": "forward", "stoichiometry": {"A": -1, "B": 1}, "orders": {"A": 1}, "k": 0.1}]
    return [
        {
            "id": "C01",
            "family": "reversible",
            "initial": {"A": 100.5, "B": 20.25},
            "reactions": one,
            "noise": [0.0],
            "duration_s": 0.5,
            "seeds": [17],
        },
        {
            "id": "C02",
            "family": "reversible",
            "initial": {"A": 100.5, "B": 20.25},
            "reactions": [{**one[0], "k": 0.0}],
            "noise": [2.0],
            "duration_s": STEP_S,
            "seeds": list(range(1, 33)),
        },
        {
            "id": "C03",
            "family": "reversible",
            "initial": {"A": 100.5, "B": 20.25},
            "reactions": [
                *one,
                {"id": "reverse", "stoichiometry": {"A": 1, "B": -1}, "orders": {"B": 1}, "k": 0.2},
            ],
            "noise": None,
            "duration_s": 0.5,
            "seeds": [17],
        },
        {
            "id": "C04",
            "family": "enzyme",
            "initial": {"E": 10.5, "S": 100.25, "ES": 3.25, "P": 5.5},
            "reactions": [
                {
                    "id": "binding",
                    "stoichiometry": {"E": -1, "S": -1, "ES": 1},
                    "orders": {"E": 1, "S": 1},
                    "k": 1e6,
                },
                {
                    "id": "unbinding",
                    "stoichiometry": {"ES": -1, "E": 1, "S": 1},
                    "orders": {"ES": 1},
                    "k": 1.0,
                },
                {
                    "id": "catalysis",
                    "stoichiometry": {"ES": -1, "E": 1, "P": 1},
                    "orders": {"ES": 1},
                    "k": 0.1,
                },
            ],
            "noise": None,
            "duration_s": STEP_S,
            "seeds": [17],
        },
    ]


def expected_fluxes(case):
    return [
        r["k"]
        * prod(case["initial"][key] ** order for key, order in r["orders"].items())
        / (AVOGADRO * VOLUME_L) ** (sum(r["orders"].values()) - 1)
        for r in case["reactions"]
    ]


def native_model(case):
    from dataclasses import replace

    from chemworld.foundation.dimensions import DimensionVector
    from chemworld.physchem import ReactionNoise, SDEParameter, StochasticSDESpec
    from chemworld.world.biochemical_worlds import biochemical_network

    network = biochemical_network(case["family"])
    originals = {r.reaction_id: r for r in network.reactions}
    reactions = tuple(
        replace(
            originals[r["id"]],
            rate_law=replace(originals[r["id"]].rate_law, parameters={"k": r["k"]}),
        )
        for r in case["reactions"]
    )
    network = replace(network, stochastic=None, reactions=reactions)
    noises = (
        None
        if case["noise"] is None
        else tuple(
            ReactionNoise(
                r["id"],
                '<math xmlns="http://www.w3.org/1998/Math/MathML"><ci>sigma</ci></math>',
                (SDEParameter("sigma", amplitude, DimensionVector(time=-0.5)),),
            )
            for r, amplitude in zip(case["reactions"], case["noise"], strict=True)
        )
    )
    return network, StochasticSDESpec(VOLUME_L, step_s=STEP_S, noises=noises)


def source_sbml(case):
    root = Element(
        "sbml", xmlns="http://www.sbml.org/sbml/level3/version2/core", level="3", version="2"
    )
    model = SubElement(
        root,
        "model",
        id="sde_" + case["id"],
        name="sde_" + case["id"],
        substanceUnits="item",
        extentUnits="item",
        timeUnits="second",
        volumeUnits="litre",
    )
    units = SubElement(model, "listOfUnitDefinitions")
    for order in sorted({sum(r["orders"].values()) for r in case["reactions"]}):
        definition = SubElement(units, "unitDefinition", id="k_order_" + str(order))
        components = SubElement(definition, "listOfUnits")
        for kind, exponent in [("second", -1), ("litre", order - 1), ("item", 1 - order)]:
            if exponent:
                SubElement(
                    components, "unit", kind=kind, exponent=str(exponent), scale="0", multiplier="1"
                )
    compartments = SubElement(model, "listOfCompartments")
    SubElement(
        compartments,
        "compartment",
        id="cell",
        name="cell",
        spatialDimensions="3",
        size=str(VOLUME_L),
        units="litre",
        constant="true",
    )
    species = SubElement(model, "listOfSpecies")
    for name, value in case["initial"].items():
        SubElement(
            species,
            "species",
            id=name,
            name=name,
            compartment="cell",
            initialAmount=str(value),
            substanceUnits="item",
            hasOnlySubstanceUnits="true",
            boundaryCondition="false",
            constant="false",
        )
    parameters = SubElement(model, "listOfParameters")
    for index, reaction in enumerate(case["reactions"]):
        order = sum(reaction["orders"].values())
        SubElement(
            parameters,
            "parameter",
            id="k" + str(index),
            constant="true",
            value=str(reaction["k"] / AVOGADRO ** (order - 1)),
            units="k_order_" + str(order),
        )
    reactions = SubElement(model, "listOfReactions")
    for index, reaction in enumerate(case["reactions"]):
        node = SubElement(
            reactions, "reaction", id=reaction["id"], name=reaction["id"], reversible="false"
        )
        for sign, tag in [(-1, "listOfReactants"), (1, "listOfProducts")]:
            terms = SubElement(node, tag)
            for key, value in reaction["stoichiometry"].items():
                if sign * value > 0:
                    SubElement(
                        terms,
                        "speciesReference",
                        species=key,
                        stoichiometry=str(abs(value)),
                        constant="true",
                    )
        law = SubElement(node, "kineticLaw")
        math = SubElement(law, "math", xmlns="http://www.w3.org/1998/Math/MathML")
        order = sum(reaction["orders"].values())
        target = math
        if order > 1:
            target = SubElement(math, "apply")
            SubElement(target, "divide")
        numerator = SubElement(target, "apply")
        SubElement(numerator, "times")
        SubElement(numerator, "ci").text = "k" + str(index)
        for key, power in reaction["orders"].items():
            if power == 1:
                SubElement(numerator, "ci").text = key
            else:
                exponent = SubElement(numerator, "apply")
                SubElement(exponent, "power")
                SubElement(exponent, "ci").text = key
                SubElement(exponent, "cn").text = str(power)
        if order > 1:
            exponent = SubElement(target, "apply")
            SubElement(exponent, "power")
            SubElement(exponent, "ci").text = "cell"
            SubElement(exponent, "cn").text = str(order - 1)
    return tostring(root, encoding="unicode")
