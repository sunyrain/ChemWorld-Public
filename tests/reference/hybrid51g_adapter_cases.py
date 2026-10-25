"""Fixed same-network inputs and particle-coordinate SBML for the method-31 adapter."""

from dataclasses import replace
from math import prod
from xml.etree.ElementTree import SubElement, fromstring, tostring

from cross_partition51_cases import CASES as CROSS_CASES
from cross_partition51_cases import author as cross_author
from sde51e_cases import AVOGADRO, source_sbml
from stochastic_hybrid51_cases import CASES as FEEDBACK_CASES
from stochastic_hybrid51_cases import author as feedback_author

SBML = "{http://www.sbml.org/sbml/level3/version2/core}"
UNITS = (
    {
        "id": "G01",
        "kind": "feedback",
        "ode": True,
        "seeds": [17],
        "parameters": {**FEEDBACK_CASES[0], "alpha": 0.0, "beta": 0.0, "k1": 0.0},
    },
    {
        "id": "G02",
        "kind": "cross",
        "ode": True,
        "seeds": [17],
        "parameters": {**CROSS_CASES[0], "release": 0.0, "capture": 0.0},
    },
    {
        "id": "G03",
        "kind": "feedback",
        "ode": False,
        "seeds": [1, 17, 37, 17],
        "parameters": FEEDBACK_CASES[0],
    },
    {
        "id": "G04",
        "kind": "cross",
        "ode": False,
        "seeds": [1, 17, 37, 17],
        "parameters": CROSS_CASES[0],
    },
)


def author(unit):
    factory = feedback_author if unit["kind"] == "feedback" else cross_author
    network, spec, amounts = factory(unit["parameters"])
    return network, replace(spec, ode_method="RK45"), amounts


def jobs(unit):
    network, spec, amounts = author(unit)
    initial = {key: value * AVOGADRO for key, value in amounts.items()}
    reactions = []
    for reaction in network.reactions:
        orders = dict(reaction.kinetic_forward_orders)
        if any(value != int(value) for value in orders.values()):
            raise ValueError("fixed adapter input requires integer kinetic orders")
        reactions.append(
            {
                "id": reaction.reaction_id,
                "stoichiometry": dict(reaction.stoichiometry),
                "orders": {key: int(value) for key, value in orders.items()},
                "k": float(reaction.rate_law.parameters["k"]),
            }
        )
    tree = fromstring(source_sbml({"id": unit["id"], "initial": initial, "reactions": reactions}))
    tree.find(".//" + SBML + "compartment").set("size", str(spec.volume_L))
    nodes = {node.get("id"): node for node in tree.findall(".//" + SBML + "reaction")}
    for reaction in reactions:
        modifiers = set(reaction["orders"]) - {
            key for key, value in reaction["stoichiometry"].items() if value
        }
        if modifiers:
            group = SubElement(nodes[reaction["id"]], SBML + "listOfModifiers")
            node = nodes[reaction["id"]]
            node.remove(group)
            node.insert(list(node).index(node.find(SBML + "kineticLaw")), group)
            for key in sorted(modifiers):
                SubElement(group, SBML + "modifierSpeciesReference", species=key)
    fluxes = {
        r["id"]: r["k"]
        * prod(initial[key] ** order for key, order in r["orders"].items())
        / (AVOGADRO * spec.volume_L) ** (sum(r["orders"].values()) - 1)
        for r in reactions
    }
    common = {
        "case": unit["id"],
        "duration_s": 3.0,
        "sbml": tostring(tree, encoding="unicode"),
        "volume_L": spec.volume_L,
        "initial": initial,
        "initial_fluxes": fluxes,
        "deterministic_reactions": [
            r.reaction_id
            for r in network.reactions
            if r.reaction_id not in spec.stochastic_reactions
        ],
    }
    return [
        {**common, "id": f"{unit['id']}-{index:02d}", "seed": seed}
        for index, seed in enumerate(unit["seeds"], 1)
    ]
