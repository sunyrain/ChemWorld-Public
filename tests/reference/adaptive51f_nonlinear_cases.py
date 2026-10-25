"""Fixed nonlinear physical networks and strict particle-coordinate source authoring."""

import math
from xml.etree.ElementTree import SubElement, fromstring, tostring

from adaptive51f_distribution_cases import AVOGADRO, VOLUME_L, reaction
from sde51e_cases import source_sbml

SBML = "{http://www.sbml.org/sbml/level3/version2/core}"


def cases():
    association = reaction("association", {"A": -2, "B": 1}, {"A": 2}, 0.003)
    exchange = [
        reaction("capture", {"A": -1, "C": 1}, {"A": 1}, 0.05),
        reaction("release", {"A": 1, "C": -1}, {"C": 1}, 0.2),
    ]
    autocatalytic = reaction("autocatalytic", {"A": 1, "B": -1}, {"A": 1, "B": 1}, 0.002)
    enzyme = [
        reaction("binding", {"E": -1, "S": -1, "ES": 1}, {"E": 1, "S": 1}, 0.001),
        reaction("unbinding", {"E": 1, "S": 1, "ES": -1}, {"ES": 1}, 0.2),
        reaction("catalysis", {"E": 1, "P": 1, "ES": -1}, {"ES": 1}, 0.1),
    ]
    data = [
        ("dimer", {"A": 200.0, "B": 200.0}, [association], 0.0, 0.0, 1.0, True),
        (
            "trimer",
            {"A": 200.0, "B": 200.0},
            [reaction("association", {"A": -3, "B": 1}, {"A": 3}, 3e-6)],
            0.0,
            0.0,
            1.0,
            True,
        ),
        ("logistic", {"A": 100.0, "B": 200.0}, [autocatalytic], 0.0, 0.0, 1.0, True),
        (
            "down",
            {"A": 200.0, "B": 200.0, "C": 0.0},
            [association, *exchange],
            100.0,
            120.0,
            1.0,
            False,
        ),
        (
            "feedback",
            {"A": 200.0, "B": 200.0, "E0": 1.0, "E1": 0.0},
            [
                {**association, "coefficient": 0.001},
                reaction("reverse", {"A": 2, "B": -1}, {"B": 1}, 0.05),
                reaction("activate", {"E0": -1, "E1": 1}, {"A": 1, "E0": 1}, 0.02),
                reaction("deactivate", {"E1": -1, "E0": 1}, {"E1": 1}, 2.0),
                reaction("catalytic", {"A": -2, "B": 1}, {"A": 2, "E1": 1}, 0.0005),
            ],
            20.0,
            40.0,
            1.0,
            False,
        ),
        ("up", {"A": 20.0, "B": 280.0}, [autocatalytic], 20.0, 40.0, 3.0, False),
        ("enzyme", {"E": 100.0, "S": 200.0, "ES": 100.0, "P": 0.0}, enzyme, 20.0, 40.0, 1.0, False),
        (
            "stiff_enzyme",
            {"E": 100.0, "S": 200.0, "ES": 100.0, "P": 0.0},
            [
                {**enzyme[0], "coefficient": 0.05},
                {**enzyme[1], "coefficient": 20.0},
                enzyme[2],
            ],
            20.0,
            40.0,
            1.0,
            False,
        ),
        (
            "fractional_down",
            {"A": 200.25, "B": 200.75, "C": 0.5},
            [association, *exchange],
            100.0,
            120.0,
            1.0,
            False,
        ),
    ]
    result = []
    for family, initial, reactions, lower, upper, duration, pure in data:
        if "enzyme" in family:
            formulas = {"E": "N2", "S": "CH2O", "ES": "CH2ON2", "P": "CH2O"}
            pools = [{"E": 1, "ES": 1}, {"S": 1, "ES": 1, "P": 1}]
        elif family in {"logistic", "up"}:
            formulas = dict.fromkeys(initial, "CH2O")
            pools = [{"A": 1, "B": 1}]
        else:
            order = 3 if family == "trimer" else 2
            formulas = {"A": "CH2O", "B": "C3H6O3" if order == 3 else "C2H4O2"}
            pool = {"A": 1, "B": order}
            if "C" in initial:
                formulas["C"] = "CH2O"
                pool["C"] = 1
            pools = [pool]
            if family == "feedback":
                formulas.update(E0="Cu", E1="Cu")
                pools.append({"E0": 1, "E1": 1})
        steps = (1 / 64, 1 / 256) if family == "stiff_enzyme" else (1 / 16, 1 / 64)
        for method in ("RK4", "LSODA"):
            for level, h in enumerate(steps):
                result.append(
                    {
                        "id": f"N{len(result) + 1:02d}",
                        "index": len(result),
                        "family": family,
                        "initial": initial,
                        "reactions": reactions,
                        "lower": lower,
                        "upper": upper,
                        "duration_s": duration,
                        "pure_ode": pure,
                        "method": method,
                        "step_s": h,
                        "level": level,
                        "formulas": formulas,
                        "pools": pools,
                        "native_replicates": 1 if pure else 1024,
                        "reference_replicates": 1 if pure else 4096,
                        "copasi_replicates": 1 if pure else 1024,
                    }
                )
    assert len(result) == 36
    assert sum(u["native_replicates"] for u in result) == 24588
    assert sum(u["reference_replicates"] for u in result) == 98316
    return result


def source_jobs(unit):
    reactions = [
        {
            "id": r["id"],
            "stoichiometry": r["changes"],
            "orders": r["orders"],
            "k": r["coefficient"] * (AVOGADRO * VOLUME_L) ** (sum(r["orders"].values()) - 1),
        }
        for r in unit["reactions"]
    ]
    tree = fromstring(source_sbml({**unit, "reactions": reactions}))
    nodes = {node.get("id"): node for node in tree.findall(".//" + SBML + "reaction")}
    for r in unit["reactions"]:
        modifiers = set(r["orders"]) - set(r["changes"])
        if modifiers:
            node = nodes[r["id"]]
            group = SubElement(node, SBML + "listOfModifiers")
            node.remove(group)
            node.insert(list(node).index(node.find(SBML + "kineticLaw")), group)
            for key in sorted(modifiers):
                SubElement(group, SBML + "modifierSpeciesReference", species=key)
    common = {
        "method": unit["method"],
        "lower": unit["lower"],
        "upper": unit["upper"],
        "step_s": unit["step_s"],
        "duration_s": unit["duration_s"],
        "volume_L": VOLUME_L,
        "output_steps": round(unit["duration_s"] / unit["step_s"]),
        "initial": unit["initial"],
        "initial_fluxes": {
            r["id"]: r["coefficient"]
            * math.prod(unit["initial"][k] ** n for k, n in r["orders"].items())
            for r in unit["reactions"]
        },
        "sbml": tostring(tree, encoding="unicode"),
    }
    return [
        {
            **common,
            "id": unit["id"] + "-" + str(index),
            "seed": 17 if unit["pure_ode"] else 511800000 + 100000 * unit["index"] + index,
        }
        for index in range(unit["copasi_replicates"])
    ]
