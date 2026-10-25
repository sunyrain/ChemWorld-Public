"""Full substrate/product SBML, with unchanged native chemistry and source controls."""

from math import prod
from xml.etree.ElementTree import SubElement, fromstring, tostring

from integer51h_distribution_cases import STRATEGIES
from sde51e_cases import AVOGADRO, source_sbml

SBML = "{http://www.sbml.org/sbml/level3/version2/core}"
BASE_SEEDS = dict(zip(STRATEGIES, (553600000, 563600000, 573600000, 583600000), strict=True))


def job(unit, strategy):
    reactions, chemistry, fluxes = [], {}, {}
    for r in unit["reactions"]:
        order = sum(r["orders"].values())
        reactions.append(
            {
                "id": r["id"],
                "orders": r["orders"],
                "stoichiometry": r["changes"],
                "k": r["coefficient"] * (AVOGADRO * 1e-15) ** (order - 1),
            }
        )
        products = {
            k: r["orders"].get(k, 0) + r["changes"].get(k, 0)
            for k in set(r["orders"]) | set(r["changes"])
        }
        if min(products.values()) < 0:
            raise ValueError("source full reactants do not cover consumption")
        chemistry[r["id"]] = {
            "substrates": r["orders"],
            "products": {k: n for k, n in products.items() if n},
        }
        fluxes[r["id"]] = r["coefficient"] * prod(
            unit["initial"][k] ** n for k, n in r["orders"].items()
        )
    tree = fromstring(
        source_sbml({"id": unit["id"], "initial": unit["initial"], "reactions": reactions})
    )
    tree.find(".//" + SBML + "compartment").set("size", "1e-15")
    nodes = {r.get("id"): r for r in tree.findall(".//" + SBML + "reaction")}
    for name, terms in chemistry.items():
        for key, tag in (("substrates", "listOfReactants"), ("products", "listOfProducts")):
            group = nodes[name].find(SBML + tag)
            group.clear()
            for species, multiplicity in sorted(terms[key].items()):
                SubElement(
                    group,
                    SBML + "speciesReference",
                    species=species,
                    stoichiometry=str(multiplicity),
                    constant="true",
                )
    return {
        "unit": unit["id"],
        "strategy": strategy,
        "method_enum": {"direct": 25, "next": 26}.get(strategy, 27),
        "epsilon": 0.1 if strategy == "coarse" else 0.01 if strategy == "fine" else None,
        "initial": unit["initial"],
        "volume_L": 1e-15,
        "initial_fluxes": fluxes,
        "chemistry": chemistry,
        "duration_s": unit["duration_s"],
        "output_steps": 24,
        "sbml": tostring(tree, encoding="unicode"),
    }


def jobs(unit, strategy, mode):
    seeds = (
        (1, 17, 37, 17)
        if mode == "wiring"
        else [17]
        if unit["frozen"]
        else [BASE_SEEDS[strategy] + 100000 * unit["index"] + i for i in range(unit["replicates"])]
    )
    common = job(unit, strategy)
    return [
        {**common, "id": f"{unit['id']}-{strategy}-{i}", "seed": seed}
        for i, seed in enumerate(seeds)
    ]
