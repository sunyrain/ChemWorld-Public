"""Two original source encodings of the fixed fourteen integer generators."""

from math import prod
from xml.etree.ElementTree import Element, SubElement, fromstring, tostring

from integer51h_distribution_cases import STRATEGIES
from integer51h_source_cases import job

AVOGADRO = 6.02214076e23
ENCODINGS = ("item_amount", "mol_concentration")
BASE_SEEDS = dict(zip(STRATEGIES, (613600000, 623600000, 633600000, 643600000), strict=True))
TRIALS = (
    [(997, "warmup")]
    + [(s, "timed") for s in (1, 7, 17, 37, 73, 101, 251, 509)]
    + [(17, "allocation")]
)
SBML = "{http://www.sbml.org/sbml/level3/version2/core}"
MATH = "{http://www.w3.org/1998/Math/MathML}"


def source_job(unit, encoding, strategy):
    result = job(unit, strategy)
    if encoding not in ENCODINGS:
        raise ValueError("unknown source encoding")
    if encoding == "mol_concentration":
        tree = fromstring(result["sbml"])
        declaration = tree.find(SBML + "model")
        declaration.set("substanceUnits", "mole")
        declaration.set("extentUnits", "mole")
        for term in tree.iter(SBML + "unit"):
            if term.get("kind") == "item":
                term.set("kind", "mole")
        for species in tree.iter(SBML + "species"):
            species.set("substanceUnits", "mole")
            species.set("hasOnlySubstanceUnits", "false")
            species.set("initialAmount", repr(unit["initial"][species.get("id")] / AVOGADRO))
        parameters = {p.get("id"): p for p in tree.iter(SBML + "parameter")}
        reactions = {r.get("id"): r for r in tree.iter(SBML + "reaction")}
        for index, raw in enumerate(unit["reactions"]):
            order = sum(raw["orders"].values())
            parameters["k" + str(index)].set(
                "value", repr(raw["coefficient"] * (AVOGADRO * 1e-15) ** (order - 1))
            )
            law = reactions[raw["id"]].find(SBML + "kineticLaw")
            law.remove(law.find(MATH + "math"))
            math = Element(MATH + "math")
            terms = SubElement(math, MATH + "apply")
            SubElement(terms, MATH + "times")
            SubElement(terms, MATH + "ci").text = "cell"
            SubElement(terms, MATH + "ci").text = "k" + str(index)
            for name, power in raw["orders"].items():
                if power == 1:
                    SubElement(terms, MATH + "ci").text = name
                else:
                    node = SubElement(terms, MATH + "apply")
                    SubElement(node, MATH + "power")
                    SubElement(node, MATH + "ci").text = name
                    SubElement(node, MATH + "cn").text = str(power)
            law.insert(0, math)
        result["sbml"] = tostring(tree, encoding="unicode")
    result.update(
        source_encoding=encoding,
        expected_quantity_to_number=AVOGADRO if encoding == "mol_concentration" else 1.0,
        expected_avogadro=AVOGADRO,
        configure_avogadro=AVOGADRO,
    )
    return result


def author(unit, encoding, strategy):
    from chemworld.data.sbml import SBMLModel
    from chemworld.data.scientific_inputs import InputSource
    from chemworld.physchem.data_conflicts import DataSourceProvenance
    from chemworld.physchem.sbml_system import SBMLSystemSpec
    from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy

    source = InputSource(
        DataSourceProvenance(
            "sbml51i-science",
            0,
            "Self-authored original elementary sources in two unit encodings",
            "1",
            "CC0-1.0",
        ),
        "synthetic",
    )
    parsed = SBMLModel(
        source_job(unit, encoding, strategy)["sbml"], source, items_per_mole=AVOGADRO
    )
    method = {"direct": "gillespie-direct-v1", "next": "gibson-bruck-next-reaction-v1"}.get(
        strategy, "adaptive-explicit-tau-leap-v1"
    )
    policy = (
        None
        if strategy in ("direct", "next")
        else TauLeapPolicy(
            epsilon=0.1 if strategy == "coarse" else 0.01,
            max_tau_s=0.1 if strategy == "coarse" else 0.005,
            ssa_threshold=0.01,
        )
    )
    return SBMLSystemSpec(
        parsed,
        numerical_form="stochastic",
        solver_method=method,
        maximum_events=1000000,
        tau_policy=policy,
    )


def independent_propensities(unit, counts):
    return {
        r["id"]: r["coefficient"]
        * prod(
            prod(counts[k] - j for j in range(n)) if counts[k] >= n else 0
            for k, n in r["orders"].items()
        )
        for r in unit["reactions"]
    }


def seed(unit, strategy, index):
    return 17 if unit["frozen"] else BASE_SEEDS[strategy] + 100000 * unit["index"] + index
