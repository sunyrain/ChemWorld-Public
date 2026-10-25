"""Fixed statistics inputs and independent scalar RI5 moment algebra."""

import math
from xml.etree.ElementTree import fromstring, tostring

import numpy as np
from sde51e_cases import AVOGADRO, VOLUME_L, source_sbml

REPLICATES = 1024
CDF_LIMIT = math.sqrt(2 * math.log(4 * 100 / 0.01) / REPLICATES)
EPSILON = 1e-8


def cases():
    result = []

    def add(kind, initial, reactions, steps, duration=1.0, **extra):
        for h in steps:
            result.append(
                {
                    "id": f"S{len(result) + 1:02d}",
                    "kind": kind,
                    "initial": initial,
                    "reactions": reactions,
                    "step_s": h,
                    "duration_s": duration,
                    "noise": None,
                    "replicates": REPLICATES,
                    **extra,
                }
            )

    forward = {"id": "forward", "stoichiometry": {"A": -1, "B": 1}, "orders": {"A": 1}}
    reverse = {"id": "reverse", "stoichiometry": {"A": 1, "B": -1}, "orders": {"B": 1}}
    grid = [0.5, 0.25, 0.125, 0.0625]
    add("ou", {"A": 51.0, "B": 49.0}, [{**forward, "k": 0.0}], grid, noise=[2.0])
    add(
        "gbm",
        {"A": 1.0, "B": 9999.0},
        [{**forward, "k": 0.0}],
        grid,
        noise=[{"population": "A", "coefficient": 0.25}],
    )
    fine = [1 / 16, 1 / 64]
    add(
        "high_boundary",
        {"A": 1000.0, "B": 0.0},
        [{**forward, "k": 0.1}, {**reverse, "k": 0.2}],
        fine,
    )
    add(
        "interior_cle",
        {"A": 500.0, "B": 500.0},
        [{**forward, "k": 0.2}, {**reverse, "k": 0.2}],
        fine,
    )
    add("death", {"A": 1.0, "B": 0.0}, [{**forward, "k": 1.0}], fine)
    add(
        "birth_death",
        {"A": 0.0, "B": 4.0},
        [{**forward, "k": 0.4}, {**reverse, "k": 0.2}],
        fine,
    )
    add(
        "enzyme",
        {"E": 5.0, "S": 90.0, "ES": 5.0, "P": 5.0},
        [
            {
                "id": "binding",
                "stoichiometry": {"E": -1, "S": -1, "ES": 1},
                "orders": {"E": 1, "S": 1},
                "k": 2e7,
            },
            {
                "id": "unbinding",
                "stoichiometry": {"ES": -1, "E": 1, "S": 1},
                "orders": {"ES": 1},
                "k": 20.0,
            },
            {
                "id": "catalysis",
                "stoichiometry": {"ES": -1, "E": 1, "P": 1},
                "orders": {"ES": 1},
                "k": 0.1,
            },
        ],
        [1 / 32, 1 / 128],
        0.25,
    )
    add(
        "third_order",
        {"A": 80.0, "B": 90.0, "C": 10.0},
        [
            {
                "id": "association",
                "stoichiometry": {"A": -2, "B": -1, "C": 1},
                "orders": {"A": 2, "B": 1},
                "k": 1e13,
            }
        ],
        fine,
        0.25,
    )
    add(
        "zero",
        {"A": 100.5, "B": 20.25},
        [{**forward, "k": 1.0}],
        [0.25, 0.0625],
        noise=[0.0],
        replicates=1,
    )
    add(
        "ou_dense",
        {"A": 51.0, "B": 49.0},
        [{**forward, "k": 0.0}],
        [0.5, 0.0625],
        23 / 32,
        noise=[2.0],
    )
    assert len(result) == 24 and sum(c["replicates"] for c in result) == 22530
    return result


def fluxes(case):
    if case["kind"].startswith("ou"):
        return [case["initial"]["A"] - 50]
    if case["kind"] == "gbm":
        return [-0.2 * case["initial"]["A"]]
    return [
        r["k"]
        * math.prod(case["initial"][key] ** order for key, order in r["orders"].items())
        / (AVOGADRO * VOLUME_L) ** (sum(r["orders"].values()) - 1)
        for r in case["reactions"]
    ]


def amplitudes(case):
    if case["noise"] is None:
        return [math.sqrt(abs(f)) for f in fluxes(case)]
    return [
        value["coefficient"] * case["initial"][value["population"]]
        if isinstance(value, dict)
        else value
        for value in case["noise"]
    ]


def pools(case):
    if case["kind"] == "enzyme":
        return [{"E": 1, "ES": 1}, {"S": 1, "ES": 1, "P": 1}]
    if case["kind"] == "third_order":
        return [{"A": 1, "C": 2}, {"B": 1, "C": 1}]
    return [{"A": 1, "B": 1}]


def native_model(case):
    from chemworld.foundation.dimensions import DimensionVector as D
    from chemworld.physchem import ReactionFlux, ReactionNoise, SDEParameter, StochasticSDESpec
    from chemworld.physchem.reaction_network import (
        RateLawSpec,
        ReactionNetworkSpec,
        ReactionSpec,
        SpeciesSpec,
    )

    formulas = (
        {"E": "N2", "S": "CH2O", "ES": "CH2N2O", "P": "CH2O"}
        if case["kind"] == "enzyme"
        else {"A": "H2", "B": "O2", "C": "H4O2"}
        if case["kind"] == "third_order"
        else {"A": "CH2O", "B": "CH2O"}
    )
    reactions = []
    for reaction in case["reactions"]:
        parts = []
        for sign in (-1, 1):
            terms = [
                (f"{abs(value)} " if abs(value) != 1 else "") + key
                for key, value in reaction["stoichiometry"].items()
                if sign * value > 0
            ]
            parts.append(" + ".join(terms))
        reactions.append(
            ReactionSpec.from_equation(
                reaction_id=reaction["id"],
                equation=" => ".join(parts),
                rate_law=RateLawSpec(reaction["id"], "mass_action", {"k": reaction["k"]}),
            )
        )
    network = ReactionNetworkSpec(
        "statistics-" + case["id"],
        tuple(SpeciesSpec(key, formulas[key]) for key in case["initial"]),
        tuple(reactions),
    )

    def mathml(body):
        return '<math xmlns="http://www.w3.org/1998/Math/MathML">' + body + "</math>"

    overrides = ()
    if case["kind"].startswith("ou"):
        overrides = (
            ReactionFlux(
                "forward",
                mathml(
                    "<apply><times/><ci>mu</ci>"
                    "<apply><minus/><ci>n0</ci><ci>center</ci></apply></apply>"
                ),
                (SDEParameter("mu", 1.0, D(time=-1)), SDEParameter("center", 50.0, D())),
            ),
        )
    elif case["kind"] == "gbm":
        overrides = (
            ReactionFlux(
                "forward",
                mathml("<apply><minus/><apply><times/><ci>mu</ci><ci>n0</ci></apply></apply>"),
                (SDEParameter("mu", 0.2, D(time=-1)),),
            ),
        )
    noises = None
    if case["kind"] == "zero":
        noises = ()
    elif case["noise"] is not None:
        noises = tuple(
            ReactionNoise(
                r["id"],
                mathml(
                    "<apply><times/><ci>sigma</ci><ci>n0</ci></apply>"
                    if isinstance(value, dict)
                    else "<ci>sigma</ci>"
                ),
                (
                    SDEParameter(
                        "sigma",
                        value["coefficient"] if isinstance(value, dict) else value,
                        D(time=-0.5),
                    ),
                ),
            )
            for r, value in zip(case["reactions"], case["noise"], strict=True)
        )
    return network, StochasticSDESpec(
        VOLUME_L,
        step_s=case["step_s"],
        output_policy=case.get("output_policy", "linear"),
        noises=noises,
        fluxes=overrides,
    )


def sbml(case):
    xml = source_sbml(case)
    if not (case["kind"].startswith("ou") or case["kind"] == "gbm"):
        return xml
    root = fromstring(xml)
    # ElementTree prefixes are harmless to SBML; manipulate explicit namespace.
    ns = "{http://www.sbml.org/sbml/level3/version2/core}"
    model = root.find(ns + "model")
    parameters = model.find(ns + "listOfParameters")
    from xml.etree.ElementTree import SubElement

    SubElement(
        parameters,
        ns + "parameter",
        id="mu",
        constant="true",
        value="0.2" if case["kind"] == "gbm" else "1",
        units="k_order_1",
    )
    SubElement(parameters, ns + "parameter", id="center", constant="true", value="50", units="item")
    reaction = model.find(ns + "listOfReactions/" + ns + "reaction")
    reaction.set("reversible", "true")
    law = reaction.find(ns + "kineticLaw")
    for child in list(law):
        law.remove(child)
    expression = (
        "<apply><minus/><apply><times/><ci>mu</ci><ci>A</ci></apply></apply>"
        if case["kind"] == "gbm"
        else "<apply><times/><ci>mu</ci><apply><minus/><ci>A</ci><ci>center</ci></apply></apply>"
    )
    law.append(
        fromstring('<math xmlns="http://www.w3.org/1998/Math/MathML">' + expression + "</math>")
    )
    return tostring(root, encoding="unicode")


def job(case, seed):
    return {
        "id": f"{case['id']}-{seed:04d}",
        "case": case["id"],
        "seed": seed,
        "step_s": case["step_s"],
        "duration_s": case["duration_s"],
        "output_steps": 1,
        "species": list(case["initial"]),
        "sbml": sbml(case),
        "noise": {
            r["id"]: None if case["noise"] is None else case["noise"][i]
            for i, r in enumerate(case["reactions"])
        },
        "preserve_model": seed == 1,
        "gate_expected": {
            "initial": case["initial"],
            "fluxes": dict(zip((r["id"] for r in case["reactions"]), fluxes(case), strict=True)),
            "noise": dict(zip((r["id"] for r in case["reactions"]), amplitudes(case), strict=True)),
        },
    }


def r3(z):
    return 1 + z + z * z / 2 + z * z * z / 6


def operators(case, engine):
    h, duration = case["step_s"], case["duration_s"]
    full = math.floor(duration / h)
    result = [(h, 1.0)] * full
    remainder = duration - full * h
    if remainder:
        linear = engine == "native" and case.get("output_policy", "linear") == "linear"
        result.append((h, remainder / h) if linear else (remainder, 1.0))
    return result


def moments(case, engine):
    """Independent finite-support RI5 affine/multiplicative recurrences, not a solver."""
    if case["kind"].startswith("ou"):
        mean, variance, low, high = case["initial"]["A"] - 50, 0.0, 1.0, 1.0
        for h, theta in operators(case, engine):
            z = -h
            a = 1 - theta + theta * r3(z)
            b = theta * 2 * math.sqrt(h) * (1 + z / 2 - z * z / 18)
            mean, variance = a * mean, a * a * variance + b * b
            low, high = (
                min(a * low, a * high) - abs(b) * math.sqrt(3),
                (max(a * low, a * high) + abs(b) * math.sqrt(3)),
            )
        return {
            "mean": 50 + mean,
            "variance": variance,
            "central_fourth": 3 * variance**2,
            "low": 50 + low,
            "high": 50 + high,
        }
    if case["kind"] != "gbm":
        return None
    raw = np.ones(4) * case["initial"]["A"] ** np.arange(1, 5)
    low = high = case["initial"]["A"]
    for h, theta in operators(case, engine):
        if theta != 1:
            raise ValueError("GBM dense not part of this fixed design")
        z, sigma = 0.2 * h, 0.25
        values = [
            r3(z)
            + sigma * math.sqrt(h) * xi * (1 + z - z * z / 18)
            + 0.5 * sigma**2 * h * (xi * xi - 1)
            for xi in (-math.sqrt(3), 0.0, math.sqrt(3))
        ]
        if min(values) <= 0:
            raise ValueError("finite GBM support is not positive")
        raw *= [
            sum(p * v**r for p, v in zip((1 / 6, 2 / 3, 1 / 6), values, strict=True))
            for r in range(1, 5)
        ]
        low, high = low * min(values), high * max(values)
    mean, second, third, fourth = map(float, raw)
    variance = second - mean**2
    return {
        "mean": mean,
        "variance": variance,
        "central_fourth": fourth - 4 * mean * third + 6 * mean**2 * second - 3 * mean**4,
        "low": low,
        "high": high,
    }


def continuous_moments(case):
    t = case["duration_s"]
    if case["kind"].startswith("ou"):
        return {"mean": 50 + math.exp(-t), "variance": 2 * (1 - math.exp(-2 * t))}
    return {"mean": math.exp(0.2 * t), "variance": math.exp(0.4 * t) * math.expm1(0.25**2 * t)}


def moment_checks(values, theory):
    values = np.array(values)
    n, mu, var = len(values), theory["mean"], theory["variance"]
    log = math.log(2 * 100 / 0.01)
    bound = max(mu - theory["low"], theory["high"] - mu)
    mean_width = math.sqrt(2 * var * log / n) + (2 / 3) * (2 * bound) * log / n
    variance_width = math.sqrt(2 * (theory["central_fourth"] - var**2) * log / n)
    variance_width += (2 / 3) * bound**2 * log / n
    observed_mean = float(values.mean())
    observed_second = float(np.mean((values - mu) ** 2))
    return {
        "passed": abs(observed_mean - mu) <= mean_width
        and abs(observed_second - var) <= variance_width,
        "observed_mean": observed_mean,
        "observed_centered_second": observed_second,
        "mean_width": mean_width,
        "second_width": variance_width,
        "theory": theory,
    }


def cdf_difference(left, right, epsilon=0.0):
    left, right = np.sort(left), np.sort(right)
    points = np.union1d(left, right)

    def one(a, b):
        return np.max(
            np.searchsorted(a, points, side="right") / len(a)
            - np.searchsorted(b, points + epsilon, side="right") / len(b)
        )

    return float(max(one(left, right), one(right, left)))
