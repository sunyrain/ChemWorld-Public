"""Fixed distribution data; native authoring is lazy and absent from reference imports."""

VOLUME_L = 1e-15
AVOGADRO = 6.02214076e23
NATIVE_REPLICATES = 1024
REFERENCE_REPLICATES = 4096


def reaction(name, changes, orders, coefficient):
    return {"id": name, "changes": changes, "orders": orders, "coefficient": coefficient}


def cases():
    forward = reaction("forward", {"A": -1, "B": 1}, {"A": 1}, 1.0)
    reverse = reaction("reverse", {"A": 1, "B": -1}, {"B": 1}, 0.2)
    pair = [forward, reverse]
    data = [
        (
            "birth_death",
            {"A": 0.0, "B": 4.0},
            [{**forward, "coefficient": 0.4}, reverse],
            1e9,
            1e9,
            2.0,
        ),
        ("death", {"A": 1.0, "B": 0.0}, [forward], 1e9, 1e9, 1.0),
        ("ode", {"A": 1200.0, "B": 1200.0}, pair, 0.0, 0.0, 1.0),
        ("down", {"A": 80.0, "B": 80.0}, pair, 50.0, 75.0, 1.0),
        ("up", {"A": 4.0, "B": 200.0}, [{**reverse, "coefficient": 1.0}], 5.0, 10.0, 1.0),
        ("fractional", {"A": 50.25, "B": 50.75}, pair, 45.0, 45.0, 1.0),
        ("fractional_boundary", {"A": 0.25, "B": 4.75}, pair, 1e9, 1e9, 2.0),
        (
            "cycle",
            {"A": 100.0, "B": 100.0, "C": 0.0},
            [
                reaction("ab", {"A": -1, "B": 1}, {"A": 1}, 5.0),
                reaction("bc", {"B": -1, "C": 1}, {"B": 1}, 1.0),
                reaction("ca", {"C": -1, "A": 1}, {"C": 1}, 0.1),
            ],
            20.0,
            40.0,
            5.0,
        ),
        (
            "catalyst",
            {"A": 100.0, "B": 100.0, "E": 1.0},
            [
                *pair,
                reaction("catalytic", {"A": -1, "B": 1}, {"A": 1, "E": 1}, 0.5),
            ],
            20.0,
            40.0,
            1.0,
        ),
        (
            "stiff",
            {"A": 1200.0, "B": 1200.0, "C": 4.0, "D": 4.0},
            [
                {**forward, "coefficient": 10.0},
                {**reverse, "coefficient": 2.0},
                reaction("cd", {"C": -1, "D": 1}, {"C": 1}, 0.2),
                reaction("dc", {"D": -1, "C": 1}, {"D": 1}, 0.1),
            ],
            10.0,
            20.0,
            1.0,
        ),
        (
            "third_order",
            {"A": 8.0, "B": 6.0, "C": 0.0},
            [
                reaction("association", {"A": -2, "B": -1, "C": 1}, {"A": 2, "B": 1}, 0.01),
            ],
            1e9,
            1e9,
            1.0,
        ),
    ]
    result = []
    for family, initial, reactions, lower, upper, duration in data:
        if family == "stiff":
            pools = [{"A": 1, "B": 1}, {"C": 1, "D": 1}]
        elif family == "catalyst":
            pools = [{"A": 1, "B": 1}, {"E": 1}]
        elif family == "third_order":
            pools = [{"A": 1, "C": 2}, {"B": 1, "C": 1}]
        else:
            pools = [dict.fromkeys(initial, 1)]
        for method in ("RK4", "LSODA"):
            for h in (0.125, 0.03125):
                result.append(
                    {
                        "id": f"D{len(result) + 1:02d}",
                        "index": len(result),
                        "family": family,
                        "initial": initial,
                        "reactions": reactions,
                        "lower": lower,
                        "upper": upper,
                        "duration_s": duration,
                        "method": method,
                        "step_s": h,
                        "pools": pools,
                        "native_replicates": 1 if family == "ode" else NATIVE_REPLICATES,
                        "reference_replicates": 1 if family == "ode" else REFERENCE_REPLICATES,
                    }
                )
    assert len(result) == 44
    assert sum(u["native_replicates"] for u in result) == 40964
    assert sum(u["reference_replicates"] for u in result) == 163844
    return result


def bounds(unit):
    return {
        key: min(
            sum(unit["initial"][k] * v for k, v in pool.items()) / pool[key]
            for pool in unit["pools"]
            if pool.get(key, 0) > 0
        )
        for key in unit["initial"]
    }


def author(unit):
    from chemworld.physchem.reaction_network import (
        RateLawSpec,
        ReactionNetworkSpec,
        ReactionSpec,
        SpeciesSpec,
    )
    from chemworld.physchem.stochastic_adaptive_hybrid import MOLECULAR_QUANTUM, AdaptiveHybridSpec

    reactions = []
    for r in unit["reactions"]:
        products = {k: r["orders"].get(k, 0) + r["changes"].get(k, 0) for k in unit["initial"]}

        def side(values):
            return " + ".join((str(n) + " " if n > 1 else "") + k for k, n in values.items() if n)

        order = sum(r["orders"].values())
        reactions.append(
            ReactionSpec.from_equation(
                reaction_id=r["id"],
                equation=side(r["orders"]) + " => " + side(products),
                forward_orders=r["orders"],
                rate_law=RateLawSpec(
                    r["id"],
                    "mass_action",
                    {
                        "k": r["coefficient"] * (AVOGADRO * VOLUME_L) ** (order - 1),
                    },
                ),
            )
        )
    species = tuple(
        SpeciesSpec(
            k,
            unit["formulas"][k]
            if "formulas" in unit
            else "Cu"
            if k == "E"
            else "C3H6O3"
            if unit["family"] == "third_order" and k == "C"
            else "CH2O",
        )
        for k in unit["initial"]
    )
    network = ReactionNetworkSpec("adaptive-statistics-" + unit["id"], species, tuple(reactions))
    for actual, expected in zip(network.reactions, unit["reactions"], strict=True):
        if (
            dict(actual.kinetic_forward_orders) != expected["orders"]
            or {k: v for k, v in actual.stoichiometry.items() if v} != expected["changes"]
        ):
            raise ValueError("native adapter changed declared kinetic orders or stoichiometry")
    spec = AdaptiveHybridSpec(
        VOLUME_L,
        lower_limit=unit["lower"],
        upper_limit=unit["upper"],
        method=unit["method"],
        step_s=unit["step_s"],
        partition_steps=1,
        rtol=1e-10,
        atol_particles=1e-12,
    )
    return network, spec, {k: n * MOLECULAR_QUANTUM for k, n in unit["initial"].items()}
