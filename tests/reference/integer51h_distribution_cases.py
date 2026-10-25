"""Fixed full-reactant chemistry as data; no native imports at module load."""

STRATEGIES = ("direct", "next", "coarse", "fine")
BASE_SEEDS = dict(zip(STRATEGIES, (513600000, 523600000, 533600000, 543600000), strict=True))


def units():
    result = []

    def add(family, initial, rows, duration, high=False, frozen=False, pools=None):
        result.append(
            {
                "id": f"S{len(result) + 1:02d}",
                "index": len(result),
                "family": family,
                "initial": initial,
                "reactions": rows,
                "duration_s": duration,
                "high": high,
                "frozen": frozen,
                "replicates": 1 if frozen else 2048,
                "pools": pools or [dict.fromkeys(initial, 1)],
            }
        )

    def reaction(identity, orders, changes, coefficient):
        return {"id": identity, "orders": orders, "changes": changes, "coefficient": coefficient}

    patterns = (
        (
            "autocatalytic",
            {"A": 1, "B": 1},
            {"A": 1, "B": -1},
            ((1, 11, 0.2, 2), (20, 180, 0.002, 3)),
        ),
        ("partial", {"A": 2}, {"A": -1, "B": 1}, ((12, 0, 0.08, 2), (200, 0, 0.002, 3))),
        ("third_A", {"A": 2, "B": 1}, {"A": 1, "B": -1}, ((2, 10, 0.01, 2), (20, 180, 2e-5, 3))),
        ("third_B", {"A": 1, "B": 2}, {"A": 1, "B": -1}, ((1, 11, 0.01, 2), (20, 180, 2e-5, 3))),
    )
    for family, orders, changes, variants in patterns:
        for index, (a, b, c, duration) in enumerate(variants):
            add(
                family,
                {"A": a, "B": b},
                [reaction("reaction", orders, changes, c)],
                duration,
                high=index == 1,
            )
    for a, b, c, high in ((1, 11, 0.2, False), (3, 197, 0.05, True)):
        add(
            "catalyst",
            {"A": a, "B": b, "C": 0},
            [reaction("reaction", {"A": 1, "B": 1}, {"B": -1, "C": 1}, c)],
            3,
            high=high,
            pools=[{"A": 1}, {"B": 1, "C": 1}],
        )
    for a, high in ((4, False), (100, True)):
        add(
            "switch",
            {"A": a, "B": 0, "C": 0, "D": 1},
            [
                reaction("catalysed", {"A": 1, "C": 1}, {"A": -1, "B": 1}, 3),
                reaction("off", {"C": 1}, {"C": -1, "D": 1}, 20),
                reaction("on", {"D": 1}, {"C": 1, "D": -1}, 20),
            ],
            1,
            high=high,
            pools=[{"A": 1, "B": 1}, {"C": 1, "D": 1}],
        )
    add(
        "zero_autocatalysis",
        {"A": 0, "B": 12},
        [reaction("reaction", {"A": 1, "B": 1}, {"A": 1, "B": -1}, 0.2)],
        2,
        frozen=True,
    )
    add(
        "partial_boundary",
        {"A": 1, "B": 11},
        [reaction("reaction", {"A": 2}, {"A": -1, "B": 1}, 0.08)],
        2,
        frozen=True,
    )
    assert len(result) == 14 and 4 * sum(u["replicates"] for u in result) == 98312
    return result


def author(unit, strategy):
    from chemworld.physchem.reaction_network import (
        RateLawSpec,
        ReactionNetworkSpec,
        ReactionSpec,
        SpeciesSpec,
    )
    from chemworld.physchem.stochastic_kinetics import AVOGADRO, TAU_ALGORITHM
    from chemworld.physchem.stochastic_tau_leap import TauLeapPolicy

    rows = []
    for r in unit["reactions"]:
        parts = [
            " + ".join(
                (str(abs(v)) + " " if abs(v) != 1 else "") + k
                for k, v in r["changes"].items()
                if sign * v > 0
            )
            for sign in (-1, 1)
        ]
        k = r["coefficient"] * (AVOGADRO * 1e-15) ** (sum(r["orders"].values()) - 1)
        rows.append(
            ReactionSpec.from_equation(
                reaction_id=r["id"],
                equation=" => ".join(parts),
                forward_orders=r["orders"],
                rate_law=RateLawSpec(r["id"], "mass_action", {"k": k}),
            )
        )
    algorithm = {"direct": "gillespie-direct-v1", "next": "gibson-bruck-next-reaction-v1"}.get(
        strategy, TAU_ALGORITHM
    )
    declaration = {
        "volume_L": 1e-15,
        "temperature_K": 298.15,
        "algorithm": algorithm,
        "max_events_per_advance": 1000000,
    }
    if algorithm == TAU_ALGORITHM:
        declaration["tau_policy"] = TauLeapPolicy(
            epsilon=0.1 if strategy == "coarse" else 0.01,
            max_tau_s=0.1 if strategy == "coarse" else 0.005,
            ssa_threshold=0.01,
        ).to_dict()
    return ReactionNetworkSpec(
        "integer51h-" + unit["id"],
        tuple(SpeciesSpec(k, "H2") for k in unit["initial"]),
        tuple(rows),
        stochastic=declaration,
    )
