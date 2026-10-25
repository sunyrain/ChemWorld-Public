"""Same physical first-order networks for four resource strategies."""

from adaptive51f_distribution_cases import reaction

SEEDS = (1, 7, 17, 37, 73, 101, 251, 509)
ENGINES = ("direct", "next", "RK4", "LSODA")


def cases():
    result = []
    for index, condition in enumerate(("low", "high", "mixed", "stiff"), 1):
        initial, reactions, pairs = {}, [], []
        for pair in range(16 if condition == "mixed" else 2 if condition == "stiff" else 1):
            a, b = "A" + str(pair), "B" + str(pair)
            high = condition != "low" and (
                condition == "high" or pair < (8 if condition == "mixed" else 1)
            )
            initial.update({a: 1000 if high else 4, b: 1000 if high else 4})
            forward, reverse = (
                (10.0, 2.0)
                if condition == "stiff" and high
                else (1.0, 0.2)
                if high
                else (0.4, 0.2)
                if condition == "low"
                else (0.1, 0.2)
            )
            reactions.extend(
                (
                    reaction("forward" + str(pair), {a: -1, b: 1}, {a: 1}, forward),
                    reaction("reverse" + str(pair), {a: 1, b: -1}, {b: 1}, reverse),
                )
            )
            pairs.append({"A": a, "B": b, "forward": forward, "reverse": reverse, "high": high})
        result.append(
            {
                "id": f"R{index:02d}",
                "condition": condition,
                "family": "resource",
                "initial": initial,
                "reactions": reactions,
                "pairs": pairs,
                "pools": [{p["A"]: 1, p["B"]: 1} for p in pairs],
                "lower": 1e9 if condition == "low" else 10.0,
                "upper": 1e9 if condition == "low" else 20.0,
                "duration_s": 1.0,
                "step_s": 1 / 32,
            }
        )
    return result
