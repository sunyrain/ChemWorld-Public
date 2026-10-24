"""Independent elementary formulas; deliberately no ChemWorld imports."""

from math import exp


def reference(
    time_s,
    *,
    kind="decay",
    initial_a=0.8,
    initial_b=0.0,
    k1=0.55,
    k2=0.23,
    volume_L=2.0,
    source_seconds=1.0,
):
    if kind == "reversible":
        total, rate = initial_a + initial_b, k1 + k2
        equilibrium = total * k2 / rate
        exponential = exp(-rate * time_s)
        amount = equilibrium + (initial_a - equilibrium) * exponential
        equilibrium_derivatives = (-total * k2 / rate**2, total * k1 / rate**2)
        slopes = {
            name: slope * (1 - exponential) - (initial_a - equilibrium) * time_s * exponential
            for name, slope in zip(("k1", "k2"), equilibrium_derivatives, strict=True)
        }
        forward, forward_slopes = (
            k1 * amount,
            {
                "k1": amount + k1 * slopes["k1"],
                "k2": k1 * slopes["k2"],
            },
        )
    else:
        rate = k1 + (k2 if kind == "confounded" else 0.0)
        amount = initial_a * exp(-rate * time_s)
        slopes = dict.fromkeys(("k1", "k2") if kind == "confounded" else ("k1",), -time_s * amount)
        forward = rate * amount
        forward_slopes = {name: amount + rate * slope for name, slope in slopes.items()}
    return {
        "amount_A": (amount, {k: v / source_seconds for k, v in slopes.items()}),
        "amount_B": (
            initial_a + initial_b - amount,
            {k: -v / source_seconds for k, v in slopes.items()},
        ),
        "concentration_A": (
            amount / volume_L,
            {k: v / volume_L / source_seconds for k, v in slopes.items()},
        ),
        "reaction_rate_forward": (
            forward,
            {k: v / source_seconds for k, v in forward_slopes.items()},
        ),
    }
