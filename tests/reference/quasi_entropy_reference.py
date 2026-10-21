"""Independent series primitive derived from the published quasi Cp formula."""

import math

# Exact SI molar gas constant; reference independent of the production module.
R = 8.31446261815324


def dilogarithm_series(x):
    if not 0 < x < 1:
        raise ValueError("fixed series reference requires 0 < T/Tc < 1")
    terms = []
    power = 1.0
    for k in range(1, 100000):
        power *= x
        terms.append(power / k**2)
        tail_bound = power * x / ((k + 1)**2 * (1 - x))
        if tail_bound <= 1e-16:
            return math.fsum(terms), tail_bound
    raise ValueError("series tail bound not reached")


def quasi_entropy_primitive(T, c):
    x = T / c["Tc"]
    value, tail = dilogarithm_series(x)
    result = R * math.fsum((
        -c["a1"] * value, c["a2"] * (math.log(x) - math.log1p(-x)),
        c["a3"] * math.log(x), c["a4"] * x, c["a5"] * x**2 / 2, c["a6"] * x**3 / 3,
    ))
    return result, tail * abs(R * c["a1"])
