"""Independent autonomous scalar RI5 polynomial with declared reaction direction."""

import math


def scalar_endpoint(case, hat):
    h, initial = case["step_s"], case["initial"]["A"]
    direction = case["reactions"][0]["stoichiometry"]["A"]
    z = (0.2 if case["kind"] == "gbm" else -1) * h
    deterministic = 1 + z + z * z / 2 + z * z * z / 6
    if case["kind"] == "gbm":
        return initial * (
            deterministic
            - direction * 0.25 * math.sqrt(h) * hat * (1 + z - z * z / 18)
            + 0.5 * 0.25**2 * h * (hat * hat - 1)
        )
    return (
        50
        + deterministic * (initial - 50)
        - direction * 2 * math.sqrt(h) * hat * (1 + z / 2 - z * z / 18)
    )
