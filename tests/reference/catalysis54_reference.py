"""Independent cycle formulas, Frumkin brackets and finite-batch ODE.

Reads declared scalar data only. Reuses the independent 53 oscillator/NASA
reference formulas, never the native kinetic or residual/advance solvers.
"""

from math import exp, log

import numpy as np
from catalysis53_reference import R, thermo
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def constants(spec, temperature, theta=0):
    entries = {s.species.species_id: s for s in spec.species}
    g = {k: thermo(s, temperature)["g"] for k, s in entries.items()}
    g["bound"] += spec.interaction_matrix_J_mol[0][0] * theta
    g["vacancy:metal"] = 0.0
    result = []
    for reaction in spec.reactions:
        delta = sum(nu * g[k] for k, nu in reaction.products.items()) - sum(
            nu * g[k] for k, nu in reaction.reactants.items()
        )
        forward = reaction.bep.intercept_J_mol + reaction.bep.alpha * delta
        result.append(
            (
                reaction.attempt_frequency_s_inv * exp(-forward / (R * temperature)),
                reaction.attempt_frequency_s_inv * exp(-(forward - delta) / (R * temperature)),
            )
        )
    return result


def cycle(spec, boundary):
    (kf, kr), (of, or_) = constants(spec, boundary.temperature_K)
    a = kf * boundary.gas_partial_pressures_Pa["A"] / 100000
    b, c = kr, of
    d = or_ * boundary.gas_partial_pressures_Pa["B"] / 100000
    total, numerator = a + b + c + d, a * c - b * d
    return {
        "coverage": (a + d) / total,
        "turnover": numerator / total,
        "eigenvalue": -total,
        "drc_adsorption": (c + d) / total,
        "drc_conversion": (a + b) / total,
        "order_A": a * c / numerator - a / total,
        "order_B": -b * d / numerator - d / total,
    }


def apparent_energy(spec, boundary):
    from dataclasses import replace

    step = 0.001
    lower = cycle(spec, replace(boundary, temperature_K=boundary.temperature_K - step))["turnover"]
    upper = cycle(spec, replace(boundary, temperature_K=boundary.temperature_K + step))["turnover"]
    return R * boundary.temperature_K**2 * (log(upper) - log(lower)) / (2 * step)


def frumkin(spec, boundary):
    entries = {s.species.species_id: s for s in spec.species}
    t, p = boundary.temperature_K, boundary.gas_partial_pressures_Pa["A"] / 100000
    delta0 = thermo(entries["bound"], t)["g"] - thermo(entries["A"], t)["g"]
    epsilon = spec.interaction_matrix_J_mol[0][0]

    def condition(theta):
        return log(theta / (1 - theta)) + (delta0 + epsilon * theta) / (R * t) - log(p)

    roots = [
        brentq(condition, a, b, xtol=1e-14) for a, b in ((1e-10, 0.2), (0.2, 0.8), (0.8, 1 - 1e-10))
    ]
    answer = []
    alpha = spec.reactions[0].bep.alpha
    for theta in roots:
        kf, kr = constants(spec, t, theta)[0]
        eigenvalue = (
            -alpha * epsilon / (R * t) * kf * p * (1 - theta)
            - kf * p
            - (1 - alpha) * epsilon / (R * t) * kr * theta
            - kr
        )
        answer.append({"coverage": theta, "eigenvalue": eigenvalue})
    return answer


def finite_batch(spec, boundary, initial_gas, initial_bound, time):
    (kf, kr), (of, or_) = constants(spec, boundary.temperature_K)
    capacity = boundary.capacities_mol["metal"]
    factor = R * boundary.temperature_K / (boundary.volume_L * 1e-3 * 100000)

    def rhs(t, y):
        a, b, bound = y
        theta = bound / capacity
        r1 = capacity * (kf * a * factor * (1 - theta) - kr * theta)
        r2 = capacity * (of * theta - or_ * b * factor * (1 - theta))
        return [-r1, r2, r1 - r2]

    solved = solve_ivp(
        rhs,
        (0, time),
        [initial_gas["A"], initial_gas["B"], initial_bound],
        method="RK45",
        rtol=1e-11,
        atol=1e-14,
    )
    if not solved.success:
        raise RuntimeError(solved.message)
    return dict(zip(("A", "B", "bound"), map(float, np.asarray(solved.y)[:, -1]), strict=True))
