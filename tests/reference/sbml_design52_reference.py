"""Analytic finite-pool objective reference, independent of native solvers."""

from math import exp

from scipy.optimize import brentq


def product(k1, k2=0.2, *, initial_a=0.8, initial_b=0.2, time_s=None):
    total = initial_a + initial_b
    steady = total * k1 / (k1 + k2)
    return steady if time_s is None else steady + (initial_b - steady) * exp(-(k1 + k2) * time_s)


def optimum(name, *, initial_a=0.8, initial_b=0.2):
    name = name.removeprefix("scaled_").removeprefix("log_")
    k2 = 0.1 if name == "steady_budget" else 0.2
    total = initial_a + initial_b
    if name == "steady_min":
        k1 = 0.75 * k2 / (total - 0.75)
    elif name == "trajectory_min":
        k1 = brentq(
            lambda value: (
                product(value, k2, initial_a=initial_a, initial_b=initial_b, time_s=2.0) - 0.6
            ),
            0.05,
            1.0,
            xtol=1e-14,
        )
    else:
        k1 = 0.7 if name == "steady_budget" else 0.65
    b = product(
        k1,
        k2,
        initial_a=initial_a,
        initial_b=initial_b,
        time_s=2.0 if name.startswith("trajectory") else None,
    )
    metrics = {
        "product": b,
        "reactant": total - b,
        "control": k1,
        "total_control": k1 + k2,
        "forward_flux": k1 * (total - b),
        "product_concentration": b / 2.0,
        "volume": 2.0,
    }
    controls = {"forward_rate": k1}
    if name == "steady_budget":
        controls["reverse_rate"] = k2
    return {
        "controls": controls,
        "objective": k1 if name.endswith("min") else b,
        "metrics": metrics,
        "k1": k1,
        "k2": k2,
    }
