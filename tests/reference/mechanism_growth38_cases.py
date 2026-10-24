"""Independent scalar four-reaction oracle, with no native kinetic/RHS calls."""

from __future__ import annotations

import json
from math import exp

import numpy as np
from scipy.integrate import solve_ivp

SPECIES = ("ethyl", "ethene", "H", "H2", "ethane", "butane")
ROWS = (
    ("beta_scission", {"ethyl": 1}, {"ethene": 1, "H": 1}),
    ("radical_recombination", {"ethyl": 2}, {"butane": 1}),
    ("radical_recombination", {"ethyl": 1, "H": 1}, {"ethane": 1}),
    ("radical_recombination", {"H": 2}, {"H2": 1}),
)
KEYS = tuple(
    json.dumps([family, sorted(left.items()), sorted(right.items())], separators=(",", ":"))
    for family, left, right in ROWS
)


def scalar_rates(amounts, temperature, volume):
    e, h = amounts.get("ethyl", 0) / volume, amounts.get("H", 0) / volume
    return dict(
        zip(
            KEYS,
            (
                0.4 * exp(-3000 / (8.31446261815324 * temperature)) * e,
                0.04 * e**2,
                8 * exp(-2000 / (8.31446261815324 * temperature)) * e * h,
                4 * h**2,
            ),
            strict=True,
        )
    )


def scalar_derivatives(amounts, temperature, volume, selected):
    rates = scalar_rates(amounts, temperature, volume)
    a, b, c, d = (rates[key] if key in selected else 0 for key in KEYS)
    return dict(zip(SPECIES, (-a - 2 * b - c, a, a - c - 2 * d, d, c, b), strict=True))


def independent_trajectory(condition, selected):
    initial = np.array([condition.initial_amounts_mol.get(s, 0) for s in SPECIES])

    def rhs(time, values):
        state = dict(zip(SPECIES, values, strict=True))
        concentration_rates = scalar_derivatives(
            state, condition.temperature_K, condition.volume_L, selected
        )
        return np.array([concentration_rates[s] * condition.volume_L for s in SPECIES])

    result = solve_ivp(
        rhs,
        (0, condition.sample_times_s[-1]),
        initial,
        method="DOP853",
        t_eval=condition.sample_times_s,
        rtol=2e-12,
        atol=1e-14,
        max_step=0.01,
    )
    if not result.success:
        raise RuntimeError(result.message)
    return [dict(zip(SPECIES, values, strict=True)) for values in result.y.T]
