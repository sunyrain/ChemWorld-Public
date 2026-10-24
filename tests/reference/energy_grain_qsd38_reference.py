"""Independent matrix assembly, two-state roots and general eig/expm oracle.

Only public finite exponential-down equations are used. No ChemWorld generator,
Boltzmann helper, QSD eigensolver or upstream implementation is imported.
"""

from math import exp, sqrt

import numpy as np
from scipy.linalg import eig, expm

R = 8.31446261815324


def matrices(card, temperature, baths):
    well = card["wells"][0]
    grains, sinks = well["grains"], card["sinks"]
    names = [g["species_id"] for g in grains] + [p["species_id"] for p in sinks]
    full = np.zeros((len(names), len(names)))
    weight_sum = sum(g["statistical_weight"] for g in grains)
    for bath in well["baths"]:
        ratio = temperature / bath["reference_temperature_K"]
        frequency = (
            bath["collision_coefficient_L_mol_s"]
            * ratio ** bath["collision_exponent"]
            * baths[bath["bath_species_id"]]
        )
        alpha = bath["alpha_J_mol"] * ratio ** bath["alpha_exponent"]
        for i, target in enumerate(grains):
            for j, source in enumerate(grains):
                if i == j:
                    continue
                gap = target["energy_J_mol"] - source["energy_J_mol"]
                rate = (
                    frequency
                    * target["statistical_weight"]
                    / weight_sum
                    * exp(-abs(gap) / alpha - max(gap, 0) / (R * temperature))
                )
                full[i, j] += rate
                full[j, j] -= rate
    for channel in card["channels"]:
        i, j = names.index(channel["target_species_id"]), names.index(channel["source_grain_id"])
        full[i, j] += channel["rate_s"]
        full[j, j] -= channel["rate_s"]
    return full, full[: len(grains), : len(grains)], full[len(grains) :, : len(grains)]


def solve(card, temperature, baths):
    full, transient, exits = matrices(card, temperature, baths)
    if len(transient) == 2:
        a, b = transient[1, 0], transient[0, 1]
        e0, e1 = exits.sum(axis=0)
        trace = a + b + e0 + e1
        determinant = a * e1 + b * e0 + e0 * e1
        rate = 2 * determinant / (trace + sqrt(trace**2 - 4 * determinant))
        second = trace - rate
        q = np.array([b, a + e0 - rate])
        q /= q.sum()
        left = np.array([a, a + e0 - rate])
        left /= left @ q
    else:
        values, lefts, rights = eig(transient, left=True, right=True)
        indices = np.argsort(values.real)[::-1]
        j = indices[0]
        assert np.max(np.abs(values.imag)) < 1e-6
        q = rights[:, j].real
        q /= q.sum()
        left = lefts[:, j].real
        left /= left @ q
        rate, second = -values[j].real, -values[indices[1]].real
    return {
        "full": full,
        "transient": transient,
        "exits": exits,
        "q": q,
        "left": left,
        "rate": float(rate),
        "second": float(second),
        "product_rates": exits @ q,
    }


def qsd_trajectory(reference, q, rate):
    initial = np.r_[q, np.zeros(reference["exits"].shape[0])]
    return expm(reference["full"] * (0.6 / rate)) @ initial


def scaled_long_time(reference, initial, rate, gap):
    operator = reference["transient"] + np.eye(len(initial)) * rate
    return expm(operator * (30 / gap)) @ initial
