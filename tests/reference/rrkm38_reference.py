"""Grouped occupation enumeration and raw master equation; no ChemWorld imports."""

from collections import Counter
from itertools import product
from math import comb, exp, expm1, fsum, log

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

R = 8.31446261815324
H = 6.62607015e-34
NA = 6.02214076e23


def ground(state):
    return state["reference_energy_J_mol"] + (
        0.5 * sum(state["vibrational_quanta_J_mol"])
        if state["energy_basis"] == "electronic"
        else 0.0
    )


def index(value, grid):
    # Source cards fix exact or explicitly nearest-half-up; validation is native's responsibility.
    return int(np.floor(value / grid["grain_width_J_mol"] + 0.5))


def counts(state, grid):
    """Enumerate occupation totals by equal-frequency groups, not BS recurrence."""
    groups = Counter(index(q, grid) for q in state["vibrational_quanta_J_mol"])
    maximum = grid["maximum_grain_index"]
    result = [0] * (maximum + 1)
    pairs = list(groups.items())
    for occupations in product(*(range(maximum // q + 1) for q, _ in pairs)):
        energy = sum(q * n for (q, _), n in zip(pairs, occupations, strict=True))
        if energy <= maximum:
            weight = state["electronic_degeneracy"]
            for (_, multiplicity), n in zip(pairs, occupations, strict=True):
                weight *= comb(n + multiplicity - 1, multiplicity - 1)
            result[energy] += weight
    return result


def micro(card):
    grid, well = card["grid"], card["well"]
    width = grid["grain_width_J_mol"]
    g = counts(well, grid)
    rows = {}
    for channel in card["channels"]:
        ts = channel["transition_state"]
        gt = counts(ts, grid)
        barrier = index(ground(ts) - ground(well), grid)
        cumulative = np.cumsum(gt, dtype=object).tolist()
        rates = [
            channel["path_degeneracy"] * width * sum(gt[: j - barrier + 1]) / (H * NA * c)
            if j >= barrier and c
            else (0.0 if c else None)
            for j, c in enumerate(g)
        ]
        rows[channel["channel_id"]] = {
            "counts": gt,
            "sums": cumulative,
            "barrier": barrier,
            "rates": rates,
        }
    return {"counts": g, "sums": np.cumsum(g, dtype=object).tolist(), "channels": rows}


def canonical(card, temperature):
    m = micro(card)
    width = card["grid"]["grain_width_J_mol"]
    weight = [c * exp(-j * width / (R * temperature)) for j, c in enumerate(m["counts"])]
    total = sum(weight)
    fractions = [v / total for v in weight]

    def log_partition(state):
        return log(state["electronic_degeneracy"]) - sum(
            log(-expm1(-index(q, card["grid"]) * width / (R * temperature)))
            for q in state["vibrational_quanta_J_mol"]
        )

    dense = any(index(q, card["grid"]) == 1 for q in card["well"]["vibrational_quanta_J_mol"])
    factor = width / (R * temperature) / (-expm1(-width / (R * temperature)))
    rows = {}
    for c in card["channels"]:
        values = m["channels"][c["channel_id"]]
        k = fsum(p * r for p, r in zip(fractions, values["rates"], strict=True) if r is not None)
        tst = (
            c["path_degeneracy"]
            * R
            * temperature
            / (H * NA)
            * exp(
                log_partition(c["transition_state"])
                - log_partition(card["well"])
                - values["barrier"] * width / (R * temperature)
            )
        )
        rows[c["channel_id"]] = {
            "finite": k,
            "tst": tst,
            "infinite": factor * tst if dense else None,
        }
    return {
        "fractions": fractions,
        "retention": exp(log(total) - log_partition(card["well"])),
        "dense": dense,
        "factor": factor,
        "channels": rows,
    }


def model_fields(card):
    m = micro(card)
    width = card["grid"]["grain_width_J_mol"]
    bins = [j for j, c in enumerate(m["counts"]) if c]
    names = [f"{card['well']['state_id']}:grain:{j}" for j in bins]
    energies = [ground(card["well"]) + j * width for j in bins]
    weights = [m["counts"][j] for j in bins]
    names += [c["product"]["species_id"] for c in card["channels"]]
    energies += [c["product"]["energy_J_mol"] for c in card["channels"]]
    return m, bins, names, np.array(energies), np.array(weights, dtype=float)


def generator(card, temperature, baths):
    m, bins, names, energies, weights = model_fields(card)
    matrix = np.zeros((len(names), len(names)))
    for bath in card["baths"]:
        alpha = (
            bath["alpha_J_mol"]
            * (temperature / bath["reference_temperature_K"]) ** bath["alpha_exponent"]
        )
        freq = (
            bath["collision_coefficient_L_mol_s"]
            * (temperature / bath["reference_temperature_K"]) ** bath["collision_exponent"]
            * baths.get(bath["bath_species_id"], 0.0)
        )
        for source in range(len(bins)):
            for target in range(len(bins)):
                if source == target:
                    continue
                gap = energies[target] - energies[source]
                rate = (
                    freq
                    * weights[target]
                    / sum(weights)
                    * exp(-abs(gap) / alpha - max(gap, 0.0) / (R * temperature))
                )
                matrix[target, source] += rate
                matrix[source, source] -= rate
    for b, c in enumerate(card["channels"]):
        for i, j in enumerate(bins):
            rate = m["channels"][c["channel_id"]]["rates"][j]
            matrix[len(bins) + b, i] += rate
            matrix[i, i] -= rate
    return names, matrix, energies


def closed(card, temperature, amounts, duration):
    names, matrix, energies = generator(card, temperature, amounts)
    initial = np.array([amounts.get(s, 0.0) for s in names])
    final = expm(matrix * duration) @ initial
    return {
        "names": names,
        "matrix": matrix.tolist(),
        "amounts": dict(zip(names, final.tolist(), strict=True)),
        "heat_J": float(energies @ (final - initial)),
    }


def finite_flow(card, temperature, initial, flow, duration):
    _, _, names, energies, _ = model_fields(card)
    bath_names = [b["species_id"] for b in card["bath_gases"]]
    all_names = names + bath_names
    size = len(all_names)
    y0 = np.array(
        [initial[v].get(s, 0.0) for v in ("cell", "feed") for s in all_names] + [0.0, 0.0]
    )
    cp = np.array(
        [card["baseline"]["cp_J_mol_K"]] * len(names)
        + [b["cp_J_mol_K"] for b in card["bath_gases"]]
    )
    h0 = np.array(
        [card["baseline"]["h_ref_J_mol"] + e for e in energies]
        + [b["h_ref_J_mol"] for b in card["bath_gases"]]
    )
    enthalpy = h0 + cp * (temperature - card["baseline"]["reference_temperature_K"])
    internal = enthalpy - R * temperature
    _, base_matrix, _ = generator(card, temperature, {})
    bath_matrices = {b: generator(card, temperature, {b: 1.0})[1] - base_matrix for b in bath_names}

    def rhs(_, y):
        output = np.zeros_like(y)
        incoming = flow * y[size : 2 * size] / sum(y[size : 2 * size])
        matrix = base_matrix + sum(
            y[len(names) + i] * bath_matrices[b] for i, b in enumerate(bath_names)
        )
        chemistry = matrix @ y[: len(names)]
        output[:size] = incoming
        output[: len(names)] += chemistry
        output[size : 2 * size] = -incoming
        transferred_h = float(incoming @ enthalpy)
        output[-2] = float(output[:size] @ internal) - transferred_h
        output[-1] = float(output[size : 2 * size] @ internal) + transferred_h
        return output

    result = solve_ivp(rhs, (0, duration), y0, method="Radau", rtol=2e-10, atol=1e-13)
    if not result.success:
        raise RuntimeError(result.message)
    return {
        "amounts": {
            v: dict(zip(all_names, result.y[i * size : (i + 1) * size, -1].tolist(), strict=True))
            for i, v in enumerate(("cell", "feed"))
        },
        "heat_J": dict(zip(("cell", "feed"), result.y[-2:, -1].tolist(), strict=True)),
        "nfev": result.nfev,
        "njev": result.njev,
        "nlu": result.nlu,
    }
