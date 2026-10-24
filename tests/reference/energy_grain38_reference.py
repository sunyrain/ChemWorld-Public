"""Independent scalar reference: raw cards only, no ChemWorld scientific imports."""

from math import exp

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

R = 8.31446261815324


def molecular_states(card):
    return [g for well in card["wells"] for g in well["grains"]] + card["sinks"]


def collision_matrix(well, temperature, bath_concentrations):
    grains = well["grains"]
    matrix = np.zeros((len(grains), len(grains)))
    weights = np.array([g["statistical_weight"] for g in grains])
    equilibrium = weights * np.exp(
        -np.array([g["energy_J_mol"] for g in grains]) / (R * temperature)
    )
    for bath in well["baths"]:
        ratio = temperature / bath["reference_temperature_K"]
        alpha = bath["alpha_J_mol"] * ratio ** bath["alpha_exponent"]
        frequency = (
            bath["collision_coefficient_L_mol_s"]
            * ratio ** bath["collision_exponent"]
            * bath_concentrations.get(bath["bath_species_id"], 0.0)
        )
        # Assemble each unordered pair downward, then obtain its partner by
        # equilibrium flux balance. This does not call/copy the native dispatcher.
        for high in range(len(grains)):
            for low in range(high):
                gap = grains[high]["energy_J_mol"] - grains[low]["energy_J_mol"]
                downward = frequency * weights[low] / sum(weights) * exp(-gap / alpha)
                upward = downward * equilibrium[high] / equilibrium[low]
                matrix[low, high] += downward
                matrix[high, low] += upward
    for column in range(len(grains)):
        matrix[column, column] = -sum(matrix[:, column])
    return matrix


def generator(card, temperature, bath_concentrations):
    states = molecular_states(card)
    names = [g["species_id"] for g in states]
    weights = {g["species_id"]: g["statistical_weight"] for g in states}
    matrix = np.zeros((len(names), len(names)))
    for well in card["wells"]:
        local = collision_matrix(well, temperature, bath_concentrations)
        indices = [names.index(g["species_id"]) for g in well["grains"]]
        matrix[np.ix_(indices, indices)] += local
    sink_ids = {s["species_id"] for s in card["sinks"]}
    for channel in card["channels"]:
        a, b = names.index(channel["source_grain_id"]), names.index(channel["target_species_id"])
        rate = channel["rate_s"]
        matrix[b, a] += rate
        matrix[a, a] -= rate
        if names[b] not in sink_ids:
            reverse = rate * weights[names[a]] / weights[names[b]]
            matrix[a, b] += reverse
            matrix[b, b] -= reverse
    return names, matrix


def trajectory(card, temperature, amounts, times, volume_L=1.0):
    baths = {b["species_id"]: amounts[b["species_id"]] / volume_L for b in card["bath_gases"]}
    names, matrix = generator(card, temperature, baths)
    initial = np.array([amounts[name] for name in names])
    result = []
    for time in times:
        values = dict(zip(names, expm(matrix * time) @ initial, strict=True))
        values.update({b["species_id"]: amounts[b["species_id"]] for b in card["bath_gases"]})
        result.append(values)
    return result


def enthalpies(card, temperature):
    baseline = card["baseline"]
    delta = temperature - baseline["reference_temperature_K"]
    output = {
        g["species_id"]: baseline["h_ref_J_mol"]
        + g["energy_J_mol"]
        + baseline["cp_J_mol_K"] * delta
        for g in molecular_states(card)
    }
    output.update(
        {b["species_id"]: b["h_ref_J_mol"] + b["cp_J_mol_K"] * delta for b in card["bath_gases"]}
    )
    return output


def finite_flow(card, amounts, temperature, molar_flow, duration):
    """Independent n/enthalpy/bath-heat RHS for pure-bath feed -> reactive receiver."""
    names = list(amounts["reactor"])
    molecular = [s["species_id"] for s in molecular_states(card)]
    count = len(names)
    h = np.array([enthalpies(card, temperature)[name] for name in names])
    u = h - R * temperature
    initial = np.array(
        [amounts[node][name] for node in ("feed", "reactor") for name in names] + [0.0, 0.0, 0.0]
    )

    def rhs(_, state):
        feed, receiver = state[:count], state[count : 2 * count]
        flow = molar_flow * feed / sum(feed)
        baths = {
            b["species_id"]: receiver[names.index(b["species_id"])] for b in card["bath_gases"]
        }
        _, matrix = generator(card, temperature, baths)
        chemistry = np.zeros(count)
        molecular_indices = [names.index(name) for name in molecular]
        chemistry[molecular_indices] = matrix @ receiver[molecular_indices]
        feed_rate, receiver_rate = -flow, flow + chemistry
        advected_h = float(h @ flow)
        return np.concatenate(
            (
                feed_rate,
                receiver_rate,
                [
                    float(u @ feed_rate) + advected_h,
                    float(u @ receiver_rate) - advected_h,
                    advected_h,
                ],
            )
        )

    result = solve_ivp(rhs, (0.0, duration), initial, method="DOP853", rtol=2e-12, atol=2e-14)
    if not result.success:
        raise RuntimeError(result.message)
    final = result.y[:, -1]
    return {
        "amounts": {
            node: dict(zip(names, final[i * count : (i + 1) * count], strict=True))
            for i, node in enumerate(("feed", "reactor"))
        },
        "bath_heat_J": {"feed": final[-3], "reactor": final[-2]},
        "transferred_enthalpy_J": final[-1],
        "nfev": result.nfev,
    }
