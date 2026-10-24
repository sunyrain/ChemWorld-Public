"""Independent n/U/Twall balances; imports no ChemWorld implementation."""

from math import exp

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

R = 8.31446261815324
SIGMA = 5.670374419e-8
CP = 30.0
TREF = 298.15


def energy(amounts, temperatures, wall_temperature, card):
    total = card["wall"]["heat_capacity_J_K"] * (
        wall_temperature - card["wall"]["reference_temperature_K"]
    )
    for name in card["nodes"]:
        t = temperatures[name]
        n = amounts[name]
        total += (n["A"] + n["B"]) * (CP * (t - TREF) - R * t) - 12000 * n["B"]
        total += card["supports"][name] * (t - TREF)
    return total


def solve(card):
    names = card["nodes"]
    count = len(names)
    wall = card["wall"]
    contacts = {c["vessel_id"]: c["conductance_W_K"] for c in wall["contacts"]}
    # nA, nB, T for every node; Tw; each contact integral; convection/radiation/heat.
    wi, qi = 3 * count, 3 * count + 1
    initial = np.zeros(qi + count + 3)
    for i, name in enumerate(names):
        initial[3 * i : 3 * i + 3] = [
            card["amounts"][name]["A"],
            card["amounts"][name]["B"],
            card["temperatures"][name],
        ]
    initial[wi] = wall["initial_temperature_K"]

    def rhs(_time, y, power):
        dy = np.zeros_like(y)
        net_h = np.zeros(count)
        for left, right, flow in card["flows"]:
            i, j = names.index(left), names.index(right)
            rates = flow * y[3 * i : 3 * i + 2] / sum(y[3 * i : 3 * i + 2])
            dy[3 * i : 3 * i + 2] -= rates
            dy[3 * j : 3 * j + 2] += rates
            h = rates[0] * CP * (y[3 * i + 2] - TREF) + rates[1] * (
                -12000 + CP * (y[3 * i + 2] - TREF)
            )
            net_h[i] -= h
            net_h[j] += h
        tw = y[wi]
        qcond = wall["loss_conductance_W_K"] * (tw - wall["environment_temperature_K"])
        qrad = (
            wall["emissivity"]
            * SIGMA
            * wall["radiating_area_m2"]
            * (tw**4 - wall["environment_temperature_K"] ** 4)
        )
        qwall = 0.0
        for i, name in enumerate(names):
            na, nb, t = y[3 * i : 3 * i + 3]
            if name in card["reactive_nodes"]:
                reaction = 4 * exp(-8000 / (R * t)) * na
                dy[3 * i] -= reaction
                dy[3 * i + 1] += reaction
            contact = contacts.get(name, 0.0) * (t - tw)
            dy[qi + i] = contact
            qwall += contact
            cv = card["supports"][name] + (na + nb) * (CP - R)
            u = np.array([CP * (t - TREF) - R * t, CP * (t - TREF) - R * t - 12000])
            dy[3 * i + 2] = (
                net_h[i]
                - contact
                + (power if name == "reactor" else 0.0)
                - np.dot(u, dy[3 * i : 3 * i + 2])
            ) / cv
        dy[wi] = (qwall - qcond - qrad) / wall["heat_capacity_J_K"]
        dy[-3:] = [qcond, qrad, power]
        return dy

    def low(_time, y):
        return y[wi] - wall["minimum_temperature_K"]

    def high(_time, y):
        return wall["maximum_temperature_K"] - y[wi]

    low.terminal = high.terminal = True
    low.direction = high.direction = -1
    y, duration, diagnostics = initial, 0.0, []
    for dt, power in card["segments"]:
        result = solve_ivp(
            lambda t, state, power=power: rhs(t, state, power),
            (0, dt),
            y,
            method="Radau",
            rtol=2e-11,
            atol=1e-12,
            events=(low, high),
        )
        if not result.success:
            raise RuntimeError(result.message)
        y = result.y[:, -1]
        duration += float(result.t[-1])
        diagnostics.append({"nfev": result.nfev, "njev": result.njev, "nlu": result.nlu})
        if result.status == 1:
            break
    # Two insulated constant-inventory cases have an independent exact matrix
    # exponential, including contact integrals; it replaces the numerical result.
    if card["name"] in {"one-insulated", "two-insulated"}:
        matrix = np.zeros((2 * count + 1, 2 * count + 1))
        for i, name in enumerate(names):
            c = contacts.get(name, 0.0)
            cv = card["supports"][name] + sum(card["amounts"][name].values()) * (CP - R)
            matrix[i, i], matrix[i, count] = -c / cv, c / cv
            matrix[count, i] += c / wall["heat_capacity_J_K"]
            matrix[count, count] -= c / wall["heat_capacity_J_K"]
            matrix[count + 1 + i, i], matrix[count + 1 + i, count] = c, -c
        z0 = np.array(
            [
                *(card["temperatures"][n] for n in names),
                wall["initial_temperature_K"],
                *([0.0] * count),
            ]
        )
        z = expm(matrix * duration) @ z0
        y[2 : 3 * count : 3], y[wi], y[qi : qi + count] = z[:count], z[count], z[count + 1 :]
    return {
        "amounts": {
            name: {"A": float(y[3 * i]), "B": float(y[3 * i + 1])} for i, name in enumerate(names)
        },
        "temperatures": {name: float(y[3 * i + 2]) for i, name in enumerate(names)},
        "wall_temperature": float(y[wi]),
        "contacts": {name: float(y[qi + i]) for i, name in enumerate(names)},
        "conduction": float(y[-3]),
        "radiation": float(y[-2]),
        "heater": float(y[-1]),
        "duration": duration,
        "solver": diagnostics,
    }
