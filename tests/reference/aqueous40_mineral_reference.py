"""Independent fixed-fixture solubility balances; no native physical imports."""

from itertools import product
from math import exp, log, sqrt

from scipy.optimize import brentq

R, WATER_MASS = 8.31446261815324, 0.018015


def constants(case):
    temperature, pressure = case["temperature_K"], case["pressure_Pa"]
    heat = -1500.0 if case["hydrate"] else -1000.0
    volume_change = 5e-7 if case["hydrate"] else 2e-7
    ka = 1e-4 * exp(
        heat / R * (1 / temperature - 1 / 298.15)
        - volume_change * (pressure - 1e5) / (R * temperature)
    )
    kb = 4e-4 * exp(
        -2000.0 / R * (1 / temperature - 1 / 298.15) - 2e-7 * (pressure - 1e5) / (R * temperature)
    )
    return ka, kb


def saturation(case, amounts):
    ka, kb = constants(case)
    water = amounts["H2O"] * WATER_MASS
    na, cl = amounts["Na+"] / water, amounts["Cl-"] / water
    gamma = 1.0
    if case["family"] == "Davies":
        ionic = 0.5 * (na + cl)
        gamma = 10 ** (-0.509 * (sqrt(ionic) / (1 + sqrt(ionic)) - 0.3 * ionic))
    mineral = "NaCl.H2O(s)" if case["hydrate"] else "NaCl(s)"
    result = {mineral: log(na * cl * gamma**2 / ka)}
    if case["hydrate"]:
        result[mineral] -= WATER_MASS * (na + cl)
    if case["two"]:
        result["KCl(s)"] = log(amounts["K+"] / water * cl / kb)
    return result


def properties(case, amounts):
    temperature, pressure = case["temperature_K"], case["pressure_Pa"]
    rows = {
        "H2O": (3, 0.0, False),
        "Na+": (1, 0.0, False),
        "Cl-": (1, 0.0, False),
        "K+": (1, 0.0, False),
        "NaCl(s)": (2, -1000.0, True),
        "KCl(s)": (2, -2000.0, True),
        "NaCl.H2O(s)": (5, -1500.0, True),
        "NaCl(aq)": (2, 0.0, False),
    }
    enthalpy, volume = 0.0, 0.0
    for key, n in amounts.items():
        atoms, initial_heat, solid = rows[key]
        v = atoms * 1e-6 * (0.9 if solid else 1.0)
        enthalpy += n * (initial_heat + 10 * atoms * (temperature - 298.15) + v * (pressure - 1e5))
        volume += n * v
    return {"H": enthalpy, "V": volume}


def reference(case):
    water = case["water_kg"]
    ka, kb = constants(case)
    mineral = "NaCl.H2O(s)" if case["hydrate"] else "NaCl(s)"
    amounts = {
        "H2O": water / WATER_MASS,
        "Na+": case["na"],
        "Cl-": case["na"] + case["k"],
        mineral: case["solid_na"],
    }
    if case["two"]:
        amounts.update({"K+": case["k"], "KCl(s)": case["solid_k"]})
    if case["association"]:
        amounts["NaCl(aq)"] = case["complex"]
    initial = dict(amounts)
    if case["two"]:
        ta, tb = case["na"] + case["solid_na"], case["k"] + case["solid_k"]
        solutions = []
        for a, b in product((False, True), repeat=2):
            if a and b:
                chloride = water * sqrt(ka + kb)
                na, nk = ka * water**2 / chloride, kb * water**2 / chloride
            elif a:
                nk = tb
                na = 2 * ka * water**2 / (nk + sqrt(nk * nk + 4 * ka * water**2))
            elif b:
                na = ta
                nk = 2 * kb * water**2 / (na + sqrt(na * na + 4 * kb * water**2))
            else:
                na, nk = ta, tb
            candidate = {
                **amounts,
                "Na+": na,
                "K+": nk,
                "Cl-": na + nk,
                mineral: ta - na,
                "KCl(s)": tb - nk,
            }
            si = saturation(case, candidate)
            if (
                candidate[mineral] >= -1e-13
                and candidate["KCl(s)"] >= -1e-13
                and (a or si[mineral] <= 1e-10)
                and (b or si["KCl(s)"] <= 1e-10)
            ):
                solutions.append(candidate)
        if len(solutions) != 1:
            raise ValueError("independent two-salt phase set not unique")
        amounts = solutions[0]
    elif case["association"]:
        total = case["na"] + case["complex"] + case["solid_na"]
        if case["omit_association"]:
            dissolved = min(case["na"] + case["solid_na"], water * sqrt(ka))
            complex_amount = case["complex"]
        else:
            saturated = water * sqrt(ka)
            free_if_absent = 2 * total / (1 + sqrt(1 + 40 * total / water))
            dissolved = min(saturated, free_if_absent)
            complex_amount = 10 * dissolved * dissolved / water
        amounts.update(
            {
                "Na+": dissolved,
                "Cl-": dissolved,
                "NaCl(aq)": complex_amount,
                mineral: total - dissolved - complex_amount,
            }
        )
    else:

        def at_extent(extent):
            return {
                **initial,
                "Na+": case["na"] + extent,
                "Cl-": case["na"] + extent,
                mineral: case["solid_na"] - extent,
                "H2O": initial["H2O"] + (extent if case["hydrate"] else 0.0),
            }

        maximum = case["solid_na"]
        if saturation(case, at_extent(maximum))[mineral] <= 0:
            extent = maximum
        else:
            extent = brentq(
                lambda x: saturation(case, at_extent(x))[mineral],
                -case["na"] + 1e-14,
                maximum,
                xtol=1e-14,
                rtol=1e-14,
            )
        amounts = at_extent(extent)
    before, after = properties(case, initial), properties(case, amounts)
    return {
        "amounts_mol": amounts,
        "initial": initial,
        "SI": saturation(case, amounts),
        "heat_J": after["H"] - before["H"],
        "work_J": case["pressure_Pa"] * (after["V"] - before["V"]),
    }


def invariants(case, amounts):
    solid = amounts["NaCl.H2O(s)" if case["hydrate"] else "NaCl(s)"]
    complex_amount = amounts.get("NaCl(aq)", 0.0)
    result = {
        "Na": amounts["Na+"] + solid + complex_amount,
        "Cl": amounts["Cl-"] + solid + complex_amount + amounts.get("KCl(s)", 0.0),
        "water": amounts["H2O"] + (solid if case["hydrate"] else 0.0),
    }
    if case["two"]:
        result["K"] = amounts["K+"] + amounts["KCl(s)"]
    if case["omit_association"]:
        result["undeclared_complex_reaction"] = complex_amount
    return result
