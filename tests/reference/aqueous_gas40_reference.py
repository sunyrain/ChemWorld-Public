"""Independent authored finite-headspace mathematics, no native physics imports."""

from math import exp, log, sqrt

from scipy.integrate import solve_ivp

R = 8.31446261815324
TREF = 298.15
VOLUME = 1e-4
WATER_KG = VOLUME / 18e-6 * 0.018015


def parameters(kind, condition):
    temperature = (298.15, 310.0)[condition]
    acid = kind in {"acid", "double"}
    ga = R * TREF * log(0.01) if acid else 2000.0
    rows = {
        "H2O": (1000.0, 0.0, 30.0),
        "I(g)": (200.0, 0.0, 10.0),
        "A(g)": (1200.0, ga + R * TREF * log(0.1), 20.0),
        "A(aq)": (-1000.0, ga, 20.0),
    }
    if acid:
        rows.update({"H+": (0.0, 0.0, 10.0), "Cl-": (0.0, 0.0, 10.0)})
    if kind == "double":
        rows.update({"B(g)": (700.0, R * TREF * log(0.2), 30.0), "B(aq)": (-500.0, 0.0, 30.0)})
    states = {}
    for name, (h, g, cp) in rows.items():
        enthalpy = h + cp * (temperature - TREF)
        entropy = (h - g) / TREF + cp * log(temperature / TREF)
        states[name] = {"h": enthalpy, "g": enthalpy - temperature * entropy}
    k = {"A": exp((states["A(g)"]["g"] - states["A(aq)"]["g"]) / (R * temperature))}
    if kind == "double":
        k["B"] = exp((states["B(g)"]["g"] - states["B(aq)"]["g"]) / (R * temperature))
    ka = (
        exp((states["A(aq)"]["g"] - states["H+"]["g"] - states["Cl-"]["g"]) / (R * temperature))
        if acid
        else None
    )
    initial = dict.fromkeys(states, 0.0)
    initial["H2O"] = VOLUME / 18e-6
    initial["I(g)"] = 8e4 * VOLUME / (R * temperature)
    initial["A(g)"] = (
        (0.0 if kind == "desorption" else 5e4 if acid else 2e4) * VOLUME / (R * temperature)
    )
    initial["A(aq)"] = 0.002 if kind == "desorption" else 0.001 if acid else 0.0
    if acid:
        initial = close_acid(initial, initial["A(aq)"], ka)
    if kind == "double":
        initial.update({"B(g)": 1e4 * VOLUME / (R * temperature), "B(aq)": 0.004})
    return {
        "T": temperature,
        "acid": acid,
        "states": states,
        "K": k,
        "Ka": ka,
        "rate": {"A": (4e-5, 7e-5)[condition], "B": 3e-5},
        "initial": initial,
    }


def close_acid(amounts, total, ka):
    concentration = total / WATER_KG
    ion = 2 * ka * concentration / (ka + sqrt(ka * ka + 4 * ka * concentration)) * WATER_KG
    return {**amounts, "H+": ion, "Cl-": ion, "A(aq)": total - ion}


def properties(p, amounts):
    pressure = sum(n for k, n in amounts.items() if k.endswith("(g)")) * R * p["T"] / VOLUME
    hl = sum(n * p["states"][k]["h"] for k, n in amounts.items() if not k.endswith("(g)"))
    hl += VOLUME * (pressure - 1e5)
    hg = sum(n * p["states"][k]["h"] for k, n in amounts.items() if k.endswith("(g)"))
    ul, ug = hl - pressure * VOLUME, hg - pressure * VOLUME
    return {
        "P": pressure,
        "H": hl + hg,
        "U": ul + ug,
        "H_liquid": hl,
        "U_liquid": ul,
        "H_gas": hg,
        "U_gas": ug,
    }


def rates(p, amounts):
    result = {}
    for name, equilibrium in p["K"].items():
        gas_activity = amounts[name + "(g)"] * R * p["T"] / VOLUME / 1e5
        aqueous_activity = amounts[name + "(aq)"] / WATER_KG
        result["uptake-" + name] = [
            p["rate"][name] * gas_activity,
            p["rate"][name] * aqueous_activity / equilibrium,
        ]
    return result


def history(kind, condition, times, max_step=1.0):
    p = parameters(kind, condition)
    initial = p["initial"]
    total = initial["A(g)"] + initial["A(aq)"] + initial.get("Cl-", 0.0)
    alpha = R * p["T"] / VOLUME / 1e5
    if p["acid"]:

        def rhs(t, y):
            neutral = close_acid({}, total - float(y[0]), p["Ka"])["A(aq)"]
            return [-p["rate"]["A"] * (alpha * y[0] - neutral / (WATER_KG * p["K"]["A"]))]

        sol = solve_ivp(
            rhs,
            (0.0, float(times[-1])),
            [initial["A(g)"]],
            t_eval=times,
            rtol=2e-12,
            atol=1e-15,
            method="DOP853",
            max_step=max_step,
        )
        if not sol.success:
            raise ValueError(sol.message)
        agas = sol.y[0]
    else:
        neq = total / (1 + WATER_KG * p["K"]["A"] * alpha)
        rate = p["rate"]["A"] * (alpha + 1 / (WATER_KG * p["K"]["A"]))
        agas = [neq + (initial["A(g)"] - neq) * exp(-rate * t) for t in times]
    result = []
    before = properties(p, initial)
    for t, ng in zip(times, agas, strict=True):
        amounts = {**initial, "A(g)": float(ng)}
        if p["acid"]:
            amounts = close_acid(amounts, total - float(ng), p["Ka"])
        else:
            amounts["A(aq)"] = total - float(ng)
        if kind == "double":
            totalb = initial["B(g)"] + initial["B(aq)"]
            neq = totalb / (1 + WATER_KG * p["K"]["B"] * alpha)
            rate = p["rate"]["B"] * (alpha + 1 / (WATER_KG * p["K"]["B"]))
            amounts["B(g)"] = neq + (initial["B(g)"] - neq) * exp(-rate * t)
            amounts["B(aq)"] = totalb - amounts["B(g)"]
        prop = properties(p, amounts)
        result.append(
            {
                "time_s": float(t),
                "amounts": amounts,
                "properties": prop,
                "heat": prop["U"] - before["U"],
            }
        )
    return result
