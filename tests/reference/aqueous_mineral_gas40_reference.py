"""Independent authored algebraic common-ion closure and finite gas/solid history.

No production imports. Uses gas moles and radii, analytic acid quadratic,
same-reference ideal thermodynamics, scipy integration/root only.
"""

from math import exp, log, sqrt

import numpy as np
from aqueous_gas40_reference import TREF, VOLUME, WATER_KG, R
from aqueous_gas40_reference import parameters as gas_parameters
from scipy.integrate import solve_ivp
from scipy.optimize import root


def parameters(kind, condition):
    p = gas_parameters("double" if kind == "double" else "acid", condition)
    p["minerals"] = ["MA(s)", "KA(s)"] if kind == "double" else ["MA(s)"]
    p["cation"], p["volume"], p["geometry"], p["coefficient"] = {}, {}, {}, {}
    initial = p["initial"]
    for mineral in p["minerals"]:
        cation, ksp, volume = ("M+", 5e-4, 1e-3) if mineral == "MA(s)" else ("K+", 8e-4, 5e-4)
        p["cation"][mineral], p["volume"][mineral] = cation, volume
        for key, h, g, cp in (
            (cation, 0.0, 0.0, 10.0),
            (mineral, -2000.0, R * TREF * log(ksp), 20.0),
        ):
            ht = h + cp * (p["T"] - TREF)
            st = (h - g) / TREF + cp * log(p["T"] / TREF)
            p["states"][key] = {"h": ht, "g": ht - p["T"] * st}
        initial[mineral] = 1e-5 if kind == "exhaustion" else 0.0 if kind == "unseeded" else 0.004
        initial[cation] = 0.0 if kind == "exhaustion" else 0.002 if mineral == "MA(s)" else 0.001
        p["geometry"][mineral] = (0.01, initial[mineral] or 0.004)
        p["coefficient"][mineral] = (0.0 if kind == "zero_mineral" else 0.002) * exp(
            -1200 / R * (1 / p["T"] - 1 / TREF)
        )
    if kind == "degassing":
        initial["A(g)"] = 0.0
        initial["A(aq)"] += 0.004
    if kind == "zero_gas":
        p["rate"] = dict.fromkeys(p["rate"], 0.0)
    p["acid_total"] = initial["A(g)"] + initial["A(aq)"] + initial["H+"]
    p["metal_total"] = {m: initial[m] + initial[p["cation"][m]] for m in p["minerals"]}
    if kind == "double":
        p["b_total"] = initial["B(g)"] + initial["B(aq)"]
    p["initial"] = close(p, initial)
    return p


def close(p, amounts):
    n = dict(amounts)
    for m in p["minerals"]:
        n[p["cation"][m]] = p["metal_total"][m] - n[m]
    salt = sum(n[p["cation"][m]] for m in p["minerals"]) / WATER_KG
    total = (p["acid_total"] - n["A(g)"]) / WATER_KG
    c = salt + p["Ka"]
    h = 2 * p["Ka"] * total / (c + sqrt(c * c + 4 * p["Ka"] * total))
    n.update({"H+": h * WATER_KG, "Cl-": (h + salt) * WATER_KG, "A(aq)": (total - h) * WATER_KG})
    if "b_total" in p:
        n["B(aq)"] = p["b_total"] - n["B(g)"]
    return n


def properties(p, n):
    volumes = {m: n[m] * p["volume"][m] for m in p["minerals"]}
    vg = VOLUME - sum(volumes.values())
    pressure = sum(v for k, v in n.items() if k.endswith("(g)")) * R * p["T"] / vg
    hg = sum(v * p["states"][k]["h"] for k, v in n.items() if k.endswith("(g)"))
    hl = sum(
        v * p["states"][k]["h"] for k, v in n.items() if not k.endswith(("(g)", "(s)"))
    ) + VOLUME * (pressure - 1e5)
    hm = {m: n[m] * p["states"][m]["h"] + volumes[m] * (pressure - 1e5) for m in p["minerals"]}
    um = {m: hm[m] - pressure * volumes[m] for m in p["minerals"]}
    ug, ul = hg - pressure * vg, hl - pressure * VOLUME
    return {
        "P": pressure,
        "Vg": vg,
        "H": hl + hg + sum(hm.values()),
        "U": ul + ug + sum(um.values()),
        "Hl": hl,
        "Ul": ul,
        "Hg": hg,
        "Ug": ug,
        "Hm": hm,
        "Um": um,
    }


def rates(p, n):
    props = properties(p, n)
    gas, minerals = {}, {}
    for name, k in p["K"].items():
        gas["uptake-" + name] = [
            p["rate"][name] * n[name + "(g)"] * R * p["T"] / props["Vg"] / 1e5,
            p["rate"][name] * n[name + "(aq)"] / WATER_KG / k,
        ]
    for m in p["minerals"]:
        cation = p["cation"][m]
        gm = p["states"][m]["g"] + p["volume"][m] * (props["P"] - 1e5)
        ksp = exp((gm - p["states"][cation]["g"] - p["states"]["Cl-"]["g"]) / (R * p["T"]))
        quotient = n[cation] * n["Cl-"] / WATER_KG**2 / ksp
        a, nr = p["geometry"][m]
        area = a * (n[m] / nr) ** (2 / 3)
        minerals[m] = {
            "si": None if quotient == 0 else log(quotient),
            "area": area,
            "drive": 1 - quotient,
            "rate": p["coefficient"][m] * area * (1 - quotient),
        }
    return gas, minerals


def history(kind, condition, times, max_step=2.0):
    p = parameters(kind, condition)
    initial = p["initial"]
    gases = [k + "(g)" for k in p["K"]]
    active = [m for m in p["minerals"] if initial[m] > 0 and p["coefficient"][m] > 0]
    current, start, segments, events = initial.copy(), 0.0, [], []
    while start < times[-1]:
        baseline, keys = current.copy(), tuple(active)

        def decode(y, baseline=baseline, keys=keys):
            n = {**baseline, **dict(zip(gases, y[: len(gases)], strict=True))}
            n.update(
                {m: baseline[m] * max(0.0, y[len(gases) + j]) ** 3 for j, m in enumerate(keys)}
            )
            return close(p, n)

        def rhs(t, y, keys=keys, baseline=baseline, decode=decode):
            n = decode(y)
            gas, mineral = rates(p, n)
            return [b - a for a, b in gas.values()] + [
                -p["coefficient"][m]
                * p["geometry"][m][0]
                / (3 * p["geometry"][m][1] ** (2 / 3) * baseline[m] ** (1 / 3))
                * mineral[m]["drive"]
                for m in keys
            ]

        roots = []
        for j in range(len(keys)):

            def event(t, y, j=j):
                return y[len(gases) + j]

            event.terminal, event.direction = True, -1
            roots.append(event)
        solution = solve_ivp(
            rhs,
            (start, times[-1]),
            [*[current[k] for k in gases], *[1.0 for m in keys]],
            method="DOP853",
            rtol=2e-12,
            atol=1e-15,
            dense_output=True,
            events=roots,
            max_step=max_step,
        )
        if not solution.success:
            raise ValueError(solution.message)
        end = float(solution.t[-1])
        y = solution.y[:, -1].copy()
        for j, found in enumerate(solution.t_events):
            if len(found):
                y[len(gases) + j] = 0.0
                events.append({"mineral_id": keys[j], "time_s": end})
                active.remove(keys[j])
        current = decode(y)
        segments.append((start, end, solution.sol, decode))
        start = end
    result = []
    before = properties(p, initial)
    for time in times:
        n = current.copy()
        for lo, hi, solution, decode in segments:
            if lo <= time < hi:
                n = decode(solution(time))
                break
        props = properties(p, n)
        gas, mineral = rates(p, n)
        result.append(
            {
                "time_s": time,
                "amounts": n,
                "properties": props,
                "gas": gas,
                "mineral": mineral,
                "heat": props["U"] - before["U"],
            }
        )
    return result, events


def final_root(kind, condition):
    p = parameters(kind, condition)
    gases = [k + "(g)" for k in p["K"]]
    keys = gases + p["minerals"]

    def decode(y):
        return close(p, {**p["initial"], **dict(zip(keys, y, strict=True))})

    def residual(y):
        n = decode(y)
        props = properties(p, n)
        equations = [
            n[k + "(aq)"] / WATER_KG - p["K"][k] * n[k + "(g)"] * R * p["T"] / props["Vg"] / 1e5
            for k in p["K"]
        ]
        for m in p["minerals"]:
            c = p["cation"][m]
            gm = p["states"][m]["g"] + p["volume"][m] * (props["P"] - 1e5)
            ksp = exp((gm - p["states"][c]["g"] - p["states"]["Cl-"]["g"]) / (R * p["T"]))
            equations.append((n[c] * n["Cl-"] / WATER_KG**2 - ksp) / sqrt(ksp))
        return equations

    solved = root(residual, [p["initial"][k] for k in keys], tol=1e-11)
    if not solved.success or max(abs(np.asarray(residual(solved.x)))) > 1e-10:
        raise ValueError("independent final coexistence root failed")
    n = decode(solved.x)
    if min(n.values()) < 0:
        raise ValueError("coexistence root left finite nonnegative domain")
    return n, residual(solved.x)
