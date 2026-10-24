"""Independent thermo PR tie-lines, linear stage balances and finite tank PH.

No ChemWorld physical implementation is imported by this reference.
"""

import numpy as np
from scipy.optimize import brentq

from reference.molecular28_caloric_reference import oracle


def homogeneous(index, amounts, *, temperature=None, enthalpy=None):
    total = sum(amounts)
    zs = [n / total for n in amounts]
    base = oracle().gas if index % 2 == 0 else oracle().liquid

    def phase(t):
        return base.to(T=t, P=6e5, zs=zs)

    if temperature is None:
        lo, hi = (325.0, 360.0) if index % 2 == 0 else (265.0, 295.0)
        temperature = brentq(lambda t: total * phase(t).H() - enthalpy, lo, hi, xtol=1e-10)
    value = phase(temperature)
    return {
        "T": temperature,
        "H": total * value.H(),
        "V": total * value.V(),
        "U": total * (value.H() - 6e5 * value.V()),
        "rho": 1 / value.V(),
        "h": value.H(),
    }


def steady(temperatures, ratio):
    count = len(temperatures)
    phases = []
    for t in temperatures:
        flash = oracle().flash(T=t, P=6e5, zs=[0.6, 0.4])
        if flash.gas is None or not flash.liquids:
            raise ValueError("independent fixed tie-line seed is outside two-phase region")
        phases.append((flash.gas, flash.liquids[0]))
    a = np.zeros((2 * count, 2 * count))
    b = np.zeros(2 * count)
    vin, lin = 0.001 * np.array([0.95, 0.05]), 0.001 * ratio * np.array([0.05, 0.95])
    for j, pair in enumerate(phases):
        for k in range(2):
            a[2 * j + k, 2 * j : 2 * j + 2] = [p.zs[k] for p in pair]
            if j == 0:
                b[2 * j + k] += lin[k]
            else:
                a[2 * j + k, 2 * (j - 1) + 1] = -phases[j - 1][1].zs[k]
            if j == count - 1:
                b[2 * j + k] += vin[k]
            else:
                a[2 * j + k, 2 * (j + 1)] = -phases[j + 1][0].zs[k]
    flows = np.linalg.solve(a, b).reshape(count, 2)
    inlet_h = [
        homogeneous(0, vin, temperature=340.0)["H"],
        homogeneous(1, lin, temperature=280.0)["H"],
    ]
    rows = []
    for j, pair in enumerate(phases):
        hin_v = inlet_h[0] if j == count - 1 else flows[j + 1, 0] * phases[j + 1][0].H()
        hin_l = inlet_h[1] if j == 0 else flows[j - 1, 1] * phases[j - 1][1].H()
        rows.append(
            {
                "rates": flows[j].tolist(),
                "compositions": [p.zs for p in pair],
                "densities": [1 / p.V() for p in pair],
                "enthalpies": [p.H() for p in pair],
                "Q": sum(f * p.H() for f, p in zip(flows[j], pair, strict=True)) - hin_v - hin_l,
            }
        )
    return {
        "success": bool(np.all(flows > 0)),
        "stages": rows,
        "component_balance_error": float(np.max(np.abs(a @ flows.ravel() - b))),
        "feed_rates": [vin.tolist(), lin.tolist()],
        "feed_enthalpies_W": inlet_h,
        "failure_reason": "" if np.all(flows > 0) else "no_positive_two_phase_flow_solution",
    }


def finite(initial_amounts, temperatures, ratio, duration):
    reference = steady(temperatures, ratio)
    if not reference["success"]:
        return {"success": False, "steady": reference}
    rows = reference["stages"]
    n0 = np.array(initial_amounts)
    initial = [
        homogeneous(i, n, temperature=340.0 if i % 2 == 0 else 280.0) for i, n in enumerate(n0)
    ]
    outgoing = [
        np.array(rows[0]["compositions"][0]) * rows[0]["rates"][0],
        np.array(rows[-1]["compositions"][1]) * rows[-1]["rates"][1],
    ]
    rates = np.array([-np.array(r) for r in reference["feed_rates"]] + outgoing)
    h_rates = np.array(
        [-h for h in reference["feed_enthalpies_W"]]
        + [
            rows[0]["rates"][0] * rows[0]["enthalpies"][0],
            rows[-1]["rates"][1] * rows[-1]["enthalpies"][1],
        ]
    )
    ns = n0 + duration * rates
    hs = np.array([p["H"] for p in initial]) + duration * h_rates
    states = [homogeneous(i, n, enthalpy=h) for i, (n, h) in enumerate(zip(ns, hs, strict=True))]
    return {
        "success": True,
        "steady": reference,
        "amounts": ns.tolist(),
        "enthalpies": hs.tolist(),
        "initial": initial,
        "final": states,
        "rates": rates.tolist(),
        "h_rates": h_rates.tolist(),
        "Q": duration * sum(s["Q"] for s in rows),
    }
