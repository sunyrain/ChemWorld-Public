"""Independent thermo flash, actual phase flows and finite PH endpoints."""

import numpy as np
from scipy.optimize import brentq

from reference.molecular28_caloric_reference import oracle


def homogeneous(label, amounts, *, temperature=None, enthalpy=None):
    total = sum(amounts)
    zs = [n / total for n in amounts]
    base = oracle().gas if label == "gas" else oracle().liquid

    def phase(t):
        return base.to(T=t, P=6e5, zs=zs)

    if temperature is None:
        bounds = (325.0, 360.0) if label == "gas" else (265.0, 295.0)
        temperature = brentq(lambda t: total * phase(t).H() - enthalpy, *bounds, xtol=1e-10)
    p = phase(temperature)
    return {
        "T": temperature,
        "V": total * p.V(),
        "H": total * p.H(),
        "U": total * (p.H() - 6e5 * p.V()),
        "rho": 1 / p.V(),
        "h": p.H(),
    }


def steady(temperature, z, hot=False):
    flash = oracle().flash(T=temperature, P=6e5, zs=[z, 1 - z])
    labels, phases, fractions = [], [], []
    if flash.gas is not None:
        labels.append("gas")
        phases.append(flash.gas)
        fractions.append(flash.VF)
    if flash.liquids:
        labels.append("liquid")
        phases.append(flash.liquids[0])
        fractions.append(1 - flash.VF)
    rows = [
        {"label": label, "rate": 0.001 * f, "x": p.zs, "rho": 1 / p.V(), "h": p.H()}
        for label, p, f in zip(labels, phases, fractions, strict=True)
    ]
    inlet = homogeneous(
        "gas" if hot else "liquid",
        [z * 0.001, (1 - z) * 0.001],
        temperature=340.0 if hot else 280.0,
    )
    return {
        "labels": labels,
        "phases": rows,
        "feed_h_rate": inlet["H"],
        "Q": sum(row["rate"] * row["h"] for row in rows) - inlet["H"],
    }


def finite(amounts, temperature, z, hot, duration):
    original = np.array(amounts, dtype=float)
    phases = ("gas" if hot else "liquid", "gas", "liquid")
    initial = [
        homogeneous(label, n, temperature=(340.0 if label == "gas" else 280.0))
        for label, n in zip(phases, original, strict=True)
    ]
    ref = steady(temperature, z, hot)
    dn = np.zeros_like(original)
    dh = np.zeros(3)
    dn[0] = -0.001 * np.array([z, 1 - z])
    dh[0] = -ref["feed_h_rate"]
    for row in ref["phases"]:
        index = 1 if row["label"] == "gas" else 2
        dn[index] = row["rate"] * np.array(row["x"])
        dh[index] = row["rate"] * row["h"]
    ns = original + duration * dn
    hs = np.array([v["H"] for v in initial]) + duration * dh
    final = [homogeneous(label, n, enthalpy=h) for label, n, h in zip(phases, ns, hs, strict=True)]
    return {
        "steady": ref,
        "amounts": ns.tolist(),
        "enthalpies": hs.tolist(),
        "initial": initial,
        "final": final,
        "Q": ref["Q"] * duration,
        "pressure_work": 6e5 * (sum(p["V"] for p in final) - sum(p["V"] for p in initial)),
    }
