"""Independent binary closed-form surfaces, scalar binodals and lever rules."""

from math import log

from scipy.optimize import brentq

R = 8.3145


def interaction(family, temperature):
    if family == "REGULAR":
        return 12000.0, 0.0
    if family == "LOOP":
        return (
            2 * R * temperature + 0.1 * (temperature - 550) * (850 - temperature),
            2 * R + 0.1 * (1400 - 2 * temperature),
        )
    return 0.0, 0.0


def surface(family, temperature, x, phase="ALPHA"):
    omega, slope = interaction(family, temperature)
    mix = sum(v * log(v) for v in (x, 1 - x) if v > 0)
    gc = 30 * (temperature - 600 - temperature * log(temperature / 600))
    offset = 10000 * (1 - temperature / 700) if phase == "BETA" else 0.0
    g = gc + R * temperature * mix + omega * x * (1 - x) + offset
    s = 30 * log(temperature / 600) - R * mix - slope * x * (1 - x)
    h = 30 * (temperature - 600) + (omega - temperature * slope) * x * (1 - x)
    if phase == "BETA":
        h += 10000
        s += 10000 / 700
    mu = {}
    if 0 < x < 1:
        gx = R * temperature * log(x / (1 - x)) + omega * (1 - 2 * x)
        mu = {"FE": g + (1 - x) * gx, "NI": g - x * gx}
    else:
        mu["FE" if x == 1 else "NI"] = g
    return {"G": g, "H": h, "S": s, "mu": mu}


def binodal(family, temperature):
    omega, _ = interaction(family, temperature)
    if omega <= 2 * R * temperature * (1 + 1e-13):
        return None
    left = brentq(
        lambda x: log(x / (1 - x)) + omega / (R * temperature) * (1 - 2 * x),
        1e-12,
        0.5 - 1e-7,
        xtol=1e-14,
    )
    return left, 1 - left


def equilibrium(family, temperature, z):
    phase = "BETA" if family == "CROSSING" and temperature > 700 else "ALPHA"
    ends = binodal(family, temperature)
    if ends is not None and ends[0] < z < ends[1]:
        left, right = ends
        pairs = [(left, (right - z) / (right - left)), (right, (z - left) / (right - left))]
    else:
        pairs = [(z, 1.0)]
    rows = [
        {"phase": phase, "x": x, "fraction": n, **surface(family, temperature, x, phase)}
        for x, n in pairs
    ]
    return {
        "phases": rows,
        "G": sum(r["fraction"] * r["G"] for r in rows),
        "H": sum(r["fraction"] * r["H"] for r in rows),
        "mu": rows[0]["mu"],
        "nonunique": family == "CROSSING" and temperature == 700,
    }
