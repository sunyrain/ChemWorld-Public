"""Elementary Gibbs and scalar RK4 oracle; no native physical evaluator.

Read only authored constants and explicit model declarations. Salt remains a
fixed scalar in each compartment and x is transferred water, independently of
the production two-inventory BDF representation.
"""

import sys
from math import log
from pathlib import Path

from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).parent))
from ro47_fixtures import ROWS, WATER_MASS, WATER_VOLUME
from ro47_reference import R, coefficient, davies_terms, properties


def thermo(family, n, t, p):
    base = properties(family, n, t, p)
    kg = n["H2O"] * WATER_MASS
    m = n["Na+"] / kg
    gamma, excess = davies_terms(m) if family == "Davies" else (0.0, 0.0)
    a = 0.509 + 2e-4 * (t - 298.15) + 1e-9 * (p - 1e5)
    g = sum(
        n[key] * (cp * (t - 298.15) + href + v * (p - 1e5) - t * (10 + cp * log(t / 298.15)))
        for key, _, _, cp, href, v in ROWS
    )
    g += R * t * (2 * n["Na+"] * (log(m) - 1) if m else 0)
    g += R * t * kg * a * excess
    partial_h_water = 75 * (t - 298.15) + WATER_VOLUME * (p - 1e5)
    partial_h_water -= R * t * t * WATER_MASS * 2e-4 * (excess - 2 * m * gamma)
    return {
        **base,
        "gibbs_J": g,
        "internal_energy_J": base["enthalpy_J"] - p * base["volume_m3"],
        "partial_water_enthalpy_J_mol": partial_h_water,
    }


def point(model, amounts):
    t, pressures = model.temperature_K, model.pressures_Pa
    family = model.properties.database.activity_model.family
    phases = [thermo(family, n, t, p) for n, p in zip(amounts, pressures, strict=True)]
    c = [n["Na+"] / (n["H2O"] * WATER_VOLUME) for n in amounts]
    delta = WATER_VOLUME * (pressures[0] - pressures[1]) + R * t * (
        phases[0]["log_water_activity"] - phases[1]["log_water_activity"]
    )
    mobility = coefficient(model.water_permeability, t, sum(pressures) / 2, sum(c) / 2)
    transfer = model.membrane_area_m2 * mobility * delta / WATER_VOLUME**2
    hdot = [
        -transfer * phases[0]["partial_water_enthalpy_J_mol"],
        transfer * phases[1]["partial_water_enthalpy_J_mol"],
    ]
    return {
        "delta_mu": delta,
        "flow": transfer,
        "enthalpy_rates_W": hdot,
        "bath_W": sum(hdot),
        "entropy_W_K": transfer * delta / t,
        "volumes_L": [1000 * q["volume_m3"] for q in phases],
        "concentrations": c,
    }


def transferred(initial, x):
    return tuple(
        {**n, "H2O": n["H2O"] + sign * x} for n, sign in zip(initial, (-1, 1), strict=True)
    )


def endpoint(model, initial, x):
    final = transferred(initial, x)
    family, t = model.properties.database.activity_model.family, model.temperature_K
    before, after = [
        [thermo(family, n, t, p) for n, p in zip(ns, model.pressures_Pa, strict=True)]
        for ns in (initial, final)
    ]
    q = sum(b["enthalpy_J"] - a["enthalpy_J"] for a, b in zip(before, after, strict=True))
    work = sum(
        p * (b["volume_m3"] - a["volume_m3"])
        for p, a, b in zip(model.pressures_Pa, before, after, strict=True)
    )
    return {
        "amounts": final,
        "enthalpies_J": [p["enthalpy_J"] for p in after],
        "volumes_L": [1000 * p["volume_m3"] for p in after],
        "heat_J": q,
        "work_J": work,
        "internal_energy_change_J": sum(
            b["internal_energy_J"] - a["internal_energy_J"]
            for a, b in zip(before, after, strict=True)
        ),
        "entropy_J_K": -sum(b["gibbs_J"] - a["gibbs_J"] for a, b in zip(before, after, strict=True))
        / t,
    }


def trajectory(model, initial, times=(0, 5, 10, 20), steps=256):
    x, now, rows = 0.0, 0.0, []

    def rhs(x):
        return point(model, transferred(initial, x))["flow"]

    for time in times:
        dt = (time - now) / steps
        for _ in range(steps if dt else 0):
            k1 = rhs(x)
            k2 = rhs(x + dt * k1 / 2)
            k3 = rhs(x + dt * k2 / 2)
            k4 = rhs(x + dt * k3)
            x += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        rows.append({"time_s": time, **endpoint(model, initial, x)})
        now = time
    return rows


def event_time(model, initial, transfer=0.005):
    return quad(
        lambda x: 1 / point(model, transferred(initial, x))["flow"],
        0,
        transfer,
        epsabs=1e-10,
        epsrel=1e-11,
    )[0]
