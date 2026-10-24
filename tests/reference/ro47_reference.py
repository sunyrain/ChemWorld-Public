"""Independent elementary equations and RK4 for the declared synthetic RO fixture.

No native property/activity/RO/hydraulic/finite-inventory evaluator is called.
Only authored scalar declarations are read from the fixture model.
"""

from math import exp, log, log1p, pi, sqrt

import numpy as np
from ro47_fixtures import ROWS, WATER_MASS, WATER_VOLUME
from scipy.optimize import root_scalar

R = 8.31446261815324


def coefficient(law, t, p, c):
    return law.value * exp(
        law.activation_energy_J_mol / R * (1 / law.reference_temperature_K - 1 / t)
        + law.per_Pa * (p - law.reference_pressure_Pa)
        + law.per_mol_m3 * c
    )


def davies_terms(m):
    if m == 0:
        return 0.0, 0.0
    root = sqrt(m)
    gamma_per_a = -log(10) * (root / (1 + root) - 0.3 * m)
    excess_per_a = -4 * log(10) * (m / 2 - root + log1p(root)) + log(10) * 0.3 * m * m
    return gamma_per_a, excess_per_a


def properties(family, n, t, p):
    if not any(n.values()):
        return {"enthalpy_J": 0.0, "volume_m3": 0.0}
    water, salt = n["H2O"], n["Na+"]
    kg, m = water * WATER_MASS, salt / (water * WATER_MASS)
    gamma, excess = davies_terms(m) if family == "Davies" else (0.0, 0.0)
    a = 0.509 + 2e-4 * (t - 298.15) + 1e-9 * (p - 1e5)
    lnaw = WATER_MASS * (a * excess - 2 * m * a * gamma - 2 * m)
    h = sum(
        n[key] * (cp * (t - 298.15) + href + volume * (p - 1e5))
        for key, _, _, cp, href, volume in ROWS
    )
    v = sum(n[key] * volume for key, _, _, _, _, volume in ROWS)
    h -= R * t * t * kg * 2e-4 * excess
    v += R * t * kg * 1e-9 * excess
    return {"enthalpy_J": h, "volume_m3": v, "log_water_activity": lnaw}


def local(model, c, pressure):
    t, pr = model.temperature_K, model.reservoir_pressure_Pa
    family = model.properties.database.activity_model.family
    a, b, k = (
        coefficient(law, t, pressure, c)
        for law in (model.water_permeability, model.salt_permeability, model.mass_transfer)
    )

    def residual(j):
        # Independently eliminate salt from Js=B(Cm-Cp), Cp=Js/Jw,
        # Cm=Cp+(Cb-Cp)exp(Jw/k).
        factor = exp(j / k)
        cp = c * b * factor / (b * factor + j)
        cm = cp + (c - cp) * factor
        nf, np_ = ({"H2O": 1.0, "Na+": z * WATER_VOLUME, "Cl-": z * WATER_VOLUME} for z in (cm, cp))
        af = properties(family, nf, t, pressure)["log_water_activity"]
        ap = properties(family, np_, t, pr)["log_water_activity"]
        delta_mu = WATER_VOLUME * (pressure - pr) + R * t * (af - ap)
        return j - a * delta_mu / WATER_VOLUME, cp, cm, delta_mu

    upper = 4 * a * (pressure - pr)
    answer = root_scalar(
        lambda j: residual(j)[0], bracket=(0.0, upper), method="bisect", xtol=2e-16
    ).root
    _, cp, cm, delta = residual(answer)
    return {
        "water_flux_m_s": answer,
        "salt_flux_mol_m2_s": answer * cp,
        "permeate_concentration_mol_m3": cp,
        "interface_concentration_mol_m3": cm,
        "water_chemical_potential_difference_J_mol": delta,
    }


def profile(model, feed, flow, cells=128):
    t, p0 = model.temperature_K, model.inlet_pressure_Pa
    family = model.properties.database.activity_model.family
    volume = properties(family, feed, t, model.reservoir_pressure_Pa)["volume_m3"]
    incoming = {k: v * flow / volume for k, v in feed.items()}
    y = np.array([incoming["H2O"], incoming["Na+"], p0], dtype=float)

    def rhs(v):
        w, s, p = v
        n = {"H2O": w, "Na+": s, "Cl-": s}
        c = s / (w * WATER_VOLUME)
        q = properties(family, n, t, p)["volume_m3"]
        mu = coefficient(model.properties.viscosity, t, p, c)
        # These authored small circular channels are laminar. The independent
        # reference is the closed Poiseuille law, not the production pipe model.
        gradient = 128 * mu * q / (pi * model.channel.diameter_m**4)
        flux = local(model, c, p)
        return np.array(
            [
                -model.membrane_area_m2 * flux["water_flux_m_s"] / WATER_VOLUME,
                -model.membrane_area_m2 * flux["salt_flux_mol_m2_s"],
                -model.channel.length_m * gradient,
            ]
        )

    step = 1 / cells
    for _ in range(cells):
        k1 = rhs(y)
        k2 = rhs(y + step * k1 / 2)
        k3 = rhs(y + step * k2 / 2)
        k4 = rhs(y + step * k3)
        y += step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return {
        "water_mol_s": float(y[0]),
        "salt_mol_s": float(y[1]),
        "pressure_Pa": float(y[2]),
        "inlet_mol_s": incoming,
        "retentate_mol_s": {"H2O": float(y[0]), "Na+": float(y[1]), "Cl-": float(y[1])},
        "permeate_mol_s": {
            "H2O": incoming["H2O"] - float(y[0]),
            "Na+": incoming["Na+"] - float(y[1]),
            "Cl-": incoming["Cl-"] - float(y[1]),
        },
    }
