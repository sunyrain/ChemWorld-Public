"""Independent scalar thermodynamics and temperature-coordinate RK45 reference.

Only scalar declarations are consumed. No production property/flux/PH/advance
evaluation is called. Constitutive constants are fixed in the prospective note.
"""

from math import exp, log, log1p, pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp

R, MW, VW, TR = 8.31446261815324, 0.018015, 18e-6, 298.15


def terms(m):
    root = sqrt(m)
    return (
        -log(10) * (root / (1 + root) - 0.3 * m),
        -4 * log(10) * (m / 2 - root + log1p(root)) + log(10) * 0.3 * m * m,
    )


def liquid(family, n, t, p):
    water, salt = n["H2O"], n["Na+"]
    kg, m = water * MW, salt / (water * MW)
    gamma, excess = terms(m) if family == "Davies" else (0.0, 0.0)
    a = 0.509 + 2e-4 * (t - TR) + 1e-9 * (p - 1e5)
    lnaw = MW * (a * excess - 2 * m * a * gamma - 2 * m)
    h = (water * 75 + (n["Na+"] * 40 + n["Cl-"] * 35)) * (t - TR) + n["Na+"] * 1000 - n["Cl-"] * 500
    v0 = water * VW + n["Na+"] * 2e-6 + n["Cl-"] * 3e-6
    h += v0 * (p - 1e5) - R * t * t * kg * 2e-4 * excess
    v = v0 + R * t * kg * 1e-9 * excess
    gl = 75 * (t - TR) - t * (10 + 75 * log(t / TR)) + VW * (p - 1e5)
    gv = 44000 + 34 * (t - TR) - t * (130 + 34 * log(t / TR))
    pv = 1e5 * exp((gl + R * t * lnaw - gv) / (R * t))
    cp = water * 75 + n["Na+"] * 40 + n["Cl-"] * 35 - 2 * R * t * kg * 2e-4 * excess
    # d(w Gex_per_kg)/dw = MW*(Gex - 2m*lngamma), at fixed ion amounts.
    partial_h_water = (
        75 * (t - TR) + VW * (p - 1e5) - R * t * t * 2e-4 * MW * (excess - 2 * m * gamma)
    )
    return {
        "enthalpy_J": h,
        "volume_m3": v,
        "vapor_pressure_Pa": pv,
        "heat_capacity_J_K": cp,
        "partial_water_H_J_mol": partial_h_water,
    }


def point(model, ns, temperatures):
    family = model.properties.database.activity_model.family
    liquids = [
        liquid(family, n, t, p)
        for n, t, p in zip(ns, temperatures, model.pressures_Pa, strict=True)
    ]
    mean = sum(temperatures) / 2
    coefficient = (
        2
        * model.porosity
        * model.pore_radius_m
        / (3 * model.tortuosity * model.thickness_m)
        * sqrt(8 / (pi * MW * R * mean))
    )
    flux = coefficient * (liquids[0]["vapor_pressure_Pa"] - liquids[1]["vapor_pressure_Pa"])
    cond = (
        model.membrane_area_m2
        * model.effective_conductivity_W_m_K
        / model.thickness_m
        * (temperatures[0] - temperatures[1])
    )
    hv = 44000 + 34 * (mean - TR)
    return {
        "water_flux_mol_m2_s": flux,
        "water_transfer_mol_s": flux * model.membrane_area_m2,
        "conductive_heat_W": cond,
        "vapor_enthalpy_flow_W": flux * model.membrane_area_m2 * hv,
        "membrane_energy_flow_W": cond + flux * model.membrane_area_m2 * hv,
    }


def utility(device, t):
    if not device.enabled:
        return 0.0
    return min(
        device.maximum_power_W,
        max(
            -device.maximum_cooling_power_W,
            device.conductance_W_K * device.calibration_factor * (device.setpoint_K - t),
        ),
    )


def finite(model, ns, temperatures, heater, cooler, duration):
    family = model.properties.database.activity_model.family

    def unpack(y):
        return tuple({**n, "H2O": y[i]} for i, n in enumerate(ns))

    def rhs(time, y):
        current = unpack(y)
        p = point(model, current, y[2:4])
        dn = np.array([-p["water_transfer_mol_s"], p["water_transfer_mol_s"]])
        qh, qc = utility(heater, y[2]), -utility(cooler, y[3])
        hdot = np.array([qh - p["membrane_energy_flow_W"], p["membrane_energy_flow_W"] - qc])
        liquids = [
            liquid(family, n, t, pressure)
            for n, t, pressure in zip(current, y[2:4], model.pressures_Pa, strict=True)
        ]
        dtemp = [
            (hdot[i] - liquids[i]["partial_water_H_J_mol"] * dn[i])
            / (liquids[i]["heat_capacity_J_K"] + model.support_heat_capacities_J_K[i])
            for i in range(2)
        ]
        return np.r_[dn, dtemp, qh, qc]

    initial = [ns[0]["H2O"], ns[1]["H2O"], *temperatures, 0, 0]
    result = solve_ivp(
        rhs,
        (0, duration),
        initial,
        method="RK45",
        rtol=1e-11,
        atol=1e-12,
        max_step=max(duration / 50, 1e-6),
    )
    if not result.success:
        raise RuntimeError(result.message)
    y = result.y[:, -1]
    final = unpack(y)
    h = [
        liquid(family, n, t, p)["enthalpy_J"] + c * (t - model.reference_temperature_K)
        for n, t, p, c in zip(
            final, y[2:4], model.pressures_Pa, model.support_heat_capacities_J_K, strict=True
        )
    ]
    return {
        "amounts_mol": final,
        "temperatures_K": list(y[2:4]),
        "enthalpies_J": h,
        "heater_J": float(y[4]),
        "cooler_J": float(y[5]),
        "evaluations": result.nfev,
    }
