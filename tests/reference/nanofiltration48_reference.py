"""Independent continuous NP collocation and explicit synthetic calorics.

Only declarations are read. No native membrane, hydraulic, activity, inventory
advance, residual or property evaluator is imported or called.
"""

import math

import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq

R = 8.31446261815324
E, EPS0, KB = 1.602176634e-19, 8.8541878128e-12, 1.380649e-23
ROWS = {
    "H2O": (75.0, 0.0, 18e-6),
    "Na+": (40.0, 1000.0, 2e-6),
    "Ca2+": (45.0, 500.0, 3e-6),
    "Cl-": (35.0, -500.0, 3e-6),
    "SO4--": (50.0, -1000.0, 4e-6),
}


def factors(model):
    result = []
    charges = {s.species_id: s.charge for s in model.properties.database.species}
    for ion in model.ions:
        ratio = ion.stokes_radius_m / model.pore_radius_m
        kd = (
            0.984 * ((1 - ratio) / ratio) ** 2.5
            if ratio > 0.95
            else (
                1
                + 1.125 * ratio * np.log(ratio)
                - 1.56034 * ratio
                + 0.528155 * ratio**2
                + 1.91521 * ratio**3
                - 2.81903 * ratio**4
                + 0.270788 * ratio**5
                - 1.10115 * ratio**6
                - 0.435933 * ratio**7
            )
            / (1 - ratio) ** 2
        )
        kc = (1 + 3.867 * ratio - 1.907 * ratio**2 - 0.834 * ratio**3) / (
            1 + 1.867 * ratio - 0.741 * ratio**2
        )
        born = np.exp(
            -(charges[ion.species_id] ** 2)
            * E**2
            / (8 * np.pi * EPS0 * ion.born_radius_m * KB * model.temperature_K)
            * (1 / model.pore_relative_permittivity - 1 / model.bulk_relative_permittivity)
        )
        result.append(((1 - ratio) ** 2 * born, kd * ion.diffusivity_m2_s, kc))
    return np.array(result).T


def donnan(model, c):
    z = np.array([s.charge for s in model.properties.database.species if s.charge])
    k = factors(model)[0]
    phi = brentq(
        lambda v: np.sum(z * k * c * np.exp(-z * v)) + model.fixed_charge_mol_m3,
        -40,
        40,
        xtol=1e-14,
    )
    return k * c * np.exp(-z * phi), phi


def point(model, concentrations, solvent_flow_m3_s=math.inf):
    ids = tuple(i.species_id for i in model.ions)
    z = np.array([s.charge for s in model.properties.database.species if s.charge])
    ci = np.array([concentrations[k] for k in ids])
    k, diffusion, convection = factors(model)
    hydraulic = model.pore_radius_m**2 / (8 * 0.001 * model.effective_thickness_m)
    dp = model.inlet_pressure_Pa - model.reservoir_pressure_Pa
    jscale, cscale = hydraulic * dp, max(max(ci), abs(model.fixed_charge_mol_m3), 1)
    count = len(ci)
    if not any(ci):
        return {"water_flux_m_s": jscale, "permeate_concentrations_mol_m3": dict.fromkeys(ids, 0.0)}

    def exterior(cp, j):
        recovery = model.membrane_area_m2 * j / solvent_flow_m3_s
        cr = (ci - recovery * cp) / (1 - recovery)
        return (ci + cr) / 2

    def ode(x, y, params):
        c, cp, j = np.exp(y[:-1]), np.exp(params[:count]), jscale * np.exp(params[-1])
        drift = j * (convection[:, None] * c - cp[:, None]) / diffusion[:, None]
        electric = (
            model.effective_thickness_m
            * np.sum(z[:, None] * drift, axis=0)
            / np.sum(z[:, None] ** 2 * c, axis=0)
        )
        derivative = model.effective_thickness_m * drift - z[:, None] * c * electric
        return np.vstack((derivative / c, electric))

    def boundary(left, right, params):
        cp, pp, j = np.exp(params[:count]), params[-2], jscale * np.exp(params[-1])
        cf = exterior(cp, j)
        return np.concatenate(
            (
                left[:-1] - np.log(k * cf) + z * left[-1],
                right[:-1] - np.log(k * cp) + z * (right[-1] - pp),
                [
                    (np.exp(left[:-1]) @ z + model.fixed_charge_mol_m3) / cscale,
                    cp @ z / cscale,
                    j / jscale - 1 + R * model.temperature_K * np.sum(cf - cp) / dp,
                ],
            )
        )

    cp0 = ci * 0.8
    cleft, pleft = donnan(model, ci)
    cright, pright = donnan(model, cp0)
    x = np.linspace(0, 1, 31)
    y = np.vstack(
        (np.log(np.linspace(cleft, cright, len(x))).T, np.linspace(pleft, pright, len(x)))
    )
    params = np.concatenate((np.log(cp0), [0.0, np.log(0.7)]))
    solved = solve_bvp(ode, boundary, x, y, p=params, tol=1e-9, bc_tol=1e-10, max_nodes=2000)
    if not solved.success:
        raise RuntimeError(f"independent continuous NP BVP failed: {solved.message}")
    cp, j = np.exp(solved.p[:count]), jscale * np.exp(solved.p[-1])
    sampled = solved.sol(np.linspace(0, 1, model.pore_segments + 1))
    return {
        "water_flux_m_s": float(j),
        "permeate_concentrations_mol_m3": dict(zip(ids, map(float, cp), strict=True)),
        "pore_concentrations_mol_m3": np.exp(sampled[:-1]).T.tolist(),
        "pore_dimensionless_potential": sampled[-1].tolist(),
        "nodes": len(solved.x),
        "max_residual": float(max(solved.rms_residuals)),
    }


def volume(amounts):
    return sum(n * ROWS[k][2] for k, n in amounts.items())


def enthalpy(amounts, temperature):
    return sum(n * (ROWS[k][1] + ROWS[k][0] * (temperature - 298.15)) for k, n in amounts.items())


def finite(model, initial, requested, duration):
    feed = initial[0]
    dp = model.inlet_pressure_Pa - model.reservoir_pressure_Pa
    pump = model.pump
    flow = (
        0.0
        if dp >= pump.maximum_pressure_rise_Pa
        else min(requested, pump.maximum_shaft_power_W * pump.efficiency / dp)
    )
    incoming = {k: n * flow / volume(feed) for k, n in feed.items()}
    qwater = incoming["H2O"] * 18e-6
    pore = point(
        model, {k: n / (feed["H2O"] * 18e-6) for k, n in feed.items() if k != "H2O"}, qwater
    )
    j = pore["water_flux_m_s"]
    permeate = {
        "H2O": model.membrane_area_m2 * j / 18e-6,
        **{
            k: model.membrane_area_m2 * j * c
            for k, c in pore["permeate_concentrations_mol_m3"].items()
        },
    }
    retained = {k: incoming[k] - permeate[k] for k in incoming}
    rates = ({k: -v for k, v in incoming.items()}, permeate, retained)
    times = [
        (duration, None),
        ((volume(feed) - model.pickup_volume_L / 1000) / flow, "feed_pickup_exhausted"),
        (
            (model.permeate_capacity_L / 1000 - volume(initial[1])) / volume(permeate),
            "permeate_full",
        ),
        (
            (model.retentate_capacity_L / 1000 - volume(initial[2])) / volume(retained),
            "retentate_full",
        ),
    ]
    dt, stop = min(times, key=lambda t: t[0])
    final = [{k: n + dt * rates[i][k] for k, n in row.items()} for i, row in enumerate(initial)]
    h = [enthalpy(row, model.temperature_K) for row in final]
    change = sum(h) - sum(enthalpy(row, model.temperature_K) for row in initial)
    hydraulic = dp * flow * dt
    shaft = hydraulic / pump.efficiency
    return {
        "amounts_mol": final,
        "enthalpies_J": h,
        "actual_duration_s": dt,
        "stopping_event": stop,
        "shaft_work_J": shaft,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": shaft - hydraulic,
        "bath_heat_into_liquid_J": change - hydraulic,
    }
