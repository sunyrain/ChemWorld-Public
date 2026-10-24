"""Independent continuous dual-pore BVP and RK45 finite-liquid reference.

Only declarations are read; no production pore, property, controller or finite
advance evaluator is called. Concentrations use original solvent-volume basis.
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq

R, F = 8.31446261815324, 96485.33212331001
CALORIC = {
    "H2O": (75.0, 0.0, 18e-6),
    "Na+": (40.0, 1000.0, 2e-6),
    "Ca2+": (45.0, 500.0, 3e-6),
    "Cl-": (35.0, -500.0, 3e-6),
    "SO4--": (50.0, -1000.0, 4e-6),
}


def partition(charges, c, fixed):
    psi = brentq(lambda v: np.sum(charges * c * np.exp(-charges * v)) + fixed, -40, 40, xtol=1e-14)
    return c * np.exp(-charges * psi), psi


def point(model, amounts, target, unit):
    ids = [k for k, _ in model.aem.effective_diffusivities_m2_s]
    z = np.array(
        [{s.species_id: s.charge for s in model.properties.database.species}[k] for k in ids]
    )
    c = [np.array([n[k] / (n["H2O"] * 18e-6) for k in ids]) for n in amounts]
    count = len(ids)
    pores = (model.aem, model.cem)
    t = model.temperature_K
    ends = []
    partitions = []
    pressure = []
    scales = []
    for pore, left, right in ((pores[0], c[1], c[0]), (pores[1], c[0], c[1])):
        a, pa = partition(z, left, pore.fixed_charge_mol_m3)
        b, pb = partition(z, right, pore.fixed_charge_mol_m3)
        ends.append((a, b))
        partitions.append((pa, pb))
        pressure.append(R * t * (sum(a) - sum(left) - sum(b) + sum(right)))
        diffusion = np.array([v for _, v in pore.effective_diffusivities_m2_s])
        cs = max(max(a), max(b), abs(pore.fixed_charge_mol_m3), 1)
        scales.append(
            (diffusion, cs, diffusion * cs / pore.thickness_m, max(diffusion) / pore.thickness_m)
        )
    js = max(1, *[F * max(s[2]) for s in scales])
    vs = R * t / F
    approximate = model.solution_resistance_Ohm_m2 + sum(
        R * t * p.thickness_m / (F * F * np.sum(z * z * s[0] * (a + b) / 2))
        for p, s, (a, b) in zip(pores, scales, ends, strict=True)
    )
    j0 = target / model.membrane_area_m2 if unit == "A" else target / approximate

    def decode(parameters, index):
        values = parameters[index * count : (index + 1) * count]
        independent = values[:-1] * scales[index][2][:-1]
        j = parameters[-1] * js
        flux = np.r_[independent, (j / F - z[:-1] @ independent) / z[-1]]
        return flux, values[-1] * scales[index][3], j

    def ode(x, y, parameters):
        rows = []
        for i, pore in enumerate(pores):
            local = y[i * count : (i + 1) * count]
            independent = np.exp(local[:-1])
            # Local electroneutrality is a coordinate identity, not an ODE
            # invariant with a redundant integration constant.
            dependent = (-pore.fixed_charge_mol_m3 - z[:-1] @ independent) / z[-1]
            conc = np.vstack((independent, dependent))
            flux, v, _ = decode(parameters, i)
            drift = (conc * v - flux[:, None]) / scales[i][0][:, None]
            electric = (
                pore.thickness_m
                * np.sum(z[:, None] * drift, axis=0)
                / np.sum(z[:, None] ** 2 * conc, axis=0)
            )
            rows.extend(
                (
                    pore.thickness_m * drift[:-1] / independent - z[:-1, None] * electric,
                    electric[None, :],
                )
            )
        return np.vstack(rows)

    def boundary(left, right, parameters):
        rows = []
        voltage = 0
        for i, pore in enumerate(pores):
            first = left[i * count : (i + 1) * count]
            r = right[i * count : (i + 1) * count]
            _, v, j = decode(parameters, i)
            water = v - pore.hydraulic_permeability_m_Pa_s * (
                pressure[i] + R * t * pore.fixed_charge_mol_m3 * (r[-1] - first[-1])
            )
            rows.extend(
                (
                    first[:-1] - np.log(ends[i][0][:-1]),
                    r[:-1] - np.log(ends[i][1][:-1]),
                    [first[-1] - partitions[i][0], water / scales[i][3]],
                )
            )
            voltage += vs * (partitions[i][1] - r[-1])
        voltage += j * model.solution_resistance_Ohm_m2
        rows.append(
            [(j - target / model.membrane_area_m2) / js if unit == "A" else (voltage - target) / vs]
        )
        return np.concatenate(rows)

    x = np.linspace(0, 1, 31)
    ys = []
    ps = []
    for i, pore in enumerate(pores):
        a, b = ends[i]
        pa, pb = partitions[i]
        diff, _, fs, ws = scales[i]
        endphi = pb - j0 * pore.thickness_m / (F * np.sum(z * z * diff * (a + b) / 2))
        phi = np.linspace(pa, endphi, len(x))
        conc = np.linspace(a, b, len(x)).T
        velocity = pore.hydraulic_permeability_m_Pa_s * (
            pressure[i] + R * t * pore.fixed_charge_mol_m3 * (endphi - pa)
        )
        flux = (
            -diff * (b - a) / pore.thickness_m
            - z * diff * (a + b) / 2 * (endphi - pa) / pore.thickness_m
            + (a + b) / 2 * velocity
        )
        ys.extend((np.log(conc[:-1]), phi[None, :]))
        ps.extend((flux[:-1] / fs[:-1], [velocity / ws]))
    parameters = np.r_[*ps, j0 / js]
    solved = solve_bvp(
        ode, boundary, x, np.vstack(ys), p=parameters, tol=1e-8, bc_tol=1e-10, max_nodes=2000
    )
    if not solved.success:
        raise RuntimeError("independent continuous ED BVP: " + solved.message)
    fluxes = []
    velocities = []
    readouts = []
    voltage = 0
    for i, pore in enumerate(pores):
        flux, v, j = decode(solved.p, i)
        fluxes.append(flux)
        velocities.append(v)
        sampled = solved.sol(np.linspace(0, 1, pore.segments + 1))[
            i * count : (i + 1) * count
        ]
        independent = np.exp(sampled[:-1])
        dependent = (-pore.fixed_charge_mol_m3 - z[:-1] @ independent) / z[-1]
        concentrations = np.vstack((independent, dependent))
        if not np.all(np.isfinite(concentrations)) or np.any(concentrations <= 0):
            raise RuntimeError("independent continuous ED BVP returned nonpositive concentration")
        drop = vs * (partitions[i][1] - sampled[-1, -1])
        voltage += drop
        readouts.append(
            {
                "ion_fluxes_mol_m2_s": dict(zip(ids, map(float, flux), strict=True)),
                "water_flux_m_s": float(v),
                "voltage_drop_V": float(drop),
                "pore_concentrations_mol_m3": concentrations.T.tolist(),
                "pore_dimensionless_potential": sampled[-1].tolist(),
            }
        )
    voltage += j * model.solution_resistance_Ohm_m2
    current = j * model.membrane_area_m2
    rates = {
        "H2O": model.membrane_area_m2 * (velocities[0] - velocities[1]) / 18e-6,
        **dict(zip(ids, map(float, model.membrane_area_m2 * (fluxes[0] - fluxes[1])), strict=True)),
    }
    return {
        "current_A": float(current),
        "voltage_V": float(voltage),
        "electrical_power_W": float(current * voltage),
        "diluate_rates_mol_s": rates,
        "aem": readouts[0],
        "cem": readouts[1],
        "bvp_nodes": len(solved.x),
        "maximum_bvp_residual": float(max(solved.rms_residuals)),
    }


def volume(n):
    return sum(v * CALORIC[k][2] for k, v in n.items())


def enthalpy(n, t):
    return sum(v * (CALORIC[k][1] + CALORIC[k][0] * (t - 298.15)) for k, v in n.items())


def finite(model, initial, target, unit, duration, *, ramp=None):
    ids = tuple(initial[0])
    width = len(ids)
    evaluations = 0
    nodes = 0

    def unpack(y):
        return tuple(dict(zip(ids, row, strict=True)) for row in y[:-1].reshape(2, width))

    def control(t):
        return target if ramp is None else ramp[0] + (target - ramp[0]) * min(t / ramp[1], 1)

    def rhs(t, y):
        nonlocal evaluations, nodes
        row = point(model, unpack(y), control(t), unit)
        evaluations += 1
        nodes += row["bvp_nodes"]
        rates = np.array([row["diluate_rates_mol_s"][k] for k in ids])
        return np.r_[rates, -rates, row["electrical_power_W"]]

    stops = []
    for i in range(2):
        for name, function in (
            (f"water_minimum:{i}", lambda t, y, i=i: unpack(y)[i]["H2O"] - model.minimum_water_mol),
            (
                f"capacity:{i}",
                lambda t, y, i=i: model.capacities_L[i] - 1000 * volume(unpack(y)[i]),
            ),
        ):
            function.terminal = True
            function.direction = -1
            stops.append((name, function))
    y0 = np.r_[[[n[k] for k in ids] for n in initial]].ravel()
    y0 = np.r_[y0, 0.0]
    boundaries = sorted({duration, *([ramp[1]] if ramp is not None and ramp[1] < duration else [])})
    current = y0
    start = 0
    stop = None
    for end in boundaries:
        solved = solve_ivp(
            rhs,
            (start, end),
            current,
            method="RK45",
            rtol=2e-9,
            atol=1e-12,
            events=[f for _, f in stops],
        )
        if not solved.success:
            raise RuntimeError("independent finite ED integration failed: " + solved.message)
        current = solved.y[:, -1]
        start = float(solved.t[-1])
        events = [
            name for (name, _), times in zip(stops, solved.t_events, strict=True) if len(times)
        ]
        if events:
            stop = events[0]
            break
    ns = unpack(current)
    return {
        "amounts_mol": ns,
        "enthalpies_J": [enthalpy(n, model.temperature_K) for n in ns],
        "electrical_work_J": float(current[-1]),
        "bath_heat_into_liquid_J": -float(current[-1]),
        "actual_duration_s": start,
        "stopping_event": stop,
        "rhs_evaluations": evaluations,
        "bvp_nodes_total": nodes,
    }
