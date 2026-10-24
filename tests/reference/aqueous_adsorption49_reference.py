"""Independent finite adsorption ODE, analytic pump and no-uptake CSTR exponential."""

from math import exp, pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from reference.aqueous_adsorption49_fixtures import AFFINITIES, DATA, ENERGIES, WATER_VOLUME


def independent(spec, initial, requested, duration, path="process"):
    keys = list(DATA)
    rows = np.asarray([[n[k] for k in keys] for n in initial.amounts_mol])
    loading = np.array([initial.sorbent_slot.values[f"loading:{k}"] for k in ("A", "B")])
    volumes = rows[:, 0] * WATER_VOLUME
    source, target = (0, 2) if path == "process" else (3, 4)
    resistance = 128 * 0.001 * spec.pipe.length_m / (pi * spec.pipe.diameter_m**4)
    q = min(
        requested,
        spec.pump.maximum_pressure_rise_Pa / resistance,
        sqrt(spec.pump.maximum_shaft_power_W * spec.pump.efficiency / resistance),
    )
    dt, stop = float(duration), None
    if q:
        for candidate, name in (
            ((volumes[source] - spec.pickup_volume_L / 1000) / q, f"{path}_source_pickup"),
            ((spec.capacities_L[target] / 1000 - volumes[target]) / q, f"{path}_collector_full"),
        ):
            if candidate < dt:
                dt, stop = max(float(candidate), 0.0), name
    inlet = rows[source] * q / volumes[source]
    dilution = q / volumes[1]
    affinity = np.array(
        [
            AFFINITIES[k]
            * exp(-ENERGIES[k] / 8.31446261815324 * (1 / spec.temperature_K - 1 / 298.15))
            for k in ("A", "B")
        ]
    )
    rates = np.array([spec.sorbent.mass_transfer_rates_s_inv[k] for k in ("A", "B")])
    capacity = spec.media_mass_kg * spec.sorbent.site_capacity_mol_per_kg

    def rhs(_, y):
        n, loaded = y[:15].reshape((5, 3)), y[15:]
        c = n[1, 1:] / (volumes[1] * 1000)
        target_loading = capacity * affinity * c / (1 + affinity @ c)
        uptake = rates * (target_loading - loaded)
        positive, negative = np.maximum(uptake, 0), np.maximum(-uptake, 0)
        available = negative.sum() + rates.max() * max(capacity - loaded.sum(), 0.0)
        if positive.sum() > available:
            uptake = positive * (available / positive.sum()) - negative
        dn = np.zeros((5, 3))
        outgoing = dilution * n[1]
        dn[source] -= inlet
        dn[1] += inlet - outgoing
        dn[target] += outgoing
        dn[1, 1:] -= uptake
        return np.r_[dn.ravel(), uptake]

    if not rates.any():
        matrix = np.zeros((16, 16))
        for j in range(3):
            matrix[source * 3 + j, 15] = -inlet[j]
            matrix[3 + j, 15] = inlet[j]
            matrix[3 + j, 3 + j] = -dilution
            matrix[target * 3 + j, 3 + j] = dilution
        output = (expm(matrix * dt) @ np.r_[rows.ravel(), 1.0])[:15].reshape((5, 3))
        bound = loading
    elif dt == 0:
        output, bound = rows, loading
    else:
        result = solve_ivp(
            rhs, (0.0, dt), np.r_[rows.ravel(), loading], method="DOP853", rtol=2e-13, atol=1e-15
        )
        if not result.success:
            raise RuntimeError(result.message)
        output, bound = result.y[:15, -1].reshape((5, 3)), result.y[15:, -1]
    h = np.array([DATA[k][2] + DATA[k][1] * (spec.temperature_K - 298.15) for k in keys])
    enthalpies = output @ h
    du = float(
        (output - rows).sum(axis=0) @ h
        + (bound - loading) @ np.array([h[1] + ENERGIES["A"], h[2] + ENERGIES["B"]])
    )
    hydraulic = resistance * q * q * dt
    shaft = hydraulic / spec.pump.efficiency
    # In the single closed batch, the physical root satisfies
    # b*q^2 -(V+b*(N+S))*q + S*b*N = 0. Stable small-root formula.
    total = rows[1, 1] + loading[0]
    b = affinity[0]
    v = volumes[1] * 1000
    linear = v + b * (total + capacity)
    equilibrium = (
        2 * capacity * b * total / (linear + sqrt(linear * linear - 4 * b * b * capacity * total))
    )
    return {
        "amounts": output.tolist(),
        "loading": bound.tolist(),
        "enthalpies": enthalpies.tolist(),
        "dt": dt,
        "stop": stop,
        "throughput_m3": q * dt,
        "pump_work_J": shaft,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": shaft - hydraulic,
        "bath_heat_into_system_J": du - hydraulic,
        "internal_energy_change_J": du,
        "equilibrium_A_mol": equilibrium,
    }
