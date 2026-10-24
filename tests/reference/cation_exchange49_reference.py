"""Independent molality/GT scalar equilibrium and finite ODE reference."""

from math import exp, pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq

from reference.cation_exchange49_fixtures import DATA, WATER_MASS, WATER_VOLUME


def independent(spec, initial, requested, duration, path="process"):
    rows = np.asarray([[n[k] for k in DATA] for n in initial.amounts_mol])
    loading = np.array(
        [initial.resin_slot.values[k] for k in ("bound_sodium_mol", "bound_calcium_mol")]
    )
    volumes = rows[:, 0] * WATER_VOLUME
    source, target = {"process": (0, 2), "regenerate": (3, 5), "rinse": (4, 5)}[path]
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
    capacity = spec.media_mass_kg * spec.charge_capacity_mol_kg
    selectivity = spec.reference_selectivity * exp(
        -spec.exchange_enthalpy_J_mol
        / 8.31446261815324
        * (1 / spec.temperature_K - 1 / spec.reference_temperature_K)
    )
    water_kg = rows[1, 0] * WATER_MASS

    def rhs(_, y):
        n, loaded = y[:24].reshape((6, 4)), y[24:]
        ana, aca = n[1, 1:3] / water_kg
        tna, tca = loaded * np.array([1.0, 2.0]) / capacity
        exchange = (
            capacity * spec.rate_coefficient_s_inv * (aca * tna**2 - ana**2 * tca / selectivity)
        )
        dn = np.zeros((6, 4))
        outgoing = dilution * n[1]
        dn[source] -= inlet
        dn[1] += inlet - outgoing
        dn[target] += outgoing
        dn[1, 1] += 2 * exchange
        dn[1, 2] -= exchange
        return np.r_[dn.ravel(), [-2 * exchange, exchange]]

    if spec.rate_coefficient_s_inv == 0:
        matrix = np.zeros((25, 25))
        for j in range(4):
            matrix[source * 4 + j, 24] = -inlet[j]
            matrix[4 + j, 24] = inlet[j]
            matrix[4 + j, 4 + j] = -dilution
            matrix[target * 4 + j, 4 + j] = dilution
        output = (expm(matrix * dt) @ np.r_[rows.ravel(), 1.0])[:24].reshape((6, 4))
        bound = loading
    elif dt == 0:
        output, bound = rows, loading
    else:
        result = solve_ivp(
            rhs, (0.0, dt), np.r_[rows.ravel(), loading], method="DOP853", rtol=2e-13, atol=1e-15
        )
        if not result.success:
            raise RuntimeError(result.message)
        output, bound = result.y[:24, -1].reshape((6, 4)), result.y[24:, -1]
    h = np.array([DATA[k][2] + DATA[k][1] * (spec.temperature_K - 298.15) for k in DATA])
    du = float(
        (output - rows).sum(axis=0) @ h
        + (bound - loading) @ np.array([h[1], h[2] + spec.exchange_enthalpy_J_mol])
    )
    hydraulic = resistance * q * q * dt
    shaft = hydraulic / spec.pump.efficiency
    total_na, total_ca = rows[1, 1:3] + loading
    lo, hi = max(0.0, (capacity - total_na) / 2), min(capacity / 2, total_ca)

    def equilibrium(x):
        ca_aq, na_aq = total_ca - x, total_na - capacity + 2 * x
        return (
            2 * x / capacity * (na_aq / water_kg) ** 2
            - selectivity * ((capacity - 2 * x) / capacity) ** 2 * ca_aq / water_kg
        )

    root = float(lo if lo == hi else brentq(equilibrium, lo, hi, xtol=1e-16))
    return {
        "amounts": output.tolist(),
        "loading": bound.tolist(),
        "enthalpies": (output @ h).tolist(),
        "dt": dt,
        "stop": stop,
        "throughput_m3": q * dt,
        "pump_work_J": shaft,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": shaft - hydraulic,
        "bath_heat_into_system_J": du - hydraulic,
        "internal_energy_change_J": du,
        "equilibrium_Ca_mol": root,
    }
