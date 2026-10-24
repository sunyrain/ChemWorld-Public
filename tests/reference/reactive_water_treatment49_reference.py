"""Independent laminar operating point, matrix exponential and explicit ODE.

Does not call production transport, rate, caloric or treatment functions.
"""

from math import pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from reference.reactive_water_treatment49_fixtures import DATA, molar_volumes


def independent(spec, initial, requested, duration):
    keys, v = list(DATA), molar_volumes(spec)
    h = np.asarray([DATA[k][3] + DATA[k][2] * (spec.temperature_K - 298.15) for k in keys])
    resistance = 128 * 0.001 * spec.pipe.length_m / (pi * spec.pipe.diameter_m**4)
    q = min(
        requested,
        spec.pump.maximum_pressure_rise_Pa / resistance,
        sqrt(spec.pump.maximum_shaft_power_W * spec.pump.efficiency / resistance),
    )
    rows = np.asarray([[n[k] for k in keys] for n in initial.amounts_mol])
    volumes = rows @ np.asarray(list(v.values()))
    dt, stop = duration, None
    if q:
        for candidate, event in (
            ((volumes[0] - spec.pickup_volume_L / 1000) / q, "feed_pickup_exhausted"),
            ((spec.capacities_L[2] / 1000 - volumes[2]) / q, "product_full"),
        ):
            if candidate < dt:
                dt, stop = max(candidate, 0.0), event
    inlet, dilution = rows[0] * q / volumes[0], q / volumes[1]
    nonlinear = spec.network.reactions[0].reaction_id == "dimer"
    if not nonlinear:
        reaction = np.zeros((4, 4))
        reaction[1, 1], reaction[2, 1] = -0.04, 0.04
        if len(spec.network.reactions) == 2:
            reaction[1, 2], reaction[2, 2] = 0.015, -0.015
        matrix = np.zeros((9, 9))
        matrix[:4, :4] = reaction - dilution * np.eye(4)
        matrix[:4, 8] = inlet
        matrix[4:8, :4] = dilution * np.eye(4)
        final = expm(matrix * dt) @ np.r_[rows[1], rows[2], 1.0]
    else:

        def rhs(_, x):
            r = 0.8 * x[1] ** 2 / (volumes[1] * 1000)
            chemistry = np.asarray([0.0, -2 * r, 0.0, r])
            return np.r_[inlet - dilution * x[:4] + chemistry, dilution * x[:4]]

        ode = solve_ivp(
            rhs, (0.0, dt), np.r_[rows[1], rows[2]], method="DOP853", rtol=2e-13, atol=1e-15
        )
        if not ode.success:
            raise RuntimeError(ode.message)
        final = ode.y[:, -1]
    output = np.vstack((rows[0] - inlet * dt, final[:4], final[4:8]))
    enthalpies = output @ h
    hydraulic = resistance * q * q * dt
    shaft = hydraulic / spec.pump.efficiency
    return {
        "amounts": output.tolist(),
        "enthalpies": enthalpies.tolist(),
        "dt": dt,
        "stop": stop,
        "throughput_m3": q * dt,
        "pump_work_J": shaft,
        "hydraulic_work_J": hydraulic,
        "motor_loss_J": shaft - hydraulic,
        "bath_heat_into_liquids_J": float((output - rows).sum(axis=0) @ h - hydraulic),
        "reactor_volume_L": float(output[1] @ np.asarray(list(v.values())) * 1000),
    }
