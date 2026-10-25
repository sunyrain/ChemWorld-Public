"""Independent authored half-cell equations: analytic one-step and DOP853 HER.

No native rate, thermo evaluator, coupled process or result is called here. This
qualifies the declared synthetic equations, not an actual CatMAP installation.
"""

from math import exp, log

import numpy as np
from scipy.integrate import solve_ivp

R = 8.31446261815324
F = 96485.33212
T0 = 298.15
CAPACITY = 1e-6


def thermo(h0, s0, cp, temperature):
    h = h0 + cp * (temperature - T0)
    s = s0 + cp * log(temperature / T0)
    return h, h - temperature * s


def coefficients(kind, temperature, potential):
    _, gh = thermo(0, 50, 15, temperature)
    _, gads = thermo(-10000, 0, 10, temperature)
    if kind == "hydroxylation":
        _, go = thermo(-25000, 0, 10, temperature)
        _, goh = thermo(-35000, 0, 20, temperature)
        dg = goh - go - gh + F * potential
    else:
        dg = gads - gh + F * potential

    def pair(delta):
        return (
            1000 * exp(-(30000 + 0.5 * delta) / (R * temperature)),
            1000 * exp(-(30000 - 0.5 * delta) / (R * temperature)),
        )

    kf, kr = pair(dg)
    _, gg = thermo(0, 100, 30, temperature)
    tf, tr = pair(gg - 2 * gads)
    return kf, kr, tf, tr


def equilibrium_potential(kind, temperature, pH, theta=0.5):
    _, gh = thermo(0, 50, 15, temperature)
    if kind == "hydroxylation":
        _, a = thermo(-25000, 0, 10, temperature)
        _, b = thermo(-35000, 0, 20, temperature)
        dg = b - a - gh
    else:
        _, b = thermo(-10000, 0, 10, temperature)
        dg = b - gh
    return -(dg + R * temperature * log(theta / ((1 - theta) * 10 ** (-pH)))) / F


def independent(kind, temperature, pH, potential, duration, *, pressure_limit=None):
    kf, kr, tf, tr = coefficients(kind, temperature, potential)
    a = 10 ** (-pH)
    theta0, gas0, gas_volume = 0.1, 1e-6, 1e-5
    if kind != "her":
        forward, reverse = kf * a, kr
        steady = forward / (forward + reverse)
        theta = steady + (theta0 - steady) * exp(-(forward + reverse) * duration)
        d_ads = CAPACITY * (theta - theta0)
        charge = -F * d_ads
        gas = gas0
        elapsed = duration
    else:

        def rhs(t, y):
            theta, gas = y[:2]
            progress = CAPACITY * (kf * a * (1 - theta) - kr * theta)
            h2 = CAPACITY * (
                tf * theta**2
                - tr * (gas * R * temperature / gas_volume / 101325) * (1 - theta) ** 2
            )
            return ((progress - 2 * h2) / CAPACITY, h2, -F * progress)

        events = None
        if pressure_limit is not None:

            def event(t, y):
                return pressure_limit - y[1] * R * temperature / gas_volume

            event.terminal = True
            event.direction = -1
            events = event
        result = solve_ivp(
            rhs,
            (0, duration),
            (theta0, gas0, 0),
            method="DOP853",
            rtol=2e-12,
            atol=np.array([1e-14, 1e-20, 1e-15]),
            events=events,
        )
        if not result.success:
            raise RuntimeError(result.message)
        theta, gas, charge = result.y[:, -1]
        d_ads = CAPACITY * (theta - theta0)
        elapsed = float(result.t[-1])
    proton_in = -charge / F
    ug = thermo(0, 100, 30, temperature)[0] - R * temperature
    if kind == "hydroxylation":
        chemical = (
            thermo(-35000, 0, 20, temperature)[0] - thermo(-25000, 0, 10, temperature)[0]
        ) * d_ads
    else:
        chemical = thermo(-10000, 0, 10, temperature)[0] * d_ads + (gas - gas0) * ug
    inlet_h = proton_in * thermo(0, 50, 15, temperature)[0]
    work = potential * charge
    return {
        "elapsed_s": elapsed,
        "theta": float(theta),
        "gas_mol": float(gas),
        "faradaic_charge_C": float(charge),
        "electrical_work_J": float(work),
        "proton_net_in_mol": float(proton_in),
        "proton_net_enthalpy_J": float(inlet_h),
        "material_U_change_J": float(chemical),
        "bath_heat_J": float(chemical - work - inlet_h),
    }
