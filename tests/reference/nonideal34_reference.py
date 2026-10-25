"""Independent Cantera PR and raw PR/extent-U numerical references.

Neither reference imports native EOS derivatives, molecular calorics or rates.
Source declarations are authored inputs, not values produced by the native model.
"""

from math import exp, log, sqrt

import cantera as ct
import numpy as np
import yaml
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .nonideal34_cases import cantera_source

R = ct.gas_constant / 1000


def phase(family, temperature, amounts, *, volume=1.0, multiple=True, interaction=0.0):
    gas = ct.Solution(
        yaml=yaml.safe_dump(cantera_source(family, multiple=multiple, interaction=interaction)),
        name="synthetic",
    )
    gas.X = dict(amounts)
    density = sum(
        amounts[k] * gas.molecular_weights[i] / volume for i, k in enumerate(gas.species_names)
    )
    gas.TD = temperature, density
    return gas


def properties(gas):
    return {
        "P": gas.P,
        "U": gas.int_energy_mole / 1000,
        "H": gas.enthalpy_mole / 1000,
        "S": gas.entropy_mole / 1000,
        "Cv": gas.cv_mole / 1000,
        "Cp": gas.cp_mole / 1000,
        "activity": (gas.activities * gas.P / (ct.gas_constant * gas.T)).tolist(),
        "rates": gas.net_rates_of_progress.tolist(),
    }


def trajectory(family, *, duration=0.3):
    gas = phase(family, 600.0, {"A": 1.0, "B": 0.3, "C": 0.2}, multiple=family == "constant-cp")
    reactor = ct.Reactor(gas, clone=True, volume=0.001)
    net = ct.ReactorNet([reactor])
    net.rtol, net.atol = 1e-11, 1e-15
    net.advance(duration)
    return {
        "T": reactor.T,
        "P": reactor.phase.P,
        "n": reactor.phase.concentrations.tolist(),
        "resources": dict(net.solver_stats),
    }


def raw_pr(temperature, amounts):
    """Public algebraic PR mixing, departures and fugacity; no native calls."""
    n = np.array(amounts, dtype=float)
    total = sum(n)
    x, rho = n / total, total * 1000
    b_i = np.array([3e-5, 4e-5, 5e-5])
    tc = np.array([280.0, 320.0, 360.0])
    w = np.array([0.05, 0.08, 0.11])
    k = 0.37464 + 1.54226 * w - 0.26992 * w**2
    a0 = 0.45723552892138218 / 0.0777960739038885 * b_i * R * tc
    q = 1 + k * (1 - np.sqrt(temperature / tc))
    ai = a0 * q * q
    dai = -a0 * q * k / np.sqrt(temperature * tc)
    aij = np.sqrt(ai[:, None] * ai[None, :])
    daij = aij * (dai[:, None] / ai[:, None] + dai[None, :] / ai[None, :]) / 2
    a, b, da = float(x @ aij @ x), float(x @ b_i), float(x @ daij @ x)
    v = 1 / rho
    p = R * temperature / (v - b) - a / (v * v + 2 * b * v - b * b)
    z = p / (rho * R * temperature)
    logratio = log((v + (1 + sqrt(2)) * b) / (v + (1 - sqrt(2)) * b))
    ur = (temperature * da - a) / (2 * sqrt(2) * b) * logratio
    h0 = np.array([-10000.0, -12000.0, -14000.0])
    hi = h0 + 32 * (temperature - 298.15)
    u = float(x @ hi) - R * temperature + ur
    ap, bp = a * p / (R * temperature) ** 2, b * p / (R * temperature)
    logphi = (
        b_i / b * (z - 1)
        - log(z - bp)
        - ap / (2 * sqrt(2) * bp) * (2 * (aij @ x) / a - b_i / b) * logratio
    )
    activities = z * np.exp(logphi) * n  # 1 L authored vessel
    g = hi - temperature * (30 + 32 * log(temperature / 298.15))
    standard = 101325 / (1000 * R * temperature)
    kc1 = exp(-(g[1] - g[0]) / (R * temperature))
    kc2 = exp(-(g[2] - 2 * g[1]) / (R * temperature)) / standard
    k1 = 0.2 * temperature**0.1 * exp(-1000 / (R * temperature))
    rates = [
        k1 * (activities[0] - activities[1] / kc1),
        0.03 * (activities[1] ** 2 - activities[2] / kc2),
    ]
    return {"P": p, "U": u, "rates": rates}


def extent_trajectory(*, duration=0.3, power=20.0):
    initial = np.array([1.0, 0.3, 0.2])
    initial_u = sum(initial) * raw_pr(600.0, initial)["U"]
    stoich = np.array([[-1.0, 0.0], [1.0, -2.0], [0.0, 1.0]])

    def unpack(time, extents):
        amounts = initial + stoich @ extents
        t = brentq(
            lambda v: sum(amounts) * raw_pr(v, amounts)["U"] - initial_u - power * time,
            400.0,
            1400.0,
            xtol=1e-10,
        )
        return amounts, t

    def rhs(time, extents):
        amounts, t = unpack(time, extents)
        return raw_pr(t, amounts)["rates"]

    result = solve_ivp(rhs, (0.0, duration), (0.0, 0.0), method="DOP853", rtol=1e-11, atol=1e-13)
    if not result.success:
        raise RuntimeError(result.message)
    amounts, t = unpack(duration, result.y[:, -1])
    return {"n": amounts.tolist(), "T": t, "nfev": result.nfev}
