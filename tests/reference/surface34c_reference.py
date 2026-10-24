"""Independent public Cantera thermochemistry, kinetics and finite history reference."""

from __future__ import annotations

import json
from itertools import product

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp

TIMES = (0.0, 0.025, 0.1, 0.5, 1.0)
COVERAGES = (0.55, 0.2, 0.15, 0.1)
R = ct.gas_constant / 1000
AREA = 0.1
VOLUME = 0.001


def cases():
    rows = []
    for t, size, density in product((400.0, 700.0, 1000.0), (2, 3), (1e-5, 3e-5)):
        rows.append(
            {
                "name": f"state-{len(rows)}",
                "kind": "state",
                "temperature": t,
                "size": size,
                "density": density,
                "expected_comparisons": 12,
            }
        )
    for t, size in product((400.0, 700.0, 1000.0), (2, 3)):
        rows.append(
            {
                "name": f"history-{len(rows) - 12}",
                "kind": "history",
                "temperature": t,
                "size": size,
                "density": 1e-5,
                "expected_comparisons": 30,
            }
        )
    for size, iso in product((2, 3), (True, False)):
        rows.append(
            {
                "name": f"world-{size}-{iso}",
                "kind": "world",
                "temperature": 700.0,
                "size": size,
                "density": 1e-5,
                "isothermal": iso,
                "expected_comparisons": 10,
            }
        )
    for size in (2, 3):
        rows.append(
            {
                "name": f"structure-{size}",
                "kind": "structure",
                "temperature": 700.0,
                "size": size,
                "density": 1e-5,
                "expected_comparisons": 6,
            }
        )
    return rows


def reference(payload, temperature, *, pressure=1e5, coverages=COVERAGES):
    surface = ct.Interface(yaml=json.dumps(payload), name="surface")
    gas = surface.adjacent["gas"]
    gas.TPX = temperature, pressure, [0.7, 0.3]
    surface.TP = temperature, pressure
    surface.coverages = coverages
    return gas, surface


def sizes(surface):
    return np.array([s.size for s in surface.species()])


def inventory(gas, surface, *, coverages=None):
    theta = surface.coverages if coverages is None else coverages
    return np.concatenate(
        (
            gas.concentrations * 1000 * VOLUME,
            theta * surface.site_density * 1000 * AREA / sizes(surface),
        )
    )


def elements(amounts, gas, surface):
    species = [*gas.species(), *surface.species()]
    symbols = sorted({key for s in species for key in s.composition})
    return np.array(
        [
            sum(n * s.composition.get(e, 0) for n, s in zip(amounts, species, strict=True))
            for e in symbols
        ]
    )


def caloric(temperature, gas, surface):
    species = [*gas.species(), *surface.species()]
    u = np.array([s.thermo.h(temperature) / 1000 for s in species])
    cv = np.array([s.thermo.cp(temperature) / 1000 for s in species])
    u[: gas.n_species] -= R * temperature
    cv[: gas.n_species] -= R
    return u, cv


def energy(amounts, temperature, gas, surface):
    u, _ = caloric(temperature, gas, surface)
    return float(amounts @ u + temperature - 298.15)


def finite_history(payload, temperature):
    gas, surface = reference(payload, temperature)
    reactor = ct.IdealGasReactor(gas, energy="off", volume=VOLUME, clone=False)
    rs = ct.ReactorSurface(surface, reactor, A=AREA, clone=False)
    network = ct.ReactorNet([reactor])
    network.rtol, network.atol, network.max_steps = 1e-11, 1e-20, 100000
    rows = []
    for t in TIMES:
        if t:
            network.advance(t)
        amounts = inventory(reactor.phase, surface, coverages=rs.coverages)
        rows.append(
            {
                "time_s": t,
                "amounts": amounts,
                "temperature": reactor.T,
                "elements": elements(amounts, gas, surface),
                "energy": energy(amounts, reactor.T, gas, surface),
            }
        )
    return rows


def thermal_endpoint(payload, temperature, isothermal):
    gas, surface = reference(payload, temperature)
    initial = inventory(gas, surface)
    capacity = surface.site_density * 1000 * AREA
    site_sizes = sizes(surface)

    def rhs(time, y):
        amounts, t = y[:-1], y[-1]
        gas.TPX = t, sum(amounts[:2]) * R * t / VOLUME, amounts[:2]
        surface.TP = t, gas.P
        surface.set_unnormalized_coverages(amounts[2:] * site_sizes / capacity)
        rates = (
            np.concatenate(
                (surface.get_net_production_rates(gas), surface.get_net_production_rates(surface))
            )
            * 1000
            * AREA
        )
        u, cv = caloric(t, gas, surface)
        dt = 0.0 if isothermal else -float(u @ rates) / (1 + float(amounts @ cv))
        return np.concatenate((rates, [dt]))

    solved = solve_ivp(
        rhs,
        (0.0, 1.0),
        np.concatenate((initial, [temperature])),
        method="Radau",
        rtol=1e-10,
        atol=1e-16,
    )
    if not solved.success:
        raise ValueError("independent thermal reference failed: " + solved.message)
    return {
        "amounts": solved.y[:-1, -1],
        "temperature": solved.y[-1, -1],
        "nfev": solved.nfev,
        "initial": initial,
    }
