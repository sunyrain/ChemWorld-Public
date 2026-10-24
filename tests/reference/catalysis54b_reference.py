"""Independent authored algebra and public Cantera kinetics for fixed 54B units."""

from __future__ import annotations

import json
from itertools import product
from math import exp

import cantera as ct
import numpy as np
from surface34b_cases import source_payload

R = 8.31446261815324
CAPACITY = 1e-6


def cases():
    rows = []
    for disconnected, temperature, pressure in product((False, True), (500.0, 900.0), (2e4, 8e4)):
        rows.append(
            {
                "name": f"analytic-{len(rows)}",
                "kind": "analytic",
                "disconnected": disconnected,
                "temperature": temperature,
                "pressure": pressure,
                "expected_comparisons": 10,
            }
        )
    for temperature, feed in product(
        (700.0, 1000.0, 1500.0), ("CH4:1,O2:2,N2:7", "CH4:1,O2:1,N2:8")
    ):
        rows.append(
            {
                "name": f"ptcombust-{len(rows) - 8}",
                "kind": "source",
                "temperature": temperature,
                "feed": feed,
                "expected_comparisons": 8,
            }
        )
    for disconnected in (False, True):
        rows.append(
            {
                "name": f"basis-{int(disconnected)}",
                "kind": "basis",
                "disconnected": disconnected,
                "expected_comparisons": 4,
            }
        )
    rows.extend(
        {"name": name, "kind": name, "expected_comparisons": 4}
        for name in ("incompatible", "zero-mode")
    )
    rows.extend(
        {"name": "world-" + str(iso), "kind": "world", "isothermal": iso, "expected_comparisons": 6}
        for iso in (True, False)
    )
    return rows


def analytic(disconnected, temperature, pressure):
    k = [
        a * exp(-ea / (R * temperature))
        for a, ea in (
            (0.5, 500),
            (2, 1000),
            (1, 2000),
            (4, 1500),
            (3, 2500),
            (0.3, 750),
            (2.5, 1250),
        )
    ]
    k[0] *= pressure / (R * temperature)
    k[5] *= pressure / (R * temperature)
    pool = 0.6 if disconnected else 0.9
    x = pool / ((k[1] + k[2]) / k[0] + 1 + k[1] / k[3] + k[2] / k[4])
    theta = {
        "E": (k[1] + k[2]) * x / k[0],
        "X": x,
        "Y": k[1] * x / k[3],
        "Z": k[2] * x / k[4],
        "I": 0.1,
    }
    forward = [k[0] * theta["E"], k[1] * x, k[2] * x, k[3] * theta["Y"], k[4] * theta["Z"]]
    jac = np.array([[-k[0] - k[1] - k[2], -k[0], -k[0]], [k[1], -k[3], 0], [k[2], 0, -k[4]]])
    eig = list(np.linalg.eigvals(jac))
    if disconnected:
        w = 0.3 * k[5] / (k[5] + k[6])
        theta.update(F=0.3 - w, W=w)
        forward.extend((k[5] * (0.3 - w), k[6] * w))
        eig.append(-k[5] - k[6])
    forward = np.asarray(forward) * CAPACITY
    gas = {
        "A": -forward[0] - (forward[5] if disconnected else 0),
        "B": forward[3],
        "C": forward[4] + (forward[6] if disconnected else 0),
    }
    selectivity = {k: gas[k] / (gas["B"] + gas["C"]) for k in ("B", "C")}
    return {
        "coverages": theta,
        "forward": forward,
        "reverse": np.zeros(len(forward)),
        "gas": gas,
        "eigenvalues": eig,
        "selectivity": selectivity,
    }


def source_reference(payload, temperature, partial_pressures, coverages, *, area=0.1):
    surface = ct.Interface(yaml=json.dumps(payload), name=payload["phases"][1]["name"])
    gas = surface.adjacent["gas"]
    pressure = sum(partial_pressures.values())
    gas.TPX = temperature, pressure, partial_pressures
    surface.TP = temperature, pressure
    surface.coverages = coverages
    surface.advance_coverages(
        1e4, rtol=1e-9, atol=1e-18, max_steps=100000, max_error_test_failures=20
    )
    theta = surface.coverages.copy()

    def rates(values):
        surface.set_unnormalized_coverages(values)
        return np.asarray(surface.get_net_production_rates(surface)) / surface.site_density

    base = rates(theta)
    jac = np.empty((len(theta) - 1, len(theta) - 1))
    for j in range(1, len(theta)):
        h = min(6e-6, max(1e-14, 0.25 * min(theta[j], theta[0])))
        delta = np.zeros(len(theta))
        delta[j], delta[0] = h, -h
        if np.min(theta + delta) >= 0 and np.min(theta - delta) >= 0:
            column = (rates(theta + delta) - rates(theta - delta)) / (2 * h)
        elif np.min(theta + delta) >= 0:
            column = (rates(theta + delta) - base) / h
        elif np.min(theta - delta) >= 0:
            column = (base - rates(theta - delta)) / h
        else:
            raise ValueError("reference root has no admissible surface derivative stencil")
        jac[:, j - 1] = column[1:]
    surface.set_unnormalized_coverages(theta)
    return {
        "coverages": dict(zip(surface.species_names, theta.tolist(), strict=True)),
        "gas": dict(
            zip(
                gas.species_names,
                (surface.get_net_production_rates(gas) * 1000 * area).tolist(),
                strict=True,
            )
        ),
        "forward": surface.forward_rates_of_progress * 1000 * area,
        "reverse": surface.reverse_rates_of_progress * 1000 * area,
        "eigenvalues": np.linalg.eigvals(jac),
        "residual": base,
        "capacity": surface.site_density * 1000 * area,
    }


def pt_payload_and_boundary(temperature, feed):
    payload = source_payload()
    gas = ct.Solution(yaml=json.dumps(payload), name="gas")
    gas.TPX = temperature, 1e5, feed
    return payload, dict(zip(gas.species_names, (gas.X * 1e5).tolist(), strict=True))
