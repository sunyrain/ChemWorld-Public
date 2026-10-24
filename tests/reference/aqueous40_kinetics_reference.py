"""Independent remaining-stock ODE histories; imports no native physics."""

from math import exp, sqrt

import numpy as np
from scipy.integrate import quad, solve_ivp

from .aqueous40_mineral_reference import WATER_MASS, invariants, properties, saturation

R = 8.31446261815324


def reference(case, times):
    mineral = "NaCl.H2O(s)" if case["hydrate"] else "NaCl(s)"
    ids = (mineral, "KCl(s)") if case["two"] else (mineral,)
    solid0 = np.array([case["solid_na"], case["solid_k"]] if case["two"] else [case["solid_na"]])
    coefficient = 1e-4 * exp(-12000 / R * (1 / case["temperature_K"] - 1 / 298.15))

    def at_extent(extent):
        water = case["water_kg"] / WATER_MASS + (extent[0] if case["hydrate"] else 0)
        na = case["na"] + extent[0]
        potassium = case["k"] + (extent[1] if case["two"] else 0)
        complex_n = case["complex"]
        if case["association"] and not case["omit_association"]:
            total = na + complex_n
            mass = water * WATER_MASS
            na = 2 * total / (1 + sqrt(1 + 40 * total / mass))
            complex_n = total - na
        result = {"H2O": water, "Na+": na, "Cl-": na + potassium, mineral: solid0[0] - extent[0]}
        if case["two"]:
            result.update({"K+": potassium, "KCl(s)": solid0[1] - extent[1]})
        if case["association"]:
            result["NaCl(aq)"] = complex_n
        return result

    def at_stock(stocks):
        result = at_extent(solid0 - stocks)
        result.update({key: float(stocks[j]) for j, key in enumerate(ids)})
        return result

    initial = at_stock(solid0)
    before = properties(case, initial)
    active = [j for j, n in enumerate(solid0) if n > 0]
    current = solid0.copy()
    segments, events = [], []
    start = 0.0
    while active and start < case["duration_s"]:
        indices = tuple(active)
        baseline = current.copy()

        def unpack(x, indices=indices, baseline=baseline):
            e = baseline.copy()
            e[list(indices)] = x
            return e

        def rhs(t, x, indices=indices, unpack=unpack):
            stocks = unpack(x)
            amounts = at_stock(stocks)
            si = saturation(case, amounts)
            # Signed-domain extension permits an event bracket. Only its root
            # (exact zero stock) is accepted, not a projected negative stock.
            return np.array(
                [
                    -coefficient * 0.01 * (abs(stocks[j]) / 0.01) ** (2 / 3) * (1 - exp(si[ids[j]]))
                    for j in indices
                ]
            )

        roots = []
        for i, j in enumerate(indices):

            def exhausted(t, x, i=i, j=j):
                return x[i]

            exhausted.terminal, exhausted.direction = True, -1
            roots.append(exhausted)
        result = solve_ivp(
            rhs,
            (start, case["duration_s"]),
            current[list(indices)],
            method="DOP853",
            rtol=2e-12,
            atol=1e-20,
            dense_output=True,
            events=roots,
        )
        if not result.success:
            raise RuntimeError(result.message)
        end = float(result.t[-1])
        current = unpack(result.y[:, -1])
        for i, event in enumerate(result.t_events):
            if len(event):
                j = indices[i]
                current[j] = 0.0
                active.remove(j)
                events.append(
                    {"event": "mineral_exhausted", "mineral_id": ids[j], "time_s": float(event[-1])}
                )
        segments.append((start, end, indices, result.sol, baseline))
        if end <= start:
            raise RuntimeError("independent remaining-stock event made no progress")
        start = end
    trajectory = []
    for t in times:
        stocks = current.copy()
        for lo, hi, indices, sol, baseline in segments:
            if lo <= t < hi:
                stocks = baseline.copy()
                stocks[list(indices)] = sol(t)
                break
        amounts = at_stock(stocks)
        if any(n < 0 for n in amounts.values()):
            raise RuntimeError("independent accepted amount is negative")
        si, props = saturation(case, amounts), properties(case, amounts)
        areas = {k: 0.01 * (amounts[k] / 0.01) ** (2 / 3) for k in ids}
        trajectory.append(
            {
                "time_s": t,
                "amounts_mol": amounts,
                "areas": areas,
                "rates": {k: coefficient * a * (1 - exp(si[k])) for k, a in areas.items()},
                "heat_J": props["H"] - before["H"],
                "work_J": case["pressure_Pa"] * (props["V"] - before["V"]),
                "invariants": invariants(case, amounts),
            }
        )
    quadrature_time = None
    if case["id"] == "exhaustion":
        area0 = 0.01 * (solid0[0] / 0.01) ** (2 / 3)
        quadrature_time = quad(
            lambda u: (
                3
                * solid0[0]
                / (coefficient * area0)
                / (
                    1
                    - exp(saturation(case, at_extent(np.array([solid0[0] * (1 - u**3)])))[mineral])
                )
            ),
            0,
            1,
            epsabs=1e-8,
            epsrel=1e-12,
        )[0]
        if len(events) != 1 or abs(events[0]["time_s"] - quadrature_time) > 0.5:
            raise RuntimeError(
                "independent remaining-stock depletion disagrees with analytic quadrature"
            )
    return {
        "initial": initial,
        "trajectory": trajectory,
        "events": events,
        "exhaustion_quadrature_s": quadrature_time,
    }
