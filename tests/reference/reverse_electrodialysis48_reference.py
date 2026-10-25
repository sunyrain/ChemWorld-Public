"""Independent continuous BVP/resistive circuit and RK45 finite RED reference.

The pre-existing independent ED reference reads declarations and implements its
own Donnan/continuous NP/Schloegl equations. No native pore, controller, phase
integrator, chemical-potential or property evaluator is called here.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from reference.electrodialysis48_reference import R, enthalpy, volume
from reference.electrodialysis48_reference import point as ed_point


def point(model, amounts, resistance, *, counters=None, progress=None):
    counters = counters if counters is not None else {"bvp_calls": 0, "bvp_nodes": 0}

    def evaluate(target, unit):
        row = ed_point(model, amounts, target, unit)
        counters["bvp_calls"] += 1
        counters["bvp_nodes"] += row["bvp_nodes"]
        if progress:
            progress(counters)
        return row

    opened = evaluate(0, "A")
    if resistance is None or abs(opened["voltage_V"]) < 1e-14:
        row = opened
    else:
        short = evaluate(0, "V")
        if resistance == 0:
            row = short
        else:

            def residual(current):
                return evaluate(current, "A")["voltage_V"] + resistance * current

            low, high = sorted((short["current_A"], 0))
            root = brentq(residual, low, high, xtol=1e-14, rtol=1e-12)
            row = evaluate(root, "A")
    load = 0 if resistance is None else row["current_A"] ** 2 * resistance
    gibbs_rate = mixing_gibbs_rate(amounts, row["diluate_rates_mol_s"], model.temperature_K)
    return {
        **row,
        "load_power_W": load,
        "gibbs_energy_rate_W": gibbs_rate,
        "dissipation_W": -gibbs_rate - load,
        "reference_costs": dict(counters),
    }


def mixing_gibbs(amounts, temperature):
    # Constant standard-state terms cancel in this closed two-reservoir change.
    # Molality's water molar mass factor also cancels because total ions persist.
    return (
        R
        * temperature
        * sum(
            value * (np.log(value / n["H2O"]) - 1)
            for n in amounts
            for key, value in n.items()
            if key != "H2O"
        )
    )


def mixing_gibbs_rate(amounts, diluate_flow, temperature):
    total = 0.0
    for n, sign in zip(amounts, (1, -1), strict=True):
        total += (
            R
            * temperature
            * sum(sign * diluate_flow[k] * np.log(n[k] / n["H2O"]) for k in n if k != "H2O")
        )
        total -= (
            R
            * temperature
            * sign
            * diluate_flow["H2O"]
            * sum(v for k, v in n.items() if k != "H2O")
            / n["H2O"]
        )
    return float(total)


def finite(model, initial, resistance, duration, *, ramp=None, progress=None):
    ids = tuple(initial[0])
    width = len(ids)
    costs = {"bvp_calls": 0, "bvp_nodes": 0}
    rhs_calls = 0
    last_progress = time.monotonic()

    def show(counters):
        nonlocal last_progress
        if progress and time.monotonic() - last_progress >= 10:
            progress({**counters, "rhs_calls": rhs_calls})
            last_progress = time.monotonic()

    def unpack(y):
        return tuple(dict(zip(ids, row, strict=True)) for row in y[: 2 * width].reshape(2, width))

    def resistance_at(t):
        return (
            resistance if ramp is None else ramp[0] + (resistance - ramp[0]) * min(t / ramp[1], 1)
        )

    def rhs(t, y):
        nonlocal rhs_calls
        ns = unpack(y)
        row = point(model, ns, resistance_at(t), counters=costs, progress=show)
        rates = np.asarray([row["diluate_rates_mol_s"][k] for k in ids])
        # Direct derivative of the independent ideal mixing G, including water.
        gdot = 0.0
        for n, flow in zip(ns, (rates, -rates), strict=True):
            gdot += (
                R
                * model.temperature_K
                * sum(flow[j] * np.log(n[k] / n["H2O"]) for j, k in enumerate(ids) if k != "H2O")
            )
            gdot -= (
                R
                * model.temperature_K
                * flow[ids.index("H2O")]
                * sum(v for k, v in n.items() if k != "H2O")
                / n["H2O"]
            )
        rhs_calls += 1
        return np.r_[rates, -rates, row["load_power_W"], -gdot - row["load_power_W"]]

    y = np.r_[np.asarray([[n[k] for k in ids] for n in initial]).ravel(), 0.0, 0.0]
    stops = []
    for i in range(2):
        for name, function in (
            (f"water_minimum:{i}", lambda t, y, i=i: unpack(y)[i]["H2O"] - model.minimum_water_mol),
            (
                f"capacity:{i}",
                lambda t, y, i=i: model.capacities_L[i] - 1000 * volume(unpack(y)[i]),
            ),
            (
                f"ion_minimum:{i}",
                lambda t, y, i=i: (
                    min(
                        n / (unpack(y)[i]["H2O"] * 18e-6)
                        for k, n in unpack(y)[i].items()
                        if k != "H2O"
                    )
                    - model.minimum_ion_concentration_mol_m3
                ),
            ),
        ):
            function.terminal = True
            function.direction = -1
            stops.append((name, function))
    boundaries = sorted({duration, *([ramp[1]] if ramp and ramp[1] < duration else [])})
    start = 0.0
    stopping_event = None
    for end in boundaries:
        solved = solve_ivp(
            rhs, (start, end), y, method="RK45", rtol=2e-9, atol=1e-12, events=[f for _, f in stops]
        )
        if not solved.success:
            raise RuntimeError("independent finite passive RED failed: " + solved.message)
        y, start = solved.y[:, -1], float(solved.t[-1])
        found = [
            name for (name, _), events in zip(stops, solved.t_events, strict=True) if len(events)
        ]
        if found:
            stopping_event = found[0]
            break
    final = unpack(y)
    return {
        "amounts_mol": final,
        "enthalpies_J": [enthalpy(n, model.temperature_K) for n in final],
        "volumes_m3": [volume(n) for n in final],
        "load_energy_delivered_J": float(y[-2]),
        "dissipation_integral_J": float(y[-1]),
        "actual_duration_s": start,
        "stopping_event": stopping_event,
        "free_energy_decrease_J": mixing_gibbs(initial, model.temperature_K)
        - mixing_gibbs(final, model.temperature_K),
        "costs": {**costs, "rhs_calls": rhs_calls},
    }
