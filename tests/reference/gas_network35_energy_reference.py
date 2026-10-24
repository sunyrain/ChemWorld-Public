"""Independent n/U reference, constant-Cp source cards only, no ChemWorld imports."""

from math import exp

import numpy as np
from scipy.integrate import solve_ivp

R = 8.31446261815324
MW = 0.002016


def temperature(card, node, amounts, energy):
    capacity = node["support_J_K"] + sum(
        amounts[i] * (card["thermo"][s]["cp"] - R) for i, s in enumerate(("A", "B"))
    )
    offset = -node["support_J_K"] * node["reference_K"] + sum(
        amounts[i] * (card["thermo"][s]["h"] - card["thermo"][s]["cp"] * 298.15)
        for i, s in enumerate(("A", "B"))
    )
    return (energy - offset) / capacity


def energy(card, node, amounts, t):
    return node["support_J_K"] * (t - node["reference_K"]) + sum(
        amounts[i] * (card["thermo"][s]["h"] + card["thermo"][s]["cp"] * (t - 298.15) - R * t)
        for i, s in enumerate(("A", "B"))
    )


def solve(card, duration, times=None, enable_events=False):
    nodes, edges, links = card["nodes"], card["edges"], card["heat_links"]
    count, edge_count = len(nodes), len(edges)
    index = {n["id"]: i for i, n in enumerate(nodes)}
    # Per node: nA,nB,U,Qheater,Qloss,Qreaction. Edge: nA,nB,H. Link: signed Q.
    initial = np.zeros(6 * count + 3 * edge_count + len(links))
    for i, n in enumerate(nodes):
        amounts = [card["amounts"][n["id"]][s] for s in ("A", "B")]
        initial[6 * i : 6 * i + 3] = [*amounts, energy(card, n, amounts, n["temperature_K"])]

    def fields(y):
        amounts = [y[6 * i : 6 * i + 2] for i in range(count)]
        temperatures = [temperature(card, n, amounts[i], y[6 * i + 2]) for i, n in enumerate(nodes)]
        pressure = [
            sum(amounts[i]) * R * temperatures[i] * 1000 / n["volume_L"]
            for i, n in enumerate(nodes)
        ]
        return amounts, temperatures, pressure

    def rhs(_, y):
        amounts, temperatures, pressure = fields(y)
        output = np.zeros_like(y)
        for i, n in enumerate(nodes):
            t, a = temperatures[i], amounts[i]
            k = (
                card["kinetics"]["A"] * exp(-card["kinetics"]["Ea"] / (R * t))
                if n["chemistry"]
                else 0.0
            )
            rate = k * a[0]
            output[6 * i], output[6 * i + 1] = -rate, rate
            h = [
                card["thermo"][s]["h"] + card["thermo"][s]["cp"] * (t - 298.15) for s in ("A", "B")
            ]
            output[6 * i + 5] = rate * (h[1] - h[0])
            device = card["device"] if n["id"] == card["anchor"] else None
            q = 0.0
            if device is not None and device["enabled"]:
                q = min(
                    device["maximum_power_W"],
                    max(
                        -device["maximum_cooling_power_W"],
                        device["conductance_W_K"]
                        * device["calibration_factor"]
                        * (device["setpoint_K"] - t),
                    ),
                )
            loss = n["loss_W_K"] * (t - n["environment_K"])
            output[6 * i + 2 : 6 * i + 5] = [q - loss, q, loss]
        for j, e in enumerate(edges):
            source, target = index[e["source"]], index[e["target"]]
            flow = e["coefficient"]
            if e["mode"] == "pressure-valve":
                flow *= max(pressure[source] - pressure[target], 0)
            if e["mode"] != "molar-flow":
                flow /= MW
            transfer = flow * amounts[source] / sum(amounts[source])
            source_h = np.array(
                [
                    card["thermo"][s]["h"]
                    + card["thermo"][s]["cp"] * (temperatures[source] - 298.15)
                    for s in ("A", "B")
                ]
            )
            q = float(source_h @ transfer)
            output[6 * source : 6 * source + 2] -= transfer
            output[6 * target : 6 * target + 2] += transfer
            output[6 * source + 2] -= q
            output[6 * target + 2] += q
            output[6 * count + 3 * j : 6 * count + 3 * j + 3] = [*transfer, q]
        for j, e in enumerate(links):
            source, target = index[e["source"]], index[e["target"]]
            q = e["conductance"] * (temperatures[source] - temperatures[target])
            output[6 * source + 2] -= q
            output[6 * target + 2] += q
            output[6 * count + 3 * edge_count + j] = q
        return output

    events = []
    if enable_events:
        for i, n in enumerate(nodes):
            for kind in ("amount", "high-T", "low-T", "pressure"):

                def event(_, y, i=i, n=n, kind=kind):
                    amounts, t, p = fields(y)
                    return {
                        "amount": sum(amounts[i]) - n["minimum_mol"],
                        "high-T": n["maximum_K"] - t[i],
                        "low-T": t[i] - n["minimum_K"],
                        "pressure": n["maximum_Pa"] - p[i],
                    }[kind]

                event.terminal = True
                event.direction = -1
                events.append(event)
    result = solve_ivp(
        rhs,
        (0, duration),
        initial,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        events=events or None,
    )
    if not result.success:
        raise RuntimeError(result.message)
    samples = []
    for t in [result.t[-1]] if times is None else times:
        y = result.sol(t)
        amounts, temps, pressures = fields(y)
        samples.append(
            {
                "time_s": float(t),
                "nodes": {
                    n["id"]: {
                        "amounts_mol": dict(zip(("A", "B"), amounts[i], strict=True)),
                        "temperature_K": temps[i],
                        "pressure_Pa": pressures[i],
                        "internal_energy_change_J": y[6 * i + 2] - initial[6 * i + 2],
                        "bath_heat_J": y[6 * i + 3],
                        "heat_loss_J": y[6 * i + 4],
                        "chemical_energy_change_J": y[6 * i + 5],
                    }
                    for i, n in enumerate(nodes)
                },
                "edges": {
                    e["id"]: {
                        "amounts_mol": dict(
                            zip(
                                ("A", "B"),
                                y[6 * count + 3 * j : 6 * count + 3 * j + 2],
                                strict=True,
                            )
                        ),
                        "enthalpy_J": y[6 * count + 3 * j + 2],
                    }
                    for j, e in enumerate(edges)
                },
                "links": {e["id"]: y[6 * count + 3 * edge_count + j] for j, e in enumerate(links)},
            }
        )
    return {"samples": samples, "final_time_s": float(result.t[-1]), "nfev": result.nfev}
