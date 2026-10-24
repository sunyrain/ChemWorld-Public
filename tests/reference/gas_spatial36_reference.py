"""Independent raw-card x/P reference; no ChemWorld imports or native RHS calls."""

from math import exp, pi, sqrt

import numpy as np
from scipy.integrate import solve_ivp

R = 8.31446261815324
NA = 6.02214076e23
KB = 1.380649e-23


def diffusivity(card, left, right, pressure):
    a, b, t = card["species"][left], card["species"][right], card["temperature"]
    reduced = t / sqrt(a["epsilon"] * b["epsilon"])
    omega = (
        1.06036 / reduced**0.15610
        + 0.19300 * exp(-0.47635 * reduced)
        + 1.03587 * exp(-1.52996 * reduced)
        + 1.76474 * exp(-3.89411 * reduced)
    )
    mu = a["mass"] * b["mass"] / (a["mass"] + b["mass"]) / NA
    sigma = (a["sigma"] + b["sigma"]) * 0.5e-10
    return 3 / 16 * KB * t / pressure * sqrt(2 * KB * t / (pi * mu)) / (sigma * sigma * omega)


def diffusion(card, x, gradient, pressure):
    names = list(card["species"])
    count, density = len(names), pressure / (R * card["temperature"])
    if count == 1:
        return np.zeros(1)
    d = [[diffusivity(card, a, b, pressure) for b in names] for a in names]
    if card["model"] == "mixture-averaged":
        masses = [card["species"][s]["mass"] for s in names]
        mean = sum(x[i] * masses[i] for i in range(count))
        y = [x[i] * masses[i] / mean for i in range(count)]
        raw = []
        for i in range(count):
            denominator = sum(x[j] / d[i][j] for j in range(count) if i != j)
            coefficient = (
                sum(y[j] for j in range(count) if i != j) / denominator if denominator else 0.0
            )
            raw.append(-density * coefficient * gradient[i])
        return np.array([raw[i] - x[i] * sum(raw) for i in range(count)])
    if count == 2:
        value = -density * d[0][1] * gradient[0]
        return np.array([value, -value])
    a11 = x[1] / d[0][1] + x[2] / d[0][2] + x[0] / d[0][2]
    a12 = -x[0] / d[0][1] + x[0] / d[0][2]
    a21 = -x[1] / d[1][0] + x[1] / d[1][2]
    a22 = x[0] / d[1][0] + x[2] / d[1][2] + x[1] / d[1][2]
    b1, b2 = -density * gradient[0], -density * gradient[1]
    det = a11 * a22 - a12 * a21
    n0, n1 = (b1 * a22 - a12 * b2) / det, (a11 * b2 - b1 * a21) / det
    return np.array([n0, n1, -n0 - n1])


def rates(card, fractions, pressure):
    names = list(card["species"])
    count, ns = card["cells"], len(names)
    volume = card["length"] * card["area"] / count
    dx, t, density = (
        card["length"] / count,
        card["temperature"],
        pressure / (R * card["temperature"]),
    )
    reaction = np.zeros((count, ns))
    if "A" in names:
        factor = exp(-card["Ea"] / (R * t))
        for k in range(count):
            rate = (
                (
                    card["kf"] * (density * fractions[k, 0]) ** card["order"]
                    - card["kr"] * density * fractions[k, 1]
                )
                * factor
                * volume
            )
            reaction[k, 0], reaction[k, 1] = -card["order"] * rate, rate
    w = reaction.sum(axis=1)
    p_rate = R * t * sum(w) / (count * volume)
    h = np.array([card["species"][s]["h"] + card["species"][s]["cp"] * (t - 298.15) for s in names])
    u = h - R * t
    result, faces, incoming_h = reaction.copy(), [], np.zeros(count)
    for j in range(1, count):
        bulk = sum(w[:j]) - j / count * sum(w)
        middle = (fractions[j - 1] + fractions[j]) / 2
        diff = card["area"] * diffusion(
            card, middle, (fractions[j] - fractions[j - 1]) / dx, pressure
        )
        convection = bulk * (fractions[j - 1] if bulk >= 0 else fractions[j])
        total = convection + diff
        q, qdiff = float(total @ h), float(diff @ h)
        result[j - 1] -= total
        result[j] += total
        incoming_h[j - 1] -= q
        incoming_h[j] += q
        mean_mass = sum(middle[i] * card["species"][s]["mass"] for i, s in enumerate(names))
        mass_velocity = sum(total[i] * card["species"][s]["mass"] for i, s in enumerate(names)) / (
            card["area"] * density * mean_mass
        )
        faces.append(
            {
                "total": total,
                "diffusion": diff,
                "bulk": bulk,
                "H": q,
                "diffusion_H": qdiff,
                "molar_velocity": bulk / (card["area"] * density),
                "mass_velocity": mass_velocity,
            }
        )
    bath = result @ u - incoming_h
    return result, p_rate, bath, faces


def solve(card, times, event=False):
    count, ns = card["cells"], len(card["species"])
    names = list(card["species"])
    nv = count * (ns - 1)
    face_size = 2 * ns + 3
    offset = nv + 1 + count
    initial = np.zeros(offset + (count - 1) * face_size)
    initial[:nv] = np.array([[r[s] for s in names[:-1]] for r in card["fractions"]]).ravel()
    initial[nv] = card["pressure"]

    def decode(y):
        x = np.zeros((count, ns))
        x[:, :-1] = y[:nv].reshape((count, ns - 1))
        x[:, -1] = 1 - x[:, :-1].sum(axis=1)
        return x, y[nv]

    def rhs(_, y):
        x, p = decode(y)
        dn, dp, bath, faces = rates(card, x, p)
        ncell = p / (R * card["temperature"]) * card["area"] * card["length"] / count
        dx = dn / ncell - x * dp / p
        out = np.zeros_like(y)
        out[:nv], out[nv], out[nv + 1 : offset] = dx[:, :-1].ravel(), dp, bath
        for i, f in enumerate(faces):
            out[offset + i * face_size : offset + (i + 1) * face_size] = [
                *f["total"],
                *f["diffusion"],
                f["bulk"],
                f["H"],
                f["diffusion_H"],
            ]
        return out

    def pressure_stop(_, y):
        return card["maximum_pressure"] - y[nv]

    pressure_stop.terminal = True
    pressure_stop.direction = -1
    result = solve_ivp(
        rhs,
        (0, times[-1]),
        initial,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        events=pressure_stop if event else None,
    )
    if not result.success:
        raise RuntimeError(result.message)
    output = []
    for time in [result.t[-1]] if event else times:
        y = result.sol(time)
        x, pressure = decode(y)
        cell_total = pressure / (R * card["temperature"]) * card["area"] * card["length"] / count
        _, _, _, instantaneous = rates(card, x, pressure)
        output.append(
            {
                "time_s": float(time),
                "pressure_Pa": pressure,
                "amounts": [
                    {s: cell_total * x[k, i] for i, s in enumerate(names)} for k in range(count)
                ],
                "bath": list(y[nv + 1 : offset]),
                "faces": [
                    {
                        "amounts": dict(
                            zip(
                                names,
                                y[offset + j * face_size : offset + j * face_size + ns],
                                strict=True,
                            )
                        ),
                        "diffusion": dict(
                            zip(
                                names,
                                y[offset + j * face_size + ns : offset + j * face_size + 2 * ns],
                                strict=True,
                            )
                        ),
                        "bulk": y[offset + j * face_size + 2 * ns],
                        "H": y[offset + j * face_size + 2 * ns + 1],
                        "diffusion_H": y[offset + j * face_size + 2 * ns + 2],
                        "rates": instantaneous[j],
                    }
                    for j in range(count - 1)
                ],
            }
        )
    return {"samples": output, "nfev": result.nfev}
