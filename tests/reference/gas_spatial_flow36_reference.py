"""Independent raw-card finite-volume equations and continuous analytic profile.

No native mechanisms, RHS, transport calls or integrators are imported.
"""

from itertools import pairwise
from math import sqrt

import numpy as np
from gas_spatial36_reference import diffusion, diffusivity
from scipy.integrate import solve_ivp
from scipy.linalg import expm

R = 8.31446261815324


def layout(card):
    n = card["cells"] + 2
    return n, (n - 1), 3 * n + 7 * (n - 1)


def initial(card):
    count, _, width = layout(card)
    volume = card["area"] * card["length"] / card["cells"]
    volumes = np.array([card["feed_volume"], *([volume] * card["cells"]), card["receiver_volume"]])
    x = np.array([card["feed_x"], *[r["A"] for r in card["fractions"]], card["receiver_x"]])
    pressure = np.array(
        [card["feed_pressure"], *([card["pressure"]] * card["cells"]), card["receiver_pressure"]]
    )
    y = np.zeros(width)
    y[: 2 * count] = (
        (pressure * volumes / (R * card["temperature"]))[:, None] * np.column_stack((x, 1 - x))
    ).ravel()
    return y, volumes


def derivative(card, values, *, linear=False):
    count, nf, width = layout(card)
    raw = values[: 2 * count].reshape(count, 2)
    volume = card["length"] * card["area"] / card["cells"]
    cell_total = card["pressure"] * volume / (R * card["temperature"])
    if linear:
        x = raw[1:-1] / cell_total
        p = card["pressure"]
        feed = np.array([card["feed_x"], 1 - card["feed_x"]])
    else:
        x = raw[1:-1] / raw[1:-1].sum(axis=1)[:, None]
        p = raw[1:-1].sum() * R * card["temperature"] / (card["length"] * card["area"])
        feed = raw[0] / sum(raw[0])
    chemical = np.zeros_like(raw)
    if card["non_equimolar"]:
        rate = card["k"] * raw[1:-1, 1]
        chemical[1:-1, 0] = 2 * rate
        chemical[1:-1, 1] = -rate
    else:
        rate = card["k"] * raw[1:-1, 0]
        chemical[1:-1, 0] = -rate
        chemical[1:-1, 1] = rate
    production = chemical[1:-1].sum(axis=1)
    all_rate = card["Fin"] - card["Fout"] + sum(production)
    h = np.array(
        [s["h"] + s["cp"] * (card["temperature"] - 298.15) for s in card["species"].values()]
    )
    u = h - R * card["temperature"]
    faces = []
    for j in range(nf):
        if j == 0:
            bulk = card["Fin"]
            diff = np.zeros(2)
            conv = bulk * feed
        elif j == nf - 1:
            bulk = card["Fout"]
            diff = np.zeros(2)
            conv = bulk * x[-1]
        else:
            bulk = card["Fin"] + sum(production[:j]) - j / card["cells"] * all_rate
            dx = card["length"] / card["cells"]
            middle = (x[j - 1] + x[j]) / 2
            if linear:
                diff = (
                    -card["area"]
                    * card["pressure"]
                    / (R * card["temperature"])
                    * diffusivity(card, "A", "B", p)
                    * (x[j] - x[j - 1])
                    / dx
                )
            else:
                diff = card["area"] * diffusion(card, middle, (x[j] - x[j - 1]) / dx, p)
            conv = bulk * (x[j - 1] if bulk >= 0 else x[j])
        flux = conv + diff
        faces.append(
            {
                "material": flux,
                "H": float(flux @ h),
                "bulk": bulk,
                "diffusion": diff,
                "diffusion_H": float(diff @ h),
            }
        )
    dn = chemical.copy()
    net_h = np.zeros(count)
    for j, f in enumerate(faces):
        dn[j] -= f["material"]
        dn[j + 1] += f["material"]
        net_h[j] -= f["H"]
        net_h[j + 1] += f["H"]
    out = np.zeros(width)
    out[: 2 * count] = dn.ravel()
    out[2 * count : 3 * count] = dn @ u - net_h
    for j, f in enumerate(faces):
        out[3 * count + 7 * j : 3 * count + 7 * (j + 1)] = [
            *f["material"],
            f["H"],
            f["bulk"],
            *f["diffusion"],
            f["diffusion_H"],
        ]
    return out, faces


def observe(card, y, time):
    count, nf, _ = layout(card)
    amounts = y[: 2 * count].reshape(count, 2)
    _, volumes = initial(card)
    faces = y[3 * count :].reshape(nf, 7)
    return {
        "time": float(time),
        "n": amounts.ravel().tolist(),
        "P": (amounts.sum(axis=1) * R * card["temperature"] / volumes).tolist(),
        "Q": y[2 * count : 3 * count].tolist(),
        "material": faces[:, :2].ravel().tolist(),
        "H": faces[:, 2].tolist(),
        "bulk": faces[:, 3].tolist(),
        "diffusion": faces[:, 4:6].ravel().tolist(),
        "diffusion_H": faces[:, 6].tolist(),
        "instant": np.array([f["material"] for f in derivative(card, y)[1]]).ravel().tolist(),
    }


def solve(card, *, matrix=False, event=False):
    y0, _ = initial(card)
    if matrix:
        width = len(y0)
        offset = derivative(card, np.zeros(width), linear=True)[0]
        jac = np.column_stack(
            [derivative(card, np.eye(width)[j], linear=True)[0] - offset for j in range(width)]
        )
        augmented = np.zeros((width + 1, width + 1))
        augmented[:width, :width] = jac
        augmented[:width, -1] = offset
        samples = [
            observe(card, (expm(augmented * t) @ np.r_[y0, 1.0])[:-1], t) for t in card["times"]
        ]
        return {"samples": samples, "method": "independent affine matrix exponential", "nfev": 0}

    def event_fn(t, y):
        n, _, _ = layout(card)
        a = y[: 2 * n].reshape(n, 2)
        if card["number"] == 13:
            return sum(a[0]) - card["minimum_feed"]
        if card["number"] == 14:
            return (
                card["maximum_receiver"]
                - sum(a[-1]) * R * card["temperature"] / card["receiver_volume"]
            )
        p = a[1:-1].sum() * R * card["temperature"] / (card["area"] * card["length"])
        if card["number"] == 15:
            return card["maximum_grid"] - p
        cp = (30.0 + 44.0) / 2
        speed = sqrt(cp / (cp - R) * R * card["temperature"] / 0.002016)
        return card["maximum_mach"] - card["Fout"] / (
            card["area"] * p / (R * card["temperature"]) * speed
        )

    event_fn.terminal = True
    event_fn.direction = -1
    result = solve_ivp(
        lambda t, y: derivative(card, y)[0],
        (0, card["times"][-1]),
        y0,
        method="DOP853",
        rtol=1e-11,
        atol=1e-13,
        dense_output=True,
        events=event_fn if event else None,
    )
    if not result.success:
        raise RuntimeError(result.message)
    times = (result.t[-1],) if event else card["times"]
    return {
        "samples": [observe(card, result.sol(t), t) for t in times],
        "method": "independent full n equations DOP853",
        "nfev": result.nfev,
    }


def continuous_profile(card):
    d = diffusivity(card, "A", "B", card["pressure"])
    velocity = card["Fin"] / (card["area"] * card["pressure"] / (R * card["temperature"]))
    length = card["length"]
    k = card["k"]
    roots = np.array(
        [
            (velocity + sqrt(velocity**2 + 4 * d * k)) / (2 * d),
            (velocity - sqrt(velocity**2 + 4 * d * k)) / (2 * d),
        ]
    )
    matrix = np.array([velocity - d * roots, roots * np.exp(roots * length)])
    coefficients = np.linalg.solve(matrix, np.array([velocity * card["feed_x"], 0.0]))
    edges = np.linspace(0, length, card["cells"] + 1)
    return np.array(
        [
            sum(coefficients * (np.exp(roots * b) - np.exp(roots * a)) / roots) / (b - a)
            for a, b in pairwise(edges)
        ]
    )
