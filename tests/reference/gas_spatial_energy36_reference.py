"""Independent n/U integration, pressure-derivative roots and raw transport formulae.

No native ChemWorld import, thermodynamics, transport or RHS is used here.
"""

from itertools import pairwise
from math import exp, pi, sin, sqrt

import numpy as np
from gas_spatial36_reference import diffusion
from scipy.integrate import solve_ivp
from scipy.optimize import root

R = 8.31446261815324
NA, KB = 6.02214076e23, 1.380649e-23


def conductivity(card, temperature, fractions):
    values, weights = [], []
    for fraction, species in zip(fractions, card["species"].values(), strict=True):
        t = temperature / species["epsilon"]
        omega = (
            1.16145 / t**0.14874
            + 0.52487 * exp(-0.7732 * t)
            + 2.16178 * exp(-2.43787 * t)
            - 0.0006435 * t**0.14874 * sin(18.0323 * t ** (-0.7683) - 7.27371)
        )
        viscosity = (
            5
            / 16
            * sqrt(species["mass"] / NA * KB * temperature / pi)
            / ((species["sigma"] * 1e-10) ** 2 * omega)
        )
        values.append(viscosity * (species["cp"] + 1.25 * R) / species["mass"])
        weights.append(fraction * sqrt(species["mass"]))
    return float(np.dot(values, weights) / sum(weights))


def initial(card):
    ns, m = len(card["species"]), card["cells"]
    volume = card["area"] * card["length"] / m
    volumes = [volume] * m
    pressures = [card["pressure"]] * m
    fractions = [list(row.values()) for row in card["fractions"]]
    if card["flow"]:
        volumes = [card["feed_volume"], *volumes, card["receiver_volume"]]
        pressures = [card["feed_pressure"], *pressures, card["receiver_pressure"]]
        fractions = [
            [card["feed_x"], 1 - card["feed_x"]],
            *fractions,
            [card["receiver_x"], 1 - card["receiver_x"]],
        ]
    temperatures = np.asarray(card["temperatures"])
    amounts = (np.array(pressures) * volumes / (R * temperatures))[:, None] * fractions
    cp = np.array([s["cp"] for s in card["species"].values()])
    offset = np.array([s["h"] - s["cp"] * 298.15 for s in card["species"].values()])
    energy = (
        (amounts @ (cp - R) + card["support"]) * temperatures
        + amounts @ offset
        - card["support"] * 298.15
    )
    return np.column_stack((amounts, energy)).ravel(), np.array(volumes), ns, cp, offset


def solve(card, duration=None):
    y0, volumes, ns, cp, offset = initial(card)
    count = len(volumes)
    cells = np.arange(1, count - 1) if card["flow"] else np.arange(count)
    internal = list(pairwise(cells))
    width = count * (ns + 1)
    # Readable receipts: per-node external loss and link/Fourier; face species/H.
    faces_count = len(internal) + (2 if card["flow"] else 0)
    start = np.r_[y0, np.zeros(count * 2 + faces_count * (ns + 2))]

    def fields(y):
        values = y[:width].reshape(count, ns + 1)
        n, energy = values[:, :ns], values[:, ns]
        temperatures = (energy - n @ offset + card["support"] * 298.15) / (
            n @ (cp - R) + card["support"]
        )
        return n, temperatures

    def balances(y, bulk):
        n, temperatures = fields(y)
        x = n / n.sum(axis=1)[:, None]
        pressure = R * n[cells].sum() / sum(volumes[cells] / temperatures[cells])
        dn, du = np.zeros_like(n), -card["loss"] * (temperatures - 300)
        incoming_heat = np.zeros(count)
        qlink = card["link"] * (temperatures[0] - temperatures[-1])
        incoming_heat[0] -= qlink
        incoming_heat[-1] += qlink
        du += incoming_heat
        for i in cells:
            if ns == 1:
                continue
            if card["flow"]:
                rate = card["k"] * n[i, 1 if card["non_equimolar"] else 0]
                dn[i, :2] = [2 * rate, -rate] if card["non_equimolar"] else [-rate, rate]
            else:
                rate = (
                    exp(-card["Ea"] / (R * temperatures[i]))
                    * volumes[i]
                    * (
                        card["kf"] * (n[i, 0] / volumes[i]) ** card["order"]
                        - card["kr"] * n[i, 1] / volumes[i]
                    )
                )
                dn[i, :2] = [-card["order"] * rate, rate]
        flows = []
        for j, (left, right) in enumerate(internal):
            middle = (temperatures[left] + temperatures[right]) / 2
            fractions = (x[left] + x[right]) / 2
            dx = card["length"] / card["cells"]
            diffusion_rate = card["area"] * diffusion(
                {**card, "temperature": middle}, fractions, (x[right] - x[left]) / dx, pressure
            )
            donor = left if bulk[j] >= 0 else right
            conv = bulk[j] * x[donor]
            hdiff = float(diffusion_rate @ (offset + cp * middle))
            hconv = float(conv @ (offset + cp * temperatures[donor]))
            q = (
                -card["area"]
                * conductivity(card, middle, fractions)
                * (temperatures[right] - temperatures[left])
                / dx
            )
            flows.append((left, right, conv + diffusion_rate, hconv + hdiff, q))
        if card["flow"]:
            for left, right, rate in ((0, 1, card["Fin"]), (count - 2, count - 1, card["Fout"])):
                species = rate * x[left]
                flows.append(
                    (left, right, species, float(species @ (offset + cp * temperatures[left])), 0.0)
                )
        for left, right, species, enthalpy, fourier in flows:
            dn[left] -= species
            dn[right] += species
            du[left] -= enthalpy + fourier
            du[right] += enthalpy + fourier
        dtemp = (du - np.sum((offset + (cp - R) * temperatures[:, None]) * dn, axis=1)) / (
            n @ (cp - R) + card["support"]
        )
        dp = R / volumes * (temperatures * dn.sum(axis=1) + n.sum(axis=1) * dtemp)
        return dn, du, dp, flows, incoming_heat

    def rhs(_, y):
        if internal:
            solution = root(
                lambda flow: np.diff(balances(y, flow)[2][cells]), np.zeros(len(internal)), tol=1e-9
            )
            if max(abs(solution.fun)) > 1e-5:
                raise RuntimeError(f"independent pressure root failed: {solution.message}")
            bulk = solution.x
        else:
            bulk = []
        dn, du, _, flows, link = balances(y, bulk)
        _, temperatures = fields(y)
        return np.r_[
            np.column_stack((dn, du)).ravel(),
            card["loss"] * (temperatures - 300),
            link,
            np.array([[*f[2], f[3], f[4]] for f in flows]).ravel(),
        ]

    result = solve_ivp(
        rhs,
        (0, card["duration"] if duration is None else duration),
        start,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    )
    if not result.success:
        raise RuntimeError(result.message)
    n, t = fields(result.y[:, -1])
    return {
        "amounts": n,
        "temperatures": t,
        "pressures": n.sum(axis=1) * R * t / volumes,
        "energy_changes": result.y[:width, -1].reshape(count, ns + 1)[:, -1]
        - y0.reshape(count, ns + 1)[:, -1],
        "loss": result.y[width : width + count, -1],
        "link": result.y[width + count : width + 2 * count, -1],
        "faces": result.y[width + 2 * count :, -1].reshape(faces_count, ns + 2),
        "nfev": result.nfev,
    }


def cantera_gas(card):
    import cantera as ct

    species = []
    for name, values in card["species"].items():
        item = ct.Species(name, values["atoms"])
        item.thermo = ct.ConstantCp(
            200, 2000, 101325, [298.15, values["h"] * 1000, 30000, values["cp"] * 1000]
        )
        species.append(item)
    reactions = []
    if "A" in card["species"]:
        order = card["order"]
        for reactants, products, rate, power in (
            ({"A": order}, {"B": 1}, card["kf"], order),
            ({"B": 1}, {"A": order}, card["kr"], 1),
        ):
            reaction = ct.Reaction(
                reactants,
                products,
                ct.ArrheniusRate(rate * 1000 ** (power - 1), 0, card["Ea"] * 1000),
            )
            reaction.reversible = False
            reactions.append(reaction)
    return ct.Solution(thermo="ideal-gas", kinetics="gas", species=species, reactions=reactions)
