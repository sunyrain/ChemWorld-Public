"""Independent Cantera and scalar conservation references; no native imports."""

from copy import deepcopy
from itertools import pairwise
from math import exp

import cantera as ct
import numpy as np
import yaml
from scipy.optimize import brentq

R = ct.gas_constant / 1000


def shift(raw, species, delta):
    varied = deepcopy(raw)
    thermo = next(s["thermo"] for s in varied["species"] if s["name"] == species)
    model = thermo["model"]
    if model in {"NASA7", "NASA9", "Shomate"}:
        index, factor = {"NASA7": (5, R), "NASA9": (7, R), "Shomate": (5, 1000)}[model]
        thermo["data"] = [
            [v + delta / factor if j == index else v for j, v in enumerate(row)]
            for row in thermo["data"]
        ]
    else:

        def mol_value(value):
            if isinstance(value, (int, float)):
                return value / 1000
            number, unit = value.split(" ", 1)
            return float(number) * {"J/mol": 1, "kJ/mol": 1000}[unit]

        thermo["h0"] = f"{mol_value(thermo.get('h0', 0)) + delta:.17g} J/mol"
        if model == "piecewise-Gibbs":
            thermo["data"] = {
                t: f"{mol_value(g) + delta:.17g} J/mol" for t, g in thermo["data"].items()
            }
    return varied


def thermochemistry(raw, temperatures):
    phase = ct.Solution(yaml=yaml.safe_dump(raw), name=raw["phases"][0]["name"])
    results = []
    for t in temperatures:
        phase.TP = t, 101325
        results.append(
            [
                phase.standard_enthalpies_RT[0] * R * t,
                phase.standard_gibbs_RT[0] * R * t,
                phase.standard_cp_R[0] * R,
                phase.standard_entropies_R[0] * R,
            ]
        )
    return np.asarray(results)


def trajectory(raw, card, amounts, times, *, surface=False):
    source = yaml.safe_dump(raw)
    thermal = bool(card.get("energy_nodes"))
    energies = {e["vessel_id"]: e for e in card.get("energy_nodes", ())}
    reactors, references, keep = {}, {}, []
    for n in card["nodes"]:
        name = n["vessel_id"]
        gas = ct.Solution(yaml=source, name=raw["phases"][0]["name"])
        t = energies[name]["initial_temperature_K"] if thermal else card["temperature_K"]
        gas.TPX = t, sum(amounts[name].values()) * R * t * 1000 / n["volume_L"], amounts[name]
        reactor = ct.IdealGasReactor(
            gas, clone=True, energy="on" if thermal else "off", volume=n["volume_L"] / 1000
        )
        reactor.chemistry_enabled = n["chemistry_enabled"]
        reactors[name] = reactor
        references[name] = reactor.mass * reactor.phase.int_energy_mass
        if thermal and energies[name]["loss_conductance_W_K"]:
            gas.TP = energies[name]["environment_temperature_K"], 1e5
            env = ct.Reservoir(gas, clone=True)
            keep.extend((env, ct.Wall(reactor, env, A=1, U=energies[name]["loss_conductance_W_K"])))
    for e in card["connections"]:
        if not e["enabled"]:
            continue
        src, dst = reactors[e["source_vessel_id"]], reactors[e["target_vessel_id"]]
        if e["mode"] == "pressure-valve":
            keep.append(ct.Valve(src, dst, K=e["coefficient"]))
        else:
            mass = e["coefficient"] * (
                src.phase.mean_molecular_weight / 1000 if e["mode"] == "molar-flow" else 1
            )
            keep.append(ct.MassFlowController(src, dst, mdot=mass))
    for h in card.get("heat_links", ()):
        keep.append(
            ct.Wall(
                reactors[h["source_vessel_id"]],
                reactors[h["target_vessel_id"]],
                A=1,
                U=h["conductance_W_K"],
            )
        )
    interface = None
    if surface:
        r = reactors["source"]
        interface = ct.Interface(yaml=source, name="surface", adjacent=[r.phase])
        interface.TP = r.T, r.phase.P
        interface.coverages = {"Pt": 0.8, "HA": 0.1, "HB": 0.1}
        keep.append(ct.ReactorSurface(interface, r, A=1000, clone=False))
    simulator = ct.ReactorNet(list(reactors.values()))
    simulator.rtol, simulator.atol = 2e-12, 1e-22
    results = []
    for time in times:
        simulator.advance(time)
        row = {"amounts": {}, "temperatures": {}, "pressures": {}, "delta_U": {}, "bath": {}}
        for name, r in reactors.items():
            row["amounts"][name] = dict(
                zip(r.phase.species_names, r.phase.concentrations * r.volume * 1000, strict=True)
            )
            row["temperatures"][name] = r.T
            row["pressures"][name] = r.phase.P
            du = r.mass * r.phase.int_energy_mass - references[name]
            row["delta_U"][name] = du
            row["bath"][name] = du if not thermal else 0.0
        if interface is not None:
            row["surface"] = dict(
                zip(
                    interface.species_names,
                    interface.coverages * interface.site_density * 1000 * 1000,
                    strict=True,
                )
            )
        row["work"] = dict(simulator.solver_stats)
        results.append(row)
    return results


def closed_steady(thermal, pair=False):
    names = ("r", "s") if pair else ("r",)

    def u(t, b):
        return -2000 * b + 0.02 * (32 * (t - 298.15) - R * t)

    initial = sum(u(t, 0.005) for t in ((500, 700) if pair else (500,)))

    def product(t):
        ratio = exp(2000 / (R * t))
        return 0.02 * ratio / (1 + ratio)

    t = brentq(lambda t: len(names) * u(t, product(t)) - initial, 200, 2000) if thermal else 500
    return {n: {"A": 0.02 - product(t), "B": product(t)} for n in names}, dict.fromkeys(names, t)


def stirred_roots():
    def amounts(t):
        n = 20080 / (R * t * 1000)
        k = 1e8 * exp(-80000 / (R * t))
        a = 0.001 / (k + 0.001 / n)
        return {"A": a, "B": n - a}

    def residual(t):
        n = amounts(t)
        return 0.001 * (32 * (300 - t) + 40000 * n["B"] / sum(n.values())) - 0.03 * (t - 300)

    grid = np.linspace(200, 1200, 1001)
    roots = [
        brentq(residual, a, b, xtol=1e-12)
        for a, b in pairwise(grid)
        if residual(a) * residual(b) < 0
    ]
    return [(amounts(t), t) for t in roots]
