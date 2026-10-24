"""Independent Cantera ReactorSurface and source-rate finite n/T equations."""

import json

import cantera as ct
import numpy as np
from scipy.integrate import simpson, solve_ivp

R = ct.gas_constant / 1000


def phases(card):
    result = {}
    for node in card["nodes"]:
        surface = ct.Interface(yaml=json.dumps(card["document"]), name="surface")
        gas = surface.adjacent["gas"]
        gas.TPX = (
            node["temperature"],
            sum(node["gas"].values()) * R * node["temperature"] / node["volume"],
            node["gas"],
        )
        surface.TP = gas.TP
        if node["surface"]:
            n = node["surface"]
            surface.coverages = [n[s.name] * s.size / (1e-5 * 0.1) for s in surface.species()]
        result[node["name"]] = (gas, surface)
    return result


def caloric(gas, surface, temperature, active):
    species = [*gas.species(), *(surface.species() if active else [])]
    u = np.array([s.thermo.h(temperature) / 1000 for s in species])
    cv = np.array([s.thermo.cp(temperature) / 1000 for s in species])
    u[: gas.n_species] -= R * temperature
    cv[: gas.n_species] -= R
    return u, cv


def summarize(card, table, amounts, temperatures, transfers, bath, initial_energy):
    row = {
        key: {}
        for key in (
            "gas",
            "surface",
            "coverage",
            "temperature",
            "pressure",
            "energy",
            "bath",
            "transport",
            "enthalpy",
            "elements",
            "sites",
        )
    }
    for node in card["nodes"]:
        name = node["name"]
        gas, surface = table[name]
        n, temperature = amounts[name], temperatures[name]
        species = [*gas.species(), *(surface.species() if node["surface"] else [])]
        u, _ = caloric(gas, surface, temperature, bool(node["surface"]))
        total_energy = float(n @ u + node["support"] * (temperature - 298.15))
        row["energy"][name] = total_energy - initial_energy[name]
        row["temperature"][name] = temperature
        row["pressure"][name] = float(sum(n[:2]) * R * temperature / node["volume"])
        row["bath"][name] = bath[name]
        for index, species_ in enumerate(species):
            group = "gas" if index < 2 else "surface"
            row[group][name + ":" + species_.name] = float(n[index])
            for element, count in species_.composition.items():
                row["elements"][element] = row["elements"].get(element, 0.0) + float(
                    n[index] * count
                )
            if index >= 2:
                occupied = float(n[index] * species_.size)
                row["coverage"][name + ":" + species_.name] = occupied / 1e-6
                row["sites"][name] = row["sites"].get(name, 0.0) + occupied
    row["transport"] = {"feed:A": float(transfers[0]), "feed:B": float(transfers[1])}
    row["enthalpy"] = {"feed": float(transfers[2])}
    return row


def reactor_history(card, times):
    table = phases(card)
    reactors, surfaces = {}, {}
    for node in card["nodes"]:
        name = node["name"]
        gas, surface = table[name]
        reactors[name] = ct.IdealGasReactor(gas, energy="off", volume=node["volume"], clone=False)
        if node["surface"]:
            surfaces[name] = ct.ReactorSurface(surface, reactors[name], A=0.1, clone=False)
    controller = ct.MassFlowController(
        reactors["source"], reactors["receiver"], mdot=card["mass_flow"]
    )
    solver = ct.ReactorNet(list(reactors.values()))
    solver.rtol, solver.atol, solver.max_steps = 1e-11, 1e-20, 100000
    grid = np.linspace(0, max(times), 1001)
    flux, snapshots, initial_energy = [], [], {}
    for time in grid:
        solver.advance(float(time))
        gas = reactors["source"].phase
        molar = controller.mass_flow_rate * 1000 / gas.mean_molecular_weight
        flux.append([*(molar * gas.X), controller.mass_flow_rate * gas.enthalpy_mass])
        amounts, temperature, energies = {}, {}, {}
        for node in card["nodes"]:
            name = node["name"]
            reactor = reactors[name]
            gas, surface = table[name]
            n = list(reactor.phase.concentrations * node["volume"] * 1000)
            if name in surfaces:
                n.extend(
                    surfaces[name].coverages
                    * surface.site_density
                    * 1000
                    * 0.1
                    / np.array([s.size for s in surface.species()])
                )
            amounts[name], temperature[name] = np.array(n), float(reactor.T)
            u, _ = caloric(gas, surface, reactor.T, name in surfaces)
            energies[name] = float(np.array(n) @ u + node["support"] * (reactor.T - 298.15))
        if time == 0:
            initial_energy = energies
        snapshots.append((amounts, temperature, energies))
    rows = []
    for time in times:
        i = round(time / max(times) * 1000)
        receipt = (
            np.zeros(3) if i == 0 else simpson(np.asarray(flux[: i + 1]), x=grid[: i + 1], axis=0)
        )
        amounts, temperature, energy = snapshots[i]
        bath = {
            name: energy[name] - initial_energy[name] + (1 if name == "source" else -1) * receipt[2]
            for name in reactors
        }
        rows.append(summarize(card, table, amounts, temperature, receipt, bath, initial_energy))
    return {
        "rows": rows,
        "method": "Cantera-3.2-ReactorSurface",
        "reference_work": solver.solver_stats,
    }


def equation_history(card, times, *, event=False):
    table = phases(card)
    slices, initial, initial_energy = {}, [], {}
    for node in card["nodes"]:
        name = node["name"]
        gas, surface = table[name]
        values = [node["gas"][s] for s in gas.species_names]
        if node["surface"]:
            values.extend(node["surface"][s] for s in surface.species_names)
        begin = len(initial)
        initial.extend([*values, node["temperature"]])
        slices[name] = (slice(begin, len(initial) - 1), len(initial) - 1)
        u, _ = caloric(gas, surface, node["temperature"], bool(node["surface"]))
        initial_energy[name] = float(
            np.array(values) @ u + node["support"] * (node["temperature"] - 298.15)
        )
    receipt_start = len(initial)
    initial.extend([0.0] * 5)  # transported A/B/H, source/receiver bath

    def rhs(time, state):
        derivative = np.zeros(len(state))
        rates, uvalues, capacities = {}, {}, {}
        for node in card["nodes"]:
            name = node["name"]
            material, temperature_index = slices[name]
            n, t = state[material], state[temperature_index]
            gas, surface = table[name]
            gas.TPX = t, sum(n[:2]) * R * t / node["volume"], n[:2]
            dn = np.zeros(len(n))
            if node["surface"]:
                surface.TP = gas.TP
                surface.set_unnormalized_coverages(
                    n[2:] * np.array([s.size for s in surface.species()]) / 1e-6
                )
                dn = (
                    np.r_[
                        surface.get_net_production_rates(gas),
                        surface.get_net_production_rates(surface),
                    ]
                    * 1000
                    * 0.1
                )
            rates[name] = dn
            uvalues[name], cv = caloric(gas, surface, t, bool(node["surface"]))
            capacities[name] = node["support"] + float(n @ cv)
        gas = table["source"][0]
        molar = card["mass_flow"] * 1000 / gas.mean_molecular_weight
        transfer = molar * gas.X
        enthalpy = card["mass_flow"] * gas.enthalpy_mass
        rates["source"][:2] -= transfer
        rates["receiver"][:2] += transfer
        conduction = card["conductance"] * (
            state[slices["source"][1]] - state[slices["receiver"][1]]
        )
        baths = []
        for node in card["nodes"]:
            name = node["name"]
            material, temperature_index = slices[name]
            chemical = float(uvalues[name] @ rates[name])
            incoming = (1 if name == "receiver" else -1) * (enthalpy + conduction)
            derivative[material] = rates[name]
            derivative[temperature_index] = (
                (incoming - chemical) / capacities[name] if card["thermal"] else 0
            )
            baths.append(0.0 if card["thermal"] else chemical - incoming)
        derivative[receipt_start:] = [*transfer, enthalpy, *baths]
        return derivative

    def stop(time, state):
        return sum(state[slices["source"][0]][:2]) - 0.0295

    stop.terminal, stop.direction = True, -1
    result = solve_ivp(
        rhs,
        (0, max(times)),
        initial,
        method="Radau",
        rtol=1e-10,
        atol=1e-15,
        t_eval=times,
        events=stop if event else None,
    )
    if not result.success:
        raise RuntimeError(result.message)
    points = list(zip(result.t, result.y.T, strict=True))
    if event and len(result.t_events[0]):
        points.append((float(result.t_events[0][0]), result.y_events[0][0]))
    rows = []
    for time, state in points:
        amounts = {name: state[sl] for name, (sl, _) in slices.items()}
        temperatures = {name: float(state[i]) for name, (_, i) in slices.items()}
        bath = {
            node["name"]: float(state[receipt_start + 3 + i])
            for i, node in enumerate(card["nodes"])
        }
        row = summarize(
            card,
            table,
            amounts,
            temperatures,
            state[receipt_start : receipt_start + 3],
            bath,
            initial_energy,
        )
        row["time_s"] = float(time)
        rows.append(row)
    return {
        "rows": rows,
        "method": "independent-Cantera-source-rate-nT-Radau",
        "reference_work": {"nfev": result.nfev, "njev": result.njev, "nlu": result.nlu},
    }
