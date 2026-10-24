"""Independent closed forms, Cantera, and n/U DOP853 references (no native imports)."""

from copy import deepcopy

import numpy as np
from scipy.integrate import simpson, solve_ivp

R = 8.31446261815324


def vary(raw, binding, factor):
    result = deepcopy(raw)
    kind, owner, _ = binding
    if kind == "reaction":
        result["kinetics"]["A"] *= factor
    else:
        family, identity, field = {
            "flow": ("edges", "id", "coefficient"),
            "heat_link": ("heat_links", "id", "conductance"),
            "loss": ("nodes", "id", "loss_W_K"),
        }[kind]
        next(item for item in result[family] if item[identity] == owner)[field] *= factor
    return result


def outputs(raw, amounts, temps, bath, loss, delta_u, transfers, links):
    result = {}
    for node in raw["nodes"]:
        name = node["id"]
        for species in ("A", "B"):
            result[f"node:{name}:amount:{species}"] = amounts[name][species]
        result[f"node:{name}:temperature"] = temps[name]
        result[f"node:{name}:pressure"] = (
            sum(amounts[name].values()) * R * temps[name] * 1000 / node["volume_L"]
        )
        result[f"node:{name}:bath"] = bath[name]
        result[f"node:{name}:loss"] = loss[name]
        result[f"node:{name}:delta_U"] = delta_u[name]
        result[f"node:{name}:conducted"] = sum(
            (int(link["target"] == name) - int(link["source"] == name)) * links[link["id"]]
            for link in raw["heat_links"]
        )
    for edge in raw["edges"]:
        for species in ("A", "B"):
            result[f"edge:{edge['id']}:amount:{species}"] = transfers[edge["id"]][species]
        result[f"edge:{edge['id']}:enthalpy"] = transfers[edge["id"]]["H"]
    for link in raw["heat_links"]:
        result[f"link:{link['id']}:heat"] = links[link["id"]]
    return result


def analytic(raw, times):
    samples = []
    for time in times:
        amounts = deepcopy(raw["amounts"])
        temps = {n["id"]: n["temperature_K"] for n in raw["nodes"]}
        zero = dict.fromkeys(temps, 0.0)
        bath, loss, du = dict(zero), dict(zero), dict(zero)
        transfers, links = {}, {}
        if raw.get("isothermal"):
            temperature = 700.0
            k = raw["kinetics"]["A"] if raw["nodes"][0]["chemistry"] else 0.0
            total_source, total_target = (
                sum(amounts["source"].values()),
                sum(amounts["receiver"].values()),
            )
            fraction = amounts["source"]["A"] / total_source
            transferred = transferred_a = 0.0
            if raw["edges"]:
                edge = raw["edges"][0]
                if edge["mode"] == "pressure-valve":
                    coefficient = edge["coefficient"] * R * temperature * 1000 / 0.002016
                    decay = coefficient * (1 / 1.0 + 1 / 3.0)
                    initial_flow = coefficient * (total_source / 1 - total_target / 3)
                    transferred = initial_flow / decay * (-np.expm1(-decay * time))
                else:
                    transferred = edge["coefficient"] * time
                transferred_a = (
                    fraction * transferred
                    if k == 0
                    else fraction * edge["coefficient"] * (-np.expm1(-k * time)) / k
                )
                h = {s: v["h"] + v["cp"] * (temperature - 298.15) for s, v in raw["thermo"].items()}
                transfers[edge["id"]] = {
                    "A": transferred_a,
                    "B": transferred - transferred_a,
                    "H": h["A"] * transferred_a + h["B"] * (transferred - transferred_a),
                }
            amounts["source"]["A"] = fraction * (total_source - transferred) * np.exp(-k * time)
            amounts["receiver"]["A"] = (amounts["receiver"]["A"] + fraction * transferred) * np.exp(
                -k * time
            )
            for name, total in (
                ("source", total_source - transferred),
                ("receiver", total_target + transferred),
            ):
                amounts[name]["B"] = total - amounts[name]["A"]
                du[name] = sum(
                    (amounts[name][s] - raw["amounts"][name][s])
                    * (v["h"] + v["cp"] * (temperature - 298.15) - R * temperature)
                    for s, v in raw["thermo"].items()
                )
                advected = (1 if name == "receiver" else -1) * sum(
                    v["H"] for v in transfers.values()
                )
                bath[name] = du[name] - advected
        else:
            capacities = {
                n["id"]: n["support_J_K"]
                + sum(amounts[n["id"]][s] * (v["cp"] - R) for s, v in raw["thermo"].items())
                for n in raw["nodes"]
            }
            if raw["unit_index"] == 6:
                link = raw["heat_links"][0]
                source, target = link["source"], link["target"]
                c1, c2 = capacities[source], capacities[target]
                t1, t2 = temps[source], temps[target]
                equilibrium = (c1 * t1 + c2 * t2) / (c1 + c2)
                difference = (t1 - t2) * np.exp(-link["conductance"] * (1 / c1 + 1 / c2) * time)
                temps[source] = equilibrium + c2 / (c1 + c2) * difference
                temps[target] = equilibrium - c1 / (c1 + c2) * difference
                links[link["id"]] = c1 * (t1 - temps[source])
            else:
                node = raw["nodes"][0]
                name = node["id"]
                temps[name] = node["environment_K"] + (
                    temps[name] - node["environment_K"]
                ) * np.exp(-node["loss_W_K"] * time / capacities[name])
                loss[name] = capacities[name] * (node["temperature_K"] - temps[name])
            du = {
                n["id"]: capacities[n["id"]] * (temps[n["id"]] - n["temperature_K"])
                for n in raw["nodes"]
            }
        samples.append(outputs(raw, amounts, temps, bath, loss, du, transfers, links))
    return samples


def energy_network(raw, times):
    nodes, edges, links = raw["nodes"], raw["edges"], raw["heat_links"]
    n, e = len(nodes), len(edges)
    index = {node["id"]: i for i, node in enumerate(nodes)}
    initial = np.zeros(5 * n + 3 * e + len(links), dtype=complex)

    def energy(node, amounts, temperature):
        return node["support_J_K"] * (temperature - node["reference_K"]) + sum(
            amounts[j]
            * (
                raw["thermo"][s]["h"]
                + raw["thermo"][s]["cp"] * (temperature - 298.15)
                - R * temperature
            )
            for j, s in enumerate(("A", "B"))
        )

    for i, node in enumerate(nodes):
        amounts = [raw["amounts"][node["id"]][s] for s in ("A", "B")]
        initial[5 * i : 5 * i + 3] = [*amounts, energy(node, amounts, node["temperature_K"])]

    def fields(y):
        amounts = [y[5 * i : 5 * i + 2] for i in range(n)]
        temperatures = []
        for i, node in enumerate(nodes):
            capacity = node["support_J_K"] + sum(
                amounts[i][j] * (raw["thermo"][s]["cp"] - R) for j, s in enumerate(("A", "B"))
            )
            offset = -node["support_J_K"] * node["reference_K"] + sum(
                amounts[i][j] * (raw["thermo"][s]["h"] - raw["thermo"][s]["cp"] * 298.15)
                for j, s in enumerate(("A", "B"))
            )
            temperatures.append((y[5 * i + 2] - offset) / capacity)
        return amounts, temperatures

    def rhs(_, y):
        amounts, temps = fields(y)
        dy = np.zeros_like(y)
        for i, node in enumerate(nodes):
            rate = (
                raw["kinetics"]["A"]
                * np.exp(-raw["kinetics"]["Ea"] / (R * temps[i]))
                * amounts[i][0]
                if node["chemistry"]
                else 0
            )
            dy[5 * i], dy[5 * i + 1] = -rate, rate
            loss = node["loss_W_K"] * (temps[i] - node["environment_K"])
            dy[5 * i + 2], dy[5 * i + 4] = -loss, loss
        for j, edge in enumerate(edges):
            a, b = index[edge["source"]], index[edge["target"]]
            transfer = edge["coefficient"] * amounts[a] / sum(amounts[a])
            enthalpies = np.array(
                [
                    raw["thermo"][s]["h"] + raw["thermo"][s]["cp"] * (temps[a] - 298.15)
                    for s in ("A", "B")
                ]
            )
            heat = enthalpies @ transfer
            dy[5 * a : 5 * a + 2] -= transfer
            dy[5 * b : 5 * b + 2] += transfer
            dy[5 * a + 2] -= heat
            dy[5 * b + 2] += heat
            dy[5 * n + 3 * j : 5 * n + 3 * j + 3] = [*transfer, heat]
        for j, link in enumerate(links):
            a, b = index[link["source"]], index[link["target"]]
            heat = link["conductance"] * (temps[a] - temps[b])
            dy[5 * a + 2] -= heat
            dy[5 * b + 2] += heat
            dy[5 * n + 3 * e + j] = heat
        return dy

    solution = solve_ivp(
        rhs, (0, max(times)), initial, method="DOP853", rtol=1e-11, atol=1e-13, t_eval=times
    )
    if not solution.success:
        raise ValueError(solution.message)
    result = []
    for y in solution.y.T:
        amounts, temperatures = fields(y)
        names = [node["id"] for node in nodes]
        result.append(
            outputs(
                raw,
                {
                    name: dict(zip(("A", "B"), amounts[i], strict=True))
                    for i, name in enumerate(names)
                },
                dict(zip(names, temperatures, strict=True)),
                {name: y[5 * i + 3] for i, name in enumerate(names)},
                {name: y[5 * i + 4] for i, name in enumerate(names)},
                {name: y[5 * i + 2] - initial[5 * i + 2] for i, name in enumerate(names)},
                {
                    edge["id"]: {
                        "A": y[5 * n + 3 * j],
                        "B": y[5 * n + 3 * j + 1],
                        "H": y[5 * n + 3 * j + 2],
                    }
                    for j, edge in enumerate(edges)
                },
                {link["id"]: y[5 * n + 3 * e + j] for j, link in enumerate(links)},
            )
        )
    return result


def cantera(raw, times):
    import cantera as ct
    import yaml
    from gas_network_sensitivity35_cases import document

    source = document(raw)
    source["elements"] = [{"symbol": "H", "atomic-weight": 1.008}]
    reactors = {}
    for node in raw["nodes"]:
        phase = ct.Solution(yaml=yaml.safe_dump(source), name=raw["id"])
        initial = raw["amounts"][node["id"]]
        phase.TPX = 700, sum(initial.values()) * R * 700 * 1000 / node["volume_L"], initial
        reactors[node["id"]] = ct.IdealGasReactor(
            phase, energy="off", volume=node["volume_L"] / 1000, clone=True
        )
    edge = raw["edges"][0]
    flow = ct.MassFlowController(
        reactors[edge["source"]], reactors[edge["target"]], mdot=edge["coefficient"]
    )
    solver = ct.ReactorNet(list(reactors.values()))
    solver.rtol, solver.atol = 1e-12, 1e-20
    initial_u = {
        name: reactor.mass * reactor.phase.int_energy_mass for name, reactor in reactors.items()
    }
    grid = np.linspace(0, max(times), 1001)
    fluxes, states = [], []
    for t in grid:
        solver.advance(float(t))
        phase = reactors[edge["source"]].phase
        molar = flow.mass_flow_rate / phase.mean_molecular_weight * 1000
        fluxes.append([*(molar * phase.X), flow.mass_flow_rate * phase.enthalpy_mass])
        states.append(
            {
                name: (
                    dict(
                        zip(
                            reactor.phase.species_names,
                            reactor.phase.concentrations * reactor.volume * 1000,
                            strict=True,
                        )
                    ),
                    reactor.mass * reactor.phase.int_energy_mass,
                )
                for name, reactor in reactors.items()
            }
        )
    result = []
    for t in times:
        position = round(t / max(times) * 1000)
        integrated = (
            np.zeros(3)
            if position == 0
            else simpson(np.asarray(fluxes[: position + 1]), x=grid[: position + 1], axis=0)
        )
        amounts = {name: values[0] for name, values in states[position].items()}
        du = {name: values[1] - initial_u[name] for name, values in states[position].items()}
        bath = {
            name: du[name] + (1 if name == edge["source"] else -1) * integrated[2]
            for name in reactors
        }
        result.append(
            outputs(
                raw,
                amounts,
                dict.fromkeys(reactors, 700.0),
                bath,
                dict.fromkeys(reactors, 0.0),
                du,
                {edge["id"]: dict(zip(("A", "B", "H"), integrated, strict=True))},
                {},
            )
        )
    return result


def reference(raw, bindings, times):
    method = (
        cantera
        if raw["unit_index"] == 4
        else energy_network
        if raw["unit_index"] == 8
        else analytic
    )
    baseline = [
        {key: float(np.real(value)) for key, value in row.items()} for row in method(raw, times)
    ]
    derivatives, auxiliary = [], []
    for binding in bindings:
        if raw["unit_index"] == 4:
            levels = []
            for h in (0.00125, 0.000625):
                plus, minus = (
                    method(vary(raw, binding, float(np.exp(h))), times),
                    method(vary(raw, binding, float(np.exp(-h))), times),
                )
                levels.append(
                    [
                        {key: float((a[key] - b[key]) / (2 * h)) for key in a}
                        for a, b in zip(plus, minus, strict=True)
                    ]
                )
            derivatives.append(levels[-1])
            auxiliary.append(levels)
        else:
            complex_samples = method(vary(raw, binding, np.exp(1e-20j)), times)
            derivatives.append(
                [
                    {key: float(np.imag(value) / 1e-20) for key, value in row.items()}
                    for row in complex_samples
                ]
            )
    return {
        "baseline": baseline,
        "derivatives": derivatives,
        "auxiliary": auxiliary,
        "method": "Cantera3.2-two-log-steps"
        if raw["unit_index"] == 4
        else "independent-nU-DOP853-complex-step"
        if raw["unit_index"] == 8
        else "closed-form-complex-step",
    }
