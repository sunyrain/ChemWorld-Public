"""Independent constant-Cp n/T ODE and actual Cantera reactor network; no native imports."""

import numpy as np
from scipy.integrate import solve_ivp

R = 8.31446261815324
CP = np.array([30.0, 44.0])
H0 = np.array([0.0, -9000.0])
MW = np.array([0.002016, 0.004032])


def heat(card, name, temperature):
    d = card["device"]
    if d is None or name != card["anchor"] or not d["enabled"]:
        return 0.0
    return min(
        d["maximum_power_W"],
        max(
            -d["maximum_cooling_power_W"],
            d["conductance_W_K"] * d["calibration_factor"] * (d["setpoint_K"] - temperature),
        ),
    )


def fields(card, y):
    nodes, piston = card["nodes"], card["piston"]
    ns = np.array([y[5 * i : 5 * i + 2] for i in range(len(nodes))])
    ts = np.array([y[5 * i + 2] for i in range(len(nodes))])
    vs = np.array(
        [
            sum(ns[i]) * R * ts[i] * 1000 / piston["pressure_Pa"]
            if n["id"] == piston["vessel_id"]
            else n["volume_L"]
            for i, n in enumerate(nodes)
        ]
    )
    ps = ns.sum(axis=1) * R * ts * 1000 / vs
    return ns, ts, vs, ps


def ode(card, times=None, events=False):
    nodes, edges = card["nodes"], card["edges"]
    index = {n["id"]: i for i, n in enumerate(nodes)}
    initial = np.zeros(5 * len(nodes) + 3 * len(edges))
    for i, n in enumerate(nodes):
        initial[5 * i : 5 * i + 3] = [
            *[card["amounts"][n["id"]][s] for s in ("A", "B")],
            n["temperature_K"],
        ]

    def rhs(_, y):
        ns, ts, _, ps = fields(card, y)
        result = np.zeros_like(y)
        dh = np.zeros(len(nodes))
        qs = np.array([heat(card, n["id"], ts[i]) for i, n in enumerate(nodes)])
        for i, n in enumerate(nodes):
            rate = 0.3 * ns[i, 1] if n["chemistry"] else 0.0
            result[5 * i : 5 * i + 2] = [2 * rate, -rate]
        for j, e in enumerate(edges):
            a, b = index[e["source_vessel_id"]], index[e["target_vessel_id"]]
            rate = e["coefficient"]
            if e["mode"] == "pressure-valve":
                rate *= max(ps[a] - ps[b], 0.0)
            if e["mode"] != "molar-flow":
                rate /= float(ns[a] @ MW / sum(ns[a]))
            flow = rate * ns[a] / sum(ns[a])
            h = H0 + CP * (ts[a] - 298.15)
            enthalpy = float(flow @ h)
            result[5 * a : 5 * a + 2] -= flow
            result[5 * b : 5 * b + 2] += flow
            dh[a] -= enthalpy
            dh[b] += enthalpy
            result[5 * len(nodes) + 3 * j : 5 * len(nodes) + 3 * j + 3] = [*flow, enthalpy]
        for link in card["heat_links"]:
            a, b = index[link["source_vessel_id"]], index[link["target_vessel_id"]]
            q = link["conductance_W_K"] * (ts[a] - ts[b])
            qs[a] -= q
            qs[b] += q
        for i, n in enumerate(nodes):
            dn = result[5 * i : 5 * i + 2]
            piston = n["id"] == card["piston"]["vessel_id"]
            h = H0 + CP * (ts[i] - 298.15)
            chem = h if piston else h - R * ts[i]
            capacity = n["support"] + float(ns[i] @ (CP if piston else CP - R))
            dt = (qs[i] + dh[i] - float(dn @ chem)) / capacity if card["thermal"] else 0.0
            work = -R * (ts[i] * sum(dn) + sum(ns[i]) * dt) if piston else 0.0
            q = qs[i] if card["thermal"] else float(dn @ (h - R * ts[i])) - dh[i] - work
            result[5 * i + 2 : 5 * i + 5] = [dt, q, work]
        return result

    event_functions = []
    labels = []
    if events:
        for i, n in enumerate(nodes):
            kinds = ["minimum_amount", "maximum_pressure"]
            if card["thermal"]:
                kinds += ["minimum_temperature", "maximum_temperature"]
            if n["id"] == card["piston"]["vessel_id"]:
                kinds += ["minimum_volume", "maximum_volume"]
            for kind in kinds:

                def event(_, y, i=i, n=n, kind=kind):
                    ns, ts, vs, ps = fields(card, y)
                    return {
                        "minimum_amount": sum(ns[i]) - n["min_n"],
                        "maximum_pressure": n["max_P"] - ps[i],
                        "minimum_temperature": ts[i] - n["min_T"],
                        "maximum_temperature": n["max_T"] - ts[i],
                        "minimum_volume": vs[i] - card["piston"]["minimum_volume_L"],
                        "maximum_volume": card["piston"]["maximum_volume_L"] - vs[i],
                    }[kind]

                event.terminal = True
                event.direction = -1
                event_functions.append(event)
                labels.append(f"gas:{n['id']}:{kind}")
    result = solve_ivp(
        rhs,
        (0, card["duration_s"]),
        initial,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        dense_output=True,
        events=event_functions or None,
    )
    if not result.success:
        raise RuntimeError(result.message)
    sample_times = [result.t[-1]] if times is None else times
    samples = []
    for t in sample_times:
        y = result.sol(t)
        ns, ts, vs, ps = fields(card, y)
        samples.append(
            {
                "time": float(t),
                "amounts": ns.tolist(),
                "temperatures": ts.tolist(),
                "volumes": vs.tolist(),
                "pressures": ps.tolist(),
                "heat": y[3 : 5 * len(nodes) : 5].tolist(),
                "work": y[4 : 5 * len(nodes) : 5].tolist(),
                "edges": y[5 * len(nodes) :].reshape(-1, 3).tolist(),
            }
        )
    stopped = next((labels[i] for i, ts in enumerate(result.t_events or []) if len(ts)), None)
    return {"samples": samples, "time": float(result.t[-1]), "event": stopped, "nfev": result.nfev}


def cantera(card, times):
    import cantera as ct
    import yaml

    class SupportedPiston(ct.ExtensibleIdealGasConstPressureReactor):
        def after_eval(self, t, lhs, rhs):
            if self.energy_enabled:
                lhs[self.component_index("temperature")] += self.support

    class SupportedTank(ct.ExtensibleIdealGasReactor):
        def after_eval(self, t, lhs, rhs):
            if self.energy_enabled:
                lhs[self.component_index("temperature")] += self.support

    reactors = {}
    for n in card["nodes"]:
        gas = ct.Solution(yaml=yaml.safe_dump(card["mechanism"]), name="piston")
        ns = card["amounts"][n["id"]]
        gas.TPX = (
            n["temperature_K"],
            sum(ns.values()) * R * n["temperature_K"] * 1000 / n["volume_L"],
            ns,
        )
        gas.set_multiplier(1.0 if n["chemistry"] else 0.0)
        cls = SupportedPiston if n["id"] == card["piston"]["vessel_id"] else SupportedTank
        reactor = cls(
            gas, energy="on" if card["thermal"] else "off", volume=n["volume_L"] * 0.001, clone=True
        )
        reactor.support = n["support"]
        reactors[n["id"]] = reactor
    links = []
    for e in card["edges"]:
        a, b = reactors[e["source_vessel_id"]], reactors[e["target_vessel_id"]]
        if e["mode"] == "pressure-valve":
            link = ct.Valve(a, b, K=e["coefficient"])
        else:

            def rate(t, a=a, e=e):
                return e["coefficient"] * (
                    a.phase.mean_molecular_weight / 1000 if e["mode"] == "molar-flow" else 1.0
                )

            link = ct.MassFlowController(a, b, mdot=rate)
        links.append(link)
    reservoir = ct.Reservoir(next(iter(reactors.values())).phase, clone=True)
    for n in card["nodes"]:
        reactor = reactors[n["id"]]
        if card["device"] is not None and n["id"] == card["anchor"]:
            links.append(
                ct.Wall(
                    reservoir,
                    reactor,
                    A=1.0,
                    Q=lambda t, r=reactor, name=n["id"]: heat(card, name, r.T),
                )
            )
    for e in card["heat_links"]:
        links.append(
            ct.Wall(
                reactors[e["source_vessel_id"]],
                reactors[e["target_vessel_id"]],
                A=1.0,
                U=e["conductance_W_K"],
            )
        )
    sim = ct.ReactorNet(list(reactors.values()))
    sim.rtol = 1e-11
    sim.atol = 1e-18
    samples = []
    for t in times:
        sim.advance(t)
        samples.append(
            {
                "time": t,
                "amounts": [
                    (r.phase.concentrations * r.volume * 1000).tolist() for r in reactors.values()
                ],
                "temperatures": [r.T for r in reactors.values()],
                "volumes": [r.volume * 1000 for r in reactors.values()],
                "pressures": [r.phase.P for r in reactors.values()],
            }
        )
    return {
        "samples": samples,
        "solver_stats": sim.solver_stats,
        "version": ct.__version__,
        "support_extension": "after_eval adds vessel support heat capacity to temperature LHS",
    }
