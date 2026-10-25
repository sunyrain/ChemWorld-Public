"""Independent n/U integration and actual Cantera3.2 SP/HP; no ChemWorld imports."""

from copy import deepcopy
from math import exp, log

import cantera as ct
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

R = 8.31446261815324


class Reference:
    def __init__(self, card):
        if ct.__version__ != "3.2.0":
            raise ValueError("independent reference must be actual Cantera3.2.0")
        self.card = deepcopy(card)
        self.spec = card["spec"]
        self.machine = self.spec["shaft_machine"]
        self.gas = ct.Solution(
            thermo="ideal-gas",
            species=[ct.Species.from_dict(s) for s in card["document"]["species"]],
        )
        self.nodes = {n["vessel_id"]: n for n in self.spec["nodes"]}
        self.energy = {n["vessel_id"]: n for n in self.spec["energy_nodes"]}
        self.constant = all(
            s["thermo"]["model"] == "constant-cp" for s in card["document"]["species"]
        )
        if self.constant:
            self.cp = np.array(
                [float(s["thermo"]["cp0"].split()[0]) for s in card["document"]["species"]]
            )
            self.h0 = np.array(
                [float(s["thermo"]["h0"].split()[0]) for s in card["document"]["species"]]
            )
        self.edge = self.spec["connections"][0]

    def h(self, t, x):
        if self.constant:
            return float(x @ (self.h0 + self.cp * (t - 298.15)))
        self.gas.TPX = t, 101325.0, x
        return self.gas.enthalpy_mole / 1000.0

    def u(self, name, n, t):
        energy = self.energy[name]
        return float(
            sum(n) * (self.h(t, n / sum(n)) - R * t)
            + energy["support_heat_capacity_J_K"] * (t - energy["reference_temperature_K"])
        )

    def temperature(self, name, n, u):
        e = self.energy[name]
        if self.constant:
            offset = (
                float(n @ (self.h0 - self.cp * 298.15))
                - e["support_heat_capacity_J_K"] * e["reference_temperature_K"]
            )
            return (u - offset) / (float(n @ (self.cp - R)) + e["support_heat_capacity_J_K"])
        return brentq(lambda t: self.u(name, n, t) - u, 150.0, 2000.0, xtol=1e-10)

    def point(self, t, p1, p2, x, request=None):
        machine = self.machine
        request = self.edge["coefficient"] if request is None else request
        if self.edge["mode"] == "mass-flow":
            request /= 0.002016
        ratio = p2 / p1 if machine["mode"] == "compressor" else p1 / p2
        hi = self.h(t, x)
        if not self.edge["enabled"] or request == 0 or ratio < 1:
            return {
                "T_s": t,
                "T_out": t,
                "h_in": hi,
                "h_out": hi,
                "w": 0.0,
                "flow": 0.0,
                "W": 0.0,
                "S": 0.0,
                "ratio": ratio,
            }
        eta = machine["isentropic_efficiency"]
        if self.constant:
            cp = float(x @ self.cp)
            ts = t * exp(R / cp * log(p2 / p1))
            tout = t + ((ts - t) / eta if machine["mode"] == "compressor" else (ts - t) * eta)
            ho = self.h(tout, x)
            entropy = cp * log(tout / t) - R * log(p2 / p1)
        else:
            self.gas.TPX = t, p1, x
            si, hi_mass = self.gas.entropy_mass, self.gas.enthalpy_mass
            self.gas.SP = si, p2
            # Cantera's public SP/HP setters have a fixed relative stopping
            # tolerance. Retain the actual setter calls, then independently
            # polish their physical equations using Cantera properties only.
            raw_ts = self.gas.T

            def entropy_residual(value):
                self.gas.TP = value, p2
                return self.gas.entropy_mass - si

            ts = brentq(entropy_residual, 150.0, 2000.0, xtol=1e-11)
            self.gas.TP = ts, p2
            hs_mass = self.gas.enthalpy_mass
            target = hi_mass + (
                (hs_mass - hi_mass) / eta
                if machine["mode"] == "compressor"
                else (hs_mass - hi_mass) * eta
            )
            self.gas.HP = target, p2
            raw_tout = self.gas.T

            def enthalpy_residual(value):
                self.gas.TP = value, p2
                return self.gas.enthalpy_mass - target

            tout = brentq(enthalpy_residual, 150.0, 2000.0, xtol=1e-11)
            self.gas.TP = tout, p2
            ho = self.gas.enthalpy_mole / 1000.0
            entropy = (self.gas.entropy_mass - si) * self.gas.mean_molecular_weight / 1000.0
        work = ho - hi
        flow = min(request, machine["maximum_shaft_power_W"] / abs(work)) if work else request
        return {
            "T_s": ts,
            "T_out": tout,
            "h_in": hi,
            "h_out": ho,
            "w": work,
            "flow": flow,
            "W": flow * work,
            "S": flow * entropy,
            "ratio": ratio,
            **(
                {"raw_cantera_SP_T": raw_ts, "raw_cantera_HP_T": raw_tout}
                if not self.constant
                else {}
            ),
        }

    def fields(self, y):
        amounts = {"source": y[:2], "receiver": y[3:5]}
        temp = {
            name: self.temperature(name, n, y[2 if name == "source" else 5])
            for name, n in amounts.items()
        }
        pressure = {
            name: sum(n) * R * temp[name] * 1000 / self.nodes[name]["volume_L"]
            for name, n in amounts.items()
        }
        p = self.point(
            temp["source"],
            pressure["source"],
            pressure["receiver"],
            amounts["source"] / sum(amounts["source"]),
        )
        return amounts, temp, pressure, p

    def initial(self):
        y = np.zeros(14)
        for name, offset in (("source", 0), ("receiver", 3)):
            n = np.array([self.card["amounts"][name][s] for s in ("A", "B")])
            y[offset : offset + 3] = [
                *n,
                self.u(name, n, self.energy[name]["initial_temperature_K"]),
            ]
        return y

    def solve(self):
        initial = self.initial()

        def rhs(_, y):
            n, t, _, p = self.fields(y)
            rates = p["flow"] * n["source"] / sum(n["source"])
            out = np.zeros(14)
            out[:2], out[3:5] = -rates, rates
            out[2], out[5] = -p["flow"] * p["h_in"], p["flow"] * p["h_out"]
            out[6:10] = [p["flow"] * p["h_in"], p["flow"] * p["h_out"], p["W"], p["S"]]
            for i, name in enumerate(("source", "receiver")):
                heat = self.card["heat_power_W"] if name == self.spec["anchor_vessel_id"] else 0.0
                e = self.energy[name]
                loss = e["loss_conductance_W_K"] * (t[name] - e["environment_temperature_K"])
                out[2 if i == 0 else 5] += heat - loss
                out[10 + i], out[12 + i] = heat, loss
            return out

        definitions = []
        for name in sorted(self.nodes):
            for kind in (
                "minimum_amount",
                "maximum_pressure",
                "minimum_temperature",
                "maximum_temperature",
            ):

                def event(_, y, name=name, kind=kind):
                    n, t, p, _ = self.fields(y)
                    return {
                        "minimum_amount": sum(n[name]) - self.nodes[name]["minimum_amount_mol"],
                        "maximum_pressure": self.nodes[name]["maximum_pressure_Pa"] - p[name],
                        "minimum_temperature": t[name] - self.energy[name]["minimum_temperature_K"],
                        "maximum_temperature": self.energy[name]["maximum_temperature_K"] - t[name],
                    }[kind]

                definitions.append((f"gas:{name}:{kind}", event))
        for kind in (
            "pressure_direction",
            "maximum_pressure_ratio",
            "minimum_outlet_temperature",
            "maximum_outlet_temperature",
        ):

            def event(_, y, kind=kind):
                p = self.fields(y)[3]
                return {
                    "pressure_direction": p["ratio"] - 1.0,
                    "maximum_pressure_ratio": self.machine["maximum_pressure_ratio"] - p["ratio"],
                    "minimum_outlet_temperature": p["T_out"]
                    - self.machine["minimum_outlet_temperature_K"],
                    "maximum_outlet_temperature": self.machine["maximum_outlet_temperature_K"]
                    - p["T_out"],
                }[kind]

            definitions.append((f"shaft:shaft-feed:{kind}", event))
        for _, event in definitions:
            event.terminal = True
            event.direction = -1
        solution = solve_ivp(
            rhs,
            (0, self.card["duration_s"]),
            initial,
            method="DOP853",
            rtol=2e-11,
            atol=1e-12,
            events=[fn for _, fn in definitions],
            dense_output=True,
        )
        if not solution.success:
            raise ValueError(solution.message)
        stopping = next(
            (
                name
                for (name, _), times in zip(definitions, solution.t_events, strict=True)
                if len(times)
            ),
            None,
        )
        times = self.card["sample_times_s"] or [float(solution.t[-1])]
        samples = []
        for time in times:
            if time > solution.t[-1] + 1e-8:
                raise ValueError("reference stopped before a declared sample")
            y = solution.sol(time)
            n, t, p, point = self.fields(y)
            samples.append(
                {
                    "time_s": time,
                    "amounts": {k: v.tolist() for k, v in n.items()},
                    "temperatures": t,
                    "pressures": {k: float(v) for k, v in p.items()},
                    "delta_U": {
                        "source": float(y[2] - initial[2]),
                        "receiver": float(y[5] - initial[5]),
                    },
                    "Hin": float(y[6]),
                    "Hout": float(y[7]),
                    "W": float(y[8]),
                    "S": float(y[9]),
                    "Q": float(y[10] + y[11]),
                    "loss": float(y[12] + y[13]),
                    "point": point,
                }
            )
        return {
            "success": True,
            "cantera_version": ct.__version__,
            "samples": samples,
            "stopping_event": stopping,
            "rhs_evaluations": solution.nfev,
            "steps": len(solution.t) - 1,
        }
