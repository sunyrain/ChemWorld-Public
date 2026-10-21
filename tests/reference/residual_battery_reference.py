"""Independent synthetic isothermal SPM reference for the fixed 21C block.

Only public parameter dictionaries cross this boundary. Geometry, fluxes,
surface reconstruction, voltage/current and quadratures are assembled here;
no ChemWorld numerical or physical implementation is imported. LSODA and
scalar Brent roots are validation tools, not a second production backend.
"""

from math import pi

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

FARADAY = 96485.33212
GAS = 8.31446261815324


class IndependentSPM:
    def __init__(self, parameters, program):
        if parameters["thermal"] is not None or parameters["sei"] is not None:
            raise ValueError("21C reference is isothermal and has no SEI")
        self.parameters = parameters
        self.program = program
        self.particles = []
        initial = []
        offset = 0
        for name in ("negative", "positive"):
            data = parameters[name]
            radius, cells = data["radius_m"], data["cells"]
            faces = radius * (1 - (1 - np.linspace(0, 1, cells + 1)) ** data["mesh_exponent"])
            count = data["active_volume_m3"] / (4 * pi * radius**3 / 3)
            volumes = count * 4 * pi * np.diff(faces**3) / 3
            areas = count * 4 * pi * faces**2
            centers = (faces[:-1] + faces[1:]) / 2
            # Independently integrate shell moments (three-point Gauss is exact
            # for r^4). The production shell-average discretization no longer
            # identifies an inventory mean with its arithmetic midpoint value.
            nodes, weights = np.polynomial.legendre.leggauss(3)
            points = centers[:, None] + np.diff(faces)[:, None] * nodes / 2
            mean_square = (points**4 @ weights) / (points**2 @ weights)
            self.particles.append(
                {
                    "name": name,
                    "data": data,
                    "volumes": volumes,
                    "areas": areas,
                    "centers": centers,
                    "gradient_spacing": np.diff(mean_square) / (2 * faces[1:-1]),
                    "surface_distance": (radius**2 - mean_square[-1]) / (2 * radius),
                    "slice": slice(offset, offset + cells),
                }
            )
            initial.extend([parameters[f"initial_{name}_fraction"]] * cells)
            offset += cells
        self.initial = np.r_[initial, 0.0, 0.0]

    def target(self, time):
        points = np.asarray(self.program["setpoints"])
        if self.program["mode"] != "linear":
            raise ValueError("fixed reference block uses linear controls")
        return float(np.interp(time, points[:, 0], points[:, 1]))

    def voltage(self, state, current):
        potentials, overpotentials, surfaces = [], [], []
        for particle, sign in zip(self.particles, (1, -1), strict=True):
            name, data = particle["name"], particle["data"]
            area = particle["areas"][-1]
            flux = sign * current / FARADAY
            surface = state[particle["slice"]][-1] - (
                flux
                * particle["surface_distance"]
                / (area * data["diffusivity_m2_s"] * data["maximum_concentration_mol_m3"])
            )
            surfaces.append(float(surface))
            # Only trial continuation outside the terminal surfaces uses this
            # smooth BV extension. Actual surface diagnostics stay unclipped.
            margin = self.parameters["surface_margin"]
            trial = np.clip(surface, margin / 2, 1 - margin / 2)
            exchange = data["exchange_current_scale_A_m2"] * np.sqrt(trial * (1 - trial))
            overpotentials.append(
                2
                * GAS
                * self.parameters["temperature_K"]
                / FARADAY
                * np.arcsinh(sign * current / (2 * area * exchange))
            )
            potentials.append(
                self.parameters[f"{name}_ocv_intercept_V"]
                + self.parameters[f"{name}_ocv_slope_V"] * surface
            )
        voltage = (
            potentials[1]
            - potentials[0]
            + overpotentials[1]
            - overpotentials[0]
            - current * self.parameters["series_resistance_ohm"]
        )
        return float(voltage), tuple(surfaces)

    def current(self, time, state):
        target = self.target(time)
        if self.program["unit"] == "A":
            return target
        limit = self.parameters["maximum_current_A"]
        if self.voltage(state, -limit)[0] < target:
            return -limit
        if self.voltage(state, limit)[0] > target:
            return limit
        return brentq(
            lambda current: self.voltage(state, current)[0] - target, -limit, limit, xtol=1e-13
        )

    def rhs(self, time, state):
        current = self.current(time, state)
        rates = []
        for particle, sign in zip(self.particles, (1, -1), strict=True):
            data = particle["data"]
            cmax = data["maximum_concentration_mol_m3"]
            internal = (
                -data["diffusivity_m2_s"]
                * particle["areas"][1:-1]
                * cmax
                * np.diff(state[particle["slice"]])
                / particle["gradient_spacing"]
            )
            faces = np.r_[0.0, internal, sign * current / FARADAY]
            rates.extend(-np.diff(faces) / (cmax * particle["volumes"]))
        return np.r_[rates, current, current * self.voltage(state, current)[0]]

    def solve(self, duration):
        def stop_values(time, state):
            current = self.current(time, state)
            voltage, surfaces = self.voltage(state, current)
            margin = self.parameters["surface_margin"]
            return (
                voltage - self.parameters["minimum_voltage_V"],
                self.parameters["maximum_voltage_V"] - voltage,
                *(surface - margin for surface in surfaces),
                *(1 - margin - surface for surface in surfaces),
            )

        names = [
            "minimum-voltage",
            "maximum-voltage",
            "negative-low",
            "positive-low",
            "negative-high",
            "positive-high",
        ]
        events = []
        for index in range(len(names)):

            def event(time, state, index=index):
                return stop_values(time, state)[index]

            event.terminal, event.direction = True, -1
            events.append(event)
        if self.program["unit"] == "V":
            limit = self.parameters["maximum_current_A"]
            for name, current in (("positive-limit", limit), ("negative-limit", -limit)):

                def transition(time, state, current=current):
                    return self.voltage(state, current)[0] - self.target(time)

                transition.terminal, transition.direction = False, 0
                names.append(name)
                events.append(transition)
        solution = solve_ivp(
            self.rhs,
            (0, duration),
            self.initial,
            method="LSODA",
            rtol=3e-10,
            atol=1e-12,
            max_step=duration / 100,
            dense_output=True,
            events=events,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution, dict(zip(names, solution.t_events, strict=True))
