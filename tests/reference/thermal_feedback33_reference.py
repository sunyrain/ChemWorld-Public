"""Independent thermal plant, matrix exponential and actual IDAES controller API."""

from copy import deepcopy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

R = 8.31446261815324


class Reference:
    def __init__(self, card, *, idaes=False):
        self.card = deepcopy(card)
        self.c, self.d, self.p = (card[k] for k in ("controller", "device", "plant"))
        self.idaes = idaes
        self.controller = None
        if idaes:
            from importlib.metadata import version

            from idaes.core import FlowsheetBlock
            from idaes.models.control.controller import ControllerType, PIDController
            from pyomo.environ import ConcreteModel, Var, units

            if version("idaes-pse") != "2.13.0" or self.c["derivative_gain_s"]:
                raise ValueError("actual IDAES reference is bounded to 2.13.0 P/PI equations")
            self.model = model = ConcreteModel()
            model.fs = FlowsheetBlock(dynamic=True, time_set=[0, 1], time_units=units.s)
            model.fs.pv = Var(model.fs.time, initialize=300, units=units.K)
            model.fs.mv = Var(model.fs.time, initialize=350, units=units.K)
            model.fs.controller = PIDController(
                process_var=model.fs.pv,
                manipulated_var=model.fs.mv,
                controller_type=ControllerType.PI
                if self.c["integral_gain_s_inv"]
                else ControllerType.P,
                calculate_initial_integral=False,
            )
            self.controller = model.fs.controller

    def point(self, y, segment):
        t, integral, filtered = y[:3]
        c, d = self.c, self.d
        target, enabled = segment["target_K"], segment["enabled"]
        error = target - t
        derivative = (t - filtered) / c["derivative_filter_time_s"]
        raw = (
            c["bias_temperature_K"]
            + c["proportional_gain"] * error
            + integral
            - c["derivative_gain_s"] * derivative
        )
        integral_base = c["integral_gain_s_inv"] * error
        if self.idaes:
            from pyomo.environ import value
            from pyomo.util.calc_var_value import calculate_variable_from_constraint

            ctl = self.controller
            ctl.process_var[0].set_value(float(t))
            ctl.setpoint[0].set_value(target)
            ctl.gain_p[0].set_value(c["proportional_gain"])
            ctl.mv_ref[0].set_value(c["bias_temperature_K"])
            if c["integral_gain_s_inv"]:
                ctl.gain_i[0].set_value(c["integral_gain_s_inv"])
                ctl.mv_integral_component[0].set_value(float(integral))
                calculate_variable_from_constraint(
                    ctl.mv_integral_component_dot[0], ctl.mv_integration_eqn[0]
                )
                integral_base = value(ctl.mv_integral_component_dot[0])
            raw = value(ctl.mv_unbounded[0])
        command = np.clip(raw, c["minimum_command_K"], c["maximum_command_K"])
        ua = d["conductance_W_K"] * d["calibration_factor"]
        power = (
            np.clip(ua * (command - t), -d["maximum_cooling_power_W"], d["maximum_power_W"])
            if enabled
            else 0.0
        )
        realized = t + power / ua
        idot = (
            integral_base + c["antiwindup_rate_s_inv"] * (realized - raw)
            if enabled and c["integral_gain_s_inv"]
            else 0.0
        )
        return {
            "raw": float(raw),
            "command": float(command),
            "power": float(power),
            "realized": float(realized),
            "Idot": float(idot),
            "Fdot": float(derivative),
        }

    def linear(self, y, dt, segment):
        c, p, d = self.c, self.p, self.d
        ua = d["conductance_W_K"] * d["calibration_factor"]
        gain = c["derivative_gain_s"] / c["derivative_filter_time_s"]
        heat = np.array(
            [
                -ua * (1 + c["proportional_gain"] + gain),
                ua,
                ua * gain,
                0.0,
                0.0,
                ua * (c["bias_temperature_K"] + c["proportional_gain"] * segment["target_K"]),
            ]
        )
        loss = np.array([p["loss_W_K"], 0.0, 0.0, 0.0, 0.0, -p["loss_W_K"] * p["ambient_K"]])
        matrix = np.zeros((6, 6))
        matrix[0] = (heat - loss) / p["capacity_J_K"]
        matrix[1, 0], matrix[1, 5] = (
            -c["integral_gain_s_inv"],
            c["integral_gain_s_inv"] * segment["target_K"],
        )
        matrix[2, 0], matrix[2, 2] = (
            1 / c["derivative_filter_time_s"],
            -1 / c["derivative_filter_time_s"],
        )
        matrix[3], matrix[4] = heat, loss
        return (expm(matrix * dt) @ np.r_[y, 1.0])[:5]

    def solve(self):
        y, elapsed, samples, resources = np.array(self.card["initial"], dtype=float), 0.0, [], []
        stopping = None
        for segment in self.card["segments"]:
            start = y.copy()

            def rhs(_, value, segment=segment):
                point = self.point(value, segment)
                loss = self.p["loss_W_K"] * (value[0] - self.p["ambient_K"])
                return [
                    (point["power"] - loss) / self.p["capacity_J_K"],
                    point["Idot"],
                    point["Fdot"],
                    point["power"],
                    loss,
                ]

            events = [
                ("minimum_temperature", lambda _, x: x[0] - self.p["minimum_temperature_K"]),
                ("temperature_limit", lambda _, x: self.p["stop_temperature_K"] - x[0]),
                (
                    "pressure_limit",
                    lambda _, x: (
                        self.p["maximum_pressure_Pa"]
                        - self.p["n_mol"] * R * x[0] * 1000 / self.p["volume_L"]
                    ),
                ),
            ]
            for _, event in events:
                event.terminal, event.direction = True, -1
            result = solve_ivp(
                rhs,
                (0, segment["duration_s"]),
                y,
                method="DOP853",
                rtol=2e-11,
                atol=1e-12,
                dense_output=True,
                events=[e for _, e in events],
            )
            if not result.success:
                raise ValueError(result.message)
            stopping = next(
                (
                    name
                    for (name, _), times in zip(events, result.t_events, strict=True)
                    if len(times)
                ),
                None,
            )
            for time in segment["sample_times_s"] or [float(result.t[-1])]:
                if time > result.t[-1] + 1e-8:
                    raise ValueError("reference stopped before fixed sample")
                values = result.sol(time)
                sample = {
                    "time_s": elapsed + time,
                    "values": values.tolist(),
                    "point": self.point(values, segment),
                }
                if self.card["linear_reference"]:
                    sample["matrix_exponential"] = self.linear(start, time, segment).tolist()
                samples.append(sample)
            y, elapsed = result.y[:, -1], elapsed + float(result.t[-1])
            resources.append({"rhs": result.nfev, "steps": len(result.t) - 1})
            if stopping:
                break
        return {
            "samples": samples,
            "stopping_event": stopping,
            "resources": resources,
            "actual_IDAES_equations": self.idaes,
            "IDAES_full_DAE_or_IPOPT": False,
        }
