"""Actual unmodified SplitOCVR ODE/variables/events; no manufactured CV or W."""

import numpy as np
import pybamm
from battery_split_ocvr_cases import PRECISIONS


def build_reference(data, precision):
    def polynomial(coefficients):
        def evaluate(theta):
            return sum(float(v) * theta**i for i, v in enumerate(coefficients))

        return evaluate

    model = pybamm.lithium_ion.SplitOCVR()
    values = pybamm.ParameterValues(
        {
            "Negative electrode capacity [A.h]": data["negative_capacity_C"] / 3600,
            "Positive electrode capacity [A.h]": data["positive_capacity_C"] / 3600,
            "Negative electrode initial stoichiometry": data["initial_negative_fraction"],
            "Positive electrode initial stoichiometry": data["initial_positive_fraction"],
            "Ohmic resistance [Ohm]": data["resistance_ohm"],
            "Negative electrode OCP [V]": polynomial(data["negative_polynomial_V"]),
            "Positive electrode OCP [V]": polynomial(data["positive_polynomial_V"]),
            "Lower voltage cut-off [V]": data["minimum_voltage_V"],
            "Upper voltage cut-off [V]": data["maximum_voltage_V"],
            "Current function [A]": "[input]",
        }
    )
    return pybamm.Simulation(
        model,
        parameter_values=values,
        solver=pybamm.ScipySolver(method="DOP853", **PRECISIONS[precision]),
    )


def reference_time(data, precision):
    progress = {"stage": "build", "current_time_s": 0.0, "completed_segments": []}
    try:
        simulation = build_reference(data, precision)
        segments = progress["completed_segments"]
        for index, (duration, target, unit) in enumerate(data["waveform"]):
            if unit != "A":
                raise ValueError("original SplitOCVR source has prescribed current only")
            start = sum(w[0] for w in data["waveform"][:index])
            times = np.linspace(
                0.0, data["observation_horizons_s"][index], data["samples_per_segment"]
            )
            if times[-1] < duration:
                times = np.r_[times, duration]
            progress.update(
                stage="integrate",
                current_time_s=start,
                active_segment={
                    "index": index,
                    "start_time_s": start,
                    "duration_s": duration,
                    "target": target,
                    "unit": unit,
                },
            )
            solution = simulation.step(
                duration, save=False, t_eval=times, inputs={"Current function [A]": target}
            )
            progress.update(stage="extract", current_time_s=float(solution.t[-1]))
            progress["active_segment"].update(
                returned_times_s=solution.t.tolist(),
                original_source_state_coordinates=np.asarray(solution.y).tolist(),
                termination=solution.termination,
            )
            names = (
                "Voltage [V]",
                "Current [A]",
                "Discharge capacity [A.h]",
                "Negative particle stoichiometry",
                "Positive particle stoichiometry",
                "Negative electrode OCP [V]",
                "Positive electrode OCP [V]",
            )
            columns = [np.asarray(solution[n].entries).reshape(-1) for n in names]
            columns[2] = columns[2] * 3600
            columns.append(columns[1] * data["resistance_ohm"])
            segments.append(
                {
                    "times_s": solution.t.tolist(),
                    "values": np.column_stack(columns).tolist(),
                    "original_source_state_coordinates": np.asarray(solution.y).tolist(),
                    "end_time_s": float(solution.t[-1]),
                    "termination": solution.termination,
                }
            )
            if solution.termination != "final time":
                break
        return {
            "implementation": "Actual PyBaMM25.4.2 SplitOCVR prescribed-current ODE",
            "solver": "ScipySolver(DOP853)",
            "precision": precision,
            "tolerances": PRECISIONS[precision],
            "segments": segments,
            "state_scope": "Original theta_n/theta_p/Q_Ah only; no W or CV added",
        }
    except Exception as exc:
        exc.partial_history = progress
        raise
