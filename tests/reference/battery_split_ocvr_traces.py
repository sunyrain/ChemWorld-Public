"""Native execution/read-only adapters; independent physics lives in oracle.py."""

from dataclasses import replace

import numpy as np
from battery_split_ocvr_cases import PRECISIONS, analysis_data, native_spec
from battery_split_ocvr_oracle import cc_work, inventories


def native_time(data, precision):
    from chemworld.physchem.battery_split_ocvr import split_ocvr_fields, split_ocvr_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    progress = {"stage": "build", "current_time_s": 0.0, "completed_segments": []}
    try:
        spec = native_spec(data)
        state = spec.initial_state()
        segments = progress["completed_segments"]
        for index, (duration, target, unit) in enumerate(data["waveform"]):
            progress.update(
                stage="construct_segment",
                current_time_s=state.time_s,
                active_segment={
                    "index": index,
                    "start_time_s": state.time_s,
                    "duration_s": duration,
                    "target": target,
                    "unit": unit,
                },
            )
            program = ScalarControlProgram(((0.0, target),), unit)
            system = split_ocvr_system(spec, program)
            system = replace(system, policy=replace(system.policy, **PRECISIONS[precision]))
            progress["stage"] = "integrate"
            result = system.advance(
                state,
                duration,
                evaluation_times_s=np.linspace(
                    0.0, data["observation_horizons_s"][index], data["samples_per_segment"]
                ),
            )
            states = list(result.trajectory)
            if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
                states.append(result.final_state)
            values, amounts, work = [], [], []
            progress.update(stage="extract", current_time_s=result.final_state.time_s)
            progress["active_segment"].update(
                times_s=[s.time_s for s in states],
                values=values,
                amounts_mol=amounts,
                work_J=work,
                termination=result.stopping_event,
                outcome=result.outcome,
            )
            for saved in states:
                f = split_ocvr_fields(spec, saved, program=program)
                values.append(
                    [
                        f["voltage_V"],
                        f["current_A"],
                        saved.coordinates["discharged_charge_C"],
                        f["negative_average_fraction"],
                        f["positive_average_fraction"],
                        f["negative_ocv_V"],
                        f["positive_ocv_V"],
                        f["ohmic_drop_V"],
                    ]
                )
                amounts.append([saved.amounts_mol[k] for k in ("negative:Li", "positive:Li")])
                work.append(saved.coordinates["delivered_work_J"])
            segments.append(
                {
                    "times_s": [s.time_s for s in states],
                    "values": values,
                    "amounts_mol": amounts,
                    "work_J": work,
                    "end_time_s": result.final_state.time_s,
                    "termination": result.stopping_event,
                    "outcome": result.outcome,
                    "solver": dict(result.solver_diagnostic),
                    "conservation_residuals": dict(result.conservation_residuals),
                }
            )
            state = result.final_state
            if result.outcome != "success":
                break
        return {
            "implementation": "Native two true Li inventories and common residual",
            "precision": precision,
            "tolerances": PRECISIONS[precision],
            "segments": segments,
        }
    except Exception as exc:
        exc.partial_history = progress
        raise


def refusal(data, name):
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_split_ocvr import split_ocvr_system
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data)
    try:
        if name in ("negative-capacity", "positive-capacity"):
            replace(spec, **{name.replace("-", "_") + "_C": 0.0})
        elif name == "negative-R":
            replace(spec, resistance_ohm=-0.1)
        elif name == "initial-fraction":
            replace(spec, initial_negative_fraction=1.0)
        elif name == "inverted-bounds":
            replace(spec, positive_minimum_fraction=0.8, positive_maximum_fraction=0.2)
        elif name == "zero-R-CV":
            z = replace(spec, resistance_ohm=0.0)
            split_ocvr_system(z, ScalarControlProgram(((0.0, 4.0),), "V")).advance(
                z.initial_state(), 1.0
            )
        elif name == "thermal":
            split_ocvr_system(
                spec, ScalarControlProgram(((0.0, 0.0),), "A"), thermal_device=ThermalDeviceSpec()
            )
        elif name == "chemical-energy":
            battery_chemical_energy_j(spec, spec.initial_state())
        else:
            raise RuntimeError("unknown authored refusal")
    except ValueError as exc:
        return {"refusal": name, "rejected": True, "reason": str(exc)}
    return {"refusal": name, "rejected": False}


def state_snapshot(spec, state):
    return {
        "spec": spec.to_dict(),
        "time_s": state.time_s,
        "amounts_mol": dict(state.amounts_mol),
        "coordinates": dict(state.coordinates),
        "temperature_K": state.temperature_K,
        "volume_L": state.volume_L,
    }


def analysis(kind, name):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_soh import soh_from_battery

    data = analysis_data(kind, name)
    spec = native_spec(data)
    state = spec.initial_state()
    q = data["analysis_charge_C"]
    if q:
        state = replace(
            state,
            time_s=q / 0.5,
            amounts_mol=dict(
                zip(("negative:Li", "positive:Li"), inventories(data, q), strict=True)
            ),
            coordinates={"discharged_charge_C": q, "delivered_work_J": cc_work(data, 0.0, q, 0.5)},
        )
    before = state_snapshot(spec, state)
    if kind == "EIS":
        frequencies = np.geomspace(1e-4, 1e2, 17)
        result = battery_impedance(spec, frequencies, state=state)
        payload = {"frequencies_Hz": frequencies.tolist(), "result": result.to_dict()}
    else:
        result = soh_from_battery(spec, state, sensitivity=False)
        payload = {
            "result": result.to_dict(),
            "status": result.status,
            "branch": result.branch() if result.status == "unique" else None,
            "soc_values": [result.at_soc(v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
            if result.status == "unique"
            else [],
        }
    return {
        "kind": kind,
        "configuration": name,
        "inputs": data,
        **payload,
        "before": before,
        "after": state_snapshot(spec, state),
    }
