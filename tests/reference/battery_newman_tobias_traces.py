"""Native history adapter and independent limiting formulas for the fixed NT block."""

from dataclasses import replace

import numpy as np
from battery_newman_tobias_cases import ATOL, PROFILE_POINTS, RTOL, native_spec


def native_time(data, cells):
    progress = {"completed_segments": [], "stage": "build", "current_time_s": 0.0}
    try:
        return _native_time(data, cells, progress)
    except Exception as exc:
        exc.partial_history = progress
        raise


def _native_time(data, cells, progress):
    from chemworld.physchem.battery_newman_tobias import (
        newman_tobias_fields,
        newman_tobias_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, cells)
    state, segments = spec.initial_state(), progress["completed_segments"]
    starts = np.cumsum((0.0, *data["lengths_m"]))
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
                "requested_end_time_s": state.time_s + duration,
            },
        )
        program = ScalarControlProgram(((0.0, target),), unit)
        system = newman_tobias_system(spec, program)
        system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
        progress["stage"] = "integrate"
        result = system.advance(
            state,
            duration,
            evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
        )
        progress.update(stage="extract", current_time_s=result.final_state.time_s)
        states = list(result.trajectory)
        if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
            states.append(result.final_state)
        values, raw, sampled = [], {}, {}
        progress["active_segment"].update(
            times_s=[s.time_s for s in states],
            values=values,
            raw_profiles=raw,
            sampled_profiles=sampled,
            returned_end_time_s=result.final_state.time_s,
            termination=result.stopping_event,
            outcome=result.outcome,
        )
        for saved in states:
            progress["current_time_s"] = saved.time_s
            fields = newman_tobias_fields(spec, saved, program=program)
            values.append(
                [
                    fields["voltage_V"],
                    fields["current_A"],
                    saved.coordinates["discharged_charge_C"],
                    saved.coordinates["delivered_work_J"],
                    fields["negative_average_fraction"],
                    fields["positive_average_fraction"],
                    saved.amounts_mol["negative:Li"],
                    saved.amounts_mol["positive:Li"],
                ]
            )
            offset = electrode_offset = 0
            for region, (name, count) in enumerate(
                zip(("negative", "separator", "positive"), cells, strict=True)
            ):
                xi, xf = (np.arange(count) + 0.5) / count, np.linspace(0.0, 1.0, count + 1)
                record = raw.setdefault(
                    name,
                    {
                        "nodes_m": (starts[region] + xi * data["lengths_m"][region]).tolist(),
                        "edges_m": (starts[region] + xf * data["lengths_m"][region]).tolist(),
                    },
                )
                selected = {
                    "electrolyte_potential_V": np.asarray(fields["electrolyte_potential_V"])[
                        offset : offset + count
                    ],
                    "electrolyte_face_current_A": np.asarray(
                        fields["electrolyte_face_current_A_m2"]
                    )[offset : offset + count + 1]
                    * data["area_m2"],
                }
                interpolated = {
                    key: np.interp(PROFILE_POINTS, xf if "face" in key else xi, value)
                    for key, value in selected.items()
                }
                if name != "separator":
                    q = np.asarray(fields["reaction_current_A"])[
                        electrode_offset : electrode_offset + count
                    ]
                    selected.update(
                        solid_potential_V=np.asarray(fields["solid_potential_V"])[
                            electrode_offset : electrode_offset + count
                        ],
                        solid_face_current_A=np.asarray(fields[name + "_solid_face_current_A_m2"])
                        * data["area_m2"],
                        reaction_current_A=q,
                        interfacial_current_density_A_m2=np.asarray(
                            fields["interfacial_current_A_m2"]
                        )[electrode_offset : electrode_offset + count],
                    )
                    interpolated.update(
                        solid_potential_V=np.interp(
                            PROFILE_POINTS, xi, selected["solid_potential_V"]
                        ),
                        solid_face_current_A=np.interp(
                            PROFILE_POINTS, xf, selected["solid_face_current_A"]
                        ),
                        reaction_density_A=np.interp(PROFILE_POINTS, xi, q * count),
                    )
                    electrode_offset += count
                for key, value in selected.items():
                    record.setdefault(key, []).append(value.tolist())
                for key, value in interpolated.items():
                    sampled.setdefault(name, {}).setdefault(key, []).append(value.tolist())
                offset += count
        segments.append(
            {
                "times_s": [s.time_s for s in states],
                "values": values,
                "raw_profiles": raw,
                "sampled_profiles": sampled,
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
        "implementation": "Native sole two mean Li inventories / common coupled residual",
        "thickness_cells": list(cells),
        "segments": segments,
        "profile_fractional_coordinates": PROFILE_POINTS.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
    }


def refusal(data, name):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_newman_tobias import (
        newman_tobias_fields,
        newman_tobias_system,
    )
    from chemworld.physchem.battery_soh import soh_from_battery
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, (12, 6, 12))
    state = spec.initial_state()
    try:
        if name == "invalid-active-volume":
            replace(spec, negative=replace(spec.negative, active_volume_m3=1.1e-6))
        elif name == "nonpositive-conductivity":
            replace(spec.geometry, conductivity_S_m=0.0)
        elif name == "invalid-inventory":
            newman_tobias_fields(
                spec,
                replace(state, amounts_mol={**state.amounts_mol, "negative:Li": -1e-5}),
                current_A=0.0,
            )
        elif name == "radial-state":
            newman_tobias_fields(
                spec,
                replace(state, amounts_mol={**state.amounts_mol, "negative:Li:0": 1e-4}),
                current_A=0.0,
            )
        elif name == "EIS":
            battery_impedance(spec, [1.0], state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        elif name == "energy":
            battery_chemical_energy_j(spec, state)
        elif name == "thermal":
            newman_tobias_system(
                spec, ScalarControlProgram(((0.0, 0.1),), "A"), thermal_device=ThermalDeviceSpec()
            )
        else:
            raise RuntimeError("unknown fixed refusal")
    except ValueError as exc:
        return {"refusal": name, "rejected": True, "reason": str(exc)}
    return {"refusal": name, "rejected": False}
