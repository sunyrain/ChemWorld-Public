"""Thin native-history adapter; independent reference equations live separately."""

from dataclasses import replace

import numpy as np
from battery_lead_acid_cases import KEYS, PRECISIONS, native_spec


def native_time(data, precision):
    progress = {"stage": "build", "current_time_s": 0.0, "completed_segments": []}
    try:
        return _native_time(data, precision, progress)
    except Exception as exc:
        exc.partial_history = progress
        raise


def _native_time(data, precision, progress):
    from chemworld.physchem.battery_lead_acid import lead_acid_fields, lead_acid_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

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
        system = lead_acid_system(spec, program)
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
        values, amounts, pore = [], [], []
        progress.update(stage="extract", current_time_s=result.final_state.time_s)
        progress["active_segment"].update(
            times_s=[s.time_s for s in states],
            values=values,
            amounts_mol=amounts,
            pore_volume_m3=pore,
            termination=result.stopping_event,
            outcome=result.outcome,
        )
        for saved in states:
            progress["current_time_s"] = saved.time_s
            f = lead_acid_fields(spec, saved, program=program)
            values.append(
                [
                    f["voltage_V"],
                    f["current_A"],
                    saved.coordinates["discharged_charge_C"],
                    saved.coordinates["delivered_work_J"],
                    f["acid_concentration_mol_m3"],
                    f["water_concentration_mol_m3"],
                    f["molality_mol_kg"],
                    *f["porosities"],
                    f["negative_ocp_V"],
                    f["positive_ocp_V"],
                    f["negative_overpotential_V"],
                    f["positive_overpotential_V"],
                    f["negative_exchange_current_A_m2"],
                    f["positive_exchange_current_A_m2"],
                ]
            )
            amounts.append([saved.amounts_mol[key] for key in KEYS])
            pore.append(f["total_pore_volume_m3"])
        segments.append(
            {
                "times_s": [s.time_s for s in states],
                "values": values,
                "amounts_mol": amounts,
                "pore_volume_m3": pore,
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
        "implementation": "Native six true species inventories / common CoupledSystem",
        "precision": precision,
        "tolerances": PRECISIONS[precision],
        "segments": segments,
    }


def refusal(data, name):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_lead_acid import lead_acid_system
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_soh import soh_from_battery
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data)
    state = spec.initial_state()
    try:
        if name == "incompatible-molar-volumes":
            replace(
                spec.material,
                molar_volumes_m3_mol=(
                    1.82539682539683e-5,
                    2.54797441364606e-5,
                    4.81717011128776e-5,
                    4.5e-5,
                    1.75e-5,
                ),
            )
        elif name == "incompatible-initial-water":
            values = list(spec.initial_amounts_mol)
            values[-1] *= 1.01
            replace(spec, initial_amounts_mol=tuple(values))
        elif name == "negative-inventory":
            values = list(spec.initial_amounts_mol)
            values[0] = -0.001
            replace(spec, initial_amounts_mol=tuple(values))
        elif name == "invalid-reaction-area":
            replace(spec.geometry, reaction_area_densities_m_inv=(0.0, 2e6))
        elif name == "thermal":
            lead_acid_system(
                spec, ScalarControlProgram(((0.0, 0.1),), "A"), thermal_device=ThermalDeviceSpec()
            )
        elif name == "chemical-energy":
            battery_chemical_energy_j(spec, state)
        elif name == "EIS":
            battery_impedance(spec, [1.0], state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        else:
            raise RuntimeError("unknown fixed refusal")
    except ValueError as exc:
        return {"refusal": name, "rejected": True, "reason": str(exc)}
    return {"refusal": name, "rejected": False}
