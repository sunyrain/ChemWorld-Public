"""Real native phase histories and independent inventory/port balance measurements."""

from dataclasses import replace

import numpy as np
from battery_composite_cases import ATOL, PROFILE_RADII, RESPONSES, RTOL, F, native_spec


def native_time(data, cells):
    from chemworld.physchem.battery_composite import composite_fields, composite_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, cells)
    state, segments = spec.initial_state(), []
    for duration, target, unit in data["waveform"]:
        program = ScalarControlProgram(((0., target),), unit)
        system = composite_system(spec, program)
        system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
        result = system.advance(state, duration,
            evaluation_times_s=np.linspace(0., duration, data["samples_per_segment"]))
        states = list(result.trajectory)
        if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
            states.append(result.final_state)
        rows = []
        for saved in states:
            fields = composite_fields(spec, saved, program=program)
            row = [fields["voltage_V"], fields["current_A"],
                   saved.coordinates["discharged_charge_C"], saved.coordinates["delivered_work_J"]]
            for i, (_, _, _, keys, _) in enumerate(spec.particle_bins):
                row.extend([fields["reaction_current_A"][i], fields["surface_fraction"][i],
                    fields["average_fraction"][i], sum(saved.amounts_mol[k] for k in keys),
                    fields["kinetic_overpotential_V"][i]])
            rows.append(row)
        profiles = []
        for _, _, particle, keys, _ in spec.particle_bins:
            profile = np.array([result.final_state.amounts_mol[k] for k in keys]) / (
                particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3)
            profiles.append(np.interp(PROFILE_RADII, particle.mesh.centers_m / particle.radius_m,
                                      profile).tolist())
        segments.append({"times_s": [s.time_s for s in states], "rows": rows,
            "end_time_s": result.final_state.time_s, "termination": result.stopping_event,
            "outcome": result.outcome, "profiles_theta": profiles,
            "conservation_residuals": dict(result.conservation_residuals),
            "solver": dict(result.solver_diagnostic)})
        state = result.final_state
        if result.outcome != "success":
            break
    return {"implementation": "native shared particle-electrode residual; three materials",
        "radial_cells": cells, "segments": segments, "single_material": False,
        "response_order": RESPONSES, "profile_r_over_R": PROFILE_RADII.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL}}


def refusal(data, name):
    from chemworld.physchem.battery_composite import composite_system
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import battery_chemical_energy_j
    from chemworld.physchem.battery_soh import soh_from_battery
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, 8)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s)
    try:
        if name == "duplicate-phase-id":
            second = replace(spec.negative_phases[1], phase_id=spec.negative_phases[0].phase_id)
            replace(spec, negative_phases=(spec.negative_phases[0], second))
        elif name == "wrong-topology":
            replace(spec, negative_phases=(spec.negative_phases[0],))
        elif name == "nonmonotone-ocv":
            original = spec.negative_phases[0]
            phase = replace(original, ocv=replace(original.ocv, polynomial_V=(.1, .1)))
            replace(spec, negative_phases=(phase, spec.negative_phases[1]))
        elif name == "zero-diffusivity":
            replace(spec.negative_phases[0].particle, diffusivity_m2_s=0.)
        elif name == "EIS":
            battery_impedance(spec, (1.,), state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        elif name == "energy":
            battery_chemical_energy_j(spec, state)
        elif name == "thermal":
            composite_system(spec, ScalarControlProgram(((0., .1),), "A"),
                             thermal_device=ThermalDeviceSpec())
        else:
            raise RuntimeError("unknown fixed refusal")
        answer = {"rejected": False}
    except Exception as exc:
        answer = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {**answer, "input_state_unchanged": before == (
        dict(state.amounts_mol), dict(state.coordinates), state.time_s)}


def history_metrics(trace, data):
    rows = np.asarray([row for segment in trace["segments"] for row in segment["rows"]])
    single = trace.get("single_material", False)
    amount = rows[:, 7::5]
    q = rows[:, 4::5]
    initial = np.array([v["active_volume_m3"] * v["maximum_concentration_mol_m3"]
                        * v["initial_fraction"] for v in data["phases"]])
    nnegative = 1 if single else 2
    if single:
        initial = np.array([sum(initial[:2]), initial[2]])
    neg_error = F * (amount[:, :nnegative].sum(axis=1) - sum(initial[:nnegative])) + rows[:, 2]
    pos_error = F * (amount[:, -1] - initial[-1]) - rows[:, 2]
    cc, rest, jumps, theta_jumps, charge_jumps = [], [], [], [], []
    previous = None
    for segment, (_, target, unit) in zip(trace["segments"], data["waveform"], strict=False):
        segment_rows, times = np.asarray(segment["rows"]), np.asarray(segment["times_s"])
        if unit == "A":
            cc.extend(abs(segment_rows[:, 2] - segment_rows[0, 2] - target * (times-times[0])))
            if target == 0:
                rest.extend(abs(segment_rows[:, 2] - segment_rows[0, 2]))
        if previous is not None:
            jumps.extend(abs(segment_rows[0, 7::5] - previous[7::5]))
            theta_jumps.extend(abs(segment_rows[0, 6::5] - previous[6::5]))
            charge_jumps.append(abs(segment_rows[0, 2] - previous[2]))
        previous = segment_rows[-1]
    return {"lithium_balance_mol": float(np.max(abs(amount.sum(axis=1) - sum(initial)))),
        "faradaic_error_C": float(max(np.max(abs(neg_error)), np.max(abs(pos_error)))),
        "phase_current_balance_A": float(max(np.max(abs(q[:, :nnegative].sum(axis=1)-rows[:, 1])),
                                              np.max(abs(q[:, -1]+rows[:, 1])))),
        "CC_charge_error_C": float(max(cc, default=0.)),
        "rest_charge_change_C": float(max(rest, default=0.)),
        "stage_inventory_jump_mol": float(max(jumps, default=0.)),
        "stage_average_jump": float(max(theta_jumps, default=0.)),
        "stage_charge_jump_C": float(max(charge_jumps, default=0.))}
