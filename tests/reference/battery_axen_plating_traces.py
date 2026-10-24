"""New traces reuse old measurement conventions, never alter the frozen block."""

from dataclasses import replace

import numpy as np
from battery_axen_plating_cases import ATOL, PROFILE_RADII, RESPONSES, RTOL, F, R, native_spec
from battery_partial_plating_traces import history_metrics as history_metrics


def native_time(data, cells, *, baseline=False):
    from chemworld.physchem.battery_models import (
        battery_control_current,
        battery_readout,
        battery_system,
    )
    from chemworld.physchem.coupled_system import assemble_coupled_rates
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, cells, baseline=baseline)
    from battery_axen_plating_cases import active_electrodes

    active = active_electrodes(data, baseline=baseline)
    state, segments = spec.initial_state(), []
    for stage, (duration, target, unit) in enumerate(data["waveform"]):
        try:
            program = ScalarControlProgram(((0.0, target),), unit)
            system = battery_system(spec, program)
            system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
            result = system.advance(
                state,
                duration,
                evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
            )
            states = list(result.trajectory)
            if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
                states.append(result.final_state)
            rows, raw_profiles = [], [[], []]
            hysteresis = {name: {"h": [], "h-rate": []} for name in active}
            for saved in states:
                current = battery_control_current(spec, saved, program)
                reading = battery_readout(spec, saved, current)
                side = getattr(reading, "stripping_current_A", 0.0)
                plated, dead = (
                    saved.amounts_mol.get("plating:Li", 0.0),
                    saved.amounts_mol.get("dead:Li", 0.0),
                )
                un = spec.operational_ocp(
                    "negative", reading.negative_surface_fraction, current, saved
                )
                phi = un + reading.negative_overpotential_V
                row = [
                    reading.voltage_V,
                    current,
                    saved.coordinates["discharged_charge_C"],
                    saved.coordinates["delivered_work_J"],
                    side,
                    plated,
                    dead,
                    phi,
                    plated / data["negative_electrode_volume_m3"],
                ]
                for i, e in enumerate(data["electrodes"]):
                    name, particle = e["name"], getattr(spec, e["name"])
                    surface = getattr(reading, name + "_surface_fraction")
                    average = getattr(reading, name + "_average_fraction")
                    amounts = np.array([saved.amounts_mol[k] for k in particle.keys(name)])
                    row.extend(
                        [
                            current - side if i == 0 else -current,
                            surface,
                            average,
                            float(sum(amounts)),
                            spec.operational_ocp(name, surface, current, saved),
                            getattr(reading, name + "_overpotential_V"),
                        ]
                    )
                    raw_profiles[i].append(
                        (
                            amounts
                            / (
                                particle.mesh.cell_volumes_m3
                                * particle.maximum_concentration_mol_m3
                            )
                        ).tolist()
                    )
                rows.append(row)
                rates = assemble_coupled_rates(
                    saved,
                    processes=system.processes,
                    thermodynamics=system.thermodynamics,
                    heat_flux=system.heat_flux,
                    isothermal=True,
                )
                for name in active:
                    key = f"battery:{name}:hysteresis"
                    hysteresis[name]["h"].append(saved.coordinates[key])
                    hysteresis[name]["h-rate"].append(rates.coordinate_rates[key])
            nodes = [
                (getattr(spec, e["name"]).mesh.centers_m / e["radius_m"]).tolist()
                for e in data["electrodes"]
            ]
            profiles = [
                [
                    np.interp(PROFILE_RADII, nodes[i], profile).tolist()
                    for profile in raw_profiles[i]
                ]
                for i in range(2)
            ]
            segments.append(
                {
                    "times_s": [s.time_s for s in states],
                    "rows": rows,
                    "hysteresis": hysteresis,
                    "raw_coordinates": [dict(s.coordinates) for s in states],
                    "start_time_s": states[0].time_s,
                    "end_time_s": result.final_state.time_s,
                    "termination": result.stopping_event,
                    "outcome": result.outcome,
                    "profiles_theta": profiles,
                    "raw_r_over_R": nodes,
                    "raw_profiles_theta": raw_profiles,
                    "raw_amounts_mol": [dict(s.amounts_mol) for s in states],
                    "solver": dict(result.solver_diagnostic),
                    "conservation_residuals": dict(result.conservation_residuals),
                    "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
                }
            )
            state = result.final_state
            if result.outcome != "success":
                break
        except Exception as exc:
            exc.partial_history = {
                "segments": segments,
                "stage": stage,
                "target": target,
                "duration_s": duration,
                "current_time_s": state.time_s,
            }
            raise
    return {
        "implementation": "native SPM Axen/plating",
        "radial_cells": cells,
        "baseline": baseline,
        "has_plating": spec.plating is not None,
        "h_electrodes": active,
        "container_volume_L": spec.initial_state().volume_L,
        "segments": segments,
        "response_order": RESPONSES,
        "profile_r_over_R": PROFILE_RADII.tolist(),
        "F_C_mol": F,
        "R_J_mol_K": R,
        "tolerances": {"rtol": RTOL, "atol": ATOL},
    }


def refusal(data, name):
    from copy import deepcopy
    from dataclasses import fields

    from battery_thevenin_cases import table

    from chemworld.physchem.battery_degradation import SEISpec
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_models import (
        BatterySPMSpec,
        BatteryThermalSpec,
        battery_chemical_energy_j,
    )
    from chemworld.physchem.battery_mpm import BatteryMPMSpec
    from chemworld.physchem.battery_ocp import ElectrodeOCPSpec
    from chemworld.physchem.battery_particles import ParticleSizeDistribution
    from chemworld.physchem.battery_soh import soh_from_battery

    spec = native_spec(data, 8)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), state.time_s, spec.to_dict())
    altered = deepcopy(data)
    try:
        if name in ("initial-h-domain", "negative-decay", "negative-plated"):
            key, value = {
                "initial-h-domain": ("negative_initial_h", 1.1),
                "negative-decay": ("lithiation_decay", -20.0),
                "negative-plated": ("initial_plated_mol", -1e-6),
            }[name]
            altered[key] = value
            native_spec(altered, 8)
        elif name in ("missing-decay", "wrong-decay-unit"):
            ocp = spec.negative_ocp
            params, row = ocp.parameters, ocp.parameters.rows[0]
            values = dict(row.values)
            if name == "missing-decay":
                values.pop("lithiation_decay")
                columns = tuple(c for c in params.columns if c.quantity_id != "lithiation_decay")
            else:
                values["lithiation_decay"] = replace(values["lithiation_decay"], unit="m")
                columns = tuple(
                    replace(c, unit="m") if c.quantity_id == "lithiation_decay" else c
                    for c in params.columns
                )
            params = replace(params, columns=columns, rows=(replace(row, values=values),))
            replace(ocp, parameters=params)
        elif name == "MPM":
            values = {f.name: getattr(spec, f.name) for f in fields(BatterySPMSpec) if f.init}
            for electrode in ("negative", "positive"):
                values[electrode + "_ocv"] = replace(
                    values[electrode + "_ocv"], supported_models=("SPM", "MPM")
                )
                ocp = values[electrode + "_ocp"]
                if ocp is not None and ocp.mode != "single":
                    values[electrode + "_ocp"] = replace(
                        ocp,
                        lithiation=replace(ocp.lithiation, supported_models=("SPM", "MPM")),
                        delithiation=replace(ocp.delithiation, supported_models=("SPM", "MPM")),
                    )
                values[electrode + "_distribution"] = ParticleSizeDistribution.from_weights(
                    (8e-6, 12e-6),
                    (0.5, 0.5),
                    basis="volume",
                    source=spec.plating.parameters.sources[0],
                )
            BatteryMPMSpec(**values)
        elif name == "sigmoid":
            ocp = spec.negative_ocp
            params = table(
                "sigmoid-refusal",
                {"collector_area_m2": "m^2", "sigmoid_slope_m2_A": "m^2/A"},
                [{"collector_area_m2": 0.02, "sigmoid_slope_m2_A": 100.0}],
            )
            replace(
                spec,
                negative_ocp=ElectrodeOCPSpec(
                    mode="current_sigmoid",
                    lithiation=ocp.lithiation,
                    delithiation=ocp.delithiation,
                    parameters=params,
                    parameter_row_id="0",
                ),
            )
        elif name == "partial":
            from battery_partial_plating_cases import native_spec as partial_spec

            partial = partial_spec({**data, "gamma_s_inv": 0.002}, 8)
            replace(spec, plating=partial.plating)
        elif name == "irreversible":
            replace(spec, plating=replace(spec.plating, mode="irreversible"))
        elif name == "SEI":
            replace(spec, sei=SEISpec())
        elif name == "thermal":
            replace(spec, thermal=BatteryThermalSpec())
        elif name == "EIS":
            battery_impedance(spec, (1.0,), state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        elif name == "chemical-energy":
            battery_chemical_energy_j(spec, state)
        else:
            raise RuntimeError("unknown fixed refusal: " + name)
        result = {"rejected": False}
    except (ValueError, NotImplementedError) as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {
        **result,
        "input_state_unchanged": before
        == (dict(state.amounts_mol), dict(state.coordinates), state.time_s, spec.to_dict()),
    }
