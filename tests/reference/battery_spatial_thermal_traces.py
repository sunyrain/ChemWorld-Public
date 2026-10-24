"""Native original-state histories; diagnostics never replace thermal receipts."""

from dataclasses import replace

import numpy as np
from battery_spatial_thermal_cases import (
    ATOL,
    PROFILE_POINTS,
    RTOL,
    F,
    initial_temperatures,
    native_spec,
    thermal_geometry,
)


def native_time(data, grid):
    from chemworld.physchem.battery_spatial_thermal import (
        spatial_thermal_fields,
        spatial_thermal_readout,
        spatial_thermal_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    cells, radial = grid
    spec = native_spec(data, grid)
    geometry = thermal_geometry(data, cells)
    state, segments = spec.initial_state(), []
    for duration, target, unit in data["waveform"]:
        program = ScalarControlProgram(((0.0, target),), unit)
        system = spatial_thermal_system(spec, program)
        system = replace(system, policy=replace(system.policy, rtol=RTOL, atol=ATOL))
        result = system.advance(
            state,
            duration,
            evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
        )
        states = list(result.trajectory)
        if not states or abs(states[-1].time_s - result.final_state.time_s) > 1e-10:
            states.append(result.final_state)
        values, temperatures, profiles = {}, [], []
        for saved in states:
            fields = spatial_thermal_fields(spec, saved, program=program)
            reading = spatial_thermal_readout(spec, saved, fields["current_A"])
            temperature = np.asarray(fields["temperature_K"])
            amounts = {
                name: sum(saved.amounts_mol[k] for k in getattr(spec, name).keys(name))
                for name in ("negative", "positive")
            }
            row = {
                "voltage_V": reading.voltage_V,
                "current_A": reading.current_A,
                "charge_C": saved.coordinates["discharged_charge_C"],
                "work_J": saved.coordinates["delivered_work_J"],
                "negative_surface": reading.negative_surface_fraction,
                "positive_surface": reading.positive_surface_fraction,
                "negative_average": reading.negative_average_fraction,
                "positive_average": reading.positive_average_fraction,
                **{name + "_amount_mol": value for name, value in amounts.items()},
                "negative_collector_K": temperature[0],
                "positive_collector_K": temperature[-1],
                "negative_mean_K": fields["negative_mean_temperature_K"],
                "separator_mean_K": np.mean(temperature[1 + cells : 1 + 2 * cells]),
                "positive_mean_K": fields["positive_mean_temperature_K"],
                "C_mean_K": saved.temperature_K,
                "volume_mean_K": fields["volume_mean_temperature_K"],
                "maximum_K": fields["maximum_temperature_K"],
                "minimum_K": fields["minimum_temperature_K"],
                "sensible_J": geometry["capacities_J_K"] @ (temperature - 298.15),
                "chemical_J": -F * sum(e["ocv_V"] * amounts[e["name"]] for e in data["electrodes"]),
                "loss_W": sum(fields["boundary_heat_loss_W"]),
                "loss_J": saved.heat_loss_J,
                "generated_W": sum(fields["generated_heat_W"]),
                "irreversible_W": sum(fields["generated_heat_W"]),
                "reversible_W": 0.0,
                "mixing_W": 0.0,
                "ohmic_W": 0.0,
                "negative_CC_ohmic_W_m3": 0.0,
                "positive_CC_ohmic_W_m3": 0.0,
            }
            for key, weight in (
                ("C_mean_K", geometry["capacities_J_K"]),
                ("volume_mean_K", geometry["volumes_m3"]),
            ):
                row[key + "_raw_nodes"] = temperature @ weight / weight.sum()
            for key, value in row.items():
                values.setdefault(key, []).append(float(value))
            temperatures.append(temperature.tolist())
            profiles.append(
                [
                    np.interp(
                        PROFILE_POINTS,
                        (np.arange(cells) + 0.5) / cells,
                        temperature[1 + i * cells : 1 + (i + 1) * cells],
                    ).tolist()
                    for i in range(3)
                ]
            )
        segments.append(
            {
                "times_s": [s.time_s for s in states],
                "values": values,
                "temperatures_K": temperatures,
                "sampled_profiles_K": profiles,
                "end_time_s": result.final_state.time_s,
                "termination": result.stopping_event,
                "outcome": result.outcome,
                "conservation_residuals": dict(result.conservation_residuals),
                "solver": dict(result.solver_diagnostic),
            }
        )
        state = result.final_state
        if result.outcome != "success":
            break
    return {
        "implementation": "Native single residual state; finite CC and n/s/p thermal nodes",
        "thermal_cells_per_layer": cells,
        "radial_cells": radial,
        "segments": segments,
        "geometry": {name: value.tolist() for name, value in geometry.items()},
        "profile_local_x_over_L": PROFILE_POINTS.tolist(),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "zero_heat_scope": "Constant OCV, zero entropy, ideal conductors; no mixing heat",
    }


def refusal(data, name):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.battery_soh import soh_from_battery
    from chemworld.physchem.battery_spatial_thermal import spatial_thermal_system
    from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, (6, 40))
    state = spec.initial_state()

    def snapshot():
        return (
            dict(state.amounts_mol),
            dict(state.coordinates),
            state.temperature_K,
            state.time_s,
            state.heat_loss_J,
        )

    before = snapshot()
    try:
        if name == "nonconstant-ocv":
            replace(spec, negative_ocv=replace(spec.negative_ocv, polynomial_V=(0.2, -0.1)))
        elif name == "invalid-layer-cells":
            replace(spec.heat_stack.negative, cells=0)
        elif name == "nonpositive-heat-capacity":
            replace(spec.heat_stack.negative, heat_capacity_J_m3_K=0.0)
        elif name == "negative-cooling":
            replace(spec.heat_stack.positive_collector, face_heat_transfer_W_m2_K=-1.0)
        elif name == "initial-hotspot-outside-domain":
            temperatures = list(initial_temperatures(data, 6))
            temperatures[2] = data["maximum_temperature_K"] + 1.0
            replace(spec, initial_temperatures_K=tuple(temperatures))
        elif name == "external-thermal-device":
            spatial_thermal_system(
                spec, ScalarControlProgram(((0.0, 0.1),), "A"), thermal_device=ThermalDeviceSpec()
            )
        elif name == "EIS":
            battery_impedance(spec, (1.0,), state=state)
        elif name == "SOH":
            soh_from_battery(spec, state)
        else:
            raise RuntimeError("unknown fixed refusal")
        answer = {"rejected": False}
    except Exception as exc:
        answer = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {**answer, "input_state_unchanged": before == snapshot()}


def history_metrics(trace, data):
    values = {
        key: np.asarray(
            [value for segment in trace["segments"] for value in segment["values"][key]]
        )
        for key in trace["segments"][0]["values"]
    }
    initial = {
        e["name"]: e["active_volume_m3"] * e["maximum_concentration_mol_m3"] * e["initial_fraction"]
        for e in data["electrodes"]
    }
    negative = values["negative_amount_mol"] - initial["negative"]
    positive = values["positive_amount_mol"] - initial["positive"]
    capacity = np.asarray(trace["geometry"]["capacities_J_K"])
    initial_sensible = capacity @ (
        initial_temperatures(data, trace["thermal_cells_per_layer"]) - 298.15
    )
    initial_chemical = -F * sum(e["ocv_V"] * initial[e["name"]] for e in data["electrodes"])
    delta_chemical = values["chemical_J"] - initial_chemical
    delta_sensible = values["sensible_J"] - initial_sensible
    balance = delta_chemical + delta_sensible + values["work_J"] + values["loss_J"]
    energy_limit = 0.002 + 1e-4 * (
        abs(delta_chemical) + abs(delta_sensible) + abs(values["work_J"]) + abs(values["loss_J"])
    )
    return {
        "lithium_balance_mol": float(np.max(abs(negative + positive))),
        "negative_faraday_error_C": float(np.max(abs(F * negative + values["charge_C"]))),
        "positive_faraday_error_C": float(np.max(abs(F * positive - values["charge_C"]))),
        "energy_closure_maximum_J": float(np.max(abs(balance))),
        "energy_closure_maximum_tolerance_ratio": float(np.max(abs(balance) / energy_limit)),
        "C_mean_reconstruction_K": float(
            np.max(abs(values["C_mean_K"] - values["C_mean_K_raw_nodes"]))
        ),
        "volume_mean_reconstruction_K": float(
            np.max(abs(values["volume_mean_K"] - values["volume_mean_K_raw_nodes"]))
        ),
        "zero_heat_terms_maximum": max(
            float(np.max(abs(values[k])))
            for k in (
                "reversible_W",
                "mixing_W",
                "ohmic_W",
                "negative_CC_ohmic_W_m3",
                "positive_CC_ohmic_W_m3",
            )
        ),
    }
