"""Native adapters for the prospective collector supplement.

The collector adapter records its actual single coupled solve. The uniform
baseline separately advances the original BatterySPMSpec, with no collector
geometry or local-current algebraics. Both retain t0, stopping endpoints,
stage continuity, all real inventories, and the actual solver diagnostics.
No function here runs a matrix, selects successful histories, or changes inputs.
"""

import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields

import numpy as np
from battery_collector_spm_cases import native_spec

from chemworld.physchem.battery_collector_spm import (
    collector_spm_fields,
    collector_spm_system,
)
from chemworld.physchem.battery_models import (
    BatterySPMSpec,
    F,
    ParticleSpec,
    battery_control_current,
    battery_control_mode,
    battery_readout,
    battery_spec_from_dict,
    battery_system,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram

REFUSAL_NAMES = (
    "thermal",
    "electrolyte",
    "SEI",
    "plating",
    "invalid-tab",
    "negative-geometry",
    "EIS",
    "SOH",
)


def _json(value):
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def state_snapshot(state):
    """Every declared CoupledState field, including cumulative receipts."""
    return {item.name: _json(getattr(state, item.name)) for item in fields(state)}


def _snapshot(spec, state):
    return {**state_snapshot(state), "spec": _json(spec.to_dict())}


def _uniform_spec(data, grid):
    particles, fractions = {}, {}
    for name in ("negative", "positive"):
        profile = np.asarray(data["initial_z_fractions"][name], dtype=float)
        if profile.ndim != 1 or not len(profile) or not np.all(profile == profile[0]):
            raise ValueError("uniform SPM baseline requires an explicitly uniform initial profile")
        fractions["initial_" + name + "_fraction"] = float(profile[0])
        particle = data[name]
        particles[name] = ParticleSpec(
            radius_m=particle["radius_m"],
            diffusivity_m2_s=particle["diffusivity_m2_s"],
            maximum_concentration_mol_m3=particle["maximum_concentration_mol_m3"],
            active_volume_m3=particle["active_volume_m3"],
            cells=grid["r"],
            mesh_exponent=1.0,
            exchange_current_scale_A_m2=particle["exchange_current_scale_A_m2"],
        )
    return BatterySPMSpec(
        **particles,
        **fractions,
        negative_ocv_intercept_V=data["negative"]["ocv"][0],
        negative_ocv_slope_V=data["negative"]["ocv"][1],
        positive_ocv_intercept_V=data["positive"]["ocv"][0],
        positive_ocv_slope_V=data["positive"]["ocv"][1],
        temperature_K=data["temperature_K"],
        surface_margin=data["surface_margin"],
        minimum_voltage_V=data["minimum_voltage_V"],
        maximum_voltage_V=data["maximum_voltage_V"],
        maximum_current_A=data["maximum_current_A"],
        series_resistance_ohm=data["series_resistance_ohm"],
        numerical_form="residual",
        provenance="collector-supplement-original-uniform-SPM",
    )


def _uniform_fields(spec, state, program):
    current = battery_control_current(spec, state, program)
    readout = battery_readout(spec, state, current)
    return {
        "current_A": current,
        "voltage_V": readout.voltage_V,
        "surface_fraction": np.array(
            [readout.negative_surface_fraction, readout.positive_surface_fraction]
        ),
        "average_fraction": np.array(
            [readout.negative_average_fraction, readout.positive_average_fraction]
        ),
        "kinetic_overpotential_V": np.array(
            [readout.negative_overpotential_V, readout.positive_overpotential_V]
        ),
        "reaction_current_A": np.array([current, -current]),
        "through_current_A": np.array([current]),
    }


def _bins(spec, collector):
    if collector:
        return spec.particle_bins
    return tuple(
        (name, 0, getattr(spec, name), getattr(spec, name).keys(name), name)
        for name in ("negative", "positive")
    )


def _profiles(spec, states, samples, data, collector):
    profiles, bins_all = {}, _bins(spec, collector)
    nz = spec.collector_cells if collector else 1
    for name in ("negative", "positive"):
        bins = [item for item in bins_all if item[0] == name]
        particle = bins[0][2]
        amounts = np.array(
            [[[s.amounts_mol[key] for key in item[3]] for item in bins] for s in states]
        )
        theta = amounts / (particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3)
        indices = np.arange(nz) + (0 if name == "negative" else nz)
        surfaces = np.array([f["surface_fraction"][indices] for f in samples])
        # Preserve the actual constitutive trial extension explicitly, alongside
        # unmodified surface values. This is a readout, not an oracle or clipping
        # of accepted inventory.
        kinetic_fraction = np.clip(surfaces, spec.surface_margin / 2, 1 - spec.surface_margin / 2)
        q = np.array([f["reaction_current_A"][indices] for f in samples])
        profiles[name] = {
            "electrode": name,
            "inventory_keys": [list(item[3]) for item in bins],
            "z_nodes_m": ((np.arange(nz) + 0.5) * data["height_m"] / nz).tolist(),
            "z_edges_m": np.linspace(0, data["height_m"], nz + 1).tolist(),
            "r_nodes_m": list(particle.mesh.centers_m),
            "r_edges_m": list(particle.mesh.faces_m),
            "shell_volumes_m3": list(particle.mesh.cell_volumes_m3),
            "interface_area_per_z_m2": particle.area_m2,
            "shell_amounts_mol": amounts.tolist(),
            "theta": theta.tolist(),
            "surface_fraction": surfaces.tolist(),
            "average_fraction": np.array(
                [f["average_fraction"][indices] for f in samples]
            ).tolist(),
            "kinetic_overpotential_V": np.array(
                [f["kinetic_overpotential_V"][indices] for f in samples]
            ).tolist(),
            "open_circuit_potential_V": (
                data[name]["ocv"][0] + data[name]["ocv"][1] * surfaces
            ).tolist(),
            "exchange_current_density_A_m2": (
                particle.exchange_current_scale_A_m2
                * np.sqrt(kinetic_fraction * (1 - kinetic_fraction))
            ).tolist(),
            "kinetics_evaluation_fraction": kinetic_fraction.tolist(),
            "reaction_current_A": q.tolist(),
            "reaction_current_density_A_m2": (q / particle.area_m2).tolist(),
            "radial_axis_order": ["time", "z", "r"],
            "local_axis_order": ["time", "z"],
            "reaction_sign": "positive delithiation/outward particle flux",
            "z_interpretation": "actual collector cell"
            if collector
            else "one uniform electrode; no collector state",
        }
        if collector:
            profiles[name]["collector_potential_V"] = np.array(
                [f["collector_potential_" + name + "_V"] for f in samples]
            ).tolist()
    return profiles


def _conservation(spec, initial, states, collector):
    bins = _bins(spec, collector)
    groups = {
        name: [key for item in bins if item[0] == name for key in item[3]]
        for name in ("negative", "positive")
    }
    totals = {
        name: np.array([sum(s.amounts_mol[k] for k in keys) for s in states])
        for name, keys in groups.items()
    }
    initial_totals = {
        name: sum(initial.amounts_mol[k] for k in keys) for name, keys in groups.items()
    }
    dq = np.array(
        [
            s.coordinates["discharged_charge_C"] - initial.coordinates["discharged_charge_C"]
            for s in states
        ]
    )
    result = {
        "negative_total_mol": totals["negative"].tolist(),
        "positive_total_mol": totals["positive"].tolist(),
        "total_lithium_change_mol": (
            totals["negative"] + totals["positive"] - sum(initial_totals.values())
        ).tolist(),
        "negative_faradaic_residual_C": (
            F * (totals["negative"] - initial_totals["negative"]) + dq
        ).tolist(),
        "positive_faradaic_residual_C": (
            F * (totals["positive"] - initial_totals["positive"]) - dq
        ).tolist(),
    }
    if collector:
        local = []
        for z in range(spec.collector_cells):
            keys = (*bins[z][3], *bins[z + spec.collector_cells][3])
            original = sum(initial.amounts_mol[k] for k in keys)
            local.append([sum(s.amounts_mol[k] for k in keys) - original for s in states])
        result["local_lithium_change_mol"] = np.array(local).T.tolist()
    return result


def _record(spec, states, samples, result, system, program, data, initial, collector):
    values = [
        [
            f["voltage_V"],
            f["current_A"],
            s.coordinates["discharged_charge_C"],
            s.coordinates["delivered_work_J"],
        ]
        for s, f in zip(states, samples, strict=True)
    ]
    voltages = [row[0] for row in values]
    record = {
        "times_s": [s.time_s for s in states],
        "end_time_s": result.final_state.time_s,
        "termination": result.stopping_event,
        "outcome": result.outcome,
        "actual_duration_s": result.actual_duration_s,
        "values": values,
        "physical_values": deepcopy(values),
        "physical_tab_voltage_V": voltages,
        "physical_work_J": [row[3] for row in values],
        "controlled_port": "physical-tab" if program.unit == "V" else "current",
        "physical_port_provenance": (
            "actual native residual terminal and original delivered-work coordinate"
        ),
        "physical_metal_tab_voltage_V": [
            row[0] + row[1] * spec.series_resistance_ohm for row in values
        ],
        "raw_profiles": _profiles(spec, states, samples, data, collector),
        "raw_states": [state_snapshot(s) for s in states],
        "raw_amounts_mol": [dict(s.amounts_mol) for s in states],
        "raw_coordinates": [dict(s.coordinates) for s in states],
        "raw_algebraic_values": [dict(s.algebraic_values) for s in states],
        "initial_state": state_snapshot(result.initial_state),
        "final_state": state_snapshot(result.final_state),
        "native_event_values_at_endpoint": {
            event.event_id: float(event.value(result.final_state)) for event in system.all_events
        },
        "event_definitions": [
            {"id": event.event_id, "direction": event.direction, "terminal": event.terminal}
            for event in system.all_events
        ],
        "control_modes": [battery_control_mode(spec, s, program)[0] for s in states],
        "solver": _json(result.solver_diagnostic),
        "native_solver_policy": asdict(system.policy),
        "conservation_residuals": dict(result.conservation_residuals),
        "sample_conservation": _conservation(spec, initial, states, collector),
        "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
        "energy_balance_residual_J": result.energy_balance_residual_J,
        "actual_exchanges": _json(result.actual_exchanges),
        "energy_scope": (
            "raw Q/W coordinates and existing receipts; no new full-cell energy qualification"
        ),
    }
    if collector:
        record["native_fields"] = [
            {key: _json(value) for key, value in sample.items()} for sample in samples
        ]
        record["source_boundary_voltage_V"] = [f["source_boundary_voltage_V"] for f in samples]
        record["independent_physical_tab_voltage_V"] = None
    else:
        record["source_boundary_voltage_V"] = None
        record["independent_physical_tab_voltage_V"] = None
        record["native_fields"] = [
            {key: _json(value) for key, value in sample.items()} for sample in samples
        ]
    return record


def _time(spec, data, grid, layout, collector):
    state = spec.initial_state()
    original = state
    segments = []
    progress = {
        "completed_segments": segments,
        "stage": "build",
        "current_time_s": state.time_s,
        "initial_state": _snapshot(spec, state),
    }
    try:
        for index, (duration, target, unit) in enumerate(data["waveform"]):
            program = ScalarControlProgram(((0.0, target),), unit)
            stage = {
                "index": index,
                "start_time_s": state.time_s,
                "duration_s": duration,
                "target": target,
                "unit": unit,
                "current_limit_A": spec.maximum_current_A,
            }
            progress.update(stage="integrate", active_segment=stage)
            system = (
                collector_spm_system(spec, program) if collector else battery_system(spec, program)
            )
            result = system.advance(
                state,
                duration,
                evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
            )
            states = list(result.trajectory)
            if not states or states[-1].time_s != result.final_state.time_s:
                states.append(result.final_state)
            progress["last_integrated_state"] = state_snapshot(result.final_state)
            # Extraction can fail independently of integration. Keep every
            # computed state and receipt until the complete record is captured.
            progress["pending_solution"] = {
                "raw_states": [state_snapshot(s) for s in states],
                "initial_state": state_snapshot(result.initial_state),
                "final_state": state_snapshot(result.final_state),
                "outcome": result.outcome,
                "stopping_event": result.stopping_event,
                "actual_duration_s": result.actual_duration_s,
                "solver": _json(result.solver_diagnostic),
                "native_solver_policy": asdict(system.policy),
                "conservation_residuals": dict(result.conservation_residuals),
                "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
                "energy_balance_residual_J": result.energy_balance_residual_J,
                "actual_exchanges": _json(result.actual_exchanges),
            }
            progress["stage"] = "capture"
            samples = [
                collector_spm_fields(spec, s, program=program)
                if collector
                else _uniform_fields(spec, s, program)
                for s in states
            ]
            record = _record(
                spec, states, samples, result, system, program, data, original, collector
            )
            record["control"] = stage
            segments.append(record)
            del progress["pending_solution"]
            state = result.final_state
            progress["current_time_s"] = state.time_s
            if result.outcome != "success":
                break
    except Exception as exc:
        progress.update(exception_type=type(exc).__name__, exception=str(exc))
        exc.partial_history = progress
        raise
    return {
        "implementation": "native" if collector else "native-uniform",
        "family": spec.model_family if collector else "original-BatterySPMSpec",
        "unit": data["id"],
        "inputs": deepcopy(data),
        "grid": dict(grid),
        "actual_grid": dict(grid),
        "requested_grid": dict(grid),
        "layout": dict(layout),
        "native_spec": _json(spec.to_dict()),
        "native_spec_class": type(spec).__name__,
        "initial_state": _snapshot(spec, original),
        "final_state": _snapshot(spec, state),
        "segments": segments,
        "faraday_C_mol": F,
        "controlled_port": (
            segments[0]["controlled_port"]
            if segments
            and all(s["controlled_port"] == segments[0]["controlled_port"] for s in segments)
            else "per-segment"
        ),
        "source_boundary_convention": (
            "source-style FV extrapolation diagnostic only; actual terminal defines V/W"
        )
        if collector
        else "uniform SPM has no collector boundary observation",
    }


def native_time(data, grid, layout):
    """Advance the actual collector and record its unchanged physical port."""
    return _time(native_spec(data, grid, layout), data, grid, layout, True)


def native_uniform_time(data, grid):
    """Advance the original uniform SPM; never replace conductivity or local q."""
    spec = _uniform_spec(data, grid)
    result = _time(
        spec,
        data,
        {"z": 1, "r": grid["r"]},
        {"kind": "uniform-no-collector", "negative_tab": None, "positive_tab": None},
        False,
    )
    result["requested_grid"] = dict(grid)
    return result


def run_refusal(name, data, grid, layout=None):
    """Invoke one real codec/analysis path; a successful return is a failed refusal.

    Exceptions are captured with the actual before/after state, declaration and
    exact entry point. Invalid names fail outside the measured call; no adapter
    raise is ever counted as a production rejection.
    """
    if name not in REFUSAL_NAMES:
        raise ValueError("unknown declared supplement refusal")
    layout = {"negative_tab": "left", "positive_tab": "right"} if layout is None else layout
    spec = native_spec(data, grid, layout)
    state = spec.initial_state()
    before = _snapshot(spec, state)
    payload = _json(spec.to_dict())
    stage, entrypoint = "codec", "chemworld.physchem.battery_models.battery_spec_from_dict"
    expected = "collector SPM is fixed-T/fixed-electrolyte without side chemistry"
    if name == "thermal":
        from chemworld.physchem.battery_models import BatteryThermalSpec

        payload["thermal"] = asdict(BatteryThermalSpec())
    elif name == "electrolyte":
        from chemworld.physchem.battery_electrolyte import BatteryElectrolyteSpec

        payload["electrolyte"] = asdict(BatteryElectrolyteSpec())
    elif name == "SEI":
        from chemworld.physchem.battery_degradation import SEISpec

        payload["sei"] = asdict(SEISpec())
    elif name == "plating":
        from chemworld.examples.battery_plating import default_spec

        payload["plating"] = default_spec().plating.to_dict()
        expected = "collector SPM does not support plating"
    elif name == "invalid-tab":
        payload["negative_tab"] = "middle"
        expected = "negative_tab must be left or right"
    elif name == "negative-geometry":
        payload["negative_collector_thickness_m"] = -abs(payload["negative_collector_thickness_m"])
        expected = "collector dimensions and conductivities must be positive"
    elif name == "EIS":
        stage, entrypoint = "analysis", "chemworld.physchem.battery_impedance.battery_impedance"
        expected = "collector SPM impedance is unsupported"
    else:
        stage, entrypoint = "analysis", "chemworld.physchem.battery_soh.soh_from_battery"
        expected = "collector SPM chemical-energy and equilibrium-SOH closure is unsupported"
    diagnostic = {
        "name": name,
        "stage": stage,
        "entrypoint": entrypoint,
        "request": payload,
        "before": before,
        "expected_exception": "ValueError",
        "expected_reason": expected,
        "expected_reason_pattern": re.escape(expected),
        "rejected": False,
        "exception": None,
    }
    try:
        if name == "EIS":
            from chemworld.physchem.battery_impedance import battery_impedance

            battery_impedance(spec, [0.1], state=state)
        elif name == "SOH":
            from chemworld.physchem.battery_soh import soh_from_battery

            soh_from_battery(spec, state)
        else:
            battery_spec_from_dict(payload)
    except Exception as exc:
        diagnostic["rejected"] = True
        diagnostic["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    diagnostic["after"] = _snapshot(spec, state)
    diagnostic["state_unchanged"] = diagnostic["after"] == diagnostic["before"]
    return diagnostic
