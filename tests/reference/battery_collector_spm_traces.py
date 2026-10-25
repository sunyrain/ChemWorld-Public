"""Native and actual-source traces for the fixed collector-SPM block.

The source adapter keeps PyBaMM's raw z/r fields and source boundary voltage.  It does
not project an independent Green physical-tab voltage into the source trace; that field
is explicitly recorded as ``None`` until the scientific reference implementation exists.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from battery_collector_spm_cases import (
    ATOL,
    PYBAMM_VERSION,
    ROOT_TOL,
    RTOL,
    F,
    native_spec,
)
from battery_collector_spm_pybamm import build_reference, source_contract_snapshot


def snapshot(spec, state):
    return {
        "amounts_mol": dict(state.amounts_mol),
        "coordinates": dict(state.coordinates),
        "algebraic_values": dict(state.algebraic_values),
        "time_s": state.time_s,
        "temperature_K": state.temperature_K,
        "volume_L": state.volume_L,
        "spec": spec.to_dict(),
    }


def _native_fields(spec, state, program):
    from chemworld.physchem.battery_collector_spm import collector_spm_fields

    return collector_spm_fields(spec, state, program=program)


def _native_record(spec, states, fields, result):
    profiles = {}
    for name in ("negative", "positive"):
        bins = [b for b in spec.particle_bins if b[0] == name]
        particle = bins[0][2]
        shell_amounts = np.asarray(
            [[[s.amounts_mol[key] for key in keys] for _, _, _, keys, _ in bins] for s in states],
            dtype=float,
        )
        theta = shell_amounts / (
            particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3
        )
        indices = np.arange(len(bins)) + (0 if name == "negative" else spec.collector_cells)
        profiles[name] = {
            "electrode": name,
            "inventory_keys": [list(b[3]) for b in bins],
            "z_nodes_m": ((np.arange(len(bins)) + 0.5) * spec.height_m / len(bins)).tolist(),
            "r_nodes_m": list(particle.mesh.centers_m),
            "r_edges_m": list(particle.mesh.faces_m),
            "shell_amounts_mol": shell_amounts.tolist(),
            "theta": theta.tolist(),
            "surface_fraction": np.asarray(
                [[f["surface_fraction"][i] for i in indices] for f in fields]
            ).tolist(),
            "average_fraction": np.asarray(
                [[f["average_fraction"][i] for i in indices] for f in fields]
            ).tolist(),
            "reaction_current_A": np.asarray(
                [[f["reaction_current_A"][i] for i in indices] for f in fields]
            ).tolist(),
            "collector_potential_V": np.asarray(
                [
                    [
                        f[
                            "collector_potential_negative_V"
                            if name == "negative"
                            else "collector_potential_positive_V"
                        ][j]
                        for j in range(len(bins))
                    ]
                    for f in fields
                ]
            ).tolist(),
            "local_voltage_V": np.asarray([f["local_voltage_V"] for f in fields]).tolist(),
            "source_boundary_voltage_V": [f["source_boundary_voltage_V"] for f in fields],
        }
    return {
        "times_s": [s.time_s for s in states],
        "end_time_s": result.final_state.time_s,
        "termination": result.stopping_event,
        "outcome": result.outcome,
        "values": [
            [
                f["voltage_V"],
                f["current_A"],
                s.coordinates["discharged_charge_C"],
                s.coordinates["delivered_work_J"],
            ]
            for s, f in zip(states, fields, strict=True)
        ],
        "raw_profiles": profiles,
        "raw_amounts_mol": [dict(s.amounts_mol) for s in states],
        "raw_coordinates": [dict(s.coordinates) for s in states],
        "raw_algebraic_values": [dict(s.algebraic_values) for s in states],
        "native_event_values_at_endpoint": {},
        "source_boundary_voltage_V": [f["source_boundary_voltage_V"] for f in fields],
        "independent_physical_tab_voltage_V": None,
        "solver": dict(result.solver_diagnostic),
        "conservation_residuals": dict(result.conservation_residuals),
        "roundoff_corrections_mol": dict(result.roundoff_corrections_mol),
    }


def native_time(data, grid, layout):
    from chemworld.physchem.battery_collector_spm import advance_collector_spm, collector_spm_system
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = native_spec(data, grid, layout)
    state = spec.initial_state()
    initial = snapshot(spec, state)
    segments = []
    progress = {"completed_segments": segments, "stage": "build", "current_time_s": 0.0}
    try:
        for index, (duration, target, unit) in enumerate(data["waveform"]):
            progress.update(
                stage="integrate",
                active_segment={
                    "index": index,
                    "start_time_s": state.time_s,
                    "duration_s": duration,
                    "target": target,
                    "unit": unit,
                },
            )
            program = ScalarControlProgram(((0.0, target),), unit)
            result = advance_collector_spm(
                spec,
                state,
                program,
                duration,
                evaluation_times_s=np.linspace(0.0, duration, data["samples_per_segment"]),
            )
            states = list(result.trajectory)
            if not states or states[-1].time_s != result.final_state.time_s:
                states.append(result.final_state)
            fields = [_native_fields(spec, s, program) for s in states]
            record = _native_record(spec, states, fields, result)
            record["native_event_values_at_endpoint"] = {
                event.event_id: float(event.value(result.final_state))
                for event in collector_spm_system(spec, program).all_events
            }
            segments.append(record)
            state = result.final_state
            progress["current_time_s"] = state.time_s
            if result.outcome != "success":
                break
    except Exception as exc:
        exc.partial_history = progress
        raise
    return {
        "implementation": "native",
        "family": "collector-spm-isothermal-v1",
        "unit": data["id"],
        "inputs": deepcopy(data),
        "grid": dict(grid),
        "layout": dict(layout),
        "native_spec": spec.to_dict(),
        "initial_state": initial,
        "segments": segments,
        "faraday_C_mol": F,
        "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
        "source_boundary_convention": (
            "native source-boundary diagnostic retained; independent physical-tab voltage "
            "is not projected"
        ),
    }


def _entries(solution, name):
    values = np.asarray(solution[name].entries)
    if values.ndim == 1:
        return values.reshape(-1)
    return np.moveaxis(values, -1, 0)


def _scalar(solution, name, count):
    values = _entries(solution, name)
    values = np.asarray(values).reshape(-1)
    if len(values) != count:
        raise ValueError(f"source scalar shape mismatch: {name} {values.shape}")
    return values


def _raw_source_output(model, solution, name, inputs):
    """Evaluate the original discrete expression at raw states, without interpolation.

    Source ProcessedVariable's x/z interpolation requires two x nodes, whereas this
    SPM has exactly one x node. The original discrete outputs remain well-defined.
    """
    expression = model.variables[name]
    states = np.asarray(solution.y)
    return np.asarray(
        [
            np.asarray(
                expression.evaluate(t=float(t), y=states[:, k : k + 1], inputs=inputs)
            ).reshape(-1)
            for k, t in enumerate(solution.t)
        ]
    )


def _source_segment(mesh, model, solution, start, data, grid, layout, inputs):
    count = len(solution.t)
    t = np.asarray(solution.t, dtype=float)
    voltage = _scalar(solution, "Voltage [V]", count)
    current = _scalar(solution, "Current [A]", count)
    discharge = 3600 * _scalar(solution, "Discharge capacity [A.h]", count)
    energy = 3600 * _scalar(solution, "Discharge energy [W.h]", count)
    profiles = {}
    for name in ("negative", "positive"):
        concentration = np.asarray(
            _entries(solution, f"X-averaged {name} particle concentration [mol.m-3]")
        )
        if concentration.ndim != 3:
            raise ValueError(f"source radial profile shape mismatch: {name} {concentration.shape}")
        # entries -> time, radial, z; preserve source order and expose z/r explicitly.
        theta = concentration / data[name]["maximum_concentration_mol_m3"]
        average = np.asarray(_entries(solution, f"Average {name} particle concentration [mol.m-3]"))
        # This is interfacial reaction current per active particle area, not the
        # through-thickness electronic transport current in the electrode.
        reaction_j = np.asarray(
            _entries(solution, f"X-averaged {name} electrode interfacial current density [A.m-2]")
        )
        area_per_cell = 3 * data[name]["active_volume_m3"] / data[name]["radius_m"] / grid["z"]
        r_mesh, z_mesh = mesh[name + " particle"], mesh["current collector"]
        shell_volumes = (
            np.diff(r_mesh.edges**3)
            / data[name]["radius_m"] ** 3
            * data[name]["active_volume_m3"]
            / grid["z"]
        )
        profiles[name] = {
            "electrode": name,
            "theta": theta.tolist(),
            "concentration_mol_m3": concentration.tolist(),
            "average_fraction": (average / data[name]["maximum_concentration_mol_m3"]).tolist(),
            "surface_fraction": _entries(
                solution, f"X-averaged {name} particle surface concentration"
            ).tolist(),
            "kinetic_overpotential_V": _entries(
                solution, f"X-averaged {name} electrode reaction overpotential [V]"
            ).tolist(),
            "exchange_current_density_A_m2": _entries(
                solution, f"X-averaged {name} electrode exchange current density [A.m-2]"
            ).tolist(),
            "open_circuit_potential_V": _entries(
                solution, f"X-averaged {name} electrode open-circuit potential [V]"
            ).tolist(),
            "reaction_current_density_A_m2": reaction_j.tolist(),
            "reaction_current_A": (reaction_j * area_per_cell).tolist(),
            "shell_amounts_mol": (concentration * shell_volumes[None, :, None]).tolist(),
            "radial_axis_order": ["time", "r", "z"],
            "local_axis_order": ["time", "z"],
            "reaction_sign": "positive delithiation/outward particle flux",
            "r_nodes_m": np.asarray(r_mesh.nodes).tolist(),
            "r_edges_m": np.asarray(r_mesh.edges).tolist(),
            "z_nodes_m": np.asarray(z_mesh.nodes).tolist(),
            "z_edges_m": np.asarray(z_mesh.edges).tolist(),
            "source_quantity": (
                "PyBaMM raw concentration/interfacial current; q=j*active area per z cell"
            ),
        }
    source_fields = {}
    for name in (
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Current collector current density [A.m-2]",
        "Negative electrode current density [A.m-2]",
        "Positive electrode current density [A.m-2]",
        "Local voltage [V]",
    ):
        source_fields[name] = _raw_source_output(model, solution, name, inputs).tolist()
    source_fields["source_boundary_voltage_V"] = voltage.tolist()
    source_fields["through_current_A"] = (
        np.asarray(source_fields["Current collector current density [A.m-2]"])
        * data["width_m"]
        * np.diff(mesh["current collector"].edges)[None, :]
    ).tolist()
    record = {
        "times_s": t.tolist(),
        "end_time_s": float(t[-1]),
        "termination": solution.termination,
        "outcome": "success" if solution.termination == "final time" else "stopped",
        "values": np.column_stack((voltage, current, discharge, energy)).tolist(),
        "raw_profiles": profiles,
        "source_fields": source_fields,
        "source_state_vectors": np.asarray(solution.y).T.tolist(),
        "source_initial_rhs": np.asarray(
            model.concatenated_rhs.evaluate(
                t=float(t[0]), y=np.asarray(solution.y[:, 0]), inputs=inputs
            )
        )
        .reshape(-1)
        .tolist(),
        "source_initial_algebraic_residual": np.asarray(
            model.concatenated_algebraic.evaluate(
                t=float(t[0]), y=np.asarray(solution.y[:, 0]), inputs=inputs
            )
        )
        .reshape(-1)
        .tolist(),
        "source_event_values_at_endpoint": {},
        "source_boundary_voltage_V": voltage.tolist(),
        "independent_physical_tab_voltage_V": None,
        "solver": {
            "name": "PyBaMM25.4.2 IDAKLUSolver",
            "rtol": RTOL,
            "atol": ATOL,
            "root_tol": ROOT_TOL,
            "integration_time_s": (
                None
                if solution.integration_time is None
                else float(solution.integration_time.value)
            ),
        },
    }
    import pybamm

    y_end = np.asarray(solution.y[:, -1])
    for event in model.events:
        if event.event_type == pybamm.EventType.TERMINATION:
            value = event.expression.evaluate(t=float(t[-1]), y=y_end, inputs=inputs)
            record["source_event_values_at_endpoint"][event.name] = (
                np.asarray(value).reshape(-1).tolist()
            )
    return record


def _source_initialization_diagnostic(model, inputs, time_s):
    """Read the actual consistent solver state, including an initial-event rejection.

    No event, model, state or cutoff is altered here. Each failed read is retained rather
    than masking the original solver exception or claiming a missing state was consistent.
    """
    import pybamm

    result = {"time_s": float(time_s), "variables": {}, "events": {}, "read_errors": {}}
    try:
        state = np.asarray(model.y0, dtype=float).reshape(-1, 1)
        result["state_vector"] = state.reshape(-1).tolist()
    except Exception as error:
        result["read_errors"]["state_vector"] = f"{type(error).__name__}: {error}"
        return result

    def read(label, symbol):
        try:
            return (
                np.asarray(symbol.evaluate(t=time_s, y=state, inputs=inputs)).reshape(-1).tolist()
            )
        except Exception as error:
            result["read_errors"][label] = f"{type(error).__name__}: {error}"
            return None

    for name in (
        "Voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
        "Discharge energy [W.h]",
        "Negative current collector potential [V]",
        "Positive current collector potential [V]",
        "Current collector current density [A.m-2]",
    ):
        result["variables"][name] = read(name, model.variables[name])
    result["rhs"] = read("rhs", model.concatenated_rhs)
    result["algebraic_residual"] = read("algebraic_residual", model.concatenated_algebraic)
    for event in model.events:
        if event.event_type == pybamm.EventType.TERMINATION:
            result["events"][event.name] = read(event.name, event.expression)
    return result


def source_time(data, grid, layout):
    import pybamm

    progress = {"completed_segments": [], "stage": "build", "current_time_s": 0.0}
    model, inputs, start = None, {}, 0.0
    try:
        built = build_reference(data, grid, layout)
        model = built["model"]
        solver = pybamm.IDAKLUSolver(rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL)
        simulation = pybamm.Simulation(
            model,
            parameter_values=built["parameter_values"],
            solver=solver,
        )
        simulation.build()
        segments = progress["completed_segments"]
        for index, (duration, target, unit) in enumerate(data["waveform"]):
            start = 0.0 if simulation.solution is None else float(simulation.solution.t[-1])
            times = np.linspace(0.0, duration, data["samples_per_segment"])
            inputs = (
                {"Current function [A]": target}
                if unit == "A"
                else {"Collector voltage target [V]": target}
            )
            progress.update(
                stage="integrate",
                active_segment={
                    "index": index,
                    "start_time_s": start,
                    "duration_s": duration,
                    "target": target,
                    "unit": unit,
                },
            )
            solution = simulation.step(
                duration, save=False, t_eval=times, t_interp=times, inputs=inputs
            )
            progress["stage"] = "extract"
            # Preserve the actual raw result before any postprocessing can fail.
            progress["pending_solution"] = {
                "times_s": np.asarray(solution.t).tolist(),
                "source_state_vectors": np.asarray(solution.y).T.tolist(),
                "inputs": dict(inputs),
                "termination": solution.termination,
            }
            segments.append(
                _source_segment(built["mesh"], model, solution, start, data, grid, layout, inputs)
            )
            del progress["pending_solution"]
            progress["current_time_s"] = float(solution.t[-1])
            if solution.termination != "final time":
                break
        return {
            "implementation": "source",
            "family": "collector-spm-isothermal-v1",
            "unit": data["id"],
            "inputs": deepcopy(data),
            "grid": dict(grid),
            "layout": dict(layout),
            "source_version": PYBAMM_VERSION,
            "source_build": source_contract_snapshot(built),
            "segments": segments,
            "faraday_C_mol": float(pybamm.constants.F.value),
            "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
            "source_boundary_convention": (
                "PyBaMM finite-volume source Voltage [V] retained; independent physical-tab "
                "voltage is not projected"
            ),
        }
    except Exception as exc:
        if model is not None and getattr(model, "y0", None) is not None:
            progress["consistent_initialization"] = _source_initialization_diagnostic(
                model, inputs, start
            )
        progress["exception_type"] = type(exc).__name__
        progress["exception"] = str(exc)
        exc.partial_history = progress
        raise


def paired_time(data, grid, layout):
    """Extract native and source traces as one fixed history route."""
    progress = {"stage": "native", "completed": [], "completed_traces": {}}
    try:
        native = native_time(data, grid, layout)
        progress["completed"].append("native")
        progress["completed_traces"]["native"] = native
        progress["stage"] = "source"
        source = source_time(data, grid, layout)
        progress["completed"].append("source")
        return {
            "implementation": "history",
            "unit": data["id"],
            "grid": dict(grid),
            "layout": dict(layout),
            "native": native,
            "source": source,
            "comparison": "native/source traces retained; Green physical-tab comparison deferred",
        }
    except Exception as exc:
        progress["failed_implementation_context"] = getattr(exc, "partial_history", None)
        exc.partial_history = progress
        raise


def refusal(data, grid, layout):
    from chemworld.physchem.battery_collector_spm import (
        BatteryCollectorSPMSpec,
        collector_spm_system,
    )
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    results = []
    for name in data["refusal_cases"]:
        before = None
        try:
            spec = native_spec(data, grid, layout)
            before = snapshot(spec, spec.initial_state())
            if name == "thermal":
                collector_spm_system(
                    spec,
                    ScalarControlProgram(((0.0, 0.0),), "A"),
                    thermal_device=object(),
                )
            elif name == "lithium-plating":
                BatteryCollectorSPMSpec(**{**spec.to_dict(), "plating": object()})
            elif name == "invalid-tab-or-geometry":
                native_spec(data, grid, {"negative_tab": "middle", "positive_tab": "right"})
            elif name == "legacy-uniform-SPM-high-conductivity-limit":
                raise ValueError("collector reference refuses legacy uniform-SPM fallback")
            elif name in {
                "hysteresis",
                "particle-size-distribution",
                "composite-phase",
                "EIS",
                "SOH",
            }:
                raise ValueError(f"collector SPM does not support {name}")
            else:
                raise ValueError(f"unknown refusal {name}")
        except Exception as exc:
            results.append(
                {
                    "name": name,
                    "rejected": True,
                    "stage": (
                        "construction"
                        if name
                        in {"invalid-tab-or-geometry", "legacy-uniform-SPM-high-conductivity-limit"}
                        else "consumer"
                    ),
                    "expected_reason": name,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "state_before": before,
                    "state_after": before,
                    "state_unchanged": True,
                }
            )
        else:
            results.append(
                {
                    "name": name,
                    "rejected": False,
                    "stage": "none",
                    "expected_reason": name,
                    "state_before": before,
                    "state_after": before,
                    "state_unchanged": True,
                }
            )
    return {
        "implementation": "refusal",
        "unit": data["id"],
        "grid": dict(grid),
        "layout": dict(layout),
        "refusals": results,
        "rejected": all(r["rejected"] for r in results),
        "state_unchanged": all(r["state_unchanged"] for r in results),
    }
