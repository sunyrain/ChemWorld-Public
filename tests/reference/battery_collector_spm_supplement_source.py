"""Actual PyBaMM collector supplement with an explicitly authored physical port.

The released Voltage, signed discharge energy and all original events survive.
The added port uses the last positive FV centre and its Neumann half-cell drop;
it is a declared observation/control boundary, not an independent Green solution.
The extra signed physical-work state has RHS I*Vphysical and no feedback.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from battery_collector_spm_cases import ATOL, PYBAMM_VERSION, ROOT_TOL, RTOL, _z_fractions
from battery_collector_spm_pybamm import _model_options, _parameter_updates
from battery_collector_spm_traces import (
    _entries,
    _raw_source_output,
    _scalar,
    _source_initialization_diagnostic,
    _source_segment,
)

PHYSICAL_V = "Supplement physical tab voltage [V]"
PHYSICAL_W = "Supplement signed physical work [J]"
TARGET_V = "Supplement physical voltage target [V]"
EVENT_PREFIX = "Supplement physical qualification: "


def _physical_voltage(variables, data, grid, uniform):
    import pybamm

    if uniform:
        return variables["Voltage [V]"]
    conductance = data["sigma_cc_S_m"] * data["collector_thickness_m"] * data["width_m"]
    return pybamm.Index(variables["Positive current collector potential [V]"], -1) - variables[
        "Current [A]"
    ] * data["height_m"] / (2 * grid["z"] * conductance)


def _validate(data, grid, layout, uniform):
    if data["series_resistance_ohm"] != 0:
        raise ValueError("supplement source declares zero series resistance only")
    if type(grid["r"]) is not int or grid["r"] < 2:
        raise ValueError("at least two radial cells are required")
    if not uniform and (type(grid["z"]) is not int or grid["z"] < 2):
        raise ValueError("collector requires at least two z cells")
    if layout not in (
        {"negative_tab": "left", "positive_tab": "right"},
        {"negative_tab": "right", "positive_tab": "right"},
    ):
        raise ValueError("only positive-right and negative-left/right tabs are supported")
    modes = {stage[2] for stage in data["waveform"]}
    if modes not in ({"A"}, {"V"}):
        raise ValueError("one control mode per history; CC/rest/reverse are all A")
    if uniform:
        for electrode in ("negative", "positive"):
            profile = np.asarray(data["initial_z_fractions"][electrode], dtype=float)
            if not len(profile) or not np.all(profile == profile[0]):
                raise ValueError("uniform SPM baseline cannot average a nonuniform SOC profile")


def build_reference(data, grid, layout, *, uniform=False):
    """Build the actual 25.4.2 modular SPM; no numerical integration."""
    import pybamm

    if pybamm.__version__ != PYBAMM_VERSION:
        raise RuntimeError(f"supplement requires PyBaMM {PYBAMM_VERSION}")
    _validate(data, grid, layout, uniform)
    original_options = _model_options(data)

    def finite_physical_source(variables):
        current = variables["Current [A]"]
        difference = _physical_voltage(variables, data, grid, uniform) - pybamm.InputParameter(
            TARGET_V
        )
        # Projected complementarity: the interior branch is exact V=target,
        # not a finite-gain current controller. The scale has units A/V.
        requested = current + data["maximum_current_A"] * difference
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], requested)
        )

    options = dict(original_options)
    options["operating mode"] = (
        finite_physical_source if data["waveform"][0][2] == "V" else "current"
    )
    if uniform:
        options.update({"dimensionality": 0, "current collector": "uniform"})
    model = pybamm.lithium_ion.SPM(options=options)
    original_events = tuple(event.name for event in model.events)
    original_rhs_names = tuple(variable.name for variable in model.rhs)
    original_voltage = model.variables["Voltage [V]"]
    original_energy = model.variables["Discharge energy [W.h]"]
    physical = _physical_voltage(model.variables, data, grid, uniform)
    work = pybamm.Variable(PHYSICAL_W)
    model.rhs[work] = model.variables["Current [A]"] * physical
    model.initial_conditions[work] = pybamm.Scalar(0)
    model.variables.update({PHYSICAL_V: physical, PHYSICAL_W: work})
    model.events.extend(
        [
            pybamm.Event(EVENT_PREFIX + "minimum voltage", physical - data["minimum_voltage_V"]),
            pybamm.Event(EVENT_PREFIX + "maximum voltage", data["maximum_voltage_V"] - physical),
        ]
    )
    for name in ("negative", "positive"):
        surface = model.variables[f"X-averaged {name} particle surface concentration"]
        model.events.extend(
            [
                pybamm.Event(
                    EVENT_PREFIX + name + " minimum surface",
                    pybamm.min(surface) - data["surface_margin"],
                ),
                pybamm.Event(
                    EVENT_PREFIX + name + " maximum surface",
                    1 - data["surface_margin"] - pybamm.max(surface),
                ),
            ]
        )
    if not uniform:
        centres = (np.arange(grid["z"]) + 0.5) * data["height_m"] / grid["z"]
        for name in ("negative", "positive"):
            fractions = _z_fractions(data["initial_z_fractions"][name], grid["z"])
            profile = pybamm.Interpolant(
                np.r_[0.0, centres, data["height_m"]],
                data[name]["maximum_concentration_mol_m3"]
                * np.r_[fractions[0], fractions, fractions[-1]],
                pybamm.standard_spatial_vars.z,
                interpolator="linear",
            )
            variable = model.variables[f"X-averaged {name} particle concentration [mol.m-3]"]
            model.initial_conditions[variable] = pybamm.PrimaryBroadcast(
                profile, f"{name} particle"
            )
    assert model.variables["Voltage [V]"] is original_voltage
    assert model.variables["Discharge energy [W.h]"] is original_energy
    values = pybamm.ParameterValues("Marquis2019")
    values.update(_parameter_updates(data, layout, grid), check_already_exists=False)
    geometry = pybamm.battery_geometry(options=options)
    values.process_geometry(geometry)
    var_pts = {"x_n": 1, "x_s": 1, "x_p": 1, "r_n": grid["r"], "r_p": grid["r"], "z": grid["z"]}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    values.process_model(model)
    disc.process_model(model)
    # Parameter/discretisation processing can add original interpolant-domain events.
    # Keep those too; the new labelled qualification events are the only additions.
    processed_original_events = tuple(
        event.name for event in model.events if not event.name.startswith(EVENT_PREFIX)
    )
    if not set(original_events) <= set(processed_original_events):
        raise ValueError("an original source event disappeared during processing")
    return {
        "model": model,
        "parameter_values": values,
        "mesh": mesh,
        "grid": {"z": 1 if uniform else grid["z"], "r": grid["r"]},
        "authored_grid": dict(grid),
        "layout": dict(layout),
        "uniform": uniform,
        "original_events": processed_original_events,
        "original_rhs_names": original_rhs_names,
        "z_nodes_m": [data["height_m"] / 2]
        if uniform
        else mesh["current collector"].nodes.tolist(),
        "control": "physical-CV-with-finite-current-limits"
        if data["waveform"][0][2] == "V"
        else "CC",
    }


def _physical_work_dependency_check(model):
    """Inspect actual processed state-vector reads, not a declared boolean flag."""
    import pybamm

    variable = next(v for v in model.rhs if v.name == PHYSICAL_W)
    parts = model.y_slices[variable]
    indices = [i for part in parts for i in range(part.start, part.stop)]
    checked = {}
    expressions = [("rhs:" + v.name, expression) for v, expression in model.rhs.items()]
    expressions += [
        ("algebraic:" + v.name, expression) for v, expression in model.algebraic.items()
    ]
    event_identities = [
        {"ordinal": i, "type": e.event_type.name, "name": e.name}
        for i, e in enumerate(model.events)
    ]
    # Upstream Interpolants can add multiple events with the same name.
    # Preserve each expression at its actual model position instead of
    # overwriting its proof with a later equal-name event.
    expressions += [
        (f"event:{i}:{e.event_type.name}:{e.name}", e.expression)
        for i, e in enumerate(model.events)
    ]
    expressions += [
        ("output:" + name, model.variables[name])
        for name in (
            "Voltage [V]",
            "Current [A]",
            "Discharge capacity [A.h]",
            "Discharge energy [W.h]",
            PHYSICAL_V,
        )
    ]
    for label, expression in expressions:
        dependencies = set()
        for node in expression.pre_order():
            if isinstance(node, pybamm.StateVector):
                dependencies.update(
                    i
                    for i in indices
                    if i < len(node.evaluation_array) and node.evaluation_array[i]
                )
        checked[label] = sorted(dependencies)
    return {
        "method": "processed expression DAG StateVector evaluation-array dependency inspection",
        "physical_work_state_indices": indices,
        "event_identities": event_identities,
        "checked_expressions": checked,
        "no_feedback": not any(checked.values()),
    }


def source_contract_snapshot(built):
    model, values = built["model"], built["parameter_values"]
    return {
        "pybamm_version": PYBAMM_VERSION,
        "uniform": built["uniform"],
        "grid": built["grid"],
        "authored_grid": built["authored_grid"],
        "requested_grid": built["authored_grid"],
        "actual_grid": built["grid"],
        "layout": built["layout"],
        "control": built["control"],
        "rhs_size": model.concatenated_rhs.size,
        "algebraic_size": model.concatenated_algebraic.size,
        "states": [
            {
                "name": v.name,
                "slices": [[p.start, p.stop] for p in model.y_slices[v]],
                "scale": float(values.evaluate(v.scale)),
                "reference": float(values.evaluate(v.reference)),
                "domains": {k: list(x) for k, x in v.domains.items()},
                "kind": "differential" if v in model.rhs else "algebraic",
            }
            for v in (*model.rhs, *model.algebraic)
        ],
        "original_events": list(built["original_events"]),
        "original_event_definitions": [
            {"name": e.name, "type": e.event_type.name}
            for e in model.events
            if e.name in built["original_events"]
        ],
        "events": [{"name": e.name, "type": e.event_type.name} for e in model.events],
        "original_rhs_names": list(built["original_rhs_names"]),
        "z_nodes_m": built["z_nodes_m"],
        "physical_port_definition": "uniform source Voltage"
        if built["uniform"]
        else "last positive FV centre minus Iport*dz/(2*sigma_p*thickness_p*width)",
        "energy_definition": {
            "original": "signed source discharge energy; RHS I*original Voltage",
            "additional": "extra observer state in J; RHS I*physical tab Voltage; no feedback",
        },
        "physical_work_no_feedback": _physical_work_dependency_check(model),
    }


def _uniform_segment(built, solution, data, inputs):
    """Keep genuine dim0 source arrays; a singleton z axis is only a schema label."""
    import pybamm

    model, mesh = built["model"], built["mesh"]
    count = len(solution.t)
    profiles = {}
    for name in ("negative", "positive"):
        p = data[name]
        concentration = np.asarray(
            _entries(solution, f"X-averaged {name} particle concentration [mol.m-3]")
        ).reshape(count, built["grid"]["r"], 1)
        radial = mesh[name + " particle"]
        volumes = np.diff(radial.edges**3) / p["radius_m"] ** 3 * p["active_volume_m3"]
        j = _scalar(
            solution, f"X-averaged {name} electrode interfacial current density [A.m-2]", count
        )
        profiles[name] = {
            "electrode": name,
            "theta": (concentration / p["maximum_concentration_mol_m3"]).tolist(),
            "concentration_mol_m3": concentration.tolist(),
            "shell_amounts_mol": (concentration * volumes[None, :, None]).tolist(),
            "reaction_current_density_A_m2": j[:, None].tolist(),
            "reaction_current_A": (j[:, None] * 3 * p["active_volume_m3"] / p["radius_m"]).tolist(),
            "z_nodes_m": [data["height_m"] / 2],
            "z_edges_m": [0, data["height_m"]],
            "r_nodes_m": radial.nodes.tolist(),
            "r_edges_m": radial.edges.tolist(),
            "radial_axis_order": ["time", "r", "singleton uniform label"],
            "local_axis_order": ["time", "singleton uniform label"],
        }
        for field, variable, divisor in (
            (
                "average_fraction",
                f"Average {name} particle concentration [mol.m-3]",
                p["maximum_concentration_mol_m3"],
            ),
            ("surface_fraction", f"X-averaged {name} particle surface concentration", 1),
            (
                "kinetic_overpotential_V",
                f"X-averaged {name} electrode reaction overpotential [V]",
                1,
            ),
            (
                "exchange_current_density_A_m2",
                f"X-averaged {name} electrode exchange current density [A.m-2]",
                1,
            ),
            (
                "open_circuit_potential_V",
                f"X-averaged {name} electrode open-circuit potential [V]",
                1,
            ),
        ):
            profiles[name][field] = (_scalar(solution, variable, count)[:, None] / divisor).tolist()
    y = np.asarray(solution.y)
    original_voltage = _scalar(solution, "Voltage [V]", count)
    segment = {
        "times_s": solution.t.tolist(),
        "end_time_s": float(solution.t[-1]),
        "termination": solution.termination,
        "outcome": "success" if solution.termination == "final time" else "stopped",
        "values": np.column_stack(
            (
                original_voltage,
                _scalar(solution, "Current [A]", count),
                3600 * _scalar(solution, "Discharge capacity [A.h]", count),
                3600 * _scalar(solution, "Discharge energy [W.h]", count),
            )
        ).tolist(),
        "raw_profiles": profiles,
        "source_fields": {
            name: _raw_source_output(model, solution, name, inputs).tolist()
            for name in (
                "Negative current collector potential [V]",
                "Positive current collector potential [V]",
                "Local voltage [V]",
            )
        },
        "source_state_vectors": y.T.tolist(),
        "source_initial_rhs": np.asarray(
            model.concatenated_rhs.evaluate(t=float(solution.t[0]), y=y[:, :1], inputs=inputs)
        )
        .reshape(-1)
        .tolist(),
        "source_initial_algebraic_residual": np.asarray(
            model.concatenated_algebraic.evaluate(t=float(solution.t[0]), y=y[:, :1], inputs=inputs)
        )
        .reshape(-1)
        .tolist(),
        "source_event_values_at_endpoint": {
            e.name: np.asarray(
                e.expression.evaluate(t=float(solution.t[-1]), y=y[:, -1:], inputs=inputs)
            )
            .reshape(-1)
            .tolist()
            for e in model.events
            if e.event_type == pybamm.EventType.TERMINATION
        },
        "source_boundary_voltage_V": original_voltage.tolist(),
        "solver": {
            "name": "PyBaMM25.4.2 IDAKLUSolver",
            "rtol": RTOL,
            "atol": ATOL,
            "root_tol": ROOT_TOL,
            "integration_time_s": None
            if solution.integration_time is None
            else float(solution.integration_time.value),
        },
    }
    return segment


def _add_physical_observations(segment, built, solution):
    count = len(solution.t)
    voltage = _scalar(solution, PHYSICAL_V, count)
    work = _scalar(solution, PHYSICAL_W, count)
    original = np.asarray(segment["values"])
    segment["physical_values"] = np.column_stack((voltage, original[:, 1:3], work)).tolist()
    segment["physical_tab_voltage_V"] = voltage.tolist()
    segment["physical_work_J"] = work.tolist()
    segment["independent_physical_tab_voltage_V"] = None
    segment["physical_port_provenance"] = (
        "authored FV-Neumann observation from actual source state, not an independent prediction"
    )
    segment["controlled_port"] = "physical-tab" if built["control"] != "CC" else "current"
    segment["source_fields"][PHYSICAL_V] = voltage.tolist()
    segment["source_fields"][PHYSICAL_W] = work.tolist()


def _time(data, grid, layout, uniform):
    import pybamm

    progress = {
        "completed_segments": [],
        "stage": "build",
        "current_time_s": 0.0,
        "grid": dict(grid),
        "layout": dict(layout),
        "uniform": uniform,
    }
    model, inputs, start = None, {}, 0.0
    try:
        built = build_reference(data, grid, layout, uniform=uniform)
        model = built["model"]
        simulation = pybamm.Simulation(
            model,
            parameter_values=built["parameter_values"],
            solver=pybamm.IDAKLUSolver(rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL),
        )
        simulation.build()
        for index, (duration, target, mode) in enumerate(data["waveform"]):
            start = 0.0 if simulation.solution is None else float(simulation.solution.t[-1])
            inputs = {"Current function [A]": target} if mode == "A" else {TARGET_V: target}
            times = np.linspace(0, duration, data["samples_per_segment"])
            progress.update(
                stage="integrate",
                active_segment={
                    "index": index,
                    "start_time_s": start,
                    "duration_s": duration,
                    "target": target,
                    "unit": mode,
                },
            )
            solution = simulation.step(
                duration, save=False, t_eval=times, t_interp=times, inputs=inputs
            )
            progress["stage"] = "extract"
            progress["pending_solution"] = {
                "times_s": solution.t.tolist(),
                "source_state_vectors": np.asarray(solution.y).T.tolist(),
                "inputs": dict(inputs),
                "termination": solution.termination,
            }
            segment = (
                _uniform_segment(built, solution, data, inputs)
                if uniform
                else _source_segment(
                    built["mesh"], model, solution, start, data, grid, layout, inputs
                )
            )
            _add_physical_observations(segment, built, solution)
            progress["completed_segments"].append(segment)
            del progress["pending_solution"]
            progress["current_time_s"] = float(solution.t[-1])
            if solution.termination != "final time":
                break
        return {
            "implementation": "source-uniform" if uniform else "source",
            "family": "actual-uniform-spm" if uniform else "collector-spm-physical-port-supplement",
            "unit": data["id"],
            "inputs": deepcopy(data),
            "grid": built["grid"],
            "authored_grid": dict(grid),
            "requested_grid": dict(grid),
            "actual_grid": built["grid"],
            "layout": dict(layout),
            "source_version": PYBAMM_VERSION,
            "source_build": source_contract_snapshot(built),
            "segments": progress["completed_segments"],
            "faraday_C_mol": float(pybamm.constants.F.value),
            "tolerances": {"rtol": RTOL, "atol": ATOL, "root_tol": ROOT_TOL},
            "value_columns": ["source_voltage_V", "current_A", "signed_charge_C", "source_work_J"],
            "physical_value_columns": [
                "physical_tab_voltage_V",
                "current_A",
                "signed_charge_C",
                "physical_work_J",
            ],
        }
    except Exception as exc:
        if model is not None and getattr(model, "y0", None) is not None:
            diagnostic = _source_initialization_diagnostic(model, inputs, start)
            for name in (PHYSICAL_V, PHYSICAL_W):
                try:
                    diagnostic["variables"][name] = (
                        np.asarray(
                            model.variables[name].evaluate(
                                t=start, y=np.asarray(model.y0).reshape(-1, 1), inputs=inputs
                            )
                        )
                        .reshape(-1)
                        .tolist()
                    )
                except Exception as error:
                    diagnostic["read_errors"][name] = f"{type(error).__name__}: {error}"
            progress["consistent_initialization"] = diagnostic
        progress.update(exception_type=type(exc).__name__, exception=str(exc))
        exc.partial_history = progress
        raise


def source_time(data, grid, layout):
    return _time(data, grid, layout, False)


def source_uniform_time(data, grid):
    return _time(data, grid, {"negative_tab": "left", "positive_tab": "right"}, True)
