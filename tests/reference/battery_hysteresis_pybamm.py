"""Released SPM/MPM OCP classes with explicit net-branch parameter mapping."""

from types import SimpleNamespace

import numpy as np
from battery_frequency_pybamm import frequency_from_simulation
from battery_hysteresis_cases import (
    FREQUENCIES,
    branch_polynomial,
    response_names,
    sweep_currents,
)
from battery_spme_pybamm import build_spme_reference


def reference_family(data):
    return "MPM" if data["family"] == "MPM" and data["size_distribution"] != "single" else "SPM"


def reference_declaration(data):
    """Scalar inputs for the reference factory, without any native size quadrature."""
    particle = SimpleNamespace(**{key: data[key] for key in (
        "radius_m", "diffusivity_m2_s", "maximum_concentration_mol_m3",
        "active_volume_m3", "exchange_current_scale_A_m2")})
    values = {key: data[key] for key in (
        "initial_negative_fraction", "initial_positive_fraction", "temperature_K",
        "surface_margin", "minimum_voltage_V", "maximum_voltage_V", "maximum_current_A")}
    values.update(negative=particle, positive=particle, electrolyte=None,
                  series_resistance_ohm=0.)
    for name in ("negative", "positive"):
        a, b = data[name+"_equilibrium_polynomial_V"]
        values.update({name+"_ocv_intercept_V": a, name+"_ocv_slope_V": b})
        values[name+"_distribution"] = SimpleNamespace(distribution_parameters=(
            ("lower_radius", SimpleNamespace(value=5e-6)),
            ("upper_radius", SimpleNamespace(value=17e-6))))
    return SimpleNamespace(**values)


def build_reference(data, radial_cells, size_cells, tight):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("OCP reference requires released PyBaMM25.4.2")
    family, spec = reference_family(data), reference_declaration(data)
    cv = data["waveform"][0][2] == "V"
    # Released MPM normalizes this independent continuous area density on its
    # own size FV grid. No native Gauss nodes, weights or node cap are involved.
    pdfs = {name: lambda radius: pybamm.exp(
        -(pybamm.log(radius/1e-5)**2)/(2*.12**2))/radius
        for name in ("negative", "positive")}
    baseline = build_spme_reference(spec, family=family, cells=5,
        radial_cells=radial_cells, size_cells=size_cells,
        size_pdf_functions=pdfs if family == "MPM" else None,
        radial_mesh="uniform", voltage_control=cv)

    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current+data["maximum_current_A"]*(
            variables["Voltage [V]"]-pybamm.InputParameter("native voltage target"))
        return current-pybamm.maximum(-data["maximum_current_A"], pybamm.minimum(
            data["maximum_current_A"], requested))

    model = getattr(pybamm.lithium_ion, family)({"thermal": "isothermal",
        "surface form": "algebraic" if family == "MPM" else "false",
        "particle": "Fickian diffusion",
        "open-circuit potential": (data["negative_mode"], data["positive_mode"]),
        **({"operating mode": finite_source} if cv else {})})
    values, area = {}, data["effective_collector_area_m2"]
    # Keep electrode active fraction .5 for every area variant; true capacity unchanged.
    values.update({"Electrode height [m]": area, "Electrode width [m]": 1.,
        "Number of electrodes connected in parallel to make a cell": 1.})

    def compensated(a, b):
        return lambda theta: a+b*theta-1e-6*(1/theta+1/(theta-1))

    for electrode in ("negative", "positive"):
        title = electrode.capitalize()
        values.update({f"{title} electrode thickness [m]": data["active_volume_m3"]/(.5*area),
                       f"{title} electrode active material volume fraction": .5,
                       f"{title} electrode OCP [V]": compensated(
                           *data[electrode+"_equilibrium_polynomial_V"])})
        if data[electrode+"_mode"] != "single":
            for name, lith in (("lithiation", True), ("delithiation", False)):
                values[f"{title} electrode {name} OCP [V]"] = compensated(
                    *branch_polynomial(data, electrode, lith))
        if data[electrode+"_mode"] == "Axen":
            values.update({f"Initial hysteresis state in {electrode} electrode":
                           data[electrode+"_initial_h"],
                f"{title} particle lithiation hysteresis decay rate": data["lithiation_decay"],
                f"{title} particle delithiation hysteresis decay rate": data["delithiation_decay"]})
            h = model.variables[f"{title} electrode hysteresis state"]
            model.variables[f"qualification:{electrode}:h-rate"] = model.rhs[h]
            # The released X-averaged h has a particle-size broadcast in MPM.
            # The actual persistent variable is the electrode-domain h above.
            model.variables[f"qualification:{electrode}:shared-h"] = pybamm.x_average(h)
        surface = model.variables[f"X-averaged {electrode} particle surface concentration"
            +( " distribution" if family == "MPM" else "")]
        model.variables[f"qualification:{electrode}:area-surface"] = (
            pybamm.size_average(surface) if family == "MPM" else surface)
        # The MPM OCP observation uses its area density, just like surface
        # stoichiometry; an unweighted arithmetic mean would be a different law.
        ocp = model.variables[f"X-averaged {electrode} electrode open-circuit potential"
            +( " distribution [V]" if family == "MPM" else " [V]")]
        model.variables[f"qualification:{electrode}:area-ocp"] = (
            pybamm.size_average(ocp) if family == "MPM" else ocp)
        model.events.extend([
            pybamm.Event(f"qualification:{electrode}:depleted",
                1e4*(pybamm.min(surface)-data["surface_margin"])),
            pybamm.Event(f"qualification:{electrode}:saturated",
                1e4*(1-data["surface_margin"]-pybamm.max(surface)))])
    parameters = baseline.parameter_values.copy()
    parameters.update(values, check_already_exists=False)
    rtol, atol = (1e-10, 1e-12) if tight else (1e-8, 1e-10)
    return pybamm.Simulation(model, parameter_values=parameters, var_pts=baseline.var_pts,
        submesh_types=baseline.submesh_types, spatial_methods=baseline.spatial_methods,
        solver=pybamm.CasadiSolver(mode="safe", rtol=rtol, atol=atol, root_tol=atol))


def output_names(data):
    names = ["Voltage [V]", "Current [A]", "qualification:negative:area-surface",
             "qualification:positive:area-surface", "Average negative particle concentration",
             "Average positive particle concentration", "Discharge capacity [A.h]"]
    names += [f"qualification:{e}:shared-h" for e in ("negative", "positive")
              if data[e+"_mode"] == "Axen"]
    names += [f"qualification:{e}:area-ocp"
              for e in ("negative", "positive")]
    return names


def scalar_outputs(solution, times, data):
    """Reject accidental spatial flattening instead of hiding it with a mean."""
    columns = []
    for key in output_names(data):
        values = np.asarray(solution[key](times))
        if values.size != len(times):
            raise ValueError(f"reference observable {key!r} is not scalar at each time")
        columns.append(values.reshape(-1))
    return np.column_stack(columns)


def shared_h_broadcast_errors(solution, times, data):
    """Verify that all spatial/size copies represent the same electrode history."""
    errors = {}
    for name in ("negative", "positive"):
        if data[name+"_mode"] != "Axen":
            continue
        shared = np.asarray(solution[f"qualification:{name}:shared-h"](times)).reshape(-1)
        for representation, key in (
            ("electrode", f"{name.capitalize()} electrode hysteresis state"),
            ("size-broadcast", f"X-averaged {name} electrode hysteresis state")):
            values = np.asarray(solution[key](times)).reshape(-1, len(times))
            error = float(np.max(abs(values-shared)))
            errors[name+":"+representation] = error
            if not np.isfinite(error) or error > 1e-12:
                raise ValueError("released shared electrode h has nonuniform broadcast copies")
    return errors


def reference_time(data, radial_cells, size_cells, tight):
    simulation, segments = build_reference(data, radial_cells, size_cells, tight), []
    for duration, target, unit in data["waveform"]:
        solution = simulation.step(duration, save=False, t_eval=np.linspace(
            0., duration, data["samples_per_segment"]),
            inputs={"native voltage target" if unit == "V" else "Current function [A]": target})
        times = np.asarray(solution.t)
        rows = scalar_outputs(solution, times, data)
        shared_h_errors = shared_h_broadcast_errors(solution, times, data)
        rows[:, 6] *= 3600.
        local = {}
        if reference_family(data) == "MPM":
            for electrode in ("negative", "positive"):
                radii = simulation.mesh[electrode+" particle size"].nodes
                surface = np.asarray(solution[f"X-averaged {electrode} particle surface "
                    "concentration distribution"](times, R=radii)).reshape(len(radii), -1).T
                current = np.asarray(solution[f"X-averaged {electrode} electrode interfacial "
                    "current density distribution [A.m-2]"](times, R=radii)).reshape(
                        len(radii), -1).T
                local[electrode] = {"radii_m": radii.tolist(), "surface": surface.tolist(),
                                    "j_A_m2": current.tolist()}
        segments.append({"times_s": times.tolist(), "rows": rows.tolist(), "local": local,
            "shared_h_broadcast_maximum_errors": shared_h_errors,
            "termination": solution.termination, "end_time_s": float(times[-1])})
        if solution.termination != "final time":
            break
    return {"implementation": "PyBaMM25.4.2 named OCP", "response_order": response_names(data),
            "radial_cells": radial_cells, "size_cells": size_cells, "tight": tight,
            "segments": segments}


def reference_static(data, radial_cells=160):
    simulation = build_reference(data, radial_cells, None, True)
    simulation.build()
    model, solver = simulation.built_model, simulation.solver
    points = []
    for current in sweep_currents():
        inputs = {"Current function [A]": current}
        solver.set_up(model, inputs=inputs)
        state = np.asarray(solver.calculate_consistent_state(model, inputs=inputs)).ravel()
        values = [float(np.asarray(model.variables[key].evaluate(
            t=0., y=state, inputs=inputs)).mean()) for key in output_names(data)]
        values[6] *= 3600.
        points.append({"current_A": current, "row": values})
    return {"response_order": response_names(data), "points": points}


def reference_frequency(data, radial_cells):
    simulation = build_reference(data, radial_cells, None, True)
    return frequency_from_simulation(simulation, FREQUENCIES, family="SPM-"+data["negative_mode"],
        grid={"radial_cells": radial_cells, "radial_mesh": "uniform"},
        output_names=["Voltage [V]"])


def reference_nonsmooth(data):
    simulation = build_reference(data, 160, None, True)
    simulation.build()
    model, solver = simulation.built_model, simulation.solver
    inputs = {"Current function [A]": 0.}
    solver.set_up(model, inputs=inputs)
    state = np.asarray(solver.calculate_consistent_state(model, inputs=inputs)).ravel()
    points = {}
    for electrode in ("negative", "positive"):
        rates = []
        for current in (-1e-8, 0., 1e-8):
            values = model.variables[f"qualification:{electrode}:h-rate"].evaluate(t=0., y=state,
                inputs={"Current function [A]": current})
            rates.append(float(np.asarray(values).mean()))
        points[electrode] = {"rates_s_inv": rates,
            "left_derivative_s_inv_A": (rates[1]-rates[0])/1e-8,
            "right_derivative_s_inv_A": (rates[2]-rates[1])/1e-8}
    return {"epsilon_A": 1e-8, "points": points,
        "note": "actual released RHS two-sided input probe; no claimed unique rest Jacobian"}
