"""Released PyBaMM DFN, validation-only; no native equation or solver import."""

import numpy as np


def build_dfn_reference(spec, *, unit="A", rtol=1e-7, atol=1e-8,
                        spatial_cells=64, radial_cells=100, radial_mesh="exponential-right",
                        ocp_functions=None):
    """Construct the same released model for time- and frequency-domain validation."""
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed DFN reference requires PyBaMM 25.4.2")
    if unit not in {"A", "V"}:
        raise ValueError("reference circuit unit must be A or V")
    if ocp_functions is None and any(getattr(spec, n + "_ocv", None) is not None
                                     for n in ("negative", "positive")):
        raise ValueError("nonlinear reference requires independent OCP functions")
    model = pybamm.lithium_ion.DFN({"thermal": "isothermal", "surface form": "false",
                                   "operating mode": "current" if unit == "A" else "voltage"})
    e = spec.electrolyte
    values = {
        "Electrode height [m]": e.area_m2, "Electrode width [m]": 1.0,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Negative electrode thickness [m]": e.lengths_m[0],
        "Separator thickness [m]": e.lengths_m[1],
        "Positive electrode thickness [m]": e.lengths_m[2],
        "Separator porosity": e.porosities[1],
        "Separator Bruggeman coefficient (electrolyte)": e.bruggeman[1],
        "Initial concentration in electrolyte [mol.m-3]": e.initial_concentration_mol_m3,
        "Electrolyte diffusivity [m2.s-1]": e.diffusivity_m2_s,
        "Electrolyte conductivity [S.m-1]": e.conductivity_S_m,
        "Cation transference number": e.cation_transference,
        "Thermodynamic factor": e.thermodynamic_factor,
        "Initial temperature [K]": spec.temperature_K,
        "Ambient temperature [K]": spec.temperature_K,
        "Reference temperature [K]": spec.temperature_K,
        "Lower voltage cut-off [V]": spec.minimum_voltage_V,
        "Upper voltage cut-off [V]": spec.maximum_voltage_V,
        "Current function [A]" if unit == "A" else "Voltage function [V]": "[input]",
    }
    for name, particle, region, sigma in (
        ("negative", spec.negative, 0, e.solid_conductivity_S_m[0]),
        ("positive", spec.positive, 2, e.solid_conductivity_S_m[1]),
    ):
        title = name.capitalize()
        a, b = (getattr(spec, f"{name}_ocv_{term}_V") for term in ("intercept", "slope"))

        def ocp(theta, a=a, b=b):
            return a + b * theta

        if ocp_functions is not None:
            ocp = ocp_functions[name]

        def exchange(c_e, surface, maximum, temperature,
                     scale=particle.exchange_current_scale_A_m2):
            theta = surface / maximum
            return scale * pybamm.sqrt(c_e / e.initial_concentration_mol_m3
                                       * theta * (1 - theta))

        values.update({
            f"{title} electrode conductivity [S.m-1]": sigma,
            f"{title} electrode Bruggeman coefficient (electrode)": 0.0,
            f"{title} electrode porosity": e.porosities[region],
            f"{title} electrode Bruggeman coefficient (electrolyte)": e.bruggeman[region],
            f"{title} electrode active material volume fraction": (
                particle.active_volume_m3 / (e.area_m2 * e.lengths_m[region])),
            f"{title} particle radius [m]": particle.radius_m,
            f"{title} particle diffusivity [m2.s-1]": particle.diffusivity_m2_s,
            f"Maximum concentration in {name} electrode [mol.m-3]": (
                particle.maximum_concentration_mol_m3),
            f"Initial concentration in {name} electrode [mol.m-3]": (
                getattr(spec, f"initial_{name}_fraction") * particle.maximum_concentration_mol_m3),
            f"{title} electrode OCP [V]": ocp,
            f"{title} electrode OCP entropic change [V.K-1]": 0.0,
            f"{title} electrode exchange-current density [A.m-2]": exchange,
            f"{title} electrode charge transfer coefficient": 0.5,
        })
        surface = model.variables[f"{title} particle surface concentration"]
        model.events.extend((
            pybamm.Event(f"native:{name}:depleted", pybamm.min(surface) - spec.surface_margin),
            pybamm.Event(f"native:{name}:saturated", 1 - spec.surface_margin - pybamm.max(surface)),
        ))
    model.events.append(pybamm.Event("native:electrolyte:depleted", pybamm.min(
        model.variables["Electrolyte concentration [mol.m-3]"])
        - e.minimum_concentration_mol_m3))
    if spec.series_resistance_ohm:
        raise ValueError("fixed reference block uses zero additional series resistance")
    parameters = model.default_parameter_values
    parameters.update(values, check_already_exists=False)
    points = {var: (radial_cells if "r_" in str(var) else spatial_cells)
              for var in model.default_var_pts}
    meshes = model.default_submesh_types
    if radial_mesh == "exponential-right":
        for domain in ("negative particle", "positive particle"):
            meshes[domain] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh,
                                                 {"side": "right", "stretch": 2.3})
    elif radial_mesh != "uniform":
        raise ValueError("unsupported reference radial mesh")
    simulation = pybamm.Simulation(model, parameter_values=parameters, var_pts=points,
        submesh_types=meshes,
        # The dimensional solid-potential divergence subtracts nearly equal
        # ~4 V values on a 64-cell mesh. Initial roots reach a ~2e-9 residual
        # roundoff floor in PyBaMM's scaled charge equation (retained attempts
        # 1/2). This is a numerical initialization tolerance, not any of the
        # fixed physical comparison bounds. IDAKLU avoids Casadi safe-mode's
        # repeated vanishing-step retries on this dimensional formulation.
        # Reference integration tolerances remain far below the predeclared
        # field bounds; these settings must also be checked against refinement.
        solver=pybamm.IDAKLUSolver(rtol=rtol, atol=atol, root_tol=1e-8,
                                  root_method="lm"))
    return simulation


def reference_dfn(spec, segments, *, evaluation_times=None, rtol=1e-7, atol=1e-8,
                  spatial_cells=64, radial_cells=100, radial_mesh="exponential-right"):
    units = {unit for _, _, unit in segments}
    if len(units) != 1:
        raise ValueError("this reference block uses one circuit mode per independent unit")
    unit = next(iter(units))
    simulation = build_dfn_reference(spec, unit=unit, rtol=rtol, atol=atol,
        spatial_cells=spatial_cells, radial_cells=radial_cells, radial_mesh=radial_mesh)
    answers = []
    for index, (duration, value, _) in enumerate(segments):
        start = 0.0 if simulation.solution is None else float(simulation.solution.t[-1])
        times = np.linspace(0, duration, 101) if evaluation_times is None else np.unique(
            np.r_[0.0, evaluation_times[index], duration])
        solution = simulation.step(duration, t_eval=times,
            inputs={"Current function [A]" if unit == "A" else "Voltage function [V]": value})
        answers.append((start, float(solution.t[-1]), solution))
        if solution.termination != "final time":
            break
    return answers
