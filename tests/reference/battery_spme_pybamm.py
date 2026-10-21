"""Independent released PyBaMM SPMe adapter, validation only (25.4.2)."""

from types import SimpleNamespace

import numpy as np


def build_spme_reference(spec, *, cells=120, radial_mesh="uniform", family="SPMe",
                          ocp_functions=None, radial_cells=None, size_cells=None,
                          size_pdf_functions=None, voltage_control=False):
    """Construct the same released model for time- and frequency-domain validation."""
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("fixed SPMe reference requires PyBaMM 25.4.2")
    if family not in {"SPM", "SPMe", "MPM"}:
        raise ValueError("reference family must be SPM, SPMe or MPM")
    e = spec.electrolyte
    if family in {"SPM", "MPM"} and e is None:
        # SPM's reference cross-section cancels with active volume fraction.
        # Leading-order electrolyte/solid losses are identically zero.
        e = SimpleNamespace(area_m2=.02, lengths_m=(1e-4, 2.5e-5, 1e-4),
            porosities=(.4, .5, .4), bruggeman=(1.5,) * 3,
            initial_concentration_mol_m3=1000., minimum_concentration_mol_m3=1.,
            diffusivity_m2_s=2e-10, conductivity_S_m=1.,
            cation_transference=.4, thermodynamic_factor=1., solid_conductivity_S_m=(100., 100.))
    if e is None:
        raise ValueError("SPMe reference requires an electrolyte declaration")
    if ocp_functions is None and any(getattr(spec, n + "_ocv", None) is not None
                                     for n in ("negative", "positive")):
        raise ValueError("nonlinear reference requires independent OCP functions")
    if family == "MPM" and (size_pdf_functions is None or size_cells is None):
        raise ValueError("MPM reference needs independent area PDFs and a size mesh")
    def finite_source(variables):
        current = variables["Current [A]"]
        requested = current+spec.maximum_current_A*(variables["Voltage [V]"]
                                                  -pybamm.InputParameter("native voltage target"))
        return current-pybamm.maximum(-spec.maximum_current_A,
                                     pybamm.minimum(spec.maximum_current_A, requested))
    model = getattr(pybamm.lithium_ion, family)({
        "thermal": "isothermal", "surface form": "algebraic" if family == "MPM" else "false",
        "particle": "Fickian diffusion",
        **({"operating mode": finite_source} if voltage_control else {}),
        **({"electrolyte conductivity": "integrated"} if family == "SPMe" else {}),
    })
    parameters = model.default_parameter_values
    values = {
        "Electrode height [m]": e.area_m2,
        "Electrode width [m]": 1.0,
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
        "Current function [A]": "[input]",
    }
    for name, particle, i, sigma in (
        ("negative", spec.negative, 0, e.solid_conductivity_S_m[0]),
        ("positive", spec.positive, 2, e.solid_conductivity_S_m[1]),
    ):
        title = name.capitalize()
        a, b = (getattr(spec, f"{name}_ocv_{term}_V") for term in ("intercept", "slope"))

        def ocp(theta, a=a, b=b):
            return a + b * theta

        if ocp_functions is not None:
            ocp = ocp_functions[name]
        if family == "MPM":
            # Compare the SAME declared net material OCV. Released U() adds
            # this numerical barrier separately; cancel it in the input law.
            original_ocp = ocp
            def ocp(theta, function=original_ocp):
                return function(theta)-1e-6*(1/theta+1/(theta-1))

        def exchange(c_e, c_s_surf, c_s_max, temperature,
                     scale=particle.exchange_current_scale_A_m2):
            theta = c_s_surf / c_s_max
            return scale * pybamm.sqrt(c_e / e.initial_concentration_mol_m3) * pybamm.sqrt(
                theta * (1 - theta))

        values.update({
            f"{title} electrode conductivity [S.m-1]": sigma,
            f"{title} electrode Bruggeman coefficient (electrode)": 0.0,
            f"{title} electrode porosity": e.porosities[i],
            f"{title} electrode Bruggeman coefficient (electrolyte)": e.bruggeman[i],
            f"{title} electrode active material volume fraction": (
                particle.active_volume_m3 / (e.area_m2 * e.lengths_m[i])),
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
        if family == "MPM":
            dist = getattr(spec, name+"_distribution")
            metadata = dict(dist.distribution_parameters)
            values.update({f"{title} minimum particle radius [m]": metadata["lower_radius"].value,
                f"{title} maximum particle radius [m]": metadata["upper_radius"].value,
                f"{title} area-weighted particle-size distribution [m-1]":
                    size_pdf_functions[name]})
        surface = model.variables[(f"X-averaged {name} particle surface concentration distribution"
                                   if family == "MPM" else
                                   f"X-averaged {name} particle surface concentration")]
        if family == "MPM":
            model.variables[f"native:{name}:area-surface"] = pybamm.size_average(surface)
        # CasadiSolver 25.4.2 subtracts an absolute 1e-5 from event
        # expressions when locating roots. Scale the same dimensionless
        # zero surface in MPM so its numerical displacement is 1e-9, not
        # 1e-5 in stoichiometry. This changes no physical stopping face.
        event_scale = 1e4 if family == "MPM" else 1.0
        model.events.extend([
            pybamm.Event(f"native:{name}:depleted",
                         event_scale * (pybamm.min(surface) - spec.surface_margin)),
            pybamm.Event(f"native:{name}:saturated",
                         event_scale * (1 - spec.surface_margin - pybamm.max(surface))),
        ])
    model.events.append(pybamm.Event(
        "native:electrolyte:depleted",
        pybamm.min(model.variables["Electrolyte concentration [mol.m-3]"])
        - e.minimum_concentration_mol_m3,
    ))
    if spec.series_resistance_ohm:
        raise ValueError("reference block uses zero extra series resistance")
    parameters.update(values)
    mesh_points = {var: (radial_cells if radial_cells is not None and "r_" in str(var) else cells)
                   for var in model.default_var_pts}
    if family == "MPM":
        mesh_points.update({var: size_cells for var in model.default_var_pts if "R_" in str(var)})
    meshes = model.default_submesh_types
    if radial_mesh == "exponential-right":
        for domain in ("negative particle", "positive particle"):
            meshes[domain] = pybamm.MeshGenerator(pybamm.Exponential1DSubMesh,
                                                 {"side": "right", "stretch": 2.3})
    elif radial_mesh != "uniform":
        raise ValueError("unsupported reference radial mesh")
    simulation = pybamm.Simulation(model, parameter_values=parameters, var_pts=mesh_points,
                                   submesh_types=meshes,
                                   solver=(pybamm.CasadiSolver(mode="safe", rtol=1e-9, atol=1e-9)
                                           if family == "MPM" else
                                           pybamm.ScipySolver(method="BDF", rtol=1e-9, atol=1e-9)))
    return simulation


def reference_spme(spec, segments, *, cells=120, evaluation_times=None):
    simulation = build_spme_reference(spec, cells=cells)
    answers = []
    for index, (duration, current) in enumerate(segments):
        previous = 0.0 if simulation.solution is None else float(simulation.solution.t[-1])
        times = (np.linspace(0, duration, 101) if evaluation_times is None
                 else np.unique(np.r_[0.0, evaluation_times[index], duration]))
        solution = simulation.step(duration, t_eval=times,
                                   inputs={"Current function [A]": current})
        answers.append((previous, float(solution.t[-1]), solution.last_state, solution))
        if solution.termination != "final time":
            break
    return answers
