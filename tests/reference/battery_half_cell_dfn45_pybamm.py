"""Actual PyBaMM25.4.2 modular/Basic half-cell source, with explicit variants.

Full Xu2019 preserves all released material functions. The constant-transport
variant replaces only D_e and kappa at ce_ref=1000,T=298.15. Basic's original
working-cmax metal argument and the explicit pure-metal normalization are kept
as different routes. No OCP/barrier compensation or native equation is imported.
"""

import inspect
from copy import deepcopy
from importlib.metadata import version

import numpy as np
from battery_half_cell_dfn45_cases import (
    ATOL,
    CASADI_VERSION,
    PYBAMM_VERSION,
    PYBAMMSOLVERS_VERSION,
    RESPONSE_ORDER,
    ROOT_TOL,
    RTOL,
    SOURCE_VARIANTS,
    XU2019_PROJECTION,
    F,
    stage_times,
    validate_case,
    validate_grid,
)

PREFIX = "half-cell-dfn45:"
SOURCE_ROOT_METHOD = "casadi"
MODE, CURRENT, VOLTAGE = (
    PREFIX + s for s in ("voltage-mode", "current-target-A", "voltage-target-V")
)
BASIC_WORK = PREFIX + "Basic signed work [J]"


def _dependencies():
    import casadi
    import pybamm

    if (pybamm.__version__, casadi.__version__) != (PYBAMM_VERSION, CASADI_VERSION):
        raise RuntimeError("half-cell source requires actual PyBaMM25.4.2/CasADi3.6.7")
    if version("pybammsolvers") != PYBAMMSOLVERS_VERSION:
        raise RuntimeError("half-cell source requires actual pybammsolvers0.3.3")
    return pybamm


def stage_inputs(stage, variant):
    if variant.startswith("basic"):
        if stage["unit"] != "A":
            raise ValueError("actual BasicDFNHalfCell is a prescribed-current cross-reference")
        return {CURRENT: stage["target"]}
    return dict(
        sorted(
            {
                MODE: float(stage["unit"] == "V"),
                CURRENT: stage["target"] if stage["unit"] == "A" else 0.0,
                VOLTAGE: stage["target"] if stage["unit"] == "V" else 0.0,
            }.items()
        )
    )


def _manifest(released):
    return {
        name: (
            {
                "kind": "function",
                "function_id": value.__name__,
                "module": value.__module__,
                "source": inspect.getsource(value),
            }
            if callable(value)
            else {
                "kind": "number" if isinstance(value, (int, float)) else "metadata",
                "value": deepcopy(value),
            }
        )
        for name, value in released.items()
    }


def build_reference(data, grid, variant="modular-full-Xu2019"):
    """Construct/discretize one source route; this function does not integrate."""
    pybamm = _dependencies()
    validate_case(data)
    validate_grid(grid)
    if min(grid["x"]) < 3:
        raise ValueError("pinned source FV boundary reconstruction needs three cells per x domain")
    if variant not in SOURCE_VARIANTS:
        raise ValueError("unknown half-cell source variant")
    basic = variant.startswith("basic")
    constant = "constant-transport" in variant
    normalized = basic and variant.endswith("pure-metal")
    if basic and any(stage["unit"] != "A" for stage in data["stages"]):
        raise ValueError("BasicDFNHalfCell cross-reference accepts CC/rest/reverse only")

    def finite_source(variables):
        current, voltage = variables["Current [A]"], variables["Voltage [V]"]
        mode = pybamm.InputParameter(MODE)
        requested = current + data["maximum_current_A"] * (voltage - pybamm.InputParameter(VOLTAGE))
        limited = pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], requested)
        )
        return (current - (1 - mode) * pybamm.InputParameter(CURRENT) - mode * limited) / data[
            "maximum_current_A"
        ]

    if basic:
        model = pybamm.lithium_ion.BasicDFNHalfCell()
    else:
        # The pinned package does not export pybamm.lithium_metal. Import its
        # actual released class, rather than substituting an unrelated model.
        from pybamm.models.full_battery_models.lithium_metal.dfn import DFN

        model = DFN(
            {
                "surface form": "false",
                "thermal": "isothermal",
                "particle": "Fickian diffusion",
                "open-circuit potential": "single",
                "SEI": "none",
                "lithium plating": "none",
                "contact resistance": "false",
                "current collector": "uniform",
                "dimensionality": 0,
                "calculate discharge energy": "true",
                "operating mode": finite_source,
            }
        )
    original_events = tuple((event.name, event.event_type.name) for event in model.events)
    original_rhs = tuple(variable.name for variable in model.rhs)
    original_voltage = model.variables["Voltage [V]"]
    original_source_ocp = model.variables["Positive electrode open-circuit potential [V]"]
    released = pybamm.parameter_sets["Xu2019"]
    parameters = pybamm.ParameterValues("Xu2019")
    updates, descriptions, control_updates = {}, {}, {}
    original_metal_law = parameters["Exchange-current density for lithium metal electrode [A.m-2]"]
    cref, temperature = data["ce_reference_mol_m3"], data["temperature_K"]
    pure_metal_concentration = 1 / parameters["Lithium metal partial molar volume [m3.mol-1]"]
    working_cmax = parameters["Maximum concentration in positive electrode [mol.m-3]"]
    if data["metal_thickness_m"] != parameters["Negative electrode thickness [m]"]:
        updates["Negative electrode thickness [m]"] = data["metal_thickness_m"]
        descriptions["Negative electrode thickness [m]"] = {
            "scope": "explicit thin-foil scenario; not released geometry",
            "original": parameters["Negative electrode thickness [m]"],
            "value": data["metal_thickness_m"],
        }
    if constant:
        for name in ("Electrolyte diffusivity [m2.s-1]", "Electrolyte conductivity [S.m-1]"):
            frozen = float(parameters[name](cref, temperature))
            updates[name] = frozen
            descriptions[name] = {
                "scope": "named constant-transport source variant",
                "original_function_id": parameters[name].__name__,
                "value": frozen,
                "reference_ce_mol_m3": cref,
                "reference_temperature_K": temperature,
            }
    if normalized:

        def pure_metal_normalization(c_e, c_Li_source_argument, T):
            # Basic passes cmax_p, while modular passes 1/Vbar_Li. Preserve
            # the original Xu law and replace only its explicitly recorded
            # second argument with the pure-phase concentration.
            return original_metal_law(c_e, pure_metal_concentration, T)

        key = "Exchange-current density for lithium metal electrode [A.m-2]"
        updates[key] = pure_metal_normalization
        descriptions[key] = {
            "scope": "Basic explicit pure-metal argument normalization",
            "original_function_id": original_metal_law.__name__,
            "source_passed_argument_mol_m3": working_cmax,
            "normalized_argument_mol_m3": pure_metal_concentration,
            "formula": "original_Xu2019_j0(c_e,1/Vbar_Li,T); no voltage compensation",
        }
    if basic:
        updates["Current function [A]"] = pybamm.InputParameter(CURRENT)
        control_updates["Current function [A]"] = {
            "original": parameters["Current function [A]"],
            "input_parameter": CURRENT,
        }
    parameters.update(updates, check_already_exists=False)
    p = model.param
    T = p.T_init
    ce = model.variables["Electrolyte concentration [mol.m-3]"]
    ce_p = model.variables["Positive electrolyte concentration [mol.m-3]"]
    cs = model.variables["Positive particle concentration [mol.m-3]"]
    surface = model.variables["Positive particle surface concentration [mol.m-3]"]
    phi_s = model.variables["Positive electrode potential [V]"]
    phi_e = model.variables["Electrolyte potential [V]"]
    phi_e_p = model.variables["Positive electrolyte potential [V]"]
    current, voltage = model.variables["Current [A]"], original_voltage
    theta = surface / p.p.prim.c_max
    eps = pybamm.concatenation(
        pybamm.PrimaryBroadcast(pybamm.Parameter("Separator porosity"), "separator"),
        pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity"), "positive electrode"
        ),
    )
    tor = pybamm.concatenation(
        pybamm.PrimaryBroadcast(pybamm.Parameter("Separator porosity") ** p.s.b_e, "separator"),
        pybamm.PrimaryBroadcast(
            pybamm.Parameter("Positive electrode porosity") ** p.p.b_e, "positive electrode"
        ),
    )
    if basic:
        j0 = p.p.prim.j0(ce_p, surface, T)
        eta = phi_s - phi_e_p - original_source_ocp
        j = 2 * j0 * pybamm.sinh(p.p.prim.ne * p.F / (2 * p.R * T) * eta)
        ionic = (
            p.kappa_e(ce, T) * tor * (p.chiRT_over_Fc(ce, T) * pybamm.grad(ce) - pybamm.grad(phi_e))
        )
        sigma_eff = (
            p.p.sigma(T)
            * pybamm.Parameter("Positive electrode active material volume fraction") ** p.p.b_s
        )
        solid = -sigma_eff * pybamm.grad(phi_s)
        metal_j0 = model.variables["Negative electrode exchange current density [A.m-2]"]
        metal_eta = model.variables["Negative electrode reaction overpotential [V]"]
        metal_drop = model.variables["Negative electrode potential drop [V]"]
        metal_phi_e = model.boundary_conditions[phi_e]["left"][0]
        work = pybamm.Variable(BASIC_WORK)
        model.rhs[work] = current * voltage
        model.initial_conditions[work] = pybamm.Scalar(0)
        model.variables[BASIC_WORK] = work
    else:
        eps = model.variables["Porosity"]
        tor = model.variables["Electrolyte transport efficiency"]
        j0 = model.variables["Positive electrode exchange current density [A.m-2]"]
        eta = model.variables["Positive electrode reaction overpotential [V]"]
        j = model.variables["Positive electrode interfacial current density [A.m-2]"]
        ionic = model.variables["Electrolyte current density [A.m-2]"]
        solid = model.variables["Positive electrode current density [A.m-2]"]
        metal_j0 = model.variables["Lithium metal interface exchange current density [A.m-2]"]
        metal_eta = model.variables["Lithium metal interface reaction overpotential [V]"]
        metal_drop = model.variables["Negative electrode potential drop [V]"]
        metal_phi_e = model.variables["Lithium metal interface electrolyte potential [V]"]
        work = 3600 * model.variables["Discharge energy [W.h]"]
    diffusion_flux = (
        -tor * p.D_e(ce, T) * pybamm.grad(ce)
        if basic
        else model.variables["Electrolyte diffusion flux [mol.m-2.s-1]"]
    )
    cation_flux = (
        diffusion_flux + p.t_plus(ce, T) * ionic / p.F
        if basic
        else model.variables["Electrolyte flux [mol.m-2.s-1]"]
    )
    area = parameters["Electrode height [m]"] * parameters["Electrode width [m]"]
    n_m0 = (
        area
        * data["metal_thickness_m"]
        / parameters["Lithium metal partial molar volume [m3.mol-1]"]
    )
    metal = n_m0 - 3600 * model.variables["Discharge capacity [A.h]"] / p.F
    aliases = {
        "voltage_V": voltage,
        "current_A": current,
        "charge_C": 3600 * model.variables["Discharge capacity [A.h]"],
        "work_J": work,
        "metal_Li_mol": metal,
        "metal_boundary_ce_mol_m3": pybamm.boundary_value(ce, "left"),
        "metal_j0_A_m2": metal_j0,
        "metal_eta_V": metal_eta,
        "metal_ohmic_V": metal_drop,
        "metal_phi_e_V": metal_phi_e,
        "ce": ce,
        "cs": cs,
        "theta_surface": theta,
        "phi_e": phi_e,
        "phi_s": phi_s,
        "ionic_current_A_m2": ionic,
        "solid_current_A_m2": solid,
        "positive_j_A_m2": j,
        "positive_j0_A_m2": j0,
        "positive_eta_V": eta,
        "positive_ocp_V": original_source_ocp,
        "diffusive_salt_flux_mol_m2_s": diffusion_flux,
        "total_cation_flux_mol_m2_s": cation_flux,
        "source_reported_positive_Li_mol": model.variables[
            "Total lithium in positive electrode [mol]"
        ],
        "source_reported_electrolyte_Li_mol": model.variables["Total lithium in electrolyte [mol]"],
        "eps_ce": eps * ce,
    }
    model.variables.update({PREFIX + name: expression for name, expression in aliases.items()})
    # The source uses Lx^2 as a charge-equation preconditioner, where Lx
    # includes the thick metal even though both differential operators act on
    # separator+positive only. Keep its original expressions and apply one
    # declared positive row-scaling policy to every source route. The physical
    # equation, its zero set and all material/control/event expressions remain.
    salt_length = (
        parameters["Separator thickness [m]"] + parameters["Positive electrode thickness [m]"]
    )
    row_factor = float((salt_length / (data["metal_thickness_m"] + salt_length)) ** 2)
    if not np.isfinite(row_factor) or row_factor <= 0:
        raise ValueError("source charge-equation conditioning requires a positive row factor")
    conditioning = []
    for ordinal, (variable, expression) in enumerate(tuple(model.algebraic.items())):
        factor = row_factor if "potential" in variable.name else 1.0
        original_name = PREFIX + f"original-algebraic:{ordinal}"
        conditioned_name = PREFIX + f"conditioned-algebraic:{ordinal}"
        conditioned = factor * expression
        model.variables[original_name] = expression
        model.variables[conditioned_name] = conditioned
        model.algebraic[variable] = conditioned
        conditioning.append(
            {
                "name": variable.name,
                "original_alias": original_name,
                "conditioned_alias": conditioned_name,
                "factor": factor,
                "charge_equation": "potential" in variable.name,
                "original_continuous_expression": str(expression),
                "conditioned_continuous_expression": str(conditioned),
            }
        )
    scale = data["common_event_scale"]
    stripping = current > 0
    metal_distance = stripping * metal + (1 - stripping) * n_m0
    model.events.extend(
        (
            pybamm.Event(PREFIX + "finite-metal-depleted", scale * metal_distance / n_m0),
            pybamm.Event(
                PREFIX + "working-minimum-surface",
                scale * (pybamm.min(theta) - data["surface_margin"]),
            ),
            pybamm.Event(
                PREFIX + "working-maximum-surface",
                scale * (1 - data["surface_margin"] - pybamm.max(theta)),
            ),
            pybamm.Event(
                PREFIX + "salt-floor",
                scale
                * (
                    pybamm.min(pybamm.minimum(pybamm.min(ce), pybamm.boundary_value(ce, "left")))
                    - data["minimum_concentration_mol_m3"]
                )
                / cref,
            ),
        )
    )
    initial_updates = []
    if data["initial_salt_profile"] != "uniform":
        projection = XU2019_PROJECTION
        lengths = np.asarray(projection["lengths_m"])
        porosities = np.asarray(projection["porosities"])
        starts = np.r_[0.0, np.cumsum(lengths)[:-1]]
        mean_x = float(
            np.sum(porosities * lengths * (starts + lengths / 2)) / np.sum(porosities * lengths)
        )

        def profile(x):
            return cref + data["initial_salt_direction"] * data["initial_salt_span_mol_m3"] * (
                x - data["metal_thickness_m"] - mean_x
            ) / sum(lengths)

        profiles = (
            profile(pybamm.standard_spatial_vars.x_s),
            profile(pybamm.standard_spatial_vars.x_p),
        )
        key = next(
            v
            for v in model.rhs
            if v.name
            == (
                "Electrolyte concentration [mol.m-3]"
                if basic
                else "Porosity times concentration [mol.m-3]"
            )
        )
        model.initial_conditions[key] = pybamm.concatenation(
            *(
                profiles
                if basic
                else (
                    pybamm.Parameter("Separator porosity") * profiles[0],
                    pybamm.Parameter("Positive electrode porosity") * profiles[1],
                )
            )
        )
        initial_updates.append(
            {
                "variable": key.name,
                "scope": "declared spatial initial salt profile; "
                "scalar reference parameter remains 1000",
                "formula": "ce=1000+direction*span*(x_local-pore_weighted_mean_x)/(Ls+Lp)",
                "span_mol_m3": data["initial_salt_span_mol_m3"],
                "direction": data["initial_salt_direction"],
            }
        )
    assert model.variables["Voltage [V]"] is original_voltage
    assert model.variables["Positive electrode open-circuit potential [V]"] is original_source_ocp
    var_pts = {
        var: (
            grid["r"]
            if str(var).startswith("r_")
            else grid["x"][0]
            if str(var) == "x_s"
            else grid["x"][1]
            if str(var) == "x_p"
            else 1
        )
        for var in model.default_var_pts
    }
    simulation = pybamm.Simulation(
        model,
        parameter_values=parameters,
        var_pts=var_pts,
        solver=pybamm.IDAKLUSolver(
            rtol=RTOL, atol=ATOL, root_tol=ROOT_TOL, root_method=SOURCE_ROOT_METHOD
        ),
    )
    simulation.build()
    return {
        "simulation": simulation,
        "model": simulation.built_model,
        "parameter_values": parameters,
        "case": deepcopy(data),
        "grid": deepcopy(grid),
        "variant": variant,
        "basic": basic,
        "constant_transport": constant,
        "pure_metal_normalized": normalized,
        "original_events": original_events,
        "original_rhs_names": original_rhs,
        "source_parameter_manifest": _manifest(released),
        "released_parameter_factory_source": inspect.getsource(
            inspect.getmodule(original_metal_law).get_parameter_values
        ),
        "material_parameter_updates": descriptions,
        "control_parameter_updates": control_updates,
        "initial_condition_updates": initial_updates,
        "metal_initial_mol": n_m0,
        "algebraic_conditioning": conditioning,
        "metal_argument_contract": {
            "raw_basic_argument_mol_m3": working_cmax,
            "modular_pure_metal_argument_mol_m3": pure_metal_concentration,
            "raw_basic_j0_at_reference_A_m2": float(
                original_metal_law(cref, working_cmax, temperature).evaluate()
            ),
            "pure_metal_j0_at_reference_A_m2": float(
                original_metal_law(cref, pure_metal_concentration, temperature).evaluate()
            ),
            "route_uses_raw_basic_argument": basic and not normalized,
        },
    }


def _read(expression, time_s, state, inputs):
    return np.asarray(expression.evaluate(t=time_s, y=state, inputs=inputs), dtype=float)


def _scalar(expression, time_s, state, inputs):
    values = _read(expression, time_s, state, inputs)
    if values.size != 1:
        raise ValueError("source scalar retained unresolved spatial dimensions")
    return float(values.ravel()[0])


def _mesh_record(mesh, *, origin=0.0):
    return {
        "class": type(mesh).__name__,
        "cells": mesh.npts,
        "original_nodes_m": np.asarray(mesh.nodes).tolist(),
        "original_edges_m": np.asarray(mesh.edges).tolist(),
        "local_nodes_m": (np.asarray(mesh.nodes) - origin).tolist(),
        "local_edges_m": (np.asarray(mesh.edges) - origin).tolist(),
        "origin_m": origin,
    }


def _work_observer_dependencies(built):
    if not built["basic"]:
        return {"added_observer": False, "scope": "original calculate-discharge-energy state"}
    pybamm = _dependencies()
    model = built["model"]
    variable = next(v for v in model.rhs if v.name == BASIC_WORK)
    indices = [i for part in model.y_slices[variable] for i in range(part.start, part.stop)]
    expressions = [("rhs:" + v.name, expr) for v, expr in model.rhs.items() if v.name != BASIC_WORK]
    expressions += [("algebraic:" + v.name, expr) for v, expr in model.algebraic.items()]
    expressions += [(f"event:{i}:{e.name}", e.expression) for i, e in enumerate(model.events)]
    checked = {}
    for name, expression in expressions:
        reads = set()
        for node in expression.pre_order():
            if isinstance(node, pybamm.StateVector):
                reads.update(
                    i
                    for i in indices
                    if i < len(node.evaluation_array) and node.evaluation_array[i]
                )
        checked[name] = sorted(reads)
    return {
        "added_observer": True,
        "state": BASIC_WORK,
        "indices": indices,
        "rhs": "I*original Voltage; signed J",
        "checked_original_expressions": checked,
        "no_feedback": not any(checked.values()),
    }


def source_contract(built):
    pybamm = _dependencies()
    sim, model, parameters = (built[k] for k in ("simulation", "model", "parameter_values"))
    origin = built["case"]["metal_thickness_m"]
    mesh = {
        domain: _mesh_record(sim.mesh[domain], origin=0.0 if "particle" in domain else origin)
        for domain in ("separator", "positive electrode", "positive particle")
    }
    layout = []
    for variable, slices in model.y_slices.items():
        if not slices:
            continue
        layout.append(
            {
                "name": variable.name,
                "domains": variable.domains,
                "slices": [[s.start, s.stop] for s in slices],
                "scale": float(parameters.evaluate(variable.scale)),
                "reference": float(parameters.evaluate(variable.reference)),
                "concatenation_view": isinstance(variable, pybamm.Concatenation),
                "kind": "differential" if slices[0].start < model.len_rhs else "algebraic",
            }
        )
    area = parameters["Electrode height [m]"] * parameters["Electrode width [m]"]
    widths = np.r_[
        np.diff(sim.mesh["separator"].edges), np.diff(sim.mesh["positive electrode"].edges)
    ]
    ns, np_ = built["grid"]["x"]
    pore_volumes = (
        area
        * widths
        * np.r_[
            np.full(ns, parameters["Separator porosity"]),
            np.full(np_, parameters["Positive electrode porosity"]),
        ]
    )
    salt_length = float(np.sum(widths))
    total_length = origin + salt_length
    conditioning = []
    for declaration in built["algebraic_conditioning"]:
        original = model.variables[declaration["original_alias"]]
        conditioned = model.variables[declaration["conditioned_alias"]]
        conditioning.append(
            {
                **declaration,
                "rows": int(original.size),
                "actual_row_factors": [declaration["factor"]] * int(original.size),
                "original_discretized_expression": str(original),
                "conditioned_discretized_expression": str(conditioned),
            }
        )
    return {
        "implementation": "actual PyBaMM25.4.2 "
        + (
            "lithium_ion.BasicDFNHalfCell"
            if built["basic"]
            else "models.full_battery_models.lithium_metal.dfn.DFN"
        ),
        "variant": built["variant"],
        "pybamm_version": PYBAMM_VERSION,
        "pybammsolvers_version": PYBAMMSOLVERS_VERSION,
        "casadi_version": CASADI_VERSION,
        "source_name": "Xu2019",
        "source_parameter_manifest": built["source_parameter_manifest"],
        "released_parameter_factory_source": built["released_parameter_factory_source"],
        "material_parameter_updates": built["material_parameter_updates"],
        "control_parameter_updates": built["control_parameter_updates"],
        "initial_condition_updates": built["initial_condition_updates"],
        "metal_argument_contract": built["metal_argument_contract"],
        "original_events": [{"name": n, "type": t} for n, t in built["original_events"]],
        "all_events": [
            {"ordinal": i, "name": e.name, "type": e.event_type.name}
            for i, e in enumerate(model.events)
        ],
        "original_rhs_names": built["original_rhs_names"],
        "rhs_size": int(model.len_rhs),
        "algebraic_size": int(model.len_alg),
        "state_layout": layout,
        "algebraic_conditioning": {
            "status": "explicit numerical policy; not unmodified source default numerics",
            "policy": "all charge rows multiply [(Ls+Lp)/(Lm+Ls+Lp)]^2; all other rows 1",
            "positive_factors_preserve_zero_set": True,
            "root_method": SOURCE_ROOT_METHOD,
            "root_tol_applies_to": "conditioned source residual; raw original always retained",
            "blocks": conditioning,
        },
        "mesh": mesh,
        "actual_FV_options": {
            domain: deepcopy(sim._disc.spatial_methods[domain].options)
            for domain in ("separator", "positive electrode", "positive particle")
        },
        "discrete_interface_scope": "actual source FV uses its recorded boundary extrapolation; "
        "native uses its own flux-aware FV surface reconstruction; all source fields and t0 "
        "are retained without compensating this mesh-dependent difference",
        "original_ocp_and_barrier_preserved": True,
        "endpoint_contract": {
            "source": "ParticleLithiumIonParameters.U",
            "clip": float(pybamm.settings.tolerances["U__c_s"]),
            "barrier_V": 1e-6,
        },
        "work_contract": _work_observer_dependencies(built),
        "finite_metal_contract": {
            "state": "not an upstream dynamic inventory",
            "formula": "A*Lm/Vbar-3600*Discharge capacity[A.h]/F",
            "event": "positive stripping current only; added qualification event",
            "initial_mol": built["metal_initial_mol"],
        },
        "control_contract": {
            "Basic": "prescribed source current only",
            "modular": "[I-(1-mode)*Icc-mode*clip(I+Imax*(V-Vtarget),-Imax,Imax)]/Imax=0",
            "maximum_current_A": built["case"]["maximum_current_A"],
        },
        "reported_inventory_scope": "original source reported totals retained alongside "
        "independent physical FV inventories; neither replaces the other",
        "inventory_geometry_contract": {
            "area_m2": area,
            "metal_length_m": origin,
            "actual_salt_domain_length_m": salt_length,
            "source_Lx_m": total_length,
            "actual_salt_cell_widths_m": widths.tolist(),
            "actual_salt_pore_volumes_m3": pore_volumes.tolist(),
            "physical_salt_formula": "sum(ce_i*eps_i*A*diff(actual_mesh_edges)_i)",
            "source_reported_salt_formula": "A*Lx*x_average(eps*ce); "
            "actual x_average integrates only separator+positive then divides by Ls+Lp",
            "source_reported_to_physical_salt_factor": total_length / salt_length,
            "qualification_uses": "physical FV salt inventory; original source total is raw only",
            "source_concentration_scale_mol_m3": float(
                parameters["Initial concentration in electrolyte [mol.m-3]"]
            ),
            "source_charge_conditioning": "charge-conservation residuals use Lx^2; "
            "this changes conditioning, not their zero set; mesh derivatives use actual widths",
        },
        "F_C_mol": float(pybamm.constants.F.value),
        "R_J_mol_K": float(pybamm.constants.R.value),
        "native_F_C_mol": F,
        "response_order": RESPONSE_ORDER,
    }


def state_record(built, time_s, state, inputs):
    model, sim, parameters = (built[k] for k in ("model", "simulation", "parameter_values"))
    raw = np.asarray(state, dtype=float).reshape(-1, 1)

    def value(name):
        return _read(model.variables[PREFIX + name], time_s, raw, inputs).ravel()

    scalar = {
        name: _scalar(model.variables[PREFIX + name], time_s, raw, inputs)
        for name in RESPONSE_ORDER
        if name not in {"positive_Li_mol", "electrolyte_salt_mol"}
    }
    ns, np_ = built["grid"]["x"]
    nr = built["grid"]["r"]
    ce, cs = value("ce"), value("cs")
    if ce.size != ns + np_ or cs.size != nr * np_:
        raise ValueError("actual source x/r layout differs from its declared grid")
    area = parameters["Electrode height [m]"] * parameters["Electrode width [m]"]
    widths = np.r_[
        np.diff(sim.mesh["separator"].edges), np.diff(sim.mesh["positive electrode"].edges)
    ]
    pore_volumes = (
        area
        * widths
        * np.r_[
            np.full(ns, parameters["Separator porosity"]),
            np.full(np_, parameters["Positive electrode porosity"]),
        ]
    )
    salt_amounts = ce * pore_volumes
    radial_edges = np.asarray(sim.mesh["positive particle"].edges)
    radial_weights = np.diff((radial_edges / parameters["Positive particle radius [m]"]) ** 3)
    active_volumes = (
        area
        * np.diff(sim.mesh["positive electrode"].edges)
        * parameters["Positive electrode active material volume fraction"]
    )
    csp = cs.reshape(np_, nr)
    shell_amounts = csp * active_volumes[:, None] * radial_weights[None, :]
    scalar.update(
        positive_Li_mol=float(np.sum(shell_amounts)),
        electrolyte_salt_mol=float(np.sum(salt_amounts)),
    )
    coefficient_ranges = {}
    for key, label in (
        ("Electrolyte diffusivity [m2.s-1]", "D_e_range_m2_s"),
        ("Electrolyte conductivity [S.m-1]", "kappa_range_S_m"),
    ):
        declaration = parameters[key]
        values = np.asarray(
            declaration(ce, built["case"]["temperature_K"])
            if callable(declaration)
            else declaration,
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("source transport law returned a nonfinite physical coefficient")
        coefficient_ranges[label] = [float(np.min(values)), float(np.max(values))]
    current_densities = {
        name: value(name)
        for name in ("ionic_current_A_m2", "solid_current_A_m2", "positive_j_A_m2")
    }
    positive_q = (
        current_densities["positive_j_A_m2"]
        * 3
        * active_volumes
        / parameters["Positive particle radius [m]"]
    )
    current_closure = {
        "ionic_cell_A": area * np.diff(current_densities["ionic_current_A_m2"])
        - np.r_[np.zeros(ns), positive_q],
        "solid_cell_A": area * np.diff(current_densities["solid_current_A_m2"]) + positive_q,
        "ionic_left_A": area * current_densities["ionic_current_A_m2"][0] - scalar["current_A"],
        "ionic_right_A": area * current_densities["ionic_current_A_m2"][-1],
        "solid_left_A": area * current_densities["solid_current_A_m2"][0],
        "solid_right_A": area * current_densities["solid_current_A_m2"][-1] - scalar["current_A"],
        "reaction_sum_plus_current_A": np.sum(positive_q) + scalar["current_A"],
    }
    return {
        "time_s": float(time_s),
        "row": [scalar[name] for name in RESPONSE_ORDER],
        "source_y": raw.ravel().tolist(),
        "source_rhs": _read(model.concatenated_rhs, time_s, raw, inputs).ravel().tolist(),
        "source_algebraic": _read(model.concatenated_algebraic, time_s, raw, inputs)
        .ravel()
        .tolist(),
        "source_algebraic_original": np.concatenate(
            [
                _read(model.variables[block["original_alias"]], time_s, raw, inputs).ravel()
                for block in built["algebraic_conditioning"]
            ]
        ).tolist(),
        "physical_current_closure": {
            "formula": "ionic A*diff(i_e)-q; solid A*diff(i_s)+q; "
            "q=actual j*3*eps_active*A*dx/R; evaluated independently of row scaling",
            "residuals_A": {
                name: np.asarray(residual).tolist() for name, residual in current_closure.items()
            },
            "maximum_absolute_A": float(
                max(np.max(np.abs(residual)) for residual in current_closure.values())
            ),
        },
        "source_events": [
            {
                "ordinal": i,
                "name": e.name,
                "type": e.event_type.name,
                "value": _read(e.expression, time_s, raw, inputs).ravel().tolist(),
            }
            for i, e in enumerate(model.events)
        ],
        "fields": {
            **{
                name: value(name).tolist()
                for name in (
                    "theta_surface",
                    "phi_e",
                    "phi_s",
                    "positive_j0_A_m2",
                    "positive_eta_V",
                    "positive_ocp_V",
                    "diffusive_salt_flux_mol_m2_s",
                    "total_cation_flux_mol_m2_s",
                )
            },
            **{name: arr.tolist() for name, arr in current_densities.items()},
            "ce_mol_m3": ce.tolist(),
            "cs_mol_m3_x_r": csp.tolist(),
            "theta_x_r": (
                csp / parameters["Maximum concentration in positive electrode [mol.m-3]"]
            ).tolist(),
            "electrolyte_salt_amounts_mol": salt_amounts.tolist(),
            "positive_shell_amounts_mol_x_r": shell_amounts.tolist(),
            "positive_q_cell_A": positive_q.tolist(),
            "metal_phi_e_V": float(value("metal_phi_e_V")[0]),
        },
        "original_source_reported_inventory": {
            name: float(value(name)[0])
            for name in ("source_reported_positive_Li_mol", "source_reported_electrolyte_Li_mol")
        },
        "native_projection_difference": coefficient_ranges,
    }


def initial_state_record(built, stage, *, time_s=0.0, previous_state=None):
    """Reclose the actual upstream algebraic equations at exact t0, without advance."""
    sim, model = built["simulation"], built["model"]
    inputs = stage_inputs(stage, built["variant"])
    sim.solver.set_up(model, inputs=inputs)
    initial = (
        np.asarray(model.concatenated_initial_conditions.evaluate(inputs=inputs), dtype=float)
        if previous_state is None
        else np.asarray(previous_state, dtype=float)
    )
    model.y0 = initial.reshape(-1, 1)
    try:
        result = np.asarray(
            sim.solver.calculate_consistent_state(model, time=time_s, inputs=inputs), dtype=float
        ).reshape(-1, 1)
    except Exception as exc:
        exc.source_boundary_attempt = {
            "time_s": float(time_s),
            "inputs": inputs,
            "initial_source_y": initial.ravel().tolist(),
            "initial_source_algebraic": _read(
                model.concatenated_algebraic, time_s, model.y0, inputs
            )
            .ravel()
            .tolist(),
            "root_solver": sim.solver.root_method.name,
            "root_tol": sim.solver.root_tol,
            "failed_final_trial": "not exposed by the upstream failed root solver",
        }
        raise
    record = state_record(built, time_s, result, inputs)
    record["differential_state_unchanged_by_boundary_root"] = bool(
        np.array_equal(result[: model.len_rhs], initial.reshape(-1, 1)[: model.len_rhs])
    )
    record["boundary_operation"] = "actual source algebraic consistency only; no integration"
    return record


def reference_time(data, grid, variant="modular-full-Xu2019", *, progress_callback=None):
    """Execute one requested route only; a separate coordinator owns formal GO."""
    built = build_reference(data, grid, variant)
    contract = source_contract(built)
    sim, segments = built["simulation"], []
    previous_time, previous_state = 0.0, None
    for index, stage in enumerate(data["stages"]):
        boundary = None
        pending_source = None
        try:
            boundary = initial_state_record(
                built, stage, time_s=previous_time, previous_state=previous_state
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "source-boundary-reclosed",
                        "stage": index,
                        "segments": segments,
                        "boundary_t0": boundary,
                    }
                )
            inputs = stage_inputs(stage, variant)
            times = np.asarray(stage_times(stage))
            # In pinned IDAKLU, each t_eval is a mandatory stop followed by
            # reinitialisation. The control is constant within this stage:
            # only its two boundaries belong there. Observations use t_interp
            # and must not introduce extra consistency solves or restarts.
            restart_times = times[[0, -1]]
            solution = sim.step(
                stage["duration_s"],
                save=False,
                t_eval=restart_times,
                t_interp=times,
                inputs=inputs,
            )
            actual_t, actual_y = np.asarray(solution.t), np.asarray(solution.y, dtype=float)
            # Preserve the actual source return before deriving any fields. An
            # extraction failure must not erase a successfully returned history.
            pending_source = {
                "times_s": actual_t.tolist(),
                "source_y": actual_y.tolist(),
                "all_ts": [np.asarray(t).tolist() for t in solution.all_ts],
                "all_ys": [np.asarray(y, dtype=float).tolist() for y in solution.all_ys],
                "termination": solution.termination,
                "t_event": None
                if solution.t_event is None
                else np.asarray(solution.t_event).tolist(),
                "y_event": None
                if solution.y_event is None
                else np.asarray(solution.y_event).tolist(),
            }
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "source-solution-returned",
                        "stage": index,
                        "segments": segments,
                        "boundary_t0": boundary,
                        "pending_source_solution": pending_source,
                    }
                )
            if actual_y.shape != (built["model"].len_rhs + built["model"].len_alg, actual_t.size):
                raise ValueError("source original returned y/t layout is incomplete")
            record = {
                "index": index,
                "request": deepcopy(stage),
                "inputs": inputs,
                "requested_start_time_s": previous_time,
                "requested_end_time_s": previous_time + stage["duration_s"],
                "requested_relative_times_s": stage_times(stage),
                "boundary_t0": boundary,
                "original_reported_times_s": actual_t.tolist(),
                "original_all_ts": [np.asarray(t).tolist() for t in solution.all_ts],
                "original_all_ys": [np.asarray(y, dtype=float).tolist() for y in solution.all_ys],
                "original_t_event": None
                if solution.t_event is None
                else np.asarray(solution.t_event).tolist(),
                "original_y_event": None
                if solution.y_event is None
                else np.asarray(solution.y_event).tolist(),
                "records": [
                    state_record(built, t, actual_y[:, i : i + 1], inputs)
                    for i, t in enumerate(actual_t)
                ],
                "termination": solution.termination,
                "actual_start_time_s": float(actual_t[0]),
                "actual_end_time_s": float(actual_t[-1]),
                "sampling_scope": "all original exposed solution states/all_ts/all_ys; "
                "unexposed internal integrator steps are unavailable; exact t0 retained separately",
                "solver": {
                    "name": "PyBaMM25.4.2 IDAKLUSolver",
                    "rtol": RTOL,
                    "atol": ATOL,
                    "root_tol": ROOT_TOL,
                    "root_method": SOURCE_ROOT_METHOD,
                    "stage_relative_restart_times_s": restart_times.tolist(),
                    "stage_relative_interpolation_times_s": times.tolist(),
                },
            }
            segments.append(record)
            previous_time, previous_state = float(actual_t[-1]), actual_y[:, -1:]
            if progress_callback is not None:
                progress_callback(
                    {"phase": "source-stage-captured", "stage": index, "segments": segments}
                )
            if solution.termination != "final time":
                break
        except Exception as exc:
            exc.partial_history = {
                "contract": contract,
                "case": deepcopy(data),
                "segments": segments,
                "failed_stage": index,
                "failed_request": deepcopy(stage),
                "failed_boundary_t0": boundary,
                "failed_source_boundary_attempt": getattr(exc, "source_boundary_attempt", None),
                "failed_source_return": pending_source,
                "last_completed_time_s": previous_time,
            }
            raise
    return {
        "contract": contract,
        "case": deepcopy(data),
        "segments": segments,
        "requested_stages": len(data["stages"]),
        "actual_stages": len(segments),
    }
