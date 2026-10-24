"""Actual optional WaterTAP 1.8.0 CPHSDM constraints, not a copied evaluator.

Only the named GAC subsystem is solved. Identical C/Q/rho/mu/D are supplied as
property boundary values; MCAS's different volume/enthalpy closures are excluded.
Native design output never seeds or enters this reference calculation.
"""

from importlib.metadata import version
from math import isfinite


def upstream(spec, feed):
    from idaes.core import FlowsheetBlock, MaterialFlowBasis
    from pyomo.common.errors import IterationLimitError
    from pyomo.core.expr.calculus.derivatives import differentiate
    from pyomo.environ import ConcreteModel, value
    from pyomo.util.calc_var_value import calculate_variable_from_constraint
    from watertap.property_models.multicomp_aq_sol_prop_pack import MCASParameterBlock
    from watertap.unit_models.gac import (
        GAC,
        FilmTransferCoefficientType,
        SurfaceDiffusionCoefficientType,
    )

    if version("watertap") != "1.8.0":
        raise ValueError("49E reference is pinned to actual WaterTAP 1.8.0")
    transport, bed, empirical = spec.transport, spec.bed, spec.empirical
    model = ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=False)
    model.fs.properties = MCASParameterBlock(
        solute_list=["target"],
        mw_data={"H2O": 0.01801528, "target": 0.1},
        charge={"target": 0},
        material_flow_basis=MaterialFlowBasis.mass,
        diffusivity_data={("Liq", "target"): transport.liquid_diffusivity_m2_s or 1e-9},
    )
    model.fs.unit = GAC(
        property_package=model.fs.properties,
        film_transfer_coefficient_type=(
            FilmTransferCoefficientType.fixed
            if transport.film_transfer_m_s is not None
            else FilmTransferCoefficientType.calculated
        ),
        surface_diffusion_coefficient_type=(
            SurfaceDiffusionCoefficientType.fixed
            if transport.surface_diffusion_m2_s is not None
            else SurfaceDiffusionCoefficientType.calculated
        ),
        finite_elements_ss_approximation=spec.breakthrough_elements,
    )
    unit = model.fs.unit
    inlet = unit.process_flow.properties_in[0]
    inlet.temperature.set_value(feed.temperature_K)
    inlet.pressure.set_value(feed.pressure_Pa)
    inlet.flow_mass_phase_comp["Liq", "target"].set_value(
        feed.volumetric_flow_m3_s * feed.concentration_kg_m3,
    )
    inlet.flow_mass_phase_comp["Liq", "H2O"].set_value(
        feed.volumetric_flow_m3_s * (feed.liquid_density_kg_m3 - feed.concentration_kg_m3),
    )
    inlet.conc_mass_phase_comp["Liq", "target"].set_value(feed.concentration_kg_m3)
    inlet.flow_vol_phase["Liq"].set_value(feed.volumetric_flow_m3_s)
    inlet.dens_mass_phase["Liq"].set_value(feed.liquid_density_kg_m3)
    model.fs.properties.visc_d_phase["Liq"].set_value(feed.dynamic_viscosity_Pa_s)
    for key, v in {
        "freund_k": empirical.freundlich_k,
        "freund_ninv": empirical.freundlich_exponent,
        "particle_dens_app": bed.apparent_density_kg_m3,
        "particle_dia": bed.particle_diameter_m,
        "bed_voidage": bed.void_fraction,
        "ebct": bed.empty_bed_contact_time_s,
        "velocity_sup": bed.superficial_velocity_m_s,
        "conc_ratio_replace": spec.replacement_concentration_ratio,
        **{f"a{i}": v for i, v in enumerate(empirical.stanton_coefficients)},
        **{f"b{i}": v for i, v in enumerate(empirical.throughput_coefficients)},
    }.items():
        getattr(unit, key).set_value(v)
    constraints = []
    solver_messages = []

    def equation_state(variable, constraint):
        lhs, rhs = constraint.expr.args
        expression = lhs - rhs
        a, b = float(value(lhs)), float(value(rhs))
        derivative = float(differentiate(expression, wrt=variable))
        scale = max(abs(a), abs(b), abs(derivative * value(variable)), 1e-30)
        return a - b, derivative, scale

    def solve(variable, constraint):
        try:
            calculate_variable_from_constraint(variable, constraint, eps=1e-14)
        except IterationLimitError as exc:
            solver_messages.append({"variable": variable.name, "message": str(exc)})
        # Refine the actual expression using its Pyomo derivative. Absolute
        # stopping at 1e-14 alone is inaccurate for small diffusion/flux terms.
        # Derivative*variable also supplies a physical scale when the upstream
        # equation is encoded as 0 == net balance rather than lhs == rhs.
        for _ in range(12):
            residual, derivative, scale = equation_state(variable, constraint)
            if abs(residual) <= 1e-12 * scale:
                break
            if not isfinite(derivative) or derivative == 0:
                raise RuntimeError(f"singular upstream equation {constraint.name}")
            variable.set_value(value(variable) - residual / derivative)
        residual, _, scale = equation_state(variable, constraint)
        if not isfinite(residual) or abs(residual) > 1e-8 * scale:
            raise RuntimeError(f"upstream equation failed {constraint.name}: {residual / scale}")
        constraints.append((variable, constraint))

    solve(unit.equil_conc, unit.eq_equilibrium_concentration[0, "target"])
    solve(unit.dg, unit.eq_dg[0, "target"])
    solve(unit.particle_dens_bulk, unit.eq_bed_voidage)
    solve(unit.velocity_int, unit.eq_velocity_relation)
    if transport.film_transfer_m_s is None:
        unit.shape_correction_factor.set_value(transport.shape_factor)
        solve(unit.N_Re, unit.eq_reynolds_number[0])
        solve(unit.N_Sc, unit.eq_schmidt_number[0, "target"])
        solve(unit.kf, unit.eq_film_transfer_rate[0, "target"])
    else:
        unit.kf.set_value(transport.film_transfer_m_s)
    if transport.surface_diffusion_m2_s is None:
        unit.particle_porosity.set_value(transport.particle_porosity)
        unit.tort.set_value(transport.tortuosity)
        unit.spdfr.set_value(transport.surface_to_pore_flux_ratio)
        solve(unit.ds, unit.eq_surface_diffusion_coefficient_calculated[0, "target"])
    else:
        unit.ds.set_value(transport.surface_diffusion_m2_s)
    solve(unit.N_Bi, unit.eq_number_bi)
    solve(unit.min_N_St, unit.eq_min_number_st_cps)
    solve(unit.throughput, unit.eq_throughput)
    solve(unit.bed_length, unit.eq_bed_length)
    solve(unit.bed_area, unit.eq_bed_area[0])
    solve(unit.bed_diameter, unit.eq_bed_diameter)
    solve(unit.bed_volume, unit.eq_bed_volume)
    solve(unit.residence_time, unit.eq_residence_time)
    solve(unit.bed_mass_gac, unit.eq_mass_gac_bed)
    solve(unit.min_ebct, unit.eq_min_ebct_cps)
    solve(unit.min_residence_time, unit.eq_min_residence_time_cps)
    solve(unit.min_operational_time, unit.eq_minimum_operational_time_cps)
    solve(unit.operational_time, unit.eq_operational_time)
    solve(unit.bed_volumes_treated, unit.eq_bed_volumes_treated)
    solve(unit.gac_usage_rate, unit.eq_gac_usage_rate)
    for index in unit.ele_index:
        solve(unit.ele_conc_ratio_replace[index], unit.eq_ele_conc_ratio_replace[index])
        solve(unit.ele_throughput[index], unit.eq_ele_throughput[index])
        solve(unit.ele_min_operational_time[index], unit.eq_ele_min_operational_time[index])
        solve(unit.ele_operational_time[index], unit.eq_ele_operational_time[index])
    for index in unit.ele_index:
        solve(unit.ele_conc_ratio_term[index], unit.eq_ele_conc_ratio_term[index])
    solve(unit.conc_ratio_avg, unit.eq_conc_ratio_avg)
    solve(
        unit.process_flow.mass_transfer_term[0, "Liq", "target"],
        unit.eq_mass_transfer_cv[0, "target"],
    )
    solve(
        unit.gac_removed[0].flow_mass_phase_comp["Liq", "target"],
        unit.eq_mass_transfer_port[0, "target"],
    )
    solve(unit.mass_adsorbed, unit.eq_mass_adsorbed["target"])
    outlet = unit.process_flow.properties_out[0].flow_mass_phase_comp["Liq", "target"]
    solve(outlet, unit.process_flow.material_balances[0, "target"])
    residuals = {}
    for variable, constraint in constraints:
        residual, _, scale = equation_state(variable, constraint)
        residuals[constraint.name] = abs(residual) / scale
    maximum = max(residuals.values())
    if not isfinite(maximum) or maximum > 1e-8:
        raise RuntimeError(f"actual WaterTAP scaled residual {maximum}: {residuals}")
    mapping = {
        "equilibrium_loading_kg_kg": "equil_conc",
        "distribution_parameter": "dg",
        "biot_number": "N_Bi",
        "minimum_stanton_number": "min_N_St",
        "throughput_parameter": "throughput",
        "bulk_density_kg_m3": "particle_dens_bulk",
        "interstitial_velocity_m_s": "velocity_int",
        "bed_length_m": "bed_length",
        "bed_diameter_m": "bed_diameter",
        "bed_area_m2": "bed_area",
        "bed_volume_m3": "bed_volume",
        "carbon_mass_kg": "bed_mass_gac",
        "residence_time_s": "residence_time",
        "minimum_ebct_s": "min_ebct",
        "minimum_residence_time_s": "min_residence_time",
        "minimum_operational_time_s": "min_operational_time",
        "operational_time_s": "operational_time",
        "bed_volumes_treated": "bed_volumes_treated",
        "carbon_usage_kg_s": "gac_usage_rate",
        "average_concentration_ratio": "conc_ratio_avg",
        "adsorbed_mass_kg": "mass_adsorbed",
        "film_transfer_m_s": "kf",
        "surface_diffusion_m2_s": "ds",
    }
    return {
        "metrics": {
            **{key: float(value(getattr(unit, path))) for key, path in mapping.items()},
            "average_outlet_target_kg_s": float(value(outlet)),
        },
        "breakthrough": [
            {
                "time_s": float(value(unit.ele_operational_time[index])),
                "concentration_ratio": float(value(unit.ele_conc_ratio_replace[index])),
            }
            for index in unit.ele_disc
        ],
        "reference": {
            "watertap": version("watertap"),
            "idaes-pse": version("idaes-pse"),
            "pyomo": version("pyomo"),
            "maximum_scaled_residual": maximum,
            "solved_constraint_count": len(constraints),
            "residuals": residuals,
            "solver_messages": solver_messages,
            "boundary": "actual GAC constraints; declared property boundaries, not MCAS parity",
        },
    }
