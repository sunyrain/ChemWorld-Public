"""Actual optional WaterTAP 1.8 jar/clarifier constraints at common boundaries.

Coagulation's port lacks total flow_vol required by Clarifier. A state-block
subclass adds only an alias to its existing single liquid flow_vol_phase. No
property or unit equation is replaced. Property closures, separator electricity,
transient events and native thermodynamics are outside this reference scope.
"""

from importlib.metadata import version


def upstream(spec, feed, flow):
    from idaes.core import FlowsheetBlock, StateBlock, declare_process_block_class
    from pyomo.core.expr.calculus.derivatives import differentiate
    from pyomo.environ import ConcreteModel, Expression, value
    from pyomo.util.calc_var_value import calculate_variable_from_constraint
    from watertap.property_models.unit_specific.coagulation_prop_pack import (
        CoagulationParameterBlock,
        CoagulationStateBlockData,
    )
    from watertap.unit_models.clarifier import Clarifier
    from watertap.unit_models.coag_floc_model import CoagulationFlocculation

    if version("watertap") != "1.8.0":
        raise ValueError("49F reference requires actual WaterTAP 1.8.0")

    @declare_process_block_class("ClarifierCoagulationStateBlock49F", block_class=StateBlock)
    class ClarifierCoagulationStateBlockData49F(CoagulationStateBlockData):
        def build(self):
            super().build()
            self.port_flow_vol = Expression(expr=self.flow_vol_phase["Liq"])

        def define_port_members(self):
            return {**self.define_state_vars(), "flow_vol": self.port_flow_vol}

    model = ConcreteModel()
    model.fs = FlowsheetBlock(dynamic=False)
    model.fs.properties = CoagulationParameterBlock()
    model.fs.unit = CoagulationFlocculation(property_package=model.fs.properties)
    # Port-only compatibility shim; both units retain their actual public constraints.
    model.fs.clarifier_properties = CoagulationParameterBlock()
    model.fs.clarifier_properties._state_block_class = globals()[
        "ClarifierCoagulationStateBlock49F"
    ]
    model.fs.clarifier = Clarifier(
        property_package=model.fs.clarifier_properties, outlet_list=["underflow", "effluent"]
    )
    unit, clarifier = model.fs.unit, model.fs.clarifier
    inlet, outlet = unit.control_volume.properties_in[0], unit.control_volume.properties_out[0]
    liquid = spec.material.liquid
    mw = {
        p.component.identifier: p.component.molecular_weight_g_mol / 1000 for p in liquid.packages
    }
    particle_mw = spec.material.particle.molecular_weight_kg_mol
    # Independent pure-volume boundary calculation, without calling the native profile.
    volumes = {
        p.component.identifier: mw[p.component.identifier]
        / p.by_property("liquid_density")[0].coefficients["rho_ref"]
        for p in liquid.packages
    }
    volume = (
        sum(feed.liquid_mol[k] * v for k, v in volumes.items())
        + (feed.dispersed_mol + feed.floc_mol) * spec.material.particle.molar_volume_m3_mol
    )
    mass = {
        "H2O": feed.liquid_mol[spec.material.water_id] * mw[spec.material.water_id],
        "TDS": sum(n * mw[k] for k, n in feed.liquid_mol.items() if k != spec.material.water_id),
        "TSS": feed.dispersed_mol * particle_mw,
        "Sludge": feed.floc_mol * particle_mw,
    }
    density = sum(mass.values()) / volume
    for state in (inlet, outlet):
        state.temperature.set_value(spec.temperature_K)
        state.pressure.set_value(spec.pressure_Pa)
        state.flow_vol_phase["Liq"].set_value(flow)
        state.dens_mass_phase["Liq"].set_value(density)
        state.visc_d_phase["Liq"].set_value(spec.material.viscosity_Pa_s)
    for k, amount in mass.items():
        inlet.flow_mass_phase_comp["Liq", k].set_value(amount * flow / volume)
    jar, mixing = spec.jar_test, spec.mixing
    for variable, number in (
        (unit.slope[0], jar.slope_mg_L_ntu),
        (unit.intercept[0], jar.intercept_mg_L),
        (unit.initial_turbidity_ntu[0], jar.initial_ntu),
        (unit.final_turbidity_ntu[0], jar.final_ntu),
        (unit.rapid_mixing_retention_time[0], mixing.rapid_retention_s),
        (unit.num_rapid_mixing_basins, mixing.rapid_basins),
        (unit.rapid_mixing_vel_grad[0], mixing.rapid_velocity_gradient_s_inv),
        (unit.floc_retention_time[0], mixing.floc_retention_s),
        (unit.single_paddle_length, mixing.paddle_length_m),
        (unit.single_paddle_width, mixing.paddle_width_m),
        (unit.paddle_rotational_speed[0], mixing.paddle_rotation_Hz),
        (unit.paddle_drag_coef[0], mixing.paddle_drag_coefficient),
        (unit.vel_fraction, mixing.paddle_velocity_fraction),
        (unit.num_paddle_wheels, mixing.paddle_wheels),
        (unit.num_paddles_per_wheel, mixing.paddles_per_wheel),
    ):
        variable.set_value(number)
    solved = []

    def solve(variable, constraint):
        calculate_variable_from_constraint(variable, constraint, eps=1e-16)
        lhs, rhs = constraint.expr.args
        for _ in range(3):
            residual = float(value(lhs - rhs))
            derivative = float(differentiate(lhs - rhs, wrt=variable))
            variable.set_value(value(variable) - residual / derivative)
        a, b = float(value(lhs)), float(value(rhs))
        residual = a - b
        scale = max(abs(a), abs(b), abs(float(value(variable))), 1e-20)
        if abs(residual) > 1e-8 * scale:
            raise RuntimeError(f"upstream constraint failed: {constraint.name}")
        solved.append(
            {"name": constraint.name, "residual": residual, "relative_residual": residual / scale}
        )

    solve(unit.tss_loss_rate[0], unit.eq_tss_loss_rate[0])
    for k in mass:
        solve(
            unit.control_volume.mass_transfer_term[0, "Liq", k],
            unit.eq_mass_transfer_term[0, "Liq", k],
        )
        solve(
            outlet.flow_mass_phase_comp["Liq", k],
            unit.control_volume.material_balances[0, "Liq", k],
        )
    for variable, constraint in (
        (unit.rapid_mixing_basin_vol[0], unit.eq_rapid_mixing_basin_vol[0]),
        (unit.rapid_mixing_power[0], unit.eq_rapid_mixing_power[0]),
        (unit.floc_basin_vol[0], unit.eq_floc_basin_vol[0]),
        (unit.floc_wheel_speed[0], unit.eq_floc_wheel_speed[0]),
        (unit.flocculation_power[0], unit.eq_flocculation_power[0]),
        (unit.total_power[0], unit.eq_total_power[0]),
    ):
        solve(variable, constraint)
    fractions = {
        "H2O": spec.clarifier.sludge_liquid_fraction,
        "TDS": spec.clarifier.sludge_liquid_fraction,
        "TSS": spec.clarifier.dispersed_capture,
        "Sludge": spec.clarifier.floc_capture,
    }
    rates = {name: {} for name in ("underflow", "effluent")}
    for k in mass:
        clarifier.mixed_state[0].flow_mass_phase_comp["Liq", k].set_value(
            value(outlet.flow_mass_phase_comp["Liq", k])
        )
        clarifier.split_fraction[0, "underflow", k].set_value(fractions[k])
        solve(clarifier.split_fraction[0, "effluent", k], clarifier.sum_split_frac[0, k])
        for name in rates:
            variable = getattr(clarifier, name + "_state")[0].flow_mass_phase_comp["Liq", k]
            solve(variable, clarifier.material_splitting_eqn[0, name, "Liq", k])
            rates[name][k] = value(variable)
    return {
        "version": version("watertap"),
        "tss_loss_kg_s": value(unit.tss_loss_rate[0]),
        "rapid_basin_volume_each_m3": value(unit.rapid_mixing_basin_vol[0]),
        "rapid_basin_volume_total_m3": value(unit.rapid_mixing_basin_vol[0]) * mixing.rapid_basins,
        "floc_basin_volume_m3": value(unit.floc_basin_vol[0]),
        "paddle_speed_m_s": value(unit.floc_wheel_speed[0]),
        "rapid_mixing_power_W": value(unit.rapid_mixing_power[0]) * 1000,
        "floc_mixing_power_W": value(unit.flocculation_power[0]) * 1000,
        "mixing_power_W": value(unit.total_power[0]) * 1000,
        "product_kg_s": rates["effluent"],
        "sludge_kg_s": rates["underflow"],
        "common_boundary": {
            "flow_m3_s": flow,
            "density_kg_m3": density,
            "viscosity_Pa_s": spec.material.viscosity_Pa_s,
        },
        "constraints": solved,
    }
