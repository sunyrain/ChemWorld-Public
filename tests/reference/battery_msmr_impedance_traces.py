"""Actual original MSMR systems, full states and observer-preserving linearization."""

import numpy as np
from battery_composite_impedance_traces import snapshot
from battery_half_cell_impedance_traces import sparse_record
from battery_msmr_cases import native_initial_state, native_spec
from battery_msmr_impedance_cases import (
    FREQUENCIES,
    F,
    R,
    binding,
    refusal_input,
    response_order,
    response_units,
)

DEFINITION = "Z=-delta_V/delta_I; positive discharge; exp(+i*2*pi*f*t)"
ABS_NAMES = ("Throughput capacity [A.h]", "Throughput energy [W.h]")
COUNTERS = (
    "Discharge capacity [A.h]",
    "Throughput capacity [A.h]",
    "Discharge energy [W.h]",
    "Throughput energy [W.h]",
)
ABS_EXCLUSION = (
    "excluded: abs(I) and abs(I*V) have no two-sided rest derivative; "
    "actual CasADi convention retained"
)


def native_observations(spec, state, variables, raw, data):
    from chemworld.physchem.battery_msmr import msmr_fields

    indices = {v["name"]: i for i, v in enumerate(variables)}
    current = raw[..., indices["battery:current_A"]]
    outputs = {
        "Z_ohm": -raw[..., indices["battery:voltage_V"]],
        "current_A_A": current,
        "charge_C_A": raw[..., indices["discharged_charge_C"]],
        "work_J_A": raw[..., indices["delivered_work_J"]],
    }
    fields = msmr_fields(spec, state, current_A=0.0)
    materials = {}
    for name, sign in (("negative", 1), ("positive", -1)):
        p, m = getattr(spec, name), getattr(spec, name + "_material")
        u0 = np.asarray(fields[name + "_particle_potentials_V"])
        x0 = np.asarray(fields[name + "_particle_fractions"])
        slopes = np.array([m.stoichiometry_derivatives(u, spec.temperature_K)[1] for u in u0])
        cap = p.mesh.cell_volumes_m3 * p.maximum_concentration_mol_m3
        dn = raw[..., [indices[k] for k in p.keys(name)]]
        dx = dn / cap
        du = dx / slopes
        us = 1.5 * u0[-1] - 0.5 * u0[-2]
        dus = 1.5 * du[..., -1] - 0.5 * du[..., -2]
        xj = np.array(m.occupancies(us, spec.temperature_K))
        factor = F / (R * spec.temperature_K)
        rows = np.array(m.reaction_parameters)
        dxj = -factor / rows[:, 2] * xj * (1 - xj / rows[:, 0])
        j0 = np.array(
            m.exchange_current_densities(
                us, spec.temperature_K, spec.electrolyte_concentration_mol_m3
            )
        )
        _, _, eta_j, eta_u = m.kinetic_response(
            us, 0.0, spec.temperature_K, spec.electrolyte_concentration_mol_m3
        )
        deta = eta_j * sign * current / p.area_m2 + eta_u * dus
        outputs.update(
            {
                f"{name}:U": dus,
                f"{name}:eta": deta,
                f"{name}:surface": sum(dxj) * dus,
                f"{name}:average": dn.sum(axis=-1) / p.capacity_mol,
                f"{name}:amount": dn.sum(axis=-1),
            }
        )
        source_sum = np.zeros_like(dus)
        for j in range(len(rows)):
            node_x = np.array([m.occupancies(u, spec.temperature_K)[j] for u in u0[-2:]])
            node_dx = -factor / rows[j, 2] * node_x * (1 - node_x / rows[j, 0])
            source_x = 1.5 * node_dx[-1] * du[..., -1] - 0.5 * node_dx[-2] * du[..., -2]
            outputs[f"{name}:kinetic_x{j}"] = dxj[j] * dus
            outputs[f"{name}:source_x{j}"] = source_x
            outputs[f"{name}:current{j}"] = p.area_m2 * factor * j0[j] * deta
            source_sum += source_x
        outputs[f"{name}:source_surface"] = source_sum
        for i in range(p.cells):
            outputs[f"{name}:profile_U:{i}"] = du[..., i]
            outputs[f"{name}:profile_x:{i}"] = dx[..., i]
        materials[name] = {
            "potential_V": float(us),
            "x": xj.tolist(),
            "dx": dxj.tolist(),
            "j0": j0.tolist(),
            "radial_initial_U_V": u0.tolist(),
            "radial_initial_x": x0.tolist(),
            "radial_xU_V_inv": slopes.tolist(),
            "chemical_diffusivity_m2_s": float(
                -p.diffusivity_m2_s * factor * sum(xj) * (1 - sum(xj)) / sum(dxj)
            ),
        }
    return np.stack(
        [outputs[k] for k in response_order(data, spec.negative.cells)], axis=-1
    ), materials


def native_spectrum(data, cells, *, frequencies=FREQUENCIES, gradient=False):
    from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
    from chemworld.physchem.battery_msmr import msmr_readout
    from chemworld.physchem.residual_system import ResidualEvaluator

    spec = native_spec(data, cells)
    state = native_initial_state(spec, data, cells)
    before = snapshot(state)
    system, initial = battery_impedance_system(spec, state)
    residual = ResidualEvaluator(system).value(state.time_s, initial, np.zeros(len(initial)))
    variables = [{"name": v.name, "unit": v.unit, "scale": v.scale} for v in system.variables]
    _, materials = native_observations(spec, state, variables, np.zeros(len(initial)), data)
    metadata = {
        "implementation": "native",
        "grid": cells,
        "binding": binding(data, "native", cells, "radial-gradient" if gradient else None),
        "response_order": response_order(data, cells),
        "response_units": response_units(data, cells),
        "frequencies_Hz": list(map(float, frequencies)),
        "definition": DEFINITION,
        "initial_state": initial.tolist(),
        "stationary_residual": residual.tolist(),
        "residual_scales": system.residual_scales.tolist(),
        "variables": variables,
        "state_before": before,
        "faraday_C_mol": F,
        "gas_constant_J_mol_K": R,
        "rest_voltage_V": msmr_readout(spec, state, 0.0).voltage_V,
        "material_static": materials,
    }
    try:
        spectrum = battery_impedance(spec, frequencies, state=state)
    except Exception as exc:
        if not gradient:
            raise
        return {
            **metadata,
            "success": False,
            "points": [],
            "rejected": True,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "state_after": snapshot(state),
            "state_unchanged": before == snapshot(state),
        }
    points = []
    for point in spectrum.response.points:
        row = {
            "frequency_Hz": point.frequency_Hz,
            "success": point.success,
            "relative_linear_residual": point.relative_linear_residual,
            "failure_reason": point.failure_reason,
        }
        if point.response is not None:
            raw = point.response[:, 0]
            value, _ = native_observations(spec, state, variables, raw, data)
            row.update(
                response_real=value.real.tolist(),
                response_imag=value.imag.tolist(),
                state_response_real=raw.real.tolist(),
                state_response_imag=raw.imag.tolist(),
            )
        points.append(row)
    return {
        **metadata,
        "success": spectrum.success,
        "points": points,
        "diagnostics": spectrum.response.diagnostics,
        "state_after": snapshot(state),
        "state_unchanged": before == snapshot(state),
        "rejected": False,
    }


def source_output_names(data, cells):
    return ["Voltage [V]", *["eis:" + name for name in response_order(data, cells)[1:]]]


def source_linearization(sim, names, initial, layout):
    import casadi
    from battery_frequency_pybamm import casadi_sparse

    model = sim.built_model
    exported = model.export_casadi_objects(names, input_parameter_order=["Current function [A]"])
    states = casadi.vertcat(exported["x"], exported["z"])
    rhs = casadi.vertcat(exported["rhs"], exported["algebraic"])
    outputs = casadi.vertcat(*exported["variables"].values())
    function = casadi.Function(
        "msmr_eis_full",
        [exported["t"], states, exported["inputs"]],
        [
            casadi.jacobian(rhs, states),
            casadi.jacobian(rhs, exported["inputs"]),
            casadi.jacobian(outputs, states),
            casadi.jacobian(outputs, exported["inputs"]),
        ],
    )
    j, b, c, d = function(0.0, initial, [0.0])
    j, c = casadi_sparse(j), casadi_sparse(c)
    observers = [
        entry["indices"][0]
        for entry in layout
        if entry["name"] in ABS_NAMES and len(entry["indices"]) == 1
    ]
    if len(observers) != 2:
        raise ValueError("MSMR actual energy=true must retain both abs-throughput states")
    physical = [i for i in range(len(initial)) if i not in observers]
    return {
        "jacobian": sparse_record(j),
        "mass": sparse_record(model.mass_matrix.entries),
        "forcing": np.asarray(b).ravel().tolist(),
        "output_state": sparse_record(c),
        "output_input": np.asarray(d).ravel().tolist(),
        "abs_observer_indices": observers,
        "physical_row_indices": physical,
        "maximum_observer_feedback_rhs": float(
            np.max(abs(j[physical][:, observers].toarray()), initial=0)
        ),
        "maximum_observer_feedback_output": float(
            np.max(abs(c[:, observers].toarray()), initial=0)
        ),
        "abs_input_derivative_convention": np.asarray(b).ravel()[observers].tolist(),
        "abs_response_qualification": ABS_EXCLUSION,
    }


def source_spectrum(data, cells, *, frequencies=FREQUENCIES, gradient=False):
    import pybamm
    from battery_frequency_pybamm import frequency_from_simulation
    from battery_msmr_pybamm import build_reference, static_reference_values

    sim = build_reference(data, cells)
    # The released material normalizes electrolyte concentration by initial ce.
    # Convert its reference exchange coefficient to the authored fixed 1000 basis.
    # This changes no constitutive equation or physical material parameter.
    reference_mapping = []
    source_parameters = {}
    ce = data["electrolyte_concentration_mol_m3"]
    for e in data["electrodes"]:
        for j, row in enumerate(e["rows"]):
            factor = (ce / 1000.0) ** (1 - row[4])
            source_value = row[3] * factor
            key = f"j0_ref_{e['name'][0]}_{j}"
            source_parameters[key] = source_value
            reference_mapping.append(
                {
                    "electrode": e["name"],
                    "row_index": j,
                    "authored_row": list(row),
                    "authored_reference_ce_mol_m3": 1000.0,
                    "source_reference_ce_mol_m3": ce,
                    "factor": factor,
                    "source_parameter": key,
                    "source_jref_A_m2": source_value,
                }
            )
    sim.parameter_values.update(source_parameters)
    original = sim.model

    def original_names():
        return {
            "rhs": [v.name for v in original.rhs],
            "algebraic": [v.name for v in original.algebraic],
            "initial_conditions": [v.name for v in original.initial_conditions],
            "variables": list(original.variables),
        }

    original_before = original_names()
    outputs = {
        "current_A_A": original.variables["Current [A]"],
        "charge_C_A": 3600 * original.variables["Discharge capacity [A.h]"],
        "work_J_A": 3600 * original.variables["Discharge energy [W.h]"],
    }
    for e in data["electrodes"]:
        name, short = e["name"], e["name"][0]
        base = "qualification:" + name + ":"
        profile = original.variables[base + "profile"]
        average = pybamm.r_average(profile)
        outputs.update(
            {
                f"{name}:U": original.variables[base + "surface_potential"],
                f"{name}:eta": original.variables[base + "eta"],
                f"{name}:surface": original.variables[base + "surface"],
                f"{name}:source_surface": original.variables[base + "source_surface"],
                f"{name}:average": average,
                f"{name}:amount": average
                * e["maximum_concentration_mol_m3"]
                * e["active_volume_m3"],
            }
        )
        for j in range(len(e["rows"])):
            outputs[f"{name}:kinetic_x{j}"] = original.variables[base + f"kinetic_x{j}"]
            outputs[f"{name}:source_x{j}"] = original.variables[f"X-averaged surface x_{short}_{j}"]
            outputs[f"{name}:current{j}"] = (
                original.variables[f"X-averaged j_{short}_{j} [A.m-2]"]
                * 3
                * e["active_volume_m3"]
                / e["radius_m"]
            )
    original.variables.update({"eis:" + k: v for k, v in outputs.items()})
    sim.build()
    model = sim.built_model
    for e in data["electrodes"]:
        name = e["name"]
        for key, original_key in (("profile_U", "potential"), ("profile_x", "profile")):
            expression = model.variables[f"qualification:{name}:{original_key}"]
            if expression.size != cells:
                raise ValueError("MSMR original source radial size mismatch")
            for i in range(cells):
                model.variables[f"eis:{name}:{key}:{i}"] = pybamm.Index(expression, i)
    names = source_output_names(data, cells)
    result = frequency_from_simulation(
        sim,
        frequencies,
        family="MSMR SPM",
        grid={"radial_cells": cells, "mesh": "uniform"},
        output_names=names,
        response_names=response_order(data, cells),
    )
    layout = []
    for variable, slices in model.y_slices.items():
        if isinstance(variable, pybamm.Concatenation):
            continue
        layout.append(
            {
                "name": variable.name,
                "indices": [i for sl in slices for i in range(sl.start, sl.stop)],
                "scale": float(sim.parameter_values.process_symbol(variable.scale).evaluate()),
                "reference": float(
                    sim.parameter_values.process_symbol(variable.reference).evaluate()
                ),
                "domains": variable.domains,
            }
        )
    by_name = {entry["name"]: entry for entry in layout}
    initial = np.asarray(result["initial_state"])
    materials = {}
    for e in data["electrodes"]:
        name = e["name"]
        entry = by_name[f"X-averaged {name} particle potential [V]"]
        u = initial[entry["indices"]] * entry["scale"] + entry["reference"]
        us = 1.5 * u[-1] - 0.5 * u[-2]
        m = static_reference_values(
            data,
            e,
            data["temperature_K"],
            float(us),
            0.0,
            data["electrolyte_concentration_mol_m3"],
            simulation=sim,
        )
        node_values = [
            static_reference_values(
                data,
                e,
                data["temperature_K"],
                float(v),
                0.0,
                data["electrolyte_concentration_mol_m3"],
                simulation=sim,
            )
            for v in u
        ]
        f = float(pybamm.constants.F.evaluate())
        r = float(pybamm.constants.R.evaluate())
        materials[name] = {
            **{k: m[k] for k in ("x", "dx", "j0")},
            "potential_V": float(us),
            "radial_initial_U_V": u.tolist(),
            "radial_initial_x": [sum(v["x"]) for v in node_values],
            "radial_xU_V_inv": [sum(v["dx"]) for v in node_values],
            "chemical_diffusivity_m2_s": -e["diffusivity_m2_s"]
            * f
            / (r * data["temperature_K"])
            * sum(m["x"])
            * (1 - sum(m["x"]))
            / sum(m["dx"]),
        }
    result.update(
        implementation="reference",
        grid=cells,
        source_state_layout=layout,
        binding=binding(data, "reference", cells, "radial-gradient" if gradient else None),
        frequencies_Hz=list(map(float, frequencies)),
        response_units=response_units(data, cells),
        faraday_C_mol=f,
        gas_constant_J_mol_K=r,
        rest_voltage_V=result["initial_outputs"][0],
        source_original_model=original_before,
        source_retained_model=original_names(),
        source_linearization=source_linearization(sim, names, initial, layout),
        material_static=materials,
        source_exchange_reference_mapping=reference_mapping,
        rejected=result.get("failure") == "reference_not_stationary",
    )
    return result


def refusal(data, name, cells=80):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.residual_system import ResidualDomainError

    spec, state, frequencies = refusal_input(data, name, cells)
    before = snapshot(state)
    try:
        battery_impedance(spec, frequencies, state=state)
        result = {"rejected": False}
    except (ValueError, ResidualDomainError) as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {
        "implementation": "refusal",
        "name": name,
        "grid": cells,
        **result,
        "binding": binding(data, "refusal", cells, name),
        "state_before": before,
        "state_after": snapshot(state),
        "state_unchanged": before == snapshot(state),
        "specification": spec.to_dict(),
        "frequencies_Hz": frequencies,
    }
