"""Read the complete original native/source stationary frequency systems."""

import numpy as np
from battery_composite_impedance_traces import snapshot
from battery_half_cell_cases import native_spec
from battery_half_cell_impedance_cases import (
    FREQUENCIES,
    F,
    R,
    binding,
    native_state,
    refusal_input,
    response_order,
    response_units,
)

DEFINITION = "Z=-delta_V/delta_I; positive discharge; exp(+i*2*pi*f*t)"


def native_observations(spec, variables, raw):
    """Project the actual residual solution onto its existing rest readout."""
    from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC

    indices = {v["name"]: i for i, v in enumerate(variables)}
    current = raw[..., indices[BATTERY_ALGEBRAIC[0]]]
    p = spec.positive
    shells = raw[..., [indices[k] for k in p.keys("positive")]]
    profile = shells / (p.mesh.cell_volumes_m3 * p.maximum_concentration_mol_m3)
    surface = profile[..., -1] + current * p.mesh.spherical_surface_distance_m / (
        F * p.area_m2 * p.diffusivity_m2_s * p.maximum_concentration_mol_m3
    )
    amount = shells.sum(axis=-1)
    theta = spec.initial_positive_fraction
    metal = spec.metal.native_parameters
    j0 = p.exchange_current_scale_A_m2 * np.sqrt(theta * (1 - theta))
    # These two expressions are observation derivatives of the original readout
    # at I=0. The independent oracle lives in a separate scalar-only module.
    em = (
        current
        * R
        * spec.temperature_K
        / (F * metal["area_m2"] * metal["exchange_current_density_A_m2"])
    )
    ep = -current * R * spec.temperature_K / (F * p.area_m2 * j0)
    out = np.stack(
        (
            -raw[..., indices[BATTERY_ALGEBRAIC[1]]],
            current,
            surface,
            amount / p.capacity_mol,
            amount,
            raw[..., indices["metal:Li"]],
            raw[..., indices["discharged_charge_C"]],
            em,
            ep,
            current * spec.metal.resistance_ohm,
        ),
        axis=-1,
    )
    return np.concatenate((out, profile), axis=-1), raw[..., indices["delivered_work_J"]]


def native_spectrum(data, cells, *, frequencies=FREQUENCIES, gradient=False):
    from chemworld.physchem.battery_half_cell import half_cell_readout
    from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
    from chemworld.physchem.residual_system import ResidualEvaluator

    spec = native_spec(data, cells)
    state = native_state(spec, gradient=gradient)
    before = snapshot(state)
    system, initial = battery_impedance_system(spec, state)
    residual = ResidualEvaluator(system).value(state.time_s, initial, np.zeros(len(initial)))
    variables = [{"name": v.name, "unit": v.unit, "scale": v.scale} for v in system.variables]
    metadata = {
        "implementation": "native",
        "grid": cells,
        "binding": binding(data, "native", cells, "radial-gradient" if gradient else None),
        "response_order": response_order(cells),
        "response_units": response_units(cells),
        "frequencies_Hz": list(map(float, frequencies)),
        "definition": DEFINITION,
        "initial_state": initial.tolist(),
        "stationary_residual": residual.tolist(),
        "residual_scales": system.residual_scales.tolist(),
        "variables": variables,
        "state_before": before,
        "faraday_C_mol": F,
        "metal_mapping_faraday_C_mol": F,
        "rest_voltage_V": half_cell_readout(spec, state, 0.0).voltage_V,
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
            value, work = native_observations(spec, variables, raw)
            row.update(
                response_real=value.real.tolist(),
                response_imag=value.imag.tolist(),
                state_response_real=raw.real.tolist(),
                state_response_imag=raw.imag.tolist(),
                work_response_real=float(work.real),
                work_response_imag=float(work.imag),
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


def source_output_names(cells):
    return [
        "Voltage [V]",
        "Current [A]",
        "X-averaged positive particle surface concentration",
        "Average positive particle concentration",
        "eis:positive amount [mol]",
        "qualification:metal amount [mol]",
        "eis:signed charge [C]",
        "Lithium metal interface reaction overpotential [V]",
        "X-averaged positive electrode reaction overpotential [V]",
        "qualification:metal ohmic [V]",
        *(f"eis:positive profile:{i}" for i in range(cells)),
    ]


def sparse_record(matrix):
    from scipy import sparse

    coo = sparse.coo_matrix(matrix)
    return {
        "shape": list(coo.shape),
        "row": coo.row.tolist(),
        "col": coo.col.tolist(),
        "data": coo.data.tolist(),
    }


def source_linearization(simulation, output_names, initial, layout):
    """Retain the actual full system and demonstrate the abs observer has no feedback."""
    import casadi
    from battery_frequency_pybamm import casadi_sparse

    model = simulation.built_model
    exported = model.export_casadi_objects(
        output_names, input_parameter_order=["Current function [A]"]
    )
    states = casadi.vertcat(exported["x"], exported["z"])
    rhs = casadi.vertcat(exported["rhs"], exported["algebraic"])
    outputs = casadi.vertcat(*exported["variables"].values())
    function = casadi.Function(
        "halfcell_full_linearization",
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
    throughput = [entry for entry in layout if entry["name"] == "Throughput capacity [A.h]"]
    if len(throughput) != 1 or len(throughput[0]["indices"]) != 1:
        raise ValueError("actual source requires its original scalar abs-throughput state")
    observer = throughput[0]["indices"]
    physical = [i for i in range(len(initial)) if i not in observer]
    return {
        "jacobian": sparse_record(j),
        "mass": sparse_record(model.mass_matrix.entries),
        "forcing": np.asarray(b).ravel().tolist(),
        "output_state": sparse_record(c),
        "output_input": np.asarray(d).ravel().tolist(),
        "abs_observer_indices": observer,
        "physical_row_indices": physical,
        "maximum_observer_feedback_rhs": float(
            np.max(abs(j[physical][:, observer].toarray()), initial=0)
        ),
        "maximum_observer_feedback_output": float(np.max(abs(c[:, observer].toarray()), initial=0)),
        "abs_input_derivative_convention": np.asarray(b).ravel()[observer].tolist(),
        "abs_response_qualification": (
            "excluded: abs(I) has no two-sided derivative at I=0; actual CasADi convention retained"
        ),
    }


def source_spectrum(data, cells, *, frequencies=FREQUENCIES, gradient=False):
    import pybamm
    from battery_frequency_pybamm import frequency_from_simulation
    from battery_half_cell_pybamm import build_reference

    simulation = build_reference(data, cells)
    original = simulation.model
    original_names = {
        "rhs": [v.name for v in original.rhs],
        "initial_conditions": [v.name for v in original.initial_conditions],
        "variables": list(original.variables),
    }
    if gradient:
        variables = [
            v
            for v in original.initial_conditions
            if "positive particle concentration" in v.name.lower()
        ]
        if len(variables) != 1:
            raise ValueError("source gradient must identify one actual working-particle state")
        r = pybamm.SpatialVariable("r_p", domains=variables[0].domains, coord_sys="spherical polar")
        edges = np.linspace(0.0, 1.0, cells + 1)
        weights = np.diff(edges**3)
        mean_square = np.dot(weights, ((edges[:-1] + edges[1:]) / 2) ** 2) / sum(weights)
        original.initial_conditions[variables[0]] = data["maximum_concentration_mol_m3"] * (
            data["initial_positive_fraction"] + 0.02 * ((r / data["radius_m"]) ** 2 - mean_square)
        )
    original.variables["eis:positive amount [mol]"] = (
        original.variables["Average positive particle concentration"]
        * data["maximum_concentration_mol_m3"]
        * data["active_volume_m3"]
    )
    original.variables["eis:signed charge [C]"] = (
        3600 * original.variables["Discharge capacity [A.h]"]
    )
    simulation.build()
    model = simulation.built_model
    profile = (
        model.variables["X-averaged positive particle concentration [mol.m-3]"]
        / data["maximum_concentration_mol_m3"]
    )
    if profile.size != cells:
        raise ValueError("actual source radial profile shape mismatch")
    for i in range(cells):
        model.variables[f"eis:positive profile:{i}"] = pybamm.Index(profile, i)
    names = source_output_names(cells)
    result = frequency_from_simulation(
        simulation,
        frequencies,
        family="positive-working SPM",
        grid={"radial_cells": cells, "mesh": "uniform"},
        output_names=names,
        response_names=response_order(cells),
    )
    layout = []
    for variable, slices in model.y_slices.items():
        if isinstance(variable, pybamm.Concatenation):
            continue
        layout.append(
            {
                "name": variable.name,
                "indices": [i for sl in slices for i in range(sl.start, sl.stop)],
                "scale": float(
                    simulation.parameter_values.process_symbol(variable.scale).evaluate()
                ),
                "reference": float(
                    simulation.parameter_values.process_symbol(variable.reference).evaluate()
                ),
                "domains": variable.domains,
            }
        )
    linear = source_linearization(simulation, names, result["initial_state"], layout)
    # Preserve the source's actual disabled-energy expression, with no dynamic interpretation.
    energy = model.variables["Discharge energy [W.h]"].evaluate(
        t=0.0, y=np.asarray(result["initial_state"])[:, None], inputs={"Current function [A]": 0.0}
    )
    result.update(
        implementation="reference",
        grid=cells,
        source_state_layout=layout,
        binding=binding(data, "reference", cells, "radial-gradient" if gradient else None),
        frequencies_Hz=list(map(float, frequencies)),
        response_units=response_units(cells),
        faraday_C_mol=float(pybamm.constants.F.evaluate()),
        metal_mapping_faraday_C_mol=F,
        rest_voltage_V=result["initial_outputs"][0],
        source_original_model=original_names,
        source_retained_model={
            "rhs": [v.name for v in original.rhs],
            "initial_conditions": [v.name for v in original.initial_conditions],
            "variables": list(original.variables),
        },
        source_linearization=linear,
        source_energy={
            "dynamic": False,
            "original_energy_Wh": float(np.asarray(energy).item()),
            "qualification": "excluded: original source calculate discharge energy is false",
        },
        source_metal_definition=(
            "derived n0 - 3600*Discharge capacity [A.h]/F; not an upstream independent metal state"
        ),
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
