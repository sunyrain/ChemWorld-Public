"""Read actual native/source small-signal states, without time stepping or relaxation."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
from battery_composite_impedance_cases import (
    FREQUENCIES,
    PHASES,
    F,
    binding,
    gradient_profile,
    native_spec,
    phase_values,
    response_order,
)


def snapshot(state):
    return {
        "amounts_mol": dict(state.amounts_mol),
        "coordinates": dict(state.coordinates),
        "algebraic_values": dict(state.algebraic_values),
        "time_s": state.time_s,
        "temperature_K": state.temperature_K,
        "volume_L": state.volume_L,
    }


def native_spectrum(data, cells, *, single=False, frequencies=FREQUENCIES, nonstationary=None):
    from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
    from chemworld.physchem.battery_models import BATTERY_ALGEBRAIC
    from chemworld.physchem.residual_system import ResidualEvaluator

    inputs = binding(data, "native", cells, single, nonstationary)
    data = deepcopy(data)
    if nonstationary == "internal-current":
        data["phases"][1]["initial_fraction"] = 0.6
    spec = native_spec(data, cells, single=single)
    state = spec.initial_state()
    if single:
        bins = [
            ("host_a", spec.negative, spec.negative.keys("negative")),
            ("cathode", spec.positive, spec.positive.keys("positive")),
        ]
    else:
        bins = [
            (p.phase_id, particle, keys)
            for p, (_, _, particle, keys, _) in zip(spec.phases, spec.particle_bins, strict=True)
        ]
    if nonstationary == "radial-gradient":
        _, particle, keys = bins[0]
        amounts = dict(state.amounts_mol)
        profile = gradient_profile(cells, data["phases"][0]["initial_fraction"])
        amounts.update(
            zip(
                keys,
                profile * particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3,
                strict=True,
            )
        )
        state = replace(state, amounts_mol=amounts)
    before = snapshot(state)
    system, initial = battery_impedance_system(spec, state)
    raw_residual = ResidualEvaluator(system).value(state.time_s, initial, np.zeros(len(initial)))
    metadata = {
        "implementation": "native",
        "single_material": single,
        "grid": cells,
        "binding": inputs,
        "response_order": response_order(data, cells, single=single),
        "faraday_C_mol": F,
        "initial_state": initial.tolist(),
        "stationary_residual": raw_residual.tolist(),
        "residual_scales": system.residual_scales.tolist(),
        "variables": [{"name": v.name, "unit": v.unit, "scale": v.scale} for v in system.variables],
        "state_before": before,
        "phase_order": [name for name, _, _ in bins],
        "radial_cells": cells,
        "frequencies_Hz": list(map(float, frequencies)),
        "definition": "Z=-delta_V/delta_I; positive discharge; exp(+i*2*pi*f*t)",
    }
    try:
        spectrum = battery_impedance(spec, frequencies, state=state)
    except Exception as exc:
        if nonstationary is None:
            raise
        return {
            **metadata,
            "success": False,
            "points": [],
            "rejected": True,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "state_unchanged": before == snapshot(state),
            "state_after": snapshot(state),
        }
    indices = {v.name: i for i, v in enumerate(spectrum.response.variables)}
    points = []
    for point, impedance in zip(spectrum.response.points, spectrum.impedance_ohm, strict=True):
        row = {
            "frequency_Hz": point.frequency_Hz,
            "success": point.success,
            "relative_linear_residual": point.relative_linear_residual,
            "failure_reason": point.failure_reason,
        }
        if point.success:
            vector = point.response[:, 0]
            if single:
                qvalues = (
                    vector[indices[BATTERY_ALGEBRAIC[0]]],
                    -vector[indices[BATTERY_ALGEBRAIC[0]]],
                )
            else:
                qvalues = spec.reaction_map @ np.array(
                    [vector[indices[k]] for k in spec.algebraic_names]
                )
            measured, profiles = {}, {}
            for (name, particle, keys), q in zip(bins, qvalues, strict=True):
                shell = vector[[indices[k] for k in keys]]
                amount = sum(shell)
                profile = shell / (
                    particle.mesh.cell_volumes_m3 * particle.maximum_concentration_mol_m3
                )
                surface = profile[-1] - q * particle.mesh.spherical_surface_distance_m / (
                    F
                    * particle.area_m2
                    * particle.diffusivity_m2_s
                    * particle.maximum_concentration_mol_m3
                )
                measured[name] = (q, surface, amount / particle.capacity_mol, amount)
                profiles[name] = profile
            order = [name for name in PHASES if name in measured]
            values = np.array(
                [
                    impedance,
                    *(v for name in order for v in measured[name]),
                    *(v for name in order for v in profiles[name]),
                ]
            )
            row.update(
                response_real=values.real.tolist(),
                response_imag=values.imag.tolist(),
                state_response_real=vector.real.tolist(),
                state_response_imag=vector.imag.tolist(),
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


def source_spectrum(data, cells, *, single=False, frequencies=FREQUENCIES, nonstationary=None):
    import pybamm
    from battery_composite_pybamm import build_reference
    from battery_frequency_pybamm import frequency_from_simulation

    inputs = binding(data, "reference", cells, single, nonstationary)
    data = deepcopy(data)
    if nonstationary == "internal-current":
        data["phases"][1]["initial_fraction"] = 0.6
    simulation = build_reference(data, cells, single=single)
    if nonstationary == "radial-gradient":
        variables = [
            v
            for v in simulation.model.initial_conditions
            if "negative primary particle concentration" in v.name.lower()
        ]
        if len(variables) != 1:
            raise ValueError("source gradient mapping requires exactly the negative primary state")
        phase = data["phases"][0]
        # X-averaged SPM particle state has current-collector auxiliary domain,
        # unlike standard r_n_prim's full electrode-domain spatial variable.
        r = pybamm.SpatialVariable(
            "r_n_prim", domains=variables[0].domains, coord_sys="spherical polar"
        )
        edges = np.linspace(0.0, 1.0, cells + 1)
        weights = np.diff(edges**3)
        mean_square = np.dot(weights, ((edges[:-1] + edges[1:]) / 2) ** 2) / sum(weights)
        simulation.model.initial_conditions[variables[0]] = phase[
            "maximum_concentration_mol_m3"
        ] * (phase["initial_fraction"] + 0.02 * ((r / phase["radius_m"]) ** 2 - mean_square))
    simulation.build()
    model = simulation.built_model
    phases = phase_values(data, single)
    identities = [k for k in PHASES if k in phases]
    names = [
        "Voltage [V]",
        *(
            f"qualification:{k}:{m}"
            for k in identities
            for m in ("q", "surface", "average", "amount")
        ),
    ]
    for name in identities:
        variable = (
            model.variables[f"qualification:{name}:profile"]
            / phases[name]["maximum_concentration_mol_m3"]
        )
        if variable.size != cells:
            raise ValueError("actual source radial profile shape mismatch")
        for index in range(cells):
            label = f"eis:{name}:profile:{index}"
            model.variables[label] = pybamm.Index(variable, index)
            names.append(label)
    result = frequency_from_simulation(
        simulation,
        frequencies,
        family="SPM" if single else "SPM composite ('2','1')",
        grid={"radial_cells": cells, "mesh": "uniform"},
        output_names=names,
        response_names=response_order(data, cells, single=single),
    )
    layout = []
    params = simulation.parameter_values
    for variable, slices in model.y_slices.items():
        if isinstance(variable, pybamm.Concatenation):
            continue
        layout.append(
            {
                "name": variable.name,
                "indices": [i for sl in slices for i in range(sl.start, sl.stop)],
                "scale": float(params.process_symbol(variable.scale).evaluate()),
                "reference": float(params.process_symbol(variable.reference).evaluate()),
                "domains": variable.domains,
            }
        )
    result.update(
        implementation="reference",
        single_material=single,
        grid=cells,
        binding=inputs,
        radial_cells=cells,
        phase_order=list(phases),
        source_state_layout=layout,
        faraday_C_mol=float(pybamm.constants.F.evaluate()),
        frequencies_Hz=list(map(float, frequencies)),
        rejected=result.get("failure") == "reference_not_stationary",
    )
    return result


def refusal(data, name):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.residual_system import ResidualDomainError

    spec = native_spec(data, 80)
    state = spec.initial_state()
    before = snapshot(state)
    frequencies = [0.01]
    stage = "input-construction"
    try:
        if name == "empty-frequency":
            frequencies = []
        elif name == "zero-frequency":
            frequencies = [0.0]
        elif name == "negative-frequency":
            frequencies = [-0.01]
        elif name == "nan-frequency":
            frequencies = [float("nan")]
        elif name == "missing-inventory":
            amounts = dict(state.amounts_mol)
            del amounts[next(iter(amounts))]
            state = replace(state, amounts_mol=amounts)
        elif name == "missing-coordinates":
            state = replace(state, coordinates={})
        elif name == "negative-inventory":
            amounts = dict(state.amounts_mol)
            amounts[next(iter(amounts))] = -1e-6
            state = replace(state, amounts_mol=amounts)
        elif name == "voltage-boundary":
            from chemworld.physchem.battery_composite import composite_fields

            boundary = composite_fields(spec, state, current_A=0.0)["voltage_V"]
            spec = replace(spec, minimum_voltage_V=boundary)
        else:
            raise ValueError("unknown fixed refusal")
        before = snapshot(state)
        stage = "analysis"
        battery_impedance(spec, frequencies, state=state)
        result = {"rejected": False}
    except (ValueError, ResidualDomainError) as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {
        "implementation": "refusal",
        "name": name,
        **result,
        "binding": binding(data, "refusal", 80),
        "rejection_stage": stage,
        "state_before": before,
        "state_after": snapshot(state),
        "state_unchanged": before == snapshot(state),
    }
