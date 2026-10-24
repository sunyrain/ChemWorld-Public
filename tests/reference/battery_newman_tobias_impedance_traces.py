"""Read original native/source NT phasors without a new state or closure."""

import numpy as np
from battery_composite_impedance_traces import snapshot
from battery_msmr_impedance_traces import DEFINITION, source_linearization
from battery_newman_tobias_cases import native_spec
from battery_newman_tobias_impedance_cases import (
    FREQUENCIES,
    F,
    R,
    binding,
    refusal_input,
    response_order,
    response_units,
)


def native_observations(spec, data, variables, raw):
    from chemworld.physchem.battery_porous_electrical import linear_electrical

    ids = {v["name"]: i for i, v in enumerate(variables)}
    raw = np.asarray(raw)
    scalar = raw.ndim == 1
    if scalar:
        raw = raw[None, :]
    out = []
    for row in raw:
        mapping = {k: row[ids[k]] for k in spec.algebraic_names}
        real = linear_electrical(
            spec.geometry,
            spec.reaction_keys,
            spec.resistance_ohm,
            {k: v.real for k, v in mapping.items()},
        )
        imag = linear_electrical(
            spec.geometry,
            spec.reaction_keys,
            spec.resistance_ohm,
            {k: v.imag for k, v in mapping.items()},
        )
        q, ie, pe, ps = [
            np.asarray(a) + 1j * np.asarray(b) for a, b in zip(real[:4], imag[:4], strict=True)
        ]
        iss = [np.asarray(a) + 1j * np.asarray(b) for a, b in zip(real[4], imag[4], strict=True)]
        current = row[ids["battery:current_A"]]
        output = {
            "Z_ohm": -row[ids["battery:voltage_V"]],
            "current": current,
            "charge": row[ids["discharged_charge_C"]],
            "work": row[ids["delivered_work_J"]],
        }
        for e in data["electrodes"]:
            n = e["name"]
            amount = row[ids[n + ":Li"]]
            theta = amount / getattr(spec, n).capacity_mol
            output.update(
                {
                    n + ":amount": amount,
                    n + ":theta": theta,
                    n + ":U": spec.ocv(n, e["initial_fraction"], derivative=True) * theta,
                }
            )
        offset = eo = 0
        for region, (n, count) in enumerate(
            zip(("negative", "separator", "positive"), spec.geometry.cells, strict=True)
        ):
            for i in range(count):
                output[f"{n}:phi_e:{i}"] = pe[offset + i]
            for i in range(count + 1):
                output[f"{n}:ie:{i}"] = ie[offset + i] * spec.geometry.area_m2
            if region != 1:
                p = getattr(spec, n)
                e = data["electrodes"][region // 2]
                j0 = p.exchange_current_scale_A_m2 * np.sqrt(
                    spec.electrolyte_concentration_mol_m3
                    / spec.reference_electrolyte_concentration_mol_m3
                    * e["initial_fraction"]
                    * (1 - e["initial_fraction"])
                )
                for i in range(count):
                    j = q[eo + i] / (p.area_m2 / count)
                    output[f"{n}:phi_s:{i}"] = ps[eo + i]
                    output[f"{n}:eta:{i}"] = R * spec.temperature_K / (F * j0) * j
                    output[f"{n}:j:{i}"] = j
                    output[f"{n}:q:{i}"] = q[eo + i]
                for i in range(count + 1):
                    output[f"{n}:is:{i}"] = iss[region // 2][i] * spec.geometry.area_m2
                eo += count
            offset += count
        out.append([output[k] for k in response_order(spec.geometry.cells)])
    array = np.asarray(out)
    return array[0] if scalar else array


def native_spectrum(data, cells, *, frequencies=FREQUENCIES):
    from chemworld.physchem.battery_impedance import battery_impedance, battery_impedance_system
    from chemworld.physchem.battery_newman_tobias import newman_tobias_readout
    from chemworld.physchem.residual_system import ResidualEvaluator

    spec = native_spec(data, cells)
    state = spec.initial_state()
    before = snapshot(state)
    system, initial = battery_impedance_system(spec, state)
    residual = ResidualEvaluator(system).value(state.time_s, initial, np.zeros(len(initial)))
    variables = [{"name": v.name, "unit": v.unit, "scale": v.scale} for v in system.variables]
    spectrum = battery_impedance(spec, frequencies, state=state)
    points = []
    for point in spectrum.response.points:
        record = {
            "frequency_Hz": point.frequency_Hz,
            "success": point.success,
            "relative_linear_residual": point.relative_linear_residual,
            "failure_reason": point.failure_reason,
        }
        if point.response is not None:
            raw = point.response[:, 0]
            output = native_observations(spec, data, variables, raw)
            record.update(
                response_real=output.real.tolist(),
                response_imag=output.imag.tolist(),
                state_response_real=raw.real.tolist(),
                state_response_imag=raw.imag.tolist(),
            )
        points.append(record)
    return {
        "implementation": "native",
        "grid": list(cells),
        "binding": binding(data, "native", cells),
        "frequencies_Hz": list(map(float, frequencies)),
        "response_order": response_order(cells),
        "response_units": response_units(cells),
        "definition": DEFINITION,
        "variables": variables,
        "initial_state": initial.tolist(),
        "stationary_residual": residual.tolist(),
        "residual_scales": system.residual_scales.tolist(),
        "state_before": before,
        "state_after": snapshot(state),
        "state_unchanged": before == snapshot(state),
        "faraday_C_mol": F,
        "gas_constant_J_mol_K": R,
        "rest_voltage_V": newman_tobias_readout(spec, state, 0.0).voltage_V,
        "success": spectrum.success,
        "points": points,
        "diagnostics": spectrum.response.diagnostics,
    }


def source_output_names(cells):
    return ["Voltage [V]", *["eis:" + name for name in response_order(cells)[1:]]]


def source_spectrum(data, cells, *, frequencies=FREQUENCIES):
    import pybamm
    from battery_frequency_pybamm import frequency_from_simulation
    from battery_newman_tobias_pybamm import build_reference

    sim = build_reference(data, cells)
    original = sim.model

    def original_names():
        return {
            "rhs": [v.name for v in original.rhs],
            "algebraic": [v.name for v in original.algebraic],
            "initial_conditions": [v.name for v in original.initial_conditions],
            "variables": list(original.variables),
        }

    retained = original_names()
    model = sim.built_model
    v = model.variables
    output = {
        "current": v["Current [A]"],
        "charge": 3600 * v["Discharge capacity [A.h]"],
        "work": 3600 * v["Discharge energy [W.h]"],
    }
    for e in data["electrodes"]:
        n = e["name"]
        theta = v[f"qualification:{n}:fraction"]
        output[n + ":amount"] = v[f"qualification:{n}:amount"]
        output[n + ":theta"] = theta
        output[n + ":U"] = sum(c * theta**i for i, c in enumerate(e["polynomial"]))
    pe = v["Electrolyte potential [V]"]
    ie = v["Electrolyte current density [A.m-2]"] * data["area_m2"]
    offset = 0
    for region, (n, count) in enumerate(
        zip(("negative", "separator", "positive"), cells, strict=True)
    ):
        for i in range(count):
            output[f"{n}:phi_e:{i}"] = pybamm.Index(pe, offset + i)
        for i in range(count + 1):
            output[f"{n}:ie:{i}"] = pybamm.Index(ie, offset + i)
        if region != 1:
            e = data["electrodes"][region // 2]
            ps = v[n.capitalize() + " electrode potential [V]"]
            iss = v[n.capitalize() + " electrode current density [A.m-2]"] * data["area_m2"]
            j = v[n.capitalize() + " electrode interfacial current density [A.m-2]"]
            eta = v[n.capitalize() + " electrode reaction overpotential [V]"]
            active_area = 3 * e["active_volume_m3"] / e["radius_m"]
            for i in range(count):
                output[f"{n}:phi_s:{i}"] = pybamm.Index(ps, i)
                output[f"{n}:eta:{i}"] = pybamm.Index(eta, i)
                output[f"{n}:j:{i}"] = pybamm.Index(j, i)
                output[f"{n}:q:{i}"] = pybamm.Index(j, i) * active_area / count
            for i in range(count + 1):
                output[f"{n}:is:{i}"] = pybamm.Index(iss, i)
        offset += count
    model.variables.update({"eis:" + k: value for k, value in output.items()})
    names = source_output_names(cells)
    result = frequency_from_simulation(
        sim,
        frequencies,
        family="NewmanTobias uniform profile",
        grid={"thickness_cells": list(cells), "mesh": "uniform"},
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
                "scale": float(sim.parameter_values.process_symbol(variable.scale).evaluate()),
                "reference": float(
                    sim.parameter_values.process_symbol(variable.reference).evaluate()
                ),
                "domains": variable.domains,
            }
        )
    result.update(
        implementation="reference",
        grid=list(cells),
        binding=binding(data, "reference", cells),
        frequencies_Hz=list(map(float, frequencies)),
        response_units=response_units(cells),
        source_state_layout=layout,
        source_original_model=retained,
        source_retained_model=original_names(),
        source_linearization=source_linearization(sim, names, result["initial_state"], layout),
        faraday_C_mol=float(pybamm.constants.F.evaluate()),
        gas_constant_J_mol_K=float(pybamm.constants.R.evaluate()),
        rest_voltage_V=result["initial_outputs"][0],
        source_material_mapping={
            "net_OCV_endpoint_barrier": "explicitly compensated by original NT builder",
            "ce_reference_mol_m3": data["reference_electrolyte_concentration_mol_m3"],
            "solid_bulk_conductivity_S_m": [
                sigma / (1 - data["porosities"][region]) ** bs
                for sigma, region, bs in zip(
                    data["solid_conductivity_S_m"], (0, 2), data["solid_bruggeman"], strict=True
                )
            ],
        },
    )
    return result


def refusal(data, name, cells=(24, 12, 24)):
    from chemworld.physchem.battery_impedance import battery_impedance
    from chemworld.physchem.residual_system import ResidualDomainError

    spec, state, freq = refusal_input(data, name, cells)
    before = snapshot(state)
    try:
        battery_impedance(spec, freq, state=state)
        result = {"rejected": False}
    except (ValueError, ResidualDomainError) as exc:
        result = {"rejected": True, "exception_type": type(exc).__name__, "exception": str(exc)}
    return {
        "implementation": "refusal",
        "name": name,
        "grid": list(cells),
        "binding": binding(data, "refusal", cells, name),
        **result,
        "state_before": before,
        "state_after": snapshot(state),
        "state_unchanged": before == snapshot(state),
        "specification": spec.to_dict(),
        "frequencies_Hz": freq,
    }
