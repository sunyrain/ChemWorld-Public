"""Actual PyBaMM25.4.2 LOQS with explicitly matched synthetic volume parameters."""

import numpy as np
from battery_lead_acid_cases import PRECISIONS, SOURCE_ROOT_TOL


def build_reference(data, precision):
    import pybamm

    if pybamm.__version__ != "25.4.2":
        raise RuntimeError("LOQS reference requires PyBaMM25.4.2")

    def finite_source(variables):
        current = variables["Current [A]"]
        request = current + data["maximum_current_A"] * (
            variables["Voltage [V]"] - pybamm.InputParameter("LOQS voltage target")
        )
        return current - pybamm.maximum(
            -data["maximum_current_A"], pybamm.minimum(data["maximum_current_A"], request)
        )

    options = {
        "surface form": "false",
        "thermal": "isothermal",
        "dimensionality": 0,
        "hydrolysis": "false",
        "convection": "none",
        "calculate discharge energy": "true",
    }
    if data["waveform"][0][2] == "V":
        options["operating mode"] = finite_source
    model = pybamm.lead_acid.LOQS(options, build=False)
    # LOQS overrides the base circuit setup and omits the optional counters.
    # Add only the released non-feedback Q/W accumulator; no native equations.
    model.submodels["qualification discharge and throughput"] = (
        pybamm.external_circuit.DischargeThroughput(model.param, model.options)
    )
    model.build_model()
    parameters = pybamm.ParameterValues("Sulzer2019")
    vp, vo, vs, ve, vw = data["molar_volumes_m3_mol"]
    cref = data["reference_acid_concentration_mol_m3"]
    cwref = (1 - cref * ve) / vw

    def negative_exchange(ce, temperature):
        return data["negative_exchange_current_density_A_m2"] * ce / cref

    def positive_exchange(ce, temperature):
        cw = (1 - ce * ve) / vw
        return data["positive_exchange_current_density_A_m2"] * (ce / cref) ** 2 * cw / cwref

    updates = {
        "Electrode width [m]": 0.1,
        "Electrode height [m]": data["area_m2"] / 0.1,
        "Number of electrodes connected in parallel to make a cell": 1.0,
        "Number of cells connected in series to make a battery": 1.0,
        "Initial temperature [K]": data["temperature_K"],
        "Ambient temperature [K]": data["temperature_K"],
        "Reference temperature [K]": data["temperature_K"],
        "Initial concentration in electrolyte [mol.m-3]": cref,
        "Molar volume of lead [m3.mol-1]": vp,
        "Molar volume of lead-dioxide [m3.mol-1]": vo,
        "Molar volume of lead sulfate [m3.mol-1]": vs,
        "Partial molar volume of water [m3.mol-1]": vw,
        "Partial molar volume of cations [m3.mol-1]": 1.35e-5,
        "Partial molar volume of anions [m3.mol-1]": ve - 1.35e-5,
        "Molar mass of water [kg.mol-1]": data["water_molar_mass_kg_mol"],
        "Negative electrode surface area to volume ratio [m-1]": data[
            "reaction_area_densities_m_inv"
        ][0],
        "Positive electrode surface area to volume ratio [m-1]": data[
            "reaction_area_densities_m_inv"
        ][1],
        "Negative electrode exchange-current density [A.m-2]": negative_exchange,
        "Positive electrode exchange-current density [A.m-2]": positive_exchange,
        "Lower voltage cut-off [V]": data["minimum_voltage_V"],
        "Upper voltage cut-off [V]": data["maximum_voltage_V"],
        "Contact resistance [Ohm]": 0.0,
        "Initial oxygen concentration [mol.m-3]": 0.0,
        "Initial State of Charge": 1.0,
        "Current function [A]": "[input]",
    }
    for index, name in enumerate(("negative electrode", "separator", "positive electrode")):
        updates[name.capitalize() + " thickness [m]"] = data["lengths_m"][index]
        updates["Maximum porosity of " + name] = data["initial_porosity"][index]
    parameters.update(updates, check_already_exists=False)
    ce = model.variables["X-averaged electrolyte concentration [mol.m-3]"]
    eps = [
        model.variables[f"X-averaged {region} porosity"]
        for region in ("negative electrode", "separator", "positive electrode")
    ]
    pore = data["area_m2"] * sum(
        length * porosity for length, porosity in zip(data["lengths_m"], eps, strict=True)
    )
    model.variables.update(
        {
            "qualification:water_concentration": (1 - ce * ve) / vw,
            "qualification:molality": ce * vw / ((1 - ce * ve) * data["water_molar_mass_kg_mol"]),
            "qualification:acid_amount": ce * pore,
            "qualification:water_amount": (1 - ce * ve) / vw * pore,
            "qualification:pore_volume": pore,
        }
    )
    return pybamm.Simulation(
        model,
        parameter_values=parameters,
        solver=pybamm.CasadiSolver(mode="safe", **PRECISIONS[precision], root_tol=SOURCE_ROOT_TOL),
    )


def _scalar(solution, name):
    array = np.asarray(solution[name].entries, dtype=float)
    return array.reshape(-1, len(solution.t))[0]


def extract_segment(solution):
    names = (
        "Voltage [V]",
        "Current [A]",
        "Discharge capacity [A.h]",
        "Discharge energy [W.h]",
        "X-averaged electrolyte concentration [mol.m-3]",
        "qualification:water_concentration",
        "qualification:molality",
        "X-averaged negative electrode porosity",
        "X-averaged separator porosity",
        "X-averaged positive electrode porosity",
        "X-averaged negative electrode open-circuit potential [V]",
        "X-averaged positive electrode open-circuit potential [V]",
        "X-averaged negative electrode reaction overpotential [V]",
        "X-averaged positive electrode reaction overpotential [V]",
        "X-averaged negative electrode exchange current density [A.m-2]",
        "X-averaged positive electrode exchange current density [A.m-2]",
    )
    values = np.column_stack([_scalar(solution, name) for name in names])
    values[:, 2:4] *= 3600
    return {
        "times_s": solution.t.tolist(),
        "values": values.tolist(),
        "source_acid_water_amounts_mol": np.column_stack(
            [
                _scalar(solution, "qualification:acid_amount"),
                _scalar(solution, "qualification:water_amount"),
            ]
        ).tolist(),
        "pore_volume_m3": _scalar(solution, "qualification:pore_volume").tolist(),
        "original_source_state_coordinates": np.asarray(solution.y).tolist(),
        "end_time_s": float(solution.t[-1]),
        "termination": solution.termination,
    }


def reference_time(data, precision):
    progress = {"stage": "build", "current_time_s": 0.0, "completed_segments": []}
    try:
        return _reference_time(data, precision, progress)
    except Exception as exc:
        exc.partial_history = progress
        raise


def _reference_time(data, precision, progress):
    simulation = build_reference(data, precision)
    segments = progress["completed_segments"]
    for index, (duration, target, unit) in enumerate(data["waveform"]):
        horizon = data["observation_horizons_s"][index]
        if data["source_stop_semantics"].startswith("author_imposed"):
            duration = horizon
        start = sum(segment[0] for segment in data["waveform"][:index])
        times = np.linspace(0.0, horizon, data["samples_per_segment"])
        # The actual voltage event remains active when the requested interval exceeds its oracle.
        if times[-1] < duration:
            times = np.r_[times, duration]
        progress.update(
            stage="integrate",
            current_time_s=start,
            active_segment={
                "index": index,
                "start_time_s": start,
                "duration_s": duration,
                "target": target,
                "unit": unit,
            },
        )
        solution = simulation.step(
            duration,
            save=False,
            t_eval=times,
            inputs={"LOQS voltage target" if unit == "V" else "Current function [A]": target},
        )
        progress.update(stage="extract", current_time_s=float(solution.t[-1]))
        progress["active_segment"].update(
            returned_times_s=solution.t.tolist(),
            original_source_state_coordinates=np.asarray(solution.y).tolist(),
            termination=solution.termination,
        )
        segments.append(extract_segment(solution))
        if solution.termination != "final time":
            break
    return {
        "implementation": "Actual PyBaMM25.4.2 lead_acid.LOQS",
        "precision": precision,
        "tolerances": PRECISIONS[precision],
        "root_tol": SOURCE_ROOT_TOL,
        "source_stop_semantics": data["source_stop_semantics"],
        "segments": segments,
        "state_scope": "Source ce/porosity/Q/W; no native finite-Pb inventories "
        "or source solid-exhaustion events",
    }
