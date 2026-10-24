"""Independent continuous 4x4 constant-coefficient boundary-value solution."""

import numpy as np
from battery_newman_tobias_impedance_cases import (
    FREQUENCIES,
    GRIDS,
    F,
    R,
    binding,
    ocv,
    response_order,
    response_units,
)
from scipy.linalg import expm


def network(data):
    area = data["area_m2"]
    J = 1 / area
    kappas = data["conductivity_S_m"] * np.power(data["porosities"], data["bruggeman"])
    phases = []
    for e, region, sigma in zip(
        data["electrodes"], (0, 2), data["solid_conductivity_S_m"], strict=True
    ):
        length = data["lengths_m"][region]
        active_a = 3 * e["active_volume_m3"] / (e["radius_m"] * area * length)
        j0 = e["exchange_current_scale_A_m2"] * np.sqrt(
            data["electrolyte_concentration_mol_m3"]
            / data["reference_electrolyte_concentration_mol_m3"]
            * e["initial_fraction"]
            * (1 - e["initial_fraction"])
        )
        g = active_a * F * j0 / (R * data["temperature_K"])
        k = kappas[region]
        matrix = np.array(
            [[0, 1 / sigma + 1 / k, 0, -1 / sigma], [g, 0, 0, 0], [0, -1 / k, 0, 0], [0, 0, 0, 0]]
        )
        propagator = expm(matrix * length)
        ie0, iel = (0.0, J) if region == 0 else (J, 0.0)
        eta0 = (iel - propagator[1, 1] * ie0 - propagator[1, 3] * J) / propagator[1, 0]
        phie0 = -eta0 if region == 0 else phases[0]["end"][2] - J * data["lengths_m"][1] / kappas[1]
        initial = np.array([eta0, ie0, phie0, J])
        phases.append(
            {
                "matrix": matrix,
                "initial": initial,
                "end": propagator @ initial,
                "a": active_a,
                "g": g,
                "j0": j0,
                "sigma": sigma,
                "kappa": k,
                "length": length,
            }
        )
    end = phases[1]["end"]
    rnetwork = -(end[0] + end[2])
    xi, wi = np.polynomial.legendre.leggauss(48)
    power = area * J**2 * data["lengths_m"][1] / kappas[1]
    for p in phases:
        y = np.array([expm(p["matrix"] * ((x + 1) * p["length"] / 2)) @ p["initial"] for x in xi])
        density = (
            (J - y[:, 1]) ** 2 / p["sigma"] + y[:, 1] ** 2 / p["kappa"] + p["g"] * y[:, 0] ** 2
        )
        power += area * p["length"] / 2 * np.dot(wi, density)
    return phases, kappas, float(rnetwork), float(power)


def continuous_values(data, cells, frequencies=FREQUENCIES):
    phases, kappa, rnetwork, power = network(data)
    omega = 2 * np.pi * np.asarray(frequencies)
    count = len(omega)
    ones = np.ones(count)
    output = {"current": ones, "charge": 1 / (1j * omega)}
    for e, sign in zip(data["electrodes"], (1, -1), strict=True):
        n = e["name"]
        amount = -sign / (1j * omega * F)
        theta = amount / (e["maximum_concentration_mol_m3"] * e["active_volume_m3"])
        output[n + ":amount"], output[n + ":theta"], output[n + ":U"] = (
            amount,
            theta,
            ocv(e, True) * theta,
        )
    un, up = output["negative:U"], output["positive:U"]
    output["Z_ohm"] = rnetwork + un - up
    v0 = ocv(data["electrodes"][1]) - ocv(data["electrodes"][0])
    output["work"] = v0 / (1j * omega)
    for region, (name, size) in enumerate(
        zip(("negative", "separator", "positive"), cells, strict=True)
    ):
        length = data["lengths_m"][region]
        nodes = (np.arange(size) + 0.5) * length / size
        faces = np.linspace(0, length, size + 1)
        if region == 1:
            pe = phases[0]["end"][2] - nodes / (data["area_m2"] * kappa[1])
            ie = np.ones(size + 1)
            for i in range(size):
                output[f"{name}:phi_e:{i}"] = pe[i] * ones - un
            for i in range(size + 1):
                output[f"{name}:ie:{i}"] = ie[i] * ones
        else:
            p = phases[region // 2]
            y = np.array([expm(p["matrix"] * x) @ p["initial"] for x in nodes])
            yf = np.array([expm(p["matrix"] * x) @ p["initial"] for x in faces])
            ie = yf[:, 1] * data["area_m2"]
            q = np.diff(ie)
            for i in range(size):
                output[f"{name}:phi_e:{i}"] = y[i, 2] * ones - un
                output[f"{name}:phi_s:{i}"] = (y[i, 2] + y[i, 0]) * ones - un + output[name + ":U"]
                output[f"{name}:eta:{i}"] = y[i, 0] * ones
                output[f"{name}:j:{i}"] = p["g"] / p["a"] * y[i, 0] * ones
                output[f"{name}:q:{i}"] = q[i] * ones
            for i in range(size + 1):
                output[f"{name}:ie:{i}"] = ie[i] * ones
                output[f"{name}:is:{i}"] = (1 - ie[i]) * ones
    values = np.column_stack([output[k] for k in response_order(cells)])
    return values, {
        "resistance_ohm": rnetwork,
        "dissipation_W_A2": power,
        "effective_ionic_conductivity_S_m": kappa.tolist(),
        "electrodes": [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
            for p in phases
        ],
        "rest_voltage_V": v0,
    }


def analytic_spectrum(data, frequencies=FREQUENCIES, grids=GRIDS):
    grid_responses = {}
    for label, cells in grids.items():
        array, meta = continuous_values(data, cells, frequencies)
        grid_responses[label] = {
            "cells": list(cells),
            "response_order": response_order(cells),
            "response_units": response_units(cells),
            "response_real": array.real.tolist(),
            "response_imag": array.imag.tolist(),
        }
    main = array[:, : len(response_order())]
    return {
        "implementation": "analytic",
        "grid": None,
        "binding": binding(data, "analytic", None),
        "frequencies_Hz": list(map(float, frequencies)),
        "response_order": response_order(),
        "response_units": response_units(),
        "faraday_C_mol": F,
        "gas_constant_J_mol_K": R,
        "rest_voltage_V": meta["rest_voltage_V"],
        "continuous_network": meta,
        "grid_responses": grid_responses,
        "success": bool(np.all(np.isfinite(main))),
        "points": [
            {
                "frequency_Hz": float(f),
                "success": True,
                "response_real": v.real.tolist(),
                "response_imag": v.imag.tolist(),
            }
            for f, v in zip(frequencies, main, strict=True)
        ],
    }
