"""Independent scalar MSMR material and continuous-sphere linear response."""

import numpy as np
from battery_msmr_impedance_cases import (
    FREQUENCIES,
    F,
    R,
    binding,
    response_order,
    response_units,
)
from scipy.optimize import brentq
from scipy.special import expit


def material(electrode, temperature, ce, *, faraday=F, gas_constant=R):
    rows = np.asarray(electrode["rows"], float)
    weight, u0, width, exchange, alpha = rows.T
    factor = faraday / (gas_constant * temperature)

    def occupation(u):
        return weight * expit(-factor * (u - u0) / width)

    u = brentq(
        lambda v: sum(occupation(v)) - electrode["initial_fraction"],
        min(u0) - 5,
        max(u0) + 5,
        xtol=1e-14,
        rtol=1e-14,
    )
    x = occupation(u)
    dx = -factor / width * x * (1 - x / weight)
    j0 = exchange * x**width * np.exp(factor * (1 - alpha) * (u - u0)) * (ce / 1000) ** (1 - alpha)
    dchem = -electrode["diffusivity_m2_s"] * factor * sum(x) * (1 - sum(x)) / sum(dx)
    return {
        "potential_V": float(u),
        "x": x.tolist(),
        "dx": dx.tolist(),
        "j0": j0.tolist(),
        "chemical_diffusivity_m2_s": float(dchem),
    }


def analytic_spectrum(data, frequencies=FREQUENCIES):
    omega = 2 * np.pi * np.asarray(frequencies)
    ones = np.ones(len(omega))
    outputs = {"current_A_A": ones, "charge_C_A": 1 / (1j * omega)}
    z = np.full(len(omega), data["resistance_ohm"], complex)
    materials = {}
    for e, sign in zip(data["electrodes"], (1, -1), strict=True):
        name = e["name"]
        m = material(e, data["temperature_K"], data["electrolyte_concentration_mol_m3"])
        materials[name] = m
        radius, cmax, volume = (
            e["radius_m"],
            e["maximum_concentration_mol_m3"],
            e["active_volume_m3"],
        )
        area, dc = 3 * volume / radius, m["chemical_diffusivity_m2_s"]
        xu, j0 = sum(m["dx"]), np.asarray(m["j0"])
        s = np.sqrt(1j * omega * radius**2 / dc)
        green = radius / (F * area * dc * cmax * (s / np.tanh(s) - 1))
        rct = R * data["temperature_K"] / (F * area * sum(j0))
        surface = -sign * green
        u = surface / xu
        amount = -sign / (1j * omega * F)
        z += rct - green / xu
        outputs.update(
            {
                f"{name}:U": u,
                f"{name}:eta": sign * rct * ones,
                f"{name}:surface": surface,
                f"{name}:source_surface": surface,
                f"{name}:average": amount / (volume * cmax),
                f"{name}:amount": amount,
            }
        )
        for j, slope in enumerate(m["dx"]):
            outputs[f"{name}:kinetic_x{j}"] = slope * u
            outputs[f"{name}:source_x{j}"] = slope * u
            outputs[f"{name}:current{j}"] = sign * j0[j] / sum(j0) * ones
    v0 = materials["positive"]["potential_V"] - materials["negative"]["potential_V"]
    outputs.update(Z_ohm=z, work_J_A=v0 / (1j * omega))
    values = np.column_stack([outputs[k] for k in response_order(data)])
    return {
        "implementation": "analytic",
        "grid": None,
        "binding": binding(data, "analytic", None),
        "frequencies_Hz": list(map(float, frequencies)),
        "response_order": response_order(data),
        "response_units": response_units(data),
        "success": bool(np.all(np.isfinite(values))),
        "faraday_C_mol": F,
        "gas_constant_J_mol_K": R,
        "rest_voltage_V": v0,
        "material_static": materials,
        "points": [
            {
                "frequency_Hz": float(f),
                "success": True,
                "response_real": row.real.tolist(),
                "response_imag": row.imag.tolist(),
            }
            for f, row in zip(frequencies, values, strict=True)
        ],
    }
