"""Continuous spherical diffusion and parallel-electrode linear solution.

Only authored scalar inputs and NumPy are used; no native model or FV code.
"""

import numpy as np
from battery_composite_impedance_cases import FREQUENCIES, PHASES, F, R, binding, response_order


def analytic_spectrum(data, frequencies=FREQUENCIES):
    phases = {p["phase_id"]: p for p in data["phases"]}
    omega = 2 * np.pi * np.asarray(frequencies)
    impedances, green = {}, {}
    for name, phase in phases.items():
        radius, d = phase["radius_m"], phase["diffusivity_m2_s"]
        volume, cmax = phase["active_volume_m3"], phase["maximum_concentration_mol_m3"]
        theta = phase["initial_fraction"]
        area = 3 * volume / radius
        j0 = phase["exchange_current_scale_A_m2"] * np.sqrt(theta * (1 - theta))
        s = np.sqrt(1j * omega * radius**2 / d)
        green[name] = radius / (F * area * d * cmax * (s / np.tanh(s) - 1))
        slope = sum(i * c * theta ** (i - 1) for i, c in enumerate(phase["polynomial"]) if i)
        impedances[name] = R * data["temperature_K"] / (F * area * j0) - slope * green[name]
    parallel = 1 / (1 / impedances["host_a"] + 1 / impedances["host_b"])
    q = {
        "host_a": parallel / impedances["host_a"],
        "host_b": parallel / impedances["host_b"],
        "cathode": -np.ones(len(omega)),
    }
    columns = [data["series_resistance_ohm"] + parallel + impedances["cathode"]]
    for name in PHASES:
        amount = -q[name] / (1j * omega * F)
        columns.extend(
            (
                q[name],
                -green[name] * q[name],
                amount
                / (phases[name]["maximum_concentration_mol_m3"] * phases[name]["active_volume_m3"]),
                amount,
            )
        )
    values = np.column_stack(columns)
    return {
        "implementation": "analytic",
        "single_material": False,
        "grid": None,
        "binding": binding(data, "analytic", None),
        "frequencies_Hz": list(map(float, frequencies)),
        "response_order": response_order(data),
        "success": bool(np.all(np.isfinite(values))),
        "faraday_C_mol": F,
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
