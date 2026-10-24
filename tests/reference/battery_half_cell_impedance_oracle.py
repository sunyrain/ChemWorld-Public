"""Independent continuous sphere, two BV interfaces and fixed metal resistor."""

import numpy as np
from battery_half_cell_impedance_cases import (
    FREQUENCIES,
    F,
    R,
    binding,
    ocv,
    response_order,
    response_units,
)


def analytic_spectrum(data, frequencies=FREQUENCIES):
    omega = 2 * np.pi * np.asarray(frequencies)
    radius, d, volume, cmax = (
        data[k]
        for k in (
            "radius_m",
            "diffusivity_m2_s",
            "active_volume_m3",
            "maximum_concentration_mol_m3",
        )
    )
    theta, temperature = data["initial_positive_fraction"], data["temperature_K"]
    area = 3 * volume / radius
    j0 = data["exchange_current_scale_A_m2"] * np.sqrt(theta * (1 - theta))
    rm = data["thickness_m"] / (data["conductivity_S_m"] * data["area_m2"])
    rct_m = R * temperature / (F * data["area_m2"] * data["exchange_current_density_A_m2"])
    rct_p = R * temperature / (F * area * j0)
    s = np.sqrt(1j * omega * radius**2 / d)
    green = radius / (F * area * d * cmax * (s / np.tanh(s) - 1))
    amount = 1 / (1j * omega * F)
    ones = np.ones(len(omega))
    values = np.column_stack(
        (
            rm + rct_m + rct_p - ocv(theta, True) * green,
            ones,
            green,
            amount / (volume * cmax),
            amount,
            -amount,
            1 / (1j * omega),
            rct_m * ones,
            -rct_p * ones,
            rm * ones,
        )
    )
    work = ocv(theta) / (1j * omega)
    return {
        "implementation": "analytic",
        "grid": None,
        "binding": binding(data, "analytic", None),
        "frequencies_Hz": list(map(float, frequencies)),
        "response_order": response_order(),
        "response_units": response_units(),
        "success": bool(np.all(np.isfinite(values))),
        "faraday_C_mol": F,
        "metal_mapping_faraday_C_mol": F,
        "rest_voltage_V": float(ocv(theta)),
        "points": [
            {
                "frequency_Hz": float(f),
                "success": True,
                "response_real": row.real.tolist(),
                "response_imag": row.imag.tolist(),
                "work_response_real": float(w.real),
                "work_response_imag": float(w.imag),
            }
            for f, row, w in zip(frequencies, values, work, strict=True)
        ],
    }
