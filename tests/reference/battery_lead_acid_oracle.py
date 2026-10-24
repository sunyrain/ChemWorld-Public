"""Independent six-species stoichiometry, Bode/BV formulas and CC work quadrature."""

import numpy as np
from battery_lead_acid_cases import STOICHIOMETRY, F, R
from scipy.integrate import quad
from scipy.optimize import brentq


def amounts(data, charge_C, faraday=F):
    return np.asarray(data["initial_amounts_mol"]) + STOICHIOMETRY * charge_C / (2 * faraday)


def fields(data, charge_C, current_A, faraday=F):
    n = amounts(data, charge_C, faraday)
    vp, vo, vs, ve, vw = data["molar_volumes_m3_mol"]
    geometry = np.asarray(data["lengths_m"]) * data["area_m2"]
    inact = data["inactive_solid_volumes_m3"]
    pore = np.array(
        [
            geometry[0] - inact[0] - n[0] * vp - n[1] * vs,
            geometry[1] * data["initial_porosity"][1],
            geometry[2] - inact[1] - n[2] * vo - n[3] * vs,
        ]
    )
    ca, cw = n[4] / sum(pore), n[5] / sum(pore)
    molality = ca / (cw * data["water_molar_mass_kg_mol"])
    logm = np.log10(molality)
    un = -0.294 - 0.074 * logm - 0.030 * logm**2 - 0.031 * logm**3 - 0.012 * logm**4
    up = 1.628 + 0.074 * logm + 0.033 * logm**2 + 0.043 * logm**3 + 0.022 * logm**4
    cref = data["reference_acid_concentration_mol_m3"]
    cwref = (1 - cref * ve) / vw
    jn = data["negative_exchange_current_density_A_m2"] * ca / cref
    jp = data["positive_exchange_current_density_A_m2"] * (ca / cref) ** 2 * cw / cwref
    reaction_areas = geometry[[0, 2]] * data["reaction_area_densities_m_inv"]
    etan = (
        R * data["temperature_K"] / faraday * np.arcsinh(current_A / (2 * reaction_areas[0] * jn))
    )
    etap = (
        R * data["temperature_K"] / faraday * np.arcsinh(-current_A / (2 * reaction_areas[1] * jp))
    )
    return {
        "voltage_V": up - un + etap - etan,
        "acid_concentration_mol_m3": ca,
        "water_concentration_mol_m3": cw,
        "molality_mol_kg": molality,
        "porosities": pore / geometry,
        "pore_volume_m3": sum(pore),
        "negative_ocp_V": un,
        "positive_ocp_V": up,
        "negative_overpotential_V": etan,
        "positive_overpotential_V": etap,
        "negative_j0_A_m2": jn,
        "positive_j0_A_m2": jp,
        "amounts_mol": n,
    }


def stopping_time(data, current_A):
    n0 = data["initial_amounts_mol"]
    kind = data["expected"]
    vp, vo, vs, _ve, _vw = data["molar_volumes_m3_mol"]
    if kind == "acid-threshold":
        pore0 = fields(data, 0, 0)["pore_volume_m3"]
        cmin = data["minimum_acid_concentration_mol_m3"]
        extent = (n0[4] - cmin * pore0) / (2 + cmin * (vp + vo - 2 * vs))
    elif kind == "negative:Pb:depleted":
        extent = n0[0]
    elif kind == "positive:PbO2:depleted":
        extent = n0[2]
    elif kind == "negative:PbSO4:depleted":
        extent = -n0[1]
    elif kind == "negative:minimum-porosity":
        extent = (
            (data["initial_porosity"][0] - data["minimum_porosity"])
            * data["area_m2"]
            * data["lengths_m"][0]
            / (vs - vp)
        )
    elif kind == "minimum-voltage":
        return brentq(
            lambda t: (
                fields(data, t * current_A, current_A)["voltage_V"] - data["minimum_voltage_V"]
            ),
            0,
            data["waveform"][0][0],
            xtol=1e-9,
        )
    else:
        raise ValueError("No authored analytic stop for " + kind)
    return 2 * F * extent / current_A


def cc_work(data, initial_charge_C, final_charge_C, current_A, faraday=F):
    if current_A == 0.0:
        return 0.0
    return quad(
        lambda q: fields(data, q, current_A, faraday)["voltage_V"],
        initial_charge_C,
        final_charge_C,
        epsabs=1e-9,
        epsrel=1e-11,
    )[0]
