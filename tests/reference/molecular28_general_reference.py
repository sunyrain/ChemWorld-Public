"""Independent authored PR/regular-solution references for parent 28.

This module never imports native physics. All constants and gauges are explicit.
phasepy is used for genuine LLV and azeotropy; thermo for caloric VLE/envelopes.
These same-equation comparisons do not assert measured material accuracy.
"""

from functools import lru_cache
from math import log

import numpy as np
import phasepy
from scipy.optimize import brentq, least_squares
from thermo import (
    CEOSGas,
    CEOSLiquid,
    ChemicalConstantsPackage,
    FlashVL,
    HeatCapacityGas,
    PropertyCorrelationsPackage,
)
from thermo.eos import PR
from thermo.eos_mix import PRMIX

R = 8.31446261815324
T_REF, P_REF = 298.15, 101325.0
# name, Tc/K, Pc/Pa, omega, mass/kg/mol, ideal Cp/J/mol/K
MATERIALS = {
    "propane": (369.83, 4.248e6, 0.152, 0.04409562, 80.0),
    "butane": (425.12, 3.796e6, 0.2, 0.0581222, 100.0),
    "methane": (190.564, 4.5992e6, 0.01142, 0.016043, 35.0),
    "ethane": (305.322, 4.8722e6, 0.0995, 0.030069, 50.0),
    "water": (647.096, 22.064e6, 0.3443, 0.01801528, 34.0),
    "ethanol": (514.71, 6.268e6, 0.646, 0.04606844, 65.0),
}
FAMILIES = {
    "vle": ("propane", "butane"),
    "envelope": ("methane", "ethane"),
    "vlle": ("methane", "butane", "water"),
    "azeotrope": ("ethanol", "water"),
}


@lru_cache(maxsize=4)
def thermo_oracle(family="vle"):
    ids = FAMILIES[family]
    values = [MATERIALS[k] for k in ids]
    constants = ChemicalConstantsPackage(
        names=list(ids),
        Tcs=[v[0] for v in values],
        Pcs=[v[1] for v in values],
        omegas=[v[2] for v in values],
        MWs=[1000 * v[3] for v in values],
    )
    cps = [HeatCapacityGas(poly_fit=(100.0, 1000.0, [v[4]])) for v in values]
    correlations = PropertyCorrelationsPackage(constants, HeatCapacityGases=cps, skip_missing=True)
    kwargs = {
        "Tcs": constants.Tcs,
        "Pcs": constants.Pcs,
        "omegas": constants.omegas,
        "kijs": np.zeros((len(ids), len(ids))).tolist(),
    }
    phases = [
        cls(PRMIX, kwargs, HeatCapacityGases=cps, T=T_REF, P=P_REF, zs=[1 / len(ids)] * len(ids))
        for cls in (CEOSGas, CEOSLiquid)
    ]
    flash = FlashVL(constants, correlations, phases[0], phases[1])
    # Pinned source defaults to 1e-5 for Newton dew/bubble compositions, which
    # leaves fugacity errors above this block's independent-reference accuracy.
    flash.DEW_BUBBLE_NEWTON_XTOL = 1e-12
    flash.DEW_BUBBLE_QUASI_NEWTON_XTOL = 1e-12
    return flash


def thermo_state(z, *, family="vle", total=1.0, **inputs):
    source = thermo_oracle(family).flash(zs=list(z), **inputs)
    phases, beta = [], []
    if source.gas is not None and source.VF > 1e-12:
        phases.append(source.gas)
        beta.append(source.VF)
    if source.liquids and source.VF < 1 - 1e-12:
        phases.append(source.liquid0)
        beta.append(1 - source.VF)
    mass = total * sum(x * MATERIALS[k][3] for x, k in zip(z, FAMILIES[family], strict=True))
    volume = total * source.V()
    return {
        "source": "thermo.PRMixture/FlashVL",
        "T": source.T,
        "P": source.P,
        "H": total * source.H(),
        "S": total * source.S(),
        "U": total * source.U(),
        "V": volume,
        "mass": mass,
        "vapor_fraction": source.VF,
        "amounts": [[total * b * x for x in p.zs] for p, b in zip(phases, beta, strict=True)],
        "densities": [1 / p.V() for p in phases],
    }


def thermo_boundary(z, kind, *, family="vle", **inputs):
    source = thermo_oracle(family).flash(zs=list(z), VF=0 if kind == "bubble" else 1, **inputs)
    if source.gas is None or source.liquid0 is None:
        raise ValueError("source boundary omitted its incipient phase")
    return {
        "T": source.T,
        "P": source.P,
        "x": source.liquid0.zs,
        "y": source.gas.zs,
        "rhoL": 1 / source.liquid0.V(),
        "rhoV": 1 / source.gas.V(),
        "source_chemical_potential_error_RT": float(
            max(
                abs(
                    np.log(source.liquid0.zs)
                    + source.liquid0.lnphis()
                    - np.log(source.gas.zs)
                    - source.gas.lnphis()
                )
            )
        ),
    }


def thermo_fixed_volume(z, temperature, volume, *, total=1.0):
    # FlashVL's direct T,V branch is not supported reliably by the pinned source.
    # A scalar pressure closure around its actual TP flashes is independent.
    pressure = brentq(
        lambda p: thermo_state(z, total=total, T=temperature, P=p)["V"] - volume,
        1e5,
        2e6,
        xtol=1e-6,
    )
    return thermo_state(z, total=total, T=temperature, P=pressure)


def mixture_critical_reference():
    import teqp

    values = [MATERIALS[k] for k in FAMILIES["envelope"]]
    model = teqp.canonical_PR(
        [v[0] for v in values], [v[1] for v in values], [v[2] for v in values]
    )
    x = np.array([0.8, 0.2])
    calls = []

    def equations(q):
        t, rho = np.exp(q)
        values = np.asarray(model.get_criticality_conditions(t, rho * x))
        scaled = values * np.array([rho / (R * t), rho * rho / (R * t)])
        calls.append({"T": float(t), "rho": float(rho), "residual": scaled.tolist()})
        return scaled

    root = least_squares(
        equations,
        np.log([230.0, 10000.0]),
        bounds=np.log([[200.0, 5000.0], [260.0, 18000.0]]),
        xtol=1e-12,
        gtol=1e-12,
        ftol=1e-12,
        max_nfev=300,
    )
    t, rho = np.exp(root.x)
    return {
        "success": bool(root.success and max(abs(root.fun)) < 2e-8),
        "T": float(t),
        "rho": float(rho),
        "P": float(rho * R * t * (1 + model.get_Ar01(t, rho, x))),
        "source_parameters": model.get_meta(),
        "evaluations": calls,
    }


def pure_nearcritical_reference(fraction):
    tc, pc, omega, _, _ = MATERIALS["propane"]
    source = PR(Tc=tc, Pc=pc, omega=omega, T=300.0, P=1e5)
    temperature = tc * fraction
    return {
        "T": temperature,
        "P": source.Psat(temperature, polish=True),
        "rhoL": 1 / source.V_l_sat(temperature),
        "rhoV": 1 / source.V_g_sat(temperature),
        "x": [1.0],
        "y": [1.0],
    }


@lru_cache(maxsize=2)
def phasepy_oracle(family="vlle"):
    ids = FAMILIES[family]
    components = [
        phasepy.component(
            name=k,
            Tc=MATERIALS[k][0],
            Pc=MATERIALS[k][1] / 1e5,
            w=MATERIALS[k][2],
            Mw=1000 * MATERIALS[k][3],
        )
        for k in ids
    ]
    mixture = components[0] + components[1]
    for component in components[2:]:
        mixture = mixture + component
    interactions = np.zeros((len(ids), len(ids)))
    if family == "vlle":
        interactions[0, 2] = interactions[2, 0] = 0.5
        interactions[1, 2] = interactions[2, 1] = 0.5
    mixture.kij_cubic(interactions)
    return phasepy.preos(mixture)


def vlle_state(temperature, pressure, z=(0.3, 0.3, 0.4)):
    eos = phasepy_oracle()
    source = phasepy.equilibrium.vlle(
        np.array([0.005, 0.005, 0.99]),
        np.array([0.1, 0.89, 0.01]),
        np.array([0.98, 0.01, 0.01]),
        np.asarray(z),
        temperature,
        pressure / 1e5,
        eos,
        full_output=True,
    )
    active = [
        (b, x, v, s)
        for b, x, v, s in zip(source.beta, source.X, source.v, source.states, strict=True)
        if b > 1e-10
    ]
    cp = np.array([MATERIALS[k][4] for k in FAMILIES["vlle"]])
    h, entropy, volume = 0.0, 0.0, 0.0
    for b, x, v, state in active:
        h += b * (
            float(x @ cp) * (temperature - T_REF)
            + eos.EnthalpyR(x, temperature, pressure / 1e5, state, v0=v)
        )
        entropy += b * (
            float(x @ cp) * log(temperature / T_REF)
            - 8.314 * log(pressure / P_REF)
            - 8.314 * float(x @ np.log(x))
            + eos.EntropyR(x, temperature, pressure / 1e5, state, v0=v)
        )
        volume += b * v * 1e-6
    return {
        "source": "phasepy.preos/vlle; R=8.314",
        "T": temperature,
        "P": pressure,
        "H": h,
        "S": entropy,
        "U": h - pressure * volume,
        "V": volume,
        "mass": sum(x * MATERIALS[k][3] for x, k in zip(z, FAMILIES["vlle"], strict=True)),
        "vapor_fraction": float(source.beta[2]),
        "amounts": [(b * x).tolist() for b, x, _, _ in active],
        "densities": [1e6 / v for _, _, v, _ in active],
        "source_error_inner": float(source.error_inner),
        "source_error_outer": float(source.error_outer),
        "source_all_beta": source.beta.tolist(),
        "source_all_tetha": source.tetha.tolist(),
    }


def azeotrope_reference():
    eos = phasepy_oracle("azeotrope")

    def bubble(x):
        return phasepy.equilibrium.bubblePy(
            np.array([x, 1 - x]), 1.0, np.array([x, 1 - x]), 350.0, eos, full_output=True
        )

    x = brentq(lambda x: bubble(x).Y[0] - x, 0.55, 0.60, xtol=2e-12)
    result = bubble(x)
    return {
        "T": 350.0,
        "P": result.P * 1e5,
        "x": [x, 1 - x],
        "y": result.Y.tolist(),
        "rhoL": 1e6 / result.v1,
        "rhoV": 1e6 / result.v2,
    }


def regular_state(z, temperature, pressure=1e5, *, omega=9000.0, total=1.0):
    """Symmetric ternary common tangents, with explicit one/two/three topology.

    The fixed matrix uses equal or two-equal feeds, so no general minimizer is
    needed. Select the lowest analytic Gibbs candidate, not a native TPD result.
    """
    z = np.asarray(z, dtype=float)
    a = omega / (R * temperature)

    def gibbs(x):
        return float(
            sum(y * log(y) for y in x if y > 0)
            + a * sum(x[i] * x[j] for i in range(3) for j in range(i + 1, 3))
        )

    candidates = [(gibbs(z), np.array([z]), np.array([1.0]))]
    # The stable three-phase root lies above 2/3 at the first-order transition.
    if a > 4 * log(2):
        big = brentq(
            lambda x: log(2 * x / (1 - x)) - a * (3 * x - 1) / 2, 2 / 3, 1 - 1e-13, xtol=2e-14
        )
        small = (1 - big) / 2
        compositions = np.full((3, 3), small)
        np.fill_diagonal(compositions, big)
        beta = np.linalg.solve(compositions.T, z)
        if beta.min() >= -1e-12:
            candidates.append(
                (
                    sum(b * gibbs(x) for b, x in zip(beta, compositions, strict=True)),
                    compositions,
                    beta,
                )
            )
    for minor in range(3):
        major = [i for i in range(3) if i != minor]
        c = z[minor]
        width = 1 - c
        if a * width <= 2:
            continue
        small = brentq(
            lambda x, width=width: log(x / (width - x)) + a * (width - 2 * x),
            1e-16,
            width * (0.5 - 1e-9),
            xtol=2e-15,
        )
        big = width - small
        fraction = (z[major[0]] - small) / (big - small)
        if not -1e-12 <= fraction <= 1 + 1e-12:
            continue
        x, y = np.zeros(3), np.zeros(3)
        x[minor] = y[minor] = c
        x[major] = [big, small]
        y[major] = [small, big]
        beta = np.array([fraction, 1 - fraction])
        compositions = np.array([x, y])
        candidates.append(
            (sum(b * gibbs(w) for b, w in zip(beta, compositions, strict=True)), compositions, beta)
        )
    _, xs, beta = min(candidates, key=lambda row: row[0])
    cp = np.array([80.0, 90.0, 100.0])
    h, entropy = 0.0, 0.0
    for b, x in zip(beta, xs, strict=True):
        h += (
            total
            * b
            * (
                float(x @ cp) * (temperature - T_REF)
                + omega * sum(x[i] * x[j] for i in range(3) for j in range(i + 1, 3))
            )
        )
        entropy += (
            total
            * b
            * (
                float(x @ cp) * log(temperature / T_REF)
                - R * log(pressure / P_REF)
                - R * sum(y * log(y) for y in x if y > 0)
            )
        )
    volume = total * R * temperature / pressure
    return {
        "source": "analytic symmetric regular-solution common tangents",
        "T": temperature,
        "P": pressure,
        "H": h,
        "S": entropy,
        "U": h - pressure * volume,
        "V": volume,
        "mass": total * float(z @ np.array([0.04, 0.05, 0.06])),
        "amounts": [(total * b * x).tolist() for b, x in zip(beta, xs, strict=True)],
        "densities": [pressure / (R * temperature)] * len(xs),
    }
