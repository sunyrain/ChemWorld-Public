"""Actual pinned CatMAP local mean-field kernels, with independent steady solve.

Molar J/R units are supplied explicitly to the upstream templates. This is a
local subsystem comparison, not a claim to execute CatMAP's descriptor mapper.
"""

from functools import lru_cache
from math import exp, log, prod
from string import Template
from textwrap import indent

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

R = 8.31446261815324
ADS = ("CO_ads", "O_ads", "CO_B", "O_B")
GAS = ("CO", "O2", "CO2", "N2")
REACTIONS = (
    ({"CO": 1, "vac_A": 1}, {"CO_ads": 1}, 1.0),
    ({"O2": 1, "vac_A": 2}, {"O_ads": 2}, 1.0),
    ({"CO_ads": 1, "O_ads": 1}, {"CO2": 1, "vac_A": 2}, 1.0),
    ({"CO": 1, "vac_B": 1}, {"CO_B": 1}, 1.0),
    ({"O2": 1, "vac_B": 2}, {"O_B": 2}, 1.0),
    ({"CO_B": 1, "O_B": 1}, {"CO2": 1, "vac_B": 2}, 1.0),
    ({"CO_ads": 1, "vac_B": 1}, {"vac_A": 1, "CO_B": 1}, 0.5),
)


@lru_cache(maxsize=1)
def upstream_kernels():
    from catmap.data import templates
    from catmap.thermodynamics.first_order_interactions import FirstOrderInteractions

    interaction = Template(templates["first_order_interaction_function"]).substitute(
        site_info_dict="site_info_dict = {'A': ([0,1],.5,{}), 'B': ([2,3],.5,{})}"
    )
    energetics = """potentials = dict(zip(('CO_ads','O_ads','CO_B','O_B'),Gf))
    potentials.update(dict(zip(('CO','O2','CO2','N2'),gas_energies)))
    potentials.update({'vac_A':0.0,'vac_B':0.0})
    G_IS = [sum(v*potentials[k] for k,v in r.items()) for r,p,s in REACTIONS]
    G_FS = [sum(v*potentials[k] for k,v in p.items()) for r,p,s in REACTIONS]
    G_TS = [max(a,b)+10000.0 for a,b in zip(G_IS,G_FS)]
    G_af, G_ar = [0.0]*7, [0.0]*7"""
    source = Template(templates["rate_constants"]).substitute(
        interaction_function=indent(interaction, "    ").lstrip(),
        n_adsorbates="n_adsorbates = 4",
        n_transition_states="n_transition_states = 0",
        kB=f"kB = {R}",
        h="h = 1.0",
        prefactor_list="prefactor_list = [100.0]*7",
        elementary_step_energetics=energetics,
    )
    scope = {"REACTIONS": REACTIONS}
    exec(compile(interaction, "<CatMAP first-order interaction template>", "exec"), scope)
    exec(compile(source, "<CatMAP rate-constant template>", "exec"), scope)
    return (
        scope["interaction_function"],
        scope["rate_constants"],
        FirstOrderInteractions.linear_response,
    )


def standard_potentials(temperature):
    # Independent thermodynamic evaluation of the fixed, documented fixture.
    internal = (-110000, 0, -390000, 0, -210000, -100000, -205000, -95000)
    cv = (20.8, 20.8, 28.8, 20.8, 10, 5, 10, 5)
    reference_g = (-10000, 0, -20000, 0, -14000, -4000, -12000, -3000)
    result = []
    for i, (u, c, g) in enumerate(zip(internal, cv, reference_g, strict=True)):
        h = u + (R * 350 if i < 4 else 0)
        cp = c + (R if i < 4 else 0)
        s = (h - g) / 350 + cp * log(temperature / 350)
        result.append(h + cp * (temperature - 350) - temperature * s)
    return result


def reference_rates(temperature, coverages, matrix, *, sites_mol=2.5e-7):
    interaction, constants, response = upstream_kernels()
    potentials = standard_potentials(temperature)
    normalized = [value / 2 for value in coverages]
    vector = np.asarray(matrix).ravel().tolist()
    _, differential, _ = interaction(normalized, potentials[4:], vector, response, True, False)
    kf, kr, _, _ = constants(
        potentials[4:] + vector,
        normalized,
        potentials[:4],
        [],
        temperature,
        response,
        float,
        np.array,
        exp,
        np.sqrt,
        False,
    )
    activities = dict(zip(ADS, coverages, strict=True))
    activities.update(
        {
            "vac_A": 1 - sum(coverages[:2]),
            "vac_B": 1 - sum(coverages[2:]),
            "CO": 0.2,
            "O2": 0.1,
            "CO2": 0.05,
            "N2": 0.65,
        }
    )
    derivatives = dict.fromkeys(ADS, 0.0)
    progress = []
    for i, (reactants, products, scale) in enumerate(REACTIONS):
        forward = sites_mol * scale * kf[i] * prod(activities[k] ** v for k, v in reactants.items())
        reverse = sites_mol * scale * kr[i] * prod(activities[k] ** v for k, v in products.items())
        progress.extend((forward, reverse))
        for k in ADS:
            derivatives[k] += (forward - reverse) * (products.get(k, 0) - reactants.get(k, 0))
    return {
        "excess_potentials_J_mol": np.array(differential) - np.array(potentials[4:]),
        "progress_mol_s": np.array(progress),
        "coverage_rates_s_inv": np.array([derivatives[k] / sites_mol for k in ADS]),
    }


def coverages_from_logits(logits):
    result = []
    for pair in (logits[:2], logits[2:]):
        weights = np.exp(np.r_[pair, 0] - max(0, *pair))
        result.extend(weights[:2] / sum(weights))
    return np.array(result)


def solve_reference_steady(temperature, initial, matrix):
    # Follow the same authored initial coverage to its attracting neighborhood
    # before the algebraic polish. Direct logit least-squares can stall at an
    # almost-empty vacancy boundary with a small parameter gradient but a
    # nonzero physical rate. The unchanged physical residual is authoritative.
    transient = solve_ivp(
        lambda t, y: reference_rates(temperature, y, matrix)["coverage_rates_s_inv"],
        (0, 1000),
        initial,
        method="BDF",
        rtol=1e-9,
        atol=1e-12,
    )
    if not transient.success:
        raise RuntimeError("CatMAP local transient preconditioning failed")
    endpoint = transient.y[:, -1]
    start = np.array(
        [
            log(c / (1 - sum(endpoint[(i // 2) * 2 : (i // 2 + 1) * 2])))
            for i, c in enumerate(endpoint)
        ]
    )
    result = least_squares(
        lambda logits: reference_rates(temperature, coverages_from_logits(logits), matrix)[
            "coverage_rates_s_inv"
        ],
        start,
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    coverage = coverages_from_logits(result.x)
    residual = reference_rates(temperature, coverage, matrix)["coverage_rates_s_inv"]
    if not result.success or max(abs(residual)) > 1e-8:
        raise RuntimeError("CatMAP local steady equations did not converge")
    return coverage
