"""Actual PhasePy interface oracles with independently solved source endpoints.

No native physics imports. Sources use R=8.314, bar, mol/cm3, Angstrom and
mN/m; all public return arrays below are converted explicitly to SI.
"""

from functools import lru_cache
from math import exp, log, sqrt

import numpy as np
import phasepy
from phasepy.equilibrium import flash
from phasepy.sgt import sgt_mix, sgt_mix_beta0, sgt_pure, sgt_zfixed
from scipy.integrate import quad
from scipy.optimize import brentq

R = 8.314
# identifier, Tc/K, Pc/bar, omega, influence/J m5 mol-2
MATERIALS = (
    ("propane", 369.83, 42.48, 0.152, 1e-20),
    ("butane", 425.12, 37.96, 0.2, 1.5e-20),
    ("pentane", 469.7, 33.7, 0.251, 2e-20),
)


def declarations(family):
    if family == "symmetric_lle":
        data = (("a", 425.12, 37.96, 0.2, 1e-20), ("b", 425.12, 37.96, 0.2, 1.5e-20))
    else:
        data = MATERIALS[: {"pure": 1, "binary": 2, "ternary": 3}[family]]
    interactions = np.zeros((len(data), len(data)))
    if family == "symmetric_lle":
        interactions[0, 1] = interactions[1, 0] = 0.15
    return data, interactions


def source_model(family, beta=0.5):
    data, interactions = declarations(family)
    components = [phasepy.component(name=k, Tc=tc, Pc=pc, w=w, cii=[c]) for k, tc, pc, w, c in data]
    mixture = components[0]
    for component in components[1:]:
        mixture = mixture + component
    if len(data) > 1:
        mixture.kij_cubic(interactions)
    model = phasepy.preos(mixture)
    if len(data) > 1:
        matrix = np.full((len(data), len(data)), beta)
        np.fill_diagonal(matrix, 0.0)
        model.beta_sgt(matrix)
    return model


@lru_cache(maxsize=1)
def miscibility_temperature():
    model = source_model("symmetric_lle")

    def curvature(temperature):
        values = []
        step = 1e-4
        for x in (0.5 - step, 0.5 + step):
            z = np.array([x, 1 - x])
            phi, _ = model.logfugef(z, temperature, 50.0, "L")
            mu = np.log(z) + phi
            values.append(mu[0] - mu[1])
        return (values[1] - values[0]) / (2 * step)

    return brentq(curvature, 250.0, 275.0, xtol=1e-9)


def source_endpoints(family, temperature, pressure=None):
    model = source_model(family)
    if family == "pure":
        _, tc, pc, omega, _ = declarations("pure")[0][0]
        seed = pc * exp(5.37 * (1 + omega) * (1 - tc / temperature))
        p, vl, vv = [float(np.asarray(v).ravel()[0]) for v in model.psat(temperature, P0=seed)]
        left, right = np.array([1e6 / vv]), np.array([1e6 / vl])
        pressure = p * 1e5
        scales = model.sgt_adim(temperature)
        mu = [
            float(
                np.asarray(model.muad(c.sum() * 1e-6 * scales[2], temperature * scales[0])).ravel()[
                    0
                ]
            )
            for c in (left, right)
        ]
        error = abs(mu[0] - mu[1])
        details = {
            "method": "phasepy.preos.psat with Wilson pressure seed",
            "volumes_cm3_mol": [vv, vl],
        }
    elif family == "symmetric_lle":
        pressure = 5e6 if pressure is None else pressure

        def mismatch(x):
            z = np.array([x, 1 - x])
            phi, _ = model.logfugef(z, temperature, pressure / 1e5, "L")
            return log(x / (1 - x)) + phi[0] - phi[1]

        x = brentq(mismatch, 1e-6, 0.49999, xtol=2e-13)
        compositions = [np.array([x, 1 - x]), np.array([1 - x, x])]
        potentials, densities = [], []
        for z in compositions:
            phi, volume = model.logfugef(z, temperature, pressure / 1e5, "L")
            potentials.append(np.log(z) + phi)
            densities.append(z * 1e6 / volume)
        left, right = densities
        error = float(max(abs(potentials[0] - potentials[1])))
        details = {
            "method": "symmetric common tangent using PhasePy logfugef and independent Brent",
            "critical_temperature_K": miscibility_temperature(),
        }
    else:
        nc = 2 if family == "binary" else 3
        z = np.array([0.6, 0.4] if nc == 2 else [0.5, 0.3, 0.2])
        x = np.array([0.4, 0.6] if nc == 2 else [0.2, 0.4, 0.4])
        y = np.array([0.8, 0.2] if nc == 2 else [0.8, 0.15, 0.05])
        result = flash(
            x, y, "LV", z, temperature, pressure / 1e5, model, K_tol=1e-18, full_output=True
        )
        if not 0 < result.beta < 1:
            raise ValueError("source feed has no two-phase branch")
        left, right = result.Y * 1e6 / result.v2, result.X * 1e6 / result.v1
        phi_l, _ = model.logfugef(result.X, temperature, pressure / 1e5, "L", v0=result.v1)
        phi_v, _ = model.logfugef(result.Y, temperature, pressure / 1e5, "V", v0=result.v2)
        error = float(max(abs(np.log(result.X) + phi_l - np.log(result.Y) - phi_v)))
        details = {
            "method": "phasepy.equilibrium.flash LV",
            "beta": float(result.beta),
            "source_error": float(result.error),
            "source_iterations": int(result.iter),
        }
    return {
        "T": temperature,
        "P": pressure,
        "left": left.tolist(),
        "right": right.tolist(),
        "source_endpoint_mu_error": error,
        "source_details": details,
    }


def source_interface(
    family,
    temperature,
    pressure=None,
    *,
    beta=0.5,
    length_m=12e-9,
    nodes=80,
    initial=None,
    method="reference",
):
    model = source_model(family, beta)
    pair = source_endpoints(family, temperature, pressure)
    left, right = np.array(pair["left"]) * 1e-6, np.array(pair["right"]) * 1e-6
    if family == "pure":
        result = sgt_pure(
            float(left[0]),
            float(right[0]),
            temperature,
            pair["P"] / 1e5,
            model,
            n=nodes,
            full_output=True,
        )
    elif beta == 0:
        result = sgt_mix_beta0(
            left,
            right,
            temperature,
            pair["P"] / 1e5,
            model,
            n=nodes,
            method=method,
            full_output=True,
        )
    elif method == "auto":
        result = sgt_mix(
            left,
            right,
            temperature,
            pair["P"] / 1e5,
            model,
            rho0="hyperbolic",
            z0=10.0,
            dz=1.5,
            itmax=40,
            n=nodes,
            ten_tol=1e-5,
            full_output=True,
            solver_opt={"maxiter": 3000, "ftol": 1e-12, "xtol": 1e-12},
        )
    else:
        result = sgt_zfixed(
            left,
            right,
            temperature,
            pair["P"] / 1e5,
            model,
            rho0="hyperbolic" if initial is None else initial,
            z=length_m * 1e10,
            n=nodes,
            full_output=True,
            solver_opt={"maxiter": 3000, "ftol": 1e-12, "xtol": 1e-12},
        )
    rho = np.asarray(result.rho).reshape((len(left), -1))
    factors = model.sgt_adim(temperature)
    if family == "pure":
        mu = model.muad(float(left[0]) * factors[2], temperature * factors[0])
        raw = [
            float(
                np.asarray(
                    model.dOm(
                        c[0] * factors[2],
                        temperature * factors[0],
                        mu,
                        pair["P"] / 1e5 * factors[1],
                    )
                ).ravel()[0]
            )
            for c in rho.T
        ]
    else:
        temp_aux = model.temperature_aux(temperature)
        mu = model.muad_aux(left * factors[2], temp_aux)
        raw = [
            float(model.dOm_aux(c * factors[2], temp_aux, mu, pair["P"] / 1e5 * factors[1]))
            for c in rho.T
        ]
    return pair | {
        "gamma": float(np.asarray(result.tension).ravel()[0]) * 1e-3,
        "rho": (rho.T * 1e6).tolist(),
        "z": (np.asarray(result.z) * 1e-10).tolist(),
        "raw_dimensionless_grand_potential": raw,
        "source_summary": {k: v for k, v in dict(result).items() if k not in {"rho", "z", "GPT"}},
        "source_result": result,
    }


def pure_interior_positions(temperature, fractions):
    """Source EOS grand-potential quadrature, independent of native energy calls."""
    model = source_model("pure")
    pair = source_endpoints("pure", temperature)
    low, high = pair["left"][0], pair["right"][0]
    factors = model.sgt_adim(temperature)
    mu = model.muad(low * 1e-6 * factors[2], temperature * factors[0])

    def integrand(rho):
        omega = float(
            np.asarray(
                model.dOm(
                    rho * 1e-6 * factors[2],
                    temperature * factors[0],
                    mu,
                    pair["P"] / 1e5 * factors[1],
                )
            ).ravel()[0]
        )
        physical = omega / factors[1] * 1e5
        return sqrt(1e-20 / (2 * physical))

    return [
        quad(
            integrand,
            (low + high) / 2,
            low + x * (high - low),
            epsabs=1e-16,
            epsrel=1e-9,
            limit=150,
        )[0]
        for x in fractions
    ]
