"""Independent closed-form weighted profiles; no ChemWorld imports."""

from math import exp

import numpy as np
from scipy.optimize import minimize_scalar

TIMES = np.array([0.0, 0.3, 0.8, 1.4, 2.2])
OBSERVATIONS = np.array([1.01, 1.23, 1.52, 1.94, 2.47])
SIGMAS = np.array([0.04, 0.06, 0.05, 0.08, 0.05])


def affine_sse(a, b):
    residual = (a + b * TIMES - OBSERVATIONS) / SIGMAS
    return float(residual @ residual)


def affine_point(parameter, value, b_upper=2.0):
    weights = 1 / SIGMAS**2
    if parameter == "a":
        nuisance = float(
            np.sum(weights * TIMES * (OBSERVATIONS - value)) / np.sum(weights * TIMES**2)
        )
        nuisance = max(0.01, min(b_upper, nuisance))
        return affine_sse(value, nuisance), nuisance
    nuisance = float(np.sum(weights * (OBSERVATIONS - value * TIMES)) / np.sum(weights))
    nuisance = max(0.01, min(3.0, nuisance))
    return affine_sse(nuisance, value), nuisance


def affine_baseline(b_upper=2.0):
    design = np.column_stack((np.ones(len(TIMES)), TIMES)) / SIGMAS[:, None]
    candidate = np.linalg.lstsq(design, OBSERVATIONS / SIGMAS, rcond=None)[0]
    candidates = []
    if 0.01 <= candidate[0] <= 3 and 0.01 <= candidate[1] <= b_upper:
        candidates.append(tuple(candidate))
    candidates += [(a, affine_point("a", a, b_upper)[1]) for a in (0.01, 3.0)]
    candidates += [(affine_point("b", b, b_upper)[1], b) for b in (0.01, b_upper)]
    a, b = min(candidates, key=lambda pair: affine_sse(*pair))
    return {"a": float(a), "b": float(b), "sse": affine_sse(a, b)}


def pool_predictions(kind, k1, k2=0.23):
    readings = []
    for initial in (0.8, 0.45):
        for t in (0, 0.2, 0.6, 1.2, 2.5, 5):
            if kind == "reversible":
                eq = (initial + 0.2) * k2 / (k1 + k2)
                a = eq + (initial - eq) * exp(-(k1 + k2) * t)
                readings.extend((a / 2, k1 * a))
            else:
                readings.append(initial * exp(-(k1 + (k2 if kind == "confounded" else 0)) * t))
    return np.asarray(readings)


def pool_sse(kind, k1, k2=0.23):
    residual = (pool_predictions(kind, k1, k2) - pool_predictions(kind, 0.55)) / 0.002
    return float(residual @ residual)


def pool_point(kind, k1):
    if kind == "decay":
        return pool_sse(kind, k1), None
    if kind == "confounded":
        nuisance = max(0.04, min(1.5, 0.78 - k1))
    else:
        result = minimize_scalar(
            lambda k2: pool_sse(kind, k1, k2),
            bounds=(0.04, 1.5),
            method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success:
            raise ValueError("independent scalar nuisance reference did not converge")
        nuisance = min((0.04, float(result.x), 1.5), key=lambda k2: pool_sse(kind, k1, k2))
    return pool_sse(kind, k1, nuisance), nuisance
