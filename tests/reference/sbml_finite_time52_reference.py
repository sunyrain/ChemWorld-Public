"""Closed formulas for amounts and tangent growth, no production rate/AD imports."""

from math import exp, sqrt

import numpy as np


def reference(kind, time_s, scales):
    metric = np.asarray(list(scales.values()))
    if kind == "nonnormal":
        amounts = [0.6, 4.0]
        first, second = exp(-0.5 * time_s), exp(-1.2 * time_s)
        physical = np.array([[first, 0], [8 * (first - second) / 0.7, second]])
        transfer = physical * metric[None, :] / metric[:, None]
        trace = float(np.sum(transfer**2))
        determinant = first * second
        largest = sqrt((trace + sqrt(trace**2 - 4 * determinant**2)) / 2)
        growth = np.asarray([largest, determinant / largest])
    else:
        if kind == "closed":
            factor = exp(-0.5 * time_s)
            a = 0.4 + 0.4 * factor
            amounts, direction = [a, 1 - a], [-1, 1]
        elif kind == "growth":
            factor = exp(0.4 * time_s)
            amounts, direction = [0.5 * factor], [1]
        else:
            alpha, beta, total, a0 = 1.4, 0.3, 1.0, 0.8
            discriminant = sqrt(beta**2 + 4 * alpha * beta * total)
            positive, negative = (
                (-beta + discriminant) / (2 * alpha),
                (-beta - discriminant) / (2 * alpha),
            )
            z0 = (a0 - positive) / (a0 - negative)
            z = z0 * exp(-discriminant * time_s)
            a = (positive - z * negative) / (1 - z)
            factor = exp(-discriminant * time_s) * (1 - z0) ** 2 / (1 - z) ** 2
            amounts, direction = [a, (total - a) / 2], [-2, 1]
        vector = np.asarray(direction) / metric
        vector /= np.linalg.norm(vector)
        transfer = factor * np.outer(vector, vector)
        growth = np.array([factor])
    return {
        "amounts": amounts,
        "transfer": transfer,
        "growth": growth,
        "rates": np.log(growth) / time_s,
        "rank": len(growth),
    }
