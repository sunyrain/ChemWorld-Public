"""Probability arithmetic for static candidate acquisition functions."""

from __future__ import annotations

import math

import numpy as np


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Evaluate the existing scalar-erf CDF on a one-dimensional array."""

    values = np.asarray(z, dtype=float)
    return 0.5 * (
        1.0
        + np.asarray(
            [math.erf(float(value) / math.sqrt(2.0)) for value in values],
            dtype=float,
        )
    )
