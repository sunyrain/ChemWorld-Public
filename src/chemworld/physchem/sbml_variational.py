"""Shared directional evaluation of original SBML rates, without state or solver."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import FirstDerivative, scalar_value


def sbml_variational_rates(
    frame: Any,
    values: Mapping[str, float],
    keys: Sequence[str],
    slopes: np.ndarray,
    time_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original coordinate rates and their supplied directional derivatives."""
    dimension = slopes.shape[1]
    jets = {key: FirstDerivative(values[key], slopes[i]) for i, key in enumerate(keys)}
    rates = frame.rates(jets, time_s).coordinate_rates
    return np.asarray([scalar_value(rates[key]) for key in keys]), np.asarray(
        [
            rates[key].slope if isinstance(rates[key], FirstDerivative) else np.zeros(dimension)
            for key in keys
        ]
    )
