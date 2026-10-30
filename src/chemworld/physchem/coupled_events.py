"""Choose the feasible side of a rounded event root without changing inventories."""

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def feasible_terminal_point(
    time: float, vector: NDArray[np.float64],
    interpolant: Callable[[float], ArrayLike] | None,
    feasible: Callable[[float, NDArray[np.float64]], bool | np.bool_],
) -> tuple[float, NDArray[np.float64]]:
    """Return one complete interpolated state within root-location roundoff.

    SciPy's event Brent solve has absolute and relative tolerances of four
    machine epsilons. The bounded interval below encloses that location error;
    it is not an inventory tolerance or a retry of the physical integration.
    A larger violation remains a failure. Ordinary feasible roots are unchanged.
    """
    if feasible(time, vector):
        return time, vector
    if interpolant is None or time <= 0:
        raise RuntimeError("terminal root has no feasible interpolated prefix")
    upper = time
    lower = max(0.0, time - 8 * np.finfo(float).eps * max(1.0, abs(time)))
    neighbor = float(np.nextafter(time, 0.0))
    candidate = np.asarray(interpolant(neighbor))
    if feasible(neighbor, candidate):
        return neighbor, candidate
    result = np.asarray(interpolant(lower))
    if not feasible(lower, result):
        raise RuntimeError("terminal root exceeds the feasible-side roundoff interval")
    # Locate the closest representable feasible prefix using this same dense
    # trajectory for every material, progress and energy coordinate together.
    for _ in range(64):
        midpoint = lower + (upper - lower) / 2
        if midpoint in (lower, upper):
            break
        candidate = np.asarray(interpolant(midpoint))
        if feasible(midpoint, candidate):
            lower, result = midpoint, candidate
        else:
            upper = midpoint
    return lower, result
