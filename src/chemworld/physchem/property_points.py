"""One temperature observation is a point, not a constant fitted curve."""

from math import isfinite

from chemworld.physchem.caloric_math import FirstDerivative, scalar_value


def source_point_value(temperature, reference_temperature, value):
    if (
        isinstance(reference_temperature, bool)
        or not isfinite(reference_temperature)
        or reference_temperature <= 0
    ):
        raise ValueError("source point temperature must be finite and positive")
    if scalar_value(temperature) != reference_temperature:
        raise ValueError("a single source observation cannot be extrapolated")
    if isinstance(temperature, FirstDerivative):
        raise ValueError("a single source observation has no temperature derivative")
    return value
